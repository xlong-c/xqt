"""Legacy SVDQuant W8A8 split executor."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from xqt.contracts.composite import CompositeAddLinear, CompositeAddModule
from xqt.runtime.modules.w4_storage_int8_mma_linear import W4StorageInt8MmaLinear

_SUPPORTED_RESIDUAL_QUANT_DTYPES = frozenset({"fp4", "int4"})


def _current_cuda_stream_id(device: torch.device) -> int:
    raw_stream = getattr(torch._C, "_cuda_getCurrentRawStream", None)
    if callable(raw_stream):
        return int(raw_stream(device.index))
    return int(torch.cuda.current_stream(device).cuda_stream)


class SVDQuantInt8MmaLinear(CompositeAddModule):
    """SVDQuant packed W4 residual with W8A8 INT8 MMA execution."""

    def __init__(
        self,
        down_weight: torch.Tensor,
        up_weight: torch.Tensor,
        packed_residual: torch.Tensor,
        residual_scale: torch.Tensor,
        *,
        bias: torch.Tensor | None,
        input_features: int,
        output_features: int,
        group_size: int,
        padded_input_features: int,
        output_dtype: torch.dtype,
        quant_dtype: str = "fp4",
        engine: str = "auto",
        fallback_engine: str = "torch_int_mm",
        block_m: int = 64,
        block_n: int = 64,
        block_k: int = 64,
        threads: int = 128,
        num_stages: int = 2,
        activation_scale_mode: str = "dynamic",
        activation_scale: torch.Tensor | float | None = None,
        activation_quant_block_size: int = 256,
        eps: float = 1e-6,
        cache_int8_compute_view: bool = True,
        min_int8_rows: int = 0,
    ) -> None:
        super().__init__()
        residual_dtype = str(quant_dtype).lower()
        if residual_dtype not in _SUPPORTED_RESIDUAL_QUANT_DTYPES:
            allowed = ", ".join(sorted(_SUPPORTED_RESIDUAL_QUANT_DTYPES))
            raise ValueError(
                f"Unsupported quant_dtype '{quant_dtype}' for "
                "SVDQuantInt8MmaLinear. Supported: "
                f"{allowed}"
            )
        self.output_dtype = output_dtype
        self.input_features = int(input_features)
        self.output_features = int(output_features)
        self.group_size = int(group_size)
        self.padded_input_features = int(padded_input_features)
        self.quant_dtype = residual_dtype
        self.rank = int(down_weight.shape[0])
        self.down_proj = nn.Linear(
            self.input_features,
            int(down_weight.shape[0]),
            bias=False,
            dtype=down_weight.dtype,
            device=down_weight.device,
        )
        self.down_proj.weight.data = down_weight.detach().clone()
        self.up_proj = nn.Linear(
            int(up_weight.shape[1]),
            self.output_features,
            bias=False,
            dtype=up_weight.dtype,
            device=up_weight.device,
        )
        self.up_proj.weight.data = up_weight.detach().clone()
        self.residual_int8 = W4StorageInt8MmaLinear(
            packed_residual,
            residual_scale,
            bias=None,
            input_features=self.input_features,
            output_features=self.output_features,
            group_size=self.group_size,
            padded_input_features=self.padded_input_features,
            engine=engine,
            fallback_engine=fallback_engine,
            block_m=block_m,
            block_n=block_n,
            block_k=block_k,
            threads=threads,
            num_stages=num_stages,
            output_dtype=output_dtype,
            activation_scale_mode=activation_scale_mode,
            activation_scale=activation_scale,
            activation_quant_block_size=activation_quant_block_size,
            eps=eps,
            cache_int8_compute_view=cache_int8_compute_view,
            min_int8_rows=min_int8_rows,
        )
        if bias is None:
            self.register_buffer("bias", None)
        else:
            self.register_buffer("bias", bias.detach().clone().to(torch.float32))
        self._fused_forward: Any = None
        self._native_w8a8_fusion_enabled = False
        self._native_w8a8_packed_cache: tuple[tuple[Any, ...], Any] | None = None
        self._native_w8a8_workspace_cache: dict[tuple[Any, ...], Any] = {}
        self._native_w8a8_hot_cache: dict[
            tuple[Any, ...], tuple[tuple[Any, ...], Any, Any, Any]
        ] = {}
        self._last_native_w8a8_used = False
        self._last_native_w8a8_fallback_reason: str | None = None
        self._last_fused_forward_used = False
        self._last_fused_forward_fallback_reason: str | None = None

    def _apply(self, fn: Any) -> "SVDQuantInt8MmaLinear":
        """Move child runtime modules and discard device-specific compiled code."""

        previous_weight_dtype = self.down_proj.weight.dtype
        previous_output_dtype = self.output_dtype
        super()._apply(fn)
        if previous_output_dtype == previous_weight_dtype:
            self.output_dtype = self.down_proj.weight.dtype
        if hasattr(self.residual_int8, "output_dtype"):
            self.residual_int8.output_dtype = self.output_dtype
        self._fused_forward = None
        self._native_w8a8_fusion_enabled = False
        self._native_w8a8_packed_cache = None
        self._native_w8a8_workspace_cache.clear()
        self._native_w8a8_hot_cache.clear()
        self._last_native_w8a8_used = False
        self._last_native_w8a8_fallback_reason = None
        self._last_fused_forward_used = False
        self._last_fused_forward_fallback_reason = None
        return self

    @classmethod
    def from_composite(
        cls,
        module: CompositeAddLinear,
        *,
        output_dtype: torch.dtype | None = None,
        quant_dtype: str | None = None,
        engine: str = "auto",
        fallback_engine: str = "torch_int_mm",
        block_m: int = 64,
        block_n: int = 64,
        block_k: int = 64,
        threads: int = 128,
        num_stages: int = 2,
        activation_scale_mode: str = "dynamic",
        activation_scale: torch.Tensor | float | None = None,
        activation_quant_block_size: int = 256,
        eps: float = 1e-6,
        cache_int8_compute_view: bool = True,
        min_int8_rows: int = 0,
    ) -> "SVDQuantInt8MmaLinear":
        """Materialize a generic composite artifact as a W8A8 runtime view."""

        if not isinstance(module, CompositeAddLinear):
            raise TypeError("module must be a CompositeAddLinear artifact")
        return cls(
            down_weight=module.down_proj.weight.detach(),
            up_weight=module.up_proj.weight.detach(),
            packed_residual=module.packed_residual.detach(),
            residual_scale=module.residual_scale.detach(),
            bias=None if module.bias is None else module.bias.detach(),
            input_features=module.input_features,
            output_features=module.output_features,
            group_size=module.group_size,
            padded_input_features=module.padded_input_features,
            output_dtype=(
                module.down_proj.weight.dtype
                if output_dtype is None
                else output_dtype
            ),
            quant_dtype=module.quant_dtype if quant_dtype is None else quant_dtype,
            engine=engine,
            fallback_engine=fallback_engine,
            block_m=block_m,
            block_n=block_n,
            block_k=block_k,
            threads=threads,
            num_stages=num_stages,
            activation_scale_mode=activation_scale_mode,
            activation_scale=activation_scale,
            activation_quant_block_size=activation_quant_block_size,
            eps=eps,
            cache_int8_compute_view=cache_int8_compute_view,
            min_int8_rows=min_int8_rows,
        )

    def dequantize_residual(self) -> torch.Tensor:
        """Return the residual weight reconstructed from packed W4 storage."""

        return self.residual_int8.dequantize_weight()

    def low_rank_weight(self) -> torch.Tensor:
        """Return the dense weight reconstructed from the low-rank branch."""

        return self.up_proj.weight @ self.down_proj.weight

    def full_weight_dequant(self) -> torch.Tensor:
        """Return the complete reconstructed weight for validation or export."""

        low_rank_weight = self.low_rank_weight()
        return low_rank_weight + self.dequantize_residual().to(
            dtype=low_rank_weight.dtype,
            device=low_rank_weight.device,
        )

    def enable_fusion(self, *, mode: str = "reduce-overhead") -> bool:
        """Enable native W8A8 fusion or the compiled split fallback."""

        self._native_w8a8_fusion_enabled = False
        try:
            from xqt.operator_opt.kernels.cute.svdq_w8a8_sm89 import (
                native_svdq_w8a8_available,
                native_svdq_w8a8_shape_supported,
            )

            native_contract = (
                self.output_dtype == torch.bfloat16
                and self.down_proj.weight.dtype == torch.bfloat16
                and self.up_proj.weight.dtype == torch.bfloat16
                and self.residual_int8.activation_scale_mode == "dynamic"
                and native_svdq_w8a8_shape_supported(
                    self.input_features,
                    self.output_features,
                    self.rank,
                )
            )
            if native_contract and native_svdq_w8a8_available(build=False):
                self._native_w8a8_fusion_enabled = True
                self._fused_forward = None
                return True
        except Exception:
            self._native_w8a8_fusion_enabled = False

        compile_fn = getattr(torch, "compile", None)
        if not callable(compile_fn):
            return False
        try:
            self._fused_forward = compile_fn(self._compute_lean, mode=mode)
        except Exception:
            self._fused_forward = None
            return False
        return True

    def disable_fusion(self) -> None:
        self._native_w8a8_fusion_enabled = False
        self._native_w8a8_hot_cache.clear()
        self._fused_forward = None

    def _native_w8a8_gate(self, inputs: torch.Tensor) -> tuple[bool, str]:
        if not self._native_w8a8_fusion_enabled or not inputs.is_cuda:
            return False, "native SVDQuant W8A8 fusion is disabled or input is not CUDA"
        if inputs.dtype != torch.bfloat16:
            return False, "native SVDQuant W8A8 follows the official BF16 contract"
        if self.output_dtype != torch.bfloat16:
            return False, "native SVDQuant W8A8 output dtype must be bfloat16"
        if self.down_proj.weight.dtype != torch.bfloat16:
            return False, "native SVDQuant W8A8 down-projection must be bfloat16"
        if self.up_proj.weight.dtype != torch.bfloat16:
            return False, "native SVDQuant W8A8 up-projection must be bfloat16"
        if self.residual_int8.activation_scale_mode != "dynamic":
            return False, "native SVDQuant W8A8 requires dynamic activation scales"
        if inputs.ndim < 1 or int(inputs.shape[-1]) != self.input_features:
            return False, "native SVDQuant W8A8 input trailing dimension is invalid"
        rows = int(inputs.reshape(-1, self.input_features).shape[0])
        if 0 < self.residual_int8.min_int8_rows and rows < self.residual_int8.min_int8_rows:
            return False, "input rows are below min_int8_rows"
        native_tensors = (
            self.residual_int8.packed_weight,
            self.residual_int8.group_scale,
            self.down_proj.weight,
            self.up_proj.weight,
        )
        if any(not tensor.is_cuda for tensor in native_tensors):
            return False, "native SVDQuant W8A8 requires all weights on CUDA"
        if any(tensor.device != inputs.device for tensor in native_tensors):
            return False, "native SVDQuant W8A8 inputs and weights must share a device"
        major, minor = torch.cuda.get_device_capability(inputs.device)
        if (major, minor) != (8, 9):
            return False, f"native SVDQuant W8A8 currently targets sm_{major}{minor}"
        try:
            from xqt.operator_opt.kernels.cute.svdq_w8a8_sm89 import (
                native_svdq_w8a8_available,
                native_svdq_w8a8_shape_supported,
            )

            if not native_svdq_w8a8_shape_supported(
                self.input_features,
                self.output_features,
                self.rank,
            ):
                return False, (
                    "native SVDQuant W8A8 requires N/K multiples of 4 and "
                    "rank <= 1024"
                )
            if not native_svdq_w8a8_available(build=False):
                return False, "native SVDQuant W8A8 backend is unavailable"
        except Exception as exc:
            return False, f"native SVDQuant W8A8 capability check failed: {exc}"
        return True, "native SVDQuant dynamic W8A8 plus LoRA fusion is available"

    def _native_w8a8_packed(self, inputs: torch.Tensor) -> Any:
        from xqt.operator_opt.kernels.cute.svdq_w8a8_sm89 import (
            pack_svdq_w8a8_linear,
        )

        tensors = (
            self.residual_int8.packed_weight,
            self.residual_int8.group_scale,
            self.down_proj.weight,
            self.up_proj.weight,
            self.bias,
        )
        signature = (
            str(inputs.device),
            self.input_features,
            self.output_features,
            self.rank,
            self.group_size,
            *(
                None
                if tensor is None
                else (
                    str(tensor.device),
                    str(tensor.dtype),
                    int(tensor.data_ptr()),
                    int(getattr(tensor, "_version", 0)),
                    tuple(int(dim) for dim in tensor.shape),
                )
                for tensor in tensors
            ),
        )
        cached = self._native_w8a8_packed_cache
        if cached is not None and cached[0] == signature:
            return cached[1]
        residual = self.dequantize_residual().to(
            device=inputs.device,
            dtype=torch.bfloat16,
        )
        bias = None
        if self.bias is not None:
            bias = self.bias.to(device=inputs.device, dtype=torch.bfloat16)
        packed = pack_svdq_w8a8_linear(
            residual,
            self.down_proj.weight,
            self.up_proj.weight,
            bias,
        )
        self._native_w8a8_packed_cache = (signature, packed)
        self._native_w8a8_workspace_cache.clear()
        self._native_w8a8_hot_cache.clear()
        return packed

    def _native_w8a8_state_signature(self) -> tuple[Any, ...]:
        packed_weight = self.residual_int8.packed_weight
        group_scale = self.residual_int8.group_scale
        down_weight = self.down_proj.weight
        up_weight = self.up_proj.weight
        bias = self.bias
        return (
            self.output_dtype,
            self.residual_int8.activation_scale_mode,
            int(self.residual_int8.min_int8_rows),
            id(packed_weight),
            int(packed_weight._version),
            id(group_scale),
            int(group_scale._version),
            id(down_weight),
            int(down_weight._version),
            id(up_weight),
            int(up_weight._version),
            0 if bias is None else id(bias),
            0 if bias is None else int(bias._version),
        )

    @staticmethod
    def _native_w8a8_hot_key(inputs: torch.Tensor) -> tuple[Any, ...]:
        stream_id = _current_cuda_stream_id(inputs.device)
        return (
            inputs.device.index,
            inputs.dtype,
            int(inputs.shape[0]),
            stream_id,
        )

    def _native_w8a8_hot_forward(self, inputs: torch.Tensor) -> torch.Tensor | None:
        if not self._native_w8a8_fusion_enabled or not inputs.is_cuda:
            return None
        flat = (
            inputs
            if inputs.ndim == 2
            else inputs.reshape(-1, self.input_features)
        )
        key = self._native_w8a8_hot_key(flat)
        cached = self._native_w8a8_hot_cache.get(key)
        if cached is None:
            return None
        state_signature, packed, workspace, native_forward = cached
        output = native_forward(flat)
        if state_signature != self._native_w8a8_state_signature():
            self._native_w8a8_hot_cache.pop(key, None)
            return None
        if not self._last_native_w8a8_used:
            self._last_fused_forward_used = False
            self._last_fused_forward_fallback_reason = None
            self._last_native_w8a8_used = True
            self._last_native_w8a8_fallback_reason = None
        if inputs.ndim == 2:
            return output
        original_shape = inputs.shape[:-1]
        return output.reshape(*original_shape, self.output_features)

    def _native_w8a8_workspace(self, inputs: torch.Tensor, packed: Any) -> Any:
        from xqt.operator_opt.kernels.cute.svdq_w8a8_sm89 import (
            allocate_svdq_w8a8_workspace,
        )

        rows = int(inputs.shape[0])
        padded_rows = ((rows + 255) // 256) * 256
        stream_id = _current_cuda_stream_id(inputs.device)
        key = (
            str(inputs.device),
            padded_rows,
            int(packed.padded_input_features),
            int(packed.padded_rank),
            stream_id,
        )
        workspace = self._native_w8a8_workspace_cache.get(key)
        if workspace is not None:
            return workspace
        workspace = allocate_svdq_w8a8_workspace(rows, packed)
        if len(self._native_w8a8_workspace_cache) >= 8:
            self._native_w8a8_workspace_cache.clear()
        self._native_w8a8_workspace_cache[key] = workspace
        return workspace

    def _native_w8a8_forward(self, inputs: torch.Tensor) -> torch.Tensor:
        from xqt.operator_opt.kernels.cute.svdq_w8a8_sm89 import (
            bind_svdq_w8a8_linear,
        )

        original_shape = tuple(int(dim) for dim in inputs.shape[:-1])
        flat = (
            inputs
            if inputs.ndim == 2
            else inputs.reshape(-1, self.input_features)
        )
        packed = self._native_w8a8_packed(flat)
        workspace = self._native_w8a8_workspace(flat, packed)
        native_forward = bind_svdq_w8a8_linear(
            packed,
            workspace,
            rows=int(flat.shape[0]),
        )
        output = native_forward(flat)
        if len(self._native_w8a8_hot_cache) >= 8:
            self._native_w8a8_hot_cache.clear()
        self._native_w8a8_hot_cache[self._native_w8a8_hot_key(flat)] = (
            self._native_w8a8_state_signature(),
            packed,
            workspace,
            native_forward,
        )
        self._last_native_w8a8_used = True
        self._last_native_w8a8_fallback_reason = None
        if inputs.ndim == 2:
            return output
        return output.reshape(*original_shape, self.output_features)

    def _low_rank(self, inputs: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        low_rank_inputs = inputs.to(
            device=self.down_proj.weight.device,
            dtype=self.down_proj.weight.dtype,
        )
        return self.up_proj(self.down_proj(low_rank_inputs)).to(
            device=reference.device,
            dtype=reference.dtype,
        )

    def _compute_lean(self, inputs: torch.Tensor) -> torch.Tensor:
        """Compile-friendly split path without Python fallback decisions."""

        residual_output = self.residual_int8(inputs)
        output = residual_output + self._low_rank(inputs, residual_output)
        if self.bias is not None:
            output = output + self.bias.to(device=output.device, dtype=output.dtype)
        return output

    def _compute_guarded(self, inputs: torch.Tensor) -> torch.Tensor:
        """Eager split path retaining residual engine fallback behavior."""

        residual_output = self.residual_int8(inputs)
        output = residual_output + self._low_rank(inputs, residual_output)
        if self.bias is not None:
            output = output + self.bias.to(device=output.device, dtype=output.dtype)
        return output

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Run W8A8 MMA on the residual and add the low-rank path."""

        if inputs.ndim < 1 or inputs.shape[-1] != self.input_features:
            raise ValueError(
                "SVDQuantInt8MmaLinear input trailing dimension does not match "
                "input_features"
            )
        rows = (
            int(inputs.shape[0])
            if inputs.ndim == 2
            else int(inputs.numel() // self.input_features)
        )
        residual = self.residual_int8
        hot_output = self._native_w8a8_hot_forward(inputs)
        if hot_output is not None:
            return hot_output
        self._last_fused_forward_used = False
        self._last_fused_forward_fallback_reason = None
        self._last_native_w8a8_used = False
        self._last_native_w8a8_fallback_reason = None
        native_allowed, native_reason = self._native_w8a8_gate(inputs)
        if native_allowed:
            try:
                return self._native_w8a8_forward(inputs)
            except Exception as exc:
                native_reason = f"native SVDQuant W8A8 execution failed: {exc}"
        if self._native_w8a8_fusion_enabled:
            self._last_native_w8a8_fallback_reason = native_reason
        use_fused = (
            self._fused_forward is not None
            and inputs.is_cuda
            and not (0 < residual.min_int8_rows and rows < residual.min_int8_rows)
        )
        if use_fused:
            try:
                output = self._fused_forward(inputs)
                self._last_fused_forward_used = True
                return output
            except Exception as exc:
                self._last_fused_forward_fallback_reason = str(exc)
                self._fused_forward = None
        return self._compute_guarded(inputs)

    def execution_metadata(self) -> dict[str, Any]:
        """Expose residual INT8 execution details with SVDQuant context."""

        metadata = self.residual_int8.execution_metadata()
        metadata.update(
            {
                "implementation": (
                    "native_svdq_w8a8_dynamic_lora"
                    if self._last_native_w8a8_used
                    else "composite_add_svd_w4_residual_int8_mma"
                ),
                "compute_contract": "composite_add",
                "low_rank_branch": "source_precision",
                "residual_storage": "packed_signed_int4_group_scale",
                "residual_compute": "w8a8_int8_mma",
                "quant_dtype": self.quant_dtype,
                "fusion_enabled": (
                    self._native_w8a8_fusion_enabled
                    or self._fused_forward is not None
                ),
                "native_w8a8_fusion_enabled": bool(
                    self._native_w8a8_fusion_enabled
                ),
                "native_w8a8_used": bool(self._last_native_w8a8_used),
                "native_w8a8_fallback_reason": self._last_native_w8a8_fallback_reason,
                "fused_forward_used": bool(self._last_fused_forward_used),
                "fused_forward_fallback_reason": self._last_fused_forward_fallback_reason,
            }
        )
        return metadata


__all__ = ["SVDQuantInt8MmaLinear"]
