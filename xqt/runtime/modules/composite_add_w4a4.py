"""Native W4A4 executor for generic additive composite artifacts."""

from __future__ import annotations

from typing import Any

import torch

from xqt.contracts.composite import (
    CompositeAddLinear,
    CompositeAddModule,
    initialize_composite_add_storage,
)


def _current_stream_id(device: torch.device) -> int:
    raw_stream = getattr(torch._C, "_cuda_getCurrentRawStream", None)
    if callable(raw_stream):
        return int(raw_stream(device.index))
    return int(torch.cuda.current_stream(device).cuda_stream)


class CompositeAddW4A4Linear(CompositeAddModule):
    """Run a generic composite artifact through the W4A4 main executor.

    ``main`` and ``smalln`` are backend layouts, not storage formats. Fused
    norm, QKV/RoPE and CUDA Graph variants remain outside this generic
    executor. Unsupported inputs fall back to the parent reference path.
    """

    _SUPPORTED_LAYOUTS = frozenset({"main", "smalln"})

    def __init__(
        self,
        *args: Any,
        native_fusion: bool = True,
        layout: str = "main",
        **kwargs: Any,
    ) -> None:
        initialize_composite_add_storage(self, *args, **kwargs)
        normalized_layout = str(layout).strip().lower()
        if normalized_layout not in self._SUPPORTED_LAYOUTS:
            allowed = ", ".join(sorted(self._SUPPORTED_LAYOUTS))
            raise ValueError(f"layout must be one of {allowed}; got {layout!r}")
        self.layout = normalized_layout
        self._native_fusion_enabled = bool(native_fusion)
        self._native_packed: tuple[tuple[Any, ...], Any] | None = None
        self._native_workspace: dict[tuple[Any, ...], Any] = {}
        self._native_forward: dict[tuple[Any, ...], Any] = {}
        self._last_native_used = False
        self._last_fallback_reason: str | None = None

    @classmethod
    def from_composite(
        cls,
        module: CompositeAddLinear,
        *,
        native_fusion: bool = True,
        layout: str = "main",
    ) -> "CompositeAddW4A4Linear":
        """Materialize a generic artifact with one W4A4 backend layout."""

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
            quant_dtype=module.quant_dtype,
            native_fusion=native_fusion,
            layout=layout,
        )

    def _clear_native_state(self) -> None:
        self._native_packed = None
        self._native_workspace.clear()
        self._native_forward.clear()

    def _apply(self, fn: Any) -> "CompositeAddW4A4Linear":
        super()._apply(fn)
        self._clear_native_state()
        self._last_native_used = False
        self._last_fallback_reason = None
        return self

    def enable_fusion(self) -> bool:
        """Enable native W4A4 dispatch when the current host can support it."""

        self._native_fusion_enabled = True
        try:
            return self._native_backend_available()
        except Exception:
            return False

    def disable_fusion(self) -> None:
        """Disable native dispatch and retain reference execution."""

        self._native_fusion_enabled = False
        self._clear_native_state()
        self._last_native_used = False
        self._last_fallback_reason = "native W4A4 fusion is disabled"

    def _native_backend_available(self) -> bool:
        from xqt.kernels.ops.quantization import (
            native_w4a4_available,
            native_w4a4_smalln_available,
        )

        if not native_w4a4_available(build=False):
            return False
        if self.layout == "smalln":
            return bool(native_w4a4_smalln_available(build=False))
        return True

    def _native_backend_label(self) -> str:
        return f"native W4A4 {self.layout} executor"

    def _native_gate(self, inputs: torch.Tensor) -> tuple[bool, str]:
        if not self._native_fusion_enabled:
            return False, "native W4A4 fusion is disabled"
        if not inputs.is_cuda:
            return False, "native W4A4 requires CUDA inputs"
        if torch.is_grad_enabled() and (
            inputs.requires_grad
            or any(parameter.requires_grad for parameter in self.parameters())
        ):
            return False, "native W4A4 is forward-only when autograd is required"
        if inputs.dtype not in {torch.float16, torch.bfloat16}:
            return False, "native W4A4 requires float16 or bfloat16 inputs"
        if self.down_proj.weight.dtype != inputs.dtype:
            return False, "down-projection dtype must match inputs"
        if self.up_proj.weight.dtype != inputs.dtype:
            return False, "up-projection dtype must match inputs"
        if int(inputs.shape[-1]) != self.input_features:
            return False, "input trailing dimension does not match artifact"
        if not self.packed_residual.is_cuda or not self.residual_scale.is_cuda:
            return False, "packed residual tensors must be CUDA"
        if self.packed_residual.device != inputs.device:
            return False, "packed residual tensors must share the input device"
        if self.bias is not None and (
            not self.bias.is_cuda or self.bias.device != inputs.device
        ):
            return False, "bias must share the input device"
        try:
            from xqt.kernels.ops.quantization import (
                native_w4a4_shape_supported,
            )

            if not native_w4a4_shape_supported(
                self.input_features,
                self.output_features,
            ):
                return False, "native W4A4 feature dimensions are not vector aligned"
            if not self._native_backend_available():
                return False, f"{self._native_backend_label()} is unavailable"
        except Exception as exc:
            return False, f"native W4A4 capability check failed: {exc}"
        return True, f"{self._native_backend_label()} is available"

    def _state_signature(self, inputs: torch.Tensor) -> tuple[Any, ...]:
        tensors = (
            self.down_proj.weight,
            self.up_proj.weight,
            self.packed_residual,
            self.residual_scale,
            self.bias,
        )
        return (
            str(inputs.device),
            str(inputs.dtype),
            tuple(
                (
                    None
                    if tensor is None
                    else (
                        int(tensor.data_ptr()),
                        int(getattr(tensor, "_version", 0)),
                        tuple(int(dim) for dim in tensor.shape),
                    )
                )
                for tensor in tensors
            ),
        )

    def _pack(self, inputs: torch.Tensor) -> Any:
        signature = self._state_signature(inputs)
        if self._native_packed is not None and self._native_packed[0] == signature:
            return self._native_packed[1]
        from xqt.kernels.ops.quantization import (
            pack_svdq_w4a4_linear,
            pack_svdq_w4a4_linear_smalln,
        )

        residual = self.dequantize_residual().to(
            device=inputs.device,
            dtype=inputs.dtype,
        )
        bias = None if self.bias is None else self.bias.to(
            device=inputs.device,
            dtype=inputs.dtype,
        )
        pack_function = (
            pack_svdq_w4a4_linear_smalln
            if self.layout == "smalln"
            else pack_svdq_w4a4_linear
        )
        packed = pack_function(
            residual,
            self.down_proj.weight,
            self.up_proj.weight,
            bias,
        )
        self._native_packed = (signature, packed)
        self._native_workspace.clear()
        self._native_forward.clear()
        return packed

    def _bind(self, inputs: torch.Tensor, packed: Any) -> Any:
        rows = int(inputs.reshape(-1, self.input_features).shape[0])
        key = (
            str(inputs.device),
            str(inputs.dtype),
            rows,
            _current_stream_id(inputs.device),
            self._state_signature(inputs),
        )
        cached = self._native_forward.get(key)
        if cached is not None:
            return cached
        from xqt.kernels.ops.quantization import (
            allocate_w4a4_workspace,
            bind_svdq_w4a4_linear,
            bind_svdq_w4a4_linear_smalln,
        )

        workspace_key = (
            str(inputs.device),
            str(inputs.dtype),
            rows,
            _current_stream_id(inputs.device),
            int(packed.padded_rank),
        )
        workspace = self._native_workspace.get(workspace_key)
        if workspace is None:
            workspace = allocate_w4a4_workspace(
                rows,
                packed,
                with_lora_rank=int(packed.padded_rank),
            )
            self._native_workspace[workspace_key] = workspace
        bind_function = (
            bind_svdq_w4a4_linear_smalln
            if self.layout == "smalln"
            else bind_svdq_w4a4_linear
        )
        native_forward = bind_function(
            packed,
            workspace,
            rows=rows,
        )
        self._native_forward[key] = native_forward
        return native_forward

    def dequantize_residual(self) -> torch.Tensor:
        """Expose the contract reference residual for fallback and inspection."""

        return CompositeAddLinear.dequantize_residual(self)

    def low_rank_weight(self) -> torch.Tensor:
        """Expose the contract reference low-rank weight."""

        return CompositeAddLinear.low_rank_weight(self)

    def full_weight_dequant(self) -> torch.Tensor:
        """Expose the contract reference complete weight."""

        return CompositeAddLinear.full_weight_dequant(self)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Run W4A4 main execution or the generic reference fallback."""

        allowed, reason = self._native_gate(inputs)
        if allowed:
            try:
                original_shape = tuple(int(dim) for dim in inputs.shape[:-1])
                flat = inputs.reshape(-1, self.input_features).contiguous()
                native_forward = self._bind(flat, self._pack(flat))
                output = native_forward(flat)
                self._last_native_used = True
                self._last_fallback_reason = None
                if inputs.ndim == 2:
                    return output
                return output.reshape(*original_shape, self.output_features)
            except Exception as exc:
                reason = f"native W4A4 execution failed: {exc}"
        self._last_native_used = False
        self._last_fallback_reason = reason
        return CompositeAddLinear.forward(self, inputs)

    def execution_metadata(self) -> dict[str, Any]:
        """Report the selected executor and its fallback reason."""

        return {
            **CompositeAddLinear.execution_metadata(self),
            "implementation": (
                f"composite_add_native_w4a4_{self.layout}"
                if self._last_native_used
                else "composite_add_reference"
            ),
            "w4a4_layout": self.layout,
            "native_fusion_enabled": bool(self._native_fusion_enabled),
            "native_w4a4_used": bool(self._last_native_used),
            "fallback_reason": self._last_fallback_reason,
        }


__all__ = ["CompositeAddW4A4Linear"]
