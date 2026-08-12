"""Runtime dual-branch SVD composite Linear modules."""

from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch import nn

from xqt.analysis.svd_analysis import decompose_weight_svd
from xqt.runtime.composite_branch import get_materializer, register_materializer
from xqt.runtime.modules.packing_int4 import _pack_int4, _unpack_int4
from xqt.runtime.modules.w4_storage_int8_mma_linear import W4StorageInt8MmaLinear

_SUPPORTED_RESIDUAL_QUANT_DTYPES = frozenset({"fp4", "int4"})


def _current_cuda_stream_id(device: torch.device) -> int:
    raw_stream = getattr(torch._C, "_cuda_getCurrentRawStream", None)
    if callable(raw_stream):
        return int(raw_stream(device.index))
    return int(torch.cuda.current_stream(device).cuda_stream)


def _can_mutate_runtime_cache() -> bool:
    """Return whether eager runtime caches may be mutated safely."""

    compiler = getattr(torch, "compiler", None)
    if compiler is not None:
        is_compiling = getattr(compiler, "is_compiling", None)
        if callable(is_compiling) and bool(is_compiling()):
            return False
    dynamo = getattr(torch, "_dynamo", None)
    if dynamo is not None:
        is_compiling = getattr(dynamo, "is_compiling", None)
        if callable(is_compiling) and bool(is_compiling()):
            return False
    return not torch.jit.is_tracing()


def _tilelang_runtime_usable() -> bool:
    """Probe the optional TileLang runtime without importing it at module load."""

    try:
        from xqt.operator_opt.kernels.tilelang._common import tilelang_runtime_usable
    except Exception:
        return False
    try:
        return bool(tilelang_runtime_usable())
    except Exception:
        return False


def _quantize_residual_int4(
    weight_res: torch.Tensor,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Quantize residual weight to signed INT4 with per-group scales.

    Returns (packed_weight, scale, padded_in_features).
    """
    out_features, in_features = weight_res.shape
    normalized_group_size = max(1, min(group_size, in_features))
    padded_in_features = (
        (in_features + normalized_group_size - 1) // normalized_group_size
    ) * normalized_group_size

    # Pad to group_size boundary
    weight_f32 = weight_res.detach().to(torch.float32)
    if padded_in_features != in_features:
        weight_f32 = F.pad(weight_f32, (0, padded_in_features - in_features))

    # Per-group absmax quantization to [-8, 7]
    grouped = weight_f32.reshape(out_features, -1, normalized_group_size)
    max_abs = grouped.abs().amax(dim=2, keepdim=True)
    scale = torch.where(
        max_abs > 0, max_abs / 7.0, torch.ones_like(max_abs)
    )  # shape: (out_features, num_groups, 1)
    quantized = torch.clamp(
        torch.round(grouped / (scale + 1e-12)), min=-8, max=7
    ).to(torch.int8)
    scale = scale.squeeze(-1)  # (out_features, num_groups)

    packed = _pack_int4(quantized.reshape(out_features, padded_in_features))
    return packed, scale.to(torch.float32), padded_in_features


def _dequantize_residual_int4(
    packed_weight: torch.Tensor,
    scale: torch.Tensor,
    input_features: int,
    group_size: int,
    padded_input_features: int,
) -> torch.Tensor:
    """Dequantize packed INT4 residual weight back to float."""
    codes = _unpack_int4(packed_weight, padded_input_features)
    grouped = codes.reshape(codes.shape[0], -1, group_size)  # (out, num_groups, gs)
    scale_expanded = scale.unsqueeze(-1)  # (out, num_groups, 1)
    dequantized = grouped * scale_expanded
    return dequantized.reshape(codes.shape[0], padded_input_features)[
        :, :input_features
    ]

class LowRankBranch(nn.Module):
    """Two-layer low-rank branch that absorbs weight outliers.

    W_lr = L2 @ L1, where:
      L1: (r, in_features) - down-projection
      L2: (out_features, r) - up-projection

    Forward: y = L2(L1(x)) = x @ L1.T @ L2.T
    """

    def __init__(
        self,
        down_weight: torch.Tensor,
        up_weight: torch.Tensor,
    ) -> None:
        super().__init__()
        rank, in_features = down_weight.shape
        out_features, rank2 = up_weight.shape
        if rank != rank2:
            raise ValueError(
                f"Rank mismatch: down_proj rank={rank}, up_proj rank={rank2}"
            )
        self.down_proj = nn.Linear(in_features, rank, bias=False)
        self.down_proj.weight.data = down_weight.detach().clone()
        self.up_proj = nn.Linear(rank, out_features, bias=False)
        self.up_proj.weight.data = up_weight.detach().clone()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute low-rank correction: L2(L1(x))."""
        return self.up_proj(self.down_proj(x))

class SVDQuantLinear(nn.Module):
    """SVDQuant Linear: low-rank FP16 branch + quantized INT4/FP4 residual.

    Shapes:
      - down_proj.weight: (r, in_features)       ← L1
      - up_proj.weight:   (out_features, r)       ← L2
      - packed_residual:  (out_features, padded_in // 2) - uint8 packed INT4
      - residual_scale:   (out_features, num_groups)
      - bias (optional):  (out_features,)

    Forward (reference path):
      x_q    = act_quantize(x)                    # simulated: pass-through in reference
      y_main = dequant_gemm(x, packed_residual)   # dequant + fp16 matmul
      y_lora = up_proj(down_proj(x))              # low-rank correction
      return y_main + y_lora + bias
    """

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
        quant_dtype: str = "int4",
    ) -> None:
        super().__init__()
        self.input_features = int(input_features)
        self.output_features = int(output_features)
        self.group_size = int(group_size)
        self.padded_input_features = int(padded_input_features)
        self.quant_dtype = str(quant_dtype)
        rank = int(down_weight.shape[0])
        self.rank = rank
        self.xqt_storage_protocol = "svd_low_rank_plus_residual"
        self.down_proj = nn.Linear(input_features, rank, bias=False)
        self.down_proj.weight.data = down_weight.detach().clone()
        self.up_proj = nn.Linear(rank, output_features, bias=False)
        self.up_proj.weight.data = up_weight.detach().clone()
        self.register_buffer("packed_residual", packed_residual.to(torch.uint8))
        self.register_buffer("residual_scale", residual_scale.to(torch.float32))
        if bias is None:
            self.register_buffer("bias", None)
        else:
            self.register_buffer("bias", bias.detach().clone().to(torch.float32))
        self._pending_activation_scale: torch.Tensor | float | None = None
        self._pending_activation_scale_mode: str = "dynamic"
        # RMSNorm fusion (native W4A4 only): when set, the module consumes
        # PRE-norm activations and folds the per-channel norm weight into the
        # packed smooth factor and LoRA-down columns; the row scale is applied
        # between the quantize and GEMM stages.  Fallback paths apply the norm
        # explicitly so the output semantics stay identical.
        self.register_buffer("fused_norm_weight", None, persistent=False)
        self.fused_norm_eps = 1e-6
        self._dequantized_residual_cache: tuple[tuple[Any, ...], torch.Tensor] | None = None
        self._native_w4a4_packed_cache: tuple[tuple[Any, ...], Any] | None = None
        self._native_w4a4_packed_smalln_cache: tuple[tuple[Any, ...], Any] | None = None
        self._native_w4a4_workspace_cache: dict[tuple[Any, ...], Any] = {}
        self._native_w4a4_hot_cache: dict[
            tuple[Any, ...], tuple[tuple[Any, ...], Any, Any, Any, str, Any]
        ] = {}
        self._cuda_fused_enabled = False
        self._cuda_graph_enabled = False
        self._cuda_graph_warmup = 2
        self._last_cuda_graph_used = False
        self._last_cuda_fused_used = False
        self._last_cuda_fused_backend: str | None = None
        self._last_cuda_fused_fallback_reason: str | None = None

    def _clear_runtime_caches(self) -> None:
        self._dequantized_residual_cache = None
        self._native_w4a4_packed_cache = None
        self._native_w4a4_packed_smalln_cache = None
        self._native_w4a4_workspace_cache.clear()
        self._native_w4a4_hot_cache.clear()

    def _apply(self, fn: Any) -> "SVDQuantLinear":
        """Move registered tensors, then invalidate derived residual storage."""

        super()._apply(fn)
        self._clear_runtime_caches()
        self._last_cuda_fused_used = False
        self._last_cuda_fused_backend = None
        self._last_cuda_fused_fallback_reason = None
        self._last_cuda_graph_used = False
        return self

    def enable_fusion(
        self,
        *,
        mode: str = "reduce-overhead",
        cuda_graph: bool = False,
        cuda_graph_warmup: int = 2,
    ) -> bool:
        """Enable the optional direct CUDA fused residual + low-rank kernel.

        With ``cuda_graph=True`` the native W4A4 hot path is captured into a
        per-shape CUDA Graph after the first call, so steady-state forwards
        replay baked kernels instead of paying the Python/pybind launch path.
        Graph replay returns the same static output tensor on every call:
        callers must consume or clone it before the next forward.
        """

        del mode
        if not torch.cuda.is_available():
            return False
        native_usable = False
        try:
            from xqt.operator_opt.kernels.cute.svdq_w4a4_sm89 import (
                native_w4a4_available,
            )

            native_usable = bool(native_w4a4_available(build=False))
        except Exception:
            native_usable = False
        if not native_usable and not _tilelang_runtime_usable():
            return False
        self._cuda_fused_enabled = True
        self._cuda_graph_enabled = bool(cuda_graph) and native_usable
        self._cuda_graph_warmup = max(0, int(cuda_graph_warmup))
        return True

    def disable_fusion(self) -> None:
        self._cuda_fused_enabled = False
        self._cuda_graph_enabled = False
        self._last_cuda_graph_used = False

    def set_fused_norm(self, norm_weight: torch.Tensor, *, eps: float = 1e-6) -> None:
        """Fuse an upstream RMSNorm into this linear (native W4A4 hot path).

        After this call the module expects PRE-norm activations: the norm is
        either folded into the native W4A4 quantize/GEMM chain or applied
        explicitly on fallback paths, so outputs stay consistent either way.
        """

        if norm_weight.ndim != 1 or int(norm_weight.numel()) != self.input_features:
            raise ValueError("fused norm weight must match input_features")
        self.fused_norm_weight = norm_weight.detach().clone()
        self.fused_norm_eps = float(eps)
        self._clear_runtime_caches()

    def clear_fused_norm(self) -> None:
        self.fused_norm_weight = None
        self._clear_runtime_caches()

    def _apply_fused_norm(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.fused_norm_weight
        if weight is None:
            return x
        weight = weight.to(device=x.device, dtype=torch.float32)
        row_scale = torch.rsqrt(
            x.float().pow(2).mean(dim=-1, keepdim=True) + self.fused_norm_eps
        )
        return (x.float() * row_scale * weight).to(x.dtype)

    def _native_w4a4_gate(self, inputs: torch.Tensor) -> tuple[bool, str]:
        if not self._cuda_fused_enabled or not inputs.is_cuda:
            return False, "native W4A4 fusion is disabled or the input is not CUDA"
        if inputs.dtype not in {torch.float16, torch.bfloat16}:
            return False, "native W4A4 requires float16 or bfloat16 activations"
        if self.down_proj.weight.dtype != inputs.dtype:
            return False, "native W4A4 down-projection dtype must match activations"
        if self.up_proj.weight.dtype != inputs.dtype:
            return False, "native W4A4 up-projection dtype must match activations"
        if inputs.ndim < 1 or int(inputs.shape[-1]) != self.input_features:
            return False, "native W4A4 input trailing dimension is invalid"
        if self.rank < 1 or self.rank > 1024:
            return False, "native W4A4 supports ranks from 1 through 1024"
        if not self.packed_residual.is_cuda or not self.residual_scale.is_cuda:
            return False, "native W4A4 requires residual buffers on CUDA"
        if inputs.device != self.packed_residual.device:
            return False, "native W4A4 input and residual buffers must share a device"
        fused_norm_weight = self.fused_norm_weight
        if fused_norm_weight is not None and (
            not fused_norm_weight.is_cuda or fused_norm_weight.device != inputs.device
        ):
            return False, "native W4A4 fused norm weight must live on the input device"
        major, minor = torch.cuda.get_device_capability(inputs.device)
        if (major, minor) != (8, 9):
            return False, f"native W4A4 currently targets sm_89, got sm_{major}{minor}"
        try:
            from xqt.operator_opt.kernels.cute.svdq_w4a4_sm89 import (
                native_w4a4_available,
                native_w4a4_shape_supported,
            )

            if not native_w4a4_shape_supported(
                self.input_features,
                self.output_features,
            ):
                return False, "native W4A4 requires N and K to be multiples of 4"
            if not native_w4a4_available(build=False):
                return False, "native W4A4 backend is unavailable"
        except Exception as exc:
            return False, f"native W4A4 capability check failed: {exc}"
        return True, "native Nunchaku two-stage W4A4 fusion is available"

    def _native_w4a4_state_signature(self) -> tuple[Any, ...]:
        # Direct _modules/_parameters/_buffers access: nn.Module.__getattr__
        # overhead dominates this hot-path guard when reading attributes
        # through the usual dotted paths.
        parameters_down = self._modules["down_proj"]._parameters
        parameters_up = self._modules["up_proj"]._parameters
        buffers = self._buffers
        packed_residual = buffers["packed_residual"]
        residual_scale = buffers["residual_scale"]
        down_weight = parameters_down["weight"]
        up_weight = parameters_up["weight"]
        bias = buffers["bias"]
        fused_norm_weight = buffers["fused_norm_weight"]
        return (
            id(packed_residual),
            int(packed_residual._version),
            id(residual_scale),
            int(residual_scale._version),
            id(down_weight),
            int(down_weight._version),
            id(up_weight),
            int(up_weight._version),
            0 if bias is None else id(bias),
            0 if bias is None else int(bias._version),
            0 if fused_norm_weight is None else id(fused_norm_weight),
            0 if fused_norm_weight is None else int(fused_norm_weight._version),
        )

    @staticmethod
    def _native_w4a4_hot_key(inputs: torch.Tensor) -> tuple[Any, ...]:
        stream_id = _current_cuda_stream_id(inputs.device)
        return (
            inputs.device.index,
            inputs.dtype,
            int(inputs.shape[0]),
            stream_id,
        )

    def _native_w4a4_hot_forward(self, inputs: torch.Tensor) -> torch.Tensor | None:
        if not self._cuda_fused_enabled or not inputs.is_cuda:
            return None
        flat = (
            inputs
            if inputs.ndim == 2
            else inputs.reshape(-1, self.input_features)
        )
        key = self._native_w4a4_hot_key(flat)
        cached = self._native_w4a4_hot_cache.get(key)
        if cached is None:
            return None
        state_signature, packed, workspace, native_forward, backend, graph_state = cached
        if state_signature != self._native_w4a4_state_signature():
            self._native_w4a4_hot_cache.pop(key, None)
            return None
        if graph_state is not None:
            from xqt.operator_opt.runtime import replay_cuda_graph_tensor_callable

            output = replay_cuda_graph_tensor_callable(graph_state, (flat,))
        else:
            output = native_forward(flat)
        self._last_cuda_graph_used = graph_state is not None
        if not self._last_cuda_fused_used:
            self._last_cuda_fused_used = True
            self._last_cuda_fused_backend = backend
            self._last_cuda_fused_fallback_reason = None
        if inputs.ndim == 2:
            return output
        original_shape = inputs.shape[:-1]
        return output.reshape(*original_shape, self.output_features)

    def _native_w4a4_packed(self, inputs: torch.Tensor, *, smalln: bool = False) -> Any:
        from xqt.operator_opt.kernels.cute.svdq_w4a4_sm89 import (
            pack_svdq_w4a4_linear,
            pack_svdq_w4a4_linear_smalln,
        )

        tensors = (
            self.packed_residual,
            self.residual_scale,
            self.down_proj.weight,
            self.up_proj.weight,
            self.bias,
            self.fused_norm_weight,
        )
        signature = (
            inputs.device.index,
            inputs.dtype,
            self.input_features,
            self.output_features,
            self.rank,
            self.group_size,
            self.fused_norm_eps if self.fused_norm_weight is not None else None,
            *(
                None
                if tensor is None
                else (
                    tensor.device.index,
                    tensor.dtype,
                    int(tensor.data_ptr()),
                    int(getattr(tensor, "_version", 0)),
                )
                for tensor in tensors
            ),
        )
        cached = (
            self._native_w4a4_packed_smalln_cache
            if smalln
            else self._native_w4a4_packed_cache
        )
        if cached is not None and cached[0] == signature:
            return cached[1]
        residual = self.dequantize_residual().to(
            device=inputs.device,
            dtype=inputs.dtype,
        )
        bias = None
        if self.bias is not None:
            bias = self.bias.to(device=inputs.device, dtype=inputs.dtype)
        down_weight = self.down_proj.weight
        smooth = None
        if self.fused_norm_weight is not None:
            norm_weight = self.fused_norm_weight.to(
                device=inputs.device, dtype=torch.float32
            )
            # The quantize epilogue divides by the smooth factor, so the norm
            # weight enters as its reciprocal there and folds directly into
            # the LoRA-down columns.
            down_weight = down_weight * norm_weight.to(down_weight.dtype).unsqueeze(0)
            smooth = (1.0 / norm_weight).to(inputs.dtype)
        pack_function = pack_svdq_w4a4_linear_smalln if smalln else pack_svdq_w4a4_linear
        packed = pack_function(
            residual,
            down_weight,
            self.up_proj.weight,
            bias,
            smooth=smooth,
        )
        if smalln:
            self._native_w4a4_packed_smalln_cache = (signature, packed)
        else:
            self._native_w4a4_packed_cache = (signature, packed)
        self._native_w4a4_workspace_cache.clear()
        return packed

    def _native_w4a4_workspace(
        self,
        inputs: torch.Tensor,
        packed: Any,
    ) -> Any:
        from xqt.operator_opt.kernels.cute.svdq_w4a4_sm89 import (
            allocate_w4a4_workspace,
        )

        rows = int(inputs.shape[0])
        padded_rows = ((rows + 255) // 256) * 256
        stream_id = _current_cuda_stream_id(inputs.device)
        with_norm = self.fused_norm_weight is not None
        key = (
            str(inputs.device),
            str(inputs.dtype),
            padded_rows,
            int(packed.padded_rank),
            stream_id,
            with_norm,
        )
        workspace = self._native_w4a4_workspace_cache.get(key)
        if workspace is not None:
            return workspace
        workspace = allocate_w4a4_workspace(
            rows,
            packed,
            with_lora_rank=int(packed.padded_rank),
            with_row_scales=with_norm,
        )
        if len(self._native_w4a4_workspace_cache) >= 8:
            self._native_w4a4_workspace_cache.clear()
        self._native_w4a4_workspace_cache[key] = workspace
        return workspace

    def _native_w4a4_smalln_enabled(self, flat: torch.Tensor) -> bool:
        """Pick the BLOCK_N=64 GEMM for short-prefill shapes when available."""

        # The RMSNorm-fused runner currently lives in the base extension only.
        if self.fused_norm_weight is not None:
            return False
        try:
            from xqt.operator_opt.kernels.cute.svdq_w4a4_sm89 import (
                native_w4a4_smalln_available,
                smalln_w4a4_beneficial,
            )

            if not native_w4a4_smalln_available(build=False):
                return False
            padded_rows = ((int(flat.shape[0]) + 255) // 256) * 256
            padded_output = ((self.output_features + 127) // 128) * 128
            return bool(smalln_w4a4_beneficial(padded_rows, padded_output))
        except Exception:
            return False

    def _can_use_cuda_fused(self, inputs: torch.Tensor) -> bool:
        native_allowed, _ = self._native_w4a4_gate(inputs)
        if native_allowed:
            return True
        tilelang_allowed, _ = self._cuda_fused_gate(inputs)
        return tilelang_allowed

    def _cuda_fused_gate(self, inputs: torch.Tensor) -> tuple[bool, str]:
        """Return the runtime promotion decision and an inspectable reason."""

        if not self._cuda_fused_enabled or not inputs.is_cuda:
            return False, "CUDA SVD fusion is disabled or the input is not CUDA"
        if inputs.dtype != torch.float16:
            return False, "SVD fused CUDA requires float16 activations"
        if self.down_proj.weight.dtype != torch.float16:
            return False, "SVD fused CUDA requires float16 down-projection weights"
        if self.up_proj.weight.dtype != torch.float16:
            return False, "SVD fused CUDA requires float16 up-projection weights"
        if inputs.ndim < 1 or inputs.shape[-1] != self.input_features:
            return False, "SVD fused CUDA input trailing dimension is misaligned"
        rows = int(inputs.reshape(-1, self.input_features).shape[0])
        if rows % 64 != 0:
            return False, "SVD fused CUDA requires M to be a multiple of 64"
        if self.input_features % 64 != 0:
            return False, "SVD fused CUDA requires K to be a multiple of 64"
        if self.output_features % 64 != 0:
            return False, "SVD fused CUDA requires N to be a multiple of 64"
        if not self.packed_residual.is_cuda or not self.residual_scale.is_cuda:
            return False, "SVD fused CUDA requires packed residual buffers on CUDA"

        major, minor = torch.cuda.get_device_capability(inputs.device)
        target_arch = f"sm_{major}{minor}"
        try:
            from xqt.operator_opt.kernels.tilelang.svd_fused import (
                resolve_svd_fused_schedule,
            )

            schedule, reason = resolve_svd_fused_schedule(
                rows,
                self.output_features,
                self.input_features,
                self.rank,
                target_arch=target_arch,
            )
        except Exception as exc:
            return False, f"SVD fused CUDA schedule resolution failed: {exc}"
        if schedule is None:
            return False, reason
        return True, reason

    @classmethod
    def from_linear(
        cls,
        module: nn.Linear,
        *,
        rank: int,
        group_size: int = 128,
        quant_dtype: str = "int4",
    ) -> "SVDQuantLinear":
        """Build an SVDQuantLinear from a regular nn.Linear via SVD decomposition."""
        weight = module.weight.detach()
        weight_2d = weight.reshape(module.out_features, module.in_features)
        if weight_2d.shape[0] != module.out_features:
            weight_2d = weight_2d.reshape(module.out_features, -1)

        # 1. SVD decomposition
        decomp = decompose_weight_svd(weight_2d, rank=rank)

        # 2. Extract low-rank components
        L1, L2 = decomp.low_rank_components()  # L1: (r, in), L2: (out, r)

        # 3. Compute residual and quantize
        weight_res = decomp.residual_weight(weight_2d).to(torch.float32)
        normalized_group_size = max(1, min(int(group_size), int(module.in_features)))
        if quant_dtype in _SUPPORTED_RESIDUAL_QUANT_DTYPES:
            packed_residual, residual_scale, padded_in = _quantize_residual_int4(
                weight_res, group_size=normalized_group_size
            )
        else:
            raise ValueError(
                f"Unsupported quant_dtype '{quant_dtype}' for SVDQuant. "
                "Supported: fp4, int4"
            )

        bias = None if module.bias is None else module.bias.detach().to(torch.float32)

        return cls(
            down_weight=L1,
            up_weight=L2,
            packed_residual=packed_residual,
            residual_scale=residual_scale,
            bias=bias,
            input_features=module.in_features,
            output_features=module.out_features,
            group_size=normalized_group_size,
            padded_input_features=padded_in,
            quant_dtype=quant_dtype,
        )

    def set_activation_materialize_hint(
        self,
        *,
        activation_scale_mode: str = "dynamic",
        activation_scale: torch.Tensor | float | None = None,
    ) -> None:
        self._pending_activation_scale_mode = str(activation_scale_mode)
        self._pending_activation_scale = activation_scale

    def materialize_compute(self, spec: Any) -> nn.Module:
        """Bind residual compute from ModuleComputeSpec (Infer / post-quant).

        Dispatch via the materialize registry keyed on
        ``(self.xqt_storage_protocol, preferred_mode, compute_precision)``.
        Falls back to the concrete ``SVDQuantInt8MmaLinear`` when no builder
        is registered for the requested triple.

        ``spec.execution`` carries the ``activation_scale_mode`` (static enables
        fused activation-quant kernels) and ``min_int8_rows`` (small-M bf16
        fallback) knobs, applied to whichever runtime module is produced.
        """
        from xqt.contracts.compute import (
            ModuleComputeSpec,
            ModuleExecutionSpec,
            normalize_compute_contract,
        )

        if not isinstance(spec, ModuleComputeSpec):
            if isinstance(spec, Mapping):
                spec = ModuleComputeSpec.from_mapping(spec)
            else:
                raise TypeError("materialize_compute expects ModuleComputeSpec")

        residual_contract = None
        preferred: list[str] = list(spec.preferred_engines)
        for branch in spec.branches:
            if str(branch.get("name", "")) != "quant_residual":
                continue
            raw = branch.get("compute_contract")
            if raw is not None:
                residual_contract = normalize_compute_contract(str(raw))
            preferred = list(branch.get("preferred_engines") or preferred)
            break
        if residual_contract is None:
            residual_contract = normalize_compute_contract(spec.compute_contract)
        if residual_contract not in {
            "int8_mma",
            "w4_storage_int8_mma",
            "mix_fp4_int8_mma",
        }:
            return self

        engine = preferred[0] if preferred else "auto"
        meta = dict(spec.metadata or {})
        fallback = str(meta.get("fallback_engine", "torch_int_mm"))
        execution = spec.execution or ModuleExecutionSpec()
        if execution.activation_scale_mode != "dynamic":
            act_mode = execution.activation_scale_mode
        else:
            act_mode = str(
                meta.get("activation_scale_mode", self._pending_activation_scale_mode)
            )
        act_scale = meta.get("activation_scale", self._pending_activation_scale)
        min_rows = int(execution.min_int8_rows)
        mode = spec.preferred_mode or "split"
        compute_precision = execution.compute_precision

        # Registry lookup
        builder = get_materializer(
            self.xqt_storage_protocol,
            mode=mode,
            compute_precision=compute_precision,
        )
        if builder is not None:
            return builder(
                self,
                act_mode=act_mode,
                act_scale=act_scale,
                min_rows=min_rows,
                meta=meta,
                engine=engine,
                fallback=fallback,
            )

        # No builder registered for this (kind, mode, precision) triple.
        # Return self unchanged - callers can detect and raise.
        return self

    def dequantize_residual(self) -> torch.Tensor:
        """Dequantize the packed residual weight for reference computation."""
        if self.quant_dtype in _SUPPORTED_RESIDUAL_QUANT_DTYPES:
            packed = self.packed_residual
            scale = self.residual_scale
            if not isinstance(packed, torch.Tensor) or not isinstance(scale, torch.Tensor):
                raise RuntimeError(
                    "packed_residual and residual_scale must be tensors"
                )
            can_cache = _can_mutate_runtime_cache()
            signature: tuple[Any, ...] | None = None
            if can_cache:
                signature = (
                    str(packed.device),
                    int(getattr(packed, "_version", 0)),
                    tuple(int(dim) for dim in packed.shape),
                    str(scale.device),
                    int(getattr(scale, "_version", 0)),
                    tuple(int(dim) for dim in scale.shape),
                )
                if self._dequantized_residual_cache is not None:
                    cached_signature, cached_weight = self._dequantized_residual_cache
                    if cached_signature == signature:
                        return cached_weight
            weight = _dequantize_residual_int4(
                packed,
                scale,
                self.input_features,
                self.group_size,
                self.padded_input_features,
            )
            weight = weight.detach()
            if can_cache and signature is not None:
                self._dequantized_residual_cache = (signature, weight)
            return weight
        raise RuntimeError(f"Unsupported quant_dtype: {self.quant_dtype}")

    def low_rank_weight(self) -> torch.Tensor:
        """Reconstruct the low-rank branch weight W_lr = L2 @ L1."""
        return self.up_proj.weight.data @ self.down_proj.weight.data

    def full_weight_dequant(self) -> torch.Tensor:
        """Reconstruct the full dequantized weight: W_lr + W_res_deq."""
        return self.low_rank_weight() + self.dequantize_residual()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reference forward pass.

        Fusion opportunities (Phase 2-3):
          - FUSE_DOWN: down_proj(x) + act_quantize(x) share input → single kernel
          - FUSE_UP: up_proj(h) + dequant_gemm_epilogue share accumulator → single kernel
        """
        if x.ndim < 1 or int(x.shape[-1]) != self.input_features:
            raise ValueError(
                "SVDQuantLinear input trailing dimension does not match input_features"
            )
        hot_output = self._native_w4a4_hot_forward(x)
        if hot_output is not None:
            return hot_output
        self._last_cuda_fused_used = False
        self._last_cuda_fused_backend = None
        self._last_cuda_fused_fallback_reason = None
        native_allowed, native_gate_reason = self._native_w4a4_gate(x)
        if native_allowed:
            try:
                from xqt.operator_opt.kernels.cute.svdq_w4a4_sm89 import (
                    bind_svdq_w4a4_linear,
                    bind_svdq_w4a4_linear_norm,
                    bind_svdq_w4a4_linear_smalln,
                )

                original_shape = tuple(int(dim) for dim in x.shape[:-1])
                flat = x if x.ndim == 2 else x.reshape(-1, self.input_features)
                with_norm = self.fused_norm_weight is not None
                smalln = self._native_w4a4_smalln_enabled(flat)
                packed = self._native_w4a4_packed(flat, smalln=smalln)
                workspace = self._native_w4a4_workspace(flat, packed)
                if with_norm:
                    native_forward = bind_svdq_w4a4_linear_norm(
                        packed,
                        workspace,
                        rows=int(flat.shape[0]),
                        eps=self.fused_norm_eps,
                    )
                else:
                    bind_function = (
                        bind_svdq_w4a4_linear_smalln if smalln else bind_svdq_w4a4_linear
                    )
                    native_forward = bind_function(
                        packed,
                        workspace,
                        rows=int(flat.shape[0]),
                    )
                fused = native_forward(flat)
                graph_state = None
                if self._cuda_graph_enabled:
                    from xqt.operator_opt.runtime import (
                        capture_cuda_graph_with_static_state,
                        replay_cuda_graph_tensor_callable,
                    )

                    try:
                        candidate = capture_cuda_graph_with_static_state(
                            (flat.contiguous(),),
                            body=native_forward,
                            warmup=self._cuda_graph_warmup,
                        )
                        # Capture only records the kernels; replay once so the
                        # static output holds real results before validating.
                        replay_output = replay_cuda_graph_tensor_callable(
                            candidate, (flat,)
                        )
                        difference = (replay_output.float() - fused.float()).abs().max()
                        if float(difference.item()) <= 1e-3:
                            graph_state = candidate
                    except Exception:
                        graph_state = None
                if len(self._native_w4a4_hot_cache) >= 8:
                    self._native_w4a4_hot_cache.clear()
                if with_norm:
                    backend = "native_w4a4_dynamic_norm"
                else:
                    backend = "native_w4a4_dynamic_smalln" if smalln else "native_w4a4_dynamic"
                self._native_w4a4_hot_cache[
                    self._native_w4a4_hot_key(flat)
                ] = (
                    self._native_w4a4_state_signature(),
                    packed,
                    workspace,
                    native_forward,
                    backend,
                    graph_state,
                )
                self._last_cuda_graph_used = False
                self._last_cuda_fused_used = True
                self._last_cuda_fused_backend = backend
                self._last_cuda_fused_fallback_reason = None
                if x.ndim == 2:
                    return fused
                return fused.reshape(*original_shape, self.output_features)
            except Exception as exc:
                native_gate_reason = f"native W4A4 execution failed: {exc}"
        # From here on the norm is no longer fused into a kernel chain: apply
        # it explicitly so fallback paths keep identical output semantics.
        x = self._apply_fused_norm(x)
        fused_allowed, fused_gate_reason = self._cuda_fused_gate(x)
        if fused_allowed:
            try:
                from xqt.runtime.svd_fusion import fused_svd_forward_cuda
                from xqt.operator_opt.kernels.tilelang.svd_fused import (
                    resolve_svd_fused_schedule,
                )

                original_shape = tuple(int(dim) for dim in x.shape[:-1])
                flat = x.reshape(-1, self.input_features)
                major, minor = torch.cuda.get_device_capability(flat.device)
                schedule, schedule_reason = resolve_svd_fused_schedule(
                    int(flat.shape[0]),
                    self.output_features,
                    self.input_features,
                    self.rank,
                    target_arch=f"sm_{major}{minor}",
                )
                if schedule is None:
                    raise RuntimeError(schedule_reason)
                fused, _ = fused_svd_forward_cuda(
                    self,
                    flat,
                    **schedule.to_dict(),
                )
                self._last_cuda_fused_used = True
                self._last_cuda_fused_backend = "tilelang_dequant_fp16"
                self._last_cuda_fused_fallback_reason = None
                return fused.reshape(*original_shape, self.output_features)
            except Exception as exc:
                self._last_cuda_fused_used = False
                self._last_cuda_fused_backend = None
                self._last_cuda_fused_fallback_reason = (
                    f"{native_gate_reason}; TileLang execution failed: {exc}"
                )
        elif self._cuda_fused_enabled:
            self._last_cuda_fused_used = False
            self._last_cuda_fused_backend = None
            self._last_cuda_fused_fallback_reason = (
                f"{native_gate_reason}; {fused_gate_reason}"
            )

        self._last_cuda_fused_used = False
        self._last_cuda_fused_backend = None
        device = x.device
        dtype = x.dtype

        # Main path: dequant residual → fp16 GEMM ([FUSE_UP candidate: epilogue fusion])
        weight_deq = self.dequantize_residual().to(device=device, dtype=dtype)
        y_main = F.linear(x, weight_deq)

        # Low-rank correction ([FUSE_DOWN candidate: input-sharing with act_quantize])
        h = self.down_proj(x)  # (batch, r)
        y_lora = self.up_proj(h)  # (batch, out)  [FUSE_UP candidate: add to accum]

        y = y_main + y_lora
        if self.bias is not None:
            y = y + self.bias.to(device=device, dtype=dtype)
        return y

    def execution_metadata(self) -> dict[str, Any]:
        implementation = "reference_svd_low_rank_plus_dequant_residual"
        if self._last_cuda_fused_backend == "native_w4a4_dynamic":
            implementation = "native_svdq_w4a4_dynamic_lora"
        elif self._last_cuda_fused_backend == "native_w4a4_dynamic_smalln":
            implementation = "native_svdq_w4a4_dynamic_lora_smalln"
        elif self._last_cuda_fused_backend == "native_w4a4_dynamic_norm":
            implementation = "native_svdq_w4a4_dynamic_norm_fused_lora"
        elif self._last_cuda_fused_backend == "tilelang_dequant_fp16":
            implementation = "tilelang_svd_dequant_low_rank_fused"
        return {
            "implementation": implementation,
            "compute_contract": "composite_add",
            "residual_storage": "packed_signed_int4_group_scale",
            "residual_compute": "dequant_fp16",
            "cuda_fusion_enabled": bool(self._cuda_fused_enabled),
            "cuda_fused_used": bool(self._last_cuda_fused_used),
            "cuda_fused_backend": self._last_cuda_fused_backend,
            "cuda_fused_fallback_reason": self._last_cuda_fused_fallback_reason,
            "cuda_graph_enabled": bool(self._cuda_graph_enabled),
            "cuda_graph_used": bool(self._last_cuda_graph_used),
            "fused_norm_enabled": bool(self.fused_norm_weight is not None),
        }

class SVDQuantInt8MmaLinear(nn.Module):
    """SVDQuant Linear with packed 4-bit residual storage and W8A8 MMA execution.

    The low-rank branch remains at the source module precision. The residual is
    stored as groupwise packed signed 4-bit values (``quant_dtype`` is ``fp4`` or
    ``int4``) and re-targeted to the existing W8A8 INT8 MMA runtime contract. This is not
    native FP4/INT4 MMA.
    """

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
                f"Unsupported quant_dtype '{quant_dtype}' for SVDQuantInt8MmaLinear. "
                f"Supported: {allowed}"
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
        # ``nn.Module._apply`` transforms Parameters and buffers, but not
        # dtype-valued configuration fields.  When the caller uses the normal
        # ``module.half()``/``module.bfloat16()`` API, keep an output dtype that
        # previously followed the source weights in sync with the moved module;
        # otherwise the sm_89 FP16 CUDA path is rejected and falls back to
        # torch_int_mm even though all operands are half precision.
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
    def from_linear(
        cls,
        module: nn.Linear,
        *,
        rank: int,
        group_size: int = 128,
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
    ) -> "SVDQuantInt8MmaLinear":
        """Build the W4-storage / INT8-MMA SVDQuant runtime module."""

        reference = SVDQuantLinear.from_linear(
            module,
            rank=rank,
            group_size=group_size,
            quant_dtype=quant_dtype,
        )
        return cls(
            down_weight=reference.down_proj.weight.detach(),
            up_weight=reference.up_proj.weight.detach(),
            packed_residual=reference.packed_residual.detach(),
            residual_scale=reference.residual_scale.detach(),
            bias=reference.bias,
            input_features=reference.input_features,
            output_features=reference.output_features,
            group_size=reference.group_size,
            padded_input_features=reference.padded_input_features,
            output_dtype=module.weight.dtype,
            quant_dtype=reference.quant_dtype,
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
        """Enable native W8A8 two-stage fusion or the compiled split fallback."""

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
            return False, f"native SVDQuant W8A8 currently targets sm_89, got sm_{major}{minor}"
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
                return False, "native SVDQuant W8A8 requires N/K multiples of 4 and rank <= 1024"
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
        """Compile-friendly split path with no Python-side fallback decisions."""

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
        """Run W8A8 MMA on the residual and add the high-precision low-rank path."""

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

class SVDQuantFp8Linear(nn.Module):
    """SVDQuant split module: fp16 low-rank branch + tensorwise FP8 residual.

    The SVDQuant-faithful form for Ada (sm_89+): the low-rank branch stays at
    source precision (absorbs outliers), while the residual runs as a tensorwise
    FP8 ``_scaled_mm`` (``Fp8MmaLinear``). The two extra low-rank GEMMs are the
    known split overhead; call :meth:`enable_fusion` to fold the branch chain
    with ``torch.compile`` and recover most of it at large M. Small-M inputs are
    routed to a dense fp16 GEMM inside the residual module (``min_fp8_rows``).
    """

    def __init__(
        self,
        *,
        down_weight: torch.Tensor,
        up_weight: torch.Tensor,
        residual_weight: torch.Tensor,
        bias: torch.Tensor | None,
        input_features: int,
        output_features: int,
        output_dtype: torch.dtype = torch.float16,
        activation_scale_mode: str = "static",
        activation_scale: torch.Tensor | float | None = None,
        min_fp8_rows: int = 0,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        from xqt.runtime.modules.fp8_mma_linear import Fp8MmaLinear

        self.input_features = int(input_features)
        self.output_features = int(output_features)
        self.output_dtype = output_dtype
        rank = int(down_weight.shape[0])
        self.rank = rank
        self.xqt_storage_protocol = "svd_low_rank_plus_residual"
        lr_dtype = down_weight.dtype if down_weight.dtype in {
            torch.float16,
            torch.bfloat16,
        } else output_dtype
        self.down_proj = nn.Linear(self.input_features, rank, bias=False, dtype=lr_dtype)
        self.down_proj.weight.data = down_weight.detach().to(lr_dtype).clone()
        self.up_proj = nn.Linear(rank, self.output_features, bias=False, dtype=lr_dtype)
        self.up_proj.weight.data = up_weight.detach().to(lr_dtype).clone()
        # Residual GEMM carries the bias so a single _scaled_mm epilogue applies it.
        self.residual_fp8 = Fp8MmaLinear.from_dense_weight(
            residual_weight,
            bias=bias,
            input_features=self.input_features,
            output_features=self.output_features,
            output_dtype=output_dtype,
            activation_scale_mode=activation_scale_mode,
            activation_scale=activation_scale,
            min_fp8_rows=int(min_fp8_rows),
            eps=float(eps),
        )
        self._fused_forward: Any = None

    def _apply(self, fn: Any) -> "SVDQuantFp8Linear":
        """Move child runtime modules and discard device-specific compiled code."""

        super()._apply(fn)
        self._fused_forward = None
        return self

    def enable_fusion(self, *, mode: str = "reduce-overhead") -> bool:
        """Fold the branch chain with torch.compile; return True on success.

        Only the lean compute (clean tensor ops) is compiled - metadata writes
        and the small-M fallback stay in eager ``forward`` so Inductor sees no
        graph breaks. Fusion recovers the low-rank overhead only in large-M
        (compute-bound) regimes and degrades gracefully to eager on failure.
        """
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
        self._fused_forward = None

    def _low_rank(self, inputs: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        low_rank_inputs = inputs.to(dtype=self.down_proj.weight.dtype)
        return self.up_proj(self.down_proj(low_rank_inputs)).to(
            device=ref.device, dtype=ref.dtype
        )

    def _compute_lean(self, inputs: torch.Tensor) -> torch.Tensor:
        """Compile-friendly path: lean FP8 residual + low-rank, no side effects."""
        residual_output = self.residual_fp8.lean_forward(inputs)
        return residual_output + self._low_rank(inputs, residual_output)

    def _compute_guarded(self, inputs: torch.Tensor) -> torch.Tensor:
        """Eager path: full residual forward (metadata + small-M fallback)."""
        residual_output = self.residual_fp8(inputs)
        return residual_output + self._low_rank(inputs, residual_output)

    def dequantize_residual(self) -> torch.Tensor:
        return self.residual_fp8.dequantize_weight()

    def low_rank_weight(self) -> torch.Tensor:
        return self.up_proj.weight @ self.down_proj.weight

    def full_weight_dequant(self) -> torch.Tensor:
        low_rank_weight = self.low_rank_weight()
        return low_rank_weight + self.dequantize_residual().to(
            dtype=low_rank_weight.dtype, device=low_rank_weight.device
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim < 1 or inputs.shape[-1] != self.input_features:
            raise ValueError(
                "SVDQuantFp8Linear input trailing dimension does not match "
                "input_features"
            )
        # Route small-M / non-CUDA to the guarded eager path (fp16 fallback);
        # only send adequately-sized CUDA inputs to the fused lean path.
        rows = int(inputs.reshape(-1, self.input_features).shape[0])
        residual = self.residual_fp8
        use_fused = (
            self._fused_forward is not None
            and inputs.is_cuda
            and not (0 < residual.min_fp8_rows and rows < residual.min_fp8_rows)
        )
        if use_fused:
            return self._fused_forward(inputs)
        return self._compute_guarded(inputs)

    def execution_metadata(self) -> dict[str, Any]:
        metadata = self.residual_fp8.execution_metadata()
        metadata.update(
            {
                "implementation": "composite_add_svd_low_rank_plus_fp8_residual",
                "compute_contract": "composite_add",
                "low_rank_branch": "source_precision",
                "residual_compute": "fp8_scaled_mm",
                "fusion_enabled": self._fused_forward is not None,
            }
        )
        return metadata

# ============================================================================
# Materialize registry builders for svd_low_rank_plus_residual
# ============================================================================


@register_materializer("svd_low_rank_plus_residual", mode="collapse", compute_precision="w8a8")
def _build_svd_collapsed_int8(shell: "SVDQuantLinear", **options: Any) -> nn.Module:
    """Merge low-rank + residual into one per-channel INT8 GEMM."""
    from xqt.runtime.modules.int8_mma_linear import Int8MmaLinear
    from xqt.runtime.modules.w4_storage_int8_mma_linear import (
        _channel_int8_from_float_weight,
    )

    meta: Mapping[str, Any] = options["meta"]  # type: ignore[assignment]
    eps = float(meta.get("eps", 1e-6))
    dense_weight = shell.full_weight_dequant()
    qweight_t, channel_scale = _channel_int8_from_float_weight(dense_weight, eps=eps)
    return Int8MmaLinear(
        qweight_t,
        channel_scale,
        bias=None if shell.bias is None else shell.bias.detach(),
        input_features=shell.input_features,
        output_features=shell.output_features,
        engine=str(options["engine"]),
        fallback_engine=str(options["fallback"]),
        block_m=int(meta.get("block_m", 64)),
        block_n=int(meta.get("block_n", 64)),
        block_k=int(meta.get("block_k", 64)),
        threads=int(meta.get("threads", 128)),
        num_stages=int(meta.get("num_stages", 2)),
        output_dtype=shell.down_proj.weight.dtype,
        activation_scale_mode=str(options["act_mode"]),
        activation_scale=options.get("act_scale"),
        activation_quant_block_size=int(meta.get("activation_quant_block_size", 256)),
        eps=eps,
        min_int8_rows=int(options["min_rows"]),
    )


@register_materializer("svd_low_rank_plus_residual", mode="collapse", compute_precision="fp8")
def _build_svd_collapsed_fp8(shell: "SVDQuantLinear", **options: Any) -> nn.Module:
    """Fold low-rank + residual into one tensorwise FP8 Linear."""
    from xqt.runtime.modules.fp8_mma_linear import Fp8MmaLinear

    meta: Mapping[str, Any] = options["meta"]  # type: ignore[assignment]
    source_dtype = shell.down_proj.weight.dtype
    output_dtype = (
        source_dtype
        if source_dtype in {torch.float16, torch.bfloat16}
        else torch.float16
    )
    dense_weight = shell.full_weight_dequant()
    return Fp8MmaLinear.from_dense_weight(
        dense_weight,
        bias=None if shell.bias is None else shell.bias.detach(),
        input_features=shell.input_features,
        output_features=shell.output_features,
        output_dtype=output_dtype,
        activation_scale_mode=str(options["act_mode"]),
        activation_scale=options.get("act_scale"),
        min_fp8_rows=int(options["min_rows"]),
        eps=float(meta.get("eps", 1e-8)),
    )


@register_materializer("svd_low_rank_plus_residual", mode="split", compute_precision="fp8")
def _build_svd_split_fp8(shell: "SVDQuantLinear", **options: Any) -> nn.Module:
    """Keep fp16 low-rank branch + tensorwise FP8 residual."""
    meta: Mapping[str, Any] = options["meta"]  # type: ignore[assignment]
    source_dtype = shell.down_proj.weight.dtype
    output_dtype = (
        source_dtype
        if source_dtype in {torch.float16, torch.bfloat16}
        else torch.float16
    )
    return SVDQuantFp8Linear(
        down_weight=shell.down_proj.weight.detach(),
        up_weight=shell.up_proj.weight.detach(),
        residual_weight=shell.dequantize_residual(),
        bias=None if shell.bias is None else shell.bias.detach(),
        input_features=shell.input_features,
        output_features=shell.output_features,
        output_dtype=output_dtype,
        activation_scale_mode=str(options["act_mode"]),
        activation_scale=options.get("act_scale"),
        min_fp8_rows=int(options["min_rows"]),
        eps=float(meta.get("eps", 1e-8)),
    )


@register_materializer("svd_low_rank_plus_residual", mode="split", compute_precision="w8a8")
def _build_svd_split_int8(shell: "SVDQuantLinear", **options: Any) -> nn.Module:
    """Keep fp16 low-rank branch + W4-storage / INT8-MMA residual."""
    meta: Mapping[str, Any] = options["meta"]  # type: ignore[assignment]
    residual_dtype = str(
        meta.get("quant_dtype", getattr(shell, "quant_dtype", "fp4"))
    ).lower()
    return SVDQuantInt8MmaLinear(
        down_weight=shell.down_proj.weight.detach(),
        up_weight=shell.up_proj.weight.detach(),
        packed_residual=shell.packed_residual.detach(),
        residual_scale=shell.residual_scale.detach(),
        bias=None if shell.bias is None else shell.bias.detach(),
        input_features=shell.input_features,
        output_features=shell.output_features,
        group_size=shell.group_size,
        padded_input_features=shell.padded_input_features,
        output_dtype=shell.down_proj.weight.dtype,
        quant_dtype=residual_dtype,
        engine=str(options["engine"]),
        fallback_engine=str(options["fallback"]),
        block_m=int(meta.get("block_m", 64)),
        block_n=int(meta.get("block_n", 64)),
        block_k=int(meta.get("block_k", 64)),
        threads=int(meta.get("threads", 128)),
        num_stages=int(meta.get("num_stages", 2)),
        activation_scale_mode=str(options["act_mode"]),
        activation_scale=options.get("act_scale"),
        activation_quant_block_size=int(meta.get("activation_quant_block_size", 256)),
        eps=float(meta.get("eps", 1e-6)),
        cache_int8_compute_view=bool(meta.get("cache_int8_compute_view", True)),
        min_int8_rows=int(options["min_rows"]),
    )


__all__ = [
    "LowRankBranch",
    "SVDQuantFp8Linear",
    "SVDQuantLinear",
    "SVDQuantInt8MmaLinear",
    "_build_svd_collapsed_fp8",
    "_build_svd_collapsed_int8",
    "_build_svd_split_fp8",
    "_build_svd_split_int8",
]
