"""Legacy SVDQuant W4A4 runtime execution shell."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch import nn

from xqt.runtime.composite_branch import get_materializer
from xqt.contracts.composite import (
    CompositeAddLinear,
    CompositeAddModule,
    initialize_composite_add_storage,
)
from xqt.contracts.packing_int4 import _unpack_int4
from xqt.core.compilation import can_mutate_runtime_cache as _can_mutate_runtime_cache

_SUPPORTED_RESIDUAL_QUANT_DTYPES = frozenset({"fp4", "int4"})
_NATIVE_W4A4_LAYOUTS = frozenset({"main", "smalln"})


def _current_cuda_stream_id(device: torch.device) -> int:
    raw_stream = getattr(torch._C, "_cuda_getCurrentRawStream", None)
    if callable(raw_stream):
        return int(raw_stream(device.index))
    return int(torch.cuda.current_stream(device).cuda_stream)




def _tilelang_runtime_usable() -> bool:
    """Probe the optional TileLang runtime without importing it at module load."""

    try:
        from xqt.kernels.ops.quantization import tilelang_runtime_usable
    except Exception:
        return False
    try:
        return bool(tilelang_runtime_usable())
    except Exception:
        return False


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

class SVDQuantLinear(CompositeAddModule):
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
        initialize_composite_add_storage(
            self,
            down_weight=down_weight,
            up_weight=up_weight,
            packed_residual=packed_residual,
            residual_scale=residual_scale,
            bias=bias,
            input_features=input_features,
            output_features=output_features,
            group_size=group_size,
            padded_input_features=padded_input_features,
            quant_dtype=quant_dtype,
        )
        self.xqt_storage_protocol = "svd_low_rank_plus_residual"
        self._pending_activation_scale: torch.Tensor | float | None = None
        self._pending_activation_scale_mode: str = "dynamic"
        # RMSNorm fusion (native W4A4 only): when set, the module consumes
        # PRE-norm activations and applies the norm inside the activation
        # quantize plus LoRA-down kernel. Fallback paths apply it explicitly.
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
        self._native_only = False
        self._native_only_packed: dict[str, Any] = {}
        self._native_only_device: torch.device | None = None
        self._native_only_dtype: torch.dtype | None = None
        self._native_only_layouts: frozenset[str] = frozenset()
        self._native_only_with_norm = False
        self._native_only_state_signature: tuple[Any, ...] | None = None
        self._native_only_released_bytes = 0

    def _clear_runtime_caches(self) -> None:
        self._dequantized_residual_cache = None
        self._native_w4a4_packed_cache = None
        self._native_w4a4_packed_smalln_cache = None
        self._native_w4a4_workspace_cache.clear()
        self._native_w4a4_hot_cache.clear()

    def _apply(self, fn: Any) -> "SVDQuantLinear":
        """Move registered tensors, then invalidate derived residual storage."""

        if self._native_only:
            raise RuntimeError(
                "native-only SVDQuantLinear cannot be moved or cast; "
                "freeze a canonical module again for the target device and dtype"
            )
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
        if self._native_only:
            raise RuntimeError(
                "native-only SVDQuantLinear execution configuration is immutable"
            )
        if not torch.cuda.is_available():
            return False
        native_usable = False
        try:
            from xqt.kernels.ops.quantization import (
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
        if self._native_only:
            raise RuntimeError(
                "native-only SVDQuantLinear cannot disable its only execution path"
            )
        self._cuda_fused_enabled = False
        self._cuda_graph_enabled = False
        self._last_cuda_graph_used = False

    def set_fused_norm(self, norm_weight: torch.Tensor, *, eps: float = 1e-6) -> None:
        """Fuse an upstream RMSNorm into this linear (native W4A4 hot path).

        After this call the module expects PRE-norm activations: the norm is
        either folded into the native W4A4 quantize/GEMM chain or applied
        explicitly on fallback paths, so outputs stay consistent either way.
        """

        if self._native_only:
            raise RuntimeError(
                "native-only SVDQuantLinear cannot change fused_norm; "
                "configure it before freezing"
            )
        if norm_weight.ndim != 1 or int(norm_weight.numel()) != self.input_features:
            raise ValueError("fused norm weight must match input_features")
        self.fused_norm_weight = norm_weight.detach().clone()
        self.fused_norm_eps = float(eps)
        self._clear_runtime_caches()

    def clear_fused_norm(self) -> None:
        if self._native_only:
            raise RuntimeError(
                "native-only SVDQuantLinear cannot clear fused_norm; "
                "configure it before freezing"
            )
        self.fused_norm_weight = None
        self._clear_runtime_caches()

    @property
    def native_only(self) -> bool:
        """Return whether canonical weights were released after native prepacking."""

        return bool(self._native_only)

    def _native_w4a4_has_fused_norm(self) -> bool:
        if self._native_only:
            return bool(self._native_only_with_norm)
        return self.fused_norm_weight is not None

    def _canonical_storage_bytes(self) -> int:
        tensors: list[torch.Tensor] = []
        for module_name in ("down_proj", "up_proj"):
            module = self._modules.get(module_name)
            if isinstance(module, nn.Module):
                tensors.extend(module.parameters(recurse=True))
        for name in ("packed_residual", "residual_scale", "bias", "fused_norm_weight"):
            tensor = self._buffers.get(name)
            if isinstance(tensor, torch.Tensor):
                tensors.append(tensor)
        seen: set[tuple[str, int]] = set()
        total = 0
        for tensor in tensors:
            key = (str(tensor.device), int(tensor.data_ptr()))
            if key in seen:
                continue
            seen.add(key)
            total += int(tensor.numel()) * int(tensor.element_size())
        return total

    @staticmethod
    def _normalize_native_device(device: torch.device | str) -> torch.device:
        target = torch.device(device)
        if target.type != "cuda":
            raise ValueError("native-only SVDQuantLinear requires a CUDA device")
        if target.index is None:
            target = torch.device("cuda", torch.cuda.current_device())
        return target

    @staticmethod
    def _packed_state_signature(packed_by_layout: Mapping[str, Any]) -> tuple[Any, ...]:
        signature: list[Any] = ["native_only_w4a4"]
        for layout in sorted(packed_by_layout):
            packed = packed_by_layout[layout]
            tensors: list[tuple[Any, ...]] = []
            for name, value in sorted(vars(packed).items()):
                if not isinstance(value, torch.Tensor):
                    continue
                tensors.append(
                    (
                        name,
                        str(value.device),
                        str(value.dtype),
                        int(value.data_ptr()),
                        int(getattr(value, "_version", 0)),
                        tuple(int(dim) for dim in value.shape),
                    )
                )
            signature.append((layout, tuple(tensors)))
        return tuple(signature)

    def freeze_native_inference(
        self,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
        layouts: Iterable[str] = ("main",),
    ) -> int:
        """Irreversibly retain only prepacked native W4A4 inference state.

        Save the canonical checkpoint before calling this method. The frozen
        module cannot be moved, cast, serialized, used with autograd, repacked,
        or sent through a reference fallback. ``layouts`` explicitly selects
        the BLOCK_N=128 ``main`` pack and/or the BLOCK_N=64 ``smalln`` pack.

        Returns the number of canonical parameter/buffer bytes released.
        """

        if self._native_only:
            raise RuntimeError("SVDQuantLinear is already frozen for native inference")
        if self.training:
            raise RuntimeError("freeze_native_inference requires eval mode")
        if dtype not in {torch.float16, torch.bfloat16}:
            raise ValueError("native-only SVDQuantLinear requires float16 or bfloat16")
        target_device = self._normalize_native_device(device)
        if not torch.cuda.is_available():
            raise RuntimeError("freeze_native_inference requires CUDA")
        major, minor = torch.cuda.get_device_capability(target_device)
        if (major, minor) != (8, 9):
            raise RuntimeError(
                f"native-only SVDQuantLinear currently targets sm_89, got sm_{major}{minor}"
            )
        raw_layouts = (layouts,) if isinstance(layouts, str) else tuple(layouts)
        normalized_layouts = frozenset(str(layout).lower() for layout in raw_layouts)
        if not normalized_layouts:
            raise ValueError("freeze_native_inference requires at least one layout")
        unknown_layouts = normalized_layouts - _NATIVE_W4A4_LAYOUTS
        if unknown_layouts:
            names = ", ".join(sorted(unknown_layouts))
            raise ValueError(f"unknown native W4A4 layouts: {names}")

        down_proj = self._modules.get("down_proj")
        up_proj = self._modules.get("up_proj")
        if not isinstance(down_proj, nn.Linear) or not isinstance(up_proj, nn.Linear):
            raise RuntimeError("canonical low-rank projections are unavailable")
        if down_proj.weight.device != target_device or up_proj.weight.device != target_device:
            raise RuntimeError("canonical low-rank weights must already be on the target device")
        if down_proj.weight.dtype != dtype or up_proj.weight.dtype != dtype:
            raise RuntimeError("canonical low-rank weights must already use the target dtype")
        for name in ("packed_residual", "residual_scale", "bias", "fused_norm_weight"):
            tensor = self._buffers.get(name)
            if isinstance(tensor, torch.Tensor) and tensor.device != target_device:
                raise RuntimeError(f"canonical {name} must already be on the target device")

        from xqt.kernels.ops.quantization import (
            native_w4a4_available,
            native_w4a4_shape_supported,
            native_w4a4_smalln_available,
        )

        if not native_w4a4_available(build=False):
            raise RuntimeError("native W4A4 backend is unavailable")
        if not native_w4a4_shape_supported(
            self.input_features,
            self.output_features,
        ):
            raise RuntimeError("native W4A4 requires N and K to be multiples of 4")
        if "smalln" in normalized_layouts and not native_w4a4_smalln_available(build=False):
            raise RuntimeError("native small-N W4A4 backend is unavailable")

        # Do not retain a cache created under torch.inference_mode(): inference
        # tensors have no version counter, so they cannot support the frozen
        # state's mutation guard. Repack under an explicit normal-tensor scope.
        self._clear_runtime_caches()
        with torch.inference_mode(False), torch.no_grad():
            probe = torch.empty(
                (1, self.input_features),
                device=target_device,
                dtype=dtype,
            )
            packed_by_layout: dict[str, Any] = {}
            for layout in sorted(normalized_layouts):
                packed_by_layout[layout] = self._native_w4a4_packed(
                    probe,
                    smalln=layout == "smalln",
                )
        torch.cuda.synchronize(target_device)

        released_bytes = self._canonical_storage_bytes()
        with_norm = self.fused_norm_weight is not None
        state_signature = self._packed_state_signature(packed_by_layout)
        self._clear_runtime_caches()
        self._native_only_packed = packed_by_layout
        self._native_only_device = target_device
        self._native_only_dtype = dtype
        self._native_only_layouts = normalized_layouts
        self._native_only_with_norm = with_norm
        self._native_only_state_signature = state_signature
        self._native_only_released_bytes = released_bytes
        self._modules["down_proj"] = nn.Identity()
        self._modules["up_proj"] = nn.Identity()
        self._buffers["packed_residual"] = None
        self._buffers["residual_scale"] = None
        self._buffers["bias"] = None
        self._buffers["fused_norm_weight"] = None
        self._native_only = True
        self._cuda_fused_enabled = True
        self._cuda_graph_enabled = False
        self._last_cuda_graph_used = False
        self._last_cuda_fused_used = False
        self._last_cuda_fused_backend = None
        self._last_cuda_fused_fallback_reason = None
        return released_bytes

    def train(self, mode: bool = True) -> "SVDQuantLinear":
        if self._native_only and mode:
            raise RuntimeError("native-only SVDQuantLinear is inference-only")
        return super().train(mode)

    def _save_to_state_dict(
        self,
        destination: dict[str, Any],
        prefix: str,
        keep_vars: bool,
    ) -> None:
        if self._native_only:
            raise RuntimeError(
                "native-only SVDQuantLinear cannot be serialized; "
                "save the canonical checkpoint before freezing"
            )
        super()._save_to_state_dict(destination, prefix, keep_vars)

    def _load_from_state_dict(
        self,
        state_dict: Mapping[str, Any],
        prefix: str,
        local_metadata: Mapping[str, Any],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        if self._native_only:
            raise RuntimeError(
                "native-only SVDQuantLinear cannot load state; "
                "rebuild it from a canonical checkpoint"
            )
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def _apply_fused_norm(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.fused_norm_weight
        if weight is None:
            return x
        weight = weight.to(device=x.device, dtype=torch.float32)
        row_scale = torch.rsqrt(
            x.float().pow(2).mean(dim=-1, keepdim=True) + self.fused_norm_eps
        )
        return (x.float() * row_scale * weight).to(x.dtype)

    def _requires_autograd_fallback(self, inputs: torch.Tensor) -> bool:
        """Keep forward-only CUDA kernels out of an active autograd graph."""

        if not torch.is_grad_enabled():
            return False
        if self._native_only:
            return True
        return bool(
            inputs.requires_grad
            or self.down_proj.weight.requires_grad
            or self.up_proj.weight.requires_grad
        )

    def _native_w4a4_gate(self, inputs: torch.Tensor) -> tuple[bool, str]:
        if not self._cuda_fused_enabled or not inputs.is_cuda:
            return False, "native W4A4 fusion is disabled or the input is not CUDA"
        if self._requires_autograd_fallback(inputs):
            return False, "native W4A4 is forward-only when autograd is required"
        if inputs.dtype not in {torch.float16, torch.bfloat16}:
            return False, "native W4A4 requires float16 or bfloat16 activations"
        if inputs.ndim < 1 or int(inputs.shape[-1]) != self.input_features:
            return False, "native W4A4 input trailing dimension is invalid"
        if self.rank < 1 or self.rank > 1024:
            return False, "native W4A4 supports ranks from 1 through 1024"
        if self._native_only:
            if inputs.device != self._native_only_device:
                return False, "native-only W4A4 input device does not match frozen state"
            if inputs.dtype != self._native_only_dtype:
                return False, "native-only W4A4 input dtype does not match frozen state"
        else:
            if self.down_proj.weight.dtype != inputs.dtype:
                return False, "native W4A4 down-projection dtype must match activations"
            if self.up_proj.weight.dtype != inputs.dtype:
                return False, "native W4A4 up-projection dtype must match activations"
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
            from xqt.kernels.ops.quantization import (
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
        if self._native_only:
            signature = self._packed_state_signature(self._native_only_packed)
            if (
                self._native_only_state_signature is None
                or signature != self._native_only_state_signature
            ):
                raise RuntimeError("native-only W4A4 packed state was mutated")
            return signature
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
        if self._requires_autograd_fallback(inputs):
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
            if self._native_only:
                raise RuntimeError("native-only W4A4 packed state was mutated")
            self._native_w4a4_hot_cache.pop(key, None)
            return None
        if graph_state is not None:
            from xqt.kernels.wrappers.runtime import replay_cuda_graph_tensor_callable

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
        if self._native_only:
            layout = "smalln" if smalln else "main"
            if inputs.device != self._native_only_device:
                raise RuntimeError(
                    "native-only W4A4 input device does not match frozen state"
                )
            if inputs.dtype != self._native_only_dtype:
                raise RuntimeError(
                    "native-only W4A4 input dtype does not match frozen state"
                )
            if layout not in self._native_only_layouts:
                raise RuntimeError(
                    f"native-only W4A4 layout '{layout}' was not frozen"
                )
            packed = self._native_only_packed.get(layout)
            if packed is None:
                raise RuntimeError(
                    f"native-only W4A4 packed layout '{layout}' is unavailable"
                )
            return packed

        from xqt.kernels.ops.quantization import (
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
        norm_weight = self.fused_norm_weight
        if norm_weight is not None:
            norm_weight = norm_weight.to(device=inputs.device, dtype=inputs.dtype)
        pack_function = pack_svdq_w4a4_linear_smalln if smalln else pack_svdq_w4a4_linear
        packed = pack_function(
            residual,
            down_weight,
            self.up_proj.weight,
            bias,
            norm_weight=norm_weight,
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
        from xqt.kernels.ops.quantization import (
            allocate_w4a4_workspace,
        )

        rows = int(inputs.shape[0])
        padded_rows = ((rows + 255) // 256) * 256
        stream_id = _current_cuda_stream_id(inputs.device)
        with_norm = self._native_w4a4_has_fused_norm()
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

        try:
            from xqt.kernels.ops.quantization import (
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

        if self._native_only:
            return False, "native-only SVDQuantLinear forbids alternate CUDA paths"
        if not self._cuda_fused_enabled or not inputs.is_cuda:
            return False, "CUDA SVD fusion is disabled or the input is not CUDA"
        if self._requires_autograd_fallback(inputs):
            return False, "SVD fused CUDA is forward-only when autograd is required"
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
            from xqt.kernels.ops.quantization import (
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
    def from_composite(
        cls,
        module: CompositeAddLinear,
    ) -> "SVDQuantLinear":
        """Materialize a canonical composite artifact as the legacy W4A4 shell."""

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
            quant_dtype=module.quant_dtype,
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
        if self._native_only:
            raise RuntimeError(
                "native-only SVDQuantLinear cannot dequantize or use a reference fallback"
            )
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
        if self._native_only:
            raise RuntimeError(
                "native-only SVDQuantLinear released its canonical low-rank weights"
            )
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
                from xqt.kernels.ops.quantization import (
                    bind_svdq_w4a4_linear,
                    bind_svdq_w4a4_linear_norm,
                    bind_svdq_w4a4_linear_smalln,
                    bind_svdq_w4a4_linear_smalln_norm,
                )

                original_shape = tuple(int(dim) for dim in x.shape[:-1])
                flat = x if x.ndim == 2 else x.reshape(-1, self.input_features)
                with_norm = self._native_w4a4_has_fused_norm()
                smalln = self._native_w4a4_smalln_enabled(flat)
                packed = self._native_w4a4_packed(flat, smalln=smalln)
                workspace = self._native_w4a4_workspace(flat, packed)
                if with_norm:
                    if smalln:
                        native_forward = bind_svdq_w4a4_linear_smalln_norm(
                            packed,
                            workspace,
                            rows=int(flat.shape[0]),
                            eps=self.fused_norm_eps,
                        )
                    else:
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
                    from xqt.kernels.wrappers.runtime import (
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
                    backend = (
                        "native_w4a4_dynamic_smalln_norm"
                        if smalln
                        else "native_w4a4_dynamic_norm"
                    )
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
                if self._native_only:
                    self._last_cuda_fused_used = False
                    self._last_cuda_fused_backend = None
                    self._last_cuda_fused_fallback_reason = native_gate_reason
                    raise RuntimeError(
                        "native-only SVDQuantLinear cannot fall back after native failure: "
                        f"{exc}"
                    ) from exc
        if self._native_only:
            self._last_cuda_fused_used = False
            self._last_cuda_fused_backend = None
            self._last_cuda_fused_fallback_reason = native_gate_reason
            raise RuntimeError(
                "native-only SVDQuantLinear cannot fall back: "
                f"{native_gate_reason}"
            )
        # From here on the norm is no longer fused into a kernel chain: apply
        # it explicitly so fallback paths keep identical output semantics.
        x = self._apply_fused_norm(x)
        fused_allowed, fused_gate_reason = self._cuda_fused_gate(x)
        if fused_allowed:
            try:
                from xqt.runtime.svd_fusion import fused_svd_forward_cuda
                from xqt.kernels.ops.quantization import (
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
        elif self._last_cuda_fused_backend == "native_w4a4_dynamic_smalln_norm":
            implementation = "native_svdq_w4a4_dynamic_norm_fused_lora_smalln"
        elif self._last_cuda_fused_backend == "native_w4a4_dynamic_norm":
            implementation = "native_svdq_w4a4_dynamic_norm_fused_lora"
        elif self._last_cuda_fused_backend == "tilelang_dequant_fp16":
            implementation = "tilelang_svd_dequant_low_rank_fused"
        return {
            "implementation": implementation,
            "compute_contract": "composite_add",
            "residual_storage": (
                "native_nunchaku_packed_int4"
                if self._native_only
                else "packed_signed_int4_group_scale"
            ),
            "residual_compute": "native_w4a4" if self._native_only else "dequant_fp16",
            "cuda_fusion_enabled": bool(self._cuda_fused_enabled),
            "cuda_fused_used": bool(self._last_cuda_fused_used),
            "cuda_fused_backend": self._last_cuda_fused_backend,
            "cuda_fused_fallback_reason": self._last_cuda_fused_fallback_reason,
            "cuda_graph_enabled": bool(self._cuda_graph_enabled),
            "cuda_graph_used": bool(self._last_cuda_graph_used),
            "fused_norm_enabled": self._native_w4a4_has_fused_norm(),
            "native_only": bool(self._native_only),
            "native_only_layouts": sorted(self._native_only_layouts),
            "native_only_device": (
                None if self._native_only_device is None else str(self._native_only_device)
            ),
            "native_only_dtype": (
                None if self._native_only_dtype is None else str(self._native_only_dtype)
            ),
            "native_only_released_bytes": int(self._native_only_released_bytes),
        }


__all__ = [
    "LowRankBranch",
    "SVDQuantLinear",
]
