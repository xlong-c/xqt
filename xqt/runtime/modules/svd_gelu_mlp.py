"""Inference-only fused GELU MLP for two SVDQuant W4A4 Linear layers."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from xqt.contracts.composite import CompositeAddLinear, CompositeAddModule
from .composite_add import materialize_composite_w4a4
from .composite_add_w4a4 import CompositeAddW4A4Linear
from .svd_w4a4_legacy import SVDQuantLinear


def _is_native_only(module: CompositeAddModule) -> bool:
    """Read the optional frozen-state flag shared by native modules."""

    return bool(getattr(module, "native_only", False))


def _has_fused_norm(module: CompositeAddModule) -> bool:
    checker = getattr(module, "_native_w4a4_has_fused_norm", None)
    return bool(checker()) if callable(checker) else False


def _state_signature(
    module: CompositeAddModule,
    inputs: torch.Tensor,
) -> tuple[Any, ...]:
    if isinstance(module, CompositeAddW4A4Linear):
        return module._state_signature(inputs)
    checker = getattr(module, "_native_w4a4_state_signature", None)
    if not callable(checker):
        raise TypeError("module does not expose a native W4A4 state signature")
    return checker()


def _pack_native(
    module: CompositeAddModule,
    inputs: torch.Tensor,
) -> Any:
    if isinstance(module, CompositeAddW4A4Linear):
        return module._pack(inputs)
    packer = getattr(module, "_native_w4a4_packed", None)
    if not callable(packer):
        raise TypeError("module does not expose a native W4A4 packer")
    return packer(inputs, smalln=False)


def _materialize_gelu_projection(module: CompositeAddModule) -> CompositeAddModule:
    if isinstance(module, (SVDQuantLinear, CompositeAddW4A4Linear)):
        return module
    if isinstance(module, CompositeAddLinear):
        return materialize_composite_w4a4(module)
    return module


class SVDQuantGeluMLP(nn.Module):
    """Pair two SVDQuant Linear layers with the native INT4 GELU fastpath.

    The native path matches Nunchaku's INT4 fused-MLP contract: the first
    epilogue applies tanh-approximate GELU, adds the ``0.171875`` non-negative shift,
    quantizes the hidden activation as unsigned INT4, and feeds the second
    SVDQuant GEMM. It is forward-only and currently targets Ada ``sm_89``.

    Unsupported inputs fall back explicitly to ``fc2(gelu(fc1(x)))`` so the
    module remains usable for autograd, CPU execution, and unsupported CUDA
    targets.
    """

    def __init__(
        self,
        fc1: CompositeAddModule,
        fc2: CompositeAddModule,
        *,
        approximate: str = "tanh",
        native_fusion: bool = True,
    ) -> None:
        super().__init__()
        if not isinstance(fc1, CompositeAddModule) or not isinstance(
            fc2,
            CompositeAddModule,
        ):
            raise TypeError("fc1 and fc2 must both be additive composite modules")
        if fc1.output_features != fc2.input_features:
            raise ValueError("fc1 output_features must match fc2 input_features")
        if approximate not in {"none", "tanh"}:
            raise ValueError("approximate must be 'none' or 'tanh'")
        self.fc1 = fc1
        self.fc2 = fc2
        self.approximate = approximate
        self._native_fusion_enabled = bool(native_fusion)
        self._native_hot_cache: dict[
            tuple[Any, ...],
            tuple[
                tuple[Any, ...],
                Any,
                Any,
                Any,
                Callable[[torch.Tensor], torch.Tensor],
            ],
        ] = {}
        self._last_fused_gelu_mlp_used = False
        self._last_fallback_reason: str | None = None

    @property
    def input_features(self) -> int:
        return self.fc1.input_features

    @property
    def hidden_features(self) -> int:
        return self.fc1.output_features

    @property
    def output_features(self) -> int:
        return self.fc2.output_features

    def _clear_runtime_cache(self) -> None:
        self._native_hot_cache.clear()

    def _apply(self, fn: Any) -> "SVDQuantGeluMLP":
        if _is_native_only(self.fc1) or _is_native_only(self.fc2):
            raise RuntimeError(
                "native-only SVDQuantGeluMLP cannot be moved or cast; "
                "freeze a canonical module again for the target device and dtype"
            )
        super()._apply(fn)
        self._clear_runtime_cache()
        self._last_fused_gelu_mlp_used = False
        self._last_fallback_reason = None
        return self

    def enable_fusion(self, *, mode: str = "reduce-overhead") -> bool:
        """Enable the native fastpath and report whether it is usable now."""

        del mode
        if _is_native_only(self.fc1) or _is_native_only(self.fc2):
            raise RuntimeError(
                "native-only SVDQuantGeluMLP execution configuration is immutable"
            )
        self._native_fusion_enabled = True
        if not torch.cuda.is_available():
            return False
        try:
            from xqt.kernels.ops.quantization import (
                native_w4a4_available,
            )

            return bool(native_w4a4_available(build=False))
        except Exception:
            return False

    def disable_fusion(self) -> None:
        """Disable the native fused path without changing either Linear."""

        if _is_native_only(self.fc1) or _is_native_only(self.fc2):
            raise RuntimeError(
                "native-only SVDQuantGeluMLP cannot disable its only execution path"
            )
        self._native_fusion_enabled = False
        self._last_fused_gelu_mlp_used = False
        self._last_fallback_reason = "native fused GELU MLP is disabled"

    def _state_signature(self, inputs: torch.Tensor) -> tuple[Any, ...]:
        return (
            _state_signature(self.fc1, inputs),
            _state_signature(self.fc2, inputs),
        )

    @staticmethod
    def _hot_key(inputs: torch.Tensor) -> tuple[Any, ...]:
        raw_stream = getattr(torch._C, "_cuda_getCurrentRawStream", None)
        if callable(raw_stream):
            stream_id = int(raw_stream(inputs.device.index))
        else:
            stream_id = int(torch.cuda.current_stream(inputs.device).cuda_stream)
        return (
            inputs.device.index,
            inputs.dtype,
            int(inputs.shape[0]),
            stream_id,
        )

    def _requires_autograd_fallback(self, inputs: torch.Tensor) -> bool:
        if not torch.is_grad_enabled():
            return False
        if _is_native_only(self.fc1) or _is_native_only(self.fc2):
            return True
        if inputs.requires_grad:
            return True
        return any(parameter.requires_grad for parameter in self.parameters())

    def _native_gate(self, inputs: torch.Tensor) -> tuple[bool, str]:
        supported_types = (SVDQuantLinear, CompositeAddW4A4Linear)
        if not isinstance(self.fc1, supported_types) or not isinstance(
            self.fc2,
            supported_types,
        ):
            return False, "native fused GELU MLP requires W4A4 executor modules"
        if not self._native_fusion_enabled:
            return False, "native fused GELU MLP is disabled"
        if not inputs.is_cuda:
            return False, "native fused GELU MLP requires CUDA inputs"
        if self._requires_autograd_fallback(inputs):
            return False, "native fused GELU MLP is forward-only when autograd is required"
        if inputs.ndim < 2 or int(inputs.shape[-1]) != self.input_features:
            return False, "input trailing dimension must match fc1 input_features"
        if inputs.dtype not in {torch.float16, torch.bfloat16}:
            return False, "native fused GELU MLP requires float16 or bfloat16 activations"
        if self.approximate != "tanh":
            return False, "native fused GELU MLP requires approximate='tanh'"
        if self.fc1.quant_dtype != "int4" or self.fc2.quant_dtype != "int4":
            return False, "native fused GELU MLP currently supports INT4 weights only"
        if _has_fused_norm(self.fc1) or _has_fused_norm(self.fc2):
            return False, "native fused GELU MLP does not yet compose with fused_norm"
        for name, module in (("fc1", self.fc1), ("fc2", self.fc2)):
            if isinstance(module, CompositeAddW4A4Linear):
                if module.layout != "main":
                    return False, f"{name} requires the main W4A4 layout"
                allowed, reason = module._native_gate(inputs)
                if not allowed:
                    return False, f"{name}: {reason}"
            elif module.native_only:
                if module._native_only_device != inputs.device:
                    return False, f"{name} frozen device must match activations"
                if module._native_only_dtype != inputs.dtype:
                    return False, f"{name} frozen dtype must match activations"
                if "main" not in module._native_only_layouts:
                    return False, f"{name} requires a frozen main layout"
            else:
                if module.down_proj.weight.dtype != inputs.dtype:
                    return False, f"{name} down-projection dtype must match activations"
                if module.up_proj.weight.dtype != inputs.dtype:
                    return False, f"{name} up-projection dtype must match activations"
                if not module.packed_residual.is_cuda or not module.residual_scale.is_cuda:
                    return False, f"{name} residual buffers must be on CUDA"
                if module.packed_residual.device != inputs.device:
                    return False, f"{name} residual buffers must share the input device"
            if module.rank < 1 or module.rank > 1024:
                return False, f"{name} rank must be between 1 and 1024"
        major, minor = torch.cuda.get_device_capability(inputs.device)
        if (major, minor) != (8, 9):
            return False, f"native fused GELU MLP currently targets sm_89, got sm_{major}{minor}"
        try:
            from xqt.kernels.ops.quantization import (
                native_w4a4_available,
                native_w4a4_shape_supported,
            )

            if not native_w4a4_shape_supported(
                self.fc1.input_features,
                self.fc1.output_features,
            ):
                return False, "fc1 native W4A4 dimensions must be multiples of 4"
            if not native_w4a4_shape_supported(
                self.fc2.input_features,
                self.fc2.output_features,
            ):
                return False, "fc2 native W4A4 dimensions must be multiples of 4"
            if not native_w4a4_available(build=False):
                return False, "native W4A4 backend is unavailable"
        except Exception as exc:
            return False, f"native fused GELU MLP capability check failed: {exc}"
        return True, "native Nunchaku-semantics fused GELU MLP is available"

    def _native_forward(self, inputs: torch.Tensor) -> torch.Tensor:
        from xqt.kernels.ops.quantization import (
            allocate_svdq_w4a4_gelu_mlp_workspace,
            bind_svdq_w4a4_gelu_mlp,
        )

        original_shape = tuple(int(dim) for dim in inputs.shape[:-1])
        flat = inputs.reshape(-1, self.input_features).contiguous()
        key = self._hot_key(flat)
        signature = self._state_signature(flat)
        cached = self._native_hot_cache.get(key)
        if cached is not None and cached[0] == signature:
            output = cached[4](flat)
        else:
            if cached is not None:
                self._native_hot_cache.pop(key, None)
            packed_fc1 = _pack_native(self.fc1, flat)
            packed_fc2 = _pack_native(self.fc2, flat)
            workspace = allocate_svdq_w4a4_gelu_mlp_workspace(
                int(flat.shape[0]),
                packed_fc1,
                packed_fc2,
            )
            native_forward = bind_svdq_w4a4_gelu_mlp(
                packed_fc1,
                packed_fc2,
                workspace,
                rows=int(flat.shape[0]),
            )
            output = native_forward(flat)
            if len(self._native_hot_cache) >= 8:
                self._native_hot_cache.clear()
            self._native_hot_cache[key] = (
                signature,
                packed_fc1,
                packed_fc2,
                workspace,
                native_forward,
            )
        self._last_fused_gelu_mlp_used = True
        self._last_fallback_reason = None
        if inputs.ndim == 2:
            return output
        return output.reshape(*original_shape, self.output_features)

    def _fallback_forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if _is_native_only(self.fc1) or _is_native_only(self.fc2):
            raise RuntimeError(
                "native-only SVDQuantGeluMLP cannot use the sequential fallback"
            )
        return self.fc2(F.gelu(self.fc1(inputs), approximate=self.approximate))

    def forward(
        self,
        inputs: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Run the native fused MLP when eligible, otherwise use PyTorch semantics."""

        del args, kwargs
        allowed, reason = self._native_gate(inputs)
        if allowed:
            try:
                return self._native_forward(inputs)
            except Exception as exc:
                reason = f"native fused GELU MLP execution failed: {exc}"
        self._last_fused_gelu_mlp_used = False
        self._last_fallback_reason = reason
        if _is_native_only(self.fc1) or _is_native_only(self.fc2):
            raise RuntimeError(
                f"native-only SVDQuantGeluMLP cannot fall back: {reason}"
            )
        return self._fallback_forward(inputs)

    def execution_metadata(self) -> dict[str, Any]:
        """Report whether the composite native fastpath handled the last call."""

        implementation = (
            "native_svdq_w4a4_gelu_mlp"
            if self._last_fused_gelu_mlp_used
            else "sequential_svdq_gelu_mlp"
        )
        return {
            "implementation": implementation,
            "compute_contract": "svdq_int4_gelu_mlp",
            "native_fusion_enabled": bool(self._native_fusion_enabled),
            "fused_gelu_mlp_used": bool(self._last_fused_gelu_mlp_used),
            "fallback_reason": self._last_fallback_reason,
            "gelu_approximate": self.approximate,
            "input_features": self.input_features,
            "hidden_features": self.hidden_features,
            "output_features": self.output_features,
            "native_only": bool(
                _is_native_only(self.fc1) and _is_native_only(self.fc2)
            ),
        }


__all__ = ["SVDQuantGeluMLP"]
