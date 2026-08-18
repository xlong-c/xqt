"""TileLang Dequant GEMM wrapper and dequant-specific helpers/builders."""

from __future__ import annotations

import copy
import inspect
from typing import Any, Callable, Mapping

import torch
from torch import nn
import torch.nn.functional as F

from xqt.core.errors import XQTBackendError
from xqt.contracts.nvfp4 import (
    NVFP4LinearBridge,
    bridge_module_to_nvfp4_linear,
    bridge_module_to_nvfp4_linear_shared,
    infer_nvfp4_tensor_layout,
)

from ..backends.tilelang import get_tilelang_kernel_spec, run_tilelang_kernel
from ..types import OperatorOptimizationTargetPlan
from ._common import _resolved_target_arch
from .linear import _TileLangEagerDenseLinearModule


class _TileLangDequantGemmWrapper(nn.Module):
    """Minimal executable wrapper for dequant GEMM TileLang targets."""

    def __init__(
        self,
        module: nn.Module,
        *,
        fallback: str,
        settings: dict[str, Any],
    ) -> None:
        super().__init__()
        self.module = module
        self.fallback = fallback
        self.settings = dict(settings)
        self.last_execution_mode = "not_run"
        self.last_execution_reason: str | None = None
        self.last_weight_source = "not_run"
        self.last_weight_representation = "unknown"
        self.last_consumes_packed_weight = False
        self.last_unpack_stage: str | None = None
        self.last_kernel_pattern = "dequant_gemm_epilogue"
        self.last_operator_family = "linear"
        self.last_fastpath = "none"
        self._cached_nvfp4_bridge: NVFP4LinearBridge | None = None
        self._preferred_patterns_config = self._preferred_patterns()
        self._dense_linear_bridge = getattr(self.module, "tilelang_dense_linear_args", None)
        self._packed_mxfp_bridge = getattr(
            self.module,
            "tilelang_packed_mxfp_dequant_gemm_args",
            None,
        )
        self._packed_nvfp4_bridge = getattr(self.module, "tilelang_packed_nvfp4_dequant_gemm_args", None)
        self._packed_fp4_bridge = getattr(self.module, "tilelang_packed_dequant_gemm_args", None)
        self._dense_fp4_bridge = getattr(self.module, "tilelang_dequant_gemm_args", None)
        if (
            self._packed_mxfp_bridge is None
            and self._packed_nvfp4_bridge is None
            and self._dense_fp4_bridge is None
            and self._packed_fp4_bridge is None
        ):
            inferred_bridge = bridge_module_to_nvfp4_linear_shared(self.module)
            if inferred_bridge is not None:
                self._cached_nvfp4_bridge = inferred_bridge

    def _weight_only_bridge_metadata(
        self,
        *,
        packed: bool,
        direct_dense_bridge: bool = False,
    ) -> tuple[str, str]:
        bits = getattr(self.module, "bits", None)
        method = getattr(self.module, "method", None)
        if isinstance(bits, int) and method in {"awq", "gptq"}:
            strategy = f"{method}_weight_only_int{bits}"
            if packed:
                return (
                    f"{strategy}_linear_packed_bridge",
                    "packed_signed_int4_plus_group_scale",
                )
            return (
                f"{strategy}_linear_dense_cache_bridge",
                f"dense_dequantized_{strategy}_weight_cache",
            )
        if packed:
            return (
                "fp4_weight_only_linear_packed_bridge",
                "packed_signed_int4_plus_group_scale",
            )
        if direct_dense_bridge:
            return (
                "fp4_weight_only_linear_dense_bridge",
                "dense_unpacked_codes_plus_expanded_scale",
            )
        return (
            "fp4_weight_only_linear_dense_cache_bridge",
            "dense_dequantized_fp4_weight_cache",
        )

    def _can_delegate_dense_native_linear(self, activation: str | None) -> bool:
        return (
            activation is None
            and hasattr(self.module, "dense_weight")
            and callable(getattr(self.module, "forward", None))
        )

    def _resolve_dense_linear_args(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None, str | None] | None:
        if callable(self._dense_linear_bridge):
            if callable(self._packed_fp4_bridge) or callable(self._dense_fp4_bridge):
                self.last_weight_source, self.last_weight_representation = (
                    self._weight_only_bridge_metadata(packed=False)
                )
            elif hasattr(self.module, "mx_precision") and hasattr(self.module, "block_size"):
                mx_precision = int(getattr(self.module, "mx_precision"))
                self.last_weight_source = "mxfp_weight_only_linear_dense_cache_bridge"
                self.last_weight_representation = (
                    f"dense_dequantized_mxfp{mx_precision}_weight_cache"
                )
            else:
                self.last_weight_source = "module_dense_cache_bridge"
                self.last_weight_representation = "dense_dequantized_weight_cache"
            return self._dense_linear_bridge(dtype=x.dtype, device=x.device)
        bridge = self._resolved_nvfp4_bridge()
        if bridge is None:
            return None
        self.last_weight_source = "auto_inferred_nvfp4_dense_cache_bridge"
        self.last_weight_representation = "dense_dequantized_weight_cache"
        return bridge.tilelang_dense_linear_args(dtype=x.dtype, device=x.device)

    def _resolve_packed_nvfp4_args(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, None, int, int, torch.Tensor | None] | None:
        if callable(self._packed_nvfp4_bridge):
            self.last_weight_source = "compressed_tensors_nvfp4_packed_bridge"
            self.last_weight_representation = "packed_nvfp4_e2m1_plus_group_scale"
            return self._packed_nvfp4_bridge(dtype=x.dtype, device=x.device)
        bridge = self._resolved_nvfp4_bridge()
        if bridge is None:
            return None
        self.last_weight_source = "auto_inferred_nvfp4_packed_bridge"
        self.last_weight_representation = "packed_nvfp4_e2m1_plus_group_scale"
        return bridge.tilelang_packed_nvfp4_dequant_gemm_args(dtype=x.dtype, device=x.device)

    def _resolve_packed_mxfp_args(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, None, int, int] | None:
        if not callable(self._packed_mxfp_bridge):
            return None
        mx_precision = int(getattr(self.module, "mx_precision", 4))
        if mx_precision != 4:
            raise XQTBackendError(
                "TileLang packed MXFP dequant GEMM path currently supports MXFP4 only"
            )
        self.last_weight_source = "mxfp_weight_only_linear_packed_bridge"
        self.last_weight_representation = "packed_mxfp4_plus_block_scale"
        return self._packed_mxfp_bridge(dtype=x.dtype, device=x.device)

    @staticmethod
    def _apply_activation(output: torch.Tensor, activation: str | None) -> torch.Tensor:
        if activation is None:
            return output
        if activation == "gelu":
            return F.gelu(output)
        if activation == "silu":
            return F.silu(output)
        if activation == "relu":
            return F.relu(output)
        raise XQTBackendError(f"unsupported activation: {activation}")

    def _preferred_patterns(self) -> list[str]:
        patterns = self.settings.get("preferred_patterns")
        if isinstance(patterns, list):
            return [str(pattern) for pattern in patterns]
        return ["dequant_gemm_epilogue"]

    def _prefer_dense_linear_fastpath(self, x: torch.Tensor) -> bool:
        if (
            callable(self._dense_linear_bridge)
            and self._packed_mxfp_bridge is None
            and self._packed_nvfp4_bridge is None
            and self._packed_fp4_bridge is None
            and self._dense_fp4_bridge is None
        ):
            return True
        mode = str(self.settings.get("linear_fastpath", "auto"))
        if mode == "dense":
            return True
        if mode == "packed":
            return False
        target_arch = _resolved_target_arch(self.settings, x)
        return target_arch == "sm_89"

    def _prefer_native_linear_fastpath(self, x: torch.Tensor) -> bool:
        mode = str(self.settings.get("linear_runtime", "auto"))
        if mode == "native":
            return True
        if mode == "tilelang":
            return False
        target_arch = _resolved_target_arch(self.settings, x)
        return target_arch == "sm_89"

    def _resolved_nvfp4_bridge(self) -> NVFP4LinearBridge | None:
        if (
            callable(self._packed_fp4_bridge)
            or callable(self._dense_fp4_bridge)
            or callable(self._packed_mxfp_bridge)
        ):
            return None
        if self._cached_nvfp4_bridge is not None:
            return self._cached_nvfp4_bridge
        existing_bridge = getattr(self.module, "_bridge", None)
        if isinstance(existing_bridge, NVFP4LinearBridge):
            self._cached_nvfp4_bridge = existing_bridge
            return existing_bridge
        inferred_bridge = bridge_module_to_nvfp4_linear(self.module)
        if inferred_bridge is not None:
            self._cached_nvfp4_bridge = inferred_bridge
        return inferred_bridge

    def _run_tilelang_or_reference(
        self,
        kernel_pattern: str,
        *args: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        try:
            return run_tilelang_kernel(
                kernel_pattern,
                *args,
                **kwargs,
            )
        except Exception as exc:
            if self.fallback != "eager":
                raise
            spec = get_tilelang_kernel_spec(kernel_pattern)
            allowed = set(inspect.signature(spec.reference).parameters)
            filtered_kwargs = {
                key: value
                for key, value in kwargs.items()
                if key in allowed
            }
            self.last_execution_mode = "reference_fallback"
            self.last_execution_reason = (
                f"TileLang {kernel_pattern} runtime fallback: {exc}"
            )
            self.last_fastpath = "eager_reference_fallback"
            self.last_unpack_stage = (
                "one_time_eager_dequant_cache"
                if kernel_pattern == "dense_linear_epilogue"
                else "eager_reference_fallback"
            )
            return spec.reference(*args, **filtered_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        prefer_dense_linear = self._prefer_dense_linear_fastpath(x) and self._preferred_patterns_config in (
            ["dequant_gemm_epilogue"],
            ["dense_linear_epilogue"],
        )
        if prefer_dense_linear:
            dense_args = self._resolve_dense_linear_args(x)
            if dense_args is not None:
                qweight, bias, activation = dense_args
                self.last_kernel_pattern = "dense_linear_epilogue"
                self.last_consumes_packed_weight = False
                self.last_unpack_stage = "one_time_eager_dequant_cache"
                self.last_fastpath = "one_time_eager_dequant_plus_dense_half_gemm"
                if x.is_cuda:
                    self.last_execution_mode = (
                        "cuda_native_fastpath"
                        if self._prefer_native_linear_fastpath(x)
                        else "cuda_tilelang_entry"
                    )
                    self.last_execution_reason = None
                else:
                    self.last_execution_mode = "reference_fallback"
                    self.last_execution_reason = (
                        "TileLang dequant GEMM kernel requires CUDA tensors; using configured fallback."
                    )
                if self.last_execution_mode in {"cuda_native_fastpath", "reference_fallback"}:
                    if self._can_delegate_dense_native_linear(activation):
                        return self.module(x)
                    return self._apply_activation(F.linear(x, qweight, bias), activation)
                if bias is not None:
                    return self._run_tilelang_or_reference(
                        "dense_linear_epilogue",
                        x,
                        qweight,
                        bias,
                        activation=activation,
                        block_m=int(self.settings.get("block_m", 64)),
                        block_n=int(self.settings.get("block_n", 64)),
                        block_k=int(self.settings.get("block_k", 64)),
                        threads=int(self.settings.get("threads", 128)),
                        num_stages=int(self.settings.get("num_stages", 2)),
                        target_arch=self.settings.get("target_arch"),
                        fallback=self.fallback,
                    )
                return self._run_tilelang_or_reference(
                    "dense_linear_epilogue",
                    x,
                    qweight,
                    activation=activation,
                    block_m=int(self.settings.get("block_m", 64)),
                    block_n=int(self.settings.get("block_n", 64)),
                    block_k=int(self.settings.get("block_k", 64)),
                    threads=int(self.settings.get("threads", 128)),
                    num_stages=int(self.settings.get("num_stages", 2)),
                    target_arch=self.settings.get("target_arch"),
                    fallback=self.fallback,
                )
        qweight: torch.Tensor | None = None
        scale: torch.Tensor | None = None
        bias: torch.Tensor | None = None
        activation: str | None = None
        kernel_pattern = "dequant_gemm_epilogue"
        extra_kwargs: dict[str, Any] = {}
        if callable(self._packed_fp4_bridge):
            packed_weight, scale, bias, activation, input_features, group_size = self._packed_fp4_bridge(
                dtype=x.dtype,
                device=x.device,
            )
            qweight = packed_weight
            kernel_pattern = "fp4_packed_dequant_gemm_epilogue"
            extra_kwargs = {
                "input_features": int(input_features),
                "group_size": int(group_size),
            }
            self.last_weight_source, self.last_weight_representation = (
                self._weight_only_bridge_metadata(packed=True)
            )
            self.last_consumes_packed_weight = True
            self.last_fastpath = "packed_fp4_fused_tilelang_kernel"
        elif callable(self._packed_mxfp_bridge):
            packed_mxfp_args = self._resolve_packed_mxfp_args(x)
            if packed_mxfp_args is None:
                raise XQTBackendError(
                    "TileLang packed MXFP dequant GEMM target requires an MXFP4 packed bridge"
                )
            packed_weight, scale, bias, activation, input_features, group_size = packed_mxfp_args
            qweight = packed_weight
            kernel_pattern = "mxfp4_packed_dequant_gemm_epilogue"
            extra_kwargs = {
                "input_features": int(input_features),
                "group_size": int(group_size),
            }
            self.last_consumes_packed_weight = True
            self.last_fastpath = "packed_mxfp4_fused_tilelang_kernel"
        elif callable(self._dense_fp4_bridge):
            qweight, scale, bias, activation = self._dense_fp4_bridge(
                dtype=x.dtype,
                device=x.device,
            )
            self.last_weight_source, self.last_weight_representation = (
                self._weight_only_bridge_metadata(
                    packed=False,
                    direct_dense_bridge=True,
                )
            )
            self.last_consumes_packed_weight = False
            self.last_fastpath = "dense_codes_times_scale"
        else:
            packed_nvfp4_args = self._resolve_packed_nvfp4_args(x)
            if packed_nvfp4_args is not None:
                (
                    packed_weight,
                    scale,
                    bias,
                    activation,
                    input_features,
                    group_size,
                    weight_global_scale,
                ) = packed_nvfp4_args
                qweight = packed_weight
                kernel_pattern = "nvfp4_packed_dequant_gemm_epilogue"
                extra_kwargs = {
                    "input_features": int(input_features),
                    "group_size": int(group_size),
                    "weight_global_scale": weight_global_scale,
                }
                self.last_consumes_packed_weight = True
                self.last_fastpath = "packed_nvfp4_fused_tilelang_kernel"
            else:
                qweight = getattr(self.module, "qweight", None)
                scale = getattr(self.module, "scale", None)
                bias = getattr(self.module, "bias", None)
                activation = getattr(self.module, "activation", None)
                self.last_weight_source = "module_qweight_scale"
                self.last_weight_representation = "dense_qweight_plus_scale"
                self.last_consumes_packed_weight = False
                self.last_fastpath = "dense_codes_times_scale"
        if not isinstance(qweight, torch.Tensor) or (
            kernel_pattern != "dense_linear_epilogue" and not isinstance(scale, torch.Tensor)
        ):
            raise XQTBackendError(
                "TileLang dequant GEMM target requires qweight/scale tensors or a tilelang_dequant_gemm_args bridge"
            )
        tensors = (
            (x, qweight)
            if kernel_pattern == "dense_linear_epilogue" and bias is None
            else (x, qweight, bias)
            if kernel_pattern == "dense_linear_epilogue"
            else (x, qweight, scale)
            if bias is None
            else (x, qweight, scale, bias)
        )
        uses_cuda = all(tensor.is_cuda for tensor in tensors)
        prefers_native_linear = (
            kernel_pattern == "dense_linear_epilogue"
            and uses_cuda
            and self._prefer_native_linear_fastpath(x)
        )
        self.last_execution_mode = (
            "cuda_native_fastpath"
            if prefers_native_linear
            else "cuda_tilelang_entry"
            if uses_cuda
            else "reference_fallback"
        )
        self.last_kernel_pattern = kernel_pattern
        self.last_unpack_stage = (
            "one_time_eager_dequant_cache"
            if kernel_pattern == "dense_linear_epilogue"
            else
            "tilelang_fused_gemm_kernel"
            if uses_cuda and kernel_pattern in {
                "fp4_packed_dequant_gemm_epilogue",
                "mxfp4_packed_dequant_gemm_epilogue",
                "nvfp4_packed_dequant_gemm_epilogue",
            }
            else "eager_reference_fallback"
            if kernel_pattern in {
                "fp4_packed_dequant_gemm_epilogue",
                "mxfp4_packed_dequant_gemm_epilogue",
                "nvfp4_packed_dequant_gemm_epilogue",
            }
            else None
        )
        self.last_execution_reason = (
            None
            if self.last_execution_mode in {"cuda_tilelang_entry", "cuda_native_fastpath"}
            else "TileLang dequant GEMM kernel requires CUDA tensors; using configured fallback."
        )
        tile_kwargs: dict[str, Any] = {
            "activation": activation,
            **extra_kwargs,
            "block_m": int(self.settings.get("block_m", 64)),
            "block_n": int(
                self.settings.get(
                    "block_n",
                    16 if kernel_pattern == "nvfp4_packed_dequant_gemm_epilogue" else 64,
                )
            ),
            "threads": int(self.settings.get("threads", 128)),
            "num_stages": int(self.settings.get("num_stages", 2)),
            "target_arch": self.settings.get("target_arch"),
            "fallback": self.fallback,
        }
        if kernel_pattern == "dense_linear_epilogue":
            tile_kwargs["block_k"] = int(self.settings.get("block_k", 64))
        if kernel_pattern == "nvfp4_packed_dequant_gemm_epilogue":
            tile_kwargs["block_k"] = int(self.settings.get("block_k", 128))
        if kernel_pattern == "dense_linear_epilogue":
            if self.last_execution_mode in {"cuda_native_fastpath", "reference_fallback"}:
                return self._apply_activation(F.linear(x, qweight, bias), activation)
            if bias is not None:
                return self._run_tilelang_or_reference(
                    kernel_pattern,
                    x,
                    qweight,
                    bias,
                    **tile_kwargs,
                )
            return self._run_tilelang_or_reference(
                kernel_pattern,
                x,
                qweight,
                **tile_kwargs,
            )
        return self._run_tilelang_or_reference(
            kernel_pattern,
            x,
            qweight,
            scale,
            bias,
            **tile_kwargs,
        )

    def execution_metadata(self) -> dict[str, Any]:
        kernel_kind = (
            "native_runtime_fastpath"
            if self.last_execution_mode == "cuda_native_fastpath"
            else
            "minimal_cuda_jit"
            if self.last_execution_mode == "cuda_tilelang_entry"
            else "reference_fallback"
            if self.last_execution_mode == "reference_fallback"
            else "unknown"
        )
        return {
            "execution_mode": self.last_execution_mode,
            "execution_reason": self.last_execution_reason,
            "kernel_kind": kernel_kind,
            "kernel_constraints": {
                "dtype": "float16",
                "batch_multiple_of_block_m": True,
                "out_features_multiple_of_block_n": True,
                "supported_activations": [None, "gelu", "silu", "relu"],
                "supported_patterns": [
                    "dense_linear_epilogue",
                    "dequant_gemm_epilogue",
                    "fp4_packed_dequant_gemm_epilogue",
                    "mxfp4_packed_dequant_gemm_epilogue",
                    "nvfp4_packed_dequant_gemm_epilogue",
                ],
                "operator_families": ["linear", "conv", "attention"],
                "supports_fp4_weight_only_linear_bridge": True,
                "supports_packed_fp4_bridge": True,
                "supports_packed_mxfp_bridge": True,
                "supports_packed_nvfp4_bridge": True,
            },
            "kernel_pattern": self.last_kernel_pattern,
            "operator_family": self.last_operator_family,
            "selected_fastpath": self.last_fastpath,
            "weight_source": self.last_weight_source,
            "weight_representation": self.last_weight_representation,
            "consumes_packed_weight": self.last_consumes_packed_weight,
            "unpack_stage": self.last_unpack_stage,
            "fusion_status": (
                "tilelang_dense_half_gemm_epilogue"
                if self.last_unpack_stage == "one_time_eager_dequant_cache"
                else
                "single_tilelang_kernel_for_unpack_dequant_gemm_epilogue"
                if self.last_unpack_stage == "tilelang_fused_gemm_kernel"
                else None
            ),
            "epilogue_stage": (
                "torch_bias_activation"
                if self.last_unpack_stage == "one_time_eager_dequant_cache"
                else
                "tilelang_fused_bias_activation"
                if self.last_unpack_stage == "tilelang_fused_gemm_kernel"
                else None
            ),
            "fallback": self.fallback,
            "settings": dict(self.settings),
        }


# ---------------------------------------------------------------------------
# Dequant-only helpers and builders
# ---------------------------------------------------------------------------


def _module_has_cuda_state(module: nn.Module) -> bool:
    return any(tensor.is_cuda for tensor in module.parameters()) or any(
        tensor.is_cuda for tensor in module.buffers()
    )


def _attach_tilelang_execution_metadata(
    module: nn.Module,
    metadata: Mapping[str, Any],
) -> nn.Module:
    setattr(module, "_xqt_tilelang_execution_metadata", dict(metadata))
    return module


def _infer_module_runtime_spec(module: nn.Module) -> tuple[torch.device, torch.dtype]:
    device: torch.device | None = None
    dtype: torch.dtype | None = None
    saw_only_float32 = False
    for tensor in tuple(module.parameters()) + tuple(module.buffers()):
        device = device or tensor.device
        if not tensor.is_floating_point():
            continue
        dtype = tensor.dtype
        if dtype != torch.float32:
            return device, dtype
        saw_only_float32 = True
    if device is None:
        device = torch.device("cpu")
    if dtype is None:
        return device, torch.float32
    return device, torch.float16 if saw_only_float32 else dtype


def _has_explicit_xqt_dequant_linear_bridge(module: nn.Module) -> bool:
    return bool(
        callable(getattr(module, "tilelang_packed_dequant_gemm_args", None))
        or callable(getattr(module, "tilelang_dequant_gemm_args", None))
        or callable(getattr(module, "tilelang_packed_mxfp_dequant_gemm_args", None))
        or (
            hasattr(module, "mx_precision")
            and callable(getattr(module, "tilelang_dense_linear_args", None))
        )
    )


def _native_dense_metadata(
    *,
    dtype: str,
    selected_fastpath: str,
    weight_source: str,
    fallback: str,
    settings: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "execution_mode": "cuda_native_fastpath",
        "execution_reason": None,
        "kernel_kind": "native_runtime_fastpath",
        "kernel_constraints": {
            "dtype": dtype,
            "batch_multiple_of_block_m": True,
            "out_features_multiple_of_block_n": True,
            "supported_activations": [None, "gelu", "silu", "relu"],
            "supported_patterns": [
                "dense_linear_epilogue",
                "dequant_gemm_epilogue",
                "fp4_packed_dequant_gemm_epilogue",
                "mxfp4_packed_dequant_gemm_epilogue",
                "nvfp4_packed_dequant_gemm_epilogue",
            ],
            "operator_families": ["linear", "conv", "attention"],
            "supports_fp4_weight_only_linear_bridge": True,
            "supports_packed_fp4_bridge": True,
            "supports_packed_mxfp_bridge": True,
            "supports_packed_nvfp4_bridge": True,
        },
        "kernel_pattern": "dense_linear_epilogue",
        "operator_family": "linear",
        "selected_fastpath": selected_fastpath,
        "weight_source": weight_source,
        "weight_representation": "dense_dequantized_weight_cache",
        "consumes_packed_weight": False,
        "unpack_stage": "one_time_eager_dequant_cache",
        "fusion_status": "tilelang_dense_half_gemm_epilogue",
        "epilogue_stage": "torch_bias_activation",
        "fallback": fallback,
        "settings": dict(settings),
    }


def _is_dequant_linear_target(module: nn.Module) -> bool:
    return bool(
        infer_nvfp4_tensor_layout(module) is not None
        or callable(getattr(module, "tilelang_dense_linear_args", None))
        or callable(getattr(module, "tilelang_packed_mxfp_dequant_gemm_args", None))
        or callable(getattr(module, "tilelang_packed_nvfp4_dequant_gemm_args", None))
        or callable(getattr(module, "tilelang_packed_dequant_gemm_args", None))
        or callable(getattr(module, "tilelang_dequant_gemm_args", None))
        or all(hasattr(module, name) for name in ("qweight", "scale"))
    )


def _build_tilelang_dequant_candidate(
    target_model: nn.Module,
    target: OperatorOptimizationTargetPlan,
    settings: Mapping[str, Any],
) -> nn.Module:
    inferred_layout = infer_nvfp4_tensor_layout(target_model)
    is_sm89 = str(settings.get("target_arch") or "") == "sm_89"
    is_dense_pattern = target.patterns == ["dequant_gemm_epilogue"]
    if (
        is_dense_pattern
        and is_sm89
        and inferred_layout is not None
        and not _has_explicit_xqt_dequant_linear_bridge(target_model)
        and _module_has_cuda_state(target_model)
    ):
        bridge = bridge_module_to_nvfp4_linear_shared(target_model)
        if bridge is not None:
            device, dtype = _infer_module_runtime_spec(target_model)
            weight, bias, _ = bridge.tilelang_dense_linear_args(dtype=dtype, device=device)
            linear = nn.Linear(
                bridge.input_features,
                bridge.output_features,
                bias=bias is not None,
                device=device,
                dtype=dtype,
            )
            with torch.no_grad():
                linear.weight.copy_(weight)
                if bias is not None and linear.bias is not None:
                    linear.bias.copy_(bias)
            metadata = _native_dense_metadata(
                dtype=str(dtype),
                selected_fastpath="eager_dense_native_linear_module",
                weight_source="auto_inferred_nvfp4_dense_cache_bridge",
                fallback=target.fallback,
                settings=settings,
            )
            return _attach_tilelang_execution_metadata(
                _TileLangEagerDenseLinearModule(linear, metadata=metadata),
                metadata,
            )
    if (
        is_dense_pattern
        and is_sm89
        and callable(getattr(target_model, "tilelang_dense_linear_args", None))
        and not _has_explicit_xqt_dequant_linear_bridge(target_model)
        and hasattr(target_model, "dense_weight")
        and _module_has_cuda_state(target_model)
    ):
        return _attach_tilelang_execution_metadata(
            target_model,
            _native_dense_metadata(
                dtype="float16",
                selected_fastpath="delegated_native_dense_linear",
                weight_source="module_dense_weight",
                fallback=target.fallback,
                settings=settings,
            ),
        )
    if _is_dequant_linear_target(target_model):
        return _TileLangDequantGemmWrapper(
            target_model,
            fallback=target.fallback,
            settings=dict(settings),
        )
    for child_name, child in target_model.named_children():
        if _is_dequant_linear_target(child):
            copied = copy.deepcopy(target_model)
            copied_child = copied.get_submodule(child_name)
            setattr(
                copied,
                child_name,
                _TileLangDequantGemmWrapper(
                    copied_child,
                    fallback=target.fallback,
                    settings=dict(settings),
                ),
            )
            return copied
    raise XQTBackendError(
        "TileLang dequant GEMM target requires a module with qweight/scale tensors or a tilelang_dequant_gemm_args bridge"
    )
