"""Triton wrappers for packed or dequantized GEMM operator targets."""

from __future__ import annotations

import copy
import inspect
from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch import nn

from xqt.core.errors import XQTBackendError
from xqt.quant.bridges.nvfp4 import (
    NVFP4LinearBridge,
    bridge_module_to_nvfp4_linear,
    bridge_module_to_nvfp4_linear_shared,
    infer_nvfp4_tensor_layout,
)

from .backends.triton import get_triton_kernel_spec, run_triton_kernel
from .types import OperatorOptimizationTargetPlan


class _TritonDequantGemmWrapper(nn.Module):
    """Executable Triton wrapper for packed low-bit linear targets."""

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
        self.last_operator_family = "linear"
        self.last_kernel_pattern = "unknown"
        self.last_fastpath = "none"
        self.last_weight_source = "not_run"
        self.last_weight_representation = "unknown"
        self.last_consumes_packed_weight = False
        self.last_unpack_stage: str | None = None
        self._cached_nvfp4_bridge: NVFP4LinearBridge | None = None
        self._packed_nvfp4_bridge = getattr(
            self.module, "triton_packed_nvfp4_dequant_gemm_args", None
        )
        self._packed_fp4_bridge = getattr(
            self.module, "triton_packed_dequant_gemm_args", None
        )
        self._mxfp_bridge = getattr(self.module, "triton_mxfp_gemm_args", None)
        if self._packed_nvfp4_bridge is None and self._packed_fp4_bridge is None:
            inferred_bridge = bridge_module_to_nvfp4_linear_shared(self.module)
            if inferred_bridge is not None:
                self._cached_nvfp4_bridge = inferred_bridge

    def _resolved_nvfp4_bridge(self) -> NVFP4LinearBridge | None:
        if self._cached_nvfp4_bridge is not None:
            return self._cached_nvfp4_bridge
        existing_bridge = getattr(self.module, "_bridge", None)
        if not isinstance(existing_bridge, NVFP4LinearBridge):
            existing_bridge = getattr(self.module, "bridge", None)
        if isinstance(existing_bridge, NVFP4LinearBridge):
            self._cached_nvfp4_bridge = existing_bridge
            return existing_bridge
        inferred_bridge = bridge_module_to_nvfp4_linear(self.module)
        if inferred_bridge is not None:
            self._cached_nvfp4_bridge = inferred_bridge
        return inferred_bridge

    def _run_triton_or_reference(
        self,
        pattern: str,
        *args: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        try:
            return run_triton_kernel(pattern, *args, fallback=self.fallback, **kwargs)
        except Exception as exc:
            if self.fallback != "eager":
                raise
            spec = get_triton_kernel_spec(pattern)
            allowed = set(inspect.signature(spec.reference).parameters)
            filtered_kwargs = {
                key: value for key, value in kwargs.items() if key in allowed
            }
            self.last_execution_mode = "reference_fallback"
            self.last_execution_reason = f"Triton {pattern} runtime fallback: {exc}"
            self.last_fastpath = "eager_reference_fallback"
            self.last_unpack_stage = "eager_reference_fallback"
            return spec.reference(*args, **filtered_kwargs)

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if callable(self._mxfp_bridge):
            packed_weight, scale, bias, block_size, mx_precision = self._mxfp_bridge(
                dtype=x.dtype,
                device=x.device,
            )
            pattern = f"gemm_mxfp{int(mx_precision)}"
            self.last_kernel_pattern = pattern
            self.last_weight_source = "mxfp_weight_only_linear_packed_bridge"
            self.last_weight_representation = f"packed_mxfp{int(mx_precision)}_plus_block_scale"
            self.last_consumes_packed_weight = True
            self.last_unpack_stage = "triton_device_unpack_then_dense_gemm"
            self.last_fastpath = f"triton_mxfp{int(mx_precision)}_composed_gemm"
            self.last_execution_mode = "cuda_triton_entry" if x.is_cuda else "reference_fallback"
            self.last_execution_reason = (
                None
                if x.is_cuda
                else "Triton low-bit GEMM requires CUDA tensors; using configured fallback."
            )
            activation = getattr(self.module, "activation", None)
            return self._run_triton_or_reference(
                pattern,
                x,
                packed_weight,
                scale,
                bias,
                mx_precision=int(mx_precision),
                block_size=int(block_size),
                activation=activation,
                transpose_b=True,
            )

        if callable(self._packed_fp4_bridge):
            packed_weight, scale, bias, _, input_features, group_size = self._packed_fp4_bridge(
                dtype=x.dtype,
                device=x.device,
            )
            self.last_kernel_pattern = "gemm_int4_dequant"
            self.last_weight_source = "fp4_weight_only_linear_packed_bridge"
            self.last_weight_representation = "packed_signed_int4_plus_group_scale"
            self.last_consumes_packed_weight = True
            self.last_unpack_stage = "triton_device_unpack_then_dense_gemm"
            self.last_fastpath = "triton_fp4_composed_gemm"
            self.last_execution_mode = "cuda_triton_entry" if x.is_cuda else "reference_fallback"
            self.last_execution_reason = (
                None
                if x.is_cuda
                else "Triton low-bit GEMM requires CUDA tensors; using configured fallback."
            )
            activation = getattr(self.module, "activation", None)
            _ = input_features
            return self._run_triton_or_reference(
                "gemm_int4_dequant",
                x,
                packed_weight,
                scale,
                None,
                bias,
                group_size=int(group_size),
                activation=activation,
            )

        packed_nvfp4_args = None
        if callable(self._packed_nvfp4_bridge):
            packed_nvfp4_args = self._packed_nvfp4_bridge(dtype=x.dtype, device=x.device)
            self.last_weight_source = "compressed_tensors_nvfp4_packed_bridge"
        else:
            bridge = self._resolved_nvfp4_bridge()
            if bridge is not None:
                packed_nvfp4_args = bridge.triton_packed_nvfp4_dequant_gemm_args(
                    dtype=x.dtype,
                    device=x.device,
                )
                self.last_weight_source = "auto_inferred_nvfp4_packed_bridge"
        if packed_nvfp4_args is None:
            raise XQTBackendError(
                "Triton dequant GEMM target requires MXFP, FP4, or NVFP4 packed bridge"
            )
        (
            packed_weight,
            scale,
            bias,
            _activation,
            input_features,
            group_size,
            weight_global_scale,
        ) = packed_nvfp4_args
        self.last_kernel_pattern = "gemm_nvfp4_packed_dequant"
        self.last_weight_representation = "packed_nvfp4_e2m1_plus_group_scale"
        self.last_consumes_packed_weight = True
        self.last_unpack_stage = "triton_device_unpack_then_dense_gemm"
        self.last_fastpath = "triton_nvfp4_composed_gemm"
        self.last_execution_mode = "cuda_triton_entry" if x.is_cuda else "reference_fallback"
        self.last_execution_reason = (
            None
            if x.is_cuda
            else "Triton low-bit GEMM requires CUDA tensors; using configured fallback."
        )
        activation = getattr(self.module, "activation", None)
        return self._run_triton_or_reference(
            "gemm_nvfp4_packed_dequant",
            x,
            packed_weight,
            scale,
            bias,
            input_features=int(input_features),
            group_size=int(group_size),
            weight_global_scale=weight_global_scale,
            activation=activation,
        )

    def execution_metadata(self) -> dict[str, Any]:
        kernel_kind = (
            "triton_composed_runtime"
            if self.last_execution_mode == "cuda_triton_entry"
            else "reference_fallback"
        )
        return {
            "execution_mode": self.last_execution_mode,
            "execution_reason": self.last_execution_reason,
            "kernel_kind": kernel_kind,
            "operator_family": self.last_operator_family,
            "kernel_pattern": self.last_kernel_pattern,
            "selected_fastpath": self.last_fastpath,
            "weight_source": self.last_weight_source,
            "weight_representation": self.last_weight_representation,
            "consumes_packed_weight": self.last_consumes_packed_weight,
            "unpack_stage": self.last_unpack_stage,
            "kernel_constraints": {
                "supported_patterns": [
                    "gemm_int4_dequant",
                    "gemm_mxfp8",
                    "gemm_mxfp6",
                    "gemm_mxfp4",
                    "gemm_nvfp4_packed_dequant",
                ],
                "operator_families": ["linear"],
                "supports_packed_fp4_bridge": True,
                "supports_packed_nvfp4_bridge": True,
                "supports_mxfp_bridge": True,
            },
            "fallback": self.fallback,
            "settings": dict(self.settings),
        }


def _is_triton_dequant_linear_target(module: nn.Module) -> bool:
    return bool(
        callable(getattr(module, "triton_packed_dequant_gemm_args", None))
        or callable(getattr(module, "triton_packed_nvfp4_dequant_gemm_args", None))
        or callable(getattr(module, "triton_mxfp_gemm_args", None))
        or infer_nvfp4_tensor_layout(module) is not None
    )


def build_triton_dequant_candidate_model(
    target_model: nn.Module,
    target: OperatorOptimizationTargetPlan,
) -> nn.Module:
    """Build a Triton dequant-GEMM candidate for one supported target."""

    settings = dict(target.options)
    settings["preferred_patterns"] = list(
        target.patterns
        or ["gemm_int4_dequant", "gemm_mxfp4", "gemm_nvfp4_packed_dequant"]
    )
    if _is_triton_dequant_linear_target(target_model):
        return _TritonDequantGemmWrapper(
            target_model,
            fallback=target.fallback,
            settings=settings,
        )
    for child_name, child in target_model.named_children():
        if _is_triton_dequant_linear_target(child):
            copied = copy.deepcopy(target_model)
            copied_child = copied.get_submodule(child_name)
            setattr(
                copied,
                child_name,
                _TritonDequantGemmWrapper(
                    copied_child,
                    fallback=target.fallback,
                    settings=settings,
                ),
            )
            return copied
    raise XQTBackendError(
        "Triton dequant GEMM target requires MXFP, FP4, or NVFP4 packed linear bridges"
    )


__all__ = [
    "_TritonDequantGemmWrapper",
    "build_triton_dequant_candidate_model",
]
