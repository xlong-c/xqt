"""Triton candidate materialization and execution metadata."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from xqt.core.errors import XQTBackendError

from .triton_dequant_wrappers import (
    _TritonDequantGemmWrapper,
    build_triton_dequant_candidate_model,
)
from .backends.triton import run_triton_kernel
from .types import OperatorOptimizationTargetPlan


class _TritonRMSNormWrapper(nn.Module):
    """Standalone half RMSNorm wrapper for Triton operator targets."""

    def __init__(
        self,
        norm: nn.Module,
        *,
        fallback: str,
        settings: dict[str, Any],
    ) -> None:
        super().__init__()
        self.norm = norm
        self.fallback = fallback
        self.settings = dict(settings)
        self.last_execution_mode = "not_run"
        self.last_execution_reason: str | None = None
        self.last_operator_family = "norm"
        self.last_fastpath = "none"
        self.last_dtype = "unknown"
        self.last_kernel_pattern = "rmsnorm"
        self.last_channel_layout = "last_dim"

    def _supports_rmsnorm(self) -> bool:
        return hasattr(self.norm, "gamma") and hasattr(self.norm, "scale")

    def _rmsnorm_weight(self, x: torch.Tensor) -> torch.Tensor:
        gamma = getattr(self.norm, "gamma", None)
        scale = getattr(self.norm, "scale", None)
        if not isinstance(gamma, torch.Tensor) or scale is None:
            raise XQTBackendError("Triton RMSNorm wrapper requires gamma and scale")
        weight = gamma.to(device=x.device, dtype=x.dtype)
        return weight.reshape(-1).contiguous() * float(scale)

    def _rmsnorm_bias(self, x: torch.Tensor) -> torch.Tensor | None:
        bias = getattr(self.norm, "bias", None)
        if isinstance(bias, torch.Tensor):
            return bias.to(device=x.device, dtype=x.dtype).reshape(-1).contiguous()
        return None

    def _is_channel_first_norm(self, x: torch.Tensor) -> bool:
        channel_first = bool(getattr(self.norm, "channel_first", False))
        return channel_first and x.ndim >= 3

    def _reference(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)

    def _precision_name(self, x: torch.Tensor) -> str:
        if x.dtype == torch.float16:
            return "half"
        if x.dtype == torch.bfloat16:
            return "bf16"
        return str(x.dtype).removeprefix("torch.")

    def _run_triton_or_reference(self, x: torch.Tensor) -> torch.Tensor:
        weight = self._rmsnorm_weight(x)
        bias = self._rmsnorm_bias(x)
        if self._is_channel_first_norm(x):
            kernel_pattern = "rmsnorm_channel_first"
            eps = float(self.settings.get("eps", 1e-12))
            kernel_args: tuple[torch.Tensor, ...] = (x, weight)
            kernel_kwargs = {
                "bias": bias,
                "eps": eps,
                "block_size": int(self.settings.get("block_size", 1024)),
                "sites_per_program": int(self.settings.get("sites_per_program", 4)),
                "num_warps": int(self.settings.get("num_warps", 4)),
                "num_stages": int(self.settings.get("num_stages", 4)),
                "fallback": self.fallback,
            }
            self.last_channel_layout = "channel_first"
        else:
            kernel_pattern = "rmsnorm"
            eps = float(self.settings.get("eps", 1e-6))
            kernel_args = (x, weight)
            kernel_kwargs = {
                "eps": eps,
                "block_size": int(self.settings.get("block_size", 1024)),
                "num_warps": int(self.settings.get("num_warps", 4)),
                "num_stages": int(self.settings.get("num_stages", 4)),
                "fallback": self.fallback,
            }
            self.last_channel_layout = "last_dim"
        self.last_kernel_pattern = kernel_pattern
        try:
            return run_triton_kernel(kernel_pattern, *kernel_args, **kernel_kwargs)
        except Exception as exc:
            if self.fallback != "eager":
                raise
            self.last_execution_mode = "reference_fallback"
            self.last_execution_reason = f"Triton rmsnorm runtime fallback: {exc}"
            self.last_fastpath = "eager_reference_fallback"
            return self._reference(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._supports_rmsnorm():
            self.last_execution_mode = "reference_fallback"
            self.last_execution_reason = "module does not expose RMSNorm gamma/scale parameters"
            self.last_fastpath = "eager_reference_fallback"
            self.last_dtype = str(x.dtype).removeprefix("torch.")
            return self._reference(x)
        self.last_dtype = str(x.dtype).removeprefix("torch.")
        precision_name = self._precision_name(x)
        is_channel_first = self._is_channel_first_norm(x)
        self.last_kernel_pattern = (
            "rmsnorm_channel_first" if is_channel_first else "rmsnorm"
        )
        self.last_channel_layout = "channel_first" if is_channel_first else "last_dim"
        self.last_execution_mode = "cuda_triton_entry" if x.is_cuda else "reference_fallback"
        kernel_suffix = "channel_first_norm" if is_channel_first else "rmsnorm"
        self.last_fastpath = (
            f"triton_{precision_name}_{kernel_suffix}"
            if self.last_execution_mode == "cuda_triton_entry"
            else "eager_reference_fallback"
        )
        self.last_execution_reason = (
            None
            if self.last_execution_mode == "cuda_triton_entry"
            else "Triton rmsnorm kernel requires CUDA tensors; using configured fallback."
        )
        if self.last_execution_mode == "cuda_triton_entry":
            return self._run_triton_or_reference(x)
        return self._reference(x)

    def execution_metadata(self) -> dict[str, Any]:
        kernel_kind = (
            "minimal_cuda_jit"
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
            "kernel_constraints": {
                "dtype": self.last_dtype,
                "supported_patterns": ["rmsnorm", "rmsnorm_channel_first"],
                "operator_families": ["norm"],
                "normalized_last_dim_only": self.last_channel_layout == "last_dim",
                "supports_channel_first": True,
                "channel_layout": self.last_channel_layout,
                "supported_dtypes": ["float16", "bfloat16"],
            },
            "fallback": self.fallback,
            "settings": dict(self.settings),
        }


def supports_triton_rmsnorm(module: nn.Module | None) -> bool:
    """Return whether a module exposes the RMSNorm protocol used by Triton."""

    return module is not None and hasattr(module, "gamma") and hasattr(module, "scale")


def _feedforward_type() -> type[nn.Module]:
    """Resolve the XQT FeedForward facade without a module-import cycle."""

    from xqt.nn import FeedForward

    return FeedForward


def supports_triton_feedforward(module: nn.Module | None) -> bool:
    """Return whether a module is the XQT FeedForward Triton facade."""

    return module is not None and isinstance(module, _feedforward_type())


def _triton_feedforward_settings(target: OperatorOptimizationTargetPlan) -> dict[str, Any]:
    """Validate and normalize runtime options for a FeedForward candidate."""

    settings = dict(target.options)
    settings["preferred_patterns"] = list(target.patterns or ["feedforward"])
    projection_policies = settings.get("projection_policies")
    if projection_policies is not None and not isinstance(projection_policies, Mapping):
        raise XQTBackendError(
            "Triton feedforward projection_policies must be a mapping when provided"
        )
    return settings


def _build_triton_feedforward_candidate(
    target_model: nn.Module,
    target: OperatorOptimizationTargetPlan,
) -> nn.Module:
    """Copy and configure an XQT FeedForward for the Triton runtime path."""

    if not supports_triton_feedforward(target_model):
        raise XQTBackendError(
            "Triton feedforward target requires an xqt.nn.FeedForward module"
        )
    settings = _triton_feedforward_settings(target)
    candidate = copy.deepcopy(target_model)
    runtime_options = {
        key: settings[key]
        for key in (
            "activation_dtype",
            "weight_dtype",
            "bias_dtype",
            "mma_dtype",
            "accum_dtype",
            "output_dtype",
            "projection_policies",
        )
        if key in settings
    }
    candidate.configure_runtime(engine="triton", **runtime_options)
    setattr(candidate, "_xqt_triton_target_settings", settings)
    return candidate


def build_triton_candidate_model(
    target_model: nn.Module,
    target: OperatorOptimizationTargetPlan,
) -> nn.Module:
    """Materialize one supported Triton candidate for one target."""

    patterns = target.patterns or ["rmsnorm"]
    settings = dict(target.options)
    settings["preferred_patterns"] = list(patterns)
    dequant_patterns = {
        "gemm_int4_dequant",
        "gemm_mxfp8",
        "gemm_mxfp6",
        "gemm_mxfp4",
        "gemm_nvfp4_packed_dequant",
        "fp4_packed_dequant_gemm_epilogue",
        "nvfp4_packed_dequant_gemm_epilogue",
    }
    if patterns == ["feedforward"]:
        return _build_triton_feedforward_candidate(target_model, target)
    if set(patterns).issubset(dequant_patterns):
        return build_triton_dequant_candidate_model(target_model, target)
    if patterns != ["rmsnorm"]:
        raise XQTBackendError(
            "built-in Triton executor currently supports rmsnorm, feedforward, and low-bit dequant GEMM patterns"
        )
    if supports_triton_rmsnorm(target_model):
        return _TritonRMSNormWrapper(
            target_model,
            fallback=target.fallback,
            settings=settings,
        )
    norm = getattr(target_model, "norm", None)
    if supports_triton_rmsnorm(norm):
        candidate = copy.deepcopy(target_model)
        candidate_norm = candidate.get_submodule("norm")
        candidate.norm = _TritonRMSNormWrapper(
            candidate_norm,
            fallback=target.fallback,
            settings=settings,
        )
        return candidate
    for child_name, child in target_model.named_children():
        if supports_triton_rmsnorm(child):
            candidate = copy.deepcopy(target_model)
            setattr(
                candidate,
                child_name,
                _TritonRMSNormWrapper(
                    candidate.get_submodule(child_name),
                    fallback=target.fallback,
                    settings=settings,
                ),
            )
            return candidate
    raise XQTBackendError(
        "Triton rmsnorm target requires a module exposing gamma/scale parameters or a child module with that interface"
    )


def triton_execution_metadata(model: nn.Module) -> dict[str, Any]:
    """Read Triton wrapper execution metadata from a candidate model tree."""

    if supports_triton_feedforward(model):
        runtime_config = model.runtime_config()
        fallback = runtime_config.get("fallback")
        if isinstance(fallback, Mapping):
            execution_mode = "reference_fallback"
            execution_reason = str(fallback.get("reason") or "Triton runtime fallback")
        else:
            execution_mode = "triton_runtime_configured"
            execution_reason = None
        fusion = runtime_config.get("fusion")
        realized_patterns = (
            list(fusion.get("realized_patterns", []))
            if isinstance(fusion, Mapping)
            else []
        )
        return {
            "execution_mode": execution_mode,
            "execution_reason": execution_reason,
            "kernel_kind": (
                "reference_fallback"
                if execution_mode == "reference_fallback"
                else "triton_composed_runtime"
            ),
            "operator_family": "feedforward",
            "kernel_pattern": "feedforward",
            "selected_fastpath": (
                "eager_reference_fallback"
                if execution_mode == "reference_fallback"
                else "triton_feedforward_runtime"
            ),
            "kernel_constraints": {
                "supported_patterns": ["feedforward"],
                "operator_families": ["feedforward"],
                "realized_patterns": realized_patterns,
            },
            "fallback": "eager",
            "settings": dict(
                getattr(model, "_xqt_triton_target_settings", {})
            ),
            "runtime_config": runtime_config,
        }
    if isinstance(model, _TritonRMSNormWrapper):
        return model.execution_metadata()
    if isinstance(model, _TritonDequantGemmWrapper):
        return model.execution_metadata()
    norm = getattr(model, "norm", None)
    if isinstance(norm, _TritonRMSNormWrapper):
        return norm.execution_metadata()
    for module in model.modules():
        if isinstance(module, (_TritonRMSNormWrapper, _TritonDequantGemmWrapper)):
            return module.execution_metadata()
    return {
        "execution_mode": "unknown",
        "execution_reason": None,
    }


__all__ = [
    "_TritonRMSNormWrapper",
    "build_triton_candidate_model",
    "supports_triton_feedforward",
    "supports_triton_rmsnorm",
    "triton_execution_metadata",
]
