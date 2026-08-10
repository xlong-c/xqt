"""Triton backend adapter for XQT operator optimization."""

from __future__ import annotations

from dataclasses import dataclass, field
import inspect
from typing import Any, Callable

import torch

from xqt.core.errors import XQTBackendError

from ..kernels.triton import (
    TRITON_ATTENTION_KERNEL_METADATA,
    TRITON_KERNEL_METADATA,
    fused_attention_forward_reference,
    fused_attention_forward_triton,
    fused_bias_gelu_reference,
    fused_bias_gelu_triton,
    fused_channel_first_l2norm_reference,
    fused_channel_first_l2norm_triton,
    fused_geglu_reference,
    fused_geglu_triton,
    fused_rope_reference,
    fused_rope_triton,
    fused_rmsnorm_reference,
    fused_rmsnorm_triton,
    fused_rmsnorm_residual_reference,
    fused_rmsnorm_residual_triton,
    fused_swiglu_reference,
    fused_swiglu_triton,
)
from ..kernels.triton.gemm import (
    TRITON_GEMM_KERNEL_METADATA,
    gemm_bf16_triton,
    gemm_fp16_triton,
    gemm_fp8_reference,
    gemm_fp8_triton,
    gemm_int4_dequant_reference,
    gemm_int4_dequant_triton,
    gemm_int8_reference,
    gemm_int8_triton,
    gemm_nvfp4_packed_dequant_reference,
    gemm_nvfp4_packed_dequant_triton,
    gemm_reference,
)
from ..kernels.triton.mxfp_gemm import (
    MXFP_GEMM_KERNEL_METADATA,
    gemm_mxfp_reference,
    gemm_mxfp_triton,
)


@dataclass(frozen=True)
class TritonKernelSpec:
    """One Triton kernel registry entry."""

    pattern: str
    reference: Callable[..., torch.Tensor]
    kernel: Callable[..., torch.Tensor]
    cuda_only: bool = True
    fallback: str = "eager"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "pattern": self.pattern,
            "cuda_only": self.cuda_only,
            "fallback": self.fallback,
            "metadata": dict(self.metadata),
        }


TRITON_KERNEL_REGISTRY: dict[str, TritonKernelSpec] = {
    "attention": TritonKernelSpec(
        pattern="attention",
        reference=fused_attention_forward_reference,
        kernel=fused_attention_forward_triton,
        metadata=dict(TRITON_ATTENTION_KERNEL_METADATA["attention"]),
    ),
    "bias_gelu": TritonKernelSpec(
        pattern="bias_gelu",
        reference=fused_bias_gelu_reference,
        kernel=fused_bias_gelu_triton,
        metadata=dict(TRITON_KERNEL_METADATA["bias_gelu"]),
    ),
    "swiglu": TritonKernelSpec(
        pattern="swiglu",
        reference=fused_swiglu_reference,
        kernel=fused_swiglu_triton,
        metadata=dict(TRITON_KERNEL_METADATA["swiglu"]),
    ),
    "geglu": TritonKernelSpec(
        pattern="geglu",
        reference=fused_geglu_reference,
        kernel=fused_geglu_triton,
        metadata=dict(TRITON_KERNEL_METADATA["geglu"]),
    ),
    "rmsnorm": TritonKernelSpec(
        pattern="rmsnorm",
        reference=fused_rmsnorm_reference,
        kernel=fused_rmsnorm_triton,
        metadata=dict(TRITON_KERNEL_METADATA["rmsnorm"]),
    ),
    "rmsnorm_channel_first": TritonKernelSpec(
        pattern="rmsnorm_channel_first",
        reference=fused_channel_first_l2norm_reference,
        kernel=fused_channel_first_l2norm_triton,
        metadata=dict(TRITON_KERNEL_METADATA["rmsnorm"]),
    ),
    "rmsnorm_residual": TritonKernelSpec(
        pattern="rmsnorm_residual",
        reference=fused_rmsnorm_residual_reference,
        kernel=fused_rmsnorm_residual_triton,
        metadata=dict(TRITON_KERNEL_METADATA["rmsnorm_residual"]),
    ),
    "rope": TritonKernelSpec(
        pattern="rope",
        reference=fused_rope_reference,
        kernel=fused_rope_triton,
        metadata=dict(TRITON_KERNEL_METADATA["rope"]),
    ),
    # Multi-precision GEMM kernels
    "gemm_fp16": TritonKernelSpec(
        pattern="gemm_fp16",
        reference=gemm_reference,
        kernel=gemm_fp16_triton,
        metadata=dict(TRITON_GEMM_KERNEL_METADATA["gemm_fp16"]),
    ),
    "gemm_bf16": TritonKernelSpec(
        pattern="gemm_bf16",
        reference=gemm_reference,
        kernel=gemm_bf16_triton,
        metadata=dict(TRITON_GEMM_KERNEL_METADATA["gemm_bf16"]),
    ),
    "gemm_int8": TritonKernelSpec(
        pattern="gemm_int8",
        reference=gemm_int8_reference,
        kernel=gemm_int8_triton,
        metadata=dict(TRITON_GEMM_KERNEL_METADATA["gemm_int8"]),
    ),
    "gemm_fp8": TritonKernelSpec(
        pattern="gemm_fp8",
        reference=gemm_fp8_reference,
        kernel=gemm_fp8_triton,
        metadata=dict(TRITON_GEMM_KERNEL_METADATA["gemm_fp8"]),
    ),
    "gemm_int4_dequant": TritonKernelSpec(
        pattern="gemm_int4_dequant",
        reference=gemm_int4_dequant_reference,
        kernel=gemm_int4_dequant_triton,
        metadata=dict(TRITON_GEMM_KERNEL_METADATA["gemm_int4_dequant"]),
    ),
    "gemm_nvfp4_packed_dequant": TritonKernelSpec(
        pattern="gemm_nvfp4_packed_dequant",
        reference=gemm_nvfp4_packed_dequant_reference,
        kernel=gemm_nvfp4_packed_dequant_triton,
        metadata=dict(TRITON_GEMM_KERNEL_METADATA["gemm_nvfp4_packed_dequant"]),
    ),
    "gemm_mxfp8": TritonKernelSpec(
        pattern="gemm_mxfp8",
        reference=gemm_mxfp_reference,
        kernel=gemm_mxfp_triton,
        metadata=dict(MXFP_GEMM_KERNEL_METADATA["gemm_mxfp8"]),
    ),
    "gemm_mxfp6": TritonKernelSpec(
        pattern="gemm_mxfp6",
        reference=gemm_mxfp_reference,
        kernel=gemm_mxfp_triton,
        metadata=dict(MXFP_GEMM_KERNEL_METADATA["gemm_mxfp6"]),
    ),
    "gemm_mxfp4": TritonKernelSpec(
        pattern="gemm_mxfp4",
        reference=gemm_mxfp_reference,
        kernel=gemm_mxfp_triton,
        metadata=dict(MXFP_GEMM_KERNEL_METADATA["gemm_mxfp4"]),
    ),
}


def get_triton_kernel_spec(pattern: str) -> TritonKernelSpec:
    """Return a registered Triton kernel spec."""

    try:
        return TRITON_KERNEL_REGISTRY[pattern]
    except KeyError as exc:
        allowed = ", ".join(sorted(TRITON_KERNEL_REGISTRY))
        raise XQTBackendError(f"Unsupported Triton pattern: {pattern}. Known: {allowed}") from exc


def run_triton_kernel(
    pattern: str,
    *args: torch.Tensor,
    fallback: str = "eager",
    **kwargs: Any,
) -> torch.Tensor:
    """Run a Triton kernel when CUDA is available, otherwise use configured fallback."""

    spec = get_triton_kernel_spec(pattern)
    if not all(isinstance(arg, torch.Tensor) for arg in args if arg is not None):
        raise TypeError("Triton kernel arguments must be tensors or None")
    tensor_args = tuple(arg for arg in args if isinstance(arg, torch.Tensor))
    if spec.cuda_only and not all(arg.is_cuda for arg in tensor_args):
        if fallback == "eager":
            allowed = set(inspect.signature(spec.reference).parameters)
            filtered_kwargs = {
                key: value for key, value in kwargs.items() if key in allowed
            }
            return spec.reference(*args, **filtered_kwargs)
        raise XQTBackendError(f"Triton pattern '{pattern}' requires CUDA tensors")
    return spec.kernel(*args, **kwargs)


def list_triton_kernel_specs() -> dict[str, dict[str, Any]]:
    """Return Triton kernel registry metadata."""

    return {name: spec.to_dict() for name, spec in sorted(TRITON_KERNEL_REGISTRY.items())}


__all__ = [
    "TRITON_KERNEL_REGISTRY",
    "TritonKernelSpec",
    "get_triton_kernel_spec",
    "list_triton_kernel_specs",
    "run_triton_kernel",
]
