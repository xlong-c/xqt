"""Triton kernel references and CUDA entry points."""

from .pointwise import (
    TRITON_KERNEL_METADATA,
    fused_bias_gelu_reference,
    fused_bias_gelu_triton,
    fused_rope_reference,
    fused_rope_triton,
    fused_rmsnorm_residual_reference,
    fused_rmsnorm_residual_triton,
    fused_swiglu_reference,
    fused_swiglu_triton,
)
from .gemm import (
    TRITON_GEMM_KERNEL_METADATA,
    gemm_bf16_triton,
    gemm_fp16_triton,
    gemm_fp8_triton,
    gemm_int4_dequant_triton,
    gemm_int8_triton,
    gemm_reference,
)
from .mxfp_gemm import (
    MXFP_GEMM_KERNEL_METADATA,
    gemm_mxfp_reference,
    gemm_mxfp_triton,
    pack_mxfp,
    unpack_mxfp,
)
from .linear import (
    TRITON_LINEAR_KERNEL_METADATA,
    linear_bf16_triton,
    linear_fp16_triton,
    linear_fp8_triton,
    linear_int4_dequant_triton,
    linear_int8_triton,
    linear_reference,
)

__all__ = [
    "TRITON_KERNEL_METADATA",
    "TRITON_GEMM_KERNEL_METADATA",
    "TRITON_LINEAR_KERNEL_METADATA",
    "MXFP_GEMM_KERNEL_METADATA",
    "fused_bias_gelu_reference",
    "fused_bias_gelu_triton",
    "fused_rope_reference",
    "fused_rope_triton",
    "fused_rmsnorm_residual_reference",
    "fused_rmsnorm_residual_triton",
    "fused_swiglu_reference",
    "fused_swiglu_triton",
    "gemm_bf16_triton",
    "gemm_fp16_triton",
    "gemm_fp8_triton",
    "gemm_int4_dequant_triton",
    "gemm_int8_triton",
    "gemm_mxfp_reference",
    "gemm_mxfp_triton",
    "gemm_reference",
    "linear_bf16_triton",
    "linear_fp16_triton",
    "linear_fp8_triton",
    "linear_int4_dequant_triton",
    "linear_int8_triton",
    "linear_reference",
    "pack_mxfp",
    "unpack_mxfp",
]
