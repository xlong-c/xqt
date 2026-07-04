"""Triton Linear operator aliases backed by GEMM kernels."""

from __future__ import annotations

from .gemm import (
    TRITON_GEMM_KERNEL_METADATA as TRITON_LINEAR_KERNEL_METADATA,
    gemm_bf16_triton as linear_bf16_triton,
    gemm_fp16_triton as linear_fp16_triton,
    gemm_fp8_triton as linear_fp8_triton,
    gemm_int4_dequant_triton as linear_int4_dequant_triton,
    gemm_int8_triton as linear_int8_triton,
    gemm_reference as linear_reference,
)

__all__ = [
    "TRITON_LINEAR_KERNEL_METADATA",
    "linear_bf16_triton",
    "linear_fp16_triton",
    "linear_fp8_triton",
    "linear_int4_dequant_triton",
    "linear_int8_triton",
    "linear_reference",
]
