"""Linear operator kernel entry points grouped across backends."""

from __future__ import annotations

from .tilelang.linear import (
    dense_linear_epilogue_reference,
    dense_linear_epilogue_tilelang,
    half_linear_reference,
    half_linear_tilelang,
)
from .triton.linear import (
    linear_bf16_triton,
    linear_fp16_triton,
    linear_fp8_triton,
    linear_int4_dequant_triton,
    linear_int8_triton,
    linear_reference,
)

__all__ = [
    "dense_linear_epilogue_reference",
    "dense_linear_epilogue_tilelang",
    "half_linear_reference",
    "half_linear_tilelang",
    "linear_bf16_triton",
    "linear_fp16_triton",
    "linear_fp8_triton",
    "linear_int4_dequant_triton",
    "linear_int8_triton",
    "linear_reference",
]
