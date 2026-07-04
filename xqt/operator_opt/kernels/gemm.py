"""GEMM operator kernel entry points grouped across backends."""

from __future__ import annotations

from .cute_dsl.gemm import gemm_epilogue_cute_dsl, gemm_epilogue_reference as cute_dsl_gemm_epilogue_reference
from .cutlass.gemm import gemm_epilogue_cutlass, gemm_epilogue_reference as cutlass_gemm_epilogue_reference
from .tilelang.dequant_gemm import (
    dequant_gemm_epilogue_reference,
    dequant_gemm_epilogue_tilelang,
    fp4_packed_dequant_gemm_epilogue_reference,
    fp4_packed_dequant_gemm_epilogue_tilelang,
    nvfp4_packed_dequant_gemm_epilogue_reference,
    nvfp4_packed_dequant_gemm_epilogue_tilelang,
)
from .triton.gemm import (
    gemm_bf16_triton,
    gemm_fp16_triton,
    gemm_fp8_triton,
    gemm_int4_dequant_triton,
    gemm_int8_triton,
    gemm_reference,
)
from .triton.mxfp_gemm import gemm_mxfp_reference, gemm_mxfp_triton, pack_mxfp, unpack_mxfp

__all__ = [
    "cute_dsl_gemm_epilogue_reference",
    "cutlass_gemm_epilogue_reference",
    "dequant_gemm_epilogue_reference",
    "dequant_gemm_epilogue_tilelang",
    "fp4_packed_dequant_gemm_epilogue_reference",
    "fp4_packed_dequant_gemm_epilogue_tilelang",
    "gemm_bf16_triton",
    "gemm_epilogue_cute_dsl",
    "gemm_epilogue_cutlass",
    "gemm_fp16_triton",
    "gemm_fp8_triton",
    "gemm_int4_dequant_triton",
    "gemm_int8_triton",
    "gemm_mxfp_reference",
    "gemm_mxfp_triton",
    "gemm_reference",
    "nvfp4_packed_dequant_gemm_epilogue_reference",
    "nvfp4_packed_dequant_gemm_epilogue_tilelang",
    "pack_mxfp",
    "unpack_mxfp",
]
