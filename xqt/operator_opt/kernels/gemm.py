"""GEMM operator kernel entry points grouped across backends."""

from __future__ import annotations

from .cute_dsl.gemm import (
    gemm_epilogue_cute_dsl,
    gemm_epilogue_reference as cute_dsl_gemm_epilogue_reference,
)
from .cutlass.gemm import (
    gemm_epilogue_cutlass,
    gemm_epilogue_reference as cutlass_gemm_epilogue_reference,
)
from .tilelang.gemm import (
    dequant_gemm_epilogue_reference,
    dequant_gemm_epilogue_tilelang,
    fp4_packed_dequant_gemm_epilogue_reference,
    fp4_packed_dequant_gemm_epilogue_tilelang,
    mxfp4_packed_dequant_gemm_epilogue_reference,
    mxfp4_packed_dequant_gemm_epilogue_tilelang,
    nvfp4_packed_dequant_gemm_epilogue_reference,
    nvfp4_packed_dequant_gemm_epilogue_tilelang,
)
from .triton.gemm import (
    TritonGemmSchedule,
    gemm_bf16_triton,
    gemm_fp16_triton,
    gemm_fp8_triton,
    gemm_int4_dequant_triton,
    gemm_int8_triton,
    gemm_reference,
    resolve_triton_bf16_gemm_schedule,
    resolve_triton_fp16_gemm_schedule,
)
from .triton.mxfp_gemm import (
    gemm_mxfp_reference,
    gemm_mxfp_triton,
    pack_mxfp,
    unpack_mxfp,
)

__all__ = [
    "cute_dsl_gemm_epilogue_reference",
    "cutlass_gemm_epilogue_reference",
    "TritonGemmSchedule",
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
    "resolve_triton_bf16_gemm_schedule",
    "resolve_triton_fp16_gemm_schedule",
    "mxfp4_packed_dequant_gemm_epilogue_reference",
    "mxfp4_packed_dequant_gemm_epilogue_tilelang",
    "nvfp4_packed_dequant_gemm_epilogue_reference",
    "nvfp4_packed_dequant_gemm_epilogue_tilelang",
    "pack_mxfp",
    "unpack_mxfp",
]
