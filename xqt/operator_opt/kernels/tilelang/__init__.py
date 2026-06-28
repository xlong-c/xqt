"""TileLang kernel design helpers, references, and guarded entry points."""

from .attention import (
    TILELANG_KERNEL_METADATA,
    TileLangAttentionDesign,
    build_tilelang_attention_design,
    dequant_gemm_epilogue_reference,
    dequant_gemm_epilogue_tilelang,
    fp4_packed_dequant_gemm_epilogue_reference,
    fp4_packed_dequant_gemm_epilogue_tilelang,
    nvfp4_packed_dequant_gemm_epilogue_reference,
    nvfp4_packed_dequant_gemm_epilogue_tilelang,
    fused_attention_forward_reference,
    fused_attention_forward_tilelang,
)
from .gemm_builder import (
    build_tilelang_fp4_fused_dequant_gemm_kernel,
    build_tilelang_fp4_unpack_dequant_kernel,
    build_tilelang_gemm_kernel,
    build_tilelang_nvfp4_fused_dequant_gemm_kernel,
)

__all__ = [
    "TILELANG_KERNEL_METADATA",
    "TileLangAttentionDesign",
    "build_tilelang_attention_design",
    "build_tilelang_fp4_fused_dequant_gemm_kernel",
    "build_tilelang_fp4_unpack_dequant_kernel",
    "build_tilelang_gemm_kernel",
    "build_tilelang_nvfp4_fused_dequant_gemm_kernel",
    "dequant_gemm_epilogue_reference",
    "dequant_gemm_epilogue_tilelang",
    "fp4_packed_dequant_gemm_epilogue_reference",
    "fp4_packed_dequant_gemm_epilogue_tilelang",
    "nvfp4_packed_dequant_gemm_epilogue_reference",
    "nvfp4_packed_dequant_gemm_epilogue_tilelang",
    "fused_attention_forward_reference",
    "fused_attention_forward_tilelang",
]
