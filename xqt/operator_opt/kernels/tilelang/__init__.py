"""TileLang kernel design helpers, references, and guarded entry points."""

from .attention import (
    TILELANG_KERNEL_METADATA,
    TileLangAttentionDesign,
    build_tilelang_attention_design,
    dequant_gemm_epilogue_reference,
    dequant_gemm_epilogue_tilelang,
    fused_attention_forward_reference,
    fused_attention_forward_tilelang,
)

__all__ = [
    "TILELANG_KERNEL_METADATA",
    "TileLangAttentionDesign",
    "build_tilelang_attention_design",
    "dequant_gemm_epilogue_reference",
    "dequant_gemm_epilogue_tilelang",
    "fused_attention_forward_reference",
    "fused_attention_forward_tilelang",
]
