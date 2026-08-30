"""CuTile kernel references and guarded entry points."""

from .attention import (
    CUTILE_ATTENTION_KERNEL_METADATA,
    CuTileAttentionDesign,
    build_cutile_attention_design,
    fused_attention_forward_cutile,
    fused_attention_forward_reference,
)
from .conv import (
    CUTILE_CONV_KERNEL_METADATA,
    conv2d_cutile,
    conv2d_reference,
)
from .gemm import (
    CUTILE_DEQUANT_GEMM_KERNEL_METADATA,
    dequant_gemm_epilogue_cutile,
    dequant_gemm_epilogue_reference,
    fp4_packed_dequant_gemm_epilogue_cutile,
    fp4_packed_dequant_gemm_epilogue_reference,
    nvfp4_packed_dequant_gemm_epilogue_cutile,
    nvfp4_packed_dequant_gemm_epilogue_reference,
)
from .linear import (
    CUTILE_LINEAR_KERNEL_METADATA,
    dense_linear_epilogue_cutile,
    dense_linear_epilogue_reference,
    half_linear_cutile,
    half_linear_reference,
)
from .norm import (
    CUTILE_NORM_KERNEL_METADATA,
    layer_norm_cutile,
    layer_norm_reference,
)
from .pointwise import (
    CUTILE_KERNEL_METADATA as CUTILE_POINTWISE_KERNEL_METADATA,
    fused_bias_silu_cutile,
    fused_bias_silu_reference,
)

CUTILE_KERNEL_METADATA = {
    **CUTILE_ATTENTION_KERNEL_METADATA,
    **CUTILE_CONV_KERNEL_METADATA,
    **CUTILE_DEQUANT_GEMM_KERNEL_METADATA,
    **CUTILE_LINEAR_KERNEL_METADATA,
    **CUTILE_NORM_KERNEL_METADATA,
    **CUTILE_POINTWISE_KERNEL_METADATA,
}

__all__ = [
    "CUTILE_ATTENTION_KERNEL_METADATA",
    "CUTILE_CONV_KERNEL_METADATA",
    "CUTILE_DEQUANT_GEMM_KERNEL_METADATA",
    "CUTILE_KERNEL_METADATA",
    "CUTILE_LINEAR_KERNEL_METADATA",
    "CUTILE_NORM_KERNEL_METADATA",
    "CUTILE_POINTWISE_KERNEL_METADATA",
    "CuTileAttentionDesign",
    "build_cutile_attention_design",
    "conv2d_cutile",
    "conv2d_reference",
    "dense_linear_epilogue_cutile",
    "dense_linear_epilogue_reference",
    "dequant_gemm_epilogue_cutile",
    "dequant_gemm_epilogue_reference",
    "fp4_packed_dequant_gemm_epilogue_cutile",
    "fp4_packed_dequant_gemm_epilogue_reference",
    "fused_attention_forward_cutile",
    "fused_attention_forward_reference",
    "fused_bias_silu_cutile",
    "fused_bias_silu_reference",
    "half_linear_cutile",
    "half_linear_reference",
    "layer_norm_cutile",
    "layer_norm_reference",
    "nvfp4_packed_dequant_gemm_epilogue_cutile",
    "nvfp4_packed_dequant_gemm_epilogue_reference",
]
