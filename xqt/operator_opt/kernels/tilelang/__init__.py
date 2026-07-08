"""TileLang kernel design helpers, references, and guarded entry points."""

from .attention import (
    TileLangAttentionDesign,
    build_tilelang_attention_design,
    fused_attention_forward_reference,
    fused_attention_forward_tilelang,
)
from .conv import (
    TILELANG_CONV_KERNEL_METADATA,
    build_tilelang_conv1x1_nchw_kernel,
    conv2d_reference,
    conv2d_tilelang,
    conv3d_1x1x1_reference,
    conv3d_1x1x1_tilelang,
)
from .gemm import (
    TILELANG_DEQUANT_GEMM_KERNEL_METADATA,
    dequant_gemm_epilogue_reference,
    dequant_gemm_epilogue_tilelang,
    fp4_packed_dequant_gemm_epilogue_reference,
    fp4_packed_dequant_gemm_epilogue_tilelang,
    nvfp4_packed_dequant_gemm_epilogue_reference,
    nvfp4_packed_dequant_gemm_epilogue_tilelang,
)
from .linear import (
    TILELANG_LINEAR_KERNEL_METADATA,
    dense_linear_epilogue_reference,
    dense_linear_epilogue_tilelang,
    half_linear_reference,
    half_linear_tilelang,
)
from .linear_marlin import (
    TILELANG_MARLIN_LINEAR_KERNEL_METADATA,
    build_tilelang_marlin_linear_kernel,
    linear_marlin_reference,
    linear_marlin_tilelang,
    quantize_int4_weight,
    quantize_int8_weight,
)
from .norm import (
    TILELANG_NORM_KERNEL_METADATA,
    layer_norm_reference,
    layer_norm_tilelang,
)
from .gemm_builder import (
    build_tilelang_fp4_fused_dequant_gemm_kernel,
    build_tilelang_fp4_unpack_dequant_kernel,
    build_tilelang_gemm_kernel,
    build_tilelang_nvfp4_fused_dequant_gemm_kernel,
)

TILELANG_KERNEL_METADATA = {
    "attention": {
        "kernel_name": "fused_attention_forward",
        "block_m": 64,
        "block_n": 64,
        "threads": 128,
        "num_stages": 2,
        "baseline": "torch.nn.functional.scaled_dot_product_attention",
        "design": build_tilelang_attention_design().to_dict(),
    },
    **TILELANG_CONV_KERNEL_METADATA,
    **TILELANG_DEQUANT_GEMM_KERNEL_METADATA,
    **TILELANG_LINEAR_KERNEL_METADATA,
    **TILELANG_MARLIN_LINEAR_KERNEL_METADATA,
    **TILELANG_NORM_KERNEL_METADATA,
}

__all__ = [
    "TILELANG_KERNEL_METADATA",
    "TILELANG_CONV_KERNEL_METADATA",
    "TILELANG_DEQUANT_GEMM_KERNEL_METADATA",
    "TILELANG_LINEAR_KERNEL_METADATA",
    "TILELANG_MARLIN_LINEAR_KERNEL_METADATA",
    "TILELANG_NORM_KERNEL_METADATA",
    "TileLangAttentionDesign",
    "build_tilelang_attention_design",
    "build_tilelang_conv1x1_nchw_kernel",
    "build_tilelang_fp4_fused_dequant_gemm_kernel",
    "build_tilelang_fp4_unpack_dequant_kernel",
    "build_tilelang_gemm_kernel",
    "build_tilelang_marlin_linear_kernel",
    "build_tilelang_nvfp4_fused_dequant_gemm_kernel",
    "conv2d_reference",
    "conv2d_tilelang",
    "conv3d_1x1x1_reference",
    "conv3d_1x1x1_tilelang",
    "dense_linear_epilogue_reference",
    "dense_linear_epilogue_tilelang",
    "dequant_gemm_epilogue_reference",
    "dequant_gemm_epilogue_tilelang",
    "fp4_packed_dequant_gemm_epilogue_reference",
    "fp4_packed_dequant_gemm_epilogue_tilelang",
    "nvfp4_packed_dequant_gemm_epilogue_reference",
    "nvfp4_packed_dequant_gemm_epilogue_tilelang",
    "fused_attention_forward_reference",
    "fused_attention_forward_tilelang",
    "half_linear_reference",
    "half_linear_tilelang",
    "layer_norm_reference",
    "layer_norm_tilelang",
    "linear_marlin_reference",
    "linear_marlin_tilelang",
    "quantize_int4_weight",
    "quantize_int8_weight",
]
