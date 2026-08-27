"""Infer-facing runtime module implementations (no quant algorithm entrypoints)."""

from .awq_w4a16_linear import AWQW4A16Linear
from .composite_add import (
    CompositeAddLinear,
    CompositeAddModule,
    materialize_composite_compute,
    materialize_composite_w4a4,
)
from .composite_add_fp8 import CompositeAddFp8Linear
from .composite_add_w4a4 import CompositeAddW4A4Linear
from .composite_norm import RMSNormCompositeLinear
from .convrot import (
    ConvRotExecutionView,
    ConvRotInt8ExecutionView,
    ConvRotW4A4ExecutionView,
    materialize_convrot_execution_views,
)
from .fp8_mma_linear import Fp8MmaLinear
from .int8_mma_linear import Int8MmaLinear
from .kv_attention import KvCacheMetadata, KvScaleAttention
from .svd_legacy import (
    LowRankBranch,
    SVDQuantFp8Linear,
    SVDQuantInt8MmaLinear,
    SVDQuantLinear,
)
from . import svd_legacy_materializers as _svd_legacy_materializers
from .svd_gelu_mlp import SVDQuantGeluMLP
from .svd_flux_attention import SVDQuantFluxAttention, SVDQuantFluxRotaryEmb
from .svd_flux_block import (
    SVDQuantAdaLayerNormZero,
    SVDQuantAdaLayerNormZeroSingle,
    SVDQuantFluxSingleTransformerBlock,
    SVDQuantFluxTransformerBlock,
)
from .svd_flux_transformer import SVDQuantFluxTransformer2DModel
from .w4_storage_int8_mma_linear import W4StorageInt8MmaLinear

__all__ = [
    "AWQW4A16Linear",
    "CompositeAddLinear",
    "CompositeAddModule",
    "materialize_composite_compute",
    "materialize_composite_w4a4",
    "CompositeAddFp8Linear",
    "CompositeAddW4A4Linear",
    "ConvRotExecutionView",
    "ConvRotInt8ExecutionView",
    "ConvRotW4A4ExecutionView",
    "materialize_convrot_execution_views",
    "RMSNormCompositeLinear",
    "Fp8MmaLinear",
    "Int8MmaLinear",
    "KvCacheMetadata",
    "KvScaleAttention",
    "LowRankBranch",
    "SVDQuantFp8Linear",
    "SVDQuantGeluMLP",
    "SVDQuantFluxAttention",
    "SVDQuantFluxRotaryEmb",
    "SVDQuantAdaLayerNormZero",
    "SVDQuantAdaLayerNormZeroSingle",
    "SVDQuantFluxSingleTransformerBlock",
    "SVDQuantFluxTransformerBlock",
    "SVDQuantFluxTransformer2DModel",
    "SVDQuantInt8MmaLinear",
    "SVDQuantLinear",
    "W4StorageInt8MmaLinear",
]
