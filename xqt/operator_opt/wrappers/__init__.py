"""TileLang wrappers sub-package — split from tilelang_wrappers.py for navigability.

Import from the parent facade: `from xqt.operator_opt.tilelang_wrappers import ...`
"""

from __future__ import annotations

from .attention import _TileLangAttentionWrapper
from .build import build_tilelang_candidate_model
from .conv import _TileLangConvWrapper
from .conv3d import _TileLangConv3dWrapper
from .dequant_gemm import _TileLangDequantGemmWrapper
from .linear import _TileLangEagerDenseLinearModule, _TileLangLinearWrapper
from .norm import _TileLangNormWrapper
from .xqt_attention import _TileLangXqtAttentionWrapper

__all__ = [
    "_TileLangAttentionWrapper",
    "_TileLangConv3dWrapper",
    "_TileLangConvWrapper",
    "_TileLangDequantGemmWrapper",
    "_TileLangEagerDenseLinearModule",
    "_TileLangLinearWrapper",
    "_TileLangNormWrapper",
    "_TileLangXqtAttentionWrapper",
    "build_tilelang_candidate_model",
]
