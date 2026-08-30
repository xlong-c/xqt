"""TileLang runtime wrappers - thin facade re-exporting from wrappers/ sub-package.

All symbols are defined in `xqt.kernels.wrappers`.  This module preserves
backward-compatible imports of the form::

    from xqt.kernels.wrappers.tilelang_wrappers import _TileLangConvWrapper
"""

from __future__ import annotations

from .attention import _TileLangAttentionWrapper
from .build import build_tilelang_candidate_model
from .conv import _TileLangConvWrapper
from .conv3d import _TileLangConv3dWrapper
from .dequant_gemm import (
    _TileLangDequantGemmWrapper,
    _module_has_cuda_state,
)
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
