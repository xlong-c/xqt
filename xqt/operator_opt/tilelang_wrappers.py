"""TileLang runtime wrappers — thin facade re-exporting from wrappers/ sub-package.

All symbols are defined in `xqt.operator_opt.wrappers`.  This module preserves
backward-compatible imports of the form::

    from xqt.operator_opt.tilelang_wrappers import _TileLangConvWrapper
"""

from __future__ import annotations

from .wrappers.attention import _TileLangAttentionWrapper
from .wrappers.build import build_tilelang_candidate_model
from .wrappers.conv import _TileLangConvWrapper
from .wrappers.conv3d import _TileLangConv3dWrapper
from .wrappers.dequant_gemm import (
    _TileLangDequantGemmWrapper,
    _module_has_cuda_state,
)
from .wrappers.linear import _TileLangEagerDenseLinearModule, _TileLangLinearWrapper
from .wrappers.norm import _TileLangNormWrapper
from .wrappers.xqt_attention import _TileLangXqtAttentionWrapper

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
