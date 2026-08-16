"""Compatibility exports for legacy SVDQuant runtime shells.

Canonical SVD storage lives in ``xqt.contracts.composite``. New code should
materialize runtime views through ``xqt.runtime`` helpers.
"""

from .svd_fp8_legacy import SVDQuantFp8Linear
from .svd_w4a4_legacy import LowRankBranch, SVDQuantLinear
from .svd_w8a8_legacy import SVDQuantInt8MmaLinear
from .svd_legacy_materializers import (
    _build_svd_collapsed_fp8,
    _build_svd_collapsed_int8,
    _build_svd_split_fp8,
    _build_svd_split_int8,
)

__all__ = [
    "LowRankBranch",
    "SVDQuantFp8Linear",
    "SVDQuantLinear",
    "SVDQuantInt8MmaLinear",
    "_build_svd_collapsed_fp8",
    "_build_svd_collapsed_int8",
    "_build_svd_split_fp8",
    "_build_svd_split_int8",
]
