"""Quantization strategy / method / compute naming helpers."""

from __future__ import annotations

from xqt.core.schema import (
    CANONICAL_QUANT_COMPUTES,
    CANONICAL_QUANT_METHODS,
    CANONICAL_QUANT_STRATEGIES,
    OPTIONAL_QUANT_STRATEGIES,
    SUPPORTED_QUANT_COMPUTES,
    SUPPORTED_QUANT_METHODS,
    SUPPORTED_QUANT_STRATEGIES,
    is_supported_quant_compute,
    is_supported_quant_method,
    is_supported_quant_strategy,
    normalize_quant_compute,
    normalize_quant_method,
    normalize_quant_strategy,
    require_supported_quant_compute,
    require_supported_quant_method,
    require_supported_quant_strategy,
)


__all__ = [
    "CANONICAL_QUANT_COMPUTES",
    "CANONICAL_QUANT_METHODS",
    "CANONICAL_QUANT_STRATEGIES",
    "OPTIONAL_QUANT_STRATEGIES",
    "SUPPORTED_QUANT_COMPUTES",
    "SUPPORTED_QUANT_METHODS",
    "SUPPORTED_QUANT_STRATEGIES",
    "is_supported_quant_compute",
    "is_supported_quant_method",
    "is_supported_quant_strategy",
    "normalize_quant_compute",
    "normalize_quant_method",
    "normalize_quant_strategy",
    "require_supported_quant_compute",
    "require_supported_quant_method",
    "require_supported_quant_strategy",
]
