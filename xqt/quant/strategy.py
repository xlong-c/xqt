"""Quantization strategy naming helpers."""

from __future__ import annotations

from xqt.core.schema import (
    CANONICAL_QUANT_STRATEGIES,
    OPTIONAL_QUANT_STRATEGIES,
    QUANT_STRATEGY_ALIASES,
    SUPPORTED_QUANT_STRATEGIES,
    is_supported_quant_strategy,
    normalize_quant_strategy,
    require_supported_quant_strategy,
)


__all__ = [
    "CANONICAL_QUANT_STRATEGIES",
    "OPTIONAL_QUANT_STRATEGIES",
    "QUANT_STRATEGY_ALIASES",
    "SUPPORTED_QUANT_STRATEGIES",
    "is_supported_quant_strategy",
    "normalize_quant_strategy",
    "require_supported_quant_strategy",
]
