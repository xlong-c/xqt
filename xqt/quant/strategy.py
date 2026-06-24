"""Quantization strategy naming helpers."""

from __future__ import annotations

from typing import Any, Mapping


CANONICAL_QUANT_STRATEGIES = (
    "dynamic_int8",
    "weight_only_int8",
    "weight_only_int4",
    "static_qdq_int8",
    "fp8_dynamic",
    "fp4_weight_only",
)

OPTIONAL_QUANT_STRATEGIES = (
    "fp8_weight_only",
)

SUPPORTED_QUANT_STRATEGIES = CANONICAL_QUANT_STRATEGIES + OPTIONAL_QUANT_STRATEGIES

QUANT_STRATEGY_ALIASES = {
    "int8_dynamic_activation_int8_weight": "dynamic_int8",
    "int8_dynamic": "dynamic_int8",
    "int8": "dynamic_int8",
    "int8_weight_only": "weight_only_int8",
    "weight_only_int8": "weight_only_int8",
    "int4_weight_only": "weight_only_int4",
    "weight_only_int4": "weight_only_int4",
    "int4": "weight_only_int4",
    "static_int8": "static_qdq_int8",
    "qdq_int8": "static_qdq_int8",
    "static_qdq_int8": "static_qdq_int8",
    "float8_dynamic_activation_float8_weight": "fp8_dynamic",
    "float8_dynamic": "fp8_dynamic",
    "fp8": "fp8_dynamic",
    "fp4": "fp4_weight_only",
    "weight_only_fp4": "fp4_weight_only",
}


def normalize_quant_strategy(
    strategy: Any | None,
    policy: Mapping[str, Any] | None = None,
) -> str | None:
    """Return the canonical strategy name inferred from strategy or policy."""

    policy = policy or {}
    raw = strategy
    if raw is None:
        raw = policy.get("strategy")
    if raw is None:
        dtype = str(policy.get("dtype") or "").lower()
        scheme = str(policy.get("scheme") or "").lower()
        if dtype == "fp4" and scheme in {"", "weight_only", "weight-only"}:
            raw = "fp4_weight_only"
        elif dtype == "int4" and scheme in {"", "weight_only", "weight-only"}:
            raw = "weight_only_int4"
        elif dtype == "int8" and scheme in {"weight_only", "weight-only"}:
            raw = "weight_only_int8"
        elif dtype == "int8" and scheme in {"", "dynamic"}:
            raw = "dynamic_int8"
        elif dtype in {"fp8", "float8"} and scheme in {"", "dynamic"}:
            raw = "fp8_dynamic"
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    return QUANT_STRATEGY_ALIASES.get(text, text)


def is_supported_quant_strategy(strategy: Any | None) -> bool:
    """Return whether a strategy name is part of the supported XQT vocabulary."""

    normalized = normalize_quant_strategy(strategy)
    return normalized in SUPPORTED_QUANT_STRATEGIES


def require_supported_quant_strategy(
    strategy: Any | None,
    *,
    location: str,
    policy: Mapping[str, Any] | None = None,
) -> str:
    """Normalize a strategy or raise a clear configuration error."""

    normalized = normalize_quant_strategy(strategy, policy)
    if normalized is None:
        allowed = ", ".join(CANONICAL_QUANT_STRATEGIES)
        raise ValueError(f"{location} must specify one of: {allowed}")
    if normalized not in SUPPORTED_QUANT_STRATEGIES:
        allowed = ", ".join(SUPPORTED_QUANT_STRATEGIES)
        raise ValueError(f"{location} must be one of: {allowed}")
    return normalized


__all__ = [
    "CANONICAL_QUANT_STRATEGIES",
    "OPTIONAL_QUANT_STRATEGIES",
    "QUANT_STRATEGY_ALIASES",
    "SUPPORTED_QUANT_STRATEGIES",
    "is_supported_quant_strategy",
    "normalize_quant_strategy",
    "require_supported_quant_strategy",
]
