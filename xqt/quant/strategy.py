"""Quantization strategy / method / compute naming helpers.

The ``WxAy_format`` strategy strings remain the config-facing naming layer and
keep their alias resolution here. Execution-layer code consumes the orthogonal
``QuantScheme`` value object instead; ``resolve_scheme`` maps a canonical
strategy (plus policy overrides) onto it at plan time.
"""

from __future__ import annotations

from typing import Any, Mapping

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

from .types import QuantScheme


# ── canonical strategy -> QuantScheme templates ─────────────────────────────
# groupwise default group_size matches the quantizer defaults (128). Block
# sizes are format-inherent: nvfp4 uses 16-element blocks, mxfp formats 32.
# ───────────────────────────────────────────────────────────────────────────
_STRATEGY_SCHEME_TEMPLATES: dict[str, dict[str, Any]] = {
    "w4a16_int4": {
        "weight_dtype": "int4",
        "weight_granularity": "groupwise",
        "group_size": 128,
    },
    "w8a16_int8": {
        "weight_dtype": "int8",
        "weight_granularity": "per_channel",
    },
    "w4a16_fp4": {
        "weight_dtype": "fp4",
        "weight_granularity": "groupwise",
        "group_size": 128,
    },
    "w4a16_nvfp4": {
        "weight_dtype": "nvfp4",
        "weight_granularity": "block",
        "group_size": 16,
    },
    "w4a16_mxfp4": {
        "weight_dtype": "mxfp4",
        "weight_granularity": "block",
        "group_size": 32,
    },
    "w8a16_mxfp8": {
        "weight_dtype": "mxfp8",
        "weight_granularity": "block",
        "group_size": 32,
    },
    "w8a16_fp8_e4m3": {
        "weight_dtype": "fp8_e4m3",
        "weight_granularity": "per_channel",
    },
    "w8a16_fp8_e5m2": {
        "weight_dtype": "fp8_e5m2",
        "weight_granularity": "per_channel",
    },
    "w8a8_int8": {
        "weight_dtype": "int8",
        "weight_granularity": "per_channel",
        "activation_dtype": "int8",
        "activation_mode": "dynamic",
    },
    "w8a8_fp8_e4m3": {
        "weight_dtype": "fp8_e4m3",
        "weight_granularity": "per_channel",
        "activation_dtype": "fp8_e4m3",
        "activation_mode": "dynamic",
    },
    "w8a8_fp8_e5m2": {
        "weight_dtype": "fp8_e5m2",
        "weight_granularity": "per_channel",
        "activation_dtype": "fp8_e5m2",
        "activation_mode": "dynamic",
    },
    "w4a4_int4": {
        "weight_dtype": "int4",
        "weight_granularity": "groupwise",
        "group_size": 128,
        "activation_dtype": "int4",
        "activation_mode": "dynamic",
    },
    "w4a4_fp4": {
        "weight_dtype": "fp4",
        "weight_granularity": "groupwise",
        "group_size": 128,
        "activation_dtype": "fp4",
        "activation_mode": "dynamic",
    },
    "w4a4_nvfp4": {
        "weight_dtype": "nvfp4",
        "weight_granularity": "block",
        "group_size": 16,
        "activation_dtype": "nvfp4",
        "activation_mode": "dynamic",
    },
    "w4a4_mxfp4": {
        "weight_dtype": "mxfp4",
        "weight_granularity": "block",
        "group_size": 32,
        "activation_dtype": "mxfp4",
        "activation_mode": "dynamic",
    },
}

_SCHEME_FIELD_NAMES = frozenset(
    {
        "weight_dtype",
        "weight_granularity",
        "group_size",
        "activation_dtype",
        "activation_mode",
        "sym",
    }
)


def canonical_quant_strategies() -> tuple[str, ...]:
    """Return the canonical strategy names (single fact source).

    ``CANONICAL_QUANT_STRATEGIES`` in ``xqt/core/schema.py`` must stay aligned
    with this set; a regression test asserts the two never drift. Strategy
    strings are config-facing aliases of a ``QuantScheme`` (storage +
    activation axes), not a mixed-axis execution enum.
    """

    return tuple(_STRATEGY_SCHEME_TEMPLATES)


def strategy_scheme_templates() -> dict[str, dict[str, Any]]:
    """Return copies of the canonical strategy -> QuantScheme templates."""

    return {
        name: dict(template)
        for name, template in _STRATEGY_SCHEME_TEMPLATES.items()
    }


def resolve_scheme(
    strategy: Any | None,
    policy: Mapping[str, Any] | None = None,
) -> QuantScheme | None:
    """Resolve a canonical strategy string plus policy overrides to a QuantScheme.

    Returns ``None`` when the strategy is empty. Raises ``ValueError`` for
    strategy names that do not map to a known scheme.
    """

    normalized = normalize_quant_strategy(strategy, policy)
    if normalized is None:
        return None
    template = _STRATEGY_SCHEME_TEMPLATES.get(normalized)
    if template is None:
        allowed = ", ".join(sorted(_STRATEGY_SCHEME_TEMPLATES))
        raise ValueError(
            f"Cannot resolve a QuantScheme from strategy {normalized!r}. "
            f"Known scheme strategies: {allowed}"
        )
    data = dict(template)
    policy = policy or {}
    if data["weight_granularity"] == "groupwise" and policy.get("group_size") is not None:
        data["group_size"] = int(policy["group_size"])
    if data.get("activation_dtype") is not None:
        mode_override = policy.get("activation_mode") or policy.get("activation")
        if isinstance(mode_override, str):
            mode_text = mode_override.strip().lower()
            if mode_text in {"dynamic", "static"}:
                data["activation_mode"] = mode_text
    if policy.get("sym") is not None:
        data["sym"] = bool(policy["sym"])
    return QuantScheme(**data)


def coerce_quant_scheme(
    value: Any,
    *,
    strategy: str | None = None,
    policy: Mapping[str, Any] | None = None,
) -> QuantScheme | None:
    """Coerce a ``quant.scheme`` config value to a QuantScheme.

    Accepts a QuantScheme instance, a canonical strategy name, or a mapping of
    QuantScheme fields. When ``value`` is None the scheme is derived from the
    legacy strategy/policy pair; an explicit ``value`` always wins over the
    strategy-derived scheme.
    """

    if value is None:
        if strategy is None:
            return None
        return resolve_scheme(strategy, policy)
    if isinstance(value, QuantScheme):
        return value
    if isinstance(value, str):
        scheme = resolve_scheme(value, policy)
        if scheme is None:
            raise ValueError(
                f"quant.scheme {value!r} does not resolve to a known quantization scheme"
            )
        return scheme
    if isinstance(value, Mapping):
        unknown = set(value) - _SCHEME_FIELD_NAMES
        if unknown:
            allowed = ", ".join(sorted(_SCHEME_FIELD_NAMES))
            raise ValueError(
                f"quant.scheme has unknown fields: {sorted(unknown)}. "
                f"Known fields: {allowed}"
            )
        data = {key: value[key] for key in _SCHEME_FIELD_NAMES if key in value}
        if data.get("group_size") is not None:
            data["group_size"] = int(data["group_size"])
        return QuantScheme(**data)
    raise TypeError(
        "quant.scheme must be a strategy name, a mapping of QuantScheme fields, "
        f"or a QuantScheme; got {type(value).__name__}"
    )


__all__ = [
    "CANONICAL_QUANT_COMPUTES",
    "CANONICAL_QUANT_METHODS",
    "CANONICAL_QUANT_STRATEGIES",
    "OPTIONAL_QUANT_STRATEGIES",
    "SUPPORTED_QUANT_COMPUTES",
    "SUPPORTED_QUANT_METHODS",
    "SUPPORTED_QUANT_STRATEGIES",
    "canonical_quant_strategies",
    "coerce_quant_scheme",
    "is_supported_quant_compute",
    "is_supported_quant_method",
    "is_supported_quant_strategy",
    "normalize_quant_compute",
    "normalize_quant_method",
    "normalize_quant_strategy",
    "require_supported_quant_compute",
    "require_supported_quant_method",
    "require_supported_quant_strategy",
    "resolve_scheme",
    "strategy_scheme_templates",
]
