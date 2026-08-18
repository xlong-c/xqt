"""Quantization strategy / method / compute naming helpers.

The ``WxAy_format`` strategy strings remain the config-facing naming layer and
keep their alias resolution here. Execution-layer code consumes the orthogonal
``QuantScheme`` value object instead; ``resolve_scheme`` maps a canonical
strategy (plus policy overrides) onto it at plan time.
"""

from __future__ import annotations

from dataclasses import asdict, fields
from typing import Any, Mapping

from xqt.contracts.quant_strategy import (
    QUANT_STRATEGY_DEFINITIONS,
    canonical_quant_strategies,
    strategy_scheme_templates,
)

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


_SCHEME_FIELD_NAMES = frozenset(field.name for field in fields(QuantScheme))


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
    definition = QUANT_STRATEGY_DEFINITIONS.get(normalized)
    if definition is None:
        allowed = ", ".join(sorted(QUANT_STRATEGY_DEFINITIONS))
        raise ValueError(
            f"Cannot resolve a QuantScheme from strategy {normalized!r}. "
            f"Known scheme strategies: {allowed}"
        )
    data = asdict(definition.scheme)
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
