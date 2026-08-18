"""Orthogonal quantization scheme value object (contracts layer)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

_WEIGHT_DTYPES = frozenset(
    {"int4", "int8", "fp4", "nvfp4", "mxfp4", "mxfp8", "fp8_e4m3", "fp8_e5m2"}
)
_WEIGHT_GRANULARITIES = frozenset({"per_tensor", "per_channel", "groupwise", "block"})
_ACTIVATION_MODES = frozenset({"none", "dynamic", "static"})


@dataclass(frozen=True)
class QuantScheme:
    """Orthogonal quantization scheme value object (DEBT-002).

    Replaces mixed-axis ``WxAy_format`` strategy strings as the execution-layer
    input. Resolved at plan time from ``quant.scheme`` or from the legacy
    strategy/policy pair via ``xqt.quant.strategy.resolve_scheme``; the
    execution layer routes on this object, not on the strategy enum.
    """

    weight_dtype: str
    weight_granularity: str
    group_size: int | None = None
    activation_dtype: str | None = None
    activation_mode: str = "none"
    sym: bool = True

    def __post_init__(self) -> None:
        if self.weight_dtype not in _WEIGHT_DTYPES:
            allowed = ", ".join(sorted(_WEIGHT_DTYPES))
            raise ValueError(
                f"QuantScheme.weight_dtype must be one of: {allowed}; "
                f"got {self.weight_dtype!r}"
            )
        if self.weight_granularity not in _WEIGHT_GRANULARITIES:
            allowed = ", ".join(sorted(_WEIGHT_GRANULARITIES))
            raise ValueError(
                f"QuantScheme.weight_granularity must be one of: {allowed}; "
                f"got {self.weight_granularity!r}"
            )
        if self.weight_granularity in {"groupwise", "block"}:
            if self.group_size is None or int(self.group_size) <= 0:
                raise ValueError(
                    "QuantScheme.group_size must be a positive int when "
                    f"weight_granularity is {self.weight_granularity!r}"
                )
        elif self.group_size is not None:
            raise ValueError(
                "QuantScheme.group_size must be None unless weight_granularity "
                "is 'groupwise' or 'block'"
            )
        if self.activation_mode not in _ACTIVATION_MODES:
            allowed = ", ".join(sorted(_ACTIVATION_MODES))
            raise ValueError(
                f"QuantScheme.activation_mode must be one of: {allowed}; "
                f"got {self.activation_mode!r}"
            )
        if self.activation_dtype is None:
            if self.activation_mode != "none":
                raise ValueError(
                    "QuantScheme.activation_mode must be 'none' for weight-only "
                    "schemes (activation_dtype is None)"
                )
        else:
            if self.activation_dtype not in _WEIGHT_DTYPES:
                allowed = ", ".join(sorted(_WEIGHT_DTYPES))
                raise ValueError(
                    f"QuantScheme.activation_dtype must be one of: {allowed}; "
                    f"got {self.activation_dtype!r}"
                )
            if self.activation_mode == "none":
                raise ValueError(
                    "QuantScheme.activation_mode must be 'dynamic' or 'static' "
                    "when activation_dtype is set"
                )

    @property
    def is_weight_only(self) -> bool:
        return self.activation_dtype is None

    def to_dict(self) -> dict[str, Any]:
        return {
            "weight_dtype": self.weight_dtype,
            "weight_granularity": self.weight_granularity,
            "group_size": self.group_size,
            "activation_dtype": self.activation_dtype,
            "activation_mode": self.activation_mode,
            "sym": self.sym,
            "weight_only": self.is_weight_only,
        }


__all__ = ["QuantScheme"]
