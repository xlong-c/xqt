"""Channel-level mixed precision helpers for hybrid inference.

Import contracts (types, normalizers, pure math) from ``xqt.contracts``.
Keep only runtime-specific apply/collect helpers here.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch
from torch import nn

from xqt.contracts import (
    SUPPORTED_CHANNEL_AXES,
    ChannelAxis,
    ChannelHybridSpec,
    SupportsChannelHybrid,
    compute_hybrid_linear,
    normalize_channel_axis,
)

# ---------------------------------------------------------------------------
# re-export contracts symbols for back-compat
# ---------------------------------------------------------------------------

__all__ = [
    "SUPPORTED_CHANNEL_AXES",
    "ChannelAxis",
    "ChannelHybridSpec",
    "SupportsChannelHybrid",
    "apply_channel_hybrid_policy",
    "channel_hybrid_linear_reference",
    "collect_channel_hybrid_map",
    "normalize_channel_axis",
]


# ---------------------------------------------------------------------------
# pure compute thin wrapper
# ---------------------------------------------------------------------------


def channel_hybrid_linear_reference(
    inputs: torch.Tensor,
    *,
    low_weight: torch.Tensor,
    high_weight: torch.Tensor,
    low_activation: torch.Tensor,
    high_activation: torch.Tensor,
    bias: torch.Tensor | None,
    axis: ChannelAxis,
    high_precision_mask: torch.Tensor,
) -> torch.Tensor:
    """Thin wrapper: delegates to ``xqt.contracts.compute_hybrid_linear``."""
    return compute_hybrid_linear(
        inputs,
        low_weight=low_weight,
        high_weight=high_weight,
        low_activation=low_activation,
        high_activation=high_activation,
        bias=bias,
        axis=axis,
        high_precision_mask=high_precision_mask,
    )


# ---------------------------------------------------------------------------
# runtime collect / apply
# ---------------------------------------------------------------------------


def collect_channel_hybrid_map(model: nn.Module) -> dict[str, dict[str, Any]]:
    """Collect active channel hybrid plans from a quantized model."""

    plans: dict[str, dict[str, Any]] = {}
    for name, module in model.named_modules():
        if not isinstance(module, SupportsChannelHybrid):
            getter = getattr(module, "channel_hybrid_spec", None)
            if not callable(getter):
                continue
            spec = getter()
        else:
            spec = module.channel_hybrid_spec()
        if spec is None:
            continue
        plans[name] = spec.to_dict()
    return plans


def apply_channel_hybrid_policy(
    model: nn.Module,
    *,
    channel_overrides: Sequence[Mapping[str, Any]] | None = None,
    inplace: bool = True,
) -> nn.Module:
    """Materialize per-module channel hybrid plans without re-quantizing."""

    import copy

    target = model if inplace else copy.deepcopy(model)
    for item in channel_overrides or []:
        if not isinstance(item, Mapping) or "module" not in item:
            continue
        module_name = str(item["module"])
        module = target.get_submodule(module_name)
        raw_spec = item.get("channel_hybrid", item)
        if not isinstance(raw_spec, Mapping):
            continue
        if raw_spec.get("enabled", True) is False or (
            "high_precision_channels" in raw_spec
            and len(list(raw_spec.get("high_precision_channels") or [])) == 0
            and raw_spec.get("clear", False)
        ):
            if isinstance(module, SupportsChannelHybrid):
                module.set_channel_hybrid_spec(None)
            elif hasattr(module, "set_channel_hybrid_spec"):
                module.set_channel_hybrid_spec(None)
            continue
        spec = ChannelHybridSpec.from_mapping(raw_spec)
        if isinstance(module, SupportsChannelHybrid):
            module.set_channel_hybrid_spec(spec)
        elif hasattr(module, "set_channel_hybrid_spec"):
            module.set_channel_hybrid_spec(spec)
        else:
            raise TypeError(
                f"module {module_name!r} type {type(module).__name__} "
                "does not support channel hybrid inference"
            )
    return target
