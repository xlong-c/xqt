"""Channel-level mixed precision helpers for hybrid inference.

Supports dual-path Linear where some channels stay high precision (e.g. 16-bit)
and the rest use low-bit storage/compute (e.g. 4-bit). Quantizers only produce
channel masks / indices; this module only consumes them at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping, Protocol, Sequence, runtime_checkable

import torch
import torch.nn.functional as F
from torch import nn

ChannelAxis = Literal["input", "output"]
SUPPORTED_CHANNEL_AXES: frozenset[str] = frozenset({"input", "output"})


@dataclass(frozen=True, slots=True)
class ChannelHybridSpec:
    """Runtime channel hybrid plan for one Linear-like module."""

    axis: ChannelAxis
    high_precision_channels: tuple[int, ...]
    high_precision: str = "bf16"
    low_precision: str = "w4a4"
    enabled: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "axis": self.axis,
            "high_precision_channels": list(self.high_precision_channels),
            "high_precision": self.high_precision,
            "low_precision": self.low_precision,
            "enabled": bool(self.enabled),
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ChannelHybridSpec":
        axis = normalize_channel_axis(str(payload.get("axis", "input")))
        raw_channels = payload.get("high_precision_channels", [])
        if isinstance(raw_channels, torch.Tensor):
            channels = tuple(int(item) for item in raw_channels.detach().cpu().tolist())
        elif isinstance(raw_channels, Sequence) and not isinstance(raw_channels, (str, bytes)):
            channels = tuple(int(item) for item in raw_channels)
        else:
            raise TypeError("high_precision_channels must be a sequence of ints")
        return cls(
            axis=axis,
            high_precision_channels=channels,
            high_precision=str(payload.get("high_precision", "bf16")),
            low_precision=str(payload.get("low_precision", "w4a4")),
            enabled=bool(payload.get("enabled", True)),
        )


@runtime_checkable
class SupportsChannelHybrid(Protocol):
    """Modules that can materialize channel-level hybrid inference."""

    def set_channel_hybrid_spec(self, spec: ChannelHybridSpec | None) -> None:
        """Apply or clear channel hybrid plan without re-quantizing."""

    def channel_hybrid_spec(self) -> ChannelHybridSpec | None:
        """Return the active channel hybrid plan, if any."""


def normalize_channel_axis(axis: str) -> ChannelAxis:
    resolved = str(axis).strip().lower()
    aliases = {
        "in": "input",
        "k": "input",
        "feature": "input",
        "out": "output",
        "n": "output",
        "channel": "output",
    }
    normalized = aliases.get(resolved, resolved)
    if normalized not in SUPPORTED_CHANNEL_AXES:
        raise ValueError("channel axis must be input or output")
    return "input" if normalized == "input" else "output"


def normalize_channel_indices(
    indices: Sequence[int] | torch.Tensor,
    *,
    dim_size: int,
) -> tuple[int, ...]:
    if isinstance(indices, torch.Tensor):
        values = [int(item) for item in indices.detach().reshape(-1).tolist()]
    else:
        values = [int(item) for item in indices]
    unique_sorted = sorted(set(values))
    for index in unique_sorted:
        if index < 0 or index >= int(dim_size):
            raise ValueError(
                f"channel index {index} out of range for dim_size={dim_size}"
            )
    return tuple(unique_sorted)


def build_channel_mask(
    dim_size: int,
    high_precision_channels: Sequence[int] | torch.Tensor,
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Boolean mask where True marks high-precision channels."""

    mask = torch.zeros(int(dim_size), dtype=torch.bool, device=device)
    indices = normalize_channel_indices(high_precision_channels, dim_size=dim_size)
    if indices:
        mask[list(indices)] = True
    return mask


def select_outlier_channels(
    scores: torch.Tensor,
    *,
    ratio: float,
    top_k: int | None = None,
) -> tuple[int, ...]:
    """Pick high-precision channels from per-channel scores (higher = keep HP)."""

    flat = scores.detach().to(torch.float32).reshape(-1)
    dim = int(flat.numel())
    if dim == 0:
        return ()
    if top_k is not None:
        count = max(0, min(int(top_k), dim))
    else:
        ratio_value = max(0.0, min(float(ratio), 1.0))
        count = int(round(ratio_value * dim))
        count = max(0, min(count, dim))
    if count == 0:
        return ()
    if count >= dim:
        return tuple(range(dim))
    _, indices = torch.topk(flat, k=count, largest=True, sorted=True)
    return tuple(sorted(int(item) for item in indices.tolist()))


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
    """Reference dual-path Linear: low-bit path + high-precision path.

    ``axis="input"`` splits the K dimension (weight columns / input channels).
    ``axis="output"`` splits the N dimension (weight rows / output channels).
    """

    if axis == "input":
        if low_weight.shape[1] != low_activation.shape[-1]:
            raise ValueError("input-axis low path weight/activation dim mismatch")
        if high_weight.shape[1] != high_activation.shape[-1]:
            raise ValueError("input-axis high path weight/activation dim mismatch")
        low_out = F.linear(low_activation, low_weight, None)
        high_out = F.linear(high_activation, high_weight, None)
        output = low_out + high_out
        if bias is not None:
            output = output + bias
        return output

    # output-axis: both paths use full input features, different output rows
    if low_weight.shape[1] != low_activation.shape[-1]:
        raise ValueError("output-axis low path weight/activation dim mismatch")
    if high_weight.shape[1] != high_activation.shape[-1]:
        raise ValueError("output-axis high path weight/activation dim mismatch")
    low_out = F.linear(low_activation, low_weight, None)
    high_out = F.linear(high_activation, high_weight, None)
    out_features = int(high_precision_mask.numel())
    output = low_activation.new_zeros(*low_activation.shape[:-1], out_features)
    low_rows = (~high_precision_mask).nonzero(as_tuple=False).reshape(-1)
    high_rows = high_precision_mask.nonzero(as_tuple=False).reshape(-1)
    if low_rows.numel() > 0:
        output[..., low_rows] = low_out
    if high_rows.numel() > 0:
        output[..., high_rows] = high_out
    if bias is not None:
        output = output + bias
    return output


def split_linear_tensors_by_channel(
    weight: torch.Tensor,
    activation: torch.Tensor,
    *,
    axis: ChannelAxis,
    high_precision_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split weight/activation for dual-path reference compute."""

    mask = high_precision_mask.to(device=weight.device, dtype=torch.bool)
    if axis == "input":
        if int(mask.numel()) != int(weight.shape[1]):
            raise ValueError("input-axis mask length must equal weight.in_features")
        if int(mask.numel()) != int(activation.shape[-1]):
            raise ValueError("input-axis mask length must equal activation features")
        low_mask = ~mask
        low_weight = weight[:, low_mask]
        high_weight = weight[:, mask]
        low_activation = activation[..., low_mask]
        high_activation = activation[..., mask]
        return low_weight, high_weight, low_activation, high_activation

    if int(mask.numel()) != int(weight.shape[0]):
        raise ValueError("output-axis mask length must equal weight.out_features")
    low_mask = ~mask
    low_weight = weight[low_mask, :]
    high_weight = weight[mask, :]
    return low_weight, high_weight, activation, activation


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
