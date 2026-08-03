"""Channel-level hybrid inference contracts shared by quant and runtime.

Pure types, protocols, normalizers, and tensor compute helpers.
No quantizer / calibration / sensitivity logic. No ``nn.Module`` mutation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping, Protocol, Sequence, runtime_checkable

import torch
import torch.nn.functional as F

ChannelAxis = Literal["input", "output"]
SUPPORTED_CHANNEL_AXES: frozenset[str] = frozenset({"input", "output"})


def normalize_channel_axis(axis: str) -> ChannelAxis:
    """Canonicalize a channel-axis string to ``"input"`` or ``"output"``.

    Accepted aliases:
      input-axis:  ``"in"``, ``"k"``, ``"feature"``
      output-axis: ``"out"``, ``"n"``, ``"channel"``
    """
    resolved = str(axis).strip().lower()
    aliases: dict[str, str] = {
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


@dataclass(frozen=True, slots=True)
class ChannelHybridSpec:
    """Pure-data channel hybrid plan for one Linear-like module.

    Carries no quantizer state, no runtime reference, and no module handle.
    Safe to share between quant build-time and runtime apply-time.
    """

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
    def from_mapping(cls, payload: Mapping[str, Any]) -> ChannelHybridSpec:
        axis = normalize_channel_axis(str(payload.get("axis", "input")))
        raw_channels = payload.get("high_precision_channels", [])
        if isinstance(raw_channels, torch.Tensor):
            channels = tuple(
                int(item) for item in raw_channels.detach().cpu().tolist()
            )
        elif isinstance(raw_channels, Sequence) and not isinstance(
            raw_channels, (str, bytes)
        ):
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


def compute_hybrid_linear(
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
    """Pure dual-path linear compute: low-bit path + high-precision path.

    ``axis="input"`` splits the K dimension (weight columns / input channels).
    ``axis="output"`` splits the N dimension (weight rows / output channels).

    No ``nn.Module`` mutation, no quantizer state, no side effects.
    Safe to call from both quant module forward and runtime engine.
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


__all__ = [
    "ChannelAxis",
    "ChannelHybridSpec",
    "SUPPORTED_CHANNEL_AXES",
    "SupportsChannelHybrid",
    "compute_hybrid_linear",
    "normalize_channel_axis",
]
