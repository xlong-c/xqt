"""Reference activation quantization helpers for W8A8 GEMM paths."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True, slots=True)
class Int8ActivationQuantization:
    """Quantized activation plus scales and measurable clipping statistics."""

    values: torch.Tensor
    scales: torch.Tensor
    granularity: str
    source: str
    saturation_ratio: float
    zero_points: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.values.dtype != torch.int8:
            raise TypeError("Int8ActivationQuantization.values must be int8")
        if self.granularity not in {"per_tensor", "per_token"}:
            raise ValueError("INT8 activation quantization supports per_tensor or per_token")
        if self.source not in {"activation_static", "activation_dynamic"}:
            raise ValueError("source must be activation_static or activation_dynamic")
        if not 0.0 <= float(self.saturation_ratio) <= 1.0:
            raise ValueError("saturation_ratio must be in [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        return {
            "activation_dtype": "int8",
            "activation_granularity": self.granularity,
            "activation_scale_source": self.source,
            "saturation_ratio": float(self.saturation_ratio),
            "scale_shape": list(self.scales.shape),
            "zero_point_shape": None if self.zero_points is None else list(self.zero_points.shape),
        }


def _expand_scale(value: torch.Tensor, *, m: int, k: int, granularity: str) -> torch.Tensor:
    scale = value.to(dtype=torch.float32)
    if scale.ndim == 0 or scale.numel() == 1:
        return scale.reshape(1, 1).expand(m, k)
    if granularity == "per_tensor":
        if tuple(scale.shape) != (1, 1):
            raise ValueError("per_tensor activation scale must be scalar or shape [1,1]")
        return scale.expand(m, k)
    if granularity == "per_token":
        if scale.ndim == 1 and scale.numel() == m:
            return scale.reshape(m, 1).expand(m, k)
        if tuple(scale.shape) == (m, 1):
            return scale.expand(m, k)
        raise ValueError(f"per_token activation scale must be [M,1] or [M], got {tuple(scale.shape)}")
    raise ValueError(f"unsupported INT8 activation granularity: {granularity!r}")


def quantize_int8_activation(
    activation: torch.Tensor,
    *,
    granularity: str,
    source: str,
    scale: torch.Tensor | None = None,
    zero_point: torch.Tensor | None = None,
) -> Int8ActivationQuantization:
    """Quantize float ``[M,K]`` activation with explicit static/dynamic semantics."""

    if not isinstance(activation, torch.Tensor) or activation.ndim != 2:
        raise ValueError("activation must be a rank-2 tensor [M,K]")
    if source not in {"activation_static", "activation_dynamic"}:
        raise ValueError("source must be activation_static or activation_dynamic")
    if granularity not in {"per_tensor", "per_token"}:
        raise ValueError("INT8 activation quantization supports per_tensor or per_token")
    m, k = (int(activation.shape[0]), int(activation.shape[1]))
    values = activation.to(dtype=torch.float32)
    if source == "activation_dynamic":
        if granularity == "per_tensor":
            scales = values.abs().amax().reshape(1, 1).clamp_min(1e-8) / 127.0
        else:
            scales = values.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / 127.0
    else:
        if scale is None:
            raise ValueError("activation_static requires an explicit scale artifact")
        scales = scale.detach().to(device=activation.device, dtype=torch.float32)
        _expand_scale(scales, m=m, k=k, granularity=granularity)
        if bool((scales <= 0).any()):
            raise ValueError("activation scales must be positive")
        if granularity == "per_tensor" and scales.numel() == 1:
            scales = scales.reshape(1, 1)
        elif granularity == "per_token" and scales.ndim == 1:
            scales = scales.reshape(m, 1)
    expanded_scale = _expand_scale(scales, m=m, k=k, granularity=granularity).clamp_min(1e-8)
    expanded_zero: torch.Tensor | None = None
    if zero_point is not None:
        expanded_zero = _expand_scale(
            zero_point.detach().to(device=activation.device),
            m=m,
            k=k,
            granularity=granularity,
        )
    unrounded = values / expanded_scale
    if expanded_zero is not None:
        unrounded = unrounded + expanded_zero
    clipped = (unrounded < -128.0) | (unrounded > 127.0)
    quantized = torch.round(unrounded).clamp(-128, 127).to(torch.int8)
    return Int8ActivationQuantization(
        values=quantized,
        scales=scales.contiguous(),
        granularity=granularity,
        source=source,
        saturation_ratio=float(clipped.to(dtype=torch.float32).mean().item()),
        zero_points=None if zero_point is None else zero_point.detach().to(torch.float32).contiguous(),
    )


def dequantize_int8_activation(result: Int8ActivationQuantization) -> torch.Tensor:
    """Dequantize a result using its exact scale/zero-point contract."""

    m, k = (int(result.values.shape[0]), int(result.values.shape[1]))
    scale = _expand_scale(result.scales, m=m, k=k, granularity=result.granularity)
    output = result.values.to(torch.float32)
    if result.zero_points is not None:
        output = output - _expand_scale(
            result.zero_points,
            m=m,
            k=k,
            granularity=result.granularity,
        )
    return output * scale


__all__ = [
    "Int8ActivationQuantization",
    "dequantize_int8_activation",
    "quantize_int8_activation",
]
