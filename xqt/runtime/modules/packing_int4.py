"""INT4/FP4 packing helpers shared by runtime storage modules."""

from __future__ import annotations

import torch
import torch.nn.functional as F

def _encode_signed_nibble(values: torch.Tensor) -> torch.Tensor:
    encoded = torch.where(values < 0, values + 16, values)
    return encoded.to(torch.uint8)

def _decode_signed_nibble(values: torch.Tensor) -> torch.Tensor:
    signed = torch.where(values >= 8, values.to(torch.int16) - 16, values.to(torch.int16))
    return signed.to(torch.float32)

def _pack_int4(values: torch.Tensor) -> torch.Tensor:
    encoded = _encode_signed_nibble(values)
    if encoded.shape[-1] % 2 != 0:
        encoded = F.pad(encoded, (0, 1), value=0)
    low = encoded[..., 0::2]
    high = encoded[..., 1::2] << 4
    return (low | high).contiguous()

def _unpack_int4(packed: torch.Tensor, input_features: int) -> torch.Tensor:
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    unpacked = torch.stack((low, high), dim=-1).reshape(*packed.shape[:-1], -1)
    unpacked = unpacked[..., :input_features]
    return _decode_signed_nibble(unpacked)

def _safe_positive(value: torch.Tensor, *, eps: float = 1e-6) -> torch.Tensor:
    return torch.clamp(value, min=eps)

def _normalize_group_size(group_size: int, input_features: int) -> int:
    return max(1, min(int(group_size), int(input_features)))

def _pad_weight_for_groups(
    weight: torch.Tensor,
    *,
    input_features: int,
    group_size: int,
) -> tuple[torch.Tensor, int]:
    padded_input_features = (
        (int(input_features) + int(group_size) - 1) // int(group_size)
    ) * int(group_size)
    if padded_input_features != input_features:
        weight = F.pad(weight, (0, padded_input_features - input_features))
    return weight, padded_input_features

def _quantize_grouped_fp4_weight(
    weight: torch.Tensor,
    *,
    group_size: int,
    input_features: int,
    output_features: int,
    group_multiplier: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    weight, padded_input_features = _pad_weight_for_groups(
        weight.detach().to(torch.float32),
        input_features=input_features,
        group_size=group_size,
    )
    grouped_weight = weight.reshape(output_features, -1, group_size)
    max_abs = grouped_weight.abs().amax(dim=2, keepdim=True)
    scale = torch.where(max_abs > 0, max_abs / 7.0, torch.ones_like(max_abs))
    if group_multiplier is not None:
        multiplier = group_multiplier.to(device=scale.device, dtype=scale.dtype)
        if multiplier.ndim == 2:
            multiplier = multiplier.unsqueeze(-1)
        scale = scale * _safe_positive(multiplier)
    quantized = torch.clamp(torch.round(grouped_weight / scale), min=-8, max=7).to(torch.int8)
    packed_weight = _pack_int4(quantized.reshape(output_features, padded_input_features))
    return packed_weight, scale, padded_input_features

__all__ = [
    "_encode_signed_nibble",
    "_decode_signed_nibble",
    "_pack_int4",
    "_unpack_int4",
    "_safe_positive",
    "_normalize_group_size",
    "_pad_weight_for_groups",
    "_quantize_grouped_fp4_weight",
]
