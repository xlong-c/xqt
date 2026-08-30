"""Packed NVFP4 E2M1 unpack helpers.

These are tensor kernels, not storage contracts. Wrapper adapters live in
``xqt.kernels.wrappers.nvfp4``.
"""

from __future__ import annotations

import torch

_NVFP4_CODEBOOK = torch.tensor(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ],
    dtype=torch.float32,
)


def unpack_nvfp4e2m1(packed_weight: torch.Tensor, *, input_features: int) -> torch.Tensor:
    """Decode packed NVFP4 E2M1 weights into float32 codes."""

    if packed_weight.dtype != torch.uint8:
        raise TypeError("packed_weight must be uint8")
    low = packed_weight & 0x0F
    high = (packed_weight >> 4) & 0x0F
    unpacked = torch.stack((low, high), dim=-1).reshape(packed_weight.shape[0], -1)
    unpacked = unpacked[:, : int(input_features)]
    codebook = _NVFP4_CODEBOOK.to(device=packed_weight.device)
    return codebook[unpacked.long()]


def normalize_group_scale(weight_scale: torch.Tensor) -> torch.Tensor:
    """Normalize a 2D or 3D per-group scale to ``[out_features, groups, 1]``."""

    if weight_scale.ndim == 2:
        return weight_scale.unsqueeze(-1)
    if weight_scale.ndim == 3 and weight_scale.shape[2] == 1:
        return weight_scale
    raise ValueError(
        "weight_scale must have shape [out_features, groups] or [out_features, groups, 1]"
    )


def expand_group_scale(
    weight_scale: torch.Tensor,
    *,
    group_size: int,
    input_features: int,
) -> torch.Tensor:
    """Expand per-group scale to ``[out_features, input_features]``."""

    normalized = normalize_group_scale(weight_scale)
    expanded = normalized.expand(-1, -1, int(group_size)).reshape(normalized.shape[0], -1)
    return expanded[:, : int(input_features)]


__all__ = [
    "expand_group_scale",
    "normalize_group_scale",
    "unpack_nvfp4e2m1",
]
