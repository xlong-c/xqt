"""Channel utility helpers for quantization algorithms.

Pure tensor utilities: normalize, mask, select. No runtime engine, no module mutation,
no calibration / sensitivity logic.
"""

from __future__ import annotations

from typing import Sequence

import torch


def normalize_channel_indices(
    indices: Sequence[int] | torch.Tensor,
    *,
    dim_size: int,
) -> tuple[int, ...]:
    """Canonicalize and validate channel indices.

    Detaches tensors, sorts, deduplicates, and bounds-checks.
    Returns a sorted tuple of valid indices.
    """
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
    """Build a boolean mask where True marks high-precision channels."""
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
    """Pick high-precision channels from per-channel scores (higher = keep HP).

    ``scores`` are per-channel magnitudes; ``ratio`` selects the top fraction.
    When ``top_k`` is provided, ``ratio`` is ignored and at most ``top_k``
    channels are selected.
    """
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
    _, indices_result = torch.topk(flat, k=count, largest=True, sorted=True)
    return tuple(sorted(int(item) for item in indices_result.tolist()))


__all__ = [
    "build_channel_mask",
    "normalize_channel_indices",
    "select_outlier_channels",
]
