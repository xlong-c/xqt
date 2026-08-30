"""CuTile pointwise operator references and guarded entry points."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from ._common import require_cuda_tensors, require_cutile


def fused_bias_silu_reference(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Reference implementation for bias + SiLU fusion."""

    if bias.ndim != 1:
        raise ValueError("bias must be one-dimensional")
    if bias.numel() != x.shape[-1]:
        raise ValueError("bias must match the last dimension of x")
    return F.silu(x + bias)


def fused_bias_silu_cutile(
    x: torch.Tensor,
    bias: torch.Tensor,
    *,
    block_size: int = 1024,
    num_warps: int = 4,
) -> torch.Tensor:
    """CUDA-only CuTile placeholder entry point for bias + SiLU."""

    del block_size, num_warps
    require_cuda_tensors(x, bias)
    require_cutile()
    return fused_bias_silu_reference(x, bias)


CUTILE_KERNEL_METADATA: dict[str, dict[str, Any]] = {
    "bias_silu": {
        "kernel_name": "fused_bias_silu",
        "block_size": 1024,
        "num_warps": 4,
        "autotune_key": ["numel"],
        "baseline": "torch.nn.functional.silu(x + bias)",
        "usage": "Small pointwise epilogue scaffold for the nvvc CuTile Python DSL path.",
        "production_status": "reference_guarded",
    },
}


__all__ = [
    "CUTILE_KERNEL_METADATA",
    "fused_bias_silu_cutile",
    "fused_bias_silu_reference",
]
