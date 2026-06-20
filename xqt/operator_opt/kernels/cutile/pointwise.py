"""CuTile operator optimization references and guarded entry points."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from xqt.core.errors import XQTBackendError


def _require_cuda_tensors(*tensors: torch.Tensor) -> None:
    if not tensors:
        raise XQTBackendError("at least one tensor is required")
    if not all(tensor.is_cuda for tensor in tensors):
        raise XQTBackendError("CuTile kernels require CUDA tensors")


def _require_cutile() -> object:
    try:
        import cutile
    except ImportError as exc:
        raise XQTBackendError(
            "cutile is required for CuTile operator kernels. Install the nvvc/cutile extras."
        ) from exc
    return cutile


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
    _require_cuda_tensors(x, bias)
    _require_cutile()
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
