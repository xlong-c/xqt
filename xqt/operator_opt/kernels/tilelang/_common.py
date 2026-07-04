"""Shared TileLang kernel guards."""

from __future__ import annotations

import torch

from xqt.core.errors import XQTBackendError


def require_cuda_tensors(*tensors: torch.Tensor) -> None:
    if not tensors:
        raise XQTBackendError("at least one tensor is required")
    if not all(tensor.is_cuda for tensor in tensors):
        raise XQTBackendError("TileLang kernels require CUDA tensors")


def require_tilelang() -> object:
    try:
        import tilelang
    except ImportError as exc:
        raise XQTBackendError(
            "tilelang is required for TileLang operator kernels. Install the optimization extras."
        ) from exc
    return tilelang


def require_fp16_tensors(*tensors: torch.Tensor) -> None:
    if not all(tensor.dtype == torch.float16 for tensor in tensors):
        raise XQTBackendError("TileLang half-kernel paths currently support only float16 tensors")


__all__ = [
    "require_cuda_tensors",
    "require_fp16_tensors",
    "require_tilelang",
]
