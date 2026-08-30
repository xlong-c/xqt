"""Shared CuTile kernel guards."""

from __future__ import annotations

import importlib
from typing import Any

import torch

from xqt.core.errors import XQTBackendError


def require_cuda_tensors(*tensors: torch.Tensor) -> None:
    if not tensors:
        raise XQTBackendError("at least one tensor is required")
    if not all(tensor.is_cuda for tensor in tensors):
        raise XQTBackendError("CuTile kernels require CUDA tensors")


def require_cutile() -> object:
    """Return the installed cuTile module, accepting both current and legacy names."""

    for module_name in ("cuda.tile", "cutile"):
        try:
            return importlib.import_module(module_name)
        except ImportError:
            continue
    raise XQTBackendError(
        "cuda-tile is required for CuTile operator kernels. Install cuda-tile or cuda-tile[tileiras]."
    )


def cutile_module_metadata() -> dict[str, Any]:
    """Return import metadata for the installed cuTile runtime."""

    for module_name in ("cuda.tile", "cutile"):
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        version = getattr(module, "__version__", None)
        metadata: dict[str, Any] = {"module": module_name}
        if version is not None:
            metadata["version"] = str(version)
        return metadata
    return {}


def require_fp16_tensors(*tensors: torch.Tensor) -> None:
    if not all(tensor.dtype == torch.float16 for tensor in tensors):
        raise XQTBackendError(
            "CuTile half-kernel paths currently support only float16 tensors"
        )


__all__ = [
    "cutile_module_metadata",
    "require_cuda_tensors",
    "require_cutile",
    "require_fp16_tensors",
]
