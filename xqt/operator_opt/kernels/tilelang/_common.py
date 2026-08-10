"""Shared TileLang kernel guards."""

from __future__ import annotations

import os
from pathlib import Path

import torch

from xqt.core.errors import XQTBackendError


_KNOWN_INCOMPATIBLE_RUNTIME_VERSIONS = {"0.1.11"}


def _tilelang_runtime_version() -> str | None:
    """Return the installed TileLang version without requiring a kernel compile."""

    try:
        import tilelang
    except ImportError:
        return None
    return str(getattr(tilelang, "__version__", "unknown"))


def _ensure_tilelang_cache_dir() -> None:
    if os.environ.get("TILELANG_CACHE_DIR"):
        return
    cache_dir = Path.cwd() / "artifacts" / "xqt" / "tilelang_cache"
    tmp_dir = cache_dir / "tmp"
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TILELANG_CACHE_DIR"] = str(cache_dir)
    os.environ.setdefault("TILELANG_TMP_DIR", str(tmp_dir))


def require_cuda_tensors(*tensors: torch.Tensor) -> None:
    if not tensors:
        raise XQTBackendError("at least one tensor is required")
    if not all(tensor.is_cuda for tensor in tensors):
        raise XQTBackendError("TileLang kernels require CUDA tensors")


def require_tilelang() -> object:
    _ensure_tilelang_cache_dir()
    try:
        import tilelang
        from tilelang import env

        env.TILELANG_CACHE_DIR = os.environ["TILELANG_CACHE_DIR"]
        env.TILELANG_TMP_DIR = os.environ["TILELANG_TMP_DIR"]
    except ImportError as exc:
        raise XQTBackendError(
            "tilelang is required for TileLang operator kernels. Install the optimization extras."
        ) from exc
    return tilelang


def tilelang_runtime_usable() -> bool:
    """Return whether the installed TileLang adapter is usable by XQT kernels."""

    version = _tilelang_runtime_version()
    if version is None:
        return False
    return version not in _KNOWN_INCOMPATIBLE_RUNTIME_VERSIONS


def tilelang_runtime_unavailability_reason() -> str | None:
    """Describe the local TileLang runtime incompatibility, when known."""

    version = _tilelang_runtime_version()
    if version is None:
        return "tilelang is not importable"
    if version in _KNOWN_INCOMPATIBLE_RUNTIME_VERSIONS:
        return (
            f"TileLang {version} has an incompatible packed-tensor ABI for current XQT kernels; "
            "use a compatible TileLang runtime or the configured eager fallback"
        )
    return None


def require_fp16_tensors(*tensors: torch.Tensor) -> None:
    if not all(tensor.dtype == torch.float16 for tensor in tensors):
        raise XQTBackendError("TileLang half-kernel paths currently support only float16 tensors")


def require_fp16_or_bf16_tensors(*tensors: torch.Tensor) -> None:
    dtypes = {tensor.dtype for tensor in tensors}
    if len(dtypes) != 1 or not dtypes.issubset(
        {torch.float16, torch.bfloat16}
    ):
        raise XQTBackendError(
            "TileLang 16-bit kernel paths require matching float16 or bfloat16 tensors"
        )


__all__ = [
    "_ensure_tilelang_cache_dir",
    "require_cuda_tensors",
    "require_fp16_or_bf16_tensors",
    "require_fp16_tensors",
    "require_tilelang",
    "tilelang_runtime_unavailability_reason",
    "tilelang_runtime_usable",
]
