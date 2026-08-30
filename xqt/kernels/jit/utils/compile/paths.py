"""Repository and cache paths for XQT JIT sources."""

from __future__ import annotations

import os
from pathlib import Path


def jit_root() -> Path:
    """Return the installed ``xqt/kernels/jit`` source root."""

    return Path(__file__).resolve().parents[2]


def csrc_root() -> Path:
    return jit_root() / "csrc"


def include_root() -> Path:
    return jit_root() / "include"


def default_cache_dir() -> Path:
    configured = os.environ.get("XQT_KERNEL_CACHE") or os.environ.get("XQT_GEMM_CACHE")
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".cache" / "xqt" / "kernels"


def cache_dir(path: str | Path | None = None, *, create: bool = False) -> Path:
    resolved = default_cache_dir() if path is None else Path(path).expanduser()
    if create:
        resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def csrc_path(group: str, filename: str) -> Path:
    """Resolve a migrated source and reject path traversal."""

    if Path(group).name != group or Path(filename).name != filename:
        raise ValueError("group and filename must be simple path components")
    path = csrc_root() / group / filename
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def artifact_metadata_path(
    cache_root: str | Path | None,
    pattern: str,
    engine: str,
) -> Path | None:
    """Return the conventional metadata artifact path without creating it."""

    if cache_root is None:
        return None
    if Path(pattern).name != pattern or Path(engine).name != engine:
        raise ValueError("pattern and engine must be simple path components")
    return Path(cache_root).expanduser() / f"{pattern}.{engine}.json"


__all__ = [
    "artifact_metadata_path",
    "cache_dir",
    "csrc_path",
    "csrc_root",
    "default_cache_dir",
    "include_root",
    "jit_root",
]
