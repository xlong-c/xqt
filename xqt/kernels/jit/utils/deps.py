"""Optional dependency probes used by JIT and AOT adapters."""

from __future__ import annotations

import importlib.util
from typing import Iterable


REGISTERED_DEPENDENCIES: tuple[str, ...] = (
    "torch",
    "tilelang",
    "cuda.tile",
    "cutlass",
    "ninja",
)


def dependency_available(module_name: str) -> bool:
    """Return whether a module can be discovered without importing it."""

    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def missing_dependencies(names: Iterable[str]) -> tuple[str, ...]:
    """Return missing names in deterministic order."""

    return tuple(name for name in names if not dependency_available(name))


__all__ = ["REGISTERED_DEPENDENCIES", "dependency_available", "missing_dependencies"]
