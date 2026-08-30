"""Core XQT configuration, artifact, and shared type helpers."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .base import ArtifactManifest, ArtifactRecord, MetricRecord
    from .types import XQTContext


_LAZY_EXPORTS = {
    "ArtifactManifest": (".base", "ArtifactManifest"),
    "ArtifactRecord": (".base", "ArtifactRecord"),
    "MetricRecord": (".base", "MetricRecord"),
    "XQTContext": (".types", "XQTContext"),
}

__all__ = [
    "ArtifactManifest",
    "ArtifactRecord",
    "MetricRecord",
    "XQTContext",
]


def __getattr__(name: str) -> Any:
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module 'xqt.core' has no attribute {name!r}")
    module_name, attribute = target
    value = getattr(importlib.import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_LAZY_EXPORTS})
