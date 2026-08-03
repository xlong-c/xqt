"""Import helpers for target strings used by XQT recipes."""

from __future__ import annotations

import importlib
from typing import Any, Mapping

from .errors import XQTConfigError


def resolve_target(target: str) -> Any:
    """Resolve a dotted target string into a Python object."""

    if not target or "." not in target:
        raise XQTConfigError(f"Invalid target string: {target!r}")
    module_name, object_name = target.rsplit(".", 1)
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise XQTConfigError(f"Failed to import module '{module_name}'") from exc
    try:
        return getattr(module, object_name)
    except AttributeError as exc:
        raise XQTConfigError(
            f"Target '{object_name}' not found in module '{module_name}'"
        ) from exc


def build_target(target: str, params: Mapping[str, Any] | None = None) -> Any:
    """Resolve and call a target with keyword params."""

    resolved = resolve_target(target)
    kwargs = dict(params or {})
    if isinstance(resolved, type):
        return resolved(**kwargs)
    if callable(resolved):
        return resolved(**kwargs)
    if kwargs:
        raise XQTConfigError(f"Target '{target}' is not callable and cannot accept params")
    return resolved


__all__ = ["build_target", "resolve_target"]
