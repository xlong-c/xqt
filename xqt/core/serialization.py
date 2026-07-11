"""JSON-safe serialization for persisted XQT workflow and report data."""

from __future__ import annotations

import types
from collections.abc import Mapping
from dataclasses import asdict, fields, is_dataclass
from pathlib import Path
from typing import Any


def json_safe_value(value: Any) -> Any:
    """Return a deterministic JSON-safe representation of a runtime value.

    Live runtime objects, such as ONNX Runtime sessions and TensorRT execution
    contexts, intentionally have no persisted representation. Their payload
    contracts carry explicit materialization and validation metadata instead.
    """

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, types.ModuleType):
        return None
    if hasattr(value, "to_dict") and not isinstance(value, type):
        try:
            return json_safe_value(value.to_dict())
        except Exception:
            return None
    if is_dataclass(value) and not isinstance(value, type):
        try:
            return json_safe_value(asdict(value))
        except Exception:
            safe: dict[str, Any] = {}
            for item in fields(value):
                field_value = getattr(value, item.name)
                try:
                    safe[item.name] = json_safe_value(field_value)
                except Exception:
                    safe[item.name] = None
            return safe
    if isinstance(value, Mapping):
        return {str(key): json_safe_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe_value(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe_value(item) for item in value]
    if hasattr(value, "item"):
        try:
            return json_safe_value(value.item())
        except Exception:
            return None
    return None


__all__ = ["json_safe_value"]
