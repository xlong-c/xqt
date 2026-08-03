"""Shared metric extraction helpers for XQT auto strategy modules."""

from __future__ import annotations

from typing import Any, Mapping


def find_nested_numeric(value: Any, key: str) -> float | None:
    """Return the first numeric value found for ``key`` in a nested mapping."""

    values: list[float] = []
    if isinstance(value, Mapping):
        raw = value.get(key)
        if isinstance(raw, (float, int)) and not isinstance(raw, bool):
            values.append(float(raw))
        for item in value.values():
            nested = find_nested_numeric(item, key)
            if nested is not None:
                values.append(nested)
    elif isinstance(value, (list, tuple)):
        for item in value:
            nested = find_nested_numeric(item, key)
            if nested is not None:
                values.append(nested)
    return max(values) if values else None


def find_nested_text(value: Any, key: str) -> str | None:
    """Return the first text value found for ``key`` in a nested mapping."""

    if isinstance(value, Mapping):
        raw = value.get(key)
        if isinstance(raw, str):
            return raw
        for item in value.values():
            nested = find_nested_text(item, key)
            if nested is not None:
                return nested
    elif isinstance(value, (list, tuple)):
        for item in value:
            nested = find_nested_text(item, key)
            if nested is not None:
                return nested
    return None


def stage_metrics(source: Any) -> dict[str, Any]:
    """Extract the metrics mapping from a stage-like object."""

    if isinstance(source, Mapping):
        raw = source.get("metrics")
        return dict(raw) if isinstance(raw, Mapping) else dict(source)
    metrics = getattr(source, "metrics", None)
    if isinstance(metrics, Mapping):
        return dict(metrics)
    return {}


def stage_accepted(source: Any) -> bool:
    """Extract the accepted flag from a stage-like object."""

    if isinstance(source, Mapping):
        raw = source.get("accepted")
        if isinstance(raw, bool):
            return raw
        raw = source.get("status")
        return str(raw) == "accepted"
    raw = getattr(source, "accepted", None)
    if isinstance(raw, bool):
        return raw
    status = getattr(source, "status", None)
    return str(status) == "accepted"


def stage_name(source: Any) -> str:
    """Extract the stage name from a stage-like object."""

    if isinstance(source, Mapping):
        raw = source.get("name")
        if raw is not None:
            return str(raw)
        raw = source.get("stage_name")
        if raw is not None:
            return str(raw)
    raw = getattr(source, "name", None)
    if raw is None:
        raw = getattr(source, "stage_name", None)
    if raw is None:
        raise ValueError("stage-like object has no name field")
    return str(raw)


def stage_kind(source: Any) -> str:
    """Extract the stage kind from a stage-like object."""

    if isinstance(source, Mapping):
        raw = source.get("kind")
        if raw is not None:
            return str(raw)
        raw = source.get("stage_kind")
        if raw is not None:
            return str(raw)
    raw = getattr(source, "kind", None)
    if raw is None:
        raw = getattr(source, "stage_kind", None)
    return str(raw) if raw is not None else ""
