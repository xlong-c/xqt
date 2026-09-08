"""Shared metric extraction helpers for XQT auto strategy modules."""

from __future__ import annotations

import math
from typing import Any, Mapping


def _numeric_value(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _resolve_numeric_path(value: Any, path: str) -> float | None:
    current = value
    for part in path.split("."):
        if isinstance(current, Mapping):
            if part not in current:
                return None
            current = current[part]
        elif isinstance(current, (list, tuple)) and part.isdigit():
            index = int(part)
            if index >= len(current):
                return None
            current = current[index]
        else:
            return None
    return _numeric_value(current)


def find_nested_numeric(
    value: Any,
    key: str,
    *,
    path: str | None = None,
    aggregation: str = "max",
) -> float | None:
    """Extract a finite numeric metric by exact path or explicit aggregation.

    ``path`` avoids ambiguous recursive lookup.  The legacy key-search path is
    retained for callers that only build display summaries; acceptance code
    passes a direction-aware aggregation explicitly.
    """

    if path is not None:
        return _resolve_numeric_path(value, path)

    values: list[float] = []
    if isinstance(value, Mapping):
        raw = _numeric_value(value.get(key))
        if raw is not None:
            values.append(raw)
        for item in value.values():
            nested = find_nested_numeric(item, key, aggregation=aggregation)
            if nested is not None:
                values.append(nested)
    elif isinstance(value, (list, tuple)):
        for item in value:
            nested = find_nested_numeric(item, key, aggregation=aggregation)
            if nested is not None:
                values.append(nested)
    if not values:
        return None
    if aggregation == "min":
        return min(values)
    if aggregation == "mean":
        return sum(values) / len(values)
    if aggregation == "first":
        return values[0]
    if aggregation != "max":
        raise ValueError(
            "aggregation must be one of 'min', 'max', 'mean', or 'first'; "
            f"got {aggregation!r}"
        )
    return max(values)


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
