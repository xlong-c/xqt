"""Typed configuration for graph-level quantization transforms (XQT-012)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from xqt.core.base import XQTConfigError


@dataclass(frozen=True)
class SingleTransformConfig:
    """Configuration for one graph rewrite transform."""

    name: str
    params: Mapping[str, Any] = field(default_factory=dict)
    enabled: bool = True

    def __post_init__(self) -> None:
        if not self.name or not str(self.name).strip():
            raise XQTConfigError("SingleTransformConfig.name must be a non-empty string")


@dataclass(frozen=True)
class GraphTransformConfig:
    """Strongly-typed container for graph transforms."""

    transforms: tuple[SingleTransformConfig, ...] = ()
    dry_run: bool = False
    verify_numerics: bool = True
    tolerance: float = 1e-3

    def to_dict(self) -> dict[str, Any]:
        return {
            "transforms": [
                {
                    "name": t.name,
                    "params": dict(t.params),
                    "enabled": t.enabled,
                }
                for t in self.transforms
            ],
            "dry_run": self.dry_run,
            "verify_numerics": self.verify_numerics,
            "tolerance": self.tolerance,
        }


def parse_graph_transform_config(raw: Any) -> GraphTransformConfig:
    """Parse raw configuration (list, dict, or GraphTransformConfig) into GraphTransformConfig.

    Raises XQTConfigError on unknown transform names or invalid parameter types.
    """
    if raw is None:
        return GraphTransformConfig()
    if isinstance(raw, GraphTransformConfig):
        return raw

    from .registry import is_known_graph_transform, available_graph_transforms

    dry_run = False
    verify_numerics = True
    tolerance = 1e-3
    transform_list: list[SingleTransformConfig] = []

    if isinstance(raw, (list, tuple)):
        items = list(raw)
    elif isinstance(raw, Mapping):
        dry_run = bool(raw.get("dry_run", False))
        verify_numerics = bool(raw.get("verify_numerics", True))
        tolerance = float(raw.get("tolerance", 1e-3))
        raw_transforms = raw.get("transforms", raw.get("names", ()))
        if isinstance(raw_transforms, (list, tuple)):
            items = list(raw_transforms)
        elif isinstance(raw_transforms, str):
            items = [raw_transforms]
        else:
            items = []
    elif isinstance(raw, str):
        items = [raw]
    else:
        raise XQTConfigError(
            f"graph_transforms must be a list, dict or str; got {type(raw).__name__}"
        )

    for item in items:
        if isinstance(item, str):
            name = item.strip().lower()
            params: dict[str, Any] = {}
            enabled = True
        elif isinstance(item, Mapping):
            name = str(item.get("name", "")).strip().lower()
            params = dict(item.get("params", {}))
            enabled = bool(item.get("enabled", True))
        elif isinstance(item, SingleTransformConfig):
            name = item.name.strip().lower()
            params = dict(item.params)
            enabled = item.enabled
        else:
            raise XQTConfigError(
                f"Invalid transform entry: {item!r}; expected str or mapping"
            )

        if not name:
            raise XQTConfigError("Empty transform name in graph_transforms configuration")

        if not is_known_graph_transform(name):
            supported = ", ".join(available_graph_transforms())
            raise XQTConfigError(
                f"Unknown graph transform {name!r}; registered transforms: [{supported}]"
            )

        if enabled:
            transform_list.append(
                SingleTransformConfig(name=name, params=params, enabled=enabled)
            )

    return GraphTransformConfig(
        transforms=tuple(transform_list),
        dry_run=dry_run,
        verify_numerics=verify_numerics,
        tolerance=tolerance,
    )


__all__ = [
    "GraphTransformConfig",
    "SingleTransformConfig",
    "parse_graph_transform_config",
]
