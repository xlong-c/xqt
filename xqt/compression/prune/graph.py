"""Dependency graph helpers for structured pruning."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, Sequence


@dataclass
class DependencyGroup:
    """Lightweight dependency-group summary for structured pruning plans."""

    name: str
    producer: str
    consumers: list[str] = field(default_factory=list)
    merge: Optional[str] = None
    shape_constraints: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "producer": self.producer,
            "consumers": list(self.consumers),
            "merge": self.merge,
            "shape_constraints": dict(self.shape_constraints),
        }


@dataclass
class PruningDependencyGraph:
    """Lightweight dependency graph for structured pruning targets."""

    groups: list[DependencyGroup] = field(default_factory=list)

    def add_group(
        self,
        *,
        name: str,
        producer: str,
        consumers: Optional[list[str]] = None,
        merge: Optional[str] = None,
        shape_constraints: Optional[dict[str, Any]] = None,
    ) -> None:
        self.groups.append(
            DependencyGroup(
                name=name,
                producer=producer,
                consumers=list(consumers or []),
                merge=merge,
                shape_constraints=dict(shape_constraints or {}),
            )
        )

    def find_group(self, name: str) -> Optional[DependencyGroup]:
        for group in self.groups:
            if group.name == name:
                return group
        return None

    def to_dict(self) -> dict[str, Any]:
        return {"groups": [group.to_dict() for group in self.groups]}


class _HasDependencyGroup(Protocol):
    dependency_group: str


def validate_candidate_dependencies(
    candidates: Sequence[_HasDependencyGroup],
    dependency_graph: PruningDependencyGraph,
) -> None:
    """Ensure every candidate references a dependency group in the graph."""

    candidate_groups = {candidate.dependency_group for candidate in candidates}
    missing_groups = sorted(
        group_name
        for group_name in candidate_groups
        if dependency_graph.find_group(group_name) is None
    )
    if missing_groups:
        raise ValueError(
            "Structured pruning candidates are missing dependency metadata for groups: "
            + ", ".join(missing_groups)
        )


__all__ = [
    "DependencyGroup",
    "PruningDependencyGraph",
    "validate_candidate_dependencies",
]
