"""Graph-level quant transform protocol (C6).

Cross-layer rewrites that can be absorbed offline belong here. Online-only
kernels are declared via ``required_kernels`` and resolved later by engine
dispatch; transforms never invent a serving runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence, runtime_checkable

from torch import nn


@dataclass(frozen=True)
class TransformPlan:
    """Concrete rewrite plan produced by ``GraphQuantTransform.match``."""

    transform_name: str
    targets: tuple[str, ...]
    absorbed_ops: tuple[str, ...] = ()
    online_ops: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "transform_name": self.transform_name,
            "targets": list(self.targets),
            "absorbed_ops": list(self.absorbed_ops),
            "online_ops": list(self.online_ops),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class TransformReport:
    """Result of applying one graph quant transform."""

    transform_name: str
    applied: bool
    absorbed_ops: tuple[str, ...] = ()
    online_ops: tuple[str, ...] = ()
    required_kernels: tuple[str, ...] = ()
    targets: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "transform_name": self.transform_name,
            "applied": self.applied,
            "absorbed_ops": list(self.absorbed_ops),
            "online_ops": list(self.online_ops),
            "required_kernels": list(self.required_kernels),
            "targets": list(self.targets),
            "notes": list(self.notes),
            "metadata": dict(self.metadata),
        }


@runtime_checkable
class GraphQuantTransform(Protocol):
    """Offline graph rewrite for quantization (absorb first, online last)."""

    name: str
    required_kernels: Sequence[str]

    def match(self, model: nn.Module) -> TransformPlan | None:
        """Return a plan when the model structure is eligible, else None."""

    def apply(self, model: nn.Module, plan: TransformPlan) -> TransformReport:
        """Apply ``plan`` in-place (or documented copy) and return a report."""


def apply_graph_transforms(
    model: nn.Module,
    transforms: Sequence[GraphQuantTransform],
    *,
    dry_run: bool = False,
) -> list[TransformReport]:
    """Run match/apply for each transform; in dry_run mode generate plans without rewriting."""

    reports: list[TransformReport] = []
    for transform in transforms:
        plan = transform.match(model)
        if plan is None:
            reports.append(
                TransformReport(
                    transform_name=transform.name,
                    applied=False,
                    required_kernels=tuple(transform.required_kernels),
                    notes=("no_match",),
                )
            )
            continue

        if dry_run:
            reports.append(
                TransformReport(
                    transform_name=transform.name,
                    applied=False,
                    absorbed_ops=plan.absorbed_ops,
                    online_ops=plan.online_ops,
                    required_kernels=tuple(transform.required_kernels),
                    targets=plan.targets,
                    notes=("dry_run",),
                    metadata={"plan": plan.to_dict()},
                )
            )
            continue

        reports.append(transform.apply(model, plan))
    return reports


__all__ = [
    "GraphQuantTransform",
    "TransformPlan",
    "TransformReport",
    "apply_graph_transforms",
]
