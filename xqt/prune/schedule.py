"""Pruning schedules for model-only pruning transforms."""

from __future__ import annotations

from dataclasses import dataclass, field

from torch import nn

from .masks import (
    PruningReport,
    apply_global_l1_unstructured_pruning,
    remove_pruning_reparameterization,
    summarize_pruning,
)
from .structured import StructuredPruningReport, apply_structured_pruning


@dataclass
class PruningSchedule:
    """A simple sparsity schedule."""

    target_sparsity: float
    steps: int = 1
    start_sparsity: float = 0.0
    schedule: str = "linear"

    def values(self) -> list[float]:
        if not 0.0 <= self.start_sparsity <= 1.0:
            raise ValueError("start_sparsity must be in [0, 1]")
        if not 0.0 <= self.target_sparsity <= 1.0:
            raise ValueError("target_sparsity must be in [0, 1]")
        if self.steps <= 0:
            raise ValueError("steps must be positive")
        if self.schedule not in {"linear", "one_shot"}:
            raise ValueError("schedule must be linear or one_shot")
        if self.schedule == "one_shot" or self.steps == 1:
            return [self.target_sparsity]
        delta = self.target_sparsity - self.start_sparsity
        return [
            self.start_sparsity + delta * ((index + 1) / self.steps)
            for index in range(self.steps)
        ]


@dataclass
class PruneScheduleStepReport:
    """A single pruning schedule step report."""

    step: int
    target_sparsity: float
    pruning: PruningReport

    def to_dict(self) -> dict[str, object]:
        return {
            "step": self.step,
            "target_sparsity": self.target_sparsity,
            "pruning": self.pruning.to_dict(),
        }


@dataclass
class StructuredPruneScheduleStepReport:
    """A single structured pruning schedule step report."""

    step: int
    target_sparsity: float
    pruning: StructuredPruningReport

    def to_dict(self) -> dict[str, object]:
        return {
            "step": self.step,
            "target_sparsity": self.target_sparsity,
            "pruning": self.pruning.to_dict(),
        }


@dataclass
class PruneScheduleReport:
    """Full pruning schedule report."""

    steps: list[PruneScheduleStepReport] = field(default_factory=list)

    @property
    def final_sparsity(self) -> float:
        if not self.steps:
            return 0.0
        return self.steps[-1].pruning.sparsity

    def to_dict(self) -> dict[str, object]:
        return {
            "final_sparsity": self.final_sparsity,
            "steps": [step.to_dict() for step in self.steps],
        }


@dataclass
class StructuredPruneScheduleReport:
    """Full structured pruning schedule report."""

    steps: list[StructuredPruneScheduleStepReport] = field(default_factory=list)

    @property
    def final_sparsity(self) -> float:
        if not self.steps:
            return 0.0
        return self.steps[-1].pruning.sparsity

    def to_dict(self) -> dict[str, object]:
        return {
            "final_sparsity": self.final_sparsity,
            "method": "structured",
            "steps": [step.to_dict() for step in self.steps],
        }


def run_prune_schedule(
    model: nn.Module,
    *,
    schedule: PruningSchedule,
) -> PruneScheduleReport:
    """Apply scheduled unstructured pruning without recovery training."""

    reports: list[PruneScheduleStepReport] = []
    for step_index, sparsity in enumerate(schedule.values()):
        apply_global_l1_unstructured_pruning(model, sparsity)
        remove_pruning_reparameterization(model)
        pruning = summarize_pruning(model)
        reports.append(
            PruneScheduleStepReport(
                step=step_index,
                target_sparsity=sparsity,
                pruning=pruning,
            )
        )
    return PruneScheduleReport(steps=reports)


def run_structured_prune_schedule(
    model: nn.Module,
    *,
    schedule: PruningSchedule,
    granularity: str,
    scope: str,
    importance: dict[str, object] | None = None,
    selection: dict[str, object] | None = None,
    example_input: object = None,
) -> StructuredPruneScheduleReport:
    """Apply scheduled structured pruning without recovery training."""

    schedule_values = schedule.values()
    if not schedule_values:
        return StructuredPruneScheduleReport()

    pruning = apply_structured_pruning(
        model,
        schedule_values[-1],
        granularity=granularity,
        scope=scope,
        importance=importance,
        selection=selection,
        example_input=example_input,
    )
    reports: list[StructuredPruneScheduleStepReport] = []
    for step_index, sparsity in enumerate(schedule_values):
        reports.append(
            StructuredPruneScheduleStepReport(
                step=step_index,
                target_sparsity=sparsity,
                pruning=pruning,
            )
        )
    return StructuredPruneScheduleReport(steps=reports)


__all__ = [
    "PruneScheduleReport",
    "PruneScheduleStepReport",
    "PruningSchedule",
    "StructuredPruneScheduleReport",
    "StructuredPruneScheduleStepReport",
    "run_prune_schedule",
    "run_structured_prune_schedule",
]
