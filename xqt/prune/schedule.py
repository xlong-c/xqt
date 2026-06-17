"""Pruning schedules and prune + KD helper loops."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

import torch
from torch import nn

from xqt.distill.training import DistillationTrainReport, train_logit_distillation

from .masks import (
    PruningReport,
    apply_global_l1_unstructured_pruning,
    remove_pruning_reparameterization,
    summarize_pruning,
)


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
class PruneKDStepReport:
    """A single prune + KD step report."""

    step: int
    target_sparsity: float
    pruning: PruningReport
    distillation: Optional[DistillationTrainReport] = None

    def to_dict(self) -> dict[str, object]:
        return {
            "step": self.step,
            "target_sparsity": self.target_sparsity,
            "pruning": self.pruning.to_dict(),
            "distillation": (
                self.distillation.to_dict() if self.distillation is not None else None
            ),
        }


@dataclass
class PruneKDReport:
    """Full prune + KD loop report."""

    steps: list[PruneKDStepReport] = field(default_factory=list)

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


def run_prune_kd_loop(
    student: nn.Module,
    teacher: Optional[nn.Module],
    dataloader: Optional[Iterable[object]],
    *,
    schedule: PruningSchedule,
    optimizer: Optional[torch.optim.Optimizer] = None,
    temperature: float = 2.0,
    alpha: float = 0.5,
    device: str | torch.device = "cpu",
    kd_steps_per_prune: Optional[int] = None,
) -> PruneKDReport:
    """Apply scheduled pruning and optionally run KD after each pruning step."""

    reports: list[PruneKDStepReport] = []
    for step_index, sparsity in enumerate(schedule.values()):
        pruning = apply_global_l1_unstructured_pruning(student, sparsity)
        remove_pruning_reparameterization(student)
        pruning = summarize_pruning(student)
        distill_report: Optional[DistillationTrainReport] = None
        if teacher is not None and dataloader is not None and optimizer is not None:
            distill_report = train_logit_distillation(
                student,
                teacher,
                dataloader,
                optimizer,
                temperature=temperature,
                alpha=alpha,
                device=device,
                max_steps=kd_steps_per_prune,
            )
        reports.append(
            PruneKDStepReport(
                step=step_index,
                target_sparsity=sparsity,
                pruning=pruning,
                distillation=distill_report,
            )
        )
    return PruneKDReport(steps=reports)


__all__ = [
    "PruneKDReport",
    "PruneKDStepReport",
    "PruningSchedule",
    "run_prune_kd_loop",
]
