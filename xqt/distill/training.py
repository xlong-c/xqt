"""Distillation training adapter for XQT."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional

from xqt.core.errors import XQTBackendError
from xqt.integrations import TrainingJob, run_training_job


@dataclass
class DistillationTrainReport:
    """Metrics collected from a provider-backed distillation run."""

    steps: int
    samples: int
    mean_loss: float
    last_loss: float
    loss_history: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "steps": self.steps,
            "samples": self.samples,
            "mean_loss": self.mean_loss,
            "last_loss": self.last_loss,
            "loss_history": list(self.loss_history),
        }

    @classmethod
    def from_mapping(cls, report: Mapping[str, Any]) -> "DistillationTrainReport":
        metrics = report.get("metrics")
        metric_values = dict(metrics) if isinstance(metrics, Mapping) else {}
        loss_history = report.get("loss_history")
        if loss_history is None:
            loss_history = metric_values.get("loss_history")
        if isinstance(loss_history, list):
            history = [float(value) for value in loss_history if isinstance(value, (int, float))]
        else:
            history = []
        steps = int(report.get("steps", metric_values.get("steps", len(history) or 0)))
        samples = int(report.get("samples", metric_values.get("samples", 0)))
        mean_loss = float(report.get("mean_loss", metric_values.get("mean_loss", 0.0)))
        last_loss = float(report.get("last_loss", metric_values.get("last_loss", 0.0)))
        return cls(
            steps=steps,
            samples=samples,
            mean_loss=mean_loss,
            last_loss=last_loss,
            loss_history=history,
        )


def train_logit_distillation(
    student: Any,
    teacher: Any,
    dataloader: Iterable[Any],
    optimizer: Any,
    *,
    temperature: float = 2.0,
    alpha: float = 0.5,
    device: str | Any = "cpu",
    max_steps: Optional[int] = None,
    training_provider: Any | None = None,
) -> DistillationTrainReport:
    """Delegate logit distillation to a task provider."""

    if training_provider is None:
        raise XQTBackendError(
            "XQT no longer owns a standalone distillation training loop. "
            "Pass a provider via training_provider and let it execute the update step."
        )
    report = run_training_job(
        training_provider,
        TrainingJob(
            name="logit_distillation",
            mode="distill",
            model=student,
            teacher=teacher,
            train_data=dataloader,
            params={
                "optimizer": optimizer,
                "temperature": temperature,
                "alpha": alpha,
                "max_steps": max_steps,
            },
            device=str(device),
        ),
    )
    return DistillationTrainReport.from_mapping(report.to_dict())


__all__ = [
    "DistillationTrainReport",
    "train_logit_distillation",
]
