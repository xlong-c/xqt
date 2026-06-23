"""XDL task training provider adapter."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .training import TrainingJob, TrainingReport


@dataclass
class XDLTrainingProvider:
    """Delegate gradient-based recovery to ``xdl.trainer.Trainer``."""

    setup: Any = None
    trainer: Any = None
    core_model: Any = None
    train_loader: Any = None
    val_loader: Any = None
    fit_kwargs: dict[str, Any] = field(default_factory=dict)

    def train(self, job: TrainingJob) -> TrainingReport:
        from xdl.trainer import CoreModel, Trainer

        setup = self.setup
        trainer = self.trainer
        core_model = self.core_model
        train_loader = self.train_loader
        val_loader = self.val_loader

        if setup is not None:
            trainer = trainer or Trainer.from_setup(setup)
            core_model = core_model or getattr(setup, "model", None)
            train_loader = train_loader or getattr(setup, "train_loader", None)
            val_loader = val_loader or getattr(setup, "val_loader", None)
        else:
            train_loader = train_loader or job.train_data
            val_loader = val_loader or job.validation_data

        core_model = core_model or job.model
        if not isinstance(core_model, CoreModel):
            raise TypeError("XDLTrainingProvider requires an xdl.trainer.CoreModel")
        if train_loader is None:
            raise ValueError("XDLTrainingProvider requires train_loader")
        if trainer is None:
            trainer = Trainer(
                max_epochs=int(job.params.get("max_epochs", job.params.get("epochs", 1))),
                device=job.device,
            )

        trainer.fit(core_model, train_loader, val_loader, **self.fit_kwargs)
        if job.context is not None:
            job.context.model = core_model

        metrics: dict[str, Any] = {}
        current_metrics = getattr(core_model, "current_metrics", None)
        if isinstance(current_metrics, dict):
            metrics.update(current_metrics)
        callback_metrics = getattr(trainer, "callback_metrics", None)
        if isinstance(callback_metrics, dict):
            metrics.update(callback_metrics)
        steps = getattr(core_model, "total_train_steps", None)
        if not isinstance(steps, int):
            steps = getattr(core_model, "_total_train_steps", None)
        return TrainingReport(
            provider="xdl",
            mode=job.mode,
            metrics=metrics,
            steps=steps if isinstance(steps, int) else None,
            message="delegated to xdl.trainer.Trainer",
        )


__all__ = ["XDLTrainingProvider"]
