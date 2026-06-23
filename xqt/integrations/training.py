"""Provider-facing training adapter boundary for XQT."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from xqt.core.errors import XQTBackendError
from xqt.core.imports import build_target


@dataclass
class TrainingJob:
    """A training request created by XQT and executed by a task provider."""

    name: str
    mode: str
    model: Any
    train_data: Any = None
    validation_data: Any = None
    teacher: Any = None
    params: dict[str, Any] = field(default_factory=dict)
    device: str = "cpu"
    task_type: str = "classification"
    context: Any = None


@dataclass
class TrainingReport:
    """Normalized report returned by provider training adapters."""

    provider: str
    mode: str
    metrics: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)
    steps: int | None = None
    samples: int | None = None
    message: str = "ok"

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "provider": self.provider,
            "mode": self.mode,
            "metrics": dict(self.metrics),
            "artifacts": dict(self.artifacts),
            "message": self.message,
        }
        if self.steps is not None:
            payload["steps"] = self.steps
        if self.samples is not None:
            payload["samples"] = self.samples
        return payload


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    return None


def _mapping_from_result(result: Any) -> dict[str, Any]:
    if isinstance(result, TrainingReport):
        return result.to_dict()
    if isinstance(result, Mapping):
        return dict(result)
    if hasattr(result, "to_dict") and callable(result.to_dict):
        mapped = result.to_dict()
        if isinstance(mapped, Mapping):
            return dict(mapped)
    return {"result": repr(result)}


def _coerce_training_report(
    result: Any,
    *,
    provider_name: str,
    mode: str,
) -> TrainingReport:
    mapped = _mapping_from_result(result)
    metrics = mapped.get("metrics")
    artifacts = mapped.get("artifacts")
    top_level_metrics = {
        key: value
        for key, value in mapped.items()
        if key
        not in {
            "provider",
            "mode",
            "artifacts",
            "message",
        }
    }
    if isinstance(metrics, Mapping):
        normalized_metrics = dict(metrics)
        for key, value in top_level_metrics.items():
            normalized_metrics.setdefault(key, value)
    else:
        normalized_metrics = top_level_metrics
    return TrainingReport(
        provider=str(mapped.get("provider", provider_name)),
        mode=str(mapped.get("mode", mode)),
        metrics=normalized_metrics,
        artifacts=dict(artifacts) if isinstance(artifacts, Mapping) else {},
        steps=_optional_int(mapped.get("steps")),
        samples=_optional_int(mapped.get("samples")),
        message=str(mapped.get("message", "ok")),
    )


def build_training_provider(provider: Any) -> Any:
    """Resolve a provider object, callable, or target mapping."""

    if provider is None:
        return None
    if isinstance(provider, str):
        return build_target(provider, {})
    if isinstance(provider, Mapping):
        provider_type = provider.get("type") or provider.get("name")
        params = dict(provider.get("params") or {})
        if provider_type == "xdl":
            from .xdl import XDLTrainingProvider

            return XDLTrainingProvider(**params)
        target = provider.get("target")
        if isinstance(target, str):
            return build_target(target, params)
    return provider


def resolve_training_provider(context: Any, params: dict[str, Any]) -> Any:
    """Find a training provider from stage params or context metadata."""

    provider = params.pop("training_provider", None)
    if provider is None:
        provider = params.pop("provider", None)
    if provider is None:
        provider = getattr(context, "training_provider", None)
    if provider is None and hasattr(context, "data"):
        provider = context.data.get("training_provider")
    if provider is None:
        task = getattr(getattr(context, "config", None), "task", None)
        task_params = getattr(task, "params", {}) if task is not None else {}
        if isinstance(task_params, Mapping):
            provider = task_params.get("training_provider")
    return build_training_provider(provider)


def run_training_job(provider: Any, job: TrainingJob) -> TrainingReport:
    """Delegate a training job to a provider and normalize its report."""

    resolved = build_training_provider(provider)
    if resolved is None:
        raise XQTBackendError(
            "XQT does not run optimizer/backward/step training loops. "
            "Provide a task training provider through context.training_provider, "
            "context.data['training_provider'], task.params.training_provider, "
            "or stage.params.training_provider."
        )

    if hasattr(resolved, "train_or_finetune") and callable(resolved.train_or_finetune):
        result = resolved.train_or_finetune(job)
        provider_name = type(resolved).__name__
    elif hasattr(resolved, "train") and callable(resolved.train):
        result = resolved.train(job)
        provider_name = type(resolved).__name__
    elif callable(resolved):
        result = resolved(job)
        provider_name = getattr(resolved, "__name__", type(resolved).__name__)
    else:
        raise TypeError(
            "training provider must be callable or expose train()/train_or_finetune()"
        )
    return _coerce_training_report(result, provider_name=provider_name, mode=job.mode)


__all__ = [
    "TrainingJob",
    "TrainingReport",
    "build_training_provider",
    "resolve_training_provider",
    "run_training_job",
]
