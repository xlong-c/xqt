"""Provider-facing evaluation adapter boundary for XQT."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from xqt.core.errors import XQTBackendError
from xqt.core.imports import build_target


@dataclass
class EvaluationJob:
    """A task metric request created by XQT and executed by a provider."""

    name: str
    mode: str
    model: Any
    data: Any = None
    reference_model: Any = None
    params: dict[str, Any] = field(default_factory=dict)
    device: str = "cpu"
    task_type: str = "classification"
    context: Any = None


@dataclass
class EvaluationReport:
    """Normalized task evaluation report returned by provider adapters."""

    provider: str
    samples: int
    metrics: dict[str, float] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    message: str = "ok"

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "samples": self.samples,
            "metrics": dict(self.metrics),
            "artifacts": dict(self.artifacts),
            "metadata": dict(self.metadata),
            "message": self.message,
        }


def _optional_int(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    return 0


def _float_metrics(value: Any) -> dict[str, float]:
    if not isinstance(value, Mapping):
        return {}
    metrics: dict[str, float] = {}
    for key, item in value.items():
        if isinstance(item, (float, int)) and not isinstance(item, bool):
            metrics[str(key)] = float(item)
    return metrics


def _mapping_from_result(result: Any) -> dict[str, Any]:
    if isinstance(result, EvaluationReport):
        return result.to_dict()
    if isinstance(result, Mapping):
        return dict(result)
    if hasattr(result, "to_dict") and callable(result.to_dict):
        mapped = result.to_dict()
        if isinstance(mapped, Mapping):
            return dict(mapped)
    return {"result": repr(result)}


def _coerce_evaluation_report(
    result: Any,
    *,
    provider_name: str,
) -> EvaluationReport:
    mapped = _mapping_from_result(result)
    metrics = mapped.get("metrics")
    artifacts = mapped.get("artifacts")
    metadata = mapped.get("metadata")
    return EvaluationReport(
        provider=str(mapped.get("provider", provider_name)),
        samples=_optional_int(mapped.get("samples")),
        metrics=_float_metrics(metrics),
        artifacts=dict(artifacts) if isinstance(artifacts, Mapping) else {},
        metadata=dict(metadata) if isinstance(metadata, Mapping) else {},
        message=str(mapped.get("message", "ok")),
    )


def build_evaluation_provider(provider: Any) -> Any:
    """Resolve an evaluation provider object, callable, or target mapping."""

    if provider is None:
        return None
    if isinstance(provider, str):
        return build_target(provider, {})
    if isinstance(provider, Mapping):
        target = provider.get("target")
        params = dict(provider.get("params") or {})
        if isinstance(target, str):
            return build_target(target, params)
    return provider


def resolve_evaluation_provider(context: Any, params: dict[str, Any]) -> Any:
    """Find a task evaluation provider from stage params or context metadata."""

    provider = params.pop("evaluation_provider", None)
    if provider is None:
        provider = getattr(context, "evaluation_provider", None)
    if provider is None and hasattr(context, "data"):
        provider = context.data.get("evaluation_provider")
    if provider is None:
        task = getattr(getattr(context, "config", None), "task", None)
        task_params = getattr(task, "params", {}) if task is not None else {}
        if isinstance(task_params, Mapping):
            provider = task_params.get("evaluation_provider")
    return build_evaluation_provider(provider)


def run_evaluation_job(provider: Any, job: EvaluationJob) -> EvaluationReport:
    """Delegate task metric evaluation to a provider and normalize its report."""

    resolved = build_evaluation_provider(provider)
    if resolved is None:
        raise XQTBackendError(
            "task metric evaluation requires an evaluation provider"
        )

    if hasattr(resolved, "evaluate") and callable(resolved.evaluate):
        result = resolved.evaluate(job)
        provider_name = type(resolved).__name__
    elif callable(resolved):
        result = resolved(job)
        provider_name = getattr(resolved, "__name__", type(resolved).__name__)
    else:
        raise TypeError("evaluation provider must be callable or expose evaluate()")
    return _coerce_evaluation_report(result, provider_name=provider_name)


__all__ = [
    "EvaluationJob",
    "EvaluationReport",
    "build_evaluation_provider",
    "resolve_evaluation_provider",
    "run_evaluation_job",
]
