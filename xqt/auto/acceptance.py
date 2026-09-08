"""Acceptance policy evaluation for XQT optimization stages.

The policy covers four dimensions: numeric diff, speedup, memory and accuracy
drop. Missing evidence fails the check with an explicit reason instead of
silently passing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping

from ._helpers import find_nested_numeric


@dataclass(frozen=True)
class AcceptanceEvaluation:
    """Result of evaluating one stage acceptance policy."""

    accepted: bool
    message: str
    checks: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "message": self.message,
            "checks": {
                key: dict(value) for key, value in self.checks.items()
            },
        }
def _metric(
    metrics: Mapping[str, Any],
    accept: Any,
    key: str,
    *,
    default_aggregation: str,
) -> tuple[float | None, str | None, str]:
    paths = getattr(accept, "metric_paths", {})
    aggregations = getattr(accept, "aggregations", {})
    path = paths.get(key) if isinstance(paths, Mapping) else None
    aggregation = (
        aggregations.get(key, default_aggregation)
        if isinstance(aggregations, Mapping)
        else default_aggregation
    )
    value = find_nested_numeric(
        metrics,
        key,
        path=str(path) if path is not None else None,
        aggregation=str(aggregation),
    )
    return value, None if path is None else str(path), str(aggregation)


def _memory_mb(metrics: Mapping[str, Any], accept: Any) -> float | None:
    for key in ("peak_memory_mb", "memory_mb"):
        value, _, _ = _metric(metrics, accept, key, default_aggregation="max")
        if value is not None:
            return value
    for key in ("peak_memory_bytes", "peak_bytes", "cuda_peak_allocated_bytes"):
        value, _, _ = _metric(metrics, accept, key, default_aggregation="max")
        if value is not None:
            return value / (1024.0 * 1024.0)
    return None


def _accuracy_drop(metrics: Mapping[str, Any], accept: Any) -> float | None:
    direct, _, _ = _metric(metrics, accept, "accuracy_drop", default_aggregation="max")
    if direct is not None:
        return direct
    baseline, _, _ = _metric(metrics, accept, "accuracy_baseline", default_aggregation="min")
    candidate, _, _ = _metric(metrics, accept, "accuracy_candidate", default_aggregation="min")
    if baseline is not None and candidate is not None:
        return baseline - candidate
    accuracy, _, _ = _metric(metrics, accept, "accuracy", default_aggregation="min")
    if baseline is not None and accuracy is not None:
        return baseline - accuracy
    return None


def _benchmark_speedup(
    reference_benchmark: Mapping[str, Any] | None,
    metrics: Mapping[str, Any],
    accept: Any,
) -> float | None:
    if reference_benchmark is None:
        return None
    reference_latency, _, _ = _metric(
        reference_benchmark,
        accept,
        "p50_ms",
        default_aggregation="max",
    )
    current_latency, _, _ = _metric(
        metrics,
        accept,
        "p50_ms",
        default_aggregation="max",
    )
    if (
        reference_latency is None
        or current_latency is None
        or reference_latency < 0
        or current_latency <= 0
    ):
        return None
    return reference_latency / current_latency


def evaluate_stage_acceptance(
    accept: Any,
    metrics: Mapping[str, Any],
    *,
    reference_benchmark: Mapping[str, Any] | None = None,
) -> AcceptanceEvaluation:
    """Evaluate all configured acceptance thresholds for one stage."""

    accepted = True
    failures: list[str] = []
    checks: dict[str, dict[str, Any]] = {}

    def record(
        key: str,
        *,
        passed: bool,
        observed: Any,
        threshold: Any,
        reason: str,
        path: str | None = None,
        aggregation: str | None = None,
    ) -> None:
        nonlocal accepted
        checks[key] = {
            "passed": passed,
            "observed": observed,
            "threshold": threshold,
            "reason": reason,
            "path": path,
            "aggregation": aggregation,
        }
        if not passed:
            accepted = False
            failures.append(f"{key}: {reason}")

    min_speedup = getattr(accept, "min_speedup", None)
    if min_speedup is not None:
        speedup = _benchmark_speedup(reference_benchmark, metrics, accept)
        _, speedup_path, speedup_aggregation = _metric(
            metrics,
            accept,
            "speedup",
            default_aggregation="min",
        )
        if speedup is None:
            speedup, speedup_path, speedup_aggregation = _metric(
                metrics,
                accept,
                "speedup",
                default_aggregation="min",
            )
        if speedup is None:
            record(
                "min_speedup",
                passed=False,
                observed=None,
                threshold=min_speedup,
                reason="speedup evidence is missing",
                path=speedup_path,
                aggregation=speedup_aggregation,
            )
        elif speedup <= 0 or not math.isfinite(speedup):
            record(
                "min_speedup",
                passed=False,
                observed=speedup,
                threshold=min_speedup,
                reason="speedup must be finite and > 0",
                path=speedup_path,
                aggregation=speedup_aggregation,
            )
        elif speedup < min_speedup:
            record(
                "min_speedup",
                passed=False,
                observed=speedup,
                threshold=min_speedup,
                reason=f"speedup {speedup:.6g} is below {min_speedup:g}",
                path=speedup_path,
                aggregation=speedup_aggregation,
            )
        else:
            record(
                "min_speedup",
                passed=True,
                observed=speedup,
                threshold=min_speedup,
                reason="speedup threshold met",
                path=speedup_path,
                aggregation=speedup_aggregation,
            )

    max_mean_abs = getattr(accept, "max_mean_abs", None)
    if max_mean_abs is not None:
        mean_abs, path, aggregation = _metric(
            metrics, accept, "mean_abs", default_aggregation="max"
        )
        if mean_abs is None:
            record(
                "max_mean_abs",
                passed=False,
                observed=None,
                threshold=max_mean_abs,
                reason="mean_abs evidence is missing",
                path=path,
                aggregation=aggregation,
            )
        elif mean_abs < 0 or mean_abs > max_mean_abs:
            record(
                "max_mean_abs",
                passed=False,
                observed=mean_abs,
                threshold=max_mean_abs,
                reason=f"mean_abs {mean_abs:.6g} is outside [0, {max_mean_abs:g}]",
                path=path,
                aggregation=aggregation,
            )
        else:
            record(
                "max_mean_abs",
                passed=True,
                observed=mean_abs,
                threshold=max_mean_abs,
                reason="mean_abs threshold met",
                path=path,
                aggregation=aggregation,
            )

    max_max_abs = getattr(accept, "max_max_abs", None)
    if max_max_abs is not None:
        max_abs, path, aggregation = _metric(
            metrics, accept, "max_abs", default_aggregation="max"
        )
        if max_abs is None:
            record(
                "max_max_abs",
                passed=False,
                observed=None,
                threshold=max_max_abs,
                reason="max_abs evidence is missing",
                path=path,
                aggregation=aggregation,
            )
        elif max_abs < 0 or max_abs > max_max_abs:
            record(
                "max_max_abs",
                passed=False,
                observed=max_abs,
                threshold=max_max_abs,
                reason=f"max_abs {max_abs:.6g} is outside [0, {max_max_abs:g}]",
                path=path,
                aggregation=aggregation,
            )
        else:
            record(
                "max_max_abs",
                passed=True,
                observed=max_abs,
                threshold=max_max_abs,
                reason="max_abs threshold met",
                path=path,
                aggregation=aggregation,
            )

    max_relative_error = getattr(accept, "max_relative_error", None)
    if max_relative_error is not None:
        relative_error, path, aggregation = _metric(
            metrics, accept, "relative_error", default_aggregation="max"
        )
        if relative_error is None:
            record(
                "max_relative_error",
                passed=False,
                observed=None,
                threshold=max_relative_error,
                reason="relative_error evidence is missing",
                path=path,
                aggregation=aggregation,
            )
        elif relative_error < 0 or relative_error > max_relative_error:
            record(
                "max_relative_error",
                passed=False,
                observed=relative_error,
                threshold=max_relative_error,
                reason=(
                    f"relative_error {relative_error:.6g} is outside "
                    f"[0, {max_relative_error:g}]"
                ),
                path=path,
                aggregation=aggregation,
            )
        else:
            record(
                "max_relative_error",
                passed=True,
                observed=relative_error,
                threshold=max_relative_error,
                reason="relative_error threshold met",
                path=path,
                aggregation=aggregation,
            )

    max_memory_mb = getattr(accept, "max_memory_mb", None)
    if max_memory_mb is not None:
        memory_mb = _memory_mb(metrics, accept)
        if memory_mb is None:
            record(
                "max_memory_mb",
                passed=False,
                observed=None,
                threshold=max_memory_mb,
                reason="peak memory evidence is missing",
            )
        elif memory_mb < 0 or memory_mb > max_memory_mb:
            record(
                "max_memory_mb",
                passed=False,
                observed=memory_mb,
                threshold=max_memory_mb,
                reason=(
                    f"peak memory {memory_mb:.6g} MB is outside "
                    f"[0, {max_memory_mb:g}] MB"
                ),
            )
        else:
            record(
                "max_memory_mb",
                passed=True,
                observed=memory_mb,
                threshold=max_memory_mb,
                reason="peak memory threshold met",
            )

    max_accuracy_drop = getattr(accept, "max_accuracy_drop", None)
    if max_accuracy_drop is not None:
        accuracy_drop = _accuracy_drop(metrics, accept)
        if accuracy_drop is None:
            record(
                "max_accuracy_drop",
                passed=False,
                observed=None,
                threshold=max_accuracy_drop,
                reason="accuracy evidence is missing (no task-level provider in XQT)",
            )
        elif accuracy_drop > max_accuracy_drop:
            record(
                "max_accuracy_drop",
                passed=False,
                observed=accuracy_drop,
                threshold=max_accuracy_drop,
                reason=f"accuracy drop {accuracy_drop:.6g} exceeds {max_accuracy_drop:g}",
            )
        else:
            record(
                "max_accuracy_drop",
                passed=True,
                observed=accuracy_drop,
                threshold=max_accuracy_drop,
                reason="accuracy drop threshold met",
            )

    if failures:
        message = "rejected by acceptance thresholds: " + "; ".join(failures)
    else:
        message = "ok"
    return AcceptanceEvaluation(
        accepted=accepted,
        message=message,
        checks=checks,
    )


__all__ = [
    "AcceptanceEvaluation",
    "evaluate_stage_acceptance",
]
