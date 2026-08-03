"""Acceptance policy evaluation for XQT optimization stages.

The policy covers four dimensions: numeric diff, speedup, memory and accuracy
drop. Missing evidence fails the check with an explicit reason instead of
silently passing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
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
def _memory_mb(metrics: Mapping[str, Any]) -> float | None:
    for key in ("peak_memory_mb", "memory_mb"):
        value = find_nested_numeric(metrics, key)
        if value is not None:
            return value
    for key in ("peak_memory_bytes", "peak_bytes", "cuda_peak_allocated_bytes"):
        value = find_nested_numeric(metrics, key)
        if value is not None:
            return value / (1024.0 * 1024.0)
    return None


def _accuracy_drop(metrics: Mapping[str, Any]) -> float | None:
    direct = find_nested_numeric(metrics, "accuracy_drop")
    if direct is not None:
        return direct
    baseline = find_nested_numeric(metrics, "accuracy_baseline")
    candidate = find_nested_numeric(metrics, "accuracy_candidate")
    if baseline is not None and candidate is not None:
        return baseline - candidate
    accuracy = find_nested_numeric(metrics, "accuracy")
    if baseline is not None and accuracy is not None:
        return baseline - accuracy
    return None


def _benchmark_speedup(
    reference_benchmark: Mapping[str, Any] | None,
    metrics: Mapping[str, Any],
) -> float | None:
    if reference_benchmark is None:
        return None
    reference_latency = find_nested_numeric(reference_benchmark, "p50_ms")
    current_latency = find_nested_numeric(metrics, "p50_ms")
    if reference_latency is None or current_latency is None or current_latency <= 0:
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
    ) -> None:
        nonlocal accepted
        checks[key] = {
            "passed": passed,
            "observed": observed,
            "threshold": threshold,
            "reason": reason,
        }
        if not passed:
            accepted = False
            failures.append(f"{key}: {reason}")

    min_speedup = getattr(accept, "min_speedup", None)
    if min_speedup is not None:
        speedup = _benchmark_speedup(reference_benchmark, metrics)
        if speedup is None:
            speedup = find_nested_numeric(metrics, "speedup")
        if speedup is None:
            record(
                "min_speedup",
                passed=False,
                observed=None,
                threshold=min_speedup,
                reason="speedup evidence is missing",
            )
        elif speedup < min_speedup:
            record(
                "min_speedup",
                passed=False,
                observed=speedup,
                threshold=min_speedup,
                reason=f"speedup {speedup:.6g} is below {min_speedup:g}",
            )
        else:
            record(
                "min_speedup",
                passed=True,
                observed=speedup,
                threshold=min_speedup,
                reason="speedup threshold met",
            )

    max_mean_abs = getattr(accept, "max_mean_abs", None)
    if max_mean_abs is not None:
        mean_abs = find_nested_numeric(metrics, "mean_abs")
        if mean_abs is None:
            record(
                "max_mean_abs",
                passed=False,
                observed=None,
                threshold=max_mean_abs,
                reason="mean_abs evidence is missing",
            )
        elif mean_abs > max_mean_abs:
            record(
                "max_mean_abs",
                passed=False,
                observed=mean_abs,
                threshold=max_mean_abs,
                reason=f"mean_abs {mean_abs:.6g} exceeds {max_mean_abs:g}",
            )
        else:
            record(
                "max_mean_abs",
                passed=True,
                observed=mean_abs,
                threshold=max_mean_abs,
                reason="mean_abs threshold met",
            )

    max_max_abs = getattr(accept, "max_max_abs", None)
    if max_max_abs is not None:
        max_abs = find_nested_numeric(metrics, "max_abs")
        if max_abs is None:
            record(
                "max_max_abs",
                passed=False,
                observed=None,
                threshold=max_max_abs,
                reason="max_abs evidence is missing",
            )
        elif max_abs > max_max_abs:
            record(
                "max_max_abs",
                passed=False,
                observed=max_abs,
                threshold=max_max_abs,
                reason=f"max_abs {max_abs:.6g} exceeds {max_max_abs:g}",
            )
        else:
            record(
                "max_max_abs",
                passed=True,
                observed=max_abs,
                threshold=max_max_abs,
                reason="max_abs threshold met",
            )

    max_relative_error = getattr(accept, "max_relative_error", None)
    if max_relative_error is not None:
        relative_error = find_nested_numeric(metrics, "relative_error")
        if relative_error is None:
            record(
                "max_relative_error",
                passed=False,
                observed=None,
                threshold=max_relative_error,
                reason="relative_error evidence is missing",
            )
        elif relative_error > max_relative_error:
            record(
                "max_relative_error",
                passed=False,
                observed=relative_error,
                threshold=max_relative_error,
                reason=f"relative_error {relative_error:.6g} exceeds {max_relative_error:g}",
            )
        else:
            record(
                "max_relative_error",
                passed=True,
                observed=relative_error,
                threshold=max_relative_error,
                reason="relative_error threshold met",
            )

    max_memory_mb = getattr(accept, "max_memory_mb", None)
    if max_memory_mb is not None:
        memory_mb = _memory_mb(metrics)
        if memory_mb is None:
            record(
                "max_memory_mb",
                passed=False,
                observed=None,
                threshold=max_memory_mb,
                reason="peak memory evidence is missing",
            )
        elif memory_mb > max_memory_mb:
            record(
                "max_memory_mb",
                passed=False,
                observed=memory_mb,
                threshold=max_memory_mb,
                reason=f"peak memory {memory_mb:.6g} MB exceeds {max_memory_mb:g} MB",
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
        accuracy_drop = _accuracy_drop(metrics)
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
