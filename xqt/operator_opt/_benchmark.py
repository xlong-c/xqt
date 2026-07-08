"""Benchmark helpers for operator optimization execution."""

from __future__ import annotations

from typing import Any, Callable, Mapping

import torch

from xqt.benchmark import LatencyReport, benchmark_callable, measure_callable_ms

from .types import OperatorOptimizationTargetPlan


def _sync_if_needed(sync_cuda: bool) -> None:
    if sync_cuda and torch.cuda.is_available():
        torch.cuda.synchronize()


def _percentile(sorted_values: list[float], percentile: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (percentile / 100.0) * (len(sorted_values) - 1)
    lower = int(rank)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = rank - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _latency_report_from_samples(
    *,
    samples_ms: list[float],
    warmup: int,
    iterations: int,
) -> LatencyReport:
    sorted_samples = sorted(samples_ms)
    mean_ms = sum(samples_ms) / len(samples_ms)
    return LatencyReport(
        iterations=iterations,
        warmup=warmup,
        mean_ms=mean_ms,
        p50_ms=_percentile(sorted_samples, 50),
        p90_ms=_percentile(sorted_samples, 90),
        p99_ms=_percentile(sorted_samples, 99),
        samples_ms=samples_ms,
    )


def _benchmark_paired_callables(
    reference_fn: Callable[[], object],
    candidate_fn: Callable[[], object],
    *,
    warmup: int,
    iterations: int,
    sync_cuda: bool,
    device: str | None,
) -> tuple[LatencyReport, LatencyReport, list[float]]:
    torch_device = torch.device(device) if device is not None else None
    should_sync_cuda = sync_cuda and (torch_device is None or torch_device.type == "cuda")
    reference_samples_ms: list[float] = []
    candidate_samples_ms: list[float] = []
    paired_speedup_ratios: list[float] = []

    with torch.no_grad():
        for index in range(warmup):
            if index % 2 == 0:
                reference_fn()
                candidate_fn()
            else:
                candidate_fn()
                reference_fn()
        _sync_if_needed(should_sync_cuda)

        for index in range(iterations):
            first_fn = reference_fn if index % 2 == 0 else candidate_fn
            second_fn = candidate_fn if index % 2 == 0 else reference_fn
            first_ms = measure_callable_ms(
                first_fn,
                sync_cuda=should_sync_cuda,
                device=device,
            )
            second_ms = measure_callable_ms(
                second_fn,
                sync_cuda=should_sync_cuda,
                device=device,
            )
            if index % 2 == 0:
                reference_samples_ms.append(first_ms)
                candidate_samples_ms.append(second_ms)
                if second_ms > 0.0:
                    paired_speedup_ratios.append(first_ms / second_ms)
            else:
                candidate_samples_ms.append(first_ms)
                reference_samples_ms.append(second_ms)
                if first_ms > 0.0:
                    paired_speedup_ratios.append(second_ms / first_ms)

    return (
        _latency_report_from_samples(
            samples_ms=reference_samples_ms,
            warmup=warmup,
            iterations=iterations,
        ),
        _latency_report_from_samples(
            samples_ms=candidate_samples_ms,
            warmup=warmup,
            iterations=iterations,
        ),
        paired_speedup_ratios,
    )


def _benchmark_paired_batched_callables(
    reference_fn: Callable[[], object],
    candidate_fn: Callable[[], object],
    *,
    warmup: int,
    iterations: int,
    sync_cuda: bool,
    device: str | None,
    inner_iterations: int,
) -> tuple[dict[str, Any], dict[str, Any], list[float]]:
    reference_report, candidate_report, paired_speedup_ratios = _benchmark_paired_callables(
        _repeat_callable(reference_fn, inner_iterations),
        _repeat_callable(candidate_fn, inner_iterations),
        warmup=warmup,
        iterations=iterations,
        sync_cuda=sync_cuda,
        device=device,
    )
    return (
        _per_call_latency(reference_report.to_dict(), inner_iterations),
        _per_call_latency(candidate_report.to_dict(), inner_iterations),
        paired_speedup_ratios,
    )


def _repeat_callable(fn: Callable[[], object], count: int) -> Callable[[], object]:
    if count <= 0:
        raise ValueError("inner_iterations must be positive")

    def repeated() -> object:
        output = fn()
        for _ in range(count - 1):
            output = fn()
        return output

    return repeated


def _per_call_latency(report: dict[str, Any], inner_iterations: int) -> dict[str, Any]:
    scaled = dict(report)
    for key in ("mean_ms", "p50_ms", "p90_ms", "p99_ms"):
        scaled[key] = float(report[key]) / float(inner_iterations)
    scaled["samples_ms"] = [
        float(sample_ms) / float(inner_iterations)
        for sample_ms in report["samples_ms"]
    ]
    scaled["inner_iterations"] = inner_iterations
    return scaled


def _benchmark_batched_callable(
    fn: Callable[[], object],
    *,
    warmup: int,
    iterations: int,
    sync_cuda: bool,
    device: str | None,
    inner_iterations: int,
) -> dict[str, Any]:
    batched_fn = _repeat_callable(fn, inner_iterations)
    report = benchmark_callable(
        batched_fn,
        warmup=warmup,
        iterations=iterations,
        sync_cuda=sync_cuda,
        device=device,
    ).to_dict()
    return _per_call_latency(report, inner_iterations)


def _tilelang_inner_iterations(
    execution_detail: Mapping[str, Any],
) -> int:
    kernel_kind = execution_detail.get("kernel_kind")
    operator_family = execution_detail.get("operator_family")
    if kernel_kind not in {"minimal_cuda_jit", "cuda_graph_replay"}:
        return 1
    if operator_family not in {"attention", "norm", "linear", "conv"}:
        return 1
    return 100


def _benchmark_callable_for_execution(
    fn: Callable[[], object],
    *,
    warmup: int,
    iterations: int,
    sync_cuda: bool,
    device: str | None,
    execution_detail: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    inner_iterations = _tilelang_inner_iterations(execution_detail)
    if inner_iterations <= 1:
        return (
            benchmark_callable(
                fn,
                warmup=warmup,
                iterations=iterations,
                sync_cuda=sync_cuda,
                device=device,
            ).to_dict(),
            "single_callable_mean",
        )
    return (
        _benchmark_batched_callable(
            fn,
            warmup=warmup,
            iterations=iterations,
            sync_cuda=sync_cuda,
            device=device,
            inner_iterations=inner_iterations,
        ),
        "steady_state_batched_mean",
    )


def _native_runtime_speedup_strategy(
    execution_detail: Mapping[str, Any],
) -> str | None:
    if execution_detail.get("kernel_kind") != "native_runtime_fastpath":
        return None
    return "paired_alternating_p50"


def _paired_steady_state_speedup_strategy(
    execution_detail: Mapping[str, Any],
) -> str | None:
    inner_iterations = _tilelang_inner_iterations(execution_detail)
    if inner_iterations <= 1:
        return None
    return "paired_steady_state_batched_mean"


def _effective_min_speedup(
    target: OperatorOptimizationTargetPlan,
    *,
    execution_detail: Mapping[str, Any],
) -> float:
    if (
        execution_detail.get("kernel_kind") == "native_runtime_fastpath"
        and float(target.min_speedup) <= 1.000001
    ):
        return 0.99
    return float(target.min_speedup)


def _native_runtime_near_equal(
    execution_detail: Mapping[str, Any],
    *,
    latency_before: Mapping[str, Any],
    latency_after: Mapping[str, Any],
) -> bool:
    if execution_detail.get("kernel_kind") != "native_runtime_fastpath":
        return False
    p50_before = latency_before.get("p50_ms")
    p50_after = latency_after.get("p50_ms")
    if not isinstance(p50_before, (float, int)) or not isinstance(p50_after, (float, int)):
        return False
    before_ms = float(p50_before)
    after_ms = float(p50_after)
    if before_ms <= 0.0 or after_ms <= 0.0:
        return False
    max_ms = max(before_ms, after_ms)
    if max_ms > 0.25:
        return False
    absolute_gap_ms = abs(before_ms - after_ms)
    if absolute_gap_ms <= 0.005:
        return True
    if max_ms <= 0.1 and absolute_gap_ms <= 0.007:
        return True
    relative_gap = absolute_gap_ms / max(before_ms, after_ms)
    return relative_gap <= 0.03


__all__ = [
    "_benchmark_callable_for_execution",
    "_benchmark_paired_batched_callables",
    "_benchmark_paired_callables",
    "_effective_min_speedup",
    "_native_runtime_near_equal",
    "_native_runtime_speedup_strategy",
    "_paired_steady_state_speedup_strategy",
    "_percentile",
    "_tilelang_inner_iterations",
]
