"""TensorRT performance parsing and threshold evaluation."""

from __future__ import annotations

import re
from typing import Any, Mapping, Optional

from .trt_types import (
    TensorRTPerformanceCheck,
    TensorRTPerformanceMetrics,
    TensorRTPerformanceThresholdReport,
)

_FLOAT_PATTERN = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_STAT_PATTERN = re.compile(
    rf"(min|max|mean|median|percentile\(({_FLOAT_PATTERN})%\))"
    rf"\s*=\s*({_FLOAT_PATTERN})\s*(ms|us|s)?",
    re.IGNORECASE,
)
_STATS_LINE_FIELDS = (
    ("Host Latency", "host_latency_ms"),
    ("H2D Latency", "h2d_latency_ms"),
    ("D2H Latency", "d2h_latency_ms"),
    ("GPU Compute Time", "gpu_compute_time_ms"),
    ("Enqueue Time", "enqueue_time_ms"),
    ("Latency", "latency_ms"),
)
_TOTAL_TIME_FIELDS = (
    ("Total Host Walltime", "total_host_walltime_ms"),
    ("Total GPU Compute Time", "total_gpu_compute_time_ms"),
)


def _duration_to_ms(value: float, unit: Optional[str]) -> float:
    normalized = (unit or "ms").lower()
    if normalized == "s":
        return value * 1000.0
    if normalized == "us":
        return value / 1000.0
    return value


def _percentile_key(percentile: str) -> str:
    if "." not in percentile:
        return f"p{percentile}"
    normalized = percentile.rstrip("0").rstrip(".").replace(".", "_")
    return f"p{normalized}"


def _parse_stats_fragment(fragment: str) -> dict[str, float]:
    stats: dict[str, float] = {}
    for match in _STAT_PATTERN.finditer(fragment):
        raw_name = match.group(1).lower()
        percentile = match.group(2)
        value = _duration_to_ms(float(match.group(3)), match.group(4))
        if percentile is not None:
            stats[_percentile_key(percentile)] = value
            continue
        stats[raw_name] = value
    return stats


def _flatten_performance_metrics(
    metrics: TensorRTPerformanceMetrics,
) -> dict[str, float]:
    flattened: dict[str, float] = {}
    for key, value in metrics.to_dict().items():
        if isinstance(value, dict):
            for stat_name, stat_value in value.items():
                flattened[f"{key}.{stat_name}"] = float(stat_value)
                if key.endswith("_ms"):
                    flattened[f"{key[:-3]}_{stat_name}_ms"] = float(stat_value)
            continue
        flattened[key] = float(value)
    return flattened


def parse_trtexec_performance(output: str) -> TensorRTPerformanceMetrics:
    """Parse trtexec performance summary text.

    The parser is intentionally permissive because TensorRT versions vary in
    timestamp prefixes and metric labels.
    """

    metrics = TensorRTPerformanceMetrics()
    for line in output.splitlines():
        throughput_match = re.search(
            rf"\bThroughput\s*:\s*({_FLOAT_PATTERN})\s*(?:qps|queries/s|inferences/s)?",
            line,
            flags=re.IGNORECASE,
        )
        if throughput_match is not None:
            metrics.throughput_qps = float(throughput_match.group(1))

        for label, field_name in _STATS_LINE_FIELDS:
            if re.search(rf"\b{re.escape(label)}\s*:", line, flags=re.IGNORECASE):
                stats = _parse_stats_fragment(line)
                if stats:
                    setattr(metrics, field_name, stats)
                break

        for label, field_name in _TOTAL_TIME_FIELDS:
            total_match = re.search(
                rf"\b{re.escape(label)}\s*:\s*({_FLOAT_PATTERN})\s*(ms|us|s)?",
                line,
                flags=re.IGNORECASE,
            )
            if total_match is not None:
                setattr(
                    metrics,
                    field_name,
                    _duration_to_ms(float(total_match.group(1)), total_match.group(2)),
                )
                break
    return metrics


def evaluate_tensorrt_performance_thresholds(
    metrics: TensorRTPerformanceMetrics,
    thresholds: Mapping[str, Any],
) -> TensorRTPerformanceThresholdReport:
    """Evaluate TensorRT performance thresholds.

    Threshold names must end with `_min` or `_max`. Examples:
    `throughput_qps_min`, `latency_mean_ms_max`,
    `gpu_compute_time_p99_ms_max`.
    """

    flattened = _flatten_performance_metrics(metrics)
    checks: list[TensorRTPerformanceCheck] = []
    for name, raw_threshold in thresholds.items():
        if name.endswith("_min"):
            metric_path = name[: -len("_min")]
            direction = ">="
        elif name.endswith("_max"):
            metric_path = name[: -len("_max")]
            direction = "<="
        else:
            raise ValueError(
                "TensorRT performance threshold names must end with _min or _max"
            )
        threshold = float(raw_threshold)
        value = flattened.get(metric_path)
        if value is None:
            checks.append(
                TensorRTPerformanceCheck(
                    name=name,
                    metric_path=metric_path,
                    value=None,
                    threshold=threshold,
                    direction=direction,
                    passed=False,
                )
            )
            continue
        checks.append(
            TensorRTPerformanceCheck(
                name=name,
                metric_path=metric_path,
                value=value,
                threshold=threshold,
                direction=direction,
                passed=value >= threshold if direction == ">=" else value <= threshold,
            )
        )
    return TensorRTPerformanceThresholdReport(
        passed=all(check.passed for check in checks),
        checks=checks,
    )
