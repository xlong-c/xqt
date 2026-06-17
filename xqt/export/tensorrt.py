"""TensorRT trtexec adapter."""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from xqt.core.artifact import file_sha256
from xqt.core.errors import XQTBackendError


@dataclass
class TensorRTBuildResult:
    """Result from a TensorRT engine build attempt."""

    engine_path: Path
    command: list[str]
    returncode: Optional[int] = None
    stdout: str = ""
    stderr: str = ""
    checksum: Optional[str] = None
    dry_run: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TensorRTPerformanceMetrics:
    """Parsed TensorRT trtexec performance summary."""

    throughput_qps: Optional[float] = None
    latency_ms: dict[str, float] = field(default_factory=dict)
    host_latency_ms: dict[str, float] = field(default_factory=dict)
    enqueue_time_ms: dict[str, float] = field(default_factory=dict)
    h2d_latency_ms: dict[str, float] = field(default_factory=dict)
    d2h_latency_ms: dict[str, float] = field(default_factory=dict)
    gpu_compute_time_ms: dict[str, float] = field(default_factory=dict)
    total_host_walltime_ms: Optional[float] = None
    total_gpu_compute_time_ms: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert non-empty metrics to a plain dictionary."""

        data: dict[str, Any] = {}
        if self.throughput_qps is not None:
            data["throughput_qps"] = self.throughput_qps
        for key in (
            "latency_ms",
            "host_latency_ms",
            "enqueue_time_ms",
            "h2d_latency_ms",
            "d2h_latency_ms",
            "gpu_compute_time_ms",
        ):
            value = getattr(self, key)
            if value:
                data[key] = dict(value)
        if self.total_host_walltime_ms is not None:
            data["total_host_walltime_ms"] = self.total_host_walltime_ms
        if self.total_gpu_compute_time_ms is not None:
            data["total_gpu_compute_time_ms"] = self.total_gpu_compute_time_ms
        return data


@dataclass
class TensorRTPerformanceCheck:
    """A single TensorRT performance threshold check."""

    name: str
    metric_path: str
    value: Optional[float]
    threshold: float
    direction: str
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        """Convert the check to a plain dictionary."""

        return {
            "name": self.name,
            "metric_path": self.metric_path,
            "value": self.value,
            "threshold": self.threshold,
            "direction": self.direction,
            "passed": self.passed,
        }


@dataclass
class TensorRTPerformanceThresholdReport:
    """TensorRT performance threshold report."""

    passed: bool
    checks: list[TensorRTPerformanceCheck] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert the report to a plain dictionary."""

        return {
            "passed": self.passed,
            "checks": [check.to_dict() for check in self.checks],
        }


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


def _shape_to_string(shape: Sequence[int]) -> str:
    if not shape:
        raise ValueError("shape must not be empty")
    return "x".join(str(int(dim)) for dim in shape)


def _profiles_to_args(profiles: Mapping[str, Any]) -> list[str]:
    args: list[str] = []
    min_shapes: list[str] = []
    opt_shapes: list[str] = []
    max_shapes: list[str] = []

    for input_name, profile in profiles.items():
        if not isinstance(profile, Mapping):
            raise ValueError("TensorRT profile entries must be mappings")
        for key in ("min", "opt", "max"):
            if key not in profile:
                raise ValueError(f"TensorRT profile for '{input_name}' missing '{key}'")
        min_shapes.append(f"{input_name}:{_shape_to_string(profile['min'])}")
        opt_shapes.append(f"{input_name}:{_shape_to_string(profile['opt'])}")
        max_shapes.append(f"{input_name}:{_shape_to_string(profile['max'])}")

    if min_shapes:
        args.extend(
            [
                f"--minShapes={','.join(min_shapes)}",
                f"--optShapes={','.join(opt_shapes)}",
                f"--maxShapes={','.join(max_shapes)}",
            ]
        )
    return args


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


def build_trtexec_command(
    onnx_path: str | Path,
    engine_path: str | Path,
    *,
    precision: Optional[str] = None,
    profiles: Optional[Mapping[str, Any]] = None,
    trtexec_path: str = "trtexec",
    extra_args: Optional[Sequence[str]] = None,
) -> list[str]:
    """Build a trtexec command for ONNX -> TensorRT engine conversion."""

    command = [
        trtexec_path,
        f"--onnx={Path(onnx_path)}",
        f"--saveEngine={Path(engine_path)}",
    ]
    if precision:
        normalized = precision.lower()
        if normalized not in {"fp16", "bf16", "int8", "fp8"}:
            raise ValueError("precision must be one of fp16, bf16, int8, fp8")
        command.append(f"--{normalized}")
    if profiles:
        command.extend(_profiles_to_args(profiles))
    command.extend(str(arg) for arg in (extra_args or ()))
    return command


def build_tensorrt_engine(
    onnx_path: str | Path,
    engine_path: str | Path,
    *,
    precision: Optional[str] = None,
    profiles: Optional[Mapping[str, Any]] = None,
    trtexec_path: str = "trtexec",
    extra_args: Optional[Sequence[str]] = None,
    timeout: Optional[float] = None,
    dry_run: bool = False,
    performance_thresholds: Optional[Mapping[str, Any]] = None,
) -> TensorRTBuildResult:
    """Build a TensorRT engine with trtexec, or return the command in dry-run mode."""

    onnx = Path(onnx_path)
    if not onnx.is_file():
        raise XQTBackendError(f"ONNX file not found: {onnx}")

    engine = Path(engine_path)
    engine.parent.mkdir(parents=True, exist_ok=True)
    command = build_trtexec_command(
        onnx,
        engine,
        precision=precision,
        profiles=profiles,
        trtexec_path=trtexec_path,
        extra_args=extra_args,
    )

    if dry_run:
        return TensorRTBuildResult(
            engine_path=engine,
            command=command,
            dry_run=True,
            metadata={
                "precision": precision,
                "profiles": dict(profiles or {}),
                "performance_thresholds": dict(performance_thresholds or {}),
            },
        )

    executable = shutil.which(trtexec_path)
    if executable is None:
        raise XQTBackendError(f"trtexec executable not found: {trtexec_path}")
    command[0] = executable

    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    checksum = file_sha256(engine) if engine.is_file() else None
    if completed.returncode != 0:
        raise XQTBackendError(
            f"trtexec failed with return code {completed.returncode}: "
            f"{completed.stderr.strip()}"
        )
    if checksum is None:
        raise XQTBackendError(f"trtexec did not create engine: {engine}")
    performance = parse_trtexec_performance(
        "\n".join(part for part in (completed.stdout, completed.stderr) if part)
    )
    threshold_report = None
    if performance_thresholds:
        threshold_report = evaluate_tensorrt_performance_thresholds(
            performance,
            performance_thresholds,
        )

    return TensorRTBuildResult(
        engine_path=engine,
        command=command,
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
        checksum=checksum,
        dry_run=False,
        metadata={
            "precision": precision,
            "profiles": dict(profiles or {}),
            "performance": performance.to_dict(),
            "performance_thresholds": dict(performance_thresholds or {}),
            "performance_threshold_report": (
                threshold_report.to_dict() if threshold_report is not None else None
            ),
        },
    )


__all__ = [
    "TensorRTBuildResult",
    "TensorRTPerformanceCheck",
    "TensorRTPerformanceMetrics",
    "TensorRTPerformanceThresholdReport",
    "build_tensorrt_engine",
    "build_trtexec_command",
    "evaluate_tensorrt_performance_thresholds",
    "parse_trtexec_performance",
]
