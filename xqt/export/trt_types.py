"""TensorRT build, inspect, performance, and runtime result types."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

import torch

from xqt.core.artifact import file_sha256
from xqt.export.base import ExportResultBase


@dataclass
class TensorRTBuildResult(ExportResultBase):
    """Result from a TensorRT engine build attempt."""

    engine_path: Path
    command: list[str]
    returncode: Optional[int] = None
    stdout: str = ""
    stderr: str = ""
    checksum: Optional[str] = None
    dry_run: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def artifact_paths(self) -> tuple[Path, ...]:
        return (self.engine_path,)


@dataclass(frozen=True)
class TensorRTPluginLibraryCheck:
    """Validation result for one TensorRT plugin shared library."""

    path: str
    exists: bool
    load_requested: bool = False
    loadable: bool | None = None
    loaded_plugin_libraries: list[str] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "exists": self.exists,
            "load_requested": self.load_requested,
            "loadable": self.loadable,
            "loaded_plugin_libraries": list(self.loaded_plugin_libraries),
            "error": self.error,
        }


@dataclass(frozen=True)
class TensorRTPluginValidationResult:
    """Structured validation report for TensorRT plugin shared libraries."""

    status: str
    loadability_requested: bool
    plugin_libraries: list[TensorRTPluginLibraryCheck] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.status in {"present", "ok", "not_requested"}

    @property
    def loaded_plugin_libraries(self) -> list[str]:
        loaded: list[str] = []
        for check in self.plugin_libraries:
            loaded.extend(check.loaded_plugin_libraries)
        return loaded

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "passed": self.passed,
            "loadability_requested": self.loadability_requested,
            "plugin_libraries": [
                check.to_dict() for check in self.plugin_libraries
            ],
            "loaded_plugin_libraries": self.loaded_plugin_libraries,
        }


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


@dataclass
class TensorRTRuntimeBenchmarkResult:
    """TensorRT engine runtime benchmark result."""

    engine_path: Path
    backend: str
    device: str
    input_shapes: dict[str, list[int]]
    latency: dict[str, Any]
    output_shapes: dict[str, list[int]]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "engine_path": str(self.engine_path),
            "backend": self.backend,
            "device": self.device,
            "input_shapes": dict(self.input_shapes),
            "latency": dict(self.latency),
            "output_shapes": dict(self.output_shapes),
            "metadata": dict(self.metadata),
        }


@dataclass
class TensorRTRuntimeExecutionResult:
    """TensorRT engine runtime execution result."""

    engine_path: Path
    backend: str
    device: str
    input_shapes: dict[str, list[int]]
    output_tensors: dict[str, torch.Tensor]
    output_shapes: dict[str, list[int]]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "engine_path": str(self.engine_path),
            "backend": self.backend,
            "device": self.device,
            "input_shapes": dict(self.input_shapes),
            "output_shapes": dict(self.output_shapes),
            "metadata": dict(self.metadata),
        }


@dataclass
class TensorRTRuntimeSession:
    """Reusable TensorRT runtime session for repeated execution."""

    engine_path: Path
    device: str
    trt: Any
    runtime: Any
    engine: Any
    context: Any
    engine_inspector: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "engine_path": str(self.engine_path),
            "device": self.device,
            "handle_materialized": self.engine is not None and self.context is not None,
            "engine_inspector": dict(self.engine_inspector),
        }


@dataclass
class TensorRTEngineInspectorSummary:
    """Structured summary extracted from TensorRT engine inspector output."""

    profiling_verbosity: Optional[str] = None
    layer_count: int = 0
    layers: list[str] = field(default_factory=list)
    io_tensors: list[dict[str, Any]] = field(default_factory=list)
    engine_metadata: dict[str, Any] = field(default_factory=dict)
    quantize_layer_count: int = 0
    dequantize_layer_count: int = 0
    quantized_conv_count: int = 0
    fused_layer_count: int = 0
    has_quantization_layers: bool = False
    has_fused_quantized_conv: bool = False
    fusion_signatures: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "profiling_verbosity": self.profiling_verbosity,
            "layer_count": self.layer_count,
            "layers": list(self.layers),
            "io_tensors": [dict(item) for item in self.io_tensors],
            "engine_metadata": dict(self.engine_metadata),
            "quantize_layer_count": self.quantize_layer_count,
            "dequantize_layer_count": self.dequantize_layer_count,
            "quantized_conv_count": self.quantized_conv_count,
            "fused_layer_count": self.fused_layer_count,
            "has_quantization_layers": self.has_quantization_layers,
            "has_fused_quantized_conv": self.has_fused_quantized_conv,
            "fusion_signatures": list(self.fusion_signatures),
        }
