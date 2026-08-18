"""TensorRT plugin, inspector, and performance diagnostics."""

from __future__ import annotations

import ctypes
import json
import re
from importlib import import_module
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from xqt.core.errors import XQTBackendError

from .trt_types import (
    TensorRTEngineInspectorSummary,
    TensorRTPerformanceCheck,
    TensorRTPerformanceMetrics,
    TensorRTPerformanceThresholdReport,
    TensorRTPluginLibraryCheck,
    TensorRTPluginValidationResult,
)


def _import_tensorrt() -> Any:
    try:
        return import_module("tensorrt")
    except ImportError as exc:
        raise XQTBackendError("tensorrt is required for TensorRT python_api builds") from exc


def _normalize_plugin_libraries(
    plugin_libraries: Optional[Sequence[str | Path]],
) -> list[Path]:
    normalized: list[Path] = []
    for raw in plugin_libraries or ():
        path = Path(raw)
        if path not in normalized:
            normalized.append(path)
    return normalized


def _load_tensorrt_plugin_libraries(
    plugin_libraries: Optional[Sequence[str | Path]],
) -> list[str]:
    loaded: list[str] = []
    for path in _normalize_plugin_libraries(plugin_libraries):
        if not path.is_file():
            raise XQTBackendError(f"TensorRT plugin library not found: {path}")
        try:
            ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
        except OSError as exc:
            raise XQTBackendError(
                f"Failed to load TensorRT plugin library '{path}': {exc}"
            ) from exc
        loaded.append(str(path))
    return loaded


def validate_tensorrt_plugin_libraries(
    plugin_libraries: Optional[Sequence[str | Path]],
    *,
    validate_loadability: bool = False,
) -> TensorRTPluginValidationResult:
    checks: list[TensorRTPluginLibraryCheck] = []
    for path in _normalize_plugin_libraries(plugin_libraries):
        exists = path.is_file()
        if not exists:
            checks.append(
                TensorRTPluginLibraryCheck(
                    path=str(path),
                    exists=False,
                    load_requested=validate_loadability,
                    error=f"TensorRT plugin library not found: {path}",
                )
            )
            continue
        if not validate_loadability:
            checks.append(TensorRTPluginLibraryCheck(path=str(path), exists=True))
            continue
        try:
            loaded = _load_tensorrt_plugin_libraries([path])
        except Exception as exc:
            checks.append(
                TensorRTPluginLibraryCheck(
                    path=str(path),
                    exists=True,
                    load_requested=True,
                    loadable=False,
                    error=str(exc),
                )
            )
            continue
        checks.append(
            TensorRTPluginLibraryCheck(
                path=str(path),
                exists=True,
                load_requested=True,
                loadable=True,
                loaded_plugin_libraries=loaded,
            )
        )
    if not checks:
        status = "not_requested"
    elif any(not check.exists for check in checks):
        status = "missing"
    elif validate_loadability and any(check.loadable is False for check in checks):
        status = "load_failed"
    elif validate_loadability:
        status = "ok"
    else:
        status = "present"
    return TensorRTPluginValidationResult(
        status=status,
        loadability_requested=validate_loadability,
        plugin_libraries=checks,
    )


def _normalize_trt_layer_names(raw_layers: Any) -> list[str]:
    if not isinstance(raw_layers, list):
        return []
    names: list[str] = []
    for item in raw_layers:
        if isinstance(item, str):
            names.append(item)
        elif isinstance(item, Mapping):
            name = item.get("Name") or item.get("LayerName") or item.get("name")
            if name is not None:
                names.append(str(name))
    return names


def summarize_tensorrt_engine_inspector(
    raw: Mapping[str, Any],
) -> TensorRTEngineInspectorSummary:
    layers = _normalize_trt_layer_names(raw.get("Layers"))
    quantize_layers = [name for name in layers if "QuantizeLinear" in name]
    dequantize_layers = [name for name in layers if "DequantizeLinear" in name]
    fusion_signatures = [name for name in layers if " + " in name]
    quantized_conv_layers = [
        name
        for name in layers
        if "quantized" in name.lower() and "conv" in name.lower()
    ]
    return TensorRTEngineInspectorSummary(
        profiling_verbosity=(
            str(raw.get("ProfilingVerbosity"))
            if raw.get("ProfilingVerbosity") is not None
            else None
        ),
        layer_count=len(layers),
        layers=layers,
        io_tensors=[
            dict(item) for item in raw.get("I/O Tensors", []) if isinstance(item, Mapping)
        ],
        engine_metadata=(
            dict(raw.get("Engine Metadata", {}))
            if isinstance(raw.get("Engine Metadata"), Mapping)
            else {}
        ),
        quantize_layer_count=len(quantize_layers),
        dequantize_layer_count=len(dequantize_layers),
        quantized_conv_count=len(quantized_conv_layers),
        fused_layer_count=len(fusion_signatures),
        has_quantization_layers=bool(quantize_layers or dequantize_layers),
        has_fused_quantized_conv=bool(quantized_conv_layers),
        fusion_signatures=fusion_signatures,
    )


def inspect_tensorrt_engine(
    engine_path: str | Path,
    *,
    profiling_verbosity: Optional[str] = None,
    plugin_libraries: Optional[Sequence[str | Path]] = None,
) -> TensorRTEngineInspectorSummary:
    engine = Path(engine_path)
    if not engine.is_file():
        raise XQTBackendError(f"TensorRT engine file not found: {engine}")
    loaded_plugins = _load_tensorrt_plugin_libraries(plugin_libraries)
    trt = _import_tensorrt()
    if hasattr(trt, "init_libnvinfer_plugins"):
        trt.init_libnvinfer_plugins(trt.Logger(trt.Logger.INFO), "")
    runtime = trt.Runtime(trt.Logger(trt.Logger.INFO))
    deserialized = runtime.deserialize_cuda_engine(engine.read_bytes())
    if deserialized is None:
        raise XQTBackendError(f"Failed to deserialize TensorRT engine: {engine}")
    inspector = deserialized.create_engine_inspector()
    if inspector is None:
        raise XQTBackendError("Failed to create TensorRT engine inspector")
    try:
        raw = json.loads(
            inspector.get_engine_information(trt.LayerInformationFormat.JSON)
        )
    except json.JSONDecodeError as exc:
        raise XQTBackendError("Failed to parse TensorRT engine inspector JSON") from exc
    if profiling_verbosity is not None:
        raw["ProfilingVerbosity"] = profiling_verbosity
    if loaded_plugins:
        raw["PluginLibraries"] = loaded_plugins
    return summarize_tensorrt_engine_inspector(raw)


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
    if (unit or "ms").lower() == "s":
        return value * 1000.0
    if (unit or "ms").lower() == "us":
        return value / 1000.0
    return value


def _parse_stats_fragment(fragment: str) -> dict[str, float]:
    stats: dict[str, float] = {}
    for match in _STAT_PATTERN.finditer(fragment):
        raw_name = match.group(1).lower()
        percentile = match.group(2)
        value = _duration_to_ms(float(match.group(3)), match.group(4))
        key = raw_name
        if percentile is not None:
            normalized = percentile.rstrip("0").rstrip(".").replace(".", "_")
            key = f"p{normalized}"
        stats[key] = value
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
        else:
            flattened[key] = float(value)
    return flattened


def parse_trtexec_performance(output: str) -> TensorRTPerformanceMetrics:
    metrics = TensorRTPerformanceMetrics()
    for line in output.splitlines():
        throughput = re.search(
            rf"\bThroughput\s*:\s*({_FLOAT_PATTERN})\s*(?:qps|queries/s|inferences/s)?",
            line,
            flags=re.IGNORECASE,
        )
        if throughput is not None:
            metrics.throughput_qps = float(throughput.group(1))
        for label, field_name in _STATS_LINE_FIELDS:
            if re.search(rf"\b{re.escape(label)}\s*:", line, flags=re.IGNORECASE):
                stats = _parse_stats_fragment(line)
                if stats:
                    setattr(metrics, field_name, stats)
                break
        for label, field_name in _TOTAL_TIME_FIELDS:
            total = re.search(
                rf"\b{re.escape(label)}\s*:\s*({_FLOAT_PATTERN})\s*(ms|us|s)?",
                line,
                flags=re.IGNORECASE,
            )
            if total is not None:
                setattr(
                    metrics,
                    field_name,
                    _duration_to_ms(float(total.group(1)), total.group(2)),
                )
                break
    return metrics


def evaluate_tensorrt_performance_thresholds(
    metrics: TensorRTPerformanceMetrics,
    thresholds: Mapping[str, Any],
) -> TensorRTPerformanceThresholdReport:
    flattened = _flatten_performance_metrics(metrics)
    checks: list[TensorRTPerformanceCheck] = []
    for name, raw_threshold in thresholds.items():
        if name.endswith("_min"):
            metric_path, direction = name[:-4], ">="
        elif name.endswith("_max"):
            metric_path, direction = name[:-4], "<="
        else:
            raise ValueError(
                "TensorRT performance threshold names must end with _min or _max"
            )
        threshold = float(raw_threshold)
        value = flattened.get(metric_path)
        checks.append(
            TensorRTPerformanceCheck(
                name=name,
                metric_path=metric_path,
                value=value,
                threshold=threshold,
                direction=direction,
                passed=(
                    False
                    if value is None
                    else value >= threshold
                    if direction == ">="
                    else value <= threshold
                ),
            )
        )
    return TensorRTPerformanceThresholdReport(
        passed=all(check.passed for check in checks),
        checks=checks,
    )


__all__ = [
    "evaluate_tensorrt_performance_thresholds",
    "inspect_tensorrt_engine",
    "parse_trtexec_performance",
    "summarize_tensorrt_engine_inspector",
    "validate_tensorrt_plugin_libraries",
]
