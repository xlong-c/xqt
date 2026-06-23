"""TensorRT build adapters."""

from __future__ import annotations

import ctypes
import json
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import torch

from xqt.core.artifact import file_sha256
from xqt.core.errors import XQTBackendError


def _materialize_tensorrt_compatible_onnx(onnx_path: Path, output_path: Path) -> tuple[Path, dict[str, Any]]:
    """Rewrite known TensorRT-incompatible QDQ bias patterns into float initializers."""

    try:
        import numpy as np
        import onnx
        from onnx import TensorProto, helper, numpy_helper
    except ImportError:
        return onnx_path, {"applied": False, "reason": "onnx or numpy unavailable"}

    try:
        model = onnx.load(str(onnx_path))
    except Exception as exc:
        return onnx_path, {
            "applied": False,
            "reason": f"failed_to_parse_onnx: {type(exc).__name__}",
        }
    initializers = {initializer.name: initializer for initializer in model.graph.initializer}
    consumers: dict[str, list[Any]] = {}
    for node in model.graph.node:
        for input_name in node.input:
            consumers.setdefault(str(input_name), []).append(node)

    rewritten_biases: list[dict[str, Any]] = []
    kept_nodes: list[Any] = []
    removed_output_names: set[str] = set()
    removed_initializer_names: set[str] = set()

    for node in model.graph.node:
        if node.op_type != "DequantizeLinear" or len(node.input) < 3 or len(node.output) != 1:
            kept_nodes.append(node)
            continue

        quantized_name = str(node.input[0])
        scale_name = str(node.input[1])
        zero_point_name = str(node.input[2])
        output_name = str(node.output[0])
        quantized_initializer = initializers.get(quantized_name)
        scale_initializer = initializers.get(scale_name)
        zero_point_initializer = initializers.get(zero_point_name)
        output_consumers = consumers.get(output_name, [])

        if (
            quantized_initializer is None
            or scale_initializer is None
            or zero_point_initializer is None
            or quantized_initializer.data_type != TensorProto.INT32
            or len(output_consumers) != 1
        ):
            kept_nodes.append(node)
            continue

        consumer = output_consumers[0]
        bias_input_index = None
        if consumer.op_type == "Conv" and len(consumer.input) >= 3 and str(consumer.input[2]) == output_name:
            bias_input_index = 2
        elif consumer.op_type == "Gemm" and len(consumer.input) >= 3 and str(consumer.input[2]) == output_name:
            bias_input_index = 2
        if bias_input_index is None:
            kept_nodes.append(node)
            continue

        quantized_values = numpy_helper.to_array(quantized_initializer).astype(np.int32, copy=False)
        scale_values = numpy_helper.to_array(scale_initializer).astype(np.float32, copy=False)
        zero_point_values = numpy_helper.to_array(zero_point_initializer).astype(np.int32, copy=False)
        float_bias = (quantized_values - zero_point_values).astype(np.float32) * scale_values.astype(np.float32)
        bias_initializer = numpy_helper.from_array(float_bias.astype(np.float32), name=output_name)

        consumer.input[bias_input_index] = output_name
        initializers[output_name] = bias_initializer
        removed_output_names.add(output_name)
        removed_initializer_names.update({quantized_name, scale_name, zero_point_name})
        rewritten_biases.append(
            {
                "node_name": str(node.name or output_name),
                "consumer_name": str(consumer.name or consumer.op_type),
                "consumer_op_type": str(consumer.op_type),
                "bias_name": output_name,
                "shape": [int(dim) for dim in float_bias.shape],
            }
        )

    if not rewritten_biases:
        return onnx_path, {"applied": False, "rewritten_bias_count": 0}

    retained_initializers = []
    for initializer in model.graph.initializer:
        if initializer.name in removed_initializer_names or initializer.name in removed_output_names:
            continue
        retained_initializers.append(initializer)
    retained_initializers.extend(
        initializers[name]
        for name in removed_output_names
        if name in initializers
    )

    del model.graph.node[:]
    model.graph.node.extend(kept_nodes)
    del model.graph.initializer[:]
    model.graph.initializer.extend(retained_initializers)

    filtered_value_info = [value for value in model.graph.value_info if value.name not in removed_output_names]
    del model.graph.value_info[:]
    model.graph.value_info.extend(filtered_value_info)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(output_path))
    return output_path, {
        "applied": True,
        "rewritten_bias_count": len(rewritten_biases),
        "rewritten_biases": rewritten_biases,
        "source_onnx": str(onnx_path),
        "sanitized_onnx": str(output_path),
    }


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


def _normalize_plugin_libraries(
    plugin_libraries: Optional[Sequence[str | Path]],
) -> list[Path]:
    normalized: list[Path] = []
    for raw in plugin_libraries or ():
        path = Path(raw)
        if path in normalized:
            continue
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


def _import_tensorrt() -> Any:
    try:
        return import_module("tensorrt")
    except ImportError as exc:
        raise XQTBackendError("tensorrt is required for TensorRT python_api builds") from exc


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


def summarize_tensorrt_engine_inspector(raw: Mapping[str, Any]) -> TensorRTEngineInspectorSummary:
    """Summarize TensorRT engine inspector JSON into quant/fusion-oriented signals."""

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
    """Inspect a TensorRT engine and summarize layer-level quantization/fusion signals."""

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

    info_json = inspector.get_engine_information(trt.LayerInformationFormat.JSON)
    try:
        raw = json.loads(info_json)
    except json.JSONDecodeError as exc:
        raise XQTBackendError("Failed to parse TensorRT engine inspector JSON") from exc
    if profiling_verbosity is not None:
        raw["ProfilingVerbosity"] = profiling_verbosity
    if loaded_plugins:
        raw["PluginLibraries"] = loaded_plugins
    return summarize_tensorrt_engine_inspector(raw)


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
    plugin_libraries: Optional[Sequence[str | Path]] = None,
    serialize_plugin_libraries: bool = True,
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
    for plugin_path in _normalize_plugin_libraries(plugin_libraries):
        command.append(f"--dynamicPlugins={plugin_path}")
        if serialize_plugin_libraries:
            command.append(f"--setPluginsToSerialize={plugin_path}")
    command.extend(str(arg) for arg in (extra_args or ()))
    return command


def _shape_to_tuple(values: Sequence[Any]) -> tuple[int, ...]:
    return tuple(int(value) for value in values)


def _set_builder_precision_flags(
    trt: Any,
    builder_config: Any,
    *,
    precision: Optional[str],
) -> tuple[list[str], list[str]]:
    if precision is None:
        return [], []
    normalized = precision.lower()
    supported = {"fp16", "bf16", "int8", "fp8"}
    if normalized not in supported:
        raise ValueError("precision must be one of fp16, bf16, int8, fp8")
    applied: list[str] = []
    notes: list[str] = []

    if normalized == "fp16":
        if hasattr(trt.BuilderFlag, "FP16"):
            builder_config.set_flag(trt.BuilderFlag.FP16)
            applied.append("fp16")
        else:
            notes.append(
                "BuilderFlag.FP16 is not exposed by this TensorRT build; "
                "continuing without an explicit FP16 weak-typing flag"
            )
    elif normalized == "bf16":
        if hasattr(trt.BuilderFlag, "BF16"):
            builder_config.set_flag(trt.BuilderFlag.BF16)
            applied.append("bf16")
        else:
            notes.append(
                "BuilderFlag.BF16 is not exposed by this TensorRT build; "
                "continuing without an explicit BF16 weak-typing flag"
            )
    elif normalized == "int8":
        if hasattr(trt.BuilderFlag, "INT8"):
            builder_config.set_flag(trt.BuilderFlag.INT8)
            applied.append("int8")
        else:
            notes.append(
                "BuilderFlag.INT8 is not exposed by this TensorRT build; "
                "assuming explicit Q/DQ or strong-typing flow"
            )
    elif normalized == "fp8":
        if hasattr(trt.BuilderFlag, "FP8"):
            builder_config.set_flag(trt.BuilderFlag.FP8)
            applied.append("fp8")
        else:
            raise XQTBackendError("current TensorRT build does not expose BuilderFlag.FP8")
    return applied, notes


def _get_network_input(network: Any, name: str) -> Any | None:
    for index in range(network.num_inputs):
        tensor = network.get_input(index)
        if tensor.name == name:
            return tensor
    return None


def _infer_static_profile(network: Any) -> dict[str, Any]:
    profile: dict[str, Any] = {}
    for index in range(network.num_inputs):
        tensor = network.get_input(index)
        shape = [int(dim) for dim in tensor.shape]
        if any(dim < 0 for dim in shape):
            raise ValueError(
                "dynamic TensorRT network requires explicit profiles for python_api build"
            )
        profile[str(tensor.name)] = {"min": shape, "opt": shape, "max": shape}
    return profile


def _apply_python_profiles(
    builder: Any,
    builder_config: Any,
    network: Any,
    profiles: Mapping[str, Any],
) -> dict[str, Any]:
    profile = builder.create_optimization_profile()
    normalized: dict[str, Any] = {}
    for input_name, spec in profiles.items():
        if not isinstance(spec, Mapping):
            raise ValueError("TensorRT profile entries must be mappings")
        for key in ("min", "opt", "max"):
            if key not in spec:
                raise ValueError(f"TensorRT profile for '{input_name}' missing '{key}'")
        tensor = _get_network_input(network, str(input_name))
        if tensor is None:
            raise KeyError(f"TensorRT network input not found: {input_name}")
        min_shape = _shape_to_tuple(spec["min"])
        opt_shape = _shape_to_tuple(spec["opt"])
        max_shape = _shape_to_tuple(spec["max"])
        ok = profile.set_shape(str(input_name), min_shape, opt_shape, max_shape)
        if ok is False:
            raise XQTBackendError(f"TensorRT rejected optimization profile for input {input_name}")
        normalized[str(input_name)] = {
            "min": list(min_shape),
            "opt": list(opt_shape),
            "max": list(max_shape),
        }
    builder_config.add_optimization_profile(profile)
    return normalized


def _parse_onnx_network(trt: Any, onnx: Path, *, log_level: Optional[str] = None) -> tuple[Any, Any, Any]:
    severity_name = str(log_level or "INFO").upper()
    severity = getattr(getattr(trt, "Logger"), severity_name, trt.Logger.INFO)
    logger = trt.Logger(severity)
    builder = trt.Builder(logger)
    if hasattr(trt.NetworkDefinitionCreationFlag, "EXPLICIT_BATCH"):
        network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    else:
        network_flags = 0
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)
    parsed = parser.parse(onnx.read_bytes())
    if not parsed:
        errors = [str(parser.get_error(index)) for index in range(parser.num_errors)]
        raise XQTBackendError("Failed to parse ONNX for TensorRT:\n" + "\n".join(errors))
    builder_config = builder.create_builder_config()
    return builder, network, builder_config


def _build_tensorrt_engine_python_api(
    onnx: Path,
    engine: Path,
    *,
    precision: Optional[str],
    profiles: Optional[Mapping[str, Any]],
    workspace_mib: int,
    builder_optimization_level: Optional[int],
    timing_cache_path: Optional[str | Path],
    dry_run: bool,
    log_level: Optional[str],
    plugin_libraries: Optional[Sequence[str | Path]],
) -> TensorRTBuildResult:
    normalized_plugins = _normalize_plugin_libraries(plugin_libraries)
    metadata: dict[str, Any] = {
        "backend": "python_api",
        "precision": precision,
        "profiles": dict(profiles or {}),
        "workspace_mib": workspace_mib,
        "builder_optimization_level": builder_optimization_level,
        "timing_cache_path": str(timing_cache_path) if timing_cache_path is not None else None,
        "profiling_verbosity": "DETAILED",
        "plugin_libraries": [str(path) for path in normalized_plugins],
    }
    if dry_run:
        return TensorRTBuildResult(
            engine_path=engine,
            command=["tensorrt-python-api", f"--onnx={onnx}", f"--saveEngine={engine}"],
            dry_run=True,
            metadata=metadata,
        )

    loaded_plugins = _load_tensorrt_plugin_libraries(normalized_plugins)
    trt = _import_tensorrt()
    if hasattr(trt, "init_libnvinfer_plugins"):
        trt.init_libnvinfer_plugins(trt.Logger(trt.Logger.INFO), "")
    builder, network, builder_config = _parse_onnx_network(trt, onnx, log_level=log_level)
    builder_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_mib) << 20)
    if hasattr(builder_config, "profiling_verbosity") and hasattr(trt, "ProfilingVerbosity"):
        builder_config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    if (
        builder_optimization_level is not None
        and hasattr(builder_config, "builder_optimization_level")
    ):
        builder_config.builder_optimization_level = int(builder_optimization_level)
    flags, notes = _set_builder_precision_flags(trt, builder_config, precision=precision)
    metadata["builder_flags"] = flags
    if notes:
        metadata["builder_notes"] = notes
    normalized_profiles = dict(profiles or {})
    if profiles:
        normalized_profiles = _apply_python_profiles(builder, builder_config, network, profiles)
    elif any(-1 in tuple(network.get_input(index).shape) for index in range(network.num_inputs)):
        normalized_profiles = _infer_static_profile(network)
        normalized_profiles = _apply_python_profiles(
            builder,
            builder_config,
            network,
            normalized_profiles,
        )
    metadata["profiles"] = normalized_profiles

    timing_cache_file = Path(timing_cache_path) if timing_cache_path is not None else None
    if timing_cache_file is not None:
        raw = timing_cache_file.read_bytes() if timing_cache_file.exists() else b""
        cache = builder_config.create_timing_cache(raw)
        builder_config.set_timing_cache(cache, False)

    serialized = builder.build_serialized_network(network, builder_config)
    if serialized is None:
        raise XQTBackendError("TensorRT python_api build returned no serialized engine")
    engine.write_bytes(bytes(serialized))
    checksum = file_sha256(engine)

    if timing_cache_file is not None:
        timing_cache_file.parent.mkdir(parents=True, exist_ok=True)
        timing_cache_file.write_bytes(bytes(builder_config.get_timing_cache().serialize()))

    metadata["input_tensors"] = [
        {
            "name": str(network.get_input(index).name),
            "shape": [int(dim) for dim in network.get_input(index).shape],
        }
        for index in range(network.num_inputs)
    ]
    metadata["output_tensors"] = [
        {
            "name": str(network.get_output(index).name),
            "shape": [int(dim) for dim in network.get_output(index).shape],
        }
        for index in range(network.num_outputs)
    ]
    try:
        inspector_summary = inspect_tensorrt_engine(
            engine,
            profiling_verbosity=metadata.get("profiling_verbosity"),
            plugin_libraries=normalized_plugins,
        )
        metadata["engine_inspector"] = inspector_summary.to_dict()
    except XQTBackendError as exc:
        metadata["engine_inspector_error"] = str(exc)
    if loaded_plugins:
        metadata["loaded_plugin_libraries"] = loaded_plugins
    return TensorRTBuildResult(
        engine_path=engine,
        command=["tensorrt-python-api", f"--onnx={onnx}", f"--saveEngine={engine}"],
        returncode=0,
        checksum=checksum,
        dry_run=False,
        metadata=metadata,
    )


def _torch_dtype_from_trt(dtype: Any, trt: Any) -> torch.dtype:
    mapping = {
        trt.float32: torch.float32,
        trt.float16: torch.float16,
        getattr(trt, "bfloat16", None): torch.bfloat16,
        trt.int8: torch.int8,
        trt.int32: torch.int32,
        trt.int64: torch.int64,
        trt.bool: torch.bool,
    }
    resolved = mapping.get(dtype)
    if resolved is None:
        raise TypeError(f"Unsupported TensorRT dtype: {dtype}")
    return resolved


def create_tensorrt_runtime_session(
    engine_path: str | Path,
    *,
    device: str = "cuda:0",
    plugin_libraries: Optional[Sequence[str | Path]] = None,
) -> TensorRTRuntimeSession:
    """Create a reusable TensorRT runtime session."""

    engine = Path(engine_path)
    if not engine.is_file():
        raise XQTBackendError(f"TensorRT engine file not found: {engine}")

    loaded_plugins = _load_tensorrt_plugin_libraries(plugin_libraries)
    trt = _import_tensorrt()
    if hasattr(trt, "init_libnvinfer_plugins"):
        trt.init_libnvinfer_plugins(trt.Logger(trt.Logger.INFO), "")
    runtime = trt.Runtime(trt.Logger(trt.Logger.INFO))
    serialized = runtime.deserialize_cuda_engine(engine.read_bytes())
    if serialized is None:
        raise XQTBackendError(f"Failed to deserialize TensorRT engine: {engine}")
    context = serialized.create_execution_context()
    if context is None:
        raise XQTBackendError("Failed to create TensorRT execution context")
    return TensorRTRuntimeSession(
        engine_path=engine,
        device=device,
        trt=trt,
        runtime=runtime,
        engine=serialized,
        context=context,
        engine_inspector=inspect_tensorrt_engine(
            engine,
            plugin_libraries=plugin_libraries,
        ).to_dict()
        | (
            {"plugin_libraries": loaded_plugins}
            if loaded_plugins
            else {}
        ),
    )


def execute_tensorrt_engine(
    engine_path: str | Path,
    *,
    inputs: Mapping[str, torch.Tensor],
    device: str = "cuda:0",
    plugin_libraries: Optional[Sequence[str | Path]] = None,
) -> TensorRTRuntimeExecutionResult:
    """Execute a TensorRT engine once with explicit input tensors."""

    session = create_tensorrt_runtime_session(
        engine_path,
        device=device,
        plugin_libraries=plugin_libraries,
    )
    return execute_tensorrt_session(session, inputs=inputs)


def execute_tensorrt_session(
    session: TensorRTRuntimeSession,
    *,
    inputs: Mapping[str, torch.Tensor],
) -> TensorRTRuntimeExecutionResult:
    """Execute a reusable TensorRT session once with explicit input tensors."""

    torch_device = torch.device(session.device)
    prepared_inputs: dict[str, torch.Tensor] = {}
    for name, tensor in inputs.items():
        tensor_on_device = tensor.to(device=torch_device)
        shape_tuple = tuple(int(dim) for dim in tensor_on_device.shape)
        ok = session.context.set_input_shape(str(name), shape_tuple)
        if ok is False:
            raise XQTBackendError(f"TensorRT rejected input shape for {name}: {shape_tuple}")
        prepared_inputs[str(name)] = tensor_on_device

    unresolved = session.context.infer_shapes()
    if unresolved:
        raise XQTBackendError(
            f"TensorRT shape inference has unresolved tensors: {list(unresolved)}"
        )

    output_tensors: dict[str, torch.Tensor] = {}
    output_shapes: dict[str, list[int]] = {}
    bindings: list[int] = []
    for name in session.engine:
        mode = session.engine.get_tensor_mode(name)
        if mode == session.trt.TensorIOMode.INPUT:
            if name not in prepared_inputs:
                raise XQTBackendError(f"TensorRT input tensor missing: {name}")
            bindings.append(int(prepared_inputs[name].data_ptr()))
            continue
        shape = tuple(int(dim) for dim in session.context.get_tensor_shape(name))
        if any(dim < 0 for dim in shape):
            raise XQTBackendError(f"TensorRT output shape is unresolved for {name}: {shape}")
        dtype = _torch_dtype_from_trt(session.engine.get_tensor_dtype(name), session.trt)
        output_tensor = torch.empty(shape, dtype=dtype, device=torch_device)
        output_tensors[str(name)] = output_tensor
        output_shapes[str(name)] = [int(dim) for dim in shape]
        bindings.append(int(output_tensor.data_ptr()))

    ok = session.context.execute_v2(bindings)
    if not ok:
        raise XQTBackendError("TensorRT engine execution failed")

    return TensorRTRuntimeExecutionResult(
        engine_path=session.engine_path,
        backend="python_api",
        device=session.device,
        input_shapes={
            str(name): [int(dim) for dim in tensor.shape]
            for name, tensor in prepared_inputs.items()
        },
        output_tensors=output_tensors,
        output_shapes=output_shapes,
        metadata={"engine_inspector": dict(session.engine_inspector)},
    )


def benchmark_tensorrt_engine(
    engine_path: str | Path,
    *,
    input_shapes: Mapping[str, Sequence[int]],
    warmup: int = 10,
    iterations: int = 50,
    device: str = "cuda:0",
    fill_random: bool = True,
    plugin_libraries: Optional[Sequence[str | Path]] = None,
) -> TensorRTRuntimeBenchmarkResult:
    """Benchmark a TensorRT engine using the TensorRT Python runtime."""

    engine = Path(engine_path)
    session = create_tensorrt_runtime_session(
        engine,
        device=device,
        plugin_libraries=plugin_libraries,
    )

    torch_device = torch.device(device)
    inputs: dict[str, torch.Tensor] = {}
    for name, shape in input_shapes.items():
        shape_tuple = _shape_to_tuple(shape)
        if fill_random:
            tensor = torch.rand(shape_tuple, dtype=torch.float32, device=torch_device)
        else:
            tensor = torch.zeros(shape_tuple, dtype=torch.float32, device=torch_device)
        inputs[str(name)] = tensor

    def run_once() -> object:
        return execute_tensorrt_session(session, inputs=inputs).output_tensors

    from xqt.benchmark import benchmark_callable

    latency = benchmark_callable(
        run_once,
        warmup=warmup,
        iterations=iterations,
        sync_cuda=True,
        device=device,
    ).to_dict()
    execution = execute_tensorrt_session(session, inputs=inputs)
    return TensorRTRuntimeBenchmarkResult(
        engine_path=engine,
        backend="python_api",
        device=device,
        input_shapes={
            str(name): [int(dim) for dim in _shape_to_tuple(shape)]
            for name, shape in input_shapes.items()
        },
        latency=latency,
        output_shapes=execution.output_shapes,
        metadata={
            "fill_random": fill_random,
            **execution.metadata,
        },
    )


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
    backend: str = "trtexec",
    workspace_mib: int = 4096,
    builder_optimization_level: Optional[int] = None,
    timing_cache_path: str | Path | None = None,
    log_level: Optional[str] = None,
    plugin_libraries: Optional[Sequence[str | Path]] = None,
    serialize_plugin_libraries: bool = True,
) -> TensorRTBuildResult:
    """Build a TensorRT engine via `trtexec` or TensorRT Python API."""

    onnx = Path(onnx_path)
    if not onnx.is_file():
        raise XQTBackendError(f"ONNX file not found: {onnx}")

    engine = Path(engine_path)
    engine.parent.mkdir(parents=True, exist_ok=True)
    compat_metadata: dict[str, Any] = {}
    compat_onnx = onnx
    if not dry_run:
        compat_onnx, compat_metadata = _materialize_tensorrt_compatible_onnx(
            onnx,
            engine.with_suffix(".trt_compatible.onnx"),
        )
    normalized_backend = backend.lower()
    if normalized_backend == "python_api":
        result = _build_tensorrt_engine_python_api(
            compat_onnx,
            engine,
            precision=precision,
            profiles=profiles,
            workspace_mib=workspace_mib,
            builder_optimization_level=builder_optimization_level,
            timing_cache_path=timing_cache_path,
            dry_run=dry_run,
            log_level=log_level,
            plugin_libraries=plugin_libraries,
        )
        if compat_metadata:
            result.metadata["onnx_compat"] = compat_metadata
        return result
    if normalized_backend != "trtexec":
        raise ValueError("backend must be one of trtexec, python_api")
    command = build_trtexec_command(
        compat_onnx,
        engine,
        precision=precision,
        profiles=profiles,
        trtexec_path=trtexec_path,
        extra_args=extra_args,
        plugin_libraries=plugin_libraries,
        serialize_plugin_libraries=serialize_plugin_libraries,
    )

    if dry_run:
        return TensorRTBuildResult(
            engine_path=engine,
            command=command,
            dry_run=True,
            metadata={
                "backend": "trtexec",
                "precision": precision,
                "profiles": dict(profiles or {}),
                "performance_thresholds": dict(performance_thresholds or {}),
                "plugin_libraries": [
                    str(path) for path in _normalize_plugin_libraries(plugin_libraries)
                ],
                "serialize_plugin_libraries": bool(serialize_plugin_libraries),
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
            "backend": "trtexec",
            "precision": precision,
            "profiles": dict(profiles or {}),
            "onnx_compat": compat_metadata,
            "performance": performance.to_dict(),
            "performance_thresholds": dict(performance_thresholds or {}),
            "performance_threshold_report": (
                threshold_report.to_dict() if threshold_report is not None else None
            ),
            "plugin_libraries": [
                str(path) for path in _normalize_plugin_libraries(plugin_libraries)
            ],
            "serialize_plugin_libraries": bool(serialize_plugin_libraries),
        },
    )


__all__ = [
    "TensorRTBuildResult",
    "TensorRTEngineInspectorSummary",
    "TensorRTRuntimeBenchmarkResult",
    "TensorRTRuntimeExecutionResult",
    "TensorRTRuntimeSession",
    "TensorRTPerformanceCheck",
    "TensorRTPerformanceMetrics",
    "TensorRTPerformanceThresholdReport",
    "benchmark_tensorrt_engine",
    "build_tensorrt_engine",
    "build_trtexec_command",
    "create_tensorrt_runtime_session",
    "execute_tensorrt_engine",
    "execute_tensorrt_session",
    "evaluate_tensorrt_performance_thresholds",
    "inspect_tensorrt_engine",
    "parse_trtexec_performance",
    "summarize_tensorrt_engine_inspector",
]
