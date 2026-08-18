"""TensorRT engine build via trtexec CLI or Python API."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from xqt.core.artifact import file_sha256
from xqt.core.errors import XQTBackendError

from .trt_diagnostics import (
    evaluate_tensorrt_performance_thresholds,
    inspect_tensorrt_engine,
    parse_trtexec_performance,
    _import_tensorrt,
    _load_tensorrt_plugin_libraries,
    _normalize_plugin_libraries,
)
from .trt_types import TensorRTBuildResult


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
