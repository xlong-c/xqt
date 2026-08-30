"""Export target capability and artifact lineage reports."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from xqt.core.schema import ExportTargetConfig

from .capability import DEFAULT_EXPORT_CAPABILITIES, ExportCapability
from .openvino import OpenVINOExportResult
from .trt_types import TensorRTBuildResult, TensorRTPluginValidationResult


def _capability_by_format() -> dict[str, ExportCapability]:
    return {record.format: record for record in DEFAULT_EXPORT_CAPABILITIES}


def _plugin_support(format_name: str) -> str:
    if format_name == "tensorrt":
        return "supported"
    if format_name in {"onnx", "openvino"}:
        return "backend-dependent"
    return "none"


def export_target_capability_report(
    targets: Sequence[ExportTargetConfig],
) -> dict[str, Any]:
    """Summarize requested export targets against the static capability matrix."""

    capabilities = _capability_by_format()
    rows: list[dict[str, Any]] = []
    unsupported: list[str] = []
    for index, target in enumerate(targets):
        capability = capabilities.get(target.format)
        precision = target.precision
        precision_supported = (
            True
            if capability is not None and precision is None
            else bool(capability is not None and precision in capability.precisions)
        )
        dynamic_requested = bool(target.dynamic_shapes)
        dynamic_supported = bool(capability.dynamic_shapes) if capability else False
        if capability is None:
            unsupported.append(target.format)
        rows.append(
            {
                "index": index,
                "format": target.format,
                "format_supported": capability is not None,
                "status": capability.status if capability else "unsupported",
                "maturity": capability.maturity if capability else "unsupported",
                "runtimes": list(capability.runtimes) if capability else [],
                "precision": precision,
                "precision_supported": precision_supported,
                "supported_precisions": list(capability.precisions) if capability else [],
                "opset": target.opset,
                "dynamic_shapes_requested": dynamic_requested,
                "dynamic_shapes_supported": dynamic_supported,
                "dynamic_shapes_compatible": (not dynamic_requested) or dynamic_supported,
                "quantization_supported": bool(capability.quantization) if capability else False,
                "sparsity_support": capability.sparse_support if capability else "unsupported",
                "plugin_support": _plugin_support(target.format),
                "plugin_libraries": list(target.tensorrt.plugin_libraries)
                if target.format == "tensorrt"
                else [],
                "plugin_loadability_requested": bool(
                    target.format == "tensorrt"
                    and target.tensorrt.validate_plugin_libraries_loadable
                ),
                "limitations": [] if capability else ["unsupported_export_format"],
            }
        )
    return {
        "target_count": len(targets),
        "supported_target_count": sum(
            1 for row in rows if bool(row["format_supported"])
        ),
        "unsupported_formats": unsupported,
        "targets": rows,
    }


def _artifact_size(path_value: object) -> int | None:
    if not isinstance(path_value, (str, Path)) or not str(path_value):
        return None
    path = Path(path_value)
    if not path.is_file():
        return None
    return path.stat().st_size


def _scalar_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "backend",
        "strategy",
        "method",
        "granularity",
        "target_sparsity",
        "execution_state",
        "applied",
        "target_count",
        "numeric_validation_failures",
    )
    return {key: payload[key] for key in keys if key in payload}


def _upstream_stage_lineage(metrics: Mapping[str, Any]) -> list[dict[str, Any]]:
    lineage: list[dict[str, Any]] = []
    for stage in ("quant", "prune", "operator_optimization"):
        payload = metrics.get(stage)
        if not isinstance(payload, Mapping):
            continue
        record: dict[str, Any] = {
            "stage": stage,
            "present": True,
            "summary": _scalar_summary(payload),
        }
        targets = payload.get("targets")
        if isinstance(targets, list):
            record["target_count"] = len(targets)
            record["applied_target_count"] = sum(
                1
                for item in targets
                if isinstance(item, Mapping) and bool(item.get("applied"))
            )
        lineage.append(record)
    return lineage


def export_artifact_lineage_report(
    *,
    exported: Sequence[Mapping[str, object]],
    target_summaries: Sequence[Mapping[str, object]],
    metrics: Mapping[str, Any],
    stage_kind: str,
    source_stage: str | None = None,
) -> dict[str, Any]:
    """Describe export artifacts and their upstream optimization lineage."""

    upstream = _upstream_stage_lineage(metrics)
    resolved_source = source_stage if source_stage is not None else stage_kind
    artifacts: list[dict[str, Any]] = []
    for index, artifact in enumerate(exported):
        target = target_summaries[index] if index < len(target_summaries) else {}
        path = artifact.get("path")
        input_names = artifact.get("input_names", target.get("input_names", []))
        output_names = artifact.get("output_names", target.get("output_names", []))
        dynamic_shapes = artifact.get(
            "dynamic_shapes",
            target.get("dynamic_shapes", {}),
        )
        artifacts.append(
            {
                "index": index,
                "format": artifact.get("format"),
                "path": str(path) if path is not None else None,
                "source_stage": resolved_source,
                "artifact_checksum": artifact.get("checksum"),
                "artifact_size_bytes": _artifact_size(path),
                "input_signature": {
                    "input_names": list(input_names)
                    if isinstance(input_names, list)
                    else [],
                    "dynamic_shapes": dict(dynamic_shapes)
                    if isinstance(dynamic_shapes, Mapping)
                    else {},
                },
                "output_signature": {
                    "output_names": list(output_names)
                    if isinstance(output_names, list)
                    else [],
                },
                "upstream_stage_lineage": [dict(item) for item in upstream],
            }
        )
    return {
        "artifact_count": len(artifacts),
        "source_stage": resolved_source,
        "upstream_stage_count": len(upstream),
        "upstream_stages": [item["stage"] for item in upstream],
        "artifacts": artifacts,
    }


def _source_to_string(source: object) -> str | None:
    if source is None:
        return None
    if isinstance(source, (str, Path)):
        return str(source)
    return "<torch.nn.Module>"


def openvino_runtime_layer_report(
    *,
    target: ExportTargetConfig,
    export_result: OpenVINOExportResult,
    output_diff: Mapping[str, Any] | None,
    source: object = None,
) -> dict[str, Any]:
    """Summarize OpenVINO export readiness as runtime-specific layers."""

    import importlib.util

    metadata = export_result.metadata
    xml_path = Path(export_result.xml_path)
    ir_materialized = (not export_result.dry_run) and (
        export_result.checksum is not None or xml_path.is_file()
    )
    runtime_diff_requested = bool(target.openvino.runtime_diff)

    conversion_layer = {
        "status": "command_only"
        if export_result.dry_run
        else ("converted" if ir_materialized else "missing_ir"),
        "passed": True if export_result.dry_run else ir_materialized,
        "dry_run": export_result.dry_run,
        "xml_path": str(xml_path),
        "bin_path": str(export_result.bin_path)
        if export_result.bin_path is not None
        else None,
        "checksum": export_result.checksum,
        "source_path": _source_to_string(export_result.source_path)
        or _source_to_string(source),
        "input_shape": metadata.get("input_shape"),
        "command": metadata.get("command"),
    }
    if export_result.dry_run:
        runtime_load_layer = {
            "status": "skipped_dry_run",
            "passed": True,
            "requested": runtime_diff_requested,
            "device": target.openvino.device,
        }
        diff_layer = {
            "status": "skipped_dry_run",
            "passed": True,
            "requested": runtime_diff_requested,
            "diff": None,
        }
    elif not runtime_diff_requested:
        runtime_load_layer = {
            "status": "not_requested",
            "passed": True,
            "requested": False,
            "device": target.openvino.device,
        }
        diff_layer = {
            "status": "not_requested",
            "passed": True,
            "requested": False,
            "diff": None,
        }
    elif output_diff is None:
        runtime_load_layer = {
            "status": "not_executed",
            "passed": False,
            "requested": True,
            "device": target.openvino.device,
        }
        diff_layer = {
            "status": "not_executed",
            "passed": False,
            "requested": True,
            "diff": None,
        }
    else:
        diff_data = dict(output_diff)
        diff_passed = bool(diff_data.get("valid")) and bool(
            diff_data.get("allclose")
        )
        runtime_load_layer = {
            "status": "loaded",
            "passed": True,
            "requested": True,
            "device": target.openvino.device,
        }
        diff_layer = {
            "status": "passed" if diff_passed else "failed",
            "passed": diff_passed,
            "requested": True,
            "diff": diff_data,
        }

    benchmark_config = target.openvino.benchmark
    benchmark_requested = bool(benchmark_config.enabled)
    benchmark_spec = {
        "warmup": benchmark_config.warmup,
        "iterations": benchmark_config.iterations,
        "measure_memory": benchmark_config.measure_memory,
    }
    if not benchmark_requested:
        benchmark_layer = {
            "status": "not_configured",
            "passed": True,
            "requested": False,
            "benchmark": None,
            "reason": "openvino runtime benchmark config is not enabled for this target",
        }
    elif export_result.dry_run:
        benchmark_layer = {
            "status": "not_run_dry_run",
            "passed": True,
            "requested": True,
            "benchmark": benchmark_spec,
            "reason": "benchmark requires a materialized IR; dry-run did not convert",
        }
    elif not ir_materialized:
        benchmark_layer = {
            "status": "not_run_missing_ir",
            "passed": True,
            "requested": True,
            "benchmark": benchmark_spec,
            "reason": "benchmark requires a materialized OpenVINO IR",
        }
    elif importlib.util.find_spec("openvino") is None:
        benchmark_layer = {
            "status": "not_run_missing_dependency",
            "passed": True,
            "requested": True,
            "benchmark": benchmark_spec,
            "reason": (
                "openvino python package is missing; real benchmark execution "
                "is an optional milestone on the target machine"
            ),
        }
    else:
        benchmark_layer = {
            "status": "configured",
            "passed": True,
            "requested": True,
            "benchmark": benchmark_spec,
            "reason": (
                "benchmark config is wired into the report; latency measurement "
                "requires running OpenVINO runtime on the target machine"
            ),
        }
    layers = {
        "conversion": conversion_layer,
        "runtime_load": runtime_load_layer,
        "output_diff": diff_layer,
        "runtime_benchmark": benchmark_layer,
    }
    blocking_failures = [
        name
        for name, layer in layers.items()
        if layer.get("passed") is False
    ]
    if blocking_failures:
        status = "attention_required"
    elif export_result.dry_run:
        status = "command_only"
    elif diff_layer["status"] == "passed":
        status = "runtime_diffed"
    else:
        status = "converted"
    return {
        "format": "openvino",
        "status": status,
        "passed": not blocking_failures,
        "blocking_failures": blocking_failures,
        "precision": target.precision,
        "device": target.openvino.device,
        "source_path": conversion_layer["source_path"],
        "xml_path": str(xml_path),
        "bin_path": conversion_layer["bin_path"],
        "layer_order": [
            "conversion",
            "runtime_load",
            "output_diff",
            "runtime_benchmark",
        ],
        "layers": layers,
    }


def _plugin_validation_to_dict(
    validation: TensorRTPluginValidationResult | Mapping[str, Any] | None,
) -> dict[str, Any]:
    if validation is None:
        return {
            "status": "not_requested",
            "passed": True,
            "loadability_requested": False,
            "plugin_libraries": [],
            "loaded_plugin_libraries": [],
        }
    if isinstance(validation, TensorRTPluginValidationResult):
        return validation.to_dict()
    return dict(validation)


def _plugin_presence_layer(
    validation: Mapping[str, Any],
) -> dict[str, Any]:
    checks = [
        dict(item)
        for item in validation.get("plugin_libraries", [])
        if isinstance(item, Mapping)
    ]
    missing = [
        str(item.get("path"))
        for item in checks
        if item.get("exists") is False
    ]
    present_count = sum(1 for item in checks if item.get("exists") is True)
    if not checks:
        status = "not_requested"
    elif missing:
        status = "missing"
    else:
        status = "present"
    return {
        "status": status,
        "passed": not missing,
        "plugin_count": len(checks),
        "present_count": present_count,
        "missing_paths": missing,
        "checks": checks,
    }


def _plugin_loadability_layer(
    validation: Mapping[str, Any],
) -> dict[str, Any]:
    checks = [
        dict(item)
        for item in validation.get("plugin_libraries", [])
        if isinstance(item, Mapping)
    ]
    requested = bool(validation.get("loadability_requested"))
    if not checks:
        return {
            "status": "not_requested",
            "passed": True,
            "requested": requested,
            "loaded_plugin_libraries": [],
            "failed_paths": [],
        }
    if not requested:
        return {
            "status": "not_requested",
            "passed": True,
            "requested": False,
            "loaded_plugin_libraries": list(
                validation.get("loaded_plugin_libraries", [])
            ),
            "failed_paths": [],
        }
    failed = [
        str(item.get("path"))
        for item in checks
        if item.get("loadable") is False or item.get("exists") is False
    ]
    return {
        "status": "ok" if not failed else str(validation.get("status", "load_failed")),
        "passed": not failed,
        "requested": True,
        "loaded_plugin_libraries": list(
            validation.get("loaded_plugin_libraries", [])
        ),
        "failed_paths": failed,
    }


def tensorrt_runtime_layer_report(
    *,
    target: ExportTargetConfig,
    build_result: TensorRTBuildResult,
    plugin_validation: TensorRTPluginValidationResult | Mapping[str, Any] | None,
    runtime_benchmark: Mapping[str, Any] | None,
    source_onnx: str | Path | None = None,
) -> dict[str, Any]:
    """Summarize TensorRT export readiness as runtime-specific layers."""

    metadata = build_result.metadata
    plugin_validation_data = _plugin_validation_to_dict(plugin_validation)
    benchmark_config = target.tensorrt.runtime_benchmark
    engine_path = Path(build_result.engine_path)
    engine_materialized = (not build_result.dry_run) and (
        build_result.checksum is not None or engine_path.is_file()
    )
    benchmark_requested = bool(benchmark_config.enabled)

    dry_run_layer = {
        "status": "command_only" if build_result.dry_run else "not_requested",
        "passed": True,
        "requested": bool(build_result.dry_run),
        "command": list(build_result.command),
    }
    engine_build_layer = {
        "status": "skipped_dry_run"
        if build_result.dry_run
        else ("built" if engine_materialized else "missing_engine"),
        "passed": True if build_result.dry_run else engine_materialized,
        "backend": metadata.get("backend", target.tensorrt.backend),
        "returncode": build_result.returncode,
        "engine_path": str(engine_path),
        "checksum": build_result.checksum,
        "engine_inspector_available": isinstance(
            metadata.get("engine_inspector"), Mapping
        ),
        "engine_inspector_error": metadata.get("engine_inspector_error"),
        "performance_available": isinstance(metadata.get("performance"), Mapping),
        "performance_threshold_report": metadata.get(
            "performance_threshold_report"
        ),
    }
    plugin_presence = _plugin_presence_layer(plugin_validation_data)
    plugin_loadability = _plugin_loadability_layer(plugin_validation_data)
    if not benchmark_requested:
        benchmark_layer = {
            "status": "not_requested",
            "passed": True,
            "requested": False,
            "benchmark": None,
            "config": None,
        }
    elif build_result.dry_run:
        benchmark_layer = {
            "status": "skipped_dry_run",
            "passed": False,
            "requested": True,
            "benchmark": None,
            "config": {
                "input_shapes": dict(benchmark_config.input_shapes),
                "warmup": benchmark_config.warmup,
                "iterations": benchmark_config.iterations,
                "device": benchmark_config.device,
                "fill_random": benchmark_config.fill_random,
            },
        }
    elif runtime_benchmark is None:
        benchmark_layer = {
            "status": "not_executed",
            "passed": False,
            "requested": True,
            "benchmark": None,
            "config": {
                "input_shapes": dict(benchmark_config.input_shapes),
                "warmup": benchmark_config.warmup,
                "iterations": benchmark_config.iterations,
                "device": benchmark_config.device,
                "fill_random": benchmark_config.fill_random,
            },
        }
    else:
        benchmark_layer = {
            "status": "executed",
            "passed": True,
            "requested": True,
            "benchmark": dict(runtime_benchmark),
            "config": {
                "input_shapes": dict(benchmark_config.input_shapes),
                "warmup": benchmark_config.warmup,
                "iterations": benchmark_config.iterations,
                "device": benchmark_config.device,
                "fill_random": benchmark_config.fill_random,
            },
        }

    layers = {
        "dry_run": dry_run_layer,
        "engine_build": engine_build_layer,
        "plugin_presence": plugin_presence,
        "plugin_loadability": plugin_loadability,
        "runtime_benchmark": benchmark_layer,
    }
    blocking_failures = [
        name
        for name, layer in layers.items()
        if layer.get("passed") is False
    ]
    if blocking_failures:
        status = "attention_required"
    elif build_result.dry_run:
        status = "command_only"
    elif benchmark_layer["status"] == "executed":
        status = "runtime_benchmarked"
    else:
        status = "engine_built"
    return {
        "format": "tensorrt",
        "status": status,
        "passed": not blocking_failures,
        "blocking_failures": blocking_failures,
        "backend": metadata.get("backend", target.tensorrt.backend),
        "precision": target.precision,
        "source_onnx": str(source_onnx) if source_onnx is not None else None,
        "engine_path": str(engine_path),
        "layer_order": [
            "dry_run",
            "engine_build",
            "plugin_presence",
            "plugin_loadability",
            "runtime_benchmark",
        ],
        "layers": layers,
    }


__all__ = [
    "export_artifact_lineage_report",
    "export_target_capability_report",
    "openvino_runtime_layer_report",
    "tensorrt_runtime_layer_report",
]
