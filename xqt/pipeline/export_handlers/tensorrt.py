"""TensorRT export handler."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from xqt.core.artifact import ArtifactRecord, MetricRecord
from xqt.core.schema import ExportTargetConfig
from xqt.core.types import XQTContext

from .. import export_pass as _ep

from ._context import _target_summary


def handle_tensorrt(
    context: XQTContext,
    target: ExportTargetConfig,
    index: int,
    *,
    artifact_dir: Path,
) -> tuple[dict[str, object], dict[str, object]]:
    tensorrt = target.tensorrt
    onnx_path = tensorrt.onnx_path or context.artifacts.get("last_onnx")
    if onnx_path is None:
        raise ValueError(
            "TensorRT export requires tensorrt.onnx_path or a prior ONNX export"
        )
    output_path = target.output_path
    if output_path is None:
        output_path = str(artifact_dir / f"model_{index}.engine")
    result = _ep.build_tensorrt_engine(
        onnx_path,
        output_path,
        precision=target.precision,
        profiles=target.profiles,
        trtexec_path=tensorrt.trtexec_path,
        extra_args=tensorrt.extra_args,
        timeout=tensorrt.timeout,
        dry_run=tensorrt.dry_run,
        performance_thresholds=tensorrt.performance_thresholds,
        backend=tensorrt.backend,
        workspace_mib=tensorrt.workspace_mib,
        builder_optimization_level=tensorrt.builder_optimization_level,
        timing_cache_path=tensorrt.timing_cache_path,
        log_level=tensorrt.log_level,
        plugin_libraries=tensorrt.plugin_libraries,
        serialize_plugin_libraries=tensorrt.serialize_plugin_libraries,
    )
    context.artifacts[f"export_{index}"] = result.engine_path
    context.artifacts["last_engine"] = result.engine_path
    tensorrt_metadata = {
        "backend": result.metadata.get("backend", "trtexec"),
        "precision": target.precision,
        "dry_run": result.dry_run,
        "command": result.command,
        "profiles": result.metadata.get(
            "profiles", dict(target.profiles or {})
        ),
        "source_onnx": str(onnx_path),
        "profiling_verbosity": result.metadata.get("profiling_verbosity"),
        "builder_flags": result.metadata.get("builder_flags"),
        "builder_notes": result.metadata.get("builder_notes"),
        "plugin_libraries": result.metadata.get("plugin_libraries"),
        "loaded_plugin_libraries": result.metadata.get(
            "loaded_plugin_libraries"
        ),
        "serialize_plugin_libraries": result.metadata.get(
            "serialize_plugin_libraries"
        ),
        "engine_inspector": result.metadata.get("engine_inspector"),
        "engine_inspector_error": result.metadata.get(
            "engine_inspector_error"
        ),
        "performance": result.metadata.get("performance"),
        "performance_threshold_report": result.metadata.get(
            "performance_threshold_report"
        ),
    }
    runtime_benchmark = None
    runtime_benchmark_config = tensorrt.runtime_benchmark
    if not result.dry_run and runtime_benchmark_config.enabled:
        runtime_benchmark = _ep.benchmark_tensorrt_engine(
            result.engine_path,
            input_shapes=runtime_benchmark_config.input_shapes,
            warmup=runtime_benchmark_config.warmup,
            iterations=runtime_benchmark_config.iterations,
            device=runtime_benchmark_config.device,
            fill_random=runtime_benchmark_config.fill_random,
            plugin_libraries=tensorrt.plugin_libraries,
        ).to_dict()
        tensorrt_metadata["runtime_benchmark"] = runtime_benchmark
    if context.manifest is not None:
        context.manifest.add_artifact(
            ArtifactRecord(
                path=str(result.engine_path),
                format="tensorrt",
                runtime="tensorrt",
                checksum=result.checksum,
                metadata=tensorrt_metadata,
            )
        )
        threshold_report = result.metadata.get(
            "performance_threshold_report"
        )
        if isinstance(threshold_report, Mapping):
            context.manifest.add_metric(
                MetricRecord(
                    name=f"export.{index}.tensorrt.performance",
                    value=result.metadata.get("performance"),
                    threshold=result.metadata.get("performance_thresholds"),
                    passed=bool(threshold_report.get("passed")),
                    metadata={"checks": threshold_report.get("checks", [])},
                )
            )
    exported_entry: dict[str, object] = {
        "path": str(result.engine_path),
        "format": "tensorrt",
        "dry_run": result.dry_run,
        "artifact_status": "command_only"
        if result.dry_run
        else "materialized",
        "backend_execution": "dry_run"
        if result.dry_run
        else "executed",
        "command": result.command,
        "checksum": result.checksum,
        "precision": target.precision,
        "profiles": result.metadata.get(
            "profiles", dict(target.profiles or {})
        ),
        "source_onnx": str(onnx_path),
        "profiling_verbosity": result.metadata.get("profiling_verbosity"),
        "builder_flags": result.metadata.get("builder_flags"),
        "builder_notes": result.metadata.get("builder_notes"),
        "engine_inspector": result.metadata.get("engine_inspector"),
        "engine_inspector_error": result.metadata.get(
            "engine_inspector_error"
        ),
        "performance": result.metadata.get("performance"),
        "runtime_benchmark": runtime_benchmark,
        "performance_threshold_report": result.metadata.get(
            "performance_threshold_report"
        ),
    }
    summary = _target_summary(target, exported_entry)
    return exported_entry, summary
