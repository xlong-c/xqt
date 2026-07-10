"""Export pass implementation for the legacy XQT pipeline."""

from __future__ import annotations

import copy
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping, cast

from omegaconf import OmegaConf
import torch
from torch import nn

from xqt.core.artifact import ArtifactRecord, MetricRecord
from xqt.core.inputs import extract_model_inputs, infer_model_input_count
from xqt.core.registry import register_pass
from xqt.core.schema import ExportTargetConfig, OutputDiffConfig
from xqt.core.types import XQTContext
from xqt.workflows.stage_specs import (
    DeployStageSpec,
    ExportStageSpec,
    stage_spec_to_params,
)
from xqt.export import (
    benchmark_tensorrt_engine,
    build_tensorrt_engine,
    compare_openvino_outputs,
    compare_onnxruntime_outputs,
    create_onnxruntime_session,
    create_tensorrt_runtime_session,
    export_executorch_program,
    export_mnn_from_onnx,
    export_ncnn_from_onnx,
    export_onnx,
    export_openvino_ir,
    optimize_onnx,
    export_torch_program,
    export_torchscript,
)
from xqt.export.input_utils import default_input_names, first_tensor_output


def call_model(model: nn.Module, inputs: Any) -> Any:
    """Call a module with mapping, tuple, or positional inputs."""

    if isinstance(inputs, Mapping):
        return model(**inputs)
    if isinstance(inputs, tuple):
        return model(*inputs)
    return model(inputs)


def resolve_export_model(context: XQTContext) -> tuple[nn.Module, dict[str, object]]:
    """Choose an export-friendly model when runtime optimization wrapped the current module."""

    model = context.require_model()
    operator_metrics = context.metrics.get("operator_optimization")
    if not isinstance(operator_metrics, dict):
        return model, {"guarded": False}

    targets = operator_metrics.get("targets")
    if not isinstance(targets, list) or not any(
        isinstance(item, Mapping) and bool(item.get("applied")) for item in targets
    ):
        return model, {"guarded": False}

    if hasattr(model, "_orig_mod") and isinstance(getattr(model, "_orig_mod"), nn.Module):
        return getattr(model, "_orig_mod"), {
            "guarded": True,
            "reason": "compiled_runtime_unwrapped",
        }
    if isinstance(context.reference_model, nn.Module):
        return context.reference_model, {
            "guarded": True,
            "reason": "reference_model_fallback",
        }
    return model, {"guarded": False}


def update_structured_prune_export_status(
    context: XQTContext,
    exported: list[dict[str, object]],
) -> None:
    """Record whether structured pruning survived export checks."""

    prune_metrics = context.metrics.get("prune")
    if not isinstance(prune_metrics, dict) or prune_metrics.get("method") != "structured":
        return
    checked_values = [
        bool(item.get("checked"))
        for item in exported
        if isinstance(item.get("checked"), bool)
    ]
    prune_metrics["export_status"] = {
        "attempted": bool(exported),
        "passed": all(checked_values) if checked_values else (True if exported else None),
        "artifact_count": len(exported),
        "formats": [str(item.get("format")) for item in exported if item.get("format") is not None],
        "artifacts": [dict(item) for item in exported],
    }


def resolve_onnx_optimization_config(params: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize ONNX optimization settings from export target params."""

    raw = params.get("onnx_optimization", params.get("optimize", False))
    if raw is True:
        return {"enabled": True}
    if raw is False or raw is None:
        return {"enabled": False}
    if not isinstance(raw, Mapping):
        raise ValueError("onnx_optimization must be a mapping or boolean")
    normalized = dict(raw)
    normalized.setdefault("enabled", True)
    return normalized


def _target_summary(
    target: ExportTargetConfig,
    artifact: Mapping[str, object],
) -> dict[str, object]:
    return {
        "format": target.format,
        "output_path": target.output_path,
        "precision": target.precision,
        "opset": target.opset,
        "dynamic_shapes": dict(target.dynamic_shapes),
        "profiles": dict(target.profiles),
        "params": dict(target.params),
        **dict(artifact),
    }


def _output_diff_runtime_config(spec: OutputDiffConfig | None) -> OutputDiffConfig | None:
    if spec is None:
        return None
    try:
        merged = OmegaConf.merge(
            OmegaConf.structured(OutputDiffConfig),
            OmegaConf.create(asdict(spec) if is_dataclass(spec) else dict(spec)),
        )
        return cast(OutputDiffConfig, OmegaConf.to_object(merged))
    except Exception as exc:
        raise ValueError(f"failed to load output diff config: {exc}") from exc


def _runtime_handle_params(request: Mapping[str, Any]) -> Mapping[str, Any]:
    params = request.get("params", {})
    if not isinstance(params, Mapping):
        raise ValueError("deploy runtime_handle.params must be a mapping")
    return params


def _runtime_handle_plugin_libraries(
    params: Mapping[str, Any],
    targets: list[dict[str, object]],
) -> list[str] | None:
    plugins = params.get("plugin_libraries")
    if plugins is None:
        for target in targets:
            if target.get("format") != "tensorrt":
                continue
            target_params = target.get("params")
            if isinstance(target_params, Mapping):
                plugins = target_params.get("plugin_libraries")
                break
    if plugins is None:
        return None
    if not isinstance(plugins, list) or not all(isinstance(item, str) for item in plugins):
        raise ValueError("TensorRT runtime_handle plugin_libraries must be a list of paths")
    return list(plugins)


def materialize_deploy_runtime_handle(
    context: XQTContext,
    request: Mapping[str, Any],
    target_summaries: list[dict[str, object]],
) -> dict[str, object]:
    """Materialize and describe one verified deploy runtime handle."""

    runtime = str(request.get("runtime") or "onnxruntime")
    params = _runtime_handle_params(request)
    handle_kind = str(request.get("handle_kind", "inference_session"))
    if runtime == "onnxruntime":
        onnx_path = context.artifacts.get("last_onnx")
        if onnx_path is None:
            raise ValueError(
                "ONNX Runtime handle requires an ONNX export target in the same deploy stage"
            )
        providers = params.get("providers")
        if providers is not None and not isinstance(providers, list):
            raise ValueError("deploy runtime_handle.params.providers must be a list")
        session = create_onnxruntime_session(
            onnx_path,
            providers=[str(provider) for provider in providers] if providers is not None else None,
        )
        return {
            "runtime": runtime,
            "handle_kind": handle_kind,
            "handle": session,
            "target_count": 1,
            "targets": [
                item for item in target_summaries if item.get("format") == "onnx"
            ],
            "artifacts": {"onnx": str(onnx_path)},
            "metadata": {
                "providers": list(session.get_providers()),
                "model_path": str(onnx_path),
                "runtime_validation": {"status": "session_created"},
            },
        }
    if runtime != "tensorrt":
        raise ValueError(
            "deploy runtime-handle materialization supports runtime=onnxruntime or runtime=tensorrt"
        )

    engine_path = context.artifacts.get("last_engine")
    if engine_path is None:
        raise ValueError(
            "TensorRT runtime handle requires a non-dry-run TensorRT export target in the same deploy stage"
        )
    tensorrt_targets = [
        item for item in target_summaries if item.get("format") == "tensorrt"
    ]
    if not tensorrt_targets:
        raise ValueError(
            "TensorRT runtime handle requires a TensorRT export target in the same deploy stage"
        )
    if any(bool(item.get("dry_run")) for item in tensorrt_targets):
        raise ValueError("TensorRT runtime handle cannot materialize a dry-run engine artifact")
    device = str(params.get("device") or context.device or "cuda:0")
    plugin_libraries = _runtime_handle_plugin_libraries(params, tensorrt_targets)
    session = create_tensorrt_runtime_session(
        engine_path,
        device=device,
        plugin_libraries=plugin_libraries,
    )
    return {
        "runtime": runtime,
        "handle_kind": handle_kind,
        "handle": session,
        "target_count": len(tensorrt_targets),
        "targets": tensorrt_targets,
        "artifacts": {"tensorrt_engine": str(engine_path)},
        "metadata": {
            "engine_path": str(session.engine_path),
            "device": session.device,
            "plugin_libraries": plugin_libraries or [],
            "engine_inspector": dict(session.engine_inspector),
            "runtime_validation": {
                "status": "session_created",
                "engine_deserialized": True,
                "execution_context_created": True,
            },
        },
    }


@register_pass("export")
class ExportPass:
    """Export configured artifacts."""

    name = "export"

    def run(
        self,
        context: XQTContext,
        *,
        targets: list[ExportTargetConfig] | None = None,
        output_diff: OutputDiffConfig | None = None,
        stage_kind: str = "export",
        runtime_handle_request: Mapping[str, Any] | None = None,
    ) -> XQTContext:
        resolved_targets = targets if targets is not None else context.export_targets
        if resolved_targets is None:
            raise ValueError("XQTContext.export_targets is required")
        export_targets = list(resolved_targets)
        context.export_targets = copy.deepcopy(export_targets)
        if not export_targets:
            return context
        resolved_output_diff = output_diff or context.output_diff_config
        if resolved_output_diff is None:
            raise ValueError("XQTContext.output_diff_config is required")
        artifact_dir = Path(context.artifact_dir)
        needs_model = False
        for target in export_targets:
            if target.format in {"torch_export", "torchscript", "onnx"}:
                needs_model = True
                break
            if target.format == "openvino" and target.params.get("onnx_path") is None:
                needs_model = True
                break

        export_model = None
        export_guard: dict[str, object] = {"guarded": False}
        example_input = None
        reference_output = None
        if needs_model:
            model = context.require_model()
            del model
            export_model, export_guard = resolve_export_model(context)
            if context.example_inputs is None:
                raise ValueError("example_inputs are required for model export")
            batch = context.example_inputs
            example_input = extract_model_inputs(
                batch,
                expected_input_count=infer_model_input_count(export_model),
            )
            with torch.no_grad():
                reference_output = first_tensor_output(call_model(export_model, example_input))

        exported: list[dict[str, object]] = []
        target_summaries: list[dict[str, object]] = []

        for index, target in enumerate(export_targets):
            if target.format == "torch_export":
                if export_model is None or example_input is None:
                    raise ValueError("torch_export requires a loaded model and example_inputs")
                output_path = target.output_path
                if output_path is None:
                    output_path = str(artifact_dir / f"model_{index}.pt2")
                result = export_torch_program(
                    export_model,
                    example_input,
                    output_path,
                    dynamic_shapes=target.dynamic_shapes,
                    strict=bool(target.params.get("strict", False)),
                    validate=bool(target.params.get("validate", True)),
                    compare_output=bool(target.params.get("runtime_diff", True)),
                    atol=resolved_output_diff.atol,
                    rtol=resolved_output_diff.rtol,
                )
                record = ArtifactRecord.from_file(
                    result.path,
                    format="torch_export",
                    runtime="pytorch",
                    metadata={
                        "checked": result.checked,
                        "output_diff": (
                            result.output_diff.to_dict()
                            if result.output_diff is not None
                            else None
                        ),
                        "export_guard": dict(export_guard),
                        **result.metadata,
                    },
                )
                context.artifacts[f"export_{index}"] = result.path
                if context.manifest is not None:
                    context.manifest.add_artifact(record)
                exported.append(
                    {
                        "path": str(result.path),
                        "format": "torch_export",
                        "checked": result.checked,
                        "checksum": result.checksum,
                        "output_diff": (
                            result.output_diff.to_dict()
                            if result.output_diff is not None
                            else None
                        ),
                        "export_guard": dict(export_guard),
                    }
                )
                target_summaries.append(_target_summary(target, exported[-1]))
                continue

            if target.format == "torchscript":
                if export_model is None or example_input is None:
                    raise ValueError("torchscript export requires a loaded model and example_inputs")
                output_path = target.output_path
                if output_path is None:
                    output_path = str(artifact_dir / f"model_{index}.pt")
                result = export_torchscript(
                    export_model,
                    example_input,
                    output_path,
                    method=str(target.params.get("method", "trace")),
                    check_trace=bool(target.params.get("check_trace", True)),
                    compare_output=bool(target.params.get("runtime_diff", True)),
                    atol=resolved_output_diff.atol,
                    rtol=resolved_output_diff.rtol,
                )
                record = ArtifactRecord.from_file(
                    result.path,
                    format="torchscript",
                    runtime="pytorch",
                    metadata={
                        "output_diff": (
                            result.output_diff.to_dict()
                            if result.output_diff is not None
                            else None
                        ),
                        "export_guard": dict(export_guard),
                        **result.metadata,
                    },
                )
                context.artifacts[f"export_{index}"] = result.path
                context.artifacts.setdefault("last_torchscript", result.path)
                if context.manifest is not None:
                    context.manifest.add_artifact(record)
                exported.append(
                    {
                        "path": str(result.path),
                        "format": "torchscript",
                        "checksum": result.checksum,
                        "output_diff": (
                            result.output_diff.to_dict()
                            if result.output_diff is not None
                            else None
                        ),
                        "export_guard": dict(export_guard),
                    }
                )
                target_summaries.append(_target_summary(target, exported[-1]))
                continue

            if target.format == "onnx":
                if export_model is None or example_input is None:
                    raise ValueError("onnx export requires a loaded model and example_inputs")
                output_path = target.output_path
                if output_path is None:
                    output_path = str(artifact_dir / f"model_{index}.onnx")
                result = export_onnx(
                    export_model,
                    example_input,
                    output_path,
                    opset=target.opset,
                    dynamic_shapes=target.dynamic_shapes,
                    input_names=target.params.get("input_names")
                    or default_input_names(example_input),
                    output_names=target.params.get("output_names"),
                    dynamo=bool(target.params.get("dynamo", True)),
                    validate=bool(target.params.get("validate", True)),
                    pre_export_fusion=target.params.get("pre_export_fusion"),
                )
                optimization = resolve_onnx_optimization_config(target.params)
                optimized_result = None
                if bool(optimization.get("enabled", False)):
                    optimized_result = optimize_onnx(
                        result.path,
                        optimization.get("output_path"),
                        backend=str(optimization.get("backend", "onnxruntime")),
                        level=str(optimization.get("level", "extended")),
                        output_suffix=str(
                            optimization.get("output_suffix", ".optimized")
                        ),
                        validate=bool(optimization.get("validate", True)),
                        providers=list(
                            optimization.get("providers", ["CPUExecutionProvider"])
                        ),
                        native_qdq=optimization.get("native_qdq", True),
                        metadata={
                            "source_export_path": str(result.path),
                        },
                    )
                    result.metadata["onnx_optimization"] = optimized_result.to_dict()
                    result.path = optimized_result.path
                    result.checksum = optimized_result.checksum
                    result.checked = optimized_result.checked
                diff = None
                if bool(target.params.get("runtime_diff", True)):
                    if reference_output is None or example_input is None:
                        raise ValueError("onnx runtime_diff requires model reference output")
                    diff = compare_onnxruntime_outputs(
                        result.path,
                        reference_output,
                        example_input,
                        input_names=result.metadata.get("input_names"),
                        atol=resolved_output_diff.atol,
                        rtol=resolved_output_diff.rtol,
                    )
                    result.output_diff = diff
                result_metadata = getattr(result, "metadata", {})
                record = ArtifactRecord.from_file(
                    result.path,
                    format="onnx",
                    runtime="onnxruntime" if diff is not None else None,
                    metadata={
                        "opset": result.opset,
                        "checked": result.checked,
                        "output_diff": diff.to_dict() if diff is not None else None,
                        "export_guard": dict(export_guard),
                        **result_metadata,
                    },
                )
                context.artifacts[f"export_{index}"] = result.path
                context.artifacts["last_onnx"] = result.path
                if optimized_result is not None:
                    context.artifacts[f"export_{index}_source_onnx"] = (
                        optimized_result.source_path
                    )
                    context.artifacts["last_onnx_source"] = optimized_result.source_path
                if context.manifest is not None:
                    context.manifest.add_artifact(record)
                exported.append(
                    {
                        "path": str(result.path),
                        "format": "onnx",
                        "checked": result.checked,
                        "checksum": result.checksum,
                        "output_diff": diff.to_dict() if diff is not None else None,
                        "pre_export_fusion": result_metadata.get("pre_export_fusion"),
                        "onnx_optimization": result_metadata.get("onnx_optimization"),
                        "export_guard": dict(export_guard),
                    }
                )
                target_summaries.append(_target_summary(target, exported[-1]))
                continue

            if target.format == "tensorrt":
                onnx_path = target.params.get("onnx_path") or context.artifacts.get(
                    "last_onnx"
                )
                if onnx_path is None:
                    raise ValueError("TensorRT export requires params.onnx_path or a prior ONNX export")
                output_path = target.output_path
                if output_path is None:
                    output_path = str(artifact_dir / f"model_{index}.engine")
                result = build_tensorrt_engine(
                    onnx_path,
                    output_path,
                    precision=target.precision,
                    profiles=target.profiles,
                    trtexec_path=str(target.params.get("trtexec_path", "trtexec")),
                    extra_args=target.params.get("extra_args"),
                    timeout=target.params.get("timeout"),
                    dry_run=bool(target.params.get("dry_run", False)),
                    performance_thresholds=target.params.get("performance_thresholds"),
                    backend=str(target.params.get("backend", "trtexec")),
                    workspace_mib=int(target.params.get("workspace_mib", 4096)),
                    builder_optimization_level=(
                        int(target.params["builder_optimization_level"])
                        if target.params.get("builder_optimization_level") is not None
                        else None
                    ),
                    timing_cache_path=target.params.get("timing_cache_path"),
                    log_level=target.params.get("log_level"),
                    plugin_libraries=target.params.get("plugin_libraries"),
                    serialize_plugin_libraries=bool(
                        target.params.get("serialize_plugin_libraries", True)
                    ),
                )
                context.artifacts[f"export_{index}"] = result.engine_path
                context.artifacts["last_engine"] = result.engine_path
                tensorrt_metadata = {
                    "backend": result.metadata.get("backend", "trtexec"),
                    "precision": target.precision,
                    "dry_run": result.dry_run,
                    "command": result.command,
                    "profiles": result.metadata.get("profiles", dict(target.profiles or {})),
                    "source_onnx": str(onnx_path),
                    "profiling_verbosity": result.metadata.get("profiling_verbosity"),
                    "builder_flags": result.metadata.get("builder_flags"),
                    "builder_notes": result.metadata.get("builder_notes"),
                    "plugin_libraries": result.metadata.get("plugin_libraries"),
                    "loaded_plugin_libraries": result.metadata.get("loaded_plugin_libraries"),
                    "serialize_plugin_libraries": result.metadata.get(
                        "serialize_plugin_libraries"
                    ),
                    "engine_inspector": result.metadata.get("engine_inspector"),
                    "engine_inspector_error": result.metadata.get("engine_inspector_error"),
                    "performance": result.metadata.get("performance"),
                    "performance_threshold_report": result.metadata.get(
                        "performance_threshold_report"
                    ),
                }
                runtime_benchmark = None
                runtime_benchmark_params = target.params.get("runtime_benchmark")
                if (
                    not result.dry_run
                    and isinstance(runtime_benchmark_params, Mapping)
                    and bool(runtime_benchmark_params.get("enabled", False))
                ):
                    input_shapes = runtime_benchmark_params.get("input_shapes")
                    if not isinstance(input_shapes, Mapping):
                        raise ValueError(
                            "TensorRT runtime_benchmark.enabled=true requires params.runtime_benchmark.input_shapes"
                        )
                    runtime_benchmark = benchmark_tensorrt_engine(
                        result.engine_path,
                        input_shapes=input_shapes,
                        warmup=int(runtime_benchmark_params.get("warmup", 10)),
                        iterations=int(runtime_benchmark_params.get("iterations", 50)),
                        device=str(runtime_benchmark_params.get("device", "cuda:0")),
                        fill_random=bool(runtime_benchmark_params.get("fill_random", True)),
                        plugin_libraries=target.params.get("plugin_libraries"),
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
                    threshold_report = result.metadata.get("performance_threshold_report")
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
                exported.append(
                    {
                        "path": str(result.engine_path),
                        "format": "tensorrt",
                        "dry_run": result.dry_run,
                        "artifact_status": "command_only" if result.dry_run else "materialized",
                        "backend_execution": "dry_run" if result.dry_run else "executed",
                        "command": result.command,
                        "checksum": result.checksum,
                        "precision": target.precision,
                        "profiles": result.metadata.get("profiles", dict(target.profiles or {})),
                        "source_onnx": str(onnx_path),
                        "profiling_verbosity": result.metadata.get("profiling_verbosity"),
                        "builder_flags": result.metadata.get("builder_flags"),
                        "builder_notes": result.metadata.get("builder_notes"),
                        "engine_inspector": result.metadata.get("engine_inspector"),
                        "engine_inspector_error": result.metadata.get("engine_inspector_error"),
                        "performance": result.metadata.get("performance"),
                        "runtime_benchmark": runtime_benchmark,
                        "performance_threshold_report": result.metadata.get(
                            "performance_threshold_report"
                        ),
                    }
                )
                target_summaries.append(_target_summary(target, exported[-1]))
                continue

            if target.format == "openvino":
                source = target.params.get("onnx_path") or context.artifacts.get(
                    "last_onnx"
                )
                if source is None:
                    if export_model is None:
                        raise ValueError(
                            "OpenVINO export requires params.onnx_path or a loaded model"
                        )
                    source = export_model
                output_path = target.output_path
                if output_path is None:
                    output_path = str(artifact_dir / f"model_{index}.xml")
                result = export_openvino_ir(
                    source,
                    output_path,
                    example_input=example_input if isinstance(source, nn.Module) else None,
                    input_shape=target.params.get("input_shape"),
                    dry_run=bool(target.params.get("dry_run", False)),
                )
                diff = None
                if (
                    not result.dry_run
                    and bool(target.params.get("runtime_diff", True))
                    and result.xml_path.is_file()
                ):
                    if reference_output is None or example_input is None:
                        raise ValueError(
                            "OpenVINO runtime_diff requires a loaded model and example_inputs"
                        )
                    diff = compare_openvino_outputs(
                        result.xml_path,
                        reference_output,
                        example_input,
                        device=str(target.params.get("device", "CPU")),
                        atol=resolved_output_diff.atol,
                        rtol=resolved_output_diff.rtol,
                    )
                    result.output_diff = diff
                metadata = {
                    "precision": target.precision,
                    "dry_run": result.dry_run,
                    "source_path": (
                        str(result.source_path)
                        if result.source_path is not None
                        else None
                    ),
                    "input_shape": result.metadata.get("input_shape"),
                    "output_diff": diff.to_dict() if diff is not None else None,
                    **result.metadata,
                }
                context.artifacts[f"export_{index}"] = result.xml_path
                if context.manifest is not None:
                    context.manifest.add_artifact(
                        ArtifactRecord(
                            path=str(result.xml_path),
                            format="openvino",
                            runtime="openvino",
                            checksum=result.checksum,
                            metadata=metadata,
                        )
                    )
                exported.append(
                    {
                        "path": str(result.xml_path),
                        "bin_path": (
                            str(result.bin_path)
                            if result.bin_path is not None
                            else None
                        ),
                        "format": "openvino",
                        "dry_run": result.dry_run,
                        "artifact_status": "command_only" if result.dry_run else "materialized",
                        "backend_execution": "dry_run" if result.dry_run else "executed",
                        "checksum": result.checksum,
                        "precision": target.precision,
                        "source_path": metadata.get("source_path"),
                        "input_shape": metadata.get("input_shape"),
                        "output_diff": diff.to_dict() if diff is not None else None,
                        "command": metadata.get("command"),
                    }
                )
                target_summaries.append(_target_summary(target, exported[-1]))
                continue

            if target.format == "executorch":
                if export_model is None or example_input is None:
                    raise ValueError("executorch export requires a loaded model and example_inputs")
                output_path = target.output_path
                if output_path is None:
                    output_path = str(artifact_dir / f"model_{index}.pte")
                result = export_executorch_program(
                    export_model,
                    example_input,
                    output_path,
                    dry_run=bool(target.params.get("dry_run", False)),
                    metadata={"precision": target.precision},
                )
                context.artifacts[f"export_{index}"] = result.pte_path
                if context.manifest is not None and result.checksum is not None:
                    context.manifest.add_artifact(
                        ArtifactRecord(
                            path=str(result.pte_path),
                            format="executorch",
                            runtime="executorch",
                            checksum=result.checksum,
                            metadata=result.metadata | {"dry_run": result.dry_run},
                        )
                    )
                exported.append(
                    {
                        "path": str(result.pte_path),
                        "format": "executorch",
                        "dry_run": result.dry_run,
                        "artifact_status": "command_only" if result.dry_run else "materialized",
                        "backend_execution": "dry_run" if result.dry_run else "executed",
                        "checksum": result.checksum,
                    }
                )
                target_summaries.append(_target_summary(target, exported[-1]))
                continue

            if target.format == "ncnn":
                onnx_path = target.params.get("onnx_path") or context.artifacts.get(
                    "last_onnx"
                )
                if onnx_path is None:
                    raise ValueError("ncnn export requires params.onnx_path or a prior ONNX export")
                param_path = target.output_path
                if param_path is None:
                    param_path = str(artifact_dir / f"model_{index}.param")
                bin_path = target.params.get("bin_path")
                if bin_path is None:
                    bin_path = str(Path(param_path).with_suffix(".bin"))
                result = export_ncnn_from_onnx(
                    onnx_path,
                    param_path,
                    bin_path,
                    onnx2ncnn_path=str(target.params.get("onnx2ncnn_path", "onnx2ncnn")),
                    extra_args=target.params.get("extra_args"),
                    timeout=target.params.get("timeout"),
                    dry_run=bool(target.params.get("dry_run", False)),
                )
                context.artifacts[f"export_{index}"] = result.output_paths
                if context.manifest is not None and result.checksums:
                    for output in result.output_paths:
                        context.manifest.add_artifact(
                            ArtifactRecord(
                                path=str(output),
                                format="ncnn",
                                runtime="ncnn",
                                checksum=result.checksums.get(str(output)),
                                metadata={
                                    "dry_run": result.dry_run,
                                    "command": result.command,
                                },
                            )
                        )
                exported.append(
                    {
                        "paths": [str(path) for path in result.output_paths],
                        "format": "ncnn",
                        "dry_run": result.dry_run,
                        "artifact_status": "command_only" if result.dry_run else "materialized",
                        "backend_execution": "dry_run" if result.dry_run else "executed",
                        "command": result.command,
                        "checksums": result.checksums,
                    }
                )
                target_summaries.append(_target_summary(target, exported[-1]))
                continue

            if target.format == "mnn":
                onnx_path = target.params.get("onnx_path") or context.artifacts.get(
                    "last_onnx"
                )
                if onnx_path is None:
                    raise ValueError("MNN export requires params.onnx_path or a prior ONNX export")
                output_path = target.output_path
                if output_path is None:
                    output_path = str(artifact_dir / f"model_{index}.mnn")
                result = export_mnn_from_onnx(
                    onnx_path,
                    output_path,
                    converter_path=str(target.params.get("converter_path", "MNNConvert")),
                    framework=str(target.params.get("framework", "ONNX")),
                    extra_args=target.params.get("extra_args"),
                    timeout=target.params.get("timeout"),
                    dry_run=bool(target.params.get("dry_run", False)),
                )
                context.artifacts[f"export_{index}"] = result.output_paths[0]
                if context.manifest is not None and result.checksums:
                    output = result.output_paths[0]
                    context.manifest.add_artifact(
                        ArtifactRecord(
                            path=str(output),
                            format="mnn",
                            runtime="mnn",
                            checksum=result.checksums.get(str(output)),
                            metadata={
                                "dry_run": result.dry_run,
                                "command": result.command,
                            },
                        )
                    )
                exported.append(
                    {
                        "path": str(result.output_paths[0]),
                        "format": "mnn",
                        "dry_run": result.dry_run,
                        "artifact_status": "command_only" if result.dry_run else "materialized",
                        "backend_execution": "dry_run" if result.dry_run else "executed",
                        "command": result.command,
                        "checksums": result.checksums,
                    }
                )
                target_summaries.append(_target_summary(target, exported[-1]))
                continue

            raise ValueError(f"Unsupported export format: {target.format}")
        metrics: dict[str, object] = {
            "artifacts": exported,
            "target_count": len(export_targets),
            "targets": target_summaries,
            "stage_kind": stage_kind,
        }
        if runtime_handle_request is not None:
            metrics["runtime_handle_request"] = dict(runtime_handle_request)
            if bool(runtime_handle_request.get("materialize", False)):
                metrics["runtime_handle"] = materialize_deploy_runtime_handle(
                    context,
                    runtime_handle_request,
                    target_summaries,
                )
        context.metrics["export"] = metrics
        update_structured_prune_export_status(context, exported)
        return context


def run_export_stage(
    context: XQTContext,
    spec: ExportStageSpec | DeployStageSpec,
    *,
    stage_kind: str = "export",
) -> XQTContext:
    """Run an export or deploy stage from a typed stage spec."""

    output_diff = _output_diff_runtime_config(spec.validate)
    runtime_handle_request: Mapping[str, Any] | None = None
    if isinstance(spec, DeployStageSpec) and spec.runtime_handle is not None:
        runtime_handle_request = stage_spec_to_params(spec.runtime_handle)
    if output_diff is not None:
        context.output_diff_config = copy.deepcopy(output_diff)
    context.export_targets = copy.deepcopy(list(spec.targets))
    return ExportPass().run(
        context,
        targets=list(spec.targets),
        output_diff=output_diff,
        stage_kind=stage_kind,
        runtime_handle_request=runtime_handle_request,
    )


__all__ = [
    "ExportPass",
    "call_model",
    "resolve_export_model",
    "resolve_onnx_optimization_config",
    "materialize_deploy_runtime_handle",
    "run_export_stage",
    "update_structured_prune_export_status",
]
