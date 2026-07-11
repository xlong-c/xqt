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
    DeployRuntimeHandleSpec,
    DeployStageSpec,
    ExportStageSpec,
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
    export_ncnn_with_pnnx,
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

    if hasattr(model, "_orig_mod") and isinstance(
        getattr(model, "_orig_mod"), nn.Module
    ):
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
    if (
        not isinstance(prune_metrics, dict)
        or prune_metrics.get("method") != "structured"
    ):
        return
    checked_values = [
        bool(item.get("checked"))
        for item in exported
        if isinstance(item.get("checked"), bool)
    ]
    prune_metrics["export_status"] = {
        "attempted": bool(exported),
        "passed": all(checked_values)
        if checked_values
        else (True if exported else None),
        "artifact_count": len(exported),
        "formats": [
            str(item.get("format"))
            for item in exported
            if item.get("format") is not None
        ],
        "artifacts": [dict(item) for item in exported],
    }


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
        "onnx": asdict(target.onnx),
        "openvino": asdict(target.openvino),
        "tensorrt": asdict(target.tensorrt),
        "torch_export": asdict(target.torch_export),
        "torchscript": asdict(target.torchscript),
        "executorch": asdict(target.executorch),
        "ncnn": asdict(target.ncnn),
        "mnn": asdict(target.mnn),
        "params": dict(target.params),
        **dict(artifact),
    }


def _output_diff_runtime_config(
    spec: OutputDiffConfig | None,
) -> OutputDiffConfig | None:
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


def materialize_deploy_runtime_handle(
    context: XQTContext,
    request: DeployRuntimeHandleSpec,
    target_summaries: list[dict[str, object]],
) -> dict[str, object]:
    """Materialize and describe one verified deploy runtime handle."""

    runtime = request.runtime or "onnxruntime"
    handle_kind = request.handle_kind
    if runtime == "onnxruntime":
        onnx_path = context.artifacts.get("last_onnx")
        if onnx_path is None:
            raise ValueError(
                "ONNX Runtime handle requires an ONNX export target in the same deploy stage"
            )
        providers = request.onnxruntime.providers
        session = create_onnxruntime_session(
            onnx_path,
            providers=list(providers) or None,
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
        raise ValueError(
            "TensorRT runtime handle cannot materialize a dry-run engine artifact"
        )
    device = request.tensorrt.device or context.device or "cuda:0"
    plugin_libraries = list(request.tensorrt.plugin_libraries) or None
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
        runtime_handle_request: DeployRuntimeHandleSpec | None = None,
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
            if target.format in {"torch_export", "torchscript", "onnx", "executorch"}:
                needs_model = True
                break
            if target.format == "openvino" and target.openvino.onnx_path is None:
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
                reference_output = first_tensor_output(
                    call_model(export_model, example_input)
                )

        exported: list[dict[str, object]] = []
        target_summaries: list[dict[str, object]] = []

        for index, target in enumerate(export_targets):
            if target.format == "torch_export":
                if export_model is None or example_input is None:
                    raise ValueError(
                        "torch_export requires a loaded model and example_inputs"
                    )
                output_path = target.output_path
                if output_path is None:
                    output_path = str(artifact_dir / f"model_{index}.pt2")
                torch_export = target.torch_export
                result = export_torch_program(
                    export_model,
                    example_input,
                    output_path,
                    dynamic_shapes=target.dynamic_shapes,
                    strict=torch_export.strict,
                    validate=torch_export.validate,
                    compare_output=torch_export.runtime_diff,
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
                    raise ValueError(
                        "torchscript export requires a loaded model and example_inputs"
                    )
                output_path = target.output_path
                if output_path is None:
                    output_path = str(artifact_dir / f"model_{index}.pt")
                torchscript = target.torchscript
                result = export_torchscript(
                    export_model,
                    example_input,
                    output_path,
                    method=torchscript.method,
                    check_trace=torchscript.check_trace,
                    compare_output=torchscript.runtime_diff,
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
                    raise ValueError(
                        "onnx export requires a loaded model and example_inputs"
                    )
                output_path = target.output_path
                if output_path is None:
                    output_path = str(artifact_dir / f"model_{index}.onnx")
                onnx = target.onnx
                result = export_onnx(
                    export_model,
                    example_input,
                    output_path,
                    opset=target.opset,
                    dynamic_shapes=target.dynamic_shapes,
                    input_names=onnx.input_names or default_input_names(example_input),
                    output_names=onnx.output_names,
                    dynamo=onnx.dynamo,
                    validate=onnx.validate,
                    pre_export_fusion=asdict(onnx.pre_export_fusion),
                    pre_export_lowering=asdict(onnx.pre_export_lowering),
                )
                optimization = onnx.optimization
                optimized_result = None
                if optimization.enabled:
                    optimized_result = optimize_onnx(
                        result.path,
                        optimization.output_path,
                        backend=optimization.backend,
                        level=optimization.level,
                        output_suffix=optimization.output_suffix,
                        validate=optimization.validate,
                        providers=list(optimization.providers),
                        native_qdq=optimization.native_qdq,
                        metadata={
                            "source_export_path": str(result.path),
                        },
                    )
                    result.metadata["onnx_optimization"] = optimized_result.to_dict()
                    result.path = optimized_result.path
                    result.checksum = optimized_result.checksum
                    result.checked = optimized_result.checked
                diff = None
                if onnx.runtime_diff:
                    if reference_output is None or example_input is None:
                        raise ValueError(
                            "onnx runtime_diff requires model reference output"
                        )
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
                        "pre_export_lowering": result_metadata.get(
                            "pre_export_lowering"
                        ),
                        "onnx_optimization": result_metadata.get("onnx_optimization"),
                        "export_guard": dict(export_guard),
                    }
                )
                target_summaries.append(_target_summary(target, exported[-1]))
                continue

            if target.format == "tensorrt":
                tensorrt = target.tensorrt
                onnx_path = tensorrt.onnx_path or context.artifacts.get("last_onnx")
                if onnx_path is None:
                    raise ValueError(
                        "TensorRT export requires tensorrt.onnx_path or a prior ONNX export"
                    )
                output_path = target.output_path
                if output_path is None:
                    output_path = str(artifact_dir / f"model_{index}.engine")
                result = build_tensorrt_engine(
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
                    runtime_benchmark = benchmark_tensorrt_engine(
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
                exported.append(
                    {
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
                        "profiling_verbosity": result.metadata.get(
                            "profiling_verbosity"
                        ),
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
                )
                target_summaries.append(_target_summary(target, exported[-1]))
                continue

            if target.format == "openvino":
                openvino = target.openvino
                source = openvino.onnx_path or context.artifacts.get("last_onnx")
                if source is None:
                    if export_model is None:
                        raise ValueError(
                            "OpenVINO export requires openvino.onnx_path or a loaded model"
                        )
                    source = export_model
                output_path = target.output_path
                if output_path is None:
                    output_path = str(artifact_dir / f"model_{index}.xml")
                result = export_openvino_ir(
                    source,
                    output_path,
                    example_input=example_input
                    if isinstance(source, nn.Module)
                    else None,
                    input_shape=openvino.input_shape,
                    dry_run=openvino.dry_run,
                )
                diff = None
                if (
                    not result.dry_run
                    and openvino.runtime_diff
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
                        device=openvino.device,
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
                        "artifact_status": "command_only"
                        if result.dry_run
                        else "materialized",
                        "backend_execution": "dry_run"
                        if result.dry_run
                        else "executed",
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
                    raise ValueError(
                        "executorch export requires a loaded model and example_inputs"
                    )
                output_path = target.output_path
                if output_path is None:
                    output_path = str(artifact_dir / f"model_{index}.pte")
                executorch = target.executorch
                result = export_executorch_program(
                    export_model,
                    example_input,
                    output_path,
                    dry_run=executorch.dry_run,
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
                        "artifact_status": "command_only"
                        if result.dry_run
                        else "materialized",
                        "backend_execution": "dry_run"
                        if result.dry_run
                        else "executed",
                        "checksum": result.checksum,
                    }
                )
                target_summaries.append(_target_summary(target, exported[-1]))
                continue

            if target.format == "ncnn":
                ncnn = target.ncnn
                source_path = ncnn.source_path
                if source_path is None:
                    if ncnn.converter == "pnnx":
                        source_path = context.artifacts.get(
                            "last_torchscript"
                        ) or context.artifacts.get("last_onnx")
                    else:
                        source_path = context.artifacts.get("last_onnx")
                if source_path is None:
                    raise ValueError(
                        "ncnn export requires ncnn.source_path or a compatible prior export"
                    )
                param_path = target.output_path
                if param_path is None:
                    param_path = str(artifact_dir / f"model_{index}.param")
                bin_path = ncnn.bin_path
                if bin_path is None:
                    bin_path = str(Path(param_path).with_suffix(".bin"))
                if ncnn.converter == "pnnx":
                    result = export_ncnn_with_pnnx(
                        source_path,
                        pnnx_path=ncnn.pnnx_path,
                        param_path=param_path,
                        bin_path=bin_path,
                        extra_args=ncnn.extra_args,
                        timeout=ncnn.timeout,
                        dry_run=ncnn.dry_run,
                    )
                else:
                    result = export_ncnn_from_onnx(
                        source_path,
                        param_path,
                        bin_path,
                        onnx2ncnn_path=ncnn.onnx2ncnn_path,
                        extra_args=ncnn.extra_args,
                        timeout=ncnn.timeout,
                        dry_run=ncnn.dry_run,
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
                        "artifact_status": "command_only"
                        if result.dry_run
                        else "materialized",
                        "backend_execution": "dry_run"
                        if result.dry_run
                        else "executed",
                        "command": result.command,
                        "checksums": result.checksums,
                    }
                )
                target_summaries.append(_target_summary(target, exported[-1]))
                continue

            if target.format == "mnn":
                mnn = target.mnn
                source_path = mnn.source_path or context.artifacts.get("last_onnx")
                if source_path is None:
                    raise ValueError(
                        "MNN export requires mnn.source_path or a prior ONNX export"
                    )
                output_path = target.output_path
                if output_path is None:
                    output_path = str(artifact_dir / f"model_{index}.mnn")
                result = export_mnn_from_onnx(
                    source_path,
                    output_path,
                    converter_path=mnn.converter_path,
                    framework=mnn.framework,
                    extra_args=mnn.extra_args,
                    timeout=mnn.timeout,
                    dry_run=mnn.dry_run,
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
                        "artifact_status": "command_only"
                        if result.dry_run
                        else "materialized",
                        "backend_execution": "dry_run"
                        if result.dry_run
                        else "executed",
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
            metrics["runtime_handle_request"] = asdict(runtime_handle_request)
            if runtime_handle_request.materialize:
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
    runtime_handle_request: DeployRuntimeHandleSpec | None = None
    if isinstance(spec, DeployStageSpec) and spec.runtime_handle is not None:
        runtime_handle_request = spec.runtime_handle
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
    "materialize_deploy_runtime_handle",
    "run_export_stage",
    "update_structured_prune_export_status",
]
