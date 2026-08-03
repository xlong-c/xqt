"""Export pass implementation for XQT stage workflows."""

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
    compare_onnxruntime_outputs,
    compare_openvino_outputs,
    create_onnxruntime_session,
    create_tensorrt_runtime_session,
    export_executorch_program,
    export_mnn_from_onnx,
    export_ncnn_from_onnx,
    export_ncnn_with_pnnx,
    export_qnn_from_onnx,
    mobile_export_diagnosis,
    export_onnx,
    export_openvino_ir,
    export_artifact_lineage_report,
    export_torch_program,
    export_torchscript,
    export_target_capability_report,
    openvino_runtime_layer_report,
    optimize_onnx,
    tensorrt_runtime_layer_report,
    validate_tensorrt_plugin_libraries,
)
from xqt.export.input_utils import first_tensor_output
from xqt.runtime.package import write_model_package

from .export_handlers._context import (
    call_model,
    resolve_export_model,
    update_structured_prune_export_status,
    _target_summary,
    _output_diff_runtime_config,
)
from .export_handlers.executorch import handle_executorch
from .export_handlers.mnn import handle_mnn
from .export_handlers.ncnn import handle_ncnn
from .export_handlers.qnn import handle_qnn
from .export_handlers.onnx import handle_onnx
from .export_handlers.openvino import handle_openvino
from .export_handlers.tensorrt import handle_tensorrt
from .export_handlers.torch_export import handle_torch_export
from .export_handlers.torchscript import handle_torchscript


def _package_io_entries(names: Any) -> list[dict[str, str]]:
    if not isinstance(names, list):
        return []
    return [{"name": str(item)} for item in names if isinstance(item, str)]


def _default_package_runtime_config(
    runtime_handle_request: DeployRuntimeHandleSpec | None,
) -> dict[str, Any]:
    providers = ["CPUExecutionProvider"]
    if runtime_handle_request is not None:
        resolved_runtime = runtime_handle_request.runtime or "onnxruntime"
        if resolved_runtime == "onnxruntime" and runtime_handle_request.onnxruntime.providers:
            providers = list(runtime_handle_request.onnxruntime.providers)
    return {
        "runtime": "onnxruntime",
        "providers": providers,
    }


def _attach_onnx_model_packages(
    context: XQTContext,
    *,
    export_targets: list[ExportTargetConfig],
    exported: list[dict[str, object]],
    target_summaries: list[dict[str, object]],
    artifact_dir: Path,
    stage_kind: str,
    runtime_handle_request: DeployRuntimeHandleSpec | None,
) -> None:
    runtime_config = _default_package_runtime_config(runtime_handle_request)
    for index, target in enumerate(export_targets):
        if target.format != "onnx":
            continue
        if index >= len(exported) or index >= len(target_summaries):
            continue
        item = exported[index]
        summary = target_summaries[index]
        raw_path = item.get("path")
        if not isinstance(raw_path, (str, Path)) or not str(raw_path):
            raise ValueError(
                "ONNX model package generation requires a materialized ONNX artifact path"
            )
        model_path = Path(raw_path)
        package_dir = artifact_dir / f"{model_path.stem}.xqtpkg"
        input_names = item.get("input_names")
        output_names = item.get("output_names")
        package_dir = write_model_package(
            model_path=model_path,
            output_dir=package_dir,
            model_format="onnx",
            runtime_name="onnxruntime",
            runtime_config=runtime_config,
            model_metadata={
                "opset": item.get("opset", target.opset),
                "checked": item.get("checked"),
                "input_names": input_names if isinstance(input_names, list) else [],
                "output_names": output_names if isinstance(output_names, list) else [],
                "dynamic_shapes": item.get("dynamic_shapes", {}),
            },
            io={
                "inputs": _package_io_entries(input_names),
                "outputs": _package_io_entries(output_names),
            },
            metadata={
                "project_name": context.project_name,
                "stage_kind": stage_kind,
                "source_artifact": str(model_path),
                "compression_axes": list(context.compression_axes),
                "export_guard": item.get("export_guard"),
                "pre_export_fusion": item.get("pre_export_fusion"),
                "pre_export_lowering": item.get("pre_export_lowering"),
                "onnx_optimization": item.get("onnx_optimization"),
            },
        )
        manifest_path = package_dir / "manifest.json"
        context.artifacts[f"model_package_{index}"] = package_dir
        context.artifacts[f"model_package_{index}_manifest"] = manifest_path
        context.artifacts["last_model_package"] = package_dir
        if context.manifest is not None:
            context.manifest.add_artifact(
                ArtifactRecord.from_file(
                    manifest_path,
                    format="xqt_model_package",
                    runtime="onnxruntime",
                    metadata={
                        "stage_kind": stage_kind,
                        "source_artifact": str(model_path),
                    },
                )
            )
        item["model_package"] = str(package_dir)
        item["model_package_manifest"] = str(manifest_path)
        summary["model_package"] = str(package_dir)
        summary["model_package_manifest"] = str(manifest_path)


def materialize_deploy_runtime_handle(
    context: XQTContext,
    request: DeployRuntimeHandleSpec,
    target_summaries: list[dict[str, object]],
) -> dict[str, object]:
    """Materialize and describe one verified deploy runtime handle."""

    runtime = request.runtime or "onnxruntime"
    handle_kind = request.handle_kind
    if runtime == "onnxruntime":
        onnx_targets = [
            item for item in target_summaries if item.get("format") == "onnx"
        ]
        if not onnx_targets:
            raise ValueError(
                "ONNX Runtime handle requires an ONNX export target in the same deploy stage"
            )
        if len(onnx_targets) != 1:
            raise ValueError(
                "ONNX Runtime handle requires exactly one ONNX export target in the same deploy stage"
            )
        onnx_path = onnx_targets[0].get("path")
        if not isinstance(onnx_path, (str, Path)) or not str(onnx_path):
            raise ValueError(
                "ONNX Runtime handle requires a materialized ONNX artifact in the same deploy stage"
            )
        onnx_path = Path(onnx_path)
        providers = request.onnxruntime.providers
        session = create_onnxruntime_session(
            onnx_path,
            providers=list(providers) or None,
        )
        return {
            "runtime": runtime,
            "handle_kind": handle_kind,
            "handle": session,
            "target_count": len(onnx_targets),
            "targets": onnx_targets,
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

    tensorrt_targets = [
        item for item in target_summaries if item.get("format") == "tensorrt"
    ]
    if not tensorrt_targets:
        raise ValueError(
            "TensorRT runtime handle requires a TensorRT export target in the same deploy stage"
        )
    if len(tensorrt_targets) != 1:
        raise ValueError(
            "TensorRT runtime handle requires exactly one TensorRT export target in the same deploy stage"
        )
    if any(bool(item.get("dry_run")) for item in tensorrt_targets):
        raise ValueError(
            "TensorRT runtime handle cannot materialize a dry-run engine artifact"
        )
    engine_path = tensorrt_targets[0].get("path")
    if not isinstance(engine_path, (str, Path)) or not str(engine_path):
        raise ValueError(
            "TensorRT runtime handle requires a materialized TensorRT engine artifact in the same deploy stage"
        )
    engine_path = Path(engine_path)
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


_FORMAT_HANDLERS: dict[str, Any] = {
    "torch_export": handle_torch_export,
    "torchscript": handle_torchscript,
    "onnx": handle_onnx,
    "tensorrt": handle_tensorrt,
    "openvino": handle_openvino,
    "executorch": handle_executorch,
    "ncnn": handle_ncnn,
    "mnn": handle_mnn,
    "qnn": handle_qnn,
}

_FORMATS_NEEDING_MODEL = frozenset(
    {"torch_export", "torchscript", "onnx", "executorch"}
)


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
            context.metrics.pop("export", None)
            return context
        resolved_output_diff = output_diff or context.output_diff_config
        if resolved_output_diff is None:
            raise ValueError("XQTContext.output_diff_config is required")
        artifact_dir = Path(context.artifact_dir)

        needs_model = any(
            target.format in _FORMATS_NEEDING_MODEL
            or (target.format == "openvino" and target.openvino.onnx_path is None)
            for target in export_targets
        )

        export_model: nn.Module | None = None
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
            handler = _FORMAT_HANDLERS.get(target.format)
            if handler is None:
                raise ValueError(f"Unsupported export format: {target.format}")

            if target.format in _FORMATS_NEEDING_MODEL:
                if export_model is None or example_input is None:
                    raise ValueError(
                        f"{target.format} requires a loaded model and example_inputs"
                    )
                kwargs: dict[str, Any] = {
                    "export_model": export_model,
                    "example_input": example_input,
                    "artifact_dir": artifact_dir,
                    "output_diff_config": resolved_output_diff,
                    "export_guard": export_guard,
                }
                if target.format == "onnx":
                    kwargs["reference_output"] = reference_output
                item, summary = handler(context, target, index, **kwargs)

            elif target.format == "openvino":
                item, summary = handler(
                    context,
                    target,
                    index,
                    export_model=export_model,
                    example_input=example_input,
                    reference_output=reference_output,
                    artifact_dir=artifact_dir,
                    output_diff_config=resolved_output_diff,
                )

            elif target.format in ("tensorrt", "ncnn", "mnn", "qnn"):
                item, summary = handler(
                    context, target, index, artifact_dir=artifact_dir
                )

            else:
                raise ValueError(f"Unsupported export format: {target.format}")

            exported.append(item)
            target_summaries.append(summary)

        metrics: dict[str, object] = {
            "artifacts": exported,
            "target_count": len(export_targets),
            "targets": target_summaries,
            "stage_kind": stage_kind,
        }
        _attach_onnx_model_packages(
            context,
            export_targets=export_targets,
            exported=exported,
            target_summaries=target_summaries,
            artifact_dir=artifact_dir,
            stage_kind=stage_kind,
            runtime_handle_request=runtime_handle_request,
        )
        target_capabilities = export_target_capability_report(export_targets)
        artifact_lineage = export_artifact_lineage_report(
            exported=exported,
            target_summaries=target_summaries,
            metrics=context.metrics,
            stage_kind=stage_kind,
        )
        metrics["target_capabilities"] = target_capabilities
        metrics["artifact_lineage"] = artifact_lineage
        if runtime_handle_request is not None:
            metrics["runtime_handle_request"] = asdict(runtime_handle_request)
            if runtime_handle_request.materialize:
                metrics["runtime_handle"] = materialize_deploy_runtime_handle(
                    context,
                    runtime_handle_request,
                    target_summaries,
                )
        context.metrics["export"] = metrics
        if context.manifest is not None:
            context.manifest.add_metric(
                MetricRecord(
                    name=f"{stage_kind}.target_capabilities",
                    value=target_capabilities["supported_target_count"],
                    threshold=target_capabilities["target_count"],
                    passed=(
                        not target_capabilities["unsupported_formats"]
                        and all(
                            bool(row["precision_supported"])
                            and bool(row["dynamic_shapes_compatible"])
                            for row in target_capabilities["targets"]
                        )
                    ),
                    metadata=target_capabilities,
                )
            )
            context.manifest.add_metric(
                MetricRecord(
                    name=f"{stage_kind}.artifact_lineage",
                    value=artifact_lineage["artifact_count"],
                    metadata=artifact_lineage,
                )
            )
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
