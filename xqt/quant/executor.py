"""Quantization plan execution helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

from torch import nn

from xqt.core.types import XQTContext
from xqt.data import extract_model_inputs, infer_model_input_count
from xqt.export import export_onnx
from xqt.export.input_utils import default_input_names

from .onnx_qdq import quantize_onnx_qdq_static
from .torchao_backend import quantize_with_torchao
from .types import (
    QuantizationComponentPlan,
    QuantizationExecutionPlan,
    QuantizationExecutionResult,
    QuantizationReport,
)


def _ordered_unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _prefix_module_names(names: Iterable[str], prefix: Optional[str]) -> list[str]:
    if not prefix:
        return list(names)
    prefixed: list[str] = []
    for name in names:
        prefixed.append(f"{prefix}.{name}" if name else prefix)
    return prefixed


def _module_structure_name(example_input: Any) -> str:
    if isinstance(example_input, Mapping):
        return "mapping"
    if isinstance(example_input, tuple):
        return "tuple"
    if isinstance(example_input, list):
        return "list"
    return "tensor"


def _resolve_component_model(
    model: nn.Module,
    target_path: Optional[str],
) -> nn.Module:
    if not target_path:
        return model
    return model.get_submodule(target_path)


def _replace_component_model(
    model: nn.Module,
    target_path: Optional[str],
    replacement: nn.Module,
) -> nn.Module:
    if not target_path:
        return replacement
    parent_path, _, attribute = target_path.rpartition(".")
    parent = model.get_submodule(parent_path) if parent_path else model
    if attribute.isdigit() and isinstance(parent, (nn.Sequential, nn.ModuleList)):
        parent[int(attribute)] = replacement
    else:
        setattr(parent, attribute, replacement)
    return model


def _resolve_loader(
    context: XQTContext,
    component: QuantizationComponentPlan,
) -> tuple[Iterable[Any], str]:
    calibration_split = component.calibration_split or "calibration"
    validation_split = component.validation_split or "validation"
    dataloader = context.data.get(calibration_split)
    if dataloader is not None:
        return dataloader, calibration_split
    if validation_split != calibration_split:
        dataloader = context.data.get(validation_split)
        if dataloader is not None:
            return dataloader, validation_split
    raise ValueError(
        f"{calibration_split} or {validation_split} data is required for component "
        f"'{component.name}' backend '{component.backend}'"
    )


def _build_torchao_policy(component: QuantizationComponentPlan) -> dict[str, Any]:
    policy = dict(component.policy)
    include_module_names = list(policy.get("include_module_names") or [])
    exclude_module_names = list(policy.get("exclude_module_names") or [])
    include_module_names.extend(component.force_quantize)
    exclude_module_names.extend(component.skip_quantize)
    exclude_module_names.extend(component.keep_high_precision)
    if include_module_names:
        policy["include_module_names"] = _ordered_unique(str(name) for name in include_module_names)
    if exclude_module_names:
        policy["exclude_module_names"] = _ordered_unique(str(name) for name in exclude_module_names)
    if component.strategy is not None:
        policy.setdefault("strategy", component.strategy)
    return policy


def _component_source_name(component: QuantizationComponentPlan) -> str:
    if component.name == "model":
        return "quant_source.onnx"
    return f"{component.name}_source.onnx"


def _component_output_name(component: QuantizationComponentPlan) -> str:
    if component.name == "model":
        return "model_qdq.onnx"
    return f"{component.name}_qdq.onnx"


def _artifact_key(prefix: str, component_name: str) -> str:
    if component_name == "model":
        return prefix
    return f"{prefix}_{component_name}"


def _execute_torchao_component(
    root_model: nn.Module,
    component: QuantizationComponentPlan,
) -> tuple[nn.Module, QuantizationReport]:
    target_model = _resolve_component_model(root_model, component.target_path)
    effective_policy = _build_torchao_policy(component)
    result = quantize_with_torchao(
        target_model,
        policy=effective_policy,
        strategy=component.strategy or effective_policy.get("strategy"),
        inplace=True,
    )
    updated_model = _replace_component_model(root_model, component.target_path, result.model)
    high_precision_modules = _prefix_module_names(
        component.keep_high_precision,
        component.target_path,
    )
    skipped_modules = _ordered_unique(
        [
            *_prefix_module_names(component.skip_quantize, component.target_path),
            *high_precision_modules,
        ]
    )
    report = QuantizationReport(
        component_name=component.name,
        backend=result.backend,
        runtime="pytorch",
        strategy=result.strategy,
        target_path=component.target_path,
        quantized_modules=_prefix_module_names(result.quantized_modules, component.target_path),
        skipped_modules=skipped_modules,
        high_precision_modules=high_precision_modules,
        metadata={
            **dict(result.metadata),
            "analysis_only": component.analysis_only,
            "policy": effective_policy,
        },
    )
    return updated_model, report


def _execute_onnx_qdq_component(
    context: XQTContext,
    root_model: nn.Module,
    component: QuantizationComponentPlan,
    *,
    export_onnx_fn: Any,
    quantize_onnx_qdq_static_fn: Any,
) -> tuple[nn.Module, QuantizationReport, dict[str, Any]]:
    component_model = _resolve_component_model(root_model, component.target_path)
    dataloader, source_split = _resolve_loader(context, component)
    policy = dict(component.policy)
    batch = next(iter(dataloader))
    example_input = extract_model_inputs(
        batch,
        expected_input_count=infer_model_input_count(component_model),
    )
    input_names = list(policy.get("input_names") or default_input_names(example_input))
    artifact_dir = Path(context.config.project.artifact_dir)
    onnx_key = _artifact_key("last_onnx", component.name)
    default_last_onnx = context.artifacts.get(onnx_key)
    if component.name == "model" and default_last_onnx is None:
        default_last_onnx = context.artifacts.get("last_onnx")
    onnx_path = policy.get("onnx_path") or default_last_onnx
    export_metadata: dict[str, Any] = {}
    if onnx_path is None:
        onnx_path = artifact_dir / str(policy.get("source_name", _component_source_name(component)))
        export_result = export_onnx_fn(
            component_model,
            example_input,
            onnx_path,
            opset=policy.get("opset"),
            input_names=input_names,
            output_names=policy.get("output_names"),
            dynamo=bool(policy.get("dynamo", True)),
            validate=bool(policy.get("validate", True)),
            pre_export_fusion=policy.get("pre_export_fusion"),
        )
        export_metadata = dict(export_result.metadata)
    output_path = policy.get("output_path")
    if output_path is None:
        output_path = str(artifact_dir / _component_output_name(component))
    result = quantize_onnx_qdq_static_fn(
        onnx_path,
        output_path,
        dataloader,
        input_names=input_names,
        sample_limit=policy.get("sample_limit"),
        activation_type=str(policy.get("activation_type", "QUInt8")),
        weight_type=str(policy.get("weight_type", "QInt8")),
        per_channel=bool(policy.get("per_channel", False)),
        reduce_range=bool(policy.get("reduce_range", False)),
        op_types_to_quantize=policy.get("op_types_to_quantize"),
        extra_options=policy.get("extra_options"),
    )
    metadata = dict(result.metadata)
    metadata.update(
        {
            "component_name": component.name,
            "source_split": source_split,
            "input_structure": _module_structure_name(example_input),
            "policy": policy,
        }
    )
    if export_metadata.get("pre_export_fusion") is not None:
        metadata["pre_export_fusion"] = dict(export_metadata["pre_export_fusion"])
    report = QuantizationReport(
        component_name=component.name,
        backend="onnxruntime_qdq",
        runtime="onnxruntime",
        strategy=component.strategy,
        target_path=component.target_path,
        skipped_modules=_prefix_module_names(component.skip_quantize, component.target_path),
        high_precision_modules=_prefix_module_names(
            component.keep_high_precision,
            component.target_path,
        ),
        artifacts={"onnx": str(result.path)},
        calibration_samples=result.calibration_samples,
        calibration_summary=metadata.get("calibration_summary"),
        source_split=source_split,
        metadata={
            **metadata,
            "path": str(result.path),
            "checksum": result.checksum,
            "analysis_only": component.analysis_only,
        },
    )
    artifact_updates = {
        _artifact_key("quant_onnx", component.name): result.path,
        _artifact_key("last_onnx", component.name): result.path,
    }
    if component.name == "model":
        artifact_updates["quant_onnx"] = result.path
        artifact_updates["last_onnx"] = result.path
    return context.require_model(), report, artifact_updates


def execute_quantization_plan(
    context: XQTContext,
    plan: QuantizationExecutionPlan,
    *,
    export_onnx_fn: Any = export_onnx,
    quantize_onnx_qdq_static_fn: Any = quantize_onnx_qdq_static,
) -> QuantizationExecutionResult:
    """Execute a normalized quantization plan and return unified reports."""

    current_model = context.require_model()
    reports: list[QuantizationReport] = []
    artifacts: dict[str, Any] = {}
    for component in plan.components:
        if component.analysis_only:
            reports.append(
                QuantizationReport(
                    component_name=component.name,
                    backend=component.backend,
                    strategy=component.strategy,
                    target_path=component.target_path,
                    high_precision_modules=_prefix_module_names(
                        component.keep_high_precision,
                        component.target_path,
                    ),
                    skipped_modules=_prefix_module_names(
                        component.skip_quantize,
                        component.target_path,
                    ),
                    metadata={"analysis_only": True, "executed": False},
                )
            )
            continue
        if component.backend == "torchao":
            current_model, report = _execute_torchao_component(current_model, component)
            reports.append(report)
            continue
        if component.backend == "onnxruntime_qdq":
            current_model, report, component_artifacts = _execute_onnx_qdq_component(
                context,
                current_model,
                component,
                export_onnx_fn=export_onnx_fn,
                quantize_onnx_qdq_static_fn=quantize_onnx_qdq_static_fn,
            )
            artifacts.update(component_artifacts)
            reports.append(report)
            continue
        if component.backend in {"gptq", "awq", "bitsandbytes"}:
            raise NotImplementedError(
                f"Quantization backend '{component.backend}' is planned but not executable yet"
            )
        raise ValueError(f"Unsupported quantization backend: {component.backend}")
    return QuantizationExecutionResult(
        model=current_model,
        reports=reports,
        artifacts=artifacts,
    )


def summarize_quantization_reports(reports: list[QuantizationReport]) -> dict[str, Any]:
    """Build a unified metrics payload from backend reports."""

    components = [report.to_dict() for report in reports]
    summary = {
        "component_count": len(reports),
        "backends": [report.backend for report in reports],
        "quantized_module_count": sum(len(report.quantized_modules) for report in reports),
        "artifact_count": sum(len(report.artifacts) for report in reports),
    }
    metrics: dict[str, Any] = {
        "mode": "multi_component" if len(reports) > 1 else "single_component",
        "components": components,
        "summary": summary,
        "artifacts": {
            report.component_name: dict(report.artifacts)
            for report in reports
            if report.artifacts
        },
    }
    if not reports:
        return metrics

    first = reports[0]
    metrics.update(
        {
            "backend": first.backend,
            "strategy": first.strategy,
            "quantized_modules": list(first.quantized_modules),
            "quantized_module_count": len(first.quantized_modules),
            "calibration_samples": first.calibration_samples,
            "calibration_summary": first.calibration_summary,
            "metadata": dict(first.metadata),
        }
    )
    if "path" in first.metadata:
        metrics["path"] = first.metadata["path"]
    if "checksum" in first.metadata:
        metrics["checksum"] = first.metadata["checksum"]
    return metrics


__all__ = [
    "execute_quantization_plan",
    "summarize_quantization_reports",
]
