"""Quantization plan execution helpers."""

from __future__ import annotations

from pathlib import Path
from itertools import chain
from typing import Any, Iterable, Mapping, Optional, Sequence

from torch import nn

from xqt.core.inputs import extract_model_inputs, infer_model_input_count
from xqt.core.types import XQTContext
from xqt.export import export_onnx
from xqt.export.input_utils import default_input_names

from .fp4_backend import quantize_with_reference_fp4
from .onnx_qdq import quantize_onnx_qdq_static
from .calibration_summary import build_calibration_summary
from .capability import describe_quant_backend_capability, _resolve_nature
from .svd_quant import quantize_with_svd
from .torchao_backend import quantize_with_torchao
from .types import (
    QuantizationComponentPlan,
    QuantizationExecutionPlan,
    QuantizationExecutionResult,
    QuantizationNature,
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
    model: nn.Module | None,
    target_path: Optional[str],
) -> nn.Module:
    if model is None:
        raise ValueError("PyTorch model is required for this quantization component")
    if not target_path:
        return model
    return model.get_submodule(target_path)


def _replace_component_model(
    model: nn.Module | None,
    target_path: Optional[str],
    replacement: nn.Module,
) -> nn.Module:
    if model is None:
        raise ValueError("PyTorch model is required to replace a quantized component")
    if not target_path:
        return replacement
    parent_path, _, attribute = target_path.rpartition(".")
    parent = model.get_submodule(parent_path) if parent_path else model
    if attribute.isdigit() and isinstance(parent, (nn.Sequential, nn.ModuleList)):
        parent[int(attribute)] = replacement
    else:
        setattr(parent, attribute, replacement)
    return model


def _resolve_calibration_inputs(
    context: XQTContext,
    component: QuantizationComponentPlan,
) -> Iterable[Any]:
    calibration_inputs = context.calibration_inputs
    if calibration_inputs is not None:
        return calibration_inputs
    raise ValueError(
        "calibration_inputs are required for component "
        f"'{component.name}' backend '{component.backend}'"
    )


def _optional_calibration_summary(
    context: XQTContext,
    component: QuantizationComponentPlan,
) -> tuple[int | None, dict[str, Any] | None]:
    calibration_inputs = context.calibration_inputs
    if calibration_inputs is None:
        return None, None
    input_names = component.policy.get("input_names")
    summary = build_calibration_summary(
        calibration_inputs,
        input_names=input_names if isinstance(input_names, Sequence) else None,
        sample_limit=component.policy.get("sample_limit"),
        calibrator_type="XQTCalibrationInputSummary",
        observer_type=f"{component.backend}.calibration_inputs",
    )
    return int(summary["sample_count"]), summary


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


def _selection_policy_metadata(component: QuantizationComponentPlan) -> dict[str, Any]:
    raw = component.policy.get("selection_policy")
    if isinstance(raw, Mapping):
        return dict(raw)
    return {
        "target_path": component.target_path,
        "selectors": {},
        "keep_high_precision": list(component.keep_high_precision),
        "skip_quantize": list(component.skip_quantize),
        "force_quantize": list(component.force_quantize),
    }


def _module_selection_reason_metadata(
    component: QuantizationComponentPlan,
    *,
    quantized_modules: list[str],
    skipped_modules: list[str],
    high_precision_modules: list[str],
) -> dict[str, dict[str, str]]:
    forced = set(_prefix_module_names(component.force_quantize, component.target_path))
    explicit_skip = set(_prefix_module_names(component.skip_quantize, component.target_path))
    high_precision = set(high_precision_modules)
    quantized: dict[str, str] = {}
    skipped: dict[str, str] = {}

    for name in quantized_modules:
        quantized[name] = "force_quantize" if name in forced else "matched_selection_policy"
    for name in skipped_modules:
        if name in high_precision:
            skipped[name] = "keep_high_precision"
        elif name in explicit_skip:
            skipped[name] = "skip_quantize"
        else:
            skipped[name] = "filtered_by_selection_policy"

    return {
        "quantized": quantized,
        "skipped": skipped,
        "high_precision": {
            name: "keep_high_precision" for name in high_precision_modules
        },
        "fallback": {},
    }


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


def _onnx_qdq_graph_summary(path: str | Path) -> dict[str, Any]:
    try:
        import onnx
    except ImportError:
        return {}
    try:
        model = onnx.load(str(path))
    except Exception:
        return {}
    op_type_counts: dict[str, int] = {}
    for node in model.graph.node:
        op_type_counts[node.op_type] = op_type_counts.get(node.op_type, 0) + 1
    return {
        "node_count": len(model.graph.node),
        "op_type_counts": op_type_counts,
        "qdq_node_count": op_type_counts.get("QuantizeLinear", 0)
        + op_type_counts.get("DequantizeLinear", 0),
        "quantize_linear_count": op_type_counts.get("QuantizeLinear", 0),
        "dequantize_linear_count": op_type_counts.get("DequantizeLinear", 0),
    }


def _infer_qdq_quantized_op_types(qdq_graph: dict[str, Any]) -> list[str]:
    op_type_counts = qdq_graph.get("op_type_counts", {})
    if not isinstance(op_type_counts, dict):
        return []
    wrapper_op_types = {"QuantizeLinear", "DequantizeLinear", "Constant"}
    return sorted(
        str(op_type)
        for op_type in op_type_counts
        if str(op_type) not in wrapper_op_types
    )


def _execute_torchao_component(
    context: XQTContext,
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
    quantized_modules = _prefix_module_names(result.quantized_modules, component.target_path)
    module_selection_reasons = _module_selection_reason_metadata(
        component,
        quantized_modules=quantized_modules,
        skipped_modules=skipped_modules,
        high_precision_modules=high_precision_modules,
    )
    calibration_samples, calibration_summary = _optional_calibration_summary(
        context,
        component,
    )
    nature = _resolve_nature(component.strategy, component.policy)
    compute_speedup = 1.0 if nature == QuantizationNature.TRUE else None
    report = QuantizationReport(
        component_name=component.name,
        backend=result.backend,
        runtime="pytorch",
        method=component.method,
        strategy=result.strategy,
        target_path=component.target_path,
        quantized_modules=quantized_modules,
        skipped_modules=skipped_modules,
        high_precision_modules=high_precision_modules,
        calibration_samples=calibration_samples,
        calibration_summary=calibration_summary,
        nature=nature,
        compute_speedup_expected=compute_speedup,
        metadata={
            **dict(result.metadata),
            "analysis_only": component.analysis_only,
            "policy": effective_policy,
            "selection_policy": _selection_policy_metadata(component),
            "module_selection_reasons": module_selection_reasons,
        },
    )
    return updated_model, report


def _execute_reference_fp4_component(
    context: XQTContext,
    root_model: nn.Module,
    component: QuantizationComponentPlan,
) -> tuple[nn.Module, QuantizationReport]:
    target_model = _resolve_component_model(root_model, component.target_path)
    effective_policy = _build_torchao_policy(component)
    result = quantize_with_reference_fp4(
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
    quantized_modules = _prefix_module_names(result.quantized_modules, component.target_path)
    module_selection_reasons = _module_selection_reason_metadata(
        component,
        quantized_modules=quantized_modules,
        skipped_modules=skipped_modules,
        high_precision_modules=high_precision_modules,
    )
    calibration_samples, calibration_summary = _optional_calibration_summary(
        context,
        component,
    )
    report = QuantizationReport(
        component_name=component.name,
        backend=result.backend,
        runtime="pytorch",
        method=component.method,
        strategy=result.strategy,
        target_path=component.target_path,
        quantized_modules=quantized_modules,
        skipped_modules=skipped_modules,
        high_precision_modules=high_precision_modules,
        calibration_samples=calibration_samples,
        calibration_summary=calibration_summary,
        nature=QuantizationNature.PSEUDO,
        compute_speedup_expected=None,
        metadata={
            **dict(result.metadata),
            "analysis_only": component.analysis_only,
            "policy": effective_policy,
            "selection_policy": _selection_policy_metadata(component),
            "module_selection_reasons": module_selection_reasons,
            "executed": True,
            "execution_state": "reference_fp4_weight_only",
        },
    )
    return updated_model, report


def _execute_svdquant_component(
    context: XQTContext,
    root_model: nn.Module,
    component: QuantizationComponentPlan,
) -> tuple[nn.Module, QuantizationReport]:
    target_model = _resolve_component_model(root_model, component.target_path)
    effective_policy = _build_torchao_policy(component)
    configured_rank = int(effective_policy.get("rank", 32))
    configured_group_size = int(effective_policy.get("group_size", 128))
    configured_quant_dtype = str(effective_policy.get("quant_dtype", "int4"))
    result = quantize_with_svd(
        target_model,
        policy=effective_policy,
        strategy=component.strategy or effective_policy.get("strategy"),
        rank=configured_rank,
        group_size=configured_group_size,
        quant_dtype=configured_quant_dtype,
        inplace=True,
        collect_analysis=True,
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
    quantized_modules = _prefix_module_names(result.quantized_modules, component.target_path)
    module_selection_reasons = _module_selection_reason_metadata(
        component,
        quantized_modules=quantized_modules,
        skipped_modules=skipped_modules,
        high_precision_modules=high_precision_modules,
    )
    nature = _resolve_nature(component.strategy, component.policy)
    report = QuantizationReport(
        component_name=component.name,
        backend=result.backend,
        runtime="pytorch",
        method=component.method,
        strategy=result.strategy,
        target_path=component.target_path,
        quantized_modules=quantized_modules,
        skipped_modules=skipped_modules,
        high_precision_modules=high_precision_modules,
        nature=nature,
        compute_speedup_expected=None,
        metadata={
            **dict(result.metadata),
            "analysis_only": component.analysis_only,
            "policy": effective_policy,
            "selection_policy": _selection_policy_metadata(component),
            "module_selection_reasons": module_selection_reasons,
            "executed": True,
            "execution_state": "svdquant_reference",
            "rank": configured_rank,
            "group_size": configured_group_size,
            "quant_dtype": configured_quant_dtype,
        },
    )
    return updated_model, report


def _execute_onnx_qdq_component(
    context: XQTContext,
    root_model: nn.Module | None,
    component: QuantizationComponentPlan,
    *,
    export_onnx_fn: Any,
    quantize_onnx_qdq_static_fn: Any,
) -> tuple[nn.Module | None, QuantizationReport, dict[str, Any]]:
    calibration_inputs = _resolve_calibration_inputs(context, component)
    policy = dict(component.policy)
    calibration_iterator = iter(calibration_inputs)
    batch = next(calibration_iterator)
    calibration_data = chain([batch], calibration_iterator)
    artifact_dir = Path(context.config.project.artifact_dir)
    onnx_key = _artifact_key("last_onnx", component.name)
    default_last_onnx = context.artifacts.get(onnx_key)
    if component.name == "model" and default_last_onnx is None:
        default_last_onnx = context.artifacts.get("last_onnx")
    onnx_path = policy.get("onnx_path") or default_last_onnx
    export_metadata: dict[str, Any] = {}
    input_names = list(policy.get("input_names") or [])
    input_structure = "external_onnx"
    if onnx_path is None:
        component_model = _resolve_component_model(root_model, component.target_path)
        example_input = extract_model_inputs(
            batch,
            expected_input_count=infer_model_input_count(component_model),
        )
        if not input_names:
            input_names = default_input_names(example_input)
        input_structure = _module_structure_name(example_input)
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
    elif not input_names:
        raise ValueError(
            "onnxruntime_qdq with external onnx_path requires policy.input_names"
        )
    output_path = policy.get("output_path")
    if output_path is None:
        output_path = str(artifact_dir / _component_output_name(component))
    result = quantize_onnx_qdq_static_fn(
        onnx_path,
        output_path,
        calibration_data,
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
    qdq_graph = _onnx_qdq_graph_summary(result.path)
    if qdq_graph:
        metadata["qdq_graph"] = qdq_graph
        requested_op_types = [
            str(item) for item in policy.get("op_types_to_quantize") or []
        ]
        op_type_counts = qdq_graph.get("op_type_counts", {})
        if requested_op_types:
            metadata["quantized_op_types"] = [
                op_type for op_type in requested_op_types if op_type in op_type_counts
            ]
        else:
            metadata["quantized_op_types"] = _infer_qdq_quantized_op_types(qdq_graph)
        metadata["qdq_node_count"] = qdq_graph["qdq_node_count"]
    metadata.update(
        {
            "component_name": component.name,
            "calibration_source": "context.calibration_inputs",
            "input_structure": input_structure,
            "policy": policy,
            "selection_policy": _selection_policy_metadata(component),
        }
    )
    if export_metadata.get("pre_export_fusion") is not None:
        metadata["pre_export_fusion"] = dict(export_metadata["pre_export_fusion"])
    report = QuantizationReport(
        component_name=component.name,
        backend="onnxruntime_qdq",
        runtime="onnxruntime",
        method=component.method,
        strategy=component.strategy,
        target_path=component.target_path,
        quantized_modules=[
            f"onnx::{op_type}"
            for op_type in metadata.get("quantized_op_types", [])
        ],
        skipped_modules=_prefix_module_names(component.skip_quantize, component.target_path),
        high_precision_modules=_prefix_module_names(
            component.keep_high_precision,
            component.target_path,
        ),
        artifacts={"onnx": str(result.path)},
        calibration_samples=result.calibration_samples,
        calibration_summary=metadata.get("calibration_summary"),
        nature=_resolve_nature(component.strategy, component.policy),
        metadata={
            **metadata,
            "path": str(result.path),
            "checksum": result.checksum,
            "analysis_only": component.analysis_only,
            "selection_policy": _selection_policy_metadata(component),
        },
    )
    artifact_updates = {
        _artifact_key("quant_onnx", component.name): result.path,
        _artifact_key("last_onnx", component.name): result.path,
    }
    if component.name == "model":
        artifact_updates["quant_onnx"] = result.path
        artifact_updates["last_onnx"] = result.path
    return root_model, report, artifact_updates


def _component_requires_model(component: QuantizationComponentPlan) -> bool:
    if component.backend == "torchao":
        return True
    if component.backend != "onnxruntime_qdq":
        return True
    if component.policy.get("onnx_path") is not None:
        return False
    return True


def _execute_planned_method_component(
    root_model: nn.Module | None,
    component: QuantizationComponentPlan,
) -> tuple[nn.Module | None, QuantizationReport, dict[str, Any]]:
    capability = describe_quant_backend_capability(
        component.backend,
        method=component.method,
        strategy=component.strategy,
        policy=component.policy,
    )
    artifact_name = f"{component.name}_{component.backend}_{component.method or 'planned'}.json"
    artifact_dir = Path("artifacts")
    planned_artifact = artifact_dir / artifact_name
    artifact_payload = {
        "component_name": component.name,
        "backend": component.backend,
        "method": component.method,
        "strategy": component.strategy,
        "target_path": component.target_path,
        "policy": dict(component.policy),
        "status": "planned",
        "runtime": capability.runtime,
        "requires_cuda": capability.requires_cuda,
        "notes": list(capability.notes),
        "limitations": list(capability.limitations),
    }
    report = QuantizationReport(
        component_name=component.name,
        backend=component.backend,
        runtime=capability.runtime,
        method=component.method,
        strategy=component.strategy,
        target_path=component.target_path,
        skipped_modules=_prefix_module_names(component.skip_quantize, component.target_path),
        high_precision_modules=_prefix_module_names(
            component.keep_high_precision,
            component.target_path,
        ),
        nature=capability.nature,
        artifacts={"planned": str(planned_artifact)},
        metadata={
            "execution_state": "planned",
            "executed": False,
            "capability": capability.to_dict(),
            "planned_artifact": str(planned_artifact),
            "policy": dict(component.policy),
            "selection_policy": _selection_policy_metadata(component),
        },
    )
    artifact_updates = {
        _artifact_key("quant_plan", component.name): planned_artifact,
    }
    return root_model, report, artifact_updates


def execute_quantization_plan(
    context: XQTContext,
    plan: QuantizationExecutionPlan,
    *,
    export_onnx_fn: Any = export_onnx,
    quantize_onnx_qdq_static_fn: Any = quantize_onnx_qdq_static,
) -> QuantizationExecutionResult:
    """Execute a normalized quantization plan and return unified reports."""

    current_model = context.model if isinstance(context.model, nn.Module) else None
    reports: list[QuantizationReport] = []
    artifacts: dict[str, Any] = {}
    for component in plan.components:
        if component.analysis_only:
            reports.append(
                QuantizationReport(
                    component_name=component.name,
                    backend=component.backend,
                    method=component.method,
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
                    nature=_resolve_nature(component.strategy, component.policy),
                    metadata={
                        "analysis_only": True,
                        "executed": False,
                        "selection_policy": _selection_policy_metadata(component),
                    },
                )
            )
            continue
        if _component_requires_model(component) and current_model is None:
            raise ValueError(
                f"PyTorch model is required for quantization component "
                f"'{component.name}' backend '{component.backend}'"
            )
        if component.backend == "torchao":
            current_model, report = _execute_torchao_component(context, current_model, component)
            reports.append(report)
            continue
        if (
            component.backend == "pytorch"
            and component.strategy == "fp4_weight_only"
        ):
            current_model, report = _execute_reference_fp4_component(
                context,
                current_model,
                component,
            )
            reports.append(report)
            continue
        if component.backend == "svdquant" or (
            component.backend == "pytorch"
            and component.strategy in {"svd_fp4", "svd_int4"}
        ):
            current_model, report = _execute_svdquant_component(
                context,
                current_model,
                component,
            )
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
        if component.backend in {"pytorch", "tilelang"} and component.method in {"awq", "gptq"}:
            current_model, report, component_artifacts = _execute_planned_method_component(
                current_model,
                component,
            )
            artifacts.update(component_artifacts)
            reports.append(report)
            continue
        if component.backend in {"transformers", "bitsandbytes"}:
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
        "skipped_module_count": sum(len(report.skipped_modules) for report in reports),
        "high_precision_module_count": sum(
            len(report.high_precision_modules) for report in reports
        ),
        "artifact_count": sum(len(report.artifacts) for report in reports),
        "calibration_component_count": sum(
            1 for report in reports if report.calibration_summary is not None
        ),
    }
    metrics: dict[str, Any] = {
        "mode": "multi_component" if len(reports) > 1 else "single_component",
        "components": components,
        "summary": summary,
        "quantized_modules_by_component": {
            report.component_name: list(report.quantized_modules)
            for report in reports
        },
        "skipped_modules_by_component": {
            report.component_name: list(report.skipped_modules)
            for report in reports
        },
        "high_precision_modules_by_component": {
            report.component_name: list(report.high_precision_modules)
            for report in reports
        },
        "module_selection_reasons_by_component": {
            report.component_name: dict(report.metadata["module_selection_reasons"])
            for report in reports
            if "module_selection_reasons" in report.metadata
        },
        "calibration_summaries": {
            report.component_name: dict(report.calibration_summary)
            for report in reports
            if report.calibration_summary is not None
        },
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
            "method": first.method,
            "strategy": first.strategy,
            "quantized_modules": list(first.quantized_modules),
            "quantized_module_count": len(first.quantized_modules),
            "skipped_modules": list(first.skipped_modules),
            "high_precision_modules": list(first.high_precision_modules),
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
