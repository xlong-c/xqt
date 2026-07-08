"""Quantization plan dispatcher."""

from __future__ import annotations

from typing import Any

from torch import nn

from xqt.core.types import XQTContext
from xqt.export import export_onnx
from xqt.quant.backends.onnx_qdq import (
    execute_onnx_qdq_component,
    onnx_qdq_graph_summary as _onnx_qdq_graph_summary,
    quantize_onnx_qdq_static,
)
from xqt.quant.backends.planned import execute_planned_method_component
from xqt.quant.backends.torchao import execute_torchao_component, quantize_with_torchao
from xqt.quant.capability import _resolve_nature
from xqt.quant.execution.component import prefix_module_names
from xqt.quant.execution.selection import selection_policy_metadata
from xqt.quant.quantizers.reference_fp4 import (
    execute_reference_fp4_component,
    quantize_with_reference_fp4,
)
from xqt.quant.quantizers.svd import execute_svdquant_component, quantize_with_svd
from xqt.quant.types import (
    QuantizationComponentPlan,
    QuantizationExecutionPlan,
    QuantizationExecutionResult,
    QuantizationReport,
)

from .reporting import summarize_quantization_reports


def _component_requires_model(component: QuantizationComponentPlan) -> bool:
    if component.backend == "torchao":
        return True
    if component.backend != "onnxruntime_qdq":
        return True
    if component.policy.get("onnx_path") is not None:
        return False
    return True


def _analysis_only_report(component: QuantizationComponentPlan) -> QuantizationReport:
    return QuantizationReport(
        component_name=component.name,
        backend=component.backend,
        method=component.method,
        strategy=component.strategy,
        target_path=component.target_path,
        high_precision_modules=prefix_module_names(
            component.keep_high_precision,
            component.target_path,
        ),
        skipped_modules=prefix_module_names(
            component.skip_quantize,
            component.target_path,
        ),
        nature=_resolve_nature(component.strategy, component.policy),
        metadata={
            "analysis_only": True,
            "executed": False,
            "selection_policy": selection_policy_metadata(component),
        },
    )


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
            reports.append(_analysis_only_report(component))
            continue
        if _component_requires_model(component) and current_model is None:
            raise ValueError(
                f"PyTorch model is required for quantization component "
                f"'{component.name}' backend '{component.backend}'"
            )
        if component.backend == "torchao":
            current_model, report = execute_torchao_component(
                context,
                current_model,
                component,
                quantize_fn=quantize_with_torchao,
            )
            reports.append(report)
            continue
        if (
            component.backend == "pytorch"
            and component.strategy == "fp4_weight_only"
        ):
            current_model, report = execute_reference_fp4_component(
                context,
                current_model,
                component,
                quantize_fn=quantize_with_reference_fp4,
            )
            reports.append(report)
            continue
        if component.backend == "svdquant" or (
            component.backend == "pytorch"
            and component.strategy in {"svd_fp4", "svd_int4"}
        ):
            current_model, report = execute_svdquant_component(
                context,
                current_model,
                component,
                quantize_fn=quantize_with_svd,
            )
            reports.append(report)
            continue
        if component.backend == "onnxruntime_qdq":
            current_model, report, component_artifacts = execute_onnx_qdq_component(
                context,
                current_model,
                component,
                export_onnx_fn=export_onnx_fn,
                quantize_onnx_qdq_static_fn=quantize_onnx_qdq_static_fn,
                graph_summary_fn=_onnx_qdq_graph_summary,
            )
            artifacts.update(component_artifacts)
            reports.append(report)
            continue
        if component.backend in {"pytorch", "tilelang"} and component.method in {"awq", "gptq"}:
            current_model, report, component_artifacts = execute_planned_method_component(
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


__all__ = [
    "_onnx_qdq_graph_summary",
    "execute_quantization_plan",
    "quantize_onnx_qdq_static",
    "quantize_with_reference_fp4",
    "quantize_with_svd",
    "quantize_with_torchao",
    "summarize_quantization_reports",
]
