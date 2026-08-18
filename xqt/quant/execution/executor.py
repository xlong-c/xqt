"""Quantization plan dispatcher.

Routing is table-driven: ``xqt.quant.registry`` is the single source of truth
for which (backend, method, strategy, compute) combination executes which
quantizer. Combinations with no registered executable route fall back to a
structured planned report (``xqt.quant.backends.planned``) instead of an
opaque dispatch error; unknown backends still fail fast through the capability
validation inside the planned-report path.
"""

from __future__ import annotations

from typing import Any

from torch import nn

from xqt.core.types import XQTContext
from xqt.quant import quantizers as _quantizers  # noqa: F401  # register routes
from xqt.quant.backends.onnx_qdq import (
    onnx_qdq_graph_summary as _onnx_qdq_graph_summary,
    quantize_onnx_qdq_static,
)
from xqt.quant.backends.planned import execute_planned_method_component
from xqt.quant.backends.torchao import quantize_with_torchao
from xqt.quant.capability import _resolve_nature
from xqt.quant.component import prefix_module_names
from xqt.quant.registry import RouteQuery, resolve_quant_route
from xqt.quant.selection import selection_policy_metadata
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
    method_semantics = "analysis_only_no_executable_algorithm"
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
        algorithm_executable=False,
        method_semantics=method_semantics,
        metadata={
            "analysis_only": True,
            "executed": False,
            "algorithm_executable": False,
            "method_semantics": method_semantics,
            "selection_policy": selection_policy_metadata(component),
        },
    )


def execute_quantization_plan(
    context: XQTContext,
    plan: QuantizationExecutionPlan,
    *,
    export_onnx_fn: Any = None,
    quantize_onnx_qdq_static_fn: Any = quantize_onnx_qdq_static,
) -> QuantizationExecutionResult:
    """Execute a normalized quantization plan and return unified reports."""

    if export_onnx_fn is None:
        from xqt.export import export_onnx

        export_onnx_fn = export_onnx
    current_model = context.model if isinstance(context.model, nn.Module) else None
    runtime = {
        "export_onnx_fn": export_onnx_fn,
        "quantize_onnx_qdq_static_fn": quantize_onnx_qdq_static_fn,
        "onnx_qdq_graph_summary_fn": _onnx_qdq_graph_summary,
        "quantize_with_torchao_fn": quantize_with_torchao,
    }
    reports: list[QuantizationReport] = []
    artifacts: dict[str, Any] = {}
    for component in plan.components:
        if component.analysis_only:
            reports.append(_analysis_only_report(component))
            continue
        if component.backend == "tilelang":
            raise ValueError(
                f"Quantization component '{component.name}' uses backend 'tilelang', "
                "but tilelang is an operator engine, not a quant backend. "
                "Use backend='pytorch' with method/strategy for quant, then "
                "operator stage engine='tilelang' for kernel materialize."
            )
        if component.backend == "svdquant":
            raise ValueError(
                f"Quantization component '{component.name}' uses backend 'svdquant', "
                "but svdquant is a quant method, not a quant backend. "
                "Use backend='pytorch' with method='svd' and an SVD strategy."
            )
        if _component_requires_model(component) and current_model is None:
            raise ValueError(
                f"PyTorch model is required for quantization component "
                f"'{component.name}' backend '{component.backend}'"
            )
        registration = resolve_quant_route(
            RouteQuery(
                backend=component.backend,
                method=(component.method or "none").lower(),
                strategy=component.strategy or "",
                compute=component.compute or "dequant_fp16",
                scheme=component.scheme,
            )
        )
        if registration is None:
            current_model, report, component_artifacts = (
                execute_planned_method_component(current_model, component)
            )
        else:
            current_model, report, component_artifacts = registration.handler(
                context,
                current_model,
                component,
                runtime=runtime,
            )
        artifacts.update(component_artifacts)
        reports.append(report)
    return QuantizationExecutionResult(
        model=current_model,
        reports=reports,
        artifacts=artifacts,
    )


__all__ = [
    "execute_quantization_plan",
    "summarize_quantization_reports",
]
