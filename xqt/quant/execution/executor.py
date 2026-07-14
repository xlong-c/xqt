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
from xqt.quant.component import prefix_module_names
from xqt.quant.quantizers.fp4_weight_only import (
    execute_fp4_weight_only_component,
    quantize_with_fp4_weight_only,
)
from xqt.quant.quantizers.awq_gptq_weight_only import (
    execute_awq_gptq_weight_only_component,
    quantize_with_awq_weight_only,
    quantize_with_gptq_weight_only,
)
from xqt.quant.quantizers.int8_mma import (
    execute_int8_mma_component,
    quantize_with_int8_mma,
)
from xqt.quant.quantizers.w4_storage_int8_mma import (
    execute_w4_storage_int8_mma_component,
    quantize_with_w4_storage_int8_mma,
)
from xqt.quant.quantizers.convrot_4bit import (
    execute_convrot_4bit_component,
    quantize_with_convrot_4bit,
)
from xqt.quant.quantizers.mxfp_weight_only import (
    execute_mxfp_weight_only_component,
    quantize_with_mxfp_weight_only,
)
from xqt.quant.quantizers.fp4_dynamic import (
    execute_dynamic_fp4_component,
    quantize_with_mxfp4_dynamic,
    quantize_with_nvfp4_dynamic,
)
from xqt.quant.selection import selection_policy_metadata
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
                "Use backend='pytorch' with method='svd' and strategy='svd_fp4'/'svd_int4'."
            )
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
            current_model, report = execute_fp4_weight_only_component(
                context,
                current_model,
                component,
                quantize_fn=quantize_with_fp4_weight_only,
            )
            reports.append(report)
            continue
        if (
            component.backend == "pytorch"
            and component.method in {"awq", "gptq"}
            and component.strategy in {"weight_only_int4", "weight_only_int8"}
        ):
            current_model, report = execute_awq_gptq_weight_only_component(
                context,
                current_model,
                component,
            )
            reports.append(report)
            continue
        if (
            component.backend == "pytorch"
            and component.strategy == "mxfp_weight_only"
        ):
            current_model, report = execute_mxfp_weight_only_component(
                context,
                current_model,
                component,
                quantize_fn=quantize_with_mxfp_weight_only,
            )
            reports.append(report)
            continue
        if (
            component.backend == "pytorch"
            and component.strategy == "nvfp4_dynamic"
        ):
            current_model, report = execute_dynamic_fp4_component(
                context,
                current_model,
                component,
                fp4_format="nvfp4",
                quantize_fn=quantize_with_nvfp4_dynamic,
            )
            reports.append(report)
            continue
        if (
            component.backend == "pytorch"
            and component.strategy == "mxfp4_dynamic"
        ):
            current_model, report = execute_dynamic_fp4_component(
                context,
                current_model,
                component,
                fp4_format="mxfp4",
                quantize_fn=quantize_with_mxfp4_dynamic,
            )
            reports.append(report)
            continue
        if (
            component.backend == "pytorch"
            and component.strategy in {"dynamic_int8_mma", "tilelang_int8_mma", "int8_mma"}
        ):
            current_model, report = execute_int8_mma_component(
                context,
                current_model,
                component,
                quantize_fn=quantize_with_int8_mma,
            )
            reports.append(report)
            continue
        if (
            component.backend == "pytorch"
            and component.strategy == "w4_storage_int8_mma"
        ):
            current_model, report = execute_w4_storage_int8_mma_component(
                context,
                current_model,
                component,
                quantize_fn=quantize_with_w4_storage_int8_mma,
            )
            reports.append(report)
            continue
        if (
            component.backend == "pytorch"
            and component.strategy == "convrot_w4a4"
        ):
            current_model, report = execute_convrot_4bit_component(
                context,
                current_model,
                component,
                quantize_fn=quantize_with_convrot_4bit,
            )
            reports.append(report)
            continue
        if component.backend == "pytorch" and (
            component.strategy in {"svd_fp4", "svd_int4"}
            or (component.method or "").lower() in {"svd", "svdquant", "svd_fp4", "svd_int4"}
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
        if component.backend == "pytorch" and component.method in {"awq", "gptq"}:
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
    "quantize_with_awq_weight_only",
    "quantize_with_convrot_4bit",
    "quantize_onnx_qdq_static",
    "quantize_with_fp4_weight_only",
    "quantize_with_gptq_weight_only",
    "quantize_with_int8_mma",
    "quantize_with_mxfp4_dynamic",
    "quantize_with_mxfp_weight_only",
    "quantize_with_nvfp4_dynamic",
    "quantize_with_svd",
    "quantize_with_torchao",
    "quantize_with_w4_storage_int8_mma",
    "summarize_quantization_reports",
]
