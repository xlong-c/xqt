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
from xqt.quant.quantizers.convrot_int8 import (
    execute_convrot_int8_component,
    quantize_with_convrot_int8,
)
from xqt.quant.quantizers.turboquant import (
    execute_turboquant_component,
    quantize_with_turboquant,
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
                "Use backend='pytorch' with method='svd' and an SVD strategy."
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
        method = (component.method or "none").lower()
        strategy = component.strategy or ""
        compute = component.compute or "dequant_fp16"

        if component.backend == "pytorch" and method == "svd":
            current_model, report = execute_svdquant_component(
                context,
                current_model,
                component,
                quantize_fn=quantize_with_svd,
            )
            reports.append(report)
            continue
        if component.backend == "pytorch" and method == "convrot":
            if strategy in {"w8a8_int8", "convrot_w8a8"} or compute == "w8a8_int8_mma":
                current_model, report = execute_convrot_int8_component(
                    context,
                    current_model,
                    component,
                    quantize_fn=quantize_with_convrot_int8,
                )
            else:
                current_model, report = execute_convrot_4bit_component(
                    context,
                    current_model,
                    component,
                    quantize_fn=quantize_with_convrot_4bit,
                )
            reports.append(report)
            continue
        if component.backend == "pytorch" and method == "turboquant":
            current_model, report = execute_turboquant_component(
                context,
                current_model,
                component,
                quantize_fn=quantize_with_turboquant,
            )
            reports.append(report)
            continue
        if (
            component.backend == "pytorch"
            and method in {"awq", "gptq"}
            and strategy in {"w4a16_int4", "w8a16_int8", "w4a16_fp4"}
        ):
            if strategy == "w4a16_fp4":
                current_model, report = execute_fp4_weight_only_component(
                    context,
                    current_model,
                    component,
                    quantize_fn=quantize_with_fp4_weight_only,
                )
            else:
                current_model, report = execute_awq_gptq_weight_only_component(
                    context,
                    current_model,
                    component,
                )
            reports.append(report)
            continue
        if component.backend == "pytorch" and strategy == "w4a16_fp4":
            current_model, report = execute_fp4_weight_only_component(
                context,
                current_model,
                component,
                quantize_fn=quantize_with_fp4_weight_only,
            )
            reports.append(report)
            continue
        if component.backend == "pytorch" and strategy in {
            "w4a16_mxfp4",
            "w8a16_mxfp8",
        }:
            current_model, report = execute_mxfp_weight_only_component(
                context,
                current_model,
                component,
                quantize_fn=quantize_with_mxfp_weight_only,
            )
            reports.append(report)
            continue
        if component.backend == "pytorch" and strategy == "w4a4_nvfp4":
            current_model, report = execute_dynamic_fp4_component(
                context,
                current_model,
                component,
                fp4_format="nvfp4",
                quantize_fn=quantize_with_nvfp4_dynamic,
            )
            reports.append(report)
            continue
        if component.backend == "pytorch" and strategy == "w4a4_mxfp4":
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
            and strategy == "w8a8_int8"
            and compute == "w8a8_int8_mma"
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
            and strategy in {"w4a16_int4", "w4a16_fp4"}
            and compute == "w8a8_int8_mma"
            and method == "none"
        ):
            current_model, report = execute_w4_storage_int8_mma_component(
                context,
                current_model,
                component,
                quantize_fn=quantize_with_w4_storage_int8_mma,
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
        if component.backend == "pytorch" and method in {"awq", "gptq"}:
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
        raise ValueError(
            f"Unsupported quantization route: backend={component.backend!r} "
            f"method={method!r} strategy={strategy!r} compute={compute!r}"
        )
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
    "quantize_with_convrot_int8",
    "quantize_onnx_qdq_static",
    "quantize_with_fp4_weight_only",
    "quantize_with_gptq_weight_only",
    "quantize_with_int8_mma",
    "quantize_with_mxfp4_dynamic",
    "quantize_with_mxfp_weight_only",
    "quantize_with_nvfp4_dynamic",
    "quantize_with_svd",
    "quantize_with_torchao",
    "quantize_with_turboquant",
    "quantize_with_w4_storage_int8_mma",
    "summarize_quantization_reports",
]
