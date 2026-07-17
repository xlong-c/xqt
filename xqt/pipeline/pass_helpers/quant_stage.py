"""Quant stage implementation helpers."""

from __future__ import annotations

from typing import Any, Callable

import torch
from torch import nn

from xqt.analysis import build_layer_analysis_payload, layer_statistics_rows
from xqt.core.artifact import ArtifactRecord, MetricRecord
from xqt.core.inputs import extract_model_inputs, infer_model_input_count
from xqt.core.schema import AnalysisConfig, OutputDiffConfig, QuantConfig
from xqt.core.types import XQTContext
from xqt.export import export_onnx
from xqt.quant import (
    build_quantization_plan,
    execute_quantization_plan,
    summarize_quantization_reports,
)
from xqt.quant.backends.onnx_qdq import quantize_onnx_qdq_static
from xqt.workflows.stage_specs import QuantStageSpec

from .context import (
    _context_analysis_config,
    _context_output_diff_config,
    _move_to_device,
    _quant_runtime_config,
)


def _build_quant_layer_analysis_summary(
    context: XQTContext,
    *,
    analysis_config: AnalysisConfig | None = None,
    output_diff: OutputDiffConfig | None = None,
    build_layer_analysis_payload_fn: Callable[..., dict[str, Any]] = (
        build_layer_analysis_payload
    ),
) -> dict[str, object]:
    """Return layer diff and sensitivity summary for a quantized PyTorch model."""

    if context.reference_model is None:
        return {"available": False, "reason": "reference_model_missing"}
    if not isinstance(context.model, nn.Module):
        return {"available": False, "reason": "quantized_pytorch_model_missing"}
    if context.example_inputs is None:
        return {"available": False, "reason": "example_inputs_missing"}

    resolved_analysis = analysis_config or _context_analysis_config(context)
    resolved_output_diff = output_diff or _context_output_diff_config(context)
    device = torch.device(context.device)
    candidate_model = context.model.to(device).eval()
    reference_model = context.reference_model.to(device).eval()
    example_input = _move_to_device(
        extract_model_inputs(
            context.example_inputs,
            expected_input_count=infer_model_input_count(candidate_model),
        ),
        device,
    )
    try:
        payload = build_layer_analysis_payload_fn(
            reference_model,
            candidate_model,
            example_input,
            module_names=resolved_analysis.module_names,
            atol=resolved_output_diff.atol,
            rtol=resolved_output_diff.rtol,
            include_weight_diff=resolved_analysis.include_weight_diff,
            include_sensitivity=True,
            include_statistics=False,
            include_avoid_list=True,
            metrics=resolved_analysis.metrics,
            row_top_k=resolved_analysis.top_k,
            avoid_top_k=resolved_analysis.top_k,
            sample_budget=resolved_analysis.sample_budget,
            sample_seed=resolved_analysis.sample_seed,
            runtime="quantized_pytorch",
            avoid_used_by="quant_keep_high_precision",
            per_channel=resolved_analysis.structured.per_channel,
            per_token=resolved_analysis.structured.per_token,
        )
        module_names = list(resolved_analysis.module_names or [])
        if not module_names:
            module_names = [
                name
                for name, module in candidate_model.named_modules()
                if name and hasattr(module, "weight")
            ]
        payload["layer_statistics"] = layer_statistics_rows(
            reference_model,
            candidate_model,
            example_input,
            module_names=module_names,
            sample_budget=resolved_analysis.sample_budget,
            sample_seed=resolved_analysis.sample_seed,
        )
        payload["available"] = True
        payload["layer_error_count"] = len(payload.get("layer_errors", []))
        payload["layer_sensitivity_count"] = len(payload.get("layer_sensitivity", []))
        payload["layer_statistics_count"] = len(payload.get("layer_statistics", []))
        if not payload["layer_errors"] and payload["layer_statistics"]:
            payload["layer_error_rows_from_statistics"] = True
        return payload
    except Exception as exc:
        return {
            "available": False,
            "reason": "analysis_failed",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "layer_errors": [],
            "layer_sensitivity": [],
            "layer_statistics": [],
            "avoid_list": [],
            "layer_error_count": 0,
            "layer_sensitivity_count": 0,
            "layer_statistics_count": 0,
        }


def _run_quant_with_resolved_config(
    context: XQTContext,
    resolved_quant: QuantConfig | QuantStageSpec,
    *,
    build_layer_analysis_payload_fn: Callable[..., dict[str, Any]] = (
        build_layer_analysis_payload
    ),
) -> XQTContext:
    context.quant_config = _quant_runtime_config(resolved_quant)
    enabled = resolved_quant.enabled if isinstance(resolved_quant, QuantConfig) else True
    if not enabled:
        return context
    plan = build_quantization_plan(resolved_quant)
    execution = execute_quantization_plan(
        context,
        plan,
        export_onnx_fn=export_onnx,
        quantize_onnx_qdq_static_fn=quantize_onnx_qdq_static,
    )
    context.model = execution.model
    context.artifacts.update(execution.artifacts)
    quant_metrics = summarize_quantization_reports(execution.reports)
    quant_metrics["layer_analysis"] = _build_quant_layer_analysis_summary(
        context,
        build_layer_analysis_payload_fn=build_layer_analysis_payload_fn,
    )
    context.metrics["quant"] = quant_metrics
    if context.manifest is not None:
        layer_analysis = quant_metrics["layer_analysis"]
        context.manifest.add_metric(
            MetricRecord(
                name="quant.layer_analysis.available",
                value=bool(layer_analysis.get("available")),
                passed=bool(layer_analysis.get("available")),
                metadata={
                    "reason": layer_analysis.get("reason"),
                    "layer_error_count": layer_analysis.get("layer_error_count", 0),
                    "layer_sensitivity_count": layer_analysis.get(
                        "layer_sensitivity_count",
                        0,
                    ),
                },
            )
        )
        for report in execution.reports:
            if report.runtime == "onnxruntime" and "path" in report.metadata:
                context.manifest.add_artifact(
                    ArtifactRecord(
                        path=str(report.metadata["path"]),
                        format="onnx",
                        runtime="onnxruntime",
                        checksum=report.metadata.get("checksum"),
                        metadata={
                            "quantization": "qdq",
                            "component_name": report.component_name,
                            **report.metadata,
                        },
                    )
                )
            context.manifest.add_metric(
                MetricRecord(
                    name=f"quant.{report.component_name}.quantized_module_count",
                    value=len(report.quantized_modules),
                    metadata={
                        "backend": report.backend,
                        "strategy": report.strategy,
                    },
                )
            )
            if report.calibration_samples is not None:
                context.manifest.add_metric(
                    MetricRecord(
                        name=f"quant.{report.component_name}.calibration_samples",
                        value=report.calibration_samples,
                        metadata={
                            "backend": report.backend,
                            "calibration_source": report.metadata.get("calibration_source"),
                        },
                    )
                )
    return context
