"""Report helpers for quantization execution."""

from __future__ import annotations

from typing import Any, Sequence

from xqt.core.types import XQTContext
from xqt.quant.calibration.summary import build_calibration_summary
from xqt.quant.types import QuantizationComponentPlan, QuantizationReport


def optional_calibration_summary(
    context: XQTContext,
    component: QuantizationComponentPlan,
) -> tuple[int | None, dict[str, Any] | None]:
    """Build a lightweight calibration input summary when inputs are available."""

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
            "nature": first.nature.value,
            "algorithm_executable": first.algorithm_executable,
            "method_semantics": first.method_semantics,
            "metadata": dict(first.metadata),
        }
    )
    if "path" in first.metadata:
        metrics["path"] = first.metadata["path"]
    if "checksum" in first.metadata:
        metrics["checksum"] = first.metadata["checksum"]
    return metrics


__all__ = [
    "optional_calibration_summary",
    "summarize_quantization_reports",
]
