"""Report helpers for quantization execution."""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from xqt.core.types import XQTContext
from xqt.quant.calibration.summary import build_calibration_summary
from xqt.quant.component import ordered_unique, prefix_module_names
from xqt.quant.selection import (
    module_selection_reason_metadata,
    selection_policy_metadata,
)
from xqt.quant.types import (
    QuantizationComponentPlan,
    QuantizationNature,
    QuantizationReport,
)


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


def build_component_quantization_report(
    context: XQTContext,
    component: QuantizationComponentPlan,
    *,
    backend: str,
    strategy: str | None,
    quantized_modules: Iterable[str],
    nature: QuantizationNature,
    algorithm_executable: bool | None,
    method_semantics: str | None,
    effective_policy: Mapping[str, Any] | None = None,
    result_metadata: Mapping[str, Any] | None = None,
    extra_metadata: Mapping[str, Any] | None = None,
    execution_state: str | None = None,
    method: str | None = None,
    runtime: str | None = "pytorch",
    additional_high_precision_modules: Iterable[str] = (),
    additional_skipped_modules: Iterable[str] = (),
) -> QuantizationReport:
    """Build the shared component report shape for one quantizer algorithm."""

    high_precision_modules = ordered_unique(
        [
            *prefix_module_names(
                component.keep_high_precision,
                component.target_path,
            ),
            *prefix_module_names(
                additional_high_precision_modules,
                component.target_path,
            ),
        ]
    )
    skipped_modules = ordered_unique(
        [
            *prefix_module_names(component.skip_quantize, component.target_path),
            *prefix_module_names(
                additional_skipped_modules,
                component.target_path,
            ),
            *high_precision_modules,
        ]
    )
    resolved_quantized_modules = prefix_module_names(
        quantized_modules,
        component.target_path,
    )
    calibration_samples, calibration_summary = optional_calibration_summary(
        context,
        component,
    )
    metadata: dict[str, Any] = {
        **dict(result_metadata or {}),
        "analysis_only": component.analysis_only,
        "selection_policy": selection_policy_metadata(component),
        "module_selection_reasons": module_selection_reason_metadata(
            component,
            quantized_modules=resolved_quantized_modules,
            skipped_modules=skipped_modules,
            high_precision_modules=high_precision_modules,
        ),
        "executed": True,
        "algorithm_executable": algorithm_executable,
        "method_semantics": method_semantics,
    }
    if effective_policy is not None:
        metadata["policy"] = dict(effective_policy)
    if execution_state is not None:
        metadata["execution_state"] = execution_state
    if extra_metadata:
        metadata.update(dict(extra_metadata))
    return QuantizationReport(
        component_name=component.name,
        backend=backend,
        runtime=runtime,
        method=component.method if method is None else method,
        strategy=strategy,
        target_path=component.target_path,
        quantized_modules=resolved_quantized_modules,
        skipped_modules=skipped_modules,
        high_precision_modules=high_precision_modules,
        calibration_samples=calibration_samples,
        calibration_summary=calibration_summary,
        nature=nature,
        algorithm_executable=algorithm_executable,
        method_semantics=method_semantics,
        compute_speedup_expected=None,
        metadata=metadata,
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
    "build_component_quantization_report",
    "optional_calibration_summary",
    "summarize_quantization_reports",
]
