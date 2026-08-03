"""Operator optimization reporting helpers."""

from __future__ import annotations

from typing import Any

from .execution_support import ordered_unique
from .types import OperatorOptimizationReport


def operator_acceptance_record(report: OperatorOptimizationReport) -> dict[str, Any]:
    """Return the acceptance decision facts for one operator target."""

    metadata = dict(report.metadata)
    numeric_diff = dict(report.numeric_diff or {})
    fallback_detail = metadata.get("fallback_detail")
    if not isinstance(fallback_detail, dict):
        fallback_detail = {}
    graph_break_report = metadata.get("graph_break_report")
    if not isinstance(graph_break_report, dict):
        graph_break_report = {}
    numeric_validation = metadata.get("numeric_validation")
    if not isinstance(numeric_validation, dict):
        numeric_validation = {}
    min_speedup = metadata.get("min_speedup")
    effective_min_speedup = metadata.get("effective_min_speedup", min_speedup)
    meets_numeric = numeric_validation.get("allclose", numeric_diff.get("allclose"))
    meets_speedup = None
    if isinstance(report.speedup, (float, int)) and isinstance(
        effective_min_speedup,
        (float, int),
    ):
        meets_speedup = float(report.speedup) >= float(effective_min_speedup)

    decision = "applied" if report.applied else "fallback"
    if not report.applied:
        if meets_numeric is False:
            decision = "rejected_numeric"
        elif meets_speedup is False or "min_speedup" in str(report.skip_reason or ""):
            decision = "rejected_min_speedup"
        elif metadata.get("execution_state") == "skipped":
            decision = "skipped"

    return {
        "target_name": report.target_name,
        "applied": report.applied,
        "decision": decision,
        "reason": report.skip_reason,
        "speedup": report.speedup,
        "min_speedup": min_speedup,
        "effective_min_speedup": effective_min_speedup,
        "meets_speedup": meets_speedup,
        "meets_numeric": meets_numeric,
        "numeric_validation_status": numeric_validation.get("status"),
        "numeric_validation_reason": numeric_validation.get("reason"),
        "fallback": report.fallback,
        "fallback_policy": report.fallback_policy,
        "fallback_reason": metadata.get("fallback_reason"),
        "graph_break_count": fallback_detail.get(
            "graph_break_count",
            graph_break_report.get("graph_break_count"),
        ),
        "graph_breaks": list(
            fallback_detail.get(
                "graph_breaks",
                graph_break_report.get("break_reasons", []),
            )
            or []
        ),
    }


def summarize_operator_optimization_reports(
    reports: list[OperatorOptimizationReport],
    *,
    candidate_reports: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a unified metrics payload from operator optimization reports."""

    items = [report.to_dict() for report in reports]
    acceptance = [operator_acceptance_record(report) for report in reports]
    applied = [report for report in reports if report.applied]
    skipped = [report for report in reports if not report.applied]
    summary = {
        "target_count": len(reports),
        "applied_count": len(applied),
        "skipped_count": len(skipped),
        "targets": items,
        "acceptance": acceptance,
        "engines": ordered_unique(report.engine for report in reports),
        "runtimes": ordered_unique(report.runtime for report in reports),
        "candidate_kinds": ordered_unique(
            report.candidate_kind for report in reports
        ),
        "candidate_layers": ordered_unique(
            str(report.metadata.get("candidate_layer"))
            for report in reports
            if report.metadata.get("candidate_layer") is not None
        ),
        "benchmark_targets": ordered_unique(
            report.benchmark_target_path for report in reports
        ),
        "fallback_policies": ordered_unique(report.fallback_policy for report in reports),
        "fallback_reasons": ordered_unique(
            report.skip_reason for report in reports if report.skip_reason
        ),
        "min_speedup_rejections": [
            item["target_name"]
            for item in acceptance
            if item["decision"] == "rejected_min_speedup"
        ],
        "numeric_rejections": [
            item["target_name"]
            for item in acceptance
            if item["decision"] == "rejected_numeric"
        ],
        "numeric_validation_failures": [
            item["target_name"]
            for item in acceptance
            if item.get("numeric_validation_status") == "failed"
        ],
        "graph_break_targets": [
            item["target_name"]
            for item in acceptance
            if item["graph_break_count"] not in {None, 0}
        ],
        "applied_targets": [report.target_name for report in applied],
        "skipped_targets": [report.target_name for report in skipped],
    }
    if candidate_reports is not None:
        summary["candidates"] = dict(candidate_reports)
    return summary
