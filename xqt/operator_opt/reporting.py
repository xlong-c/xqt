"""Operator optimization reporting helpers."""

from __future__ import annotations

from typing import Any

from .execution_support import ordered_unique
from .types import OperatorOptimizationReport


def summarize_operator_optimization_reports(
    reports: list[OperatorOptimizationReport],
    *,
    candidate_reports: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a unified metrics payload from operator optimization reports."""

    items = [report.to_dict() for report in reports]
    applied = [report for report in reports if report.applied]
    skipped = [report for report in reports if not report.applied]
    summary = {
        "target_count": len(reports),
        "applied_count": len(applied),
        "skipped_count": len(skipped),
        "targets": items,
        "engines": ordered_unique(report.engine for report in reports),
        "runtimes": ordered_unique(report.runtime for report in reports),
        "fallback_policies": ordered_unique(report.fallback_policy for report in reports),
        "fallback_reasons": ordered_unique(
            report.skip_reason for report in reports if report.skip_reason
        ),
        "applied_targets": [report.target_name for report in applied],
        "skipped_targets": [report.target_name for report in skipped],
    }
    if candidate_reports is not None:
        summary["candidates"] = dict(candidate_reports)
    return summary
