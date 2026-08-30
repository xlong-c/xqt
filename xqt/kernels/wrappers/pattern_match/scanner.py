"""Stable multi-source operator candidate scanner report."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from torch import nn

from .candidate import (
    OperatorPatternCandidate,
    operator_pattern_coverage_report,
    summarize_candidate_report,
)
from .export import scan_export_candidates
from .fx import scan_fx_candidates


Scanner = Callable[[nn.Module, Any], list[OperatorPatternCandidate]]


_SCANNERS: dict[str, Scanner] = {
    "fx": scan_fx_candidates,
    "torch_export": scan_export_candidates,
}


def _empty_scan_report(status: str, error: str | None) -> dict[str, Any]:
    return {
        "status": status,
        "error": error,
        "candidate_count": 0,
        "sources": [],
        "patterns": [],
        "pattern_counts": {},
        "source_counts": {},
        "coverage": operator_pattern_coverage_report([]),
        "recommended_backends": [],
        "shape_signatures": [],
        "devices": [],
        "dtypes": [],
        "candidates": [],
    }


def _run_candidate_scan(
    scanner: Scanner,
    model: nn.Module,
    example_input: Any,
) -> tuple[dict[str, Any], list[OperatorPatternCandidate]]:
    try:
        candidates = scanner(model, example_input)
    except Exception as exc:
        return _empty_scan_report("error", str(exc)), []
    return {
        **summarize_candidate_report(candidates),
        "coverage": operator_pattern_coverage_report(candidates),
        "status": "ok",
        "error": None,
    }, candidates


def scan_candidate_report(
    scanner: Scanner,
    model: nn.Module,
    example_input: Any,
) -> dict[str, Any]:
    """Run one pattern scanner while preserving scanner failure in the report."""

    report, _ = _run_candidate_scan(scanner, model, example_input)
    return report


def scan_operator_candidate_reports(
    model: nn.Module,
    example_input: Any,
    *,
    sources: Iterable[str] = ("fx", "torch_export"),
) -> dict[str, Any]:
    """Scan operator candidates across supported graph sources."""

    source_reports: dict[str, dict[str, Any]] = {}
    candidates: list[OperatorPatternCandidate] = []
    errors: dict[str, str] = {}
    for source in sources:
        scanner = _SCANNERS.get(source)
        if scanner is None:
            source_reports[source] = _empty_scan_report(
                "error",
                f"unknown operator candidate scanner source: {source}",
            )
            errors[source] = str(source_reports[source]["error"])
            continue
        report, scanned_candidates = _run_candidate_scan(scanner, model, example_input)
        source_reports[source] = report
        if report["status"] == "ok":
            candidates.extend(scanned_candidates)
        elif report["error"]:
            errors[source] = str(report["error"])

    summary = summarize_candidate_report(candidates)
    coverage = operator_pattern_coverage_report(candidates)
    status = "ok"
    if errors and candidates:
        status = "partial"
    elif errors:
        status = "error"
    payload: dict[str, Any] = {
        **summary,
        "coverage": coverage,
        "status": status,
        "error": errors or None,
        "source_reports": source_reports,
    }
    payload.update(source_reports)
    return payload


__all__ = [
    "scan_candidate_report",
    "scan_operator_candidate_reports",
]
