"""FX and torch.export candidate pattern discovery for operator optimization."""

from .pattern_match.candidate import (
    OperatorPatternCandidate,
    operator_pattern_coverage_report,
    summarize_candidate_report,
)
from .pattern_match.export import scan_export_candidates
from .pattern_match.fx import scan_fx_candidates
from .pattern_match.scanner import scan_candidate_report, scan_operator_candidate_reports

__all__ = [
    "OperatorPatternCandidate",
    "operator_pattern_coverage_report",
    "scan_candidate_report",
    "scan_export_candidates",
    "scan_fx_candidates",
    "scan_operator_candidate_reports",
    "summarize_candidate_report",
]
