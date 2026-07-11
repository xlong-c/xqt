"""FX and torch.export candidate pattern discovery for operator optimization."""

from .pattern_match.candidate import (
    OperatorPatternCandidate,
    summarize_candidate_report,
)
from .pattern_match.export import scan_export_candidates
from .pattern_match.fx import scan_fx_candidates

__all__ = [
    "OperatorPatternCandidate",
    "scan_export_candidates",
    "scan_fx_candidates",
    "summarize_candidate_report",
]
