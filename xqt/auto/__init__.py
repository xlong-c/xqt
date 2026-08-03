"""Automatic strategy and tuning helpers for XQT."""

from .acceptance import AcceptanceEvaluation, evaluate_stage_acceptance
from .backend_suggestion import (
    BackendSuggestion,
    QuantBackendSuggestionReport,
    suggest_quant_backends,
)
from .precision_suggestion import (
    PrecisionActionSuggestion,
    PrecisionSuggestionReport,
    suggest_precision_actions,
)
from .stage_history import (
    StageBenchmarkLeaderboard,
    StageBenchmarkRanking,
    rank_stage_benchmark_history,
)
from .strategy_search import (
    SchemeCandidate,
    StrategyAttemptRecord,
    StrategySearchReport,
    build_scheme_search_space,
    plan_hash_for,
    run_scheme_search,
)

__all__ = [
    "AcceptanceEvaluation",
    "BackendSuggestion",
    "PrecisionActionSuggestion",
    "PrecisionSuggestionReport",
    "QuantBackendSuggestionReport",
    "SchemeCandidate",
    "StageBenchmarkLeaderboard",
    "StageBenchmarkRanking",
    "StrategyAttemptRecord",
    "StrategySearchReport",
    "build_scheme_search_space",
    "evaluate_stage_acceptance",
    "plan_hash_for",
    "rank_stage_benchmark_history",
    "run_scheme_search",
    "suggest_precision_actions",
    "suggest_quant_backends",
]
