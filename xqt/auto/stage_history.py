"""Benchmark-history based stage ranking for XQT optimization sessions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Sequence

from ._helpers import find_nested_numeric, stage_accepted, stage_kind, stage_metrics, stage_name


Verdict = Literal["best", "rejected", "neutral"]


@dataclass(frozen=True)
class StageBenchmarkRanking:
    """One ranked stage entry derived from benchmark history."""

    stage: str
    kind: str
    accepted: bool
    speedup: float | None
    peak_memory_mb: float | None
    mean_abs: float | None
    verdict: Verdict
    reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "kind": self.kind,
            "accepted": self.accepted,
            "speedup": self.speedup,
            "peak_memory_mb": self.peak_memory_mb,
            "mean_abs": self.mean_abs,
            "verdict": self.verdict,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class StageBenchmarkLeaderboard:
    """Best / rejected stage summary derived from benchmark history."""

    best_stage: str | None
    rejected_stages: tuple[str, ...]
    rankings: tuple[StageBenchmarkRanking, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "best_stage": self.best_stage,
            "rejected_stages": list(self.rejected_stages),
            "rankings": [item.to_dict() for item in self.rankings],
        }


def _memory_mb(metrics: dict[str, Any]) -> float | None:
    for key in ("peak_memory_mb", "memory_mb"):
        value = find_nested_numeric(metrics, key)
        if value is not None:
            return value
    for key in ("peak_memory_bytes", "peak_bytes", "cuda_peak_allocated_bytes"):
        value = find_nested_numeric(metrics, key)
        if value is not None:
            return value / (1024.0 * 1024.0)
    return None


def rank_stage_benchmark_history(
    stages: Sequence[Any],
    *,
    metric: str = "speedup",
) -> StageBenchmarkLeaderboard:
    """Rank stages by benchmark history and report best / rejected stages.

    ``stages`` accepts ``OptimizationStageResult``, ``SessionStage``, or plain
    mappings with ``name`` / ``kind`` / ``accepted`` / ``metrics``. The result
    is a pure view; it never mutates session state or reports.
    """

    rankings: list[StageBenchmarkRanking] = []
    for source in stages:
        name = stage_name(source)
        kind = stage_kind(source)
        accepted = stage_accepted(source)
        metrics = stage_metrics(source)
        speedup = find_nested_numeric(metrics, "speedup")
        memory_mb = _memory_mb(metrics)
        mean_abs = find_nested_numeric(metrics, "mean_abs")
        reasons: list[str] = []
        if not accepted:
            reasons.append("stage was rejected by acceptance policy")
        rankings.append(
            StageBenchmarkRanking(
                stage=name,
                kind=kind,
                accepted=accepted,
                speedup=speedup,
                peak_memory_mb=memory_mb,
                mean_abs=mean_abs,
                verdict="rejected" if not accepted else "neutral",
                reasons=tuple(reasons),
            )
        )

    best: StageBenchmarkRanking | None = None
    for ranking in rankings:
        if not ranking.accepted:
            continue
        if metric == "memory":
            score = -ranking.peak_memory_mb if ranking.peak_memory_mb is not None else None
        elif metric == "numeric":
            score = -ranking.mean_abs if ranking.mean_abs is not None else None
        else:
            score = ranking.speedup if ranking.speedup is not None else None
        if score is None:
            continue
        if best is None or score > (best_score := _score_for(best, metric)):
            best = ranking

    if best is not None:
        object.__setattr__(best, "verdict", "best")
        best_reasons = list(best.reasons) + [f"best by {metric}"]
        object.__setattr__(best, "reasons", tuple(best_reasons))

    rejected = tuple(
        ranking.stage for ranking in rankings if ranking.verdict == "rejected"
    )
    ordered = tuple(
        sorted(rankings, key=lambda item: (_rank_verdict(item), -(item.speedup or 0.0)))
    )
    return StageBenchmarkLeaderboard(
        best_stage=best.stage if best is not None else None,
        rejected_stages=rejected,
        rankings=ordered,
    )


def _score_for(ranking: StageBenchmarkRanking, metric: str) -> float:
    if metric == "memory":
        return -ranking.peak_memory_mb if ranking.peak_memory_mb is not None else float("-inf")
    if metric == "numeric":
        return -ranking.mean_abs if ranking.mean_abs is not None else float("-inf")
    return ranking.speedup if ranking.speedup is not None else float("-inf")


def _rank_verdict(ranking: StageBenchmarkRanking) -> int:
    return {"best": 0, "rejected": 2, "neutral": 1}[ranking.verdict]


__all__ = [
    "StageBenchmarkLeaderboard",
    "StageBenchmarkRanking",
    "rank_stage_benchmark_history",
]
