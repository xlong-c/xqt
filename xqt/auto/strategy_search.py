"""Safe strategy search over orthogonal QuantScheme candidates.

The search space is defined on ``QuantScheme`` (not the mixed-axis strategy
enum). The runner limits attempts, records every attempt, and keeps the whole
run reproducible through a seed and a plan hash. It only returns suggestions;
it never rewrites the caller's configuration.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Mapping, Sequence

from xqt.compression.quant.strategy import resolve_scheme, strategy_scheme_templates
from xqt.compression.quant.types import QuantScheme

from ._helpers import find_nested_numeric


AttemptStatus = Literal["ok", "failed", "skipped"]


@dataclass(frozen=True)
class SchemeCandidate:
    """One deterministic search candidate."""

    index: int
    strategy: str | None
    scheme: QuantScheme

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "strategy": self.strategy,
            "scheme": {
                "weight_dtype": self.scheme.weight_dtype,
                "weight_granularity": self.scheme.weight_granularity,
                "group_size": self.scheme.group_size,
                "activation_dtype": self.scheme.activation_dtype,
                "activation_mode": self.scheme.activation_mode,
                "sym": self.scheme.sym,
            },
        }


@dataclass(frozen=True)
class StrategyAttemptRecord:
    """One recorded search attempt."""

    attempt_index: int
    strategy: str | None
    scheme: QuantScheme
    status: AttemptStatus
    message: str
    metrics: dict[str, Any] = field(default_factory=dict)
    error_class: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempt_index": self.attempt_index,
            "strategy": self.strategy,
            "scheme": {
                "weight_dtype": self.scheme.weight_dtype,
                "weight_granularity": self.scheme.weight_granularity,
                "group_size": self.scheme.group_size,
                "activation_dtype": self.scheme.activation_dtype,
                "activation_mode": self.scheme.activation_mode,
                "sym": self.scheme.sym,
            },
            "status": self.status,
            "message": self.message,
            "metrics": dict(self.metrics),
            "error_class": self.error_class,
        }


@dataclass(frozen=True)
class StrategySearchReport:
    """Reproducible search report with attempts and recommendations."""

    seed: int
    plan_hash: str
    max_attempts: int
    attempts: tuple[StrategyAttemptRecord, ...]
    recommended: tuple[SchemeCandidate, ...]
    failures: tuple[StrategyAttemptRecord, ...]
    reproducible: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "plan_hash": self.plan_hash,
            "max_attempts": self.max_attempts,
            "attempts": [item.to_dict() for item in self.attempts],
            "recommended": [item.to_dict() for item in self.recommended],
            "failures": [item.to_dict() for item in self.failures],
            "reproducible": self.reproducible,
        }


def build_scheme_search_space(
    *,
    include_strategies: Sequence[str] | None = None,
    exclude_strategies: Sequence[str] | None = None,
    max_candidates: int | None = None,
    seed: int = 0,
) -> tuple[SchemeCandidate, ...]:
    """Build a deterministic QuantScheme search space from the strategy fact source.

    Candidate ordering is fixed by the canonical strategy order plus ``seed``
    as a stable tie-breaker suffix, so the same inputs always produce the same
    space.
    """

    templates = strategy_scheme_templates()
    names = list(templates)
    if include_strategies is not None:
        requested = [str(name) for name in include_strategies]
        unknown = sorted(set(requested) - set(names))
        if unknown:
            raise ValueError(f"unknown strategies: {', '.join(unknown)}")
        names = requested
    if exclude_strategies is not None:
        excluded = {str(name) for name in exclude_strategies}
        names = [name for name in names if name not in excluded]
    if max_candidates is not None and max_candidates > 0:
        names = names[:max_candidates]

    candidates: list[SchemeCandidate] = []
    for index, name in enumerate(names):
        scheme = resolve_scheme(name)
        candidates.append(
            SchemeCandidate(
                index=index,
                strategy=name,
                scheme=scheme,
            )
        )
    return tuple(candidates)


def plan_hash_for(
    candidates: Sequence[SchemeCandidate],
    *,
    seed: int,
) -> str:
    """Return a stable hash describing the search plan and seed."""

    payload = json.dumps(
        {
            "seed": seed,
            "candidates": [candidate.to_dict() for candidate in candidates],
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def run_scheme_search(
    quantize_fn: Callable[[str | None, QuantScheme], Mapping[str, Any]],
    *,
    include_strategies: Sequence[str] | None = None,
    exclude_strategies: Sequence[str] | None = None,
    max_attempts: int = 8,
    seed: int = 0,
    accept_fn: Callable[[Mapping[str, Any]], bool] | None = None,
) -> StrategySearchReport:
    """Run a bounded, recorded, reproducible scheme search.

    ``quantize_fn(strategy, scheme)`` is invoked once per candidate and must
    return a metrics mapping (for example ``mean_abs`` / ``speedup``). Failed
    attempts are recorded, not silently dropped. The report recommends at most
    ``max_attempts`` candidates, ordered by numeric drift then speedup.
    """

    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    candidates = build_scheme_search_space(
        include_strategies=include_strategies,
        exclude_strategies=exclude_strategies,
        seed=seed,
    )
    plan_hash = plan_hash_for(candidates, seed=seed)
    attempts: list[StrategyAttemptRecord] = []

    for candidate in candidates[:max_attempts]:
        try:
            raw = quantize_fn(candidate.strategy, candidate.scheme)
        except Exception as exc:  # record every failure for reproducibility
            attempts.append(
                StrategyAttemptRecord(
                    attempt_index=len(attempts),
                    strategy=candidate.strategy,
                    scheme=candidate.scheme,
                    status="failed",
                    message=str(exc),
                    error_class=type(exc).__name__,
                )
            )
            continue
        metrics = dict(raw)
        if accept_fn is not None and not accept_fn(metrics):
            attempts.append(
                StrategyAttemptRecord(
                    attempt_index=len(attempts),
                    strategy=candidate.strategy,
                    scheme=candidate.scheme,
                    status="skipped",
                    message="rejected by acceptance callback",
                    metrics=metrics,
                )
            )
            continue
        attempts.append(
            StrategyAttemptRecord(
                attempt_index=len(attempts),
                strategy=candidate.strategy,
                scheme=candidate.scheme,
                status="ok",
                message="attempt completed",
                metrics=metrics,
            )
        )

    ok_attempts = [attempt for attempt in attempts if attempt.status == "ok"]
    ok_attempts.sort(
        key=lambda attempt: (
            find_nested_numeric(attempt.metrics, "mean_abs") or float("inf"),
            -(find_nested_numeric(attempt.metrics, "speedup") or 0.0),
        )
    )
    recommended = tuple(
        SchemeCandidate(
            index=attempt.attempt_index,
            strategy=attempt.strategy,
            scheme=attempt.scheme,
        )
        for attempt in ok_attempts[:max_attempts]
    )
    failures = tuple(attempt for attempt in attempts if attempt.status == "failed")
    return StrategySearchReport(
        seed=seed,
        plan_hash=plan_hash,
        max_attempts=max_attempts,
        attempts=tuple(attempts),
        recommended=recommended,
        failures=failures,
        reproducible=True,
    )


__all__ = [
    "SchemeCandidate",
    "StrategyAttemptRecord",
    "StrategySearchReport",
    "build_scheme_search_space",
    "plan_hash_for",
    "run_scheme_search",
]
