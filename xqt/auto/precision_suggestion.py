"""Layer-sensitivity based precision action suggestions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Sequence

from xqt.compression.quant.sensitivity import LayerAnalysisRecord


PrecisionAction = Literal["keep_high_precision", "skip_quantize", "quantize"]


@dataclass(frozen=True)
class PrecisionActionSuggestion:
    """One per-module precision suggestion derived from layer analysis."""

    module_name: str
    module_type: str
    action: PrecisionAction
    mean_abs: float
    max_abs: float
    relative_error: float | None
    cosine: float | None
    reason: str
    tags: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "module_name": self.module_name,
            "module_type": self.module_type,
            "action": self.action,
            "mean_abs": self.mean_abs,
            "max_abs": self.max_abs,
            "relative_error": self.relative_error,
            "cosine": self.cosine,
            "reason": self.reason,
            "tags": list(self.tags),
        }


@dataclass(frozen=True)
class PrecisionSuggestionReport:
    """Structured precision actions plus an explicit-config protection note."""

    suggestions: tuple[PrecisionActionSuggestion, ...] = ()
    protected_modules: tuple[str, ...] = ()
    policy_delta: dict[str, list[str]] = field(default_factory=dict)
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "suggestions": [item.to_dict() for item in self.suggestions],
            "protected_modules": list(self.protected_modules),
            "policy_delta": {key: list(value) for key, value in self.policy_delta.items()},
            "summary": self.summary,
        }

def _diff_metric(record: LayerAnalysisRecord, key: str) -> float | None:
    diff = record.diff
    value = getattr(diff, key, None)
    return float(value) if isinstance(value, (float, int)) else None


def suggest_precision_actions(
    records: Sequence[LayerAnalysisRecord],
    *,
    max_mean_abs: float | None = None,
    max_max_abs: float | None = None,
    max_relative_error: float | None = None,
    min_cosine: float | None = None,
    explicit_keep_high_precision: Sequence[str] = (),
    explicit_skip_quantize: Sequence[str] = (),
) -> PrecisionSuggestionReport:
    """Suggest per-module precision actions without mutating any policy.

    Modules listed in the explicit keep/skip sets are protected: the report
    never suggests a conflicting action for them and records them separately.
    """

    explicit_keep = set(explicit_keep_high_precision)
    explicit_skip = set(explicit_skip_quantize)
    suggestions: list[PrecisionActionSuggestion] = []
    keep_delta: list[str] = []
    skip_delta: list[str] = []
    protected: list[str] = []

    for record in records:
        mean_abs = _diff_metric(record, "mean_abs") or 0.0
        max_abs = _diff_metric(record, "max_abs") or 0.0
        relative_error = _diff_metric(record, "relative_error")
        cosine = _diff_metric(record, "cosine")

        if record.name in explicit_keep or record.name in explicit_skip:
            protected.append(record.name)
            continue

        reason: str | None = None
        tags = list(record.tags)
        if max_mean_abs is not None and mean_abs > max_mean_abs:
            action: PrecisionAction = "keep_high_precision"
            reason = f"mean_abs {mean_abs:.6g} exceeds threshold {max_mean_abs:g}"
            keep_delta.append(record.name)
        elif max_max_abs is not None and max_abs > max_max_abs:
            action = "keep_high_precision"
            reason = f"max_abs {max_abs:.6g} exceeds threshold {max_max_abs:g}"
            keep_delta.append(record.name)
        elif max_relative_error is not None and relative_error is not None and relative_error > max_relative_error:
            action = "skip_quantize"
            reason = (
                f"relative_error {relative_error:.6g} exceeds threshold "
                f"{max_relative_error:g}"
            )
            skip_delta.append(record.name)
        elif min_cosine is not None and cosine is not None and cosine < min_cosine:
            action = "skip_quantize"
            reason = f"cosine {cosine:.6g} is below threshold {min_cosine:g}"
            skip_delta.append(record.name)
        else:
            action = "quantize"
            reason = "layer drift stays within configured tolerance"

        if reason is None:
            continue
        suggestions.append(
            PrecisionActionSuggestion(
                module_name=record.name,
                module_type=record.module_type,
                action=action,
                mean_abs=mean_abs,
                max_abs=max_abs,
                relative_error=relative_error,
                cosine=cosine,
                reason=reason,
                tags=tuple(tags),
            )
        )

    summary = (
        f"suggested {len(suggestions)} precision action(s); "
        f"{len(protected)} module(s) protected by explicit configuration."
    )
    return PrecisionSuggestionReport(
        suggestions=tuple(suggestions),
        protected_modules=tuple(sorted(protected)),
        policy_delta={"keep_high_precision": keep_delta, "skip_quantize": skip_delta},
        summary=summary,
    )


__all__ = [
    "PrecisionActionSuggestion",
    "PrecisionSuggestionReport",
    "suggest_precision_actions",
]
