"""Capability-driven quant backend suggestions.

This module only produces suggestions. It never rewrites user configuration;
callers decide whether to adopt the suggested backend / strategy combination.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from xqt.quant.capability import (
    QuantBackendCapability,
    describe_quant_backend_capability,
    supported_quant_backends,
)


@dataclass(frozen=True)
class BackendSuggestion:
    """One suggested or excluded quant backend combination."""

    rank: int
    backend: str
    status: str
    maturity: str
    requires_cuda: bool = False
    method: str | None = None
    strategy: str | None = None
    compute: str | None = None
    reasons: tuple[str, ...] = ()
    risks: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "backend": self.backend,
            "status": self.status,
            "maturity": self.maturity,
            "requires_cuda": self.requires_cuda,
            "method": self.method,
            "strategy": self.strategy,
            "compute": self.compute,
            "reasons": list(self.reasons),
            "risks": list(self.risks),
        }


@dataclass(frozen=True)
class QuantBackendSuggestionReport:
    """Ordered backend suggestions plus explicitly excluded combinations."""

    device: str
    explicit_backend: str | None
    explicit_strategy: str | None
    explicit_method: str | None
    suggestions: tuple[BackendSuggestion, ...] = ()
    excluded: tuple[BackendSuggestion, ...] = ()
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "device": self.device,
            "explicit_backend": self.explicit_backend,
            "explicit_strategy": self.explicit_strategy,
            "explicit_method": self.explicit_method,
            "suggestions": [item.to_dict() for item in self.suggestions],
            "excluded": [item.to_dict() for item in self.excluded],
            "summary": self.summary,
        }


def _suggestion_rank(suggestion: BackendSuggestion) -> tuple[int, int, int]:
    """Return a stable sort key: maturity first, then CPU preference."""

    maturity_order = {"executable": 0, "reference_guarded": 1, "metadata_only": 2, "planned": 3}
    return (
        maturity_order.get(suggestion.maturity, 9),
        1 if suggestion.requires_cuda else 0,
        0 if suggestion.status == "available" else 1,
    )


def suggest_quant_backends(
    *,
    device: str = "cpu",
    explicit_backend: str | None = None,
    strategy: str | None = None,
    method: str | None = None,
    constraints: Mapping[str, Any] | None = None,
) -> QuantBackendSuggestionReport:
    """Suggest quant backend combinations from the single capability source.

    ``explicit_backend`` is honored as the first suggestion when the capability
    is available; otherwise it is listed in ``excluded`` with reasons. The
    function never mutates any config object.
    """

    device_key = str(device).strip().lower() or "cpu"
    requires_cuda = device_key not in {"cpu", "tpu", "xpu", "mps"}
    constraints = dict(constraints or {})
    suggested_strategy = strategy
    suggested_method = method

    suggestions: list[BackendSuggestion] = []
    excluded: list[BackendSuggestion] = []
    rank_counter = 1

    for backend in supported_quant_backends():
        try:
            capability = describe_quant_backend_capability(
                backend,
                method=suggested_method,
                strategy=suggested_strategy,
                compute=constraints.get("compute"),
                policy=constraints.get("policy"),
            )
        except ValueError as exc:
            excluded.append(
                BackendSuggestion(
                    rank=0,
                    backend=backend,
                    status="unsupported",
                    maturity="planned",
                    requires_cuda=False,
                    method=suggested_method,
                    strategy=suggested_strategy,
                    reasons=(str(exc),),
                )
            )
            continue

        reasons: list[str] = []
        risks: list[str] = list(capability.limitations)
        if capability.requires_cuda and not requires_cuda:
            excluded.append(
                BackendSuggestion(
                    rank=0,
                    backend=backend,
                    status=capability.status,
                    maturity=capability.maturity,
                    requires_cuda=capability.requires_cuda,
                    method=suggested_method,
                    strategy=suggested_strategy,
                    compute=constraints.get("compute"),
                    reasons=("requires CUDA but the target device does not expose it",),
                    risks=risks,
                )
            )
            continue
        if capability.status != "available":
            excluded.append(
                BackendSuggestion(
                    rank=0,
                    backend=backend,
                    status=capability.status,
                    maturity=capability.maturity,
                    requires_cuda=capability.requires_cuda,
                    method=suggested_method,
                    strategy=suggested_strategy,
                    compute=constraints.get("compute"),
                    reasons=(f"status is {capability.status}, not available",),
                    risks=risks,
                )
            )
            continue

        if explicit_backend is not None and backend != explicit_backend:
            excluded.append(
                BackendSuggestion(
                    rank=0,
                    backend=backend,
                    status=capability.status,
                    maturity=capability.maturity,
                    requires_cuda=capability.requires_cuda,
                    method=suggested_method,
                    strategy=suggested_strategy,
                    compute=constraints.get("compute"),
                    reasons=(
                        f"explicit backend {explicit_backend!r} takes precedence",
                    ),
                    risks=risks,
                )
            )
            continue

        if backend == explicit_backend:
            reasons.append("matches the explicitly requested backend")
        if suggested_strategy is not None and suggested_strategy in capability.storage_strategies:
            reasons.append("requested strategy is supported by this backend")
        if suggested_method is not None and suggested_method in capability.methods:
            reasons.append("requested method is supported by this backend")
        reasons.extend(capability.notes)
        suggestions.append(
            BackendSuggestion(
                rank=rank_counter,
                backend=backend,
                status=capability.status,
                maturity=capability.maturity,
                requires_cuda=capability.requires_cuda,
                method=suggested_method,
                strategy=suggested_strategy,
                compute=constraints.get("compute"),
                reasons=tuple(reasons),
                risks=tuple(risks),
            )
        )
        rank_counter += 1

    suggestions.sort(key=_suggestion_rank)
    for index, item in enumerate(suggestions):
        object.__setattr__(item, "rank", index + 1)

    summary = (
        f"recommended {len(suggestions)} backend candidate(s) for device={device_key}; "
        f"excluded {len(excluded)} candidate(s). "
        "Suggestions never rewrite explicit user configuration."
    )
    return QuantBackendSuggestionReport(
        device=device_key,
        explicit_backend=explicit_backend,
        explicit_strategy=strategy,
        explicit_method=method,
        suggestions=tuple(suggestions),
        excluded=tuple(excluded),
        summary=summary,
    )
__all__ = [
    "BackendSuggestion",
    "QuantBackendSuggestionReport",
    "suggest_quant_backends",
]
