"""Canonical pruning method metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PruneMethodSpec:
    method: str
    family: str
    default_granularity: str
    rewrites_structure: bool
    pattern_present: bool
    baseline_kind: str | None
    speedup_claimed: bool
    speedup_verified: bool
    report_note: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "family": self.family,
            "default_granularity": self.default_granularity,
            "rewrites_structure": self.rewrites_structure,
            "pattern_present": self.pattern_present,
            "baseline_kind": self.baseline_kind,
            "speedup_claimed": self.speedup_claimed,
            "speedup_verified": self.speedup_verified,
            "report_note": self.report_note,
        }


_PRUNE_METHODS: dict[str, PruneMethodSpec] = {
    "global_l1_unstructured": PruneMethodSpec(
        method="global_l1_unstructured",
        family="unstructured",
        default_granularity="parameter",
        rewrites_structure=False,
        pattern_present=False,
        baseline_kind="unstructured_sparsity_report",
        speedup_claimed=False,
        speedup_verified=False,
        report_note=(
            "Unstructured pruning is a sparsity baseline; it does not imply latency "
            "speedup without a sparse runtime."
        ),
    ),
    "structured": PruneMethodSpec(
        method="structured",
        family="structured",
        default_granularity="channel",
        rewrites_structure=True,
        pattern_present=False,
        baseline_kind=None,
        speedup_claimed=False,
        speedup_verified=False,
        report_note=(
            "Structured pruning rewrites model topology when an adapter supports the "
            "requested granularity; speedup still requires benchmark evidence."
        ),
    ),
    "nm_structured": PruneMethodSpec(
        method="nm_structured",
        family="semi_structured",
        default_granularity="nm",
        rewrites_structure=False,
        pattern_present=True,
        baseline_kind=None,
        speedup_claimed=False,
        speedup_verified=False,
        report_note=(
            "N:M pruning records pattern compliance; hardware speedup is only a "
            "candidate until a sparse runtime benchmark verifies it."
        ),
    ),
    "block_sparse": PruneMethodSpec(
        method="block_sparse",
        family="block_sparse",
        default_granularity="block_sparse",
        rewrites_structure=False,
        pattern_present=True,
        baseline_kind=None,
        speedup_claimed=False,
        speedup_verified=False,
        report_note=(
            "Block-sparse pruning records block masks and sparsity; no generic XQT "
            "sparse runtime speedup is claimed."
        ),
    ),
}


SUPPORTED_PRUNE_METHODS: tuple[str, ...] = tuple(_PRUNE_METHODS)


def describe_prune_method(method: str) -> PruneMethodSpec:
    normalized = str(method).strip().lower()
    spec = _PRUNE_METHODS.get(normalized)
    if spec is None:
        allowed = ", ".join(SUPPORTED_PRUNE_METHODS)
        raise ValueError(f"Unsupported prune method: {method}. Allowed: {allowed}")
    return spec


def prune_method_report_fields(method: str) -> dict[str, Any]:
    spec = describe_prune_method(method)
    return {
        "method_family": spec.family,
        "rewrites_structure": spec.rewrites_structure,
        "pattern_present": spec.pattern_present,
        "baseline_kind": spec.baseline_kind,
        "speedup_claimed": spec.speedup_claimed,
        "speedup_verified": spec.speedup_verified,
        "report_note": spec.report_note,
    }


def supported_prune_method_specs() -> list[dict[str, Any]]:
    return [spec.to_dict() for spec in _PRUNE_METHODS.values()]


__all__ = [
    "PruneMethodSpec",
    "SUPPORTED_PRUNE_METHODS",
    "describe_prune_method",
    "prune_method_report_fields",
    "supported_prune_method_specs",
]
