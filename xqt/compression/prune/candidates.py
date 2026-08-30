"""Internal candidate discovery helpers for structured pruning."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, Sequence

import torch
from torch import nn

from .graph import PruningDependencyGraph
from .report import PruningTarget


@dataclass
class _StructuredCandidate:
    adapter: str
    structure_family: str
    action_type: str
    module_name: str
    module_type: str
    granularity: str
    dependency_group: str
    consumer_name: Optional[str]
    consumer_type: Optional[str]
    normalization_name: Optional[str]
    feature_block_size: int
    scores: torch.Tensor
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class _CandidateDiscoveryResult:
    candidates: list[_StructuredCandidate] = field(default_factory=list)
    dependency_graph: PruningDependencyGraph = field(default_factory=PruningDependencyGraph)
    blocked_modules: list[str] = field(default_factory=list)
    protected_modules: list[str] = field(default_factory=list)


class _CandidateCollector(Protocol):
    def __call__(
        self,
        model: nn.Module,
        *,
        importance_metric: str,
    ) -> _CandidateDiscoveryResult: ...


@dataclass(frozen=True)
class _StructuredPruningAdapter:
    name: str
    granularity: str
    structure_family: str
    collect: _CandidateCollector


@dataclass(frozen=True)
class _PruneBatch:
    candidate_index: int
    unit_indices: tuple[int, ...]
    score: float


def _candidate_to_target(candidate: _StructuredCandidate) -> PruningTarget:
    return PruningTarget(
        module_name=candidate.module_name,
        module_type=candidate.module_type,
        granularity=candidate.granularity,
        group_size=int(candidate.scores.numel()),
        dependency_group=candidate.dependency_group,
        adapter=candidate.adapter,
        structure_family=candidate.structure_family,
        metadata=dict(candidate.metadata),
    )


def collect_candidates(
    model: nn.Module,
    *,
    granularity: str,
    importance_metric: str,
    adapters: Sequence[_StructuredPruningAdapter],
    supported_granularities: Sequence[str],
) -> _CandidateDiscoveryResult:
    """Collect and merge all candidates for one structured pruning granularity."""

    matching_adapters = [adapter for adapter in adapters if adapter.granularity == granularity]
    if not matching_adapters:
        allowed = ", ".join(supported_granularities)
        raise ValueError(f"Unsupported structured granularity '{granularity}'. Allowed: {allowed}")

    combined = _CandidateDiscoveryResult()
    for adapter in matching_adapters:
        result = adapter.collect(model, importance_metric=importance_metric)
        combined.candidates.extend(result.candidates)
        combined.blocked_modules.extend(result.blocked_modules)
        combined.dependency_graph.groups.extend(result.dependency_graph.groups)
    return combined

