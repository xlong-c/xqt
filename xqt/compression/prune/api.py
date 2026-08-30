"""Public structured-pruning operations."""

from __future__ import annotations

from functools import partial
from typing import Any, Mapping, Optional

from torch import nn

from xqt.contracts import ModelStructureContract

from .candidates import _candidate_to_target
from .discovery import (
    SUPPORTED_GRANULARITIES,
    SUPPORTED_IMPORTANCE_METRICS,
    SUPPORTED_SCOPES,
    collect_structured_candidates,
    validate_action_keep_indices,
)
from .graph import validate_candidate_dependencies
from .plan import build_structured_pruning_plan
from .report import PruningTarget, StructuredPruningPlan, StructuredPruningReport
from .rewrite import apply_structured_pruning_plan, topology_changes_from_actions


def find_structured_pruning_targets(
    model: nn.Module,
    *,
    granularity: str = "channel",
    importance: Optional[Mapping[str, Any]] = None,
    structure_contract: Optional[ModelStructureContract] = None,
) -> list[PruningTarget]:
    """Enumerate supported structured-pruning targets for one granularity."""

    importance_config = dict(importance or {})
    importance_metric = str(
        importance_config.get("metric", importance_config.get("type", "l1"))
    )
    discovery = collect_structured_candidates(
        model,
        granularity=granularity,
        importance_metric=importance_metric,
        structure_contract=structure_contract,
    )
    validate_candidate_dependencies(discovery.candidates, discovery.dependency_graph)
    return [_candidate_to_target(candidate) for candidate in discovery.candidates]


def plan_structured_pruning(
    model: nn.Module,
    target_sparsity: float,
    *,
    granularity: str = "channel",
    scope: str = "global",
    importance: Optional[Mapping[str, Any]] = None,
    selection: Optional[Mapping[str, Any]] = None,
    structure_contract: Optional[ModelStructureContract] = None,
) -> StructuredPruningPlan:
    """Build a validated structured-pruning plan for a supported model family."""

    return build_structured_pruning_plan(
        model,
        target_sparsity,
        granularity=granularity,
        scope=scope,
        importance=importance,
        selection=selection,
        collect_candidates=(
            collect_structured_candidates
            if structure_contract is None
            else partial(collect_structured_candidates, structure_contract=structure_contract)
        ),
        supported_granularities=SUPPORTED_GRANULARITIES,
        supported_scopes=SUPPORTED_SCOPES,
        supported_importance_metrics=SUPPORTED_IMPORTANCE_METRICS,
        validate_action_keep_indices=validate_action_keep_indices,
        topology_changes_from_actions=topology_changes_from_actions,
    )


def apply_structured_pruning(
    model: nn.Module,
    target_sparsity: float,
    *,
    granularity: str = "channel",
    scope: str = "global",
    importance: Optional[Mapping[str, Any]] = None,
    selection: Optional[Mapping[str, Any]] = None,
    example_input: Any = None,
    task_type: Optional[str] = None,
    structure_contract: Optional[ModelStructureContract] = None,
) -> StructuredPruningReport:
    """Plan and apply a structured-pruning rewrite in place."""

    plan = plan_structured_pruning(
        model,
        target_sparsity,
        granularity=granularity,
        scope=scope,
        importance=importance,
        selection=selection,
        structure_contract=structure_contract,
    )
    return apply_structured_pruning_plan(
        model,
        plan,
        example_input=example_input,
        task_type=task_type,
    )


__all__ = [
    "apply_structured_pruning",
    "apply_structured_pruning_plan",
    "find_structured_pruning_targets",
    "plan_structured_pruning",
]
