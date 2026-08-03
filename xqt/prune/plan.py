"""Structured pruning plan assembly helpers."""

from __future__ import annotations

import math
from typing import Any, Mapping, Optional, Protocol

from torch import nn

from .candidates import (
    _CandidateDiscoveryResult,
    _PruneBatch,
    _StructuredCandidate,
    _candidate_to_target,
)
from .graph import validate_candidate_dependencies
from .granularity import describe_prune_granularity, normalize_prune_granularity
from .report import StructuredPruningAction, StructuredPruningPlan


class _CollectCandidatesFn(Protocol):
    def __call__(
        self,
        model: nn.Module,
        *,
        granularity: str,
        importance_metric: str,
    ) -> _CandidateDiscoveryResult: ...


class _ValidateActionKeepIndicesFn(Protocol):
    def __call__(
        self,
        model: nn.Module,
        action: StructuredPruningAction,
    ) -> None: ...


class _TopologyChangesFromActionsFn(Protocol):
    def __call__(
        self,
        actions: list[StructuredPruningAction],
    ) -> list[dict[str, Any]]: ...


def _group_alignment_constraints(
    candidate: _StructuredCandidate,
) -> list[Mapping[str, Any]]:
    raw = candidate.metadata.get("group_alignment_constraints", [])
    if not isinstance(raw, list):
        raise ValueError("metadata.group_alignment_constraints must be a list")
    constraints: list[Mapping[str, Any]] = []
    for item in raw:
        if not isinstance(item, Mapping):
            raise ValueError("metadata.group_alignment_constraints entries must be mappings")
        constraints.append(item)
    return constraints


def _candidate_prune_batch_groups(candidate: _StructuredCandidate) -> list[list[int]]:
    unit_count = int(candidate.scores.numel())
    constraints = _group_alignment_constraints(candidate)
    if not constraints:
        return [[index] for index in range(unit_count)]

    group_counts: set[int] = set()
    channels_per_group_values: set[int] = set()
    for constraint in constraints:
        total_channels = int(constraint["total_channels"])
        groups = int(constraint["groups"])
        channels_per_group = int(constraint["channels_per_group"])
        if total_channels != unit_count:
            raise ValueError(
                f"group alignment constraint for '{candidate.module_name}' has "
                f"total_channels={total_channels}, expected {unit_count}"
            )
        group_counts.add(groups)
        channels_per_group_values.add(channels_per_group)
    if len(group_counts) != 1 or len(channels_per_group_values) != 1:
        raise ValueError(
            f"group alignment constraints for '{candidate.module_name}' are incompatible"
        )

    groups = group_counts.pop()
    channels_per_group = channels_per_group_values.pop()
    return [
        [group_index * channels_per_group + local_index for group_index in range(groups)]
        for local_index in range(channels_per_group)
    ]


def _candidate_min_keep_units(
    candidate: _StructuredCandidate,
    *,
    min_keep: int,
) -> int:
    unit_count = int(candidate.scores.numel())
    constraints = _group_alignment_constraints(candidate)
    if not constraints:
        return min(unit_count, min_keep)
    groups = int(constraints[0]["groups"])
    return min(unit_count, max(groups, math.ceil(min_keep / groups) * groups))


def _candidate_prune_batches(
    candidate: _StructuredCandidate,
    *,
    candidate_index: int,
) -> list[_PruneBatch]:
    batches: list[_PruneBatch] = []
    for unit_indices in _candidate_prune_batch_groups(candidate):
        score = float(candidate.scores[list(unit_indices)].sum().item())
        batches.append(
            _PruneBatch(
                candidate_index=candidate_index,
                unit_indices=tuple(int(index) for index in unit_indices),
                score=score,
            )
        )
    return batches


def _build_keep_indices_global(
    candidates: list[_StructuredCandidate],
    *,
    target_sparsity: float,
    min_keep: int,
) -> list[list[int]]:
    total_units = sum(int(candidate.scores.numel()) for candidate in candidates)
    min_keep_units = [
        _candidate_min_keep_units(candidate, min_keep=min_keep)
        for candidate in candidates
    ]
    max_prunable = sum(
        max(int(candidate.scores.numel()) - min_keep_units[index], 0)
        for index, candidate in enumerate(candidates)
    )
    target_pruned = min(int(round(total_units * target_sparsity)), max_prunable)

    keep_masks: list[list[bool]] = [
        [True] * int(candidate.scores.numel())
        for candidate in candidates
    ]
    keep_counts = [int(candidate.scores.numel()) for candidate in candidates]
    ranked_batches: list[_PruneBatch] = []
    for candidate_index, candidate in enumerate(candidates):
        ranked_batches.extend(
            _candidate_prune_batches(candidate, candidate_index=candidate_index)
        )
    ranked_batches.sort(key=lambda item: item.score)

    pruned = 0
    for batch in ranked_batches:
        candidate_index = batch.candidate_index
        unit_indices = list(batch.unit_indices)
        if pruned >= target_pruned:
            break
        if keep_counts[candidate_index] - len(unit_indices) < min_keep_units[candidate_index]:
            continue
        if pruned + len(unit_indices) > target_pruned:
            continue
        if any(not keep_masks[candidate_index][unit_index] for unit_index in unit_indices):
            continue
        for unit_index in unit_indices:
            keep_masks[candidate_index][unit_index] = False
        keep_counts[candidate_index] -= len(unit_indices)
        pruned += len(unit_indices)

    return [
        [unit_index for unit_index, keep in enumerate(mask) if keep]
        for mask in keep_masks
    ]


def _build_keep_indices_per_layer(
    candidates: list[_StructuredCandidate],
    *,
    target_sparsity: float,
    min_keep: int,
) -> list[list[int]]:
    keep_indices: list[list[int]] = []
    for candidate in candidates:
        unit_count = int(candidate.scores.numel())
        min_keep_units = _candidate_min_keep_units(candidate, min_keep=min_keep)
        target_prune_count = min(
            int(round(unit_count * target_sparsity)),
            max(unit_count - min_keep_units, 0),
        )
        ranked_batches = _candidate_prune_batches(candidate, candidate_index=0)
        ranked_batches.sort(key=lambda item: item.score)
        prune_set: set[int] = set()
        pruned = 0
        for batch in ranked_batches:
            if pruned >= target_prune_count:
                break
            if pruned + len(batch.unit_indices) > target_prune_count:
                continue
            prune_set.update(batch.unit_indices)
            pruned += len(batch.unit_indices)
        keep_indices.append(
            [index for index in range(unit_count) if index not in prune_set]
        )
    return keep_indices


def _selection_keep_indices_map(
    selection: Mapping[str, Any],
) -> dict[str, list[int]]:
    raw = selection.get("keep_indices")
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ValueError("selection.keep_indices must be a mapping of module_name -> indices")
    keep_indices_map: dict[str, list[int]] = {}
    for module_name, indices in raw.items():
        if not isinstance(indices, (list, tuple)):
            raise ValueError("selection.keep_indices values must be lists or tuples")
        keep: list[int] = []
        seen: set[int] = set()
        for value in indices:
            index = int(value)
            if index in seen:
                raise ValueError(
                    f"selection.keep_indices for '{module_name}' contains duplicate index {index}"
                )
            keep.append(index)
            seen.add(index)
        if not keep:
            raise ValueError(f"selection.keep_indices for '{module_name}' must not be empty")
        keep_indices_map[str(module_name)] = keep
    return keep_indices_map


def _keep_indices_from_selection(
    candidates: list[_StructuredCandidate],
    *,
    keep_indices_map: Mapping[str, list[int]],
) -> list[list[int]]:
    keep_by_candidate: list[list[int]] = []
    candidate_names = {candidate.module_name for candidate in candidates}
    unknown_names = sorted(set(keep_indices_map) - candidate_names)
    if unknown_names:
        raise ValueError(
            "selection.keep_indices references unknown candidate(s): "
            + ", ".join(unknown_names)
        )
    for candidate in candidates:
        unit_count = int(candidate.scores.numel())
        if candidate.module_name not in keep_indices_map:
            keep_by_candidate.append(list(range(unit_count)))
            continue
        keep = sorted(int(index) for index in keep_indices_map[candidate.module_name])
        if keep[0] < 0 or keep[-1] >= unit_count:
            if (
                candidate.action_type == "attention_heads"
                and str(candidate.metadata.get("attention_variant", "mha")) in {"gqa", "mqa"}
            ):
                raise ValueError(
                    f"selection.keep_indices for '{candidate.module_name}' must reference kv heads"
                )
            raise ValueError(
                f"selection.keep_indices for '{candidate.module_name}' is out of range"
            )
        if len(set(keep)) != len(keep):
            raise ValueError(
                f"selection.keep_indices for '{candidate.module_name}' must be unique"
            )
        keep_by_candidate.append(keep)
    return keep_by_candidate


def _build_structured_pruning_actions(
    model: nn.Module,
    *,
    candidates: list[_StructuredCandidate],
    keep_by_candidate: list[list[int]],
    granularity: str,
    validate_action_keep_indices: _ValidateActionKeepIndicesFn,
) -> list[StructuredPruningAction]:
    actions: list[StructuredPruningAction] = []
    for candidate, keep_indices in zip(candidates, keep_by_candidate):
        original_units = int(candidate.scores.numel())
        keep = sorted(int(index) for index in keep_indices)
        keep_set = set(keep)
        prune = [index for index in range(original_units) if index not in keep_set]
        action = StructuredPruningAction(
            action_type=candidate.action_type,
            module_name=candidate.module_name,
            module_type=candidate.module_type,
            granularity=granularity,
            original_units=original_units,
            keep_indices=keep,
            prune_indices=prune,
            consumer_name=candidate.consumer_name,
            consumer_type=candidate.consumer_type,
            dependency_group=candidate.dependency_group,
            adapter=candidate.adapter,
            structure_family=candidate.structure_family,
            normalization_name=candidate.normalization_name,
            feature_block_size=candidate.feature_block_size,
            score_min=float(candidate.scores.min().item()),
            score_max=float(candidate.scores.max().item()),
            score_mean=float(candidate.scores.mean().item()),
            metadata=dict(candidate.metadata),
        )
        validate_action_keep_indices(model, action)
        actions.append(action)
    return actions


def build_structured_pruning_plan(
    model: nn.Module,
    target_sparsity: float,
    *,
    granularity: str = "channel",
    scope: str = "global",
    importance: Optional[Mapping[str, Any]] = None,
    selection: Optional[Mapping[str, Any]] = None,
    collect_candidates: _CollectCandidatesFn,
    supported_granularities: tuple[str, ...],
    supported_scopes: tuple[str, ...],
    supported_importance_metrics: tuple[str, ...],
    validate_action_keep_indices: _ValidateActionKeepIndicesFn,
    topology_changes_from_actions: _TopologyChangesFromActionsFn,
) -> StructuredPruningPlan:
    """Build a validated structured pruning plan from discovered candidates."""

    if target_sparsity < 0.0 or target_sparsity > 1.0:
        raise ValueError("target_sparsity must be in [0, 1]")
    canonical_granularity = normalize_prune_granularity(granularity)
    if canonical_granularity not in supported_granularities:
        spec = describe_prune_granularity(canonical_granularity)
        if not spec["rewrites_structure"]:
            raise ValueError(
                f"Structured prune granularity '{canonical_granularity}' is "
                f"{spec['runtime_support']} in XQT: it records metadata but has no "
                "generic structural rewrite, so structured pruning cannot proceed"
            )
        allowed = ", ".join(supported_granularities)
        raise ValueError(
            f"Unsupported structured granularity '{canonical_granularity}'. Allowed: {allowed}"
        )
    if scope not in supported_scopes:
        allowed = ", ".join(supported_scopes)
        raise ValueError(f"Unsupported structured scope '{scope}'. Allowed: {allowed}")

    importance_config = dict(importance or {})
    selection_config = dict(selection or {})
    importance_metric = str(
        importance_config.get("metric", importance_config.get("type", "l1"))
    )
    min_keep = int(selection_config.get("min_keep", 1))
    if min_keep <= 0:
        raise ValueError("selection.min_keep must be positive")
    if importance_metric not in supported_importance_metrics:
        allowed = ", ".join(supported_importance_metrics)
        raise ValueError(
            f"Unsupported structured importance metric '{importance_metric}'. Allowed: {allowed}"
        )
    if importance_metric == "bn_gamma" and canonical_granularity not in {
        "channel",
        "filter",
    }:
        raise ValueError("importance.metric=bn_gamma only supports channel/filter pruning")
    if importance_metric == "usage" and canonical_granularity != "expert":
        raise ValueError("importance.metric=usage only supports expert pruning")

    discovery = collect_candidates(
        model,
        granularity=canonical_granularity,
        importance_metric=importance_metric,
    )
    candidates = discovery.candidates
    validate_candidate_dependencies(candidates, discovery.dependency_graph)
    if not candidates:
        raise ValueError(
            f"No supported {granularity} structured pruning candidates were found in the model"
        )
    targets = [_candidate_to_target(candidate) for candidate in candidates]
    keep_indices_map = _selection_keep_indices_map(selection_config)

    if keep_indices_map:
        keep_by_candidate = _keep_indices_from_selection(
            candidates,
            keep_indices_map=keep_indices_map,
        )
    elif scope == "global":
        keep_by_candidate = _build_keep_indices_global(
            candidates,
            target_sparsity=target_sparsity,
            min_keep=min_keep,
        )
    else:
        keep_by_candidate = _build_keep_indices_per_layer(
            candidates,
            target_sparsity=target_sparsity,
            min_keep=min_keep,
        )

    actions = _build_structured_pruning_actions(
        model,
        candidates=candidates,
        keep_by_candidate=keep_by_candidate,
        granularity=canonical_granularity,
        validate_action_keep_indices=validate_action_keep_indices,
    )

    return StructuredPruningPlan(
        method="structured",
        granularity=canonical_granularity,
        scope=scope,
        target_sparsity=target_sparsity,
        importance_metric=importance_metric,
        adapters=sorted({candidate.adapter for candidate in candidates}),
        structure_families=sorted({candidate.structure_family for candidate in candidates}),
        blocked_modules=sorted(set(discovery.blocked_modules)),
        dependency_graph=discovery.dependency_graph.to_dict(),
        topology_changes=topology_changes_from_actions(actions),
        targets=targets,
        actions=actions,
    )
