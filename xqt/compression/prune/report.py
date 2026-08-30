"""Structured pruning plan and report dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from .methods import prune_method_report_fields


@dataclass
class PruningTarget:
    """One discovered structured pruning target."""

    module_name: str
    module_type: str
    granularity: str
    group_size: int
    dependency_group: str
    adapter: Optional[str] = None
    structure_family: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "module_name": self.module_name,
            "module_type": self.module_type,
            "granularity": self.granularity,
            "group_size": self.group_size,
            "dependency_group": self.dependency_group,
            "adapter": self.adapter,
            "structure_family": self.structure_family,
            "metadata": dict(self.metadata),
        }


@dataclass
class StructuredPruningAction:
    """One structured pruning action rooted at a producer module."""

    action_type: str
    module_name: str
    module_type: str
    granularity: str
    original_units: int
    keep_indices: list[int]
    prune_indices: list[int]
    consumer_name: Optional[str]
    consumer_type: Optional[str]
    dependency_group: Optional[str] = None
    adapter: Optional[str] = None
    structure_family: Optional[str] = None
    normalization_name: Optional[str] = None
    feature_block_size: int = 1
    score_min: float = 0.0
    score_max: float = 0.0
    score_mean: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def kept_units(self) -> int:
        return len(self.keep_indices)

    @property
    def pruned_units(self) -> int:
        return len(self.prune_indices)

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_type": self.action_type,
            "module_name": self.module_name,
            "module_type": self.module_type,
            "granularity": self.granularity,
            "original_units": self.original_units,
            "kept_units": self.kept_units,
            "pruned_units": self.pruned_units,
            "keep_indices": list(self.keep_indices),
            "prune_indices": list(self.prune_indices),
            "consumer_name": self.consumer_name,
            "consumer_type": self.consumer_type,
            "dependency_group": self.dependency_group,
            "adapter": self.adapter,
            "structure_family": self.structure_family,
            "normalization_name": self.normalization_name,
            "feature_block_size": self.feature_block_size,
            "score_min": self.score_min,
            "score_max": self.score_max,
            "score_mean": self.score_mean,
            "metadata": dict(self.metadata),
        }


@dataclass
class StructuredPruningPlan:
    """Plan describing the selected structured pruning actions."""

    method: str
    granularity: str
    scope: str
    target_sparsity: float
    importance_metric: str
    adapters: list[str] = field(default_factory=list)
    structure_families: list[str] = field(default_factory=list)
    blocked_modules: list[str] = field(default_factory=list)
    dependency_graph: dict[str, Any] = field(default_factory=dict)
    topology_changes: list[dict[str, Any]] = field(default_factory=list)
    targets: list[PruningTarget] = field(default_factory=list)
    actions: list[StructuredPruningAction] = field(default_factory=list)

    @property
    def total_units(self) -> int:
        return sum(action.original_units for action in self.actions)

    @property
    def pruned_units(self) -> int:
        return sum(action.pruned_units for action in self.actions)

    @property
    def kept_units(self) -> int:
        return sum(action.kept_units for action in self.actions)

    @property
    def sparsity(self) -> float:
        if self.total_units == 0:
            return 0.0
        return self.pruned_units / self.total_units

    def to_dict(self) -> dict[str, Any]:
        return {
            **prune_method_report_fields(self.method),
            "method": self.method,
            "granularity": self.granularity,
            "scope": self.scope,
            "target_sparsity": self.target_sparsity,
            "importance_metric": self.importance_metric,
            "adapters": list(self.adapters),
            "structure_families": list(self.structure_families),
            "blocked_modules": list(self.blocked_modules),
            "dependency_graph": dict(self.dependency_graph),
            "topology_changes": [dict(item) for item in self.topology_changes],
            "targets": [target.to_dict() for target in self.targets],
            "total_units": self.total_units,
            "kept_units": self.kept_units,
            "pruned_units": self.pruned_units,
            "sparsity": self.sparsity,
            "actions": [action.to_dict() for action in self.actions],
        }


@dataclass
class StructuredPruningReport:
    """Aggregate report for a structured pruning rewrite."""

    method: str
    granularity: str
    scope: str
    target_sparsity: float
    importance_metric: str
    parameter_count_before: int
    parameter_count_after: int
    forward_checked: bool
    adapters: list[str] = field(default_factory=list)
    structure_families: list[str] = field(default_factory=list)
    blocked_modules: list[str] = field(default_factory=list)
    dependency_graph: dict[str, Any] = field(default_factory=dict)
    topology_changes: list[dict[str, Any]] = field(default_factory=list)
    export_status: dict[str, Any] = field(default_factory=dict)
    benchmark_status: dict[str, Any] = field(default_factory=dict)
    removed_modules: list[dict[str, Any]] = field(default_factory=list)
    changed_dimensions: list[dict[str, Any]] = field(default_factory=list)
    mask_only_modules: list[str] = field(default_factory=list)
    flops_before: Optional[float] = None
    flops_after: Optional[float] = None
    flops_reduction_ratio: Optional[float] = None
    safety: dict[str, Any] = field(default_factory=dict)
    forward_diff: dict[str, Any] = field(default_factory=dict)
    export_readiness: dict[str, Any] = field(default_factory=dict)
    targets: list[PruningTarget] = field(default_factory=list)
    actions: list[StructuredPruningAction] = field(default_factory=list)

    @property
    def total_units(self) -> int:
        return sum(action.original_units for action in self.actions)

    @property
    def pruned_units(self) -> int:
        return sum(action.pruned_units for action in self.actions)

    @property
    def kept_units(self) -> int:
        return sum(action.kept_units for action in self.actions)

    @property
    def sparsity(self) -> float:
        if self.total_units == 0:
            return 0.0
        return self.pruned_units / self.total_units

    @property
    def parameter_reduction(self) -> int:
        return self.parameter_count_before - self.parameter_count_after

    @property
    def parameter_reduction_ratio(self) -> float:
        if self.parameter_count_before == 0:
            return 0.0
        return self.parameter_reduction / self.parameter_count_before

    def to_dict(self) -> dict[str, Any]:
        return {
            **prune_method_report_fields(self.method),
            "method": self.method,
            "granularity": self.granularity,
            "scope": self.scope,
            "target_sparsity": self.target_sparsity,
            "importance_metric": self.importance_metric,
            "adapters": list(self.adapters),
            "structure_families": list(self.structure_families),
            "blocked_modules": list(self.blocked_modules),
            "dependency_graph": dict(self.dependency_graph),
            "topology_changes": [dict(item) for item in self.topology_changes],
            "export_status": dict(self.export_status),
            "benchmark_status": dict(self.benchmark_status),
            "removed_modules": [dict(item) for item in self.removed_modules],
            "changed_dimensions": [dict(item) for item in self.changed_dimensions],
            "mask_only_modules": list(self.mask_only_modules),
            "flops_before": self.flops_before,
            "flops_after": self.flops_after,
            "flops_reduction_ratio": self.flops_reduction_ratio,
            "safety": dict(self.safety),
            "forward_diff": dict(self.forward_diff),
            "export_readiness": dict(self.export_readiness),
            "parameter_count_before": self.parameter_count_before,
            "parameter_count_after": self.parameter_count_after,
            "parameter_reduction": self.parameter_reduction,
            "parameter_reduction_ratio": self.parameter_reduction_ratio,
            "targets": [target.to_dict() for target in self.targets],
            "total_units": self.total_units,
            "kept_units": self.kept_units,
            "pruned_units": self.pruned_units,
            "sparsity": self.sparsity,
            "forward_checked": self.forward_checked,
            "actions": [action.to_dict() for action in self.actions],
        }


@dataclass
class NMStructuredLayerReport:
    """Per-module summary for N:M structured sparsity."""

    module_name: str
    module_type: str
    parameter_name: str
    total_parameters: int
    zero_parameters: int
    sparsity: float
    pattern_n: int
    pattern_m: int
    compliant_groups: int
    total_groups: int
    compliance_ratio: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "module_name": self.module_name,
            "module_type": self.module_type,
            "parameter_name": self.parameter_name,
            "total_parameters": self.total_parameters,
            "zero_parameters": self.zero_parameters,
            "sparsity": self.sparsity,
            "pattern_n": self.pattern_n,
            "pattern_m": self.pattern_m,
            "compliant_groups": self.compliant_groups,
            "total_groups": self.total_groups,
            "compliance_ratio": self.compliance_ratio,
        }


@dataclass
class NMStructuredPruningReport:
    """Aggregate report for N:M structured sparsity."""

    method: str
    granularity: str
    parameter_count_before: int
    parameter_count_after: int
    zero_parameters_before: int
    zero_parameters_after: int
    pattern_n: int
    pattern_m: int
    module_types: list[str]
    layers: list[NMStructuredLayerReport] = field(default_factory=list)
    mask_only_modules: list[str] = field(default_factory=list)

    @property
    def total_parameters(self) -> int:
        return self.parameter_count_after

    @property
    def sparsity(self) -> float:
        if self.parameter_count_after == 0:
            return 0.0
        return self.zero_parameters_after / self.parameter_count_after

    @property
    def compliance_ratio(self) -> float:
        total_groups = sum(layer.total_groups for layer in self.layers)
        if total_groups == 0:
            return 0.0
        return sum(layer.compliant_groups for layer in self.layers) / total_groups

    def to_dict(self) -> dict[str, Any]:
        return {
            **prune_method_report_fields(self.method),
            "method": self.method,
            "granularity": self.granularity,
            "parameter_count_before": self.parameter_count_before,
            "parameter_count_after": self.parameter_count_after,
            "zero_parameters_before": self.zero_parameters_before,
            "zero_parameters_after": self.zero_parameters_after,
            "pattern_n": self.pattern_n,
            "pattern_m": self.pattern_m,
            "module_types": list(self.module_types),
            "sparsity": self.sparsity,
            "compliance_ratio": self.compliance_ratio,
            "layers": [layer.to_dict() for layer in self.layers],
            "mask_only_modules": list(self.mask_only_modules),
        }


@dataclass
class BlockSparseLayerReport:
    """Per-module summary for block-sparse pruning."""

    module_name: str
    module_type: str
    parameter_name: str
    block_shape: tuple[int, int]
    total_blocks: int
    zero_blocks: int
    pruned_blocks: int
    block_sparsity: float
    total_parameters: int
    zero_parameters: int
    parameter_sparsity: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "module_name": self.module_name,
            "module_type": self.module_type,
            "parameter_name": self.parameter_name,
            "block_shape": list(self.block_shape),
            "total_blocks": self.total_blocks,
            "zero_blocks": self.zero_blocks,
            "pruned_blocks": self.pruned_blocks,
            "block_sparsity": self.block_sparsity,
            "total_parameters": self.total_parameters,
            "zero_parameters": self.zero_parameters,
            "parameter_sparsity": self.parameter_sparsity,
        }


@dataclass
class BlockSparsePruningReport:
    """Aggregate report for block-sparse structured pruning."""

    method: str
    granularity: str
    target_sparsity: float
    block_shape: tuple[int, int]
    parameter_count_before: int
    parameter_count_after: int
    zero_parameters_before: int
    zero_parameters_after: int
    module_types: list[str]
    layers: list[BlockSparseLayerReport] = field(default_factory=list)
    mask_only_modules: list[str] = field(default_factory=list)

    @property
    def sparsity(self) -> float:
        total_blocks = sum(layer.total_blocks for layer in self.layers)
        if total_blocks == 0:
            return 0.0
        return sum(layer.zero_blocks for layer in self.layers) / total_blocks

    @property
    def parameter_sparsity(self) -> float:
        if self.parameter_count_after == 0:
            return 0.0
        return self.zero_parameters_after / self.parameter_count_after

    def to_dict(self) -> dict[str, Any]:
        return {
            **prune_method_report_fields(self.method),
            "method": self.method,
            "granularity": self.granularity,
            "target_sparsity": self.target_sparsity,
            "block_shape": list(self.block_shape),
            "parameter_count_before": self.parameter_count_before,
            "parameter_count_after": self.parameter_count_after,
            "zero_parameters_before": self.zero_parameters_before,
            "zero_parameters_after": self.zero_parameters_after,
            "module_types": list(self.module_types),
            "sparsity": self.sparsity,
            "parameter_sparsity": self.parameter_sparsity,
            "layers": [layer.to_dict() for layer in self.layers],
            "mask_only_modules": list(self.mask_only_modules),
        }


__all__ = [
    "BlockSparseLayerReport",
    "BlockSparsePruningReport",
    "NMStructuredLayerReport",
    "NMStructuredPruningReport",
    "PruningTarget",
    "StructuredPruningAction",
    "StructuredPruningPlan",
    "StructuredPruningReport",
]
