"""Pruning helpers for XQT."""

from .capability import (
    PruneRuntimeCapability,
    describe_prune_runtime_capability,
    prune_runtime_capability_from_report,
)
from .support_matrix import PruneGranularitySupport, structured_prune_support_matrix
from .granularity import (
    PruneGranularitySpec,
    describe_prune_granularity,
    normalize_prune_granularity,
    rewrite_supported_granularities,
    supported_prune_granularities,
)
from .methods import (
    PruneMethodSpec,
    SUPPORTED_PRUNE_METHODS,
    describe_prune_method,
    prune_method_report_fields,
    supported_prune_method_specs,
)
from .graph import DependencyGroup, PruningDependencyGraph
from .masks import (
    PruningEntry,
    PruningReport,
    apply_global_l1_unstructured_pruning,
    remove_pruning_reparameterization,
    summarize_pruning,
    tensor_sparsity,
)
from .importance import (
    DEFAULT_PRUNABLE_TYPES,
    ModuleImportanceRecord,
    PruneCandidateRecord,
    collect_module_importance,
    rank_prune_candidates,
)
from .rewrite import (
    prune_batchnorm_channels,
    prune_conv2d_in_channels,
    prune_conv2d_out_channels,
    prune_linear_in_features,
    prune_linear_out_features,
    validate_conv2d_keep_indices,
)
from .schedule import (
    PruneScheduleReport,
    PruneScheduleStepReport,
    PruningSchedule,
    StructuredPruneScheduleReport,
    StructuredPruneScheduleStepReport,
    run_prune_schedule,
    run_structured_prune_schedule,
)
from .report import (
    BlockSparseLayerReport,
    BlockSparsePruningReport,
    NMStructuredLayerReport,
    NMStructuredPruningReport,
    PruningTarget,
    StructuredPruningAction,
    StructuredPruningPlan,
    StructuredPruningReport,
)
from .api import (
    apply_structured_pruning,
    apply_structured_pruning_plan,
    find_structured_pruning_targets,
    plan_structured_pruning,
)
from .discovery import (
    SUPPORTED_GRANULARITIES,
    SUPPORTED_IMPORTANCE_METRICS,
    SUPPORTED_SCOPES,
)
from .sparsity import (
    apply_block_sparse_pruning,
    apply_nm_structured_sparsity,
)
from .dimensions import diff_module_dimensions, snapshot_module_dimensions
from .flops import estimate_model_flops, estimate_module_flops
from .safety import PruneSafetyCheck, PruneSafetyReport, assess_prune_safety

__all__ = [
    "PruneScheduleReport",
    "PruneScheduleStepReport",
    "PruningEntry",
    "PruningReport",
    "PruningSchedule",
    "StructuredPruneScheduleReport",
    "StructuredPruneScheduleStepReport",
    "DEFAULT_PRUNABLE_TYPES",
    "DependencyGroup",
    "describe_prune_runtime_capability",
    "describe_prune_granularity",
    "diff_module_dimensions",
    "estimate_model_flops",
    "estimate_module_flops",
    "apply_block_sparse_pruning",
    "apply_global_l1_unstructured_pruning",
    "apply_nm_structured_sparsity",
    "assess_prune_safety",
    "BlockSparseLayerReport",
    "BlockSparsePruningReport",
    "collect_module_importance",
    "ModuleImportanceRecord",
    "NMStructuredLayerReport",
    "NMStructuredPruningReport",
    "PruneCandidateRecord",
    "PruneGranularitySupport",
    "PruneGranularitySpec",
    "PruneMethodSpec",
    "PruneRuntimeCapability",
    "PruneSafetyCheck",
    "PruneSafetyReport",
    "PruningDependencyGraph",
    "PruningTarget",
    "prune_batchnorm_channels",
    "prune_conv2d_in_channels",
    "prune_conv2d_out_channels",
    "prune_linear_in_features",
    "prune_linear_out_features",
    "validate_conv2d_keep_indices",
    "find_structured_pruning_targets",
    "prune_runtime_capability_from_report",
    "rank_prune_candidates",
    "remove_pruning_reparameterization",
    "normalize_prune_granularity",
    "rewrite_supported_granularities",
    "run_prune_schedule",
    "run_structured_prune_schedule",
    "summarize_pruning",
    "snapshot_module_dimensions",
    "supported_prune_granularities",
    "SUPPORTED_GRANULARITIES",
    "SUPPORTED_IMPORTANCE_METRICS",
    "SUPPORTED_PRUNE_METHODS",
    "SUPPORTED_SCOPES",
    "StructuredPruningAction",
    "StructuredPruningPlan",
    "StructuredPruningReport",
    "structured_prune_support_matrix",
    "describe_prune_method",
    "prune_method_report_fields",
    "supported_prune_method_specs",
    "tensor_sparsity",
    "apply_structured_pruning",
    "apply_structured_pruning_plan",
    "plan_structured_pruning",
]
