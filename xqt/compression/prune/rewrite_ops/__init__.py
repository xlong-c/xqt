"""Structured pruning rewrite operation modules."""

from .linear_conv import (
    prune_batchnorm_channels,
    prune_conv2d_in_channels,
    prune_conv2d_out_channels,
    prune_linear_in_features,
    prune_linear_in_out_features,
    prune_linear_out_features,
    validate_conv2d_keep_indices,
)
from .attention import (
    PrunedGroupedQueryAttention,
    PrunedMultiHeadAttention,
    PrunedSplitProjectionAttention,
    attention_variant,
    infer_attention_role,
)
from .topology import topology_changes_from_actions
from .apply import apply_structured_pruning_plan

__all__ = [
    "attention_variant",
    "apply_structured_pruning_plan",
    "infer_attention_role",
    "prune_batchnorm_channels",
    "prune_conv2d_in_channels",
    "prune_conv2d_out_channels",
    "prune_linear_in_out_features",
    "prune_linear_in_features",
    "prune_linear_out_features",
    "topology_changes_from_actions",
    "validate_conv2d_keep_indices",
]
