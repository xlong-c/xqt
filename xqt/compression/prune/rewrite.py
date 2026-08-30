"""Structured pruning rewrite helpers (facade).

Re-exports public API from rewrite_ops subpackage.
"""

from __future__ import annotations

from .rewrite_ops.attention import attention_variant, infer_attention_role
from .rewrite_ops.apply import apply_structured_pruning_plan
from .rewrite_ops.linear_conv import (
    prune_batchnorm_channels,
    prune_conv2d_in_channels,
    prune_conv2d_out_channels,
    prune_linear_in_features,
    prune_linear_in_out_features,
    prune_linear_out_features,
    validate_conv2d_keep_indices,
)
from .rewrite_ops.topology import topology_changes_from_actions

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
