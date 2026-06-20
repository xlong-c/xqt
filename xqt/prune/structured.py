"""Structured pruning helpers for CNN and ViT-like models."""

from __future__ import annotations

import inspect
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional

import torch
import torch.nn.functional as F
from torch import nn

from .rewrite import (
    prune_batchnorm_channels,
    prune_conv2d_in_channels,
    prune_conv2d_out_channels,
    prune_linear_in_out_features,
    prune_linear_in_features,
    prune_linear_out_features,
    validate_conv2d_keep_indices,
)

SUPPORTED_GRANULARITIES = (
    "channel",
    "filter",
    "mlp_neuron",
    "head",
    "block",
    "stage",
    "hidden_width",
    "embedding_width",
    "expert",
)
SUPPORTED_SCOPES = ("global", "per_layer")
SUPPORTED_IMPORTANCE_METRICS = ("l1", "l2", "bn_gamma", "usage")
_PASSTHROUGH_TYPES = (
    nn.ReLU,
    nn.ReLU6,
    nn.GELU,
    nn.SiLU,
    nn.Identity,
    nn.Dropout,
    nn.Dropout2d,
    nn.Dropout3d,
    nn.Flatten,
    nn.AvgPool2d,
    nn.MaxPool2d,
    nn.AdaptiveAvgPool2d,
    nn.AdaptiveMaxPool2d,
)


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
        }


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
class DependencyGroup:
    """Lightweight dependency-group summary for structured pruning plans."""

    name: str
    producer: str
    consumers: list[str] = field(default_factory=list)
    merge: Optional[str] = None
    shape_constraints: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "producer": self.producer,
            "consumers": list(self.consumers),
            "merge": self.merge,
            "shape_constraints": dict(self.shape_constraints),
        }


@dataclass
class PruningDependencyGraph:
    """Lightweight dependency graph for structured pruning targets."""

    groups: list[DependencyGroup] = field(default_factory=list)

    def add_group(
        self,
        *,
        name: str,
        producer: str,
        consumers: Optional[list[str]] = None,
        merge: Optional[str] = None,
        shape_constraints: Optional[dict[str, Any]] = None,
    ) -> None:
        self.groups.append(
            DependencyGroup(
                name=name,
                producer=producer,
                consumers=list(consumers or []),
                merge=merge,
                shape_constraints=dict(shape_constraints or {}),
            )
        )

    def find_group(self, name: str) -> Optional[DependencyGroup]:
        for group in self.groups:
            if group.name == name:
                return group
        return None

    def to_dict(self) -> dict[str, Any]:
        return {"groups": [group.to_dict() for group in self.groups]}


@dataclass
class _CandidateDiscoveryResult:
    candidates: list[_StructuredCandidate] = field(default_factory=list)
    dependency_graph: PruningDependencyGraph = field(default_factory=PruningDependencyGraph)
    blocked_modules: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class _StructuredPruningAdapter:
    name: str
    granularity: str
    structure_family: str
    collect: Callable[[nn.Module, str], _CandidateDiscoveryResult]


@dataclass(frozen=True)
class _PruneBatch:
    candidate_index: int
    unit_indices: tuple[int, ...]
    score: float


class StructuredPruningToyCNN(nn.Module):
    """Small Conv-BN-ReLU CNN used by structured pruning smoke tests."""

    def __init__(
        self,
        *,
        in_channels: int = 3,
        hidden_channels: int = 8,
        out_channels: int = 16,
        num_classes: int = 4,
    ) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.head = nn.Linear(out_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = self.flatten(x)
        return self.head(x)


class PrunedMultiHeadAttention(nn.Module):
    """Attention module with fewer heads but unchanged input/output embed dim."""

    def __init__(
        self,
        source: nn.Module,
        keep_head_indices: list[int],
    ) -> None:
        super().__init__()
        qkv = getattr(source, "qkv")
        proj = getattr(source, "proj")
        attn_dropout = getattr(source, "attn_dropout")
        proj_dropout = getattr(source, "proj_dropout")
        embed_dim = int(getattr(source, "embed_dim"))
        head_dim = int(getattr(source, "head_dim"))

        keep_features = [
            head * head_dim + offset
            for head in keep_head_indices
            for offset in range(head_dim)
        ]
        qkv_keep = (
            keep_features
            + [embed_dim + index for index in keep_features]
            + [2 * embed_dim + index for index in keep_features]
        )

        self.embed_dim = embed_dim
        self.num_heads = len(keep_head_indices)
        self.head_dim = head_dim
        self.inner_dim = self.num_heads * self.head_dim
        self.scale = self.head_dim**-0.5
        self.qkv = prune_linear_out_features(qkv, qkv_keep)
        self.attn_dropout = nn.Dropout(attn_dropout.p)
        self.proj = prune_linear_in_features(proj, keep_features)
        self.proj_dropout = nn.Dropout(proj_dropout.p)
        self.train(source.training)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, token_count, _ = x.shape
        qkv = self.qkv(x).reshape(
            batch_size,
            token_count,
            3,
            self.num_heads,
            self.head_dim,
        )
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        x = (attn @ v).transpose(1, 2).reshape(batch_size, token_count, self.inner_dim)
        x = self.proj(x)
        x = self.proj_dropout(x)
        return x


class PrunedSplitProjectionAttention(nn.Module):
    """Attention module rewritten from split q/k/v/out projections."""

    def __init__(
        self,
        source: nn.Module,
        keep_head_indices: list[int],
    ) -> None:
        super().__init__()
        q_proj = getattr(source, "q_proj")
        k_proj = getattr(source, "k_proj")
        v_proj = getattr(source, "v_proj")
        out_proj = getattr(source, "out_proj")
        num_heads = int(getattr(source, "num_heads"))
        head_dim = int(getattr(source, "head_dim"))
        embed_dim = int(getattr(source, "embed_dim"))
        dropout = float(getattr(source, "dropout", 0.0))

        if not all(
            isinstance(module, nn.Linear)
            for module in (q_proj, k_proj, v_proj, out_proj)
        ):
            raise TypeError("split attention pruning requires q_proj/k_proj/v_proj/out_proj")

        if len(keep_head_indices) >= num_heads:
            raise ValueError("keep_head_indices must prune at least one attention head")

        keep_features = [
            head * head_dim + offset
            for head in keep_head_indices
            for offset in range(head_dim)
        ]
        self.embed_dim = embed_dim
        self.num_heads = len(keep_head_indices)
        self.head_dim = head_dim
        self.inner_dim = self.num_heads * self.head_dim
        self.scale = self.head_dim**-0.5
        self.attention_role = _infer_attention_role(source)
        self.q_proj = prune_linear_out_features(q_proj, keep_features)
        self.k_proj = prune_linear_out_features(k_proj, keep_features)
        self.v_proj = prune_linear_out_features(v_proj, keep_features)
        self.out_proj = prune_linear_in_features(out_proj, keep_features)
        self.dropout = dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(_dropout_probability(getattr(source, "proj_dropout", dropout)))
        self.train(source.training)

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, token_count, _ = x.shape
        if self.attention_role == "cross":
            if context is None:
                raise ValueError("cross-attention pruning rewrite requires context input")
            kv_source = context
        else:
            kv_source = x if context is None else context
        q = self.q_proj(x).reshape(batch_size, token_count, self.num_heads, self.head_dim)
        kv_batch_size, kv_token_count, _ = kv_source.shape
        if kv_batch_size != batch_size:
            raise ValueError("cross-attention context batch size must match query batch size")
        k = self.k_proj(kv_source).reshape(
            kv_batch_size,
            kv_token_count,
            self.num_heads,
            self.head_dim,
        )
        v = self.v_proj(kv_source).reshape(
            kv_batch_size,
            kv_token_count,
            self.num_heads,
            self.head_dim,
        )
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        x = (attn @ v).transpose(1, 2).reshape(batch_size, token_count, self.inner_dim)
        x = self.out_proj(x)
        x = self.proj_dropout(x)
        return x


class PrunedGroupedQueryAttention(nn.Module):
    """Attention rewrite for GQA/MQA split q/k/v/out projections."""

    def __init__(
        self,
        source: nn.Module,
        keep_kv_head_indices: list[int],
    ) -> None:
        super().__init__()
        q_proj = getattr(source, "q_proj")
        k_proj = getattr(source, "k_proj")
        v_proj = getattr(source, "v_proj")
        out_proj = getattr(source, "out_proj")
        num_heads = int(getattr(source, "num_heads"))
        num_kv_heads = int(getattr(source, "num_kv_heads"))
        head_dim = int(getattr(source, "head_dim"))
        embed_dim = int(getattr(source, "embed_dim"))
        dropout = float(getattr(source, "dropout", 0.0))

        if not all(
            isinstance(module, nn.Linear)
            for module in (q_proj, k_proj, v_proj, out_proj)
        ):
            raise TypeError("grouped-query attention pruning requires q_proj/k_proj/v_proj/out_proj")
        if num_kv_heads <= 0 or num_heads <= 0:
            raise ValueError("grouped-query attention requires positive num_heads and num_kv_heads")
        if num_heads % num_kv_heads != 0:
            raise ValueError(
                "grouped-query attention requires num_heads divisible by num_kv_heads"
            )
        if len(keep_kv_head_indices) >= num_kv_heads:
            raise ValueError("keep_kv_head_indices must prune at least one kv head")

        query_heads_per_kv_head = num_heads // num_kv_heads
        keep_q_head_indices = [
            kv_head_index * query_heads_per_kv_head + query_head_offset
            for kv_head_index in keep_kv_head_indices
            for query_head_offset in range(query_heads_per_kv_head)
        ]
        q_keep_features = [
            head_index * head_dim + offset
            for head_index in keep_q_head_indices
            for offset in range(head_dim)
        ]
        kv_keep_features = [
            head_index * head_dim + offset
            for head_index in keep_kv_head_indices
            for offset in range(head_dim)
        ]

        self.embed_dim = embed_dim
        self.num_heads = len(keep_q_head_indices)
        self.num_kv_heads = len(keep_kv_head_indices)
        self.query_heads_per_kv_head = query_heads_per_kv_head
        self.head_dim = head_dim
        self.inner_dim = self.num_heads * self.head_dim
        self.scale = self.head_dim**-0.5
        self.attention_role = _infer_attention_role(source)
        self.attention_variant = _attention_variant(
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
        )
        self.q_proj = prune_linear_out_features(q_proj, q_keep_features)
        self.k_proj = prune_linear_out_features(k_proj, kv_keep_features)
        self.v_proj = prune_linear_out_features(v_proj, kv_keep_features)
        self.out_proj = prune_linear_in_features(out_proj, q_keep_features)
        self.dropout = dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(
            _dropout_probability(getattr(source, "proj_dropout", dropout))
        )
        self.train(source.training)

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, token_count, _ = x.shape
        if self.attention_role == "cross":
            if context is None:
                raise ValueError("cross-attention pruning rewrite requires context input")
            kv_source = context
        else:
            kv_source = x if context is None else context
        q = self.q_proj(x).reshape(batch_size, token_count, self.num_heads, self.head_dim)
        kv_batch_size, kv_token_count, _ = kv_source.shape
        if kv_batch_size != batch_size:
            raise ValueError("cross-attention context batch size must match query batch size")
        k = self.k_proj(kv_source).reshape(
            kv_batch_size,
            kv_token_count,
            self.num_kv_heads,
            self.head_dim,
        )
        v = self.v_proj(kv_source).reshape(
            kv_batch_size,
            kv_token_count,
            self.num_kv_heads,
            self.head_dim,
        )
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if self.num_heads != self.num_kv_heads:
            repeat_factor = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(repeat_factor, dim=1)
            v = v.repeat_interleave(repeat_factor, dim=1)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        x = (attn @ v).transpose(1, 2).reshape(batch_size, token_count, self.inner_dim)
        x = self.out_proj(x)
        x = self.proj_dropout(x)
        return x


def _tensor_zero_count(tensor: torch.Tensor) -> int:
    return int(torch.count_nonzero(tensor == 0).item())


def _dropout_probability(module_or_probability: Any) -> float:
    if isinstance(module_or_probability, nn.Dropout):
        return float(module_or_probability.p)
    return float(module_or_probability)


def _infer_attention_role(module: nn.Module) -> str:
    for attribute_name in ("attention_role", "attn_role"):
        role = getattr(module, attribute_name, None)
        if isinstance(role, str) and role in {"self", "cross"}:
            return role
    for attribute_name in ("is_cross_attention", "cross_attention"):
        value = getattr(module, attribute_name, None)
        if isinstance(value, bool):
            return "cross" if value else "self"
    try:
        parameter_names = list(inspect.signature(module.forward).parameters)
    except (TypeError, ValueError):
        return "self"
    cross_names = {
        "context",
        "encoder_hidden_states",
        "memory",
        "key_value_states",
        "kv",
    }
    return "cross" if any(name in cross_names for name in parameter_names[1:]) else "self"


def _attention_variant(
    *,
    num_heads: int,
    num_kv_heads: int,
) -> str:
    if num_kv_heads == num_heads:
        return "mha"
    if num_kv_heads == 1:
        return "mqa"
    return "gqa"


def _count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _count_zero_parameters(
    model: nn.Module,
    *,
    module_types: tuple[type[nn.Module], ...] = (nn.Linear, nn.Conv2d),
    parameter_name: str = "weight",
) -> int:
    zero_parameters = 0
    for module in model.modules():
        if not isinstance(module, module_types):
            continue
        parameter = getattr(module, parameter_name, None)
        if isinstance(parameter, torch.Tensor):
            zero_parameters += _tensor_zero_count(parameter)
    return zero_parameters


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


def _named_leaf_modules(model: nn.Module) -> list[tuple[str, nn.Module]]:
    leaves: list[tuple[str, nn.Module]] = []
    for name, module in model.named_modules():
        if name and not any(module.children()):
            leaves.append((name, module))
    if not leaves:
        leaves.append(("<root>", model))
    return leaves


def _get_module(root: nn.Module, name: str) -> nn.Module:
    if name in {"", "<root>"}:
        return root
    module = root
    for part in name.split("."):
        if part.isdigit():
            module = module[int(part)]
        else:
            module = getattr(module, part)
    return module


def _set_module(root: nn.Module, name: str, module: nn.Module) -> None:
    if name in {"", "<root>"}:
        raise ValueError("Replacing the root module is not supported")
    parts = name.split(".")
    parent = _get_module(root, ".".join(parts[:-1])) if len(parts) > 1 else root
    leaf = parts[-1]
    if leaf.isdigit():
        parent[int(leaf)] = module
        return
    setattr(parent, leaf, module)


def _rewrite_indexed_container(
    container: nn.Module,
    keep_indices: list[int],
) -> nn.Module:
    if isinstance(container, nn.ModuleList):
        selected = [container[index] for index in keep_indices]
        rewritten = nn.ModuleList(selected)
    elif isinstance(container, nn.Sequential):
        selected = [container[index] for index in keep_indices]
        rewritten = nn.Sequential(*selected)
    else:
        raise TypeError("container must be a ModuleList or Sequential")
    rewritten.train(container.training)
    return rewritten


def _prune_parameter_last_dim(
    parameter: nn.Parameter,
    keep_indices: list[int],
) -> nn.Parameter:
    keep = torch.tensor(list(keep_indices), dtype=torch.long, device=parameter.device)
    data = parameter.data.index_select(parameter.data.dim() - 1, keep).clone()
    return nn.Parameter(data, requires_grad=parameter.requires_grad)


def _layernorm_feature_count(module: nn.LayerNorm) -> Optional[int]:
    shape = module.normalized_shape
    if isinstance(shape, int):
        return int(shape)
    if isinstance(shape, tuple) and len(shape) == 1:
        return int(shape[0])
    return None


def _prune_layernorm_features(
    module: nn.LayerNorm,
    keep_indices: list[int],
) -> nn.LayerNorm:
    keep = torch.tensor(
        list(keep_indices),
        dtype=torch.long,
        device=(module.weight.device if module.weight is not None else None),
    )
    kwargs: dict[str, Any] = {
        "eps": module.eps,
        "elementwise_affine": module.elementwise_affine,
    }
    if module.weight is not None:
        kwargs["device"] = module.weight.device
        kwargs["dtype"] = module.weight.dtype
    try:
        new_module = nn.LayerNorm(
            int(keep.numel()),
            bias=module.bias is not None,
            **kwargs,
        )
    except TypeError:
        new_module = nn.LayerNorm(int(keep.numel()), **kwargs)
    if module.weight is not None and new_module.weight is not None:
        new_module.weight.data.copy_(module.weight.data.index_select(0, keep))
    if module.bias is not None and new_module.bias is not None:
        new_module.bias.data.copy_(module.bias.data.index_select(0, keep))
    new_module.train(module.training)
    return new_module


def _is_chain_like_module_tree(model: nn.Module) -> bool:
    for name, module in model.named_modules():
        if not name:
            continue
        if any(module.children()) and not isinstance(module, nn.Sequential):
            return False
    return True


def _is_passthrough_module(module: nn.Module) -> bool:
    return isinstance(module, _PASSTHROUGH_TYPES)


def _channel_scores(
    module: nn.Conv2d,
    normalization: Optional[nn.modules.batchnorm._BatchNorm],
    metric: str,
) -> torch.Tensor:
    if metric == "bn_gamma":
        if normalization is None or normalization.weight is None:
            raise ValueError("importance.metric=bn_gamma requires a following BatchNorm")
        return normalization.weight.detach().abs().to(dtype=torch.float32, device="cpu")

    weight = module.weight.detach().to(dtype=torch.float32, device="cpu")
    flattened = weight.reshape(weight.shape[0], -1)
    if metric == "l1":
        return flattened.abs().sum(dim=1)
    if metric == "l2":
        return torch.linalg.vector_norm(flattened, dim=1)
    raise ValueError(f"Unsupported structured importance metric: {metric}")


def _find_consumer(
    leaves: list[tuple[str, nn.Module]],
    start_index: int,
    producer: nn.Conv2d,
) -> tuple[Optional[str], Optional[str], Optional[str], int]:
    normalization_name: Optional[str] = None
    current_index = start_index + 1

    if current_index < len(leaves):
        maybe_norm_name, maybe_norm_module = leaves[current_index]
        if isinstance(maybe_norm_module, nn.modules.batchnorm._BatchNorm):
            if maybe_norm_module.num_features != producer.out_channels:
                raise ValueError(
                    f"BatchNorm '{maybe_norm_name}' does not match producer "
                    f"'{leaves[start_index][0]}' out_channels"
                )
            normalization_name = maybe_norm_name
            current_index += 1

    while current_index < len(leaves):
        consumer_name, consumer_module = leaves[current_index]
        if isinstance(consumer_module, nn.Conv2d):
            if consumer_module.in_channels != producer.out_channels:
                raise ValueError(
                    f"Consumer '{consumer_name}' in_channels do not match "
                    f"producer '{leaves[start_index][0]}' out_channels"
                )
            return consumer_name, "Conv2d", normalization_name, 1
        if isinstance(consumer_module, nn.Linear):
            if consumer_module.in_features % producer.out_channels != 0:
                raise ValueError(
                    f"Linear consumer '{consumer_name}' in_features must be divisible by "
                    f"producer '{leaves[start_index][0]}' out_channels"
                )
            block_size = consumer_module.in_features // producer.out_channels
            return consumer_name, "Linear", normalization_name, block_size
        if not _is_passthrough_module(consumer_module):
            raise ValueError(
                f"Unsupported module '{consumer_name}' of type "
                f"{type(consumer_module).__name__} between structured pruning candidates"
            )
        current_index += 1

    return None, None, normalization_name, 1


def _is_depthwise_conv2d(module: nn.Conv2d) -> bool:
    return module.groups == module.in_channels == module.out_channels


def _conv2d_group_alignment_constraint(
    module: nn.Conv2d,
    *,
    module_name: str,
    axis: str,
) -> Optional[dict[str, Any]]:
    if axis not in {"in", "out"}:
        raise ValueError("axis must be 'in' or 'out'")
    if module.groups == 1 or _is_depthwise_conv2d(module):
        return None
    total_channels = module.in_channels if axis == "in" else module.out_channels
    if total_channels % module.groups != 0:
        raise ValueError(
            f"Grouped Conv2d '{module_name}' must have {axis}_channels divisible by groups"
        )
    return {
        "module_name": module_name,
        "axis": axis,
        "groups": int(module.groups),
        "total_channels": int(total_channels),
        "channels_per_group": int(total_channels // module.groups),
    }


def _conv_channel_group_constraints(
    model: nn.Module,
    *,
    producer_name: str,
    producer: nn.Conv2d,
    consumer_name: Optional[str],
    consumer_type: Optional[str],
) -> list[dict[str, Any]]:
    constraints: list[dict[str, Any]] = []
    producer_constraint = _conv2d_group_alignment_constraint(
        producer,
        module_name=producer_name,
        axis="out",
    )
    if producer_constraint is not None:
        constraints.append(producer_constraint)
    if consumer_name is not None and consumer_type == "Conv2d":
        consumer = _get_module(model, consumer_name)
        if not isinstance(consumer, nn.Conv2d):
            raise TypeError(f"{consumer_name} is not a Conv2d module")
        consumer_constraint = _conv2d_group_alignment_constraint(
            consumer,
            module_name=consumer_name,
            axis="in",
        )
        if consumer_constraint is not None:
            constraints.append(consumer_constraint)
    return constraints


def _collect_conv_candidates(
    model: nn.Module,
    *,
    importance_metric: str,
) -> _CandidateDiscoveryResult:
    if not _is_chain_like_module_tree(model):
        return _CandidateDiscoveryResult()
    leaves = _named_leaf_modules(model)
    candidates: list[_StructuredCandidate] = []
    graph = PruningDependencyGraph()
    blocked_producers: set[str] = set()
    for module_name, module in model.named_modules():
        descriptor_module_name = module_name or "<root>"
        descriptor = _concat_branch_descriptor(descriptor_module_name, module)
        if descriptor is None:
            continue
        for branch_spec in descriptor["branch_specs"]:
            if isinstance(branch_spec, Mapping):
                conv_name = branch_spec.get("conv_name")
                if isinstance(conv_name, str):
                    blocked_producers.add(conv_name)

    for index, (module_name, module) in enumerate(leaves):
        if not isinstance(module, nn.Conv2d):
            continue
        if module_name in blocked_producers:
            continue
        try:
            consumer_name, consumer_type, normalization_name, feature_block_size = _find_consumer(
                leaves,
                index,
                module,
            )
        except ValueError:
            continue
        if consumer_name is None or consumer_type is None:
            continue
        normalization = None
        if normalization_name is not None:
            normalization = _get_module(model, normalization_name)
            if not isinstance(normalization, nn.modules.batchnorm._BatchNorm):
                raise TypeError(f"{normalization_name} is not a BatchNorm module")
        scores = _channel_scores(module, normalization, importance_metric)
        group_constraints = _conv_channel_group_constraints(
            model,
            producer_name=module_name,
            producer=module,
            consumer_name=consumer_name,
            consumer_type=consumer_type,
        )
        candidates.append(
            _StructuredCandidate(
                adapter="cnn_chain_adapter",
                structure_family="cnn",
                action_type="conv_channel_group",
                module_name=module_name,
                module_type="Conv2d",
                granularity="channel",
                dependency_group=module_name,
                consumer_name=consumer_name,
                consumer_type=consumer_type,
                normalization_name=normalization_name,
                feature_block_size=feature_block_size,
                scores=scores,
                metadata={
                    "normalization_name": normalization_name,
                    "consumer_name": consumer_name,
                    "consumer_type": consumer_type,
                    "feature_block_size": feature_block_size,
                    "group_alignment_constraints": [dict(item) for item in group_constraints],
                },
            )
        )
        graph.add_group(
            name=module_name,
            producer=module_name,
            consumers=[consumer_name],
            merge=None,
            shape_constraints={
                "consumer_type": consumer_type,
                "feature_block_size": feature_block_size,
                "normalization_name": normalization_name,
                "group_alignment_constraints": [dict(item) for item in group_constraints],
            },
        )
    return _CandidateDiscoveryResult(candidates=candidates, dependency_graph=graph)


def _concat_branch_descriptor(
    module_name: str,
    module: nn.Module,
) -> Optional[dict[str, Any]]:
    prefix = "" if module_name in {"", "<root>"} else f"{module_name}."
    branch_specs: list[dict[str, Any]] = []
    for branch_index, branch_attr in enumerate(("branch1", "branch2")):
        branch_module = getattr(module, branch_attr, None)
        if not isinstance(branch_module, nn.Sequential):
            return None
        children = list(branch_module.children())
        if not children or not isinstance(children[0], nn.Conv2d):
            return None
        conv = children[0]
        if conv.groups != 1:
            return None
        bn_name: Optional[str] = None
        if len(children) >= 2:
            maybe_bn = children[1]
            if isinstance(maybe_bn, nn.modules.batchnorm._BatchNorm):
                if maybe_bn.num_features != conv.out_channels:
                    return None
                bn_name = f"{module_name}.{branch_attr}.1"
        branch_specs.append(
            {
                "branch_attr": branch_attr,
                "branch_index": branch_index,
                "conv_name": f"{prefix}{branch_attr}.0",
                "bn_name": (
                    f"{prefix}{branch_attr}.1" if bn_name is not None else None
                ),
                "out_channels": int(conv.out_channels),
            }
        )

    consumer = getattr(module, "fuse", None)
    if not isinstance(consumer, nn.Conv2d) or consumer.groups != 1:
        return None
    total_out_channels = sum(int(spec["out_channels"]) for spec in branch_specs)
    if consumer.in_channels != total_out_channels:
        return None
    return {
        "module_name": module_name,
        "branch_specs": branch_specs,
        "consumer_name": f"{prefix}fuse",
        "consumer_type": "Conv2d",
        "total_out_channels": total_out_channels,
    }


def _collect_concat_branch_candidates(
    model: nn.Module,
    *,
    importance_metric: str,
) -> _CandidateDiscoveryResult:
    candidates: list[_StructuredCandidate] = []
    graph = PruningDependencyGraph()
    for module_name, module in model.named_modules():
        descriptor_module_name = module_name or "<root>"
        descriptor = _concat_branch_descriptor(descriptor_module_name, module)
        if descriptor is None:
            continue
        branch_specs = descriptor["branch_specs"]
        consumer_name = str(descriptor["consumer_name"])
        for branch_spec in branch_specs:
            conv_name = str(branch_spec["conv_name"])
            conv = _get_module(model, conv_name)
            if not isinstance(conv, nn.Conv2d):
                raise TypeError(f"{conv_name} is not a Conv2d module")
            bn_name = branch_spec.get("bn_name")
            normalization = None
            if isinstance(bn_name, str):
                normalization = _get_module(model, bn_name)
                if not isinstance(normalization, nn.modules.batchnorm._BatchNorm):
                    raise TypeError(f"{bn_name} is not a BatchNorm module")
            scores = _channel_scores(conv, normalization, importance_metric)
            metadata = {
                "container_name": module_name,
                "branch_index": int(branch_spec["branch_index"]),
                "branch_attr": str(branch_spec["branch_attr"]),
                "branch_specs": [dict(item) for item in branch_specs],
                "merge": "concat",
                "consumer_name": consumer_name,
                "consumer_type": "Conv2d",
            }
            candidates.append(
                _StructuredCandidate(
                    adapter="concat_branch_adapter",
                    structure_family="cnn_branch",
                    action_type="concat_branch_channels",
                    module_name=conv_name,
                    module_type="Conv2d",
                    granularity="channel",
                    dependency_group=f"{module_name}:{branch_spec['branch_attr']}",
                    consumer_name=consumer_name,
                    consumer_type="Conv2d",
                    normalization_name=bn_name if isinstance(bn_name, str) else None,
                    feature_block_size=1,
                    scores=scores,
                    metadata=metadata,
                )
            )
            graph.add_group(
                name=f"{module_name}:{branch_spec['branch_attr']}",
                producer=conv_name,
                consumers=[consumer_name],
                merge="concat",
                shape_constraints={
                    "branch_index": int(branch_spec["branch_index"]),
                    "branch_attr": str(branch_spec["branch_attr"]),
                    "branch_specs": [dict(item) for item in branch_specs],
                    "consumer_name": consumer_name,
                },
            )
    return _CandidateDiscoveryResult(candidates=candidates, dependency_graph=graph)


def _mbconv_descriptor(
    module_name: str,
    module: nn.Module,
) -> Optional[dict[str, Any]]:
    expand = getattr(module, "expand_conv", None)
    expand_bn = getattr(module, "expand_bn", None)
    depthwise = getattr(module, "depthwise_conv", None)
    depthwise_bn = getattr(module, "depthwise_bn", None)
    project = getattr(module, "project_conv", None)
    project_bn = getattr(module, "project_bn", None)
    if not (
        isinstance(expand, nn.Conv2d)
        and isinstance(expand_bn, nn.modules.batchnorm._BatchNorm)
        and isinstance(depthwise, nn.Conv2d)
        and isinstance(depthwise_bn, nn.modules.batchnorm._BatchNorm)
        and isinstance(project, nn.Conv2d)
        and isinstance(project_bn, nn.modules.batchnorm._BatchNorm)
    ):
        return None
    mid_channels = int(expand.out_channels)
    if (
        expand.groups != 1
        or expand_bn.num_features != mid_channels
        or depthwise.in_channels != mid_channels
        or depthwise.out_channels != mid_channels
        or depthwise.groups != mid_channels
        or depthwise_bn.num_features != mid_channels
        or project.in_channels != mid_channels
    ):
        return None
    return {
        "module_name": module_name,
        "mid_channels": mid_channels,
        "expand_conv_name": f"{module_name}.expand_conv",
        "expand_bn_name": f"{module_name}.expand_bn",
        "depthwise_conv_name": f"{module_name}.depthwise_conv",
        "depthwise_bn_name": f"{module_name}.depthwise_bn",
        "project_conv_name": f"{module_name}.project_conv",
        "project_bn_name": f"{module_name}.project_bn",
    }


def _mbconv_scores(
    descriptor: Mapping[str, Any],
    model: nn.Module,
    metric: str,
) -> torch.Tensor:
    expand_conv = _get_module(model, str(descriptor["expand_conv_name"]))
    expand_bn = _get_module(model, str(descriptor["expand_bn_name"]))
    depthwise_conv = _get_module(model, str(descriptor["depthwise_conv_name"]))
    depthwise_bn = _get_module(model, str(descriptor["depthwise_bn_name"]))
    project_conv = _get_module(model, str(descriptor["project_conv_name"]))
    if not all(
        isinstance(module, nn.Conv2d)
        for module in (expand_conv, depthwise_conv, project_conv)
    ):
        raise TypeError("MBConv pruning requires Conv2d modules")
    if not all(
        isinstance(module, nn.modules.batchnorm._BatchNorm)
        for module in (expand_bn, depthwise_bn)
    ):
        raise TypeError("MBConv pruning requires BatchNorm modules")
    expand_weight = expand_conv.weight.detach().to(dtype=torch.float32, device="cpu")
    depthwise_weight = depthwise_conv.weight.detach().to(dtype=torch.float32, device="cpu")
    project_weight = project_conv.weight.detach().to(dtype=torch.float32, device="cpu")
    if metric == "bn_gamma":
        return (
            expand_bn.weight.detach().abs().to(dtype=torch.float32, device="cpu")
            + depthwise_bn.weight.detach().abs().to(dtype=torch.float32, device="cpu")
        )
    if metric == "l1":
        return (
            expand_weight.abs().sum(dim=(1, 2, 3))
            + depthwise_weight.abs().sum(dim=(1, 2, 3))
            + project_weight.abs().sum(dim=(0, 2, 3))
        )
    if metric == "l2":
        return torch.sqrt(
            torch.linalg.vector_norm(expand_weight.reshape(expand_conv.out_channels, -1), dim=1).square()
            + torch.linalg.vector_norm(depthwise_weight.reshape(depthwise_conv.out_channels, -1), dim=1).square()
            + torch.linalg.vector_norm(
                project_weight.permute(1, 0, 2, 3).reshape(project_conv.in_channels, -1),
                dim=1,
            ).square()
        )
    raise ValueError(f"Unsupported structured importance metric: {metric}")


def _collect_mbconv_candidates(
    model: nn.Module,
    *,
    importance_metric: str,
) -> _CandidateDiscoveryResult:
    candidates: list[_StructuredCandidate] = []
    graph = PruningDependencyGraph()
    for module_name, module in model.named_modules():
        if not module_name:
            continue
        descriptor = _mbconv_descriptor(module_name, module)
        if descriptor is None:
            continue
        scores = _mbconv_scores(descriptor, model, importance_metric)
        candidates.append(
            _StructuredCandidate(
                adapter="mbconv_adapter",
                structure_family="cnn_mbconv",
                action_type="mbconv_mid_channels",
                module_name=module_name,
                module_type=type(module).__name__,
                granularity="channel",
                dependency_group=module_name,
                consumer_name=str(descriptor["project_conv_name"]),
                consumer_type="Conv2d",
                normalization_name=None,
                feature_block_size=1,
                scores=scores,
                metadata=dict(descriptor),
            )
        )
        graph.add_group(
            name=module_name,
            producer=str(descriptor["expand_conv_name"]),
            consumers=[str(descriptor["project_conv_name"])],
            merge=None,
            shape_constraints={
                "mid_channels": int(descriptor["mid_channels"]),
                "depthwise_groups": int(descriptor["mid_channels"]),
            },
        )
    return _CandidateDiscoveryResult(candidates=candidates, dependency_graph=graph)


def _residual_block_descriptor(
    module_name: str,
    module: nn.Module,
) -> Optional[dict[str, Any]]:
    conv1 = getattr(module, "conv1", None)
    conv2 = getattr(module, "conv2", None)
    bn1 = getattr(module, "bn1", None)
    bn2 = getattr(module, "bn2", None)
    if (
        isinstance(conv1, nn.Conv2d)
        and isinstance(conv2, nn.Conv2d)
        and isinstance(bn1, nn.modules.batchnorm._BatchNorm)
        and isinstance(bn2, nn.modules.batchnorm._BatchNorm)
        and conv1.out_channels == conv2.out_channels == bn1.num_features == bn2.num_features
    ):
        block_type = type(module).__name__
        if block_type == "BasicBlock":
            return {
                "block_name": module_name,
                "block_type": block_type,
                "out_channels": conv2.out_channels,
                "conv1_name": f"{module_name}.conv1",
                "bn1_name": f"{module_name}.bn1",
                "conv2_name": f"{module_name}.conv2",
                "bn2_name": f"{module_name}.bn2",
                "downsample_conv_name": (
                    f"{module_name}.downsample.0"
                    if isinstance(getattr(module, "downsample", None), nn.Sequential)
                    and len(getattr(module, "downsample")) >= 2
                    and isinstance(module.downsample[0], nn.Conv2d)
                    and isinstance(module.downsample[1], nn.modules.batchnorm._BatchNorm)
                    else None
                ),
                "downsample_bn_name": (
                    f"{module_name}.downsample.1"
                    if isinstance(getattr(module, "downsample", None), nn.Sequential)
                    and len(getattr(module, "downsample")) >= 2
                    and isinstance(module.downsample[0], nn.Conv2d)
                    and isinstance(module.downsample[1], nn.modules.batchnorm._BatchNorm)
                    else None
                ),
            }

    conv3 = getattr(module, "conv3", None)
    bn3 = getattr(module, "bn3", None)
    if (
        isinstance(conv1, nn.Conv2d)
        and isinstance(conv2, nn.Conv2d)
        and isinstance(conv3, nn.Conv2d)
        and isinstance(bn1, nn.modules.batchnorm._BatchNorm)
        and isinstance(bn2, nn.modules.batchnorm._BatchNorm)
        and isinstance(bn3, nn.modules.batchnorm._BatchNorm)
        and conv2.out_channels == bn2.num_features
        and conv3.out_channels == bn3.num_features
        and type(module).__name__ == "Bottleneck"
    ):
        return {
            "block_name": module_name,
            "block_type": "Bottleneck",
            "out_channels": conv2.out_channels,
            "conv1_name": f"{module_name}.conv1",
            "bn1_name": f"{module_name}.bn1",
            "conv2_name": f"{module_name}.conv2",
            "bn2_name": f"{module_name}.bn2",
            "conv3_name": f"{module_name}.conv3",
            "bn3_name": f"{module_name}.bn3",
            "project_out_channels": conv3.out_channels,
            "downsample_conv_name": (
                f"{module_name}.downsample.0"
                if isinstance(getattr(module, "downsample", None), nn.Sequential)
                and len(getattr(module, "downsample")) >= 2
                and isinstance(module.downsample[0], nn.Conv2d)
                and isinstance(module.downsample[1], nn.modules.batchnorm._BatchNorm)
                else None
            ),
            "downsample_bn_name": (
                f"{module_name}.downsample.1"
                if isinstance(getattr(module, "downsample", None), nn.Sequential)
                and len(getattr(module, "downsample")) >= 2
                and isinstance(module.downsample[0], nn.Conv2d)
                and isinstance(module.downsample[1], nn.modules.batchnorm._BatchNorm)
                else None
            ),
        }
    return None


def _find_next_feature_consumer(
    model: nn.Module,
    *,
    module_name: str,
    out_channels: int,
) -> tuple[Optional[str], Optional[str], int]:
    module_prefix = f"{module_name}."
    passed = False
    for name, module in model.named_modules():
        if name == module_name:
            passed = True
            continue
        if not passed:
            continue
        if name.startswith(module_prefix):
            continue
        if isinstance(module, nn.Conv2d) and module.in_channels == out_channels:
            return name, "Conv2d", 1
        if isinstance(module, nn.Linear):
            if module.in_features % out_channels != 0:
                continue
            return name, "Linear", module.in_features // out_channels
    return None, None, 1


def _residual_stage_descriptor(
    stage_name: str,
    stage: nn.Module,
) -> Optional[dict[str, Any]]:
    if not isinstance(stage, nn.Sequential):
        return None
    blocks = list(stage.children())
    if not blocks:
        return None
    block_type = type(blocks[0]).__name__
    if block_type not in {"BasicBlock", "Bottleneck"}:
        return None
    if any(type(block).__name__ != block_type for block in blocks):
        return None

    block_descriptors: list[dict[str, Any]] = []
    for block_index, block in enumerate(blocks):
        descriptor = _residual_block_descriptor(f"{stage_name}.{block_index}", block)
        if descriptor is None:
            return None
        block_descriptors.append(descriptor)

    first = block_descriptors[0]
    if first.get("downsample_conv_name") is None or first.get("downsample_bn_name") is None:
        return None

    if block_type == "BasicBlock":
        stage_out_channels = int(first["out_channels"])
        for descriptor in block_descriptors:
            if int(descriptor["out_channels"]) != stage_out_channels:
                return None
    else:
        stage_out_channels = int(first["project_out_channels"])
        for descriptor in block_descriptors:
            if int(descriptor["project_out_channels"]) != stage_out_channels:
                return None

    return {
        "stage_name": stage_name,
        "block_type": block_type,
        "stage_out_channels": stage_out_channels,
        "block_count": len(block_descriptors),
        "blocks": block_descriptors,
    }


def _residual_stage_scores(
    descriptor: Mapping[str, Any],
    model: nn.Module,
    metric: str,
) -> torch.Tensor:
    block_type = str(descriptor["block_type"])
    stage_out_channels = int(descriptor["stage_out_channels"])
    score = torch.zeros(stage_out_channels, dtype=torch.float32)
    blocks = descriptor["blocks"]
    if not isinstance(blocks, list):
        raise TypeError("descriptor.blocks must be a list")

    for block_index, block in enumerate(blocks):
        if not isinstance(block, Mapping):
            raise TypeError("descriptor.blocks items must be mappings")
        if block_type == "BasicBlock":
            conv1 = _get_module(model, str(block["conv1_name"]))
            conv2 = _get_module(model, str(block["conv2_name"]))
            bn1 = _get_module(model, str(block["bn1_name"]))
            bn2 = _get_module(model, str(block["bn2_name"]))
            if not isinstance(conv1, nn.Conv2d) or not isinstance(conv2, nn.Conv2d):
                raise TypeError("BasicBlock residual pruning requires conv1/conv2")
            if not isinstance(bn1, nn.modules.batchnorm._BatchNorm) or not isinstance(
                bn2, nn.modules.batchnorm._BatchNorm
            ):
                raise TypeError("BasicBlock residual pruning requires bn1/bn2")
            conv1_weight = conv1.weight.detach().to(dtype=torch.float32, device="cpu")
            conv2_weight = conv2.weight.detach().to(dtype=torch.float32, device="cpu")
            if metric == "bn_gamma":
                score = score + bn2.weight.detach().abs().to(dtype=torch.float32, device="cpu")
                continue
            if metric == "l1":
                score = score + conv1_weight.abs().sum(dim=(1, 2, 3))
                if conv1.in_channels == stage_out_channels:
                    score = score + conv1_weight.abs().sum(dim=(0, 2, 3))
                score = score + conv2_weight.abs().sum(dim=(0, 2, 3))
                score = score + conv2_weight.abs().sum(dim=(1, 2, 3))
                score = score + bn1.weight.detach().abs().to(dtype=torch.float32, device="cpu")
                score = score + bn2.weight.detach().abs().to(dtype=torch.float32, device="cpu")
                continue
            if metric == "l2":
                conv1_out = torch.linalg.vector_norm(
                    conv1_weight.reshape(conv1.out_channels, -1),
                    dim=1,
                ).square()
                conv1_in = torch.zeros_like(conv1_out)
                if conv1.in_channels == stage_out_channels:
                    conv1_in = torch.linalg.vector_norm(
                        conv1_weight.permute(1, 0, 2, 3).reshape(stage_out_channels, -1),
                        dim=1,
                    ).square()
                score = score + torch.sqrt(
                    conv1_out
                    + conv1_in
                    + torch.linalg.vector_norm(
                        conv2_weight.permute(1, 0, 2, 3).reshape(stage_out_channels, -1),
                        dim=1,
                    ).square()
                    + torch.linalg.vector_norm(
                        conv2_weight.reshape(stage_out_channels, -1),
                        dim=1,
                    ).square()
                )
                continue
            raise ValueError(f"Unsupported structured importance metric: {metric}")

        conv1 = _get_module(model, str(block["conv1_name"]))
        conv3 = _get_module(model, str(block["conv3_name"]))
        bn3 = _get_module(model, str(block["bn3_name"]))
        if not isinstance(conv1, nn.Conv2d) or not isinstance(conv3, nn.Conv2d):
            raise TypeError("Bottleneck residual pruning requires conv1/conv3")
        if not isinstance(bn3, nn.modules.batchnorm._BatchNorm):
            raise TypeError("Bottleneck residual pruning requires bn3")
        conv1_weight = conv1.weight.detach().to(dtype=torch.float32, device="cpu")
        conv3_weight = conv3.weight.detach().to(dtype=torch.float32, device="cpu")
        if metric == "bn_gamma":
            score = score + bn3.weight.detach().abs().to(dtype=torch.float32, device="cpu")
            continue
        if metric == "l1":
            if conv1.in_channels == stage_out_channels:
                score = score + conv1_weight.abs().sum(dim=(0, 2, 3))
            score = score + conv3_weight.abs().sum(dim=(1, 2, 3))
            score = score + bn3.weight.detach().abs().to(dtype=torch.float32, device="cpu")
            continue
        if metric == "l2":
            conv1_in = torch.zeros_like(score)
            if conv1.in_channels == stage_out_channels:
                conv1_in = torch.linalg.vector_norm(
                    conv1_weight.permute(1, 0, 2, 3).reshape(stage_out_channels, -1),
                    dim=1,
                ).square()
            score = score + torch.sqrt(
                conv1_in
                + torch.linalg.vector_norm(
                    conv3_weight.reshape(stage_out_channels, -1),
                    dim=1,
                ).square()
            )
            continue
        raise ValueError(f"Unsupported structured importance metric: {metric}")

    if block_type == "Bottleneck":
        first = blocks[0]
        if isinstance(first, Mapping) and first.get("downsample_conv_name") is not None:
            downsample_conv = _get_module(model, str(first["downsample_conv_name"]))
            if isinstance(downsample_conv, nn.Conv2d):
                downsample_weight = downsample_conv.weight.detach().to(
                    dtype=torch.float32,
                    device="cpu",
                )
                if metric == "l1":
                    score = score + downsample_weight.abs().sum(dim=(1, 2, 3))
                elif metric == "l2":
                    score = score + torch.sqrt(
                        torch.linalg.vector_norm(
                            downsample_weight.reshape(stage_out_channels, -1),
                            dim=1,
                        ).square()
                    )
    return score


def _collect_residual_conv_candidates(
    model: nn.Module,
    *,
    importance_metric: str,
) -> _CandidateDiscoveryResult:
    candidates: list[_StructuredCandidate] = []
    graph = PruningDependencyGraph()
    stage_descriptors: list[dict[str, Any]] = []
    for stage_name, stage in model.named_modules():
        if not stage_name:
            continue
        descriptor = _residual_stage_descriptor(stage_name, stage)
        if descriptor is not None:
            stage_descriptors.append(descriptor)

    for stage_index, descriptor in enumerate(stage_descriptors):
        module_name = str(descriptor["stage_name"])
        scores = _residual_stage_scores(descriptor, model, importance_metric)
        consumers: list[dict[str, Any]] = []
        if stage_index + 1 < len(stage_descriptors):
            next_stage = stage_descriptors[stage_index + 1]
            next_blocks = next_stage["blocks"]
            if not isinstance(next_blocks, list) or not next_blocks:
                raise TypeError("next_stage.blocks must be a non-empty list")
            first_block = next_blocks[0]
            if not isinstance(first_block, Mapping):
                raise TypeError("next_stage first block descriptor must be a mapping")
            consumers.append(
                {"name": str(first_block["conv1_name"]), "type": "Conv2d", "feature_block_size": 1}
            )
            if first_block.get("downsample_conv_name") is not None:
                consumers.append(
                    {
                        "name": str(first_block["downsample_conv_name"]),
                        "type": "Conv2d",
                        "feature_block_size": 1,
                    }
                )
        else:
            next_consumer_name, next_consumer_type, feature_block_size = _find_next_feature_consumer(
                model,
                module_name=module_name,
                out_channels=int(descriptor["stage_out_channels"]),
            )
            if next_consumer_name is not None and next_consumer_type is not None:
                consumers.append(
                    {
                        "name": next_consumer_name,
                        "type": next_consumer_type,
                        "feature_block_size": feature_block_size,
                    }
                )

        primary_consumer = consumers[0] if consumers else None
        metadata = dict(descriptor)
        metadata["consumers"] = consumers
        metadata["consumer_name"] = primary_consumer["name"] if primary_consumer is not None else None
        metadata["consumer_type"] = primary_consumer["type"] if primary_consumer is not None else None
        metadata["feature_block_size"] = (
            int(primary_consumer["feature_block_size"]) if primary_consumer is not None else 1
        )
        metadata["merge"] = "add"
        candidates.append(
            _StructuredCandidate(
                adapter="residual_cnn_adapter",
                structure_family="cnn_residual",
                action_type="residual_stage_channels",
                module_name=module_name,
                module_type="Sequential",
                granularity="channel",
                dependency_group=module_name,
                consumer_name=metadata["consumer_name"],
                consumer_type=metadata["consumer_type"],
                normalization_name=None,
                feature_block_size=int(metadata["feature_block_size"]),
                scores=scores,
                metadata=metadata,
            )
        )
        graph.add_group(
            name=module_name,
            producer=module_name,
            consumers=[str(item["name"]) for item in consumers],
            merge="add",
            shape_constraints={
                "block_type": str(descriptor["block_type"]),
                "stage_out_channels": int(descriptor["stage_out_channels"]),
                "block_count": int(descriptor["block_count"]),
                "consumers": [dict(item) for item in consumers],
            },
        )
    return _CandidateDiscoveryResult(candidates=candidates, dependency_graph=graph)


def _mlp_neuron_scores(
    fc1: nn.Linear,
    fc2: nn.Linear,
    metric: str,
) -> torch.Tensor:
    fc1_weight = fc1.weight.detach().to(dtype=torch.float32, device="cpu")
    fc2_weight = fc2.weight.detach().to(dtype=torch.float32, device="cpu")

    if metric == "l1":
        score = fc1_weight.abs().sum(dim=1) + fc2_weight.abs().sum(dim=0)
        if fc1.bias is not None:
            score = score + fc1.bias.detach().abs().to(dtype=torch.float32, device="cpu")
        return score
    if metric == "l2":
        fc1_score = torch.linalg.vector_norm(fc1_weight, dim=1)
        fc2_score = torch.linalg.vector_norm(fc2_weight, dim=0)
        score = torch.sqrt(fc1_score.square() + fc2_score.square())
        if fc1.bias is not None:
            bias = fc1.bias.detach().to(dtype=torch.float32, device="cpu")
            score = torch.sqrt(score.square() + bias.square())
        return score
    raise ValueError(f"Unsupported structured importance metric: {metric}")


def _gated_mlp_neuron_scores(
    gate_proj: nn.Linear,
    up_proj: nn.Linear,
    down_proj: nn.Linear,
    metric: str,
) -> torch.Tensor:
    gate_weight = gate_proj.weight.detach().to(dtype=torch.float32, device="cpu")
    up_weight = up_proj.weight.detach().to(dtype=torch.float32, device="cpu")
    down_weight = down_proj.weight.detach().to(dtype=torch.float32, device="cpu")

    if metric == "l1":
        score = (
            gate_weight.abs().sum(dim=1)
            + up_weight.abs().sum(dim=1)
            + down_weight.abs().sum(dim=0)
        )
        if gate_proj.bias is not None:
            score = score + gate_proj.bias.detach().abs().to(dtype=torch.float32, device="cpu")
        if up_proj.bias is not None:
            score = score + up_proj.bias.detach().abs().to(dtype=torch.float32, device="cpu")
        return score
    if metric == "l2":
        score = (
            torch.linalg.vector_norm(gate_weight, dim=1).square()
            + torch.linalg.vector_norm(up_weight, dim=1).square()
            + torch.linalg.vector_norm(down_weight, dim=0).square()
        )
        if gate_proj.bias is not None:
            bias = gate_proj.bias.detach().to(dtype=torch.float32, device="cpu")
            score = score + bias.square()
        if up_proj.bias is not None:
            bias = up_proj.bias.detach().to(dtype=torch.float32, device="cpu")
            score = score + bias.square()
        return torch.sqrt(score)
    raise ValueError(f"Unsupported structured importance metric: {metric}")


def _collect_mlp_candidates(
    model: nn.Module,
    *,
    importance_metric: str,
) -> _CandidateDiscoveryResult:
    candidates: list[_StructuredCandidate] = []
    graph = PruningDependencyGraph()
    for module_name, module in model.named_modules():
        if not module_name:
            continue
        gate_proj = getattr(module, "gate_proj", None)
        up_proj = getattr(module, "up_proj", None)
        down_proj = getattr(module, "down_proj", None)
        if all(
            isinstance(linear_module, nn.Linear)
            for linear_module in (gate_proj, up_proj, down_proj)
        ):
            if not isinstance(gate_proj, nn.Linear):
                raise TypeError("gate_proj must be Linear")
            if not isinstance(up_proj, nn.Linear):
                raise TypeError("up_proj must be Linear")
            if not isinstance(down_proj, nn.Linear):
                raise TypeError("down_proj must be Linear")
            if gate_proj.out_features != up_proj.out_features:
                raise ValueError(
                    f"Gated MLP '{module_name}' has incompatible gate/up dimensions: "
                    f"{gate_proj.out_features} vs {up_proj.out_features}"
                )
            if up_proj.out_features != down_proj.in_features:
                raise ValueError(
                    f"Gated MLP '{module_name}' has incompatible up/down dimensions: "
                    f"{up_proj.out_features} vs {down_proj.in_features}"
                )
            scores = _gated_mlp_neuron_scores(
                gate_proj,
                up_proj,
                down_proj,
                importance_metric,
            )
            gate_name = f"{module_name}.gate_proj"
            up_name = f"{module_name}.up_proj"
            down_name = f"{module_name}.down_proj"
            candidates.append(
                _StructuredCandidate(
                    adapter="mlp_pair_adapter",
                    structure_family="transformer",
                    action_type="gated_mlp_neuron_group",
                    module_name=up_name,
                    module_type="Linear",
                    granularity="mlp_neuron",
                    dependency_group=module_name,
                    consumer_name=down_name,
                    consumer_type="Linear",
                    normalization_name=None,
                    feature_block_size=1,
                    scores=scores,
                    metadata={
                        "parent_name": module_name,
                        "mlp_kind": "gated",
                        "gate_proj_name": gate_name,
                        "up_proj_name": up_name,
                        "down_proj_name": down_name,
                    },
                )
            )
            graph.add_group(
                name=module_name,
                producer=up_name,
                consumers=[gate_name, down_name],
                merge="mul",
                shape_constraints={
                    "mlp_kind": "gated",
                    "shared_intermediate_dim": up_proj.out_features,
                    "gate_proj_name": gate_name,
                    "up_proj_name": up_name,
                    "down_proj_name": down_name,
                },
            )
            continue

        fc1 = getattr(module, "fc1", None)
        fc2 = getattr(module, "fc2", None)
        if not isinstance(fc1, nn.Linear) or not isinstance(fc2, nn.Linear):
            continue
        if fc1.out_features != fc2.in_features:
            raise ValueError(
                f"MLP pair '{module_name}' has incompatible fc1/fc2 dimensions: "
                f"{fc1.out_features} vs {fc2.in_features}"
            )
        scores = _mlp_neuron_scores(fc1, fc2, importance_metric)
        fc1_name = f"{module_name}.fc1"
        fc2_name = f"{module_name}.fc2"
        candidates.append(
            _StructuredCandidate(
                adapter="mlp_pair_adapter",
                structure_family="transformer",
                action_type="mlp_neuron_group",
                module_name=fc1_name,
                module_type="Linear",
                granularity="mlp_neuron",
                dependency_group=module_name,
                consumer_name=fc2_name,
                consumer_type="Linear",
                normalization_name=None,
                feature_block_size=1,
                scores=scores,
                metadata={
                    "parent_name": module_name,
                    "partner_name": fc2_name,
                },
            )
        )
        graph.add_group(
            name=module_name,
            producer=fc1_name,
            consumers=[fc2_name],
            merge=None,
            shape_constraints={"shared_intermediate_dim": fc1.out_features},
        )
    return _CandidateDiscoveryResult(candidates=candidates, dependency_graph=graph)


def _head_scores(
    descriptor: Mapping[str, Any],
    metric: str,
) -> torch.Tensor:
    attention_kind = str(descriptor["attention_kind"])
    num_heads = int(descriptor["num_heads"])
    head_dim = int(descriptor["head_dim"])
    embed_dim = int(descriptor["embed_dim"])
    num_kv_heads = int(descriptor.get("num_kv_heads", num_heads))
    attention_variant = str(descriptor.get("attention_variant", "mha"))
    scores: list[float] = []
    score_units = num_heads
    if attention_kind == "split_qkv" and attention_variant in {"gqa", "mqa"}:
        score_units = num_kv_heads
    for head_index in range(score_units):
        start = head_index * head_dim
        end = start + head_dim
        if attention_kind == "fused_qkv":
            qkv = descriptor["qkv"]
            proj = descriptor["proj"]
            if not isinstance(qkv, nn.Linear) or not isinstance(proj, nn.Linear):
                raise ValueError(
                    "attention head pruning requires qkv/proj for fused_qkv attention"
                )
            qkv_weight = qkv.weight.detach().to(dtype=torch.float32, device="cpu")
            proj_weight = proj.weight.detach().to(dtype=torch.float32, device="cpu")
            q_weight = qkv_weight[start:end]
            k_weight = qkv_weight[embed_dim + start : embed_dim + end]
            v_weight = qkv_weight[2 * embed_dim + start : 2 * embed_dim + end]
            proj_slice = proj_weight[:, start:end]
        elif attention_kind == "split_qkv":
            q_proj = descriptor["q_proj"]
            k_proj = descriptor["k_proj"]
            v_proj = descriptor["v_proj"]
            out_proj = descriptor["out_proj"]
            if not all(
                isinstance(module, nn.Linear)
                for module in (q_proj, k_proj, v_proj, out_proj)
            ):
                raise ValueError(
                    "attention head pruning requires q_proj/k_proj/v_proj/out_proj "
                    "for split_qkv attention"
                )
            q_proj_weight = q_proj.weight.detach().to(dtype=torch.float32, device="cpu")
            k_proj_weight = k_proj.weight.detach().to(dtype=torch.float32, device="cpu")
            v_proj_weight = v_proj.weight.detach().to(dtype=torch.float32, device="cpu")
            out_proj_weight = out_proj.weight.detach().to(dtype=torch.float32, device="cpu")
            if attention_variant in {"gqa", "mqa"}:
                query_heads_per_kv_head = num_heads // num_kv_heads
                q_slices = []
                proj_slices = []
                for query_head_offset in range(query_heads_per_kv_head):
                    q_head_index = head_index * query_heads_per_kv_head + query_head_offset
                    q_start = q_head_index * head_dim
                    q_end = q_start + head_dim
                    q_slices.append(q_proj_weight[q_start:q_end])
                    proj_slices.append(out_proj_weight[:, q_start:q_end])
                q_weight = torch.cat(q_slices, dim=0)
                k_weight = k_proj_weight[start:end]
                v_weight = v_proj_weight[start:end]
                proj_slice = torch.cat(proj_slices, dim=1)
            else:
                q_weight = q_proj_weight[start:end]
                k_weight = k_proj_weight[start:end]
                v_weight = v_proj_weight[start:end]
                proj_slice = out_proj_weight[:, start:end]
        else:
            raise ValueError(f"Unsupported attention_kind: {attention_kind}")
        if metric == "l1":
            score = q_weight.abs().sum() + k_weight.abs().sum() + v_weight.abs().sum()
            score = score + proj_slice.abs().sum()
        elif metric == "l2":
            score = torch.linalg.vector_norm(
                torch.cat(
                    [
                        q_weight.flatten(),
                        k_weight.flatten(),
                        v_weight.flatten(),
                        proj_slice.flatten(),
                    ]
                )
            )
        else:
            raise ValueError(f"Unsupported structured importance metric: {metric}")
        scores.append(float(score.item()))
    return torch.tensor(scores, dtype=torch.float32)


def _build_attention_descriptor(
    module_name: str,
    module: nn.Module,
) -> Optional[dict[str, Any]]:
    num_heads = getattr(module, "num_heads", None)
    head_dim = getattr(module, "head_dim", None)
    embed_dim = getattr(module, "embed_dim", None)
    num_kv_heads = getattr(module, "num_kv_heads", num_heads)
    if (
        not isinstance(num_heads, int)
        or not isinstance(head_dim, int)
        or not isinstance(embed_dim, int)
        or not isinstance(num_kv_heads, int)
    ):
        return None
    if num_kv_heads <= 0 or num_heads <= 0 or num_heads % num_kv_heads != 0:
        return None
    attention_role = _infer_attention_role(module)
    attention_variant = _attention_variant(num_heads=num_heads, num_kv_heads=num_kv_heads)

    qkv = getattr(module, "qkv", None)
    proj = getattr(module, "proj", None)
    if isinstance(qkv, nn.Linear) and isinstance(proj, nn.Linear):
        return {
            "module_name": module_name,
            "attention_kind": "fused_qkv",
            "attention_role": attention_role,
            "attention_variant": attention_variant,
            "module_type": type(module).__name__,
            "num_heads": num_heads,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
            "embed_dim": embed_dim,
            "qkv": qkv,
            "proj": proj,
            "qkv_name": f"{module_name}.qkv",
            "proj_name": f"{module_name}.proj",
        }

    q_proj = getattr(module, "q_proj", None)
    k_proj = getattr(module, "k_proj", None)
    v_proj = getattr(module, "v_proj", None)
    out_proj = getattr(module, "out_proj", None)
    if all(
        isinstance(proj_module, nn.Linear)
        for proj_module in (q_proj, k_proj, v_proj, out_proj)
    ):
        return {
            "module_name": module_name,
            "attention_kind": "split_qkv",
            "attention_role": attention_role,
            "attention_variant": attention_variant,
            "module_type": type(module).__name__,
            "num_heads": num_heads,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
            "embed_dim": embed_dim,
            "q_proj": q_proj,
            "k_proj": k_proj,
            "v_proj": v_proj,
            "out_proj": out_proj,
            "q_proj_name": f"{module_name}.q_proj",
            "k_proj_name": f"{module_name}.k_proj",
            "v_proj_name": f"{module_name}.v_proj",
            "out_proj_name": f"{module_name}.out_proj",
        }
    return None


def _collect_head_candidates(
    model: nn.Module,
    *,
    importance_metric: str,
) -> _CandidateDiscoveryResult:
    candidates: list[_StructuredCandidate] = []
    graph = PruningDependencyGraph()
    for module_name, module in model.named_modules():
        if not module_name:
            continue
        descriptor = _build_attention_descriptor(module_name, module)
        if descriptor is None:
            continue
        scores = _head_scores(descriptor, importance_metric)
        candidates.append(
            _StructuredCandidate(
                adapter="attention_adapter",
                structure_family="transformer",
                action_type="attention_heads",
                module_name=module_name,
                module_type=str(descriptor["module_type"]),
                granularity="head",
                dependency_group=module_name,
                consumer_name=None,
                consumer_type=None,
                normalization_name=None,
                feature_block_size=1,
                scores=scores,
                metadata={
                    "attention_kind": str(descriptor["attention_kind"]),
                    "attention_role": str(descriptor["attention_role"]),
                    "attention_variant": str(descriptor["attention_variant"]),
                    "num_heads": int(descriptor["num_heads"]),
                    "num_kv_heads": int(descriptor["num_kv_heads"]),
                    "head_dim": int(descriptor["head_dim"]),
                    "embed_dim": int(descriptor["embed_dim"]),
                    **{
                        key: value
                        for key, value in descriptor.items()
                        if key.endswith("_name")
                    },
                },
            )
        )
        graph.add_group(
            name=module_name,
            producer=module_name,
            consumers=[],
            merge=None,
            shape_constraints={
                "attention_kind": str(descriptor["attention_kind"]),
                "attention_role": str(descriptor["attention_role"]),
                "attention_variant": str(descriptor["attention_variant"]),
                "num_heads": int(descriptor["num_heads"]),
                "num_kv_heads": int(descriptor["num_kv_heads"]),
                "head_dim": int(descriptor["head_dim"]),
                "embed_dim": int(descriptor["embed_dim"]),
            },
        )
    return _CandidateDiscoveryResult(candidates=candidates, dependency_graph=graph)


def _is_xdl_vit_like_model(module: nn.Module) -> bool:
    patch_embed = getattr(module, "patch_embed", None)
    patch_proj = getattr(patch_embed, "proj", None)
    blocks = getattr(module, "blocks", None)
    norm = getattr(module, "norm", None)
    head = getattr(module, "head", None)
    cls_token = getattr(module, "cls_token", None)
    pos_embed = getattr(module, "pos_embed", None)
    return (
        isinstance(patch_proj, nn.Conv2d)
        and isinstance(blocks, nn.ModuleList)
        and isinstance(norm, nn.LayerNorm)
        and isinstance(cls_token, nn.Parameter)
        and isinstance(pos_embed, nn.Parameter)
        and isinstance(head, (nn.Linear, nn.Identity))
    )


def _vit_hidden_descriptor(
    module_name: str,
    module: nn.Module,
) -> Optional[dict[str, Any]]:
    if not _is_xdl_vit_like_model(module):
        return None
    embed_dim = getattr(module, "embed_dim", None)
    patch_embed = getattr(module, "patch_embed")
    patch_proj = getattr(patch_embed, "proj")
    blocks = getattr(module, "blocks")
    norm = getattr(module, "norm")
    head = getattr(module, "head")
    cls_token = getattr(module, "cls_token")
    pos_embed = getattr(module, "pos_embed")
    if not isinstance(embed_dim, int):
        embed_dim = int(patch_proj.out_channels)
    if (
        patch_proj.out_channels != embed_dim
        or not isinstance(norm, nn.LayerNorm)
        or _layernorm_feature_count(norm) != embed_dim
        or cls_token.shape[-1] != embed_dim
        or pos_embed.shape[-1] != embed_dim
    ):
        return None
    if isinstance(head, nn.Linear) and head.in_features != embed_dim:
        return None

    prefix = "" if module_name in {"", "<root>"} else f"{module_name}."
    block_descriptors: list[dict[str, Any]] = []
    num_heads_values: set[int] = set()
    for block_index, block in enumerate(blocks):
        norm1 = getattr(block, "norm1", None)
        attn = getattr(block, "attn", None)
        norm2 = getattr(block, "norm2", None)
        mlp = getattr(block, "mlp", None)
        qkv = getattr(attn, "qkv", None)
        proj = getattr(attn, "proj", None)
        fc1 = getattr(mlp, "fc1", None)
        fc2 = getattr(mlp, "fc2", None)
        num_heads = getattr(attn, "num_heads", None)
        if not (
            isinstance(norm1, nn.LayerNorm)
            and isinstance(norm2, nn.LayerNorm)
            and isinstance(qkv, nn.Linear)
            and isinstance(proj, nn.Linear)
            and isinstance(fc1, nn.Linear)
            and isinstance(fc2, nn.Linear)
            and isinstance(num_heads, int)
        ):
            return None
        if (
            _layernorm_feature_count(norm1) != embed_dim
            or _layernorm_feature_count(norm2) != embed_dim
            or qkv.in_features != embed_dim
            or qkv.out_features != embed_dim * 3
            or proj.in_features != embed_dim
            or proj.out_features != embed_dim
            or fc1.in_features != embed_dim
            or fc2.out_features != embed_dim
        ):
            return None
        if embed_dim % num_heads != 0:
            return None
        num_heads_values.add(num_heads)
        block_name = f"{prefix}blocks.{block_index}"
        block_descriptors.append(
            {
                "block_name": block_name,
                "norm1_name": f"{block_name}.norm1",
                "attn_name": f"{block_name}.attn",
                "qkv_name": f"{block_name}.attn.qkv",
                "proj_name": f"{block_name}.attn.proj",
                "norm2_name": f"{block_name}.norm2",
                "fc1_name": f"{block_name}.mlp.fc1",
                "fc2_name": f"{block_name}.mlp.fc2",
                "num_heads": num_heads,
                "mlp_hidden_features": int(fc1.out_features),
            }
        )
    if not block_descriptors:
        return None
    if len(num_heads_values) != 1:
        return None
    num_heads = num_heads_values.pop()
    return {
        "model_name": module_name,
        "module_type": type(module).__name__,
        "model_family": "xdl_vit",
        "embed_dim": embed_dim,
        "num_heads": num_heads,
        "head_dim": embed_dim // num_heads,
        "block_count": len(block_descriptors),
        "patch_proj_name": f"{prefix}patch_embed.proj",
        "cls_token_name": f"{prefix}cls_token",
        "pos_embed_name": f"{prefix}pos_embed",
        "norm_name": f"{prefix}norm",
        "head_name": f"{prefix}head" if isinstance(head, nn.Linear) else None,
        "blocks": block_descriptors,
    }


def _vit_hidden_width_scores(
    descriptor: Mapping[str, Any],
    model: nn.Module,
    metric: str,
) -> torch.Tensor:
    embed_dim = int(descriptor["embed_dim"])
    score = torch.zeros(embed_dim, dtype=torch.float32)
    patch_proj = _get_module(model, str(descriptor["patch_proj_name"]))
    norm = _get_module(model, str(descriptor["norm_name"]))
    head_name = descriptor.get("head_name")
    if not isinstance(patch_proj, nn.Conv2d) or not isinstance(norm, nn.LayerNorm):
        raise TypeError("hidden width pruning requires ViT patch projection and LayerNorm")
    patch_weight = patch_proj.weight.detach().to(dtype=torch.float32, device="cpu")
    if metric == "bn_gamma":
        raise ValueError("importance.metric=bn_gamma does not support hidden_width pruning")
    if metric == "l1":
        score = score + patch_weight.abs().sum(dim=(1, 2, 3))
        if patch_proj.bias is not None:
            score = score + patch_proj.bias.detach().abs().to(dtype=torch.float32, device="cpu")
        if norm.weight is not None:
            score = score + norm.weight.detach().abs().to(dtype=torch.float32, device="cpu")
    elif metric == "l2":
        score = score + torch.linalg.vector_norm(patch_weight.reshape(embed_dim, -1), dim=1)
        if patch_proj.bias is not None:
            score = torch.sqrt(
                score.square()
                + patch_proj.bias.detach().to(dtype=torch.float32, device="cpu").square()
            )
    else:
        raise ValueError(f"Unsupported structured importance metric: {metric}")

    if isinstance(head_name, str):
        head = _get_module(model, head_name)
        if isinstance(head, nn.Linear):
            head_weight = head.weight.detach().to(dtype=torch.float32, device="cpu")
            if metric == "l1":
                score = score + head_weight.abs().sum(dim=0)
            elif metric == "l2":
                score = torch.sqrt(
                    score.square()
                    + torch.linalg.vector_norm(head_weight, dim=0).square()
                )

    blocks = descriptor["blocks"]
    if not isinstance(blocks, list):
        raise TypeError("descriptor.blocks must be a list")
    for block in blocks:
        if not isinstance(block, Mapping):
            raise TypeError("descriptor.blocks entries must be mappings")
        qkv = _get_module(model, str(block["qkv_name"]))
        proj = _get_module(model, str(block["proj_name"]))
        fc1 = _get_module(model, str(block["fc1_name"]))
        fc2 = _get_module(model, str(block["fc2_name"]))
        norm1 = _get_module(model, str(block["norm1_name"]))
        norm2 = _get_module(model, str(block["norm2_name"]))
        if not all(isinstance(module, nn.Linear) for module in (qkv, proj, fc1, fc2)):
            raise TypeError("hidden width pruning requires qkv/proj/fc1/fc2 Linear modules")
        if not isinstance(norm1, nn.LayerNorm) or not isinstance(norm2, nn.LayerNorm):
            raise TypeError("hidden width pruning requires LayerNorm modules")
        qkv_weight = qkv.weight.detach().to(dtype=torch.float32, device="cpu")
        proj_weight = proj.weight.detach().to(dtype=torch.float32, device="cpu")
        fc1_weight = fc1.weight.detach().to(dtype=torch.float32, device="cpu")
        fc2_weight = fc2.weight.detach().to(dtype=torch.float32, device="cpu")
        if metric == "l1":
            qkv_out = qkv_weight.reshape(3, embed_dim, embed_dim).abs().sum(dim=(0, 2))
            score = score + qkv_weight.abs().sum(dim=0)
            score = score + qkv_out
            score = score + proj_weight.abs().sum(dim=0)
            score = score + proj_weight.abs().sum(dim=1)
            score = score + fc1_weight.abs().sum(dim=0)
            score = score + fc2_weight.abs().sum(dim=1)
            if norm1.weight is not None:
                score = score + norm1.weight.detach().abs().to(dtype=torch.float32, device="cpu")
            if norm2.weight is not None:
                score = score + norm2.weight.detach().abs().to(dtype=torch.float32, device="cpu")
            continue
        if metric == "l2":
            qkv_out = torch.linalg.vector_norm(
                qkv_weight.reshape(3, embed_dim, embed_dim),
                dim=2,
            ).square().sum(dim=0)
            score = torch.sqrt(
                score.square()
                + torch.linalg.vector_norm(qkv_weight, dim=0).square()
                + qkv_out
                + torch.linalg.vector_norm(proj_weight, dim=0).square()
                + torch.linalg.vector_norm(proj_weight, dim=1).square()
                + torch.linalg.vector_norm(fc1_weight, dim=0).square()
                + torch.linalg.vector_norm(fc2_weight, dim=1).square()
            )
            continue
        raise ValueError(f"Unsupported structured importance metric: {metric}")
    return score


def _collect_vit_hidden_width_candidates(
    model: nn.Module,
    *,
    importance_metric: str,
) -> _CandidateDiscoveryResult:
    candidates: list[_StructuredCandidate] = []
    graph = PruningDependencyGraph()
    for module_name, module in model.named_modules():
        descriptor_module_name = module_name or "<root>"
        descriptor = _vit_hidden_descriptor(descriptor_module_name, module)
        if descriptor is None:
            continue
        scores = _vit_hidden_width_scores(descriptor, model, importance_metric)
        embed_dim = int(descriptor["embed_dim"])
        num_heads = int(descriptor["num_heads"])
        candidates.append(
            _StructuredCandidate(
                adapter="vit_hidden_width_adapter",
                structure_family="transformer_width",
                action_type="vit_hidden_width",
                module_name=descriptor_module_name,
                module_type=str(descriptor["module_type"]),
                granularity="hidden_width",
                dependency_group=descriptor_module_name,
                consumer_name=str(descriptor["head_name"]) if descriptor.get("head_name") else None,
                consumer_type="Linear" if descriptor.get("head_name") else None,
                normalization_name=str(descriptor["norm_name"]),
                feature_block_size=1,
                scores=scores,
                metadata={
                    **dict(descriptor),
                    "group_alignment_constraints": [
                        {
                            "module_name": descriptor_module_name,
                            "axis": "hidden",
                            "groups": num_heads,
                            "total_channels": embed_dim,
                            "channels_per_group": embed_dim // num_heads,
                        }
                    ],
                    "width_kind": "hidden",
                },
            )
        )
        graph.add_group(
            name=descriptor_module_name,
            producer=str(descriptor["patch_proj_name"]),
            consumers=[
                str(descriptor["norm_name"]),
                *[
                    str(block["block_name"])
                    for block in descriptor["blocks"]
                    if isinstance(block, Mapping)
                ],
            ],
            merge="residual",
            shape_constraints={
                "model_family": str(descriptor["model_family"]),
                "embed_dim": int(descriptor["embed_dim"]),
                "num_heads": int(descriptor["num_heads"]),
                "block_count": int(descriptor["block_count"]),
                "global_residual_width": True,
            },
        )
    return _CandidateDiscoveryResult(candidates=candidates, dependency_graph=graph)


def _collect_vit_embedding_width_candidates(
    model: nn.Module,
    *,
    importance_metric: str,
) -> _CandidateDiscoveryResult:
    result = _collect_vit_hidden_width_candidates(
        model,
        importance_metric=importance_metric,
    )
    for candidate in result.candidates:
        candidate.adapter = "vit_embedding_width_adapter"
        candidate.granularity = "embedding_width"
        candidate.metadata["width_kind"] = "embedding"
    return result


def _expert_usage_scores(module: nn.Module, router: nn.Linear, metric: str) -> torch.Tensor:
    raw_usage = getattr(module, "expert_usage", None)
    if raw_usage is None:
        raw_usage = getattr(module, "expert_usage_counts", None)
    if raw_usage is None:
        raw_usage = getattr(module, "router_usage", None)
    if raw_usage is not None:
        usage = torch.as_tensor(raw_usage, dtype=torch.float32, device="cpu").flatten()
        if int(usage.numel()) != router.out_features:
            raise ValueError("expert usage length must match router.out_features")
        return usage
    if metric == "usage":
        if router.bias is None:
            raise ValueError("importance.metric=usage requires expert_usage or router bias")
        return torch.softmax(
            router.bias.detach().to(dtype=torch.float32, device="cpu"),
            dim=0,
        )
    weight = router.weight.detach().to(dtype=torch.float32, device="cpu")
    if metric == "l1":
        score = weight.abs().sum(dim=1)
        if router.bias is not None:
            score = score + router.bias.detach().abs().to(dtype=torch.float32, device="cpu")
        return score
    if metric == "l2":
        score = torch.linalg.vector_norm(weight, dim=1)
        if router.bias is not None:
            score = torch.sqrt(
                score.square()
                + router.bias.detach().to(dtype=torch.float32, device="cpu").square()
            )
        return score
    raise ValueError(f"Unsupported structured importance metric: {metric}")


def _collect_expert_candidates(
    model: nn.Module,
    *,
    importance_metric: str,
) -> _CandidateDiscoveryResult:
    candidates: list[_StructuredCandidate] = []
    graph = PruningDependencyGraph()
    for module_name, module in model.named_modules():
        descriptor_module_name = module_name or "<root>"
        prefix = "" if descriptor_module_name == "<root>" else f"{module_name}."
        router = getattr(module, "router", None)
        experts = getattr(module, "experts", None)
        if not isinstance(router, nn.Linear):
            continue
        if not isinstance(experts, (nn.ModuleList, nn.Sequential)):
            continue
        expert_modules = list(experts.children())
        if len(expert_modules) <= 1:
            continue
        if router.out_features != len(expert_modules):
            raise ValueError(
                f"MoE module '{descriptor_module_name}' router.out_features must match expert count"
            )
        scores = _expert_usage_scores(module, router, importance_metric)
        expert_names = [f"{prefix}experts.{index}" for index in range(len(expert_modules))]
        metadata = {
            "router_name": f"{prefix}router",
            "experts_name": f"{prefix}experts",
            "expert_names": expert_names,
            "expert_count": len(expert_modules),
            "usage_scores": [float(value) for value in scores.tolist()],
            "usage_source": (
                "module_usage"
                if any(
                    getattr(module, attr_name, None) is not None
                    for attr_name in ("expert_usage", "expert_usage_counts", "router_usage")
                )
                else ("router_bias" if importance_metric == "usage" else "router_weight")
            ),
        }
        candidates.append(
            _StructuredCandidate(
                adapter="moe_expert_adapter",
                structure_family="moe_expert",
                action_type="drop_experts",
                module_name=descriptor_module_name,
                module_type=type(module).__name__,
                granularity="expert",
                dependency_group=descriptor_module_name,
                consumer_name=f"{prefix}router",
                consumer_type="Linear",
                normalization_name=None,
                feature_block_size=1,
                scores=scores,
                metadata=metadata,
            )
        )
        graph.add_group(
            name=descriptor_module_name,
            producer=f"{prefix}router",
            consumers=[f"{prefix}experts"],
            merge="router",
            shape_constraints={
                "expert_count": len(expert_modules),
                "router_out_features": int(router.out_features),
                "usage_scores": [float(value) for value in scores.tolist()],
            },
        )
    return _CandidateDiscoveryResult(candidates=candidates, dependency_graph=graph)


def _block_scores(
    block: nn.Module,
    metric: str,
) -> float:
    parameters = [
        parameter.detach().to(dtype=torch.float32, device="cpu").flatten()
        for parameter in block.parameters()
    ]
    if not parameters:
        return 0.0
    flat = torch.cat(parameters)
    if metric == "l1":
        return float(flat.abs().sum().item())
    if metric == "l2":
        return float(torch.linalg.vector_norm(flat).item())
    raise ValueError(f"Unsupported structured importance metric: {metric}")


def _is_cnn_stage_passthrough_module(module: nn.Module) -> bool:
    return isinstance(
        module,
        (
            nn.ReLU,
            nn.ReLU6,
            nn.GELU,
            nn.SiLU,
            nn.Identity,
            nn.Dropout,
            nn.Dropout2d,
            nn.Dropout3d,
        ),
    )


def _conv2d_preserves_spatial_shape(module: nn.Conv2d) -> bool:
    if module.stride != (1, 1):
        return False
    if isinstance(module.padding, str):
        return module.padding == "same"
    return all(
        2 * int(module.padding[index])
        == int(module.dilation[index]) * (int(module.kernel_size[index]) - 1)
        for index in range(2)
    )


def _cnn_stage_descriptor(
    stage_name: str,
    stage: nn.Module,
) -> Optional[dict[str, Any]]:
    leaves = [
        (leaf_name, leaf_module)
        for leaf_name, leaf_module in stage.named_modules()
        if leaf_name and not any(leaf_module.children())
    ]
    if not leaves:
        return None

    convs: list[tuple[str, nn.Conv2d]] = []
    batchnorm_features: list[int] = []
    for leaf_name, leaf_module in leaves:
        if isinstance(leaf_module, nn.Conv2d):
            if not _conv2d_preserves_spatial_shape(leaf_module):
                return None
            convs.append((leaf_name, leaf_module))
            continue
        if isinstance(leaf_module, nn.modules.batchnorm._BatchNorm):
            batchnorm_features.append(int(leaf_module.num_features))
            continue
        if _is_cnn_stage_passthrough_module(leaf_module):
            continue
        return None

    if not convs:
        return None
    input_channels = int(convs[0][1].in_channels)
    output_channels = int(convs[-1][1].out_channels)
    if input_channels != output_channels:
        return None
    conv_output_channels = {int(conv.out_channels) for _, conv in convs}
    if any(num_features not in conv_output_channels for num_features in batchnorm_features):
        return None
    return {
        "stage_name": stage_name,
        "stage_type": type(stage).__name__,
        "input_channels": input_channels,
        "output_channels": output_channels,
        "conv_count": len(convs),
        "conv_names": [f"{stage_name}.{leaf_name}" for leaf_name, _ in convs],
        "shape_compatible": True,
    }


def _collect_cnn_stage_candidates(
    model: nn.Module,
    *,
    importance_metric: str,
) -> _CandidateDiscoveryResult:
    candidates: list[_StructuredCandidate] = []
    graph = PruningDependencyGraph()
    for module_name, module in model.named_modules():
        if not module_name:
            continue
        if not isinstance(module, (nn.ModuleList, nn.Sequential)):
            continue
        named_children = list(module.named_children())
        if len(named_children) <= 1:
            continue

        descriptors: list[dict[str, Any]] = []
        for child_name, child in named_children:
            descriptor = _cnn_stage_descriptor(f"{module_name}.{child_name}", child)
            if descriptor is None:
                descriptors = []
                break
            descriptors.append(descriptor)
        if not descriptors:
            continue

        input_channels = {int(descriptor["input_channels"]) for descriptor in descriptors}
        output_channels = {int(descriptor["output_channels"]) for descriptor in descriptors}
        if len(input_channels) != 1 or len(output_channels) != 1:
            continue
        if input_channels != output_channels:
            continue

        stage_types = [str(descriptor["stage_type"]) for descriptor in descriptors]
        heterogeneous_children = len(set(stage_types)) != 1
        children = [child for _, child in named_children]
        scores = torch.tensor(
            [_block_scores(child, importance_metric) for child in children],
            dtype=torch.float32,
        )
        metadata = {
            "container_type": type(module).__name__,
            "stage_type": stage_types[0] if not heterogeneous_children else "heterogeneous",
            "stage_types": list(stage_types),
            "heterogeneous_children": heterogeneous_children,
            "stage_count": len(descriptors),
            "stage_names": [str(descriptor["stage_name"]) for descriptor in descriptors],
            "input_channels": input_channels.pop(),
            "output_channels": output_channels.pop(),
            "shape_compatible": True,
            "compatibility_rule": "same input/output channels and spatial-preserving Conv2d leaves",
            "stage_descriptors": [dict(descriptor) for descriptor in descriptors],
        }
        candidates.append(
            _StructuredCandidate(
                adapter="cnn_stage_adapter",
                structure_family="cnn_stage",
                action_type="drop_stages",
                module_name=module_name,
                module_type=type(module).__name__,
                granularity="stage",
                dependency_group=module_name,
                consumer_name=None,
                consumer_type=None,
                normalization_name=None,
                feature_block_size=1,
                scores=scores,
                metadata=metadata,
            )
        )
        graph.add_group(
            name=module_name,
            producer=module_name,
            consumers=[],
            merge=None,
            shape_constraints={
                "container_type": type(module).__name__,
                "stage_count": len(descriptors),
                "input_channels": int(metadata["input_channels"]),
                "output_channels": int(metadata["output_channels"]),
                "shape_compatible": True,
                "compatibility_rule": str(metadata["compatibility_rule"]),
            },
        )
    return _CandidateDiscoveryResult(candidates=candidates, dependency_graph=graph)


def _is_composite_block_module(module: nn.Module) -> bool:
    return any(True for _ in module.children())


def _collect_block_candidates(
    model: nn.Module,
    *,
    importance_metric: str,
) -> _CandidateDiscoveryResult:
    candidates: list[_StructuredCandidate] = []
    graph = PruningDependencyGraph()
    for module_name, module in model.named_modules():
        if not module_name:
            continue
        if not isinstance(module, (nn.ModuleList, nn.Sequential)):
            continue
        children = list(module.children())
        if len(children) <= 1:
            continue
        if not all(_is_composite_block_module(child) for child in children):
            continue
        block_types = [child.__class__.__name__ for child in children]
        heterogeneous_children = len(set(block_types)) != 1
        scores = torch.tensor(
            [_block_scores(child, importance_metric) for child in children],
            dtype=torch.float32,
        )
        candidates.append(
            _StructuredCandidate(
                adapter="container_adapter",
                structure_family="container",
                action_type="drop_blocks",
                module_name=module_name,
                module_type=type(module).__name__,
                granularity="block",
                dependency_group=module_name,
                consumer_name=None,
                consumer_type=None,
                normalization_name=None,
                feature_block_size=1,
                scores=scores,
                metadata={
                    "container_type": type(module).__name__,
                    "block_type": block_types[0] if not heterogeneous_children else "heterogeneous",
                    "block_types": list(block_types),
                    "heterogeneous_children": heterogeneous_children,
                },
            )
        )
        graph.add_group(
            name=module_name,
            producer=module_name,
            consumers=[],
            merge=None,
            shape_constraints={
                "container_type": type(module).__name__,
                "block_type": block_types[0] if not heterogeneous_children else "heterogeneous",
                "block_types": list(block_types),
                "heterogeneous_children": heterogeneous_children,
            },
        )
    return _CandidateDiscoveryResult(candidates=candidates, dependency_graph=graph)


_STRUCTURED_ADAPTERS: tuple[_StructuredPruningAdapter, ...] = (
    _StructuredPruningAdapter(
        name="cnn_chain_adapter",
        granularity="channel",
        structure_family="cnn",
        collect=_collect_conv_candidates,
    ),
    _StructuredPruningAdapter(
        name="mbconv_adapter",
        granularity="channel",
        structure_family="cnn_mbconv",
        collect=_collect_mbconv_candidates,
    ),
    _StructuredPruningAdapter(
        name="concat_branch_adapter",
        granularity="channel",
        structure_family="cnn_branch",
        collect=_collect_concat_branch_candidates,
    ),
    _StructuredPruningAdapter(
        name="residual_cnn_adapter",
        granularity="channel",
        structure_family="cnn_residual",
        collect=_collect_residual_conv_candidates,
    ),
    _StructuredPruningAdapter(
        name="cnn_chain_adapter",
        granularity="filter",
        structure_family="cnn",
        collect=_collect_conv_candidates,
    ),
    _StructuredPruningAdapter(
        name="mlp_pair_adapter",
        granularity="mlp_neuron",
        structure_family="transformer",
        collect=_collect_mlp_candidates,
    ),
    _StructuredPruningAdapter(
        name="attention_adapter",
        granularity="head",
        structure_family="transformer",
        collect=_collect_head_candidates,
    ),
    _StructuredPruningAdapter(
        name="container_adapter",
        granularity="block",
        structure_family="container",
        collect=_collect_block_candidates,
    ),
    _StructuredPruningAdapter(
        name="cnn_stage_adapter",
        granularity="stage",
        structure_family="cnn_stage",
        collect=_collect_cnn_stage_candidates,
    ),
    _StructuredPruningAdapter(
        name="vit_hidden_width_adapter",
        granularity="hidden_width",
        structure_family="transformer_width",
        collect=_collect_vit_hidden_width_candidates,
    ),
    _StructuredPruningAdapter(
        name="vit_embedding_width_adapter",
        granularity="embedding_width",
        structure_family="transformer_width",
        collect=_collect_vit_embedding_width_candidates,
    ),
    _StructuredPruningAdapter(
        name="moe_expert_adapter",
        granularity="expert",
        structure_family="moe_expert",
        collect=_collect_expert_candidates,
    ),
)


def _collect_candidates(
    model: nn.Module,
    *,
    granularity: str,
    importance_metric: str,
) -> _CandidateDiscoveryResult:
    matching_adapters = [adapter for adapter in _STRUCTURED_ADAPTERS if adapter.granularity == granularity]
    if not matching_adapters:
        allowed = ", ".join(SUPPORTED_GRANULARITIES)
        raise ValueError(f"Unsupported structured granularity '{granularity}'. Allowed: {allowed}")

    combined = _CandidateDiscoveryResult()
    matched = False
    for adapter in _STRUCTURED_ADAPTERS:
        if adapter.granularity != granularity:
            continue
        matched = True
        result = adapter.collect(model, importance_metric=importance_metric)
        combined.candidates.extend(result.candidates)
        combined.blocked_modules.extend(result.blocked_modules)
        combined.dependency_graph.groups.extend(result.dependency_graph.groups)
    if not matched:
        allowed = ", ".join(SUPPORTED_GRANULARITIES)
        raise ValueError(f"Unsupported structured granularity '{granularity}'. Allowed: {allowed}")
    return combined


def _validate_candidate_dependencies(
    candidates: list[_StructuredCandidate],
    dependency_graph: PruningDependencyGraph,
) -> None:
    candidate_groups = {candidate.dependency_group for candidate in candidates}
    missing_groups = sorted(
        group_name
        for group_name in candidate_groups
        if dependency_graph.find_group(group_name) is None
    )
    if missing_groups:
        raise ValueError(
            "Structured pruning candidates are missing dependency metadata for groups: "
            + ", ".join(missing_groups)
        )


def _validate_action_keep_indices(model: nn.Module, action: StructuredPruningAction) -> None:
    if action.action_type == "residual_stage_channels":
        consumers = action.metadata.get("consumers")
        if consumers is None:
            raise ValueError("residual_stage_channels requires metadata.consumers")
        if not isinstance(consumers, list):
            raise ValueError("metadata.consumers must be a list")
        for item in consumers:
            if not isinstance(item, Mapping):
                raise ValueError("metadata.consumers entries must be mappings")
            consumer_name = item.get("name")
            consumer_type = item.get("type")
            if consumer_type != "Conv2d" or not isinstance(consumer_name, str):
                continue
            consumer = _get_module(model, consumer_name)
            if not isinstance(consumer, nn.Conv2d):
                raise TypeError(f"{consumer_name} is not a Conv2d module")
            validate_conv2d_keep_indices(consumer, action.keep_indices, axis="in")
        return
    if action.action_type == "attention_heads":
        attention = _get_module(model, action.module_name)
        num_heads = getattr(attention, "num_heads", None)
        num_kv_heads = getattr(attention, "num_kv_heads", num_heads)
        if not isinstance(num_heads, int) or not isinstance(num_kv_heads, int):
            raise TypeError(f"{action.module_name} is not a supported attention module")
        attention_variant = str(action.metadata.get("attention_variant", "mha"))
        if attention_variant in {"gqa", "mqa"}:
            if num_kv_heads <= 0 or num_heads % num_kv_heads != 0:
                raise ValueError(
                    f"{action.module_name} has invalid num_heads/num_kv_heads configuration"
                )
            if action.keep_indices[0] < 0 or action.keep_indices[-1] >= num_kv_heads:
                raise ValueError(
                    f"selection.keep_indices for '{action.module_name}' must reference kv heads"
                )
        return
    if action.action_type == "drop_stages":
        container = _get_module(model, action.module_name)
        if not isinstance(container, (nn.ModuleList, nn.Sequential)):
            raise TypeError(
                f"{action.module_name} is not a ModuleList or Sequential stage container"
            )
        if not bool(action.metadata.get("shape_compatible", False)):
            raise ValueError(
                f"CNN stage pruning candidate '{action.module_name}' is not shape compatible"
            )
        return
    if action.action_type == "vit_hidden_width":
        num_heads = int(action.metadata.get("num_heads", 0))
        if num_heads <= 0:
            raise ValueError("vit_hidden_width requires metadata.num_heads")
        if len(action.keep_indices) % num_heads != 0:
            raise ValueError("vit_hidden_width keep count must be divisible by num_heads")
        return
    if action.action_type == "drop_experts":
        router = _get_module(model, str(action.metadata["router_name"]))
        experts = _get_module(model, str(action.metadata["experts_name"]))
        if not isinstance(router, nn.Linear):
            raise TypeError("drop_experts requires a Linear router")
        if not isinstance(experts, (nn.ModuleList, nn.Sequential)):
            raise TypeError("drop_experts requires an expert container")
        if router.out_features != len(list(experts.children())):
            raise ValueError("router.out_features must match expert count")
        return
    if action.action_type != "conv_channel_group":
        return
    producer = _get_module(model, action.module_name)
    if not isinstance(producer, nn.Conv2d):
        raise TypeError(f"{action.module_name} is not a Conv2d module")
    validate_conv2d_keep_indices(producer, action.keep_indices, axis="out")
    if action.consumer_name is None or action.consumer_type != "Conv2d":
        return
    consumer = _get_module(model, action.consumer_name)
    if not isinstance(consumer, nn.Conv2d):
        raise TypeError(f"{action.consumer_name} is not a Conv2d module")
    validate_conv2d_keep_indices(consumer, action.keep_indices, axis="in")


def find_structured_pruning_targets(
    model: nn.Module,
    *,
    granularity: str = "channel",
    importance: Optional[Mapping[str, Any]] = None,
) -> list[PruningTarget]:
    """Enumerate supported structured pruning targets for one granularity."""

    importance_config = dict(importance or {})
    importance_metric = str(
        importance_config.get("metric", importance_config.get("type", "l1"))
    )
    discovery = _collect_candidates(
        model,
        granularity=granularity,
        importance_metric=importance_metric,
    )
    _validate_candidate_dependencies(discovery.candidates, discovery.dependency_graph)
    return [_candidate_to_target(candidate) for candidate in discovery.candidates]


def _nm_group_compliance(
    flat_tensor: torch.Tensor,
    *,
    pattern_n: int,
    pattern_m: int,
) -> tuple[int, int]:
    total_groups = int(flat_tensor.numel()) // pattern_m
    if total_groups == 0:
        return 0, 0
    groups = flat_tensor[: total_groups * pattern_m].reshape(total_groups, pattern_m)
    zero_counts = torch.count_nonzero(groups == 0, dim=1)
    compliant_groups = int(torch.count_nonzero(zero_counts == (pattern_m - pattern_n)).item())
    return compliant_groups, total_groups


def apply_nm_structured_sparsity(
    model: nn.Module,
    *,
    pattern_n: int,
    pattern_m: int,
    module_types: tuple[type[nn.Module], ...] = (nn.Linear, nn.Conv2d),
    parameter_name: str = "weight",
) -> NMStructuredPruningReport:
    """Apply in-place N:M structured sparsity to supported module weights."""

    if pattern_n <= 0 or pattern_m <= 0:
        raise ValueError("pattern_n and pattern_m must be positive")
    if pattern_n >= pattern_m:
        raise ValueError("pattern_n must be smaller than pattern_m")

    parameter_count_before = _count_parameters(model)
    zero_before = _count_zero_parameters(
        model,
        module_types=module_types,
        parameter_name=parameter_name,
    )
    layer_reports: list[NMStructuredLayerReport] = []

    for module_name, module in model.named_modules():
        if not isinstance(module, module_types):
            continue
        parameter = getattr(module, parameter_name, None)
        if not isinstance(parameter, torch.Tensor):
            continue

        weight = parameter.data
        flat = weight.view(-1)
        usable = (flat.numel() // pattern_m) * pattern_m
        if usable == 0:
            layer_reports.append(
                NMStructuredLayerReport(
                    module_name=module_name or "<root>",
                    module_type=type(module).__name__,
                    parameter_name=parameter_name,
                    total_parameters=int(flat.numel()),
                    zero_parameters=_tensor_zero_count(weight),
                    sparsity=float(_tensor_zero_count(weight) / flat.numel()) if flat.numel() else 0.0,
                    pattern_n=pattern_n,
                    pattern_m=pattern_m,
                    compliant_groups=0,
                    total_groups=0,
                    compliance_ratio=0.0,
                )
            )
            continue

        groups = flat[:usable].view(-1, pattern_m)
        _, prune_indices = torch.topk(
            groups.abs(),
            k=pattern_m - pattern_n,
            dim=1,
            largest=False,
        )
        groups.scatter_(1, prune_indices, 0.0)

        zero_parameters = _tensor_zero_count(weight)
        compliant_groups, total_groups = _nm_group_compliance(
            flat,
            pattern_n=pattern_n,
            pattern_m=pattern_m,
        )
        layer_reports.append(
            NMStructuredLayerReport(
                module_name=module_name or "<root>",
                module_type=type(module).__name__,
                parameter_name=parameter_name,
                total_parameters=int(flat.numel()),
                zero_parameters=zero_parameters,
                sparsity=float(zero_parameters / flat.numel()) if flat.numel() else 0.0,
                pattern_n=pattern_n,
                pattern_m=pattern_m,
                compliant_groups=compliant_groups,
                total_groups=total_groups,
                compliance_ratio=(compliant_groups / total_groups) if total_groups else 0.0,
            )
        )

    return NMStructuredPruningReport(
        method="nm_structured",
        granularity="nm",
        parameter_count_before=parameter_count_before,
        parameter_count_after=_count_parameters(model),
        zero_parameters_before=zero_before,
        zero_parameters_after=_count_zero_parameters(
            model,
            module_types=module_types,
            parameter_name=parameter_name,
        ),
        pattern_n=pattern_n,
        pattern_m=pattern_m,
        module_types=[module_type.__name__ for module_type in module_types],
        layers=layer_reports,
    )


def _block_sparse_matrix_view(weight: torch.Tensor) -> torch.Tensor:
    if weight.ndim == 2:
        return weight
    if weight.ndim == 4:
        return weight.reshape(weight.shape[0], -1)
    raise ValueError("block_sparse pruning only supports 2D Linear or 4D Conv2d weights")


def _block_sparse_zero_blocks(
    matrix: torch.Tensor,
    *,
    block_rows: int,
    block_cols: int,
) -> tuple[int, int]:
    usable_rows = (matrix.shape[0] // block_rows) * block_rows
    usable_cols = (matrix.shape[1] // block_cols) * block_cols
    if usable_rows == 0 or usable_cols == 0:
        return 0, 0
    block_view = matrix[:usable_rows, :usable_cols].reshape(
        usable_rows // block_rows,
        block_rows,
        usable_cols // block_cols,
        block_cols,
    )
    block_view = block_view.permute(0, 2, 1, 3).reshape(-1, block_rows * block_cols)
    total_blocks = int(block_view.shape[0])
    zero_blocks = int(torch.count_nonzero(torch.count_nonzero(block_view, dim=1) == 0).item())
    return zero_blocks, total_blocks


def apply_block_sparse_pruning(
    model: nn.Module,
    *,
    target_sparsity: float,
    block_shape: tuple[int, int] = (4, 4),
    module_types: tuple[type[nn.Module], ...] = (nn.Linear, nn.Conv2d),
    parameter_name: str = "weight",
) -> BlockSparsePruningReport:
    """Apply in-place block-sparse pruning to supported module weights."""

    if target_sparsity < 0.0 or target_sparsity > 1.0:
        raise ValueError("target_sparsity must be in [0, 1]")
    block_rows, block_cols = int(block_shape[0]), int(block_shape[1])
    if block_rows <= 0 or block_cols <= 0:
        raise ValueError("block_shape values must be positive")

    parameter_count_before = _count_parameters(model)
    zero_before = _count_zero_parameters(
        model,
        module_types=module_types,
        parameter_name=parameter_name,
    )
    layer_reports: list[BlockSparseLayerReport] = []

    for module_name, module in model.named_modules():
        if not isinstance(module, module_types):
            continue
        parameter = getattr(module, parameter_name, None)
        if not isinstance(parameter, torch.Tensor):
            continue
        matrix = _block_sparse_matrix_view(parameter.data)
        usable_rows = (matrix.shape[0] // block_rows) * block_rows
        usable_cols = (matrix.shape[1] // block_cols) * block_cols
        zero_blocks_before, total_blocks = _block_sparse_zero_blocks(
            matrix,
            block_rows=block_rows,
            block_cols=block_cols,
        )
        pruned_blocks = 0
        if total_blocks > 0:
            block_region = matrix[:usable_rows, :usable_cols].reshape(
                usable_rows // block_rows,
                block_rows,
                usable_cols // block_cols,
                block_cols,
            )
            block_region = block_region.permute(0, 2, 1, 3)
            flat_blocks = block_region.detach().reshape(total_blocks, block_rows, block_cols)
            block_scores = flat_blocks.abs().sum(dim=(1, 2))
            existing_zero_blocks = torch.count_nonzero(flat_blocks, dim=(1, 2)) == 0
            block_scores = block_scores.masked_fill(existing_zero_blocks, float("inf"))
            target_zero_blocks = int(round(total_blocks * target_sparsity))
            nonzero_block_count = total_blocks - zero_blocks_before
            prune_count = max(0, min(target_zero_blocks - zero_blocks_before, nonzero_block_count))
            if prune_count > 0:
                ranked = torch.argsort(block_scores)[:prune_count]
                row_block_count = usable_rows // block_rows
                col_block_count = usable_cols // block_cols
                for flat_index in ranked.tolist():
                    row_block_index = int(flat_index) // col_block_count
                    col_block_index = int(flat_index) % col_block_count
                    row_start = row_block_index * block_rows
                    col_start = col_block_index * block_cols
                    matrix[
                        row_start : row_start + block_rows,
                        col_start : col_start + block_cols,
                    ] = 0
                pruned_blocks = int(prune_count)

        zero_blocks_after, total_blocks_after = _block_sparse_zero_blocks(
            matrix,
            block_rows=block_rows,
            block_cols=block_cols,
        )
        zero_parameters = _tensor_zero_count(parameter.data)
        total_parameters = int(parameter.data.numel())
        layer_reports.append(
            BlockSparseLayerReport(
                module_name=module_name or "<root>",
                module_type=type(module).__name__,
                parameter_name=parameter_name,
                block_shape=(block_rows, block_cols),
                total_blocks=total_blocks_after,
                zero_blocks=zero_blocks_after,
                pruned_blocks=pruned_blocks,
                block_sparsity=(
                    zero_blocks_after / total_blocks_after if total_blocks_after else 0.0
                ),
                total_parameters=total_parameters,
                zero_parameters=zero_parameters,
                parameter_sparsity=(
                    zero_parameters / total_parameters if total_parameters else 0.0
                ),
            )
        )

    return BlockSparsePruningReport(
        method="block_sparse",
        granularity="block_sparse",
        target_sparsity=target_sparsity,
        block_shape=(block_rows, block_cols),
        parameter_count_before=parameter_count_before,
        parameter_count_after=_count_parameters(model),
        zero_parameters_before=zero_before,
        zero_parameters_after=_count_zero_parameters(
            model,
            module_types=module_types,
            parameter_name=parameter_name,
        ),
        module_types=[module_type.__name__ for module_type in module_types],
        layers=layer_reports,
    )


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


def plan_structured_pruning(
    model: nn.Module,
    target_sparsity: float,
    *,
    granularity: str = "channel",
    scope: str = "global",
    importance: Optional[Mapping[str, Any]] = None,
    selection: Optional[Mapping[str, Any]] = None,
) -> StructuredPruningPlan:
    """Plan simple Conv2d channel or filter pruning for chain-like CNNs."""

    if target_sparsity < 0.0 or target_sparsity > 1.0:
        raise ValueError("target_sparsity must be in [0, 1]")
    if granularity not in SUPPORTED_GRANULARITIES:
        allowed = ", ".join(SUPPORTED_GRANULARITIES)
        raise ValueError(f"Unsupported structured granularity '{granularity}'. Allowed: {allowed}")
    if scope not in SUPPORTED_SCOPES:
        allowed = ", ".join(SUPPORTED_SCOPES)
        raise ValueError(f"Unsupported structured scope '{scope}'. Allowed: {allowed}")

    importance_config = dict(importance or {})
    selection_config = dict(selection or {})
    importance_metric = str(
        importance_config.get("metric", importance_config.get("type", "l1"))
    )
    min_keep = int(selection_config.get("min_keep", 1))
    if min_keep <= 0:
        raise ValueError("selection.min_keep must be positive")
    if importance_metric not in SUPPORTED_IMPORTANCE_METRICS:
        allowed = ", ".join(SUPPORTED_IMPORTANCE_METRICS)
        raise ValueError(
            f"Unsupported structured importance metric '{importance_metric}'. Allowed: {allowed}"
        )
    if importance_metric == "bn_gamma" and granularity not in {"channel", "filter"}:
        raise ValueError("importance.metric=bn_gamma only supports channel/filter pruning")
    if importance_metric == "usage" and granularity != "expert":
        raise ValueError("importance.metric=usage only supports expert pruning")

    discovery = _collect_candidates(
        model,
        granularity=granularity,
        importance_metric=importance_metric,
    )
    candidates = discovery.candidates
    _validate_candidate_dependencies(candidates, discovery.dependency_graph)
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

    actions: list[StructuredPruningAction] = []
    for candidate, keep_indices in zip(candidates, keep_by_candidate):
        original_units = int(candidate.scores.numel())
        keep = sorted(int(index) for index in keep_indices)
        prune = [index for index in range(original_units) if index not in set(keep)]
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
        _validate_action_keep_indices(model, action)
        actions.append(action)

    return StructuredPruningPlan(
        method="structured",
        granularity=granularity,
        scope=scope,
        target_sparsity=target_sparsity,
        importance_metric=importance_metric,
        adapters=sorted({candidate.adapter for candidate in candidates}),
        structure_families=sorted({candidate.structure_family for candidate in candidates}),
        blocked_modules=sorted(set(discovery.blocked_modules)),
        dependency_graph=discovery.dependency_graph.to_dict(),
        topology_changes=_topology_changes_from_actions(actions),
        targets=targets,
        actions=actions,
    )


def _expand_linear_keep_indices(
    keep_indices: list[int],
    *,
    block_size: int,
) -> list[int]:
    expanded: list[int] = []
    for keep_index in keep_indices:
        start = keep_index * block_size
        expanded.extend(range(start, start + block_size))
    return expanded


def _topology_changes_from_actions(
    actions: list[StructuredPruningAction],
) -> list[dict[str, Any]]:
    changes: list[dict[str, Any]] = []
    for action in actions:
        if not action.prune_indices:
            continue
        if action.action_type == "drop_experts":
            usage_scores = action.metadata.get("usage_scores", [])
            expert_names = action.metadata.get("expert_names", [])
            if not isinstance(usage_scores, list):
                usage_scores = []
            if not isinstance(expert_names, list):
                expert_names = []
            changes.append(
                {
                    "kind": "expert",
                    "module_name": action.module_name,
                    "adapter": action.adapter,
                    "structure_family": action.structure_family,
                    "router_name": action.metadata.get("router_name"),
                    "experts_name": action.metadata.get("experts_name"),
                    "usage_source": action.metadata.get("usage_source"),
                    "kept_indices": list(action.keep_indices),
                    "pruned_indices": list(action.prune_indices),
                    "kept_experts": [
                        expert_names[index]
                        for index in action.keep_indices
                        if index < len(expert_names)
                    ],
                    "pruned_experts": [
                        expert_names[index]
                        for index in action.prune_indices
                        if index < len(expert_names)
                    ],
                    "kept_usage": [
                        float(usage_scores[index])
                        for index in action.keep_indices
                        if index < len(usage_scores)
                    ],
                    "pruned_usage": [
                        float(usage_scores[index])
                        for index in action.prune_indices
                        if index < len(usage_scores)
                    ],
                }
            )
            continue
        if action.action_type in {"concat_branch_channels", "drop_stages", "drop_blocks"}:
            changes.append(
                {
                    "kind": (
                        "branch"
                        if action.action_type == "concat_branch_channels"
                        else action.granularity
                    ),
                    "module_name": action.module_name,
                    "adapter": action.adapter,
                    "structure_family": action.structure_family,
                    "kept_indices": list(action.keep_indices),
                    "pruned_indices": list(action.prune_indices),
                    "merge": action.metadata.get("merge"),
                }
            )
    return changes


def _apply_residual_stage_action(
    model: nn.Module,
    action: StructuredPruningAction,
) -> None:
    stage = _get_module(model, action.module_name)
    if not isinstance(stage, nn.Sequential):
        raise TypeError(f"{action.module_name} is not a Sequential stage")
    block_type = str(action.metadata.get("block_type"))
    blocks = action.metadata.get("blocks")
    if not isinstance(blocks, list):
        raise ValueError("residual_stage_channels requires metadata.blocks")

    keep = action.keep_indices
    stage_out_channels = int(action.metadata.get("stage_out_channels", action.original_units))
    for block in blocks:
        if not isinstance(block, Mapping):
            raise ValueError("metadata.blocks entries must be mappings")
        conv1_name = str(block["conv1_name"])
        bn1_name = str(block["bn1_name"])
        conv2_name = str(block["conv2_name"])
        bn2_name = str(block["bn2_name"])

        conv1 = _get_module(model, conv1_name)

        if block_type == "BasicBlock":
            bn1 = _get_module(model, bn1_name)
            conv2 = _get_module(model, conv2_name)
            bn2 = _get_module(model, bn2_name)
            if not isinstance(conv1, nn.Conv2d) or not isinstance(conv2, nn.Conv2d):
                raise TypeError("BasicBlock residual stage pruning requires conv1/conv2")
            if not isinstance(bn1, nn.modules.batchnorm._BatchNorm) or not isinstance(
                bn2, nn.modules.batchnorm._BatchNorm
            ):
                raise TypeError("BasicBlock residual stage pruning requires bn1/bn2")
            if conv1.in_channels == stage_out_channels:
                _set_module(model, conv1_name, prune_conv2d_in_channels(conv1, keep))
                conv1 = _get_module(model, conv1_name)
                if not isinstance(conv1, nn.Conv2d):
                    raise TypeError(f"{conv1_name} is not a Conv2d module")
            _set_module(model, conv1_name, prune_conv2d_out_channels(conv1, keep))
            _set_module(model, bn1_name, prune_batchnorm_channels(bn1, keep))
            conv2 = _get_module(model, conv2_name)
            if not isinstance(conv2, nn.Conv2d):
                raise TypeError(f"{conv2_name} is not a Conv2d module")
            _set_module(model, conv2_name, prune_conv2d_in_channels(conv2, keep))
            conv2 = _get_module(model, conv2_name)
            if not isinstance(conv2, nn.Conv2d):
                raise TypeError(f"{conv2_name} is not a Conv2d module")
            _set_module(model, conv2_name, prune_conv2d_out_channels(conv2, keep))
            _set_module(model, bn2_name, prune_batchnorm_channels(bn2, keep))
        elif block_type == "Bottleneck":
            conv3_name = str(block["conv3_name"])
            bn3_name = str(block["bn3_name"])
            if not isinstance(conv1, nn.Conv2d):
                raise TypeError(f"{conv1_name} is not a Conv2d module")
            if conv1.in_channels == stage_out_channels:
                _set_module(model, conv1_name, prune_conv2d_in_channels(conv1, keep))
            conv3 = _get_module(model, conv3_name)
            bn3 = _get_module(model, bn3_name)
            if not isinstance(conv3, nn.Conv2d):
                raise TypeError(f"{conv3_name} is not a Conv2d module")
            if not isinstance(bn3, nn.modules.batchnorm._BatchNorm):
                raise TypeError(f"{bn3_name} is not a BatchNorm module")
            _set_module(model, conv3_name, prune_conv2d_out_channels(conv3, keep))
            _set_module(model, bn3_name, prune_batchnorm_channels(bn3, keep))
        else:
            raise ValueError(f"Unsupported residual block_type: {block_type}")

        downsample_conv_name = block.get("downsample_conv_name")
        downsample_bn_name = block.get("downsample_bn_name")
        if isinstance(downsample_conv_name, str) and isinstance(downsample_bn_name, str):
            downsample_conv = _get_module(model, downsample_conv_name)
            downsample_bn = _get_module(model, downsample_bn_name)
            if not isinstance(downsample_conv, nn.Conv2d):
                raise TypeError(f"{downsample_conv_name} is not a Conv2d module")
            if not isinstance(downsample_bn, nn.modules.batchnorm._BatchNorm):
                raise TypeError(f"{downsample_bn_name} is not a BatchNorm module")
            _set_module(
                model,
                downsample_conv_name,
                prune_conv2d_out_channels(downsample_conv, keep),
            )
            _set_module(model, downsample_bn_name, prune_batchnorm_channels(downsample_bn, keep))

    consumers = action.metadata.get("consumers")
    if not isinstance(consumers, list):
        raise ValueError("residual_stage_channels requires metadata.consumers")
    for item in consumers:
        if not isinstance(item, Mapping):
            raise ValueError("metadata.consumers entries must be mappings")
        consumer_name = item.get("name")
        consumer_type = item.get("type")
        feature_block_size = int(item.get("feature_block_size", 1))
        if not isinstance(consumer_name, str) or not isinstance(consumer_type, str):
            continue
        consumer = _get_module(model, consumer_name)
        if consumer_type == "Conv2d":
            if not isinstance(consumer, nn.Conv2d):
                raise TypeError(f"{consumer_name} is not a Conv2d module")
            _set_module(model, consumer_name, prune_conv2d_in_channels(consumer, keep))
            continue
        if consumer_type == "Linear":
            if not isinstance(consumer, nn.Linear):
                raise TypeError(f"{consumer_name} is not a Linear module")
            feature_keep_indices = _expand_linear_keep_indices(
                keep,
                block_size=feature_block_size,
            )
            _set_module(model, consumer_name, prune_linear_in_features(consumer, feature_keep_indices))
            continue
        raise ValueError(f"Unsupported residual consumer type: {consumer_type}")


def _apply_concat_branch_action(
    model: nn.Module,
    action: StructuredPruningAction,
    branch_keep_map: Mapping[str, list[int]],
    rewritten_consumers: set[str],
) -> None:
    producer = _get_module(model, action.module_name)
    if not isinstance(producer, nn.Conv2d):
        raise TypeError(f"{action.module_name} is not a Conv2d module")
    _set_module(
        model,
        action.module_name,
        prune_conv2d_out_channels(producer, action.keep_indices),
    )
    if action.normalization_name is not None:
        normalization = _get_module(model, action.normalization_name)
        if not isinstance(normalization, nn.modules.batchnorm._BatchNorm):
            raise TypeError(f"{action.normalization_name} is not a BatchNorm module")
        _set_module(
            model,
            action.normalization_name,
            prune_batchnorm_channels(normalization, action.keep_indices),
        )

    if action.consumer_name is None or action.consumer_type != "Conv2d":
        return
    if action.consumer_name in rewritten_consumers:
        return
    branch_specs = action.metadata.get("branch_specs")
    if not isinstance(branch_specs, list):
        raise ValueError("concat_branch_channels requires metadata.branch_specs")
    consumer = _get_module(model, action.consumer_name)
    if not isinstance(consumer, nn.Conv2d):
        raise TypeError(f"{action.consumer_name} is not a Conv2d module")
    input_keep_indices: list[int] = []
    offset = 0
    for branch_spec in branch_specs:
        if not isinstance(branch_spec, Mapping):
            raise ValueError("metadata.branch_specs entries must be mappings")
        conv_name = branch_spec.get("conv_name")
        out_channels = int(branch_spec.get("out_channels", 0))
        if not isinstance(conv_name, str):
            raise ValueError("metadata.branch_specs[*].conv_name must be a string")
        branch_keep = branch_keep_map.get(conv_name)
        if branch_keep is None:
            current_branch = _get_module(model, conv_name)
            if not isinstance(current_branch, nn.Conv2d):
                raise TypeError(f"{conv_name} is not a Conv2d module")
            branch_keep = list(range(current_branch.out_channels))
        input_keep_indices.extend(offset + int(index) for index in branch_keep)
        offset += out_channels
    _set_module(
        model,
        action.consumer_name,
        prune_conv2d_in_channels(consumer, input_keep_indices),
    )
    rewritten_consumers.add(action.consumer_name)


def _apply_mbconv_action(
    model: nn.Module,
    action: StructuredPruningAction,
) -> None:
    expand_conv_name = str(action.metadata["expand_conv_name"])
    expand_bn_name = str(action.metadata["expand_bn_name"])
    depthwise_conv_name = str(action.metadata["depthwise_conv_name"])
    depthwise_bn_name = str(action.metadata["depthwise_bn_name"])
    project_conv_name = str(action.metadata["project_conv_name"])

    expand_conv = _get_module(model, expand_conv_name)
    expand_bn = _get_module(model, expand_bn_name)
    depthwise_conv = _get_module(model, depthwise_conv_name)
    depthwise_bn = _get_module(model, depthwise_bn_name)
    project_conv = _get_module(model, project_conv_name)
    if not all(
        isinstance(module, nn.Conv2d)
        for module in (expand_conv, depthwise_conv, project_conv)
    ):
        raise TypeError("MBConv pruning requires Conv2d modules")
    if not all(
        isinstance(module, nn.modules.batchnorm._BatchNorm)
        for module in (expand_bn, depthwise_bn)
    ):
        raise TypeError("MBConv pruning requires BatchNorm modules")

    keep = action.keep_indices
    _set_module(model, expand_conv_name, prune_conv2d_out_channels(expand_conv, keep))
    _set_module(model, expand_bn_name, prune_batchnorm_channels(expand_bn, keep))
    depthwise_conv = _get_module(model, depthwise_conv_name)
    if not isinstance(depthwise_conv, nn.Conv2d):
        raise TypeError(f"{depthwise_conv_name} is not a Conv2d module")
    _set_module(model, depthwise_conv_name, prune_conv2d_in_channels(depthwise_conv, keep))
    _set_module(model, depthwise_bn_name, prune_batchnorm_channels(depthwise_bn, keep))
    _set_module(model, project_conv_name, prune_conv2d_in_channels(project_conv, keep))


def _apply_vit_hidden_width_action(
    model: nn.Module,
    action: StructuredPruningAction,
) -> None:
    descriptor = action.metadata
    target_model = _get_module(model, action.module_name)
    keep = action.keep_indices
    new_embed_dim = len(keep)
    old_embed_dim = int(descriptor["embed_dim"])
    num_heads = int(descriptor["num_heads"])
    if new_embed_dim <= 0 or new_embed_dim % num_heads != 0:
        raise ValueError("vit_hidden_width keep_indices must keep a width divisible by num_heads")
    if max(keep) >= old_embed_dim:
        raise ValueError("vit_hidden_width keep_indices are out of range")

    patch_proj_name = str(descriptor["patch_proj_name"])
    patch_proj = _get_module(model, patch_proj_name)
    if not isinstance(patch_proj, nn.Conv2d):
        raise TypeError(f"{patch_proj_name} is not a Conv2d module")
    _set_module(model, patch_proj_name, prune_conv2d_out_channels(patch_proj, keep))

    if not hasattr(target_model, "cls_token") or not hasattr(target_model, "pos_embed"):
        raise TypeError("vit_hidden_width requires cls_token and pos_embed parameters")
    setattr(target_model, "cls_token", _prune_parameter_last_dim(target_model.cls_token, keep))
    setattr(target_model, "pos_embed", _prune_parameter_last_dim(target_model.pos_embed, keep))
    if hasattr(target_model, "embed_dim"):
        setattr(target_model, "embed_dim", new_embed_dim)

    norm_name = str(descriptor["norm_name"])
    norm = _get_module(model, norm_name)
    if not isinstance(norm, nn.LayerNorm):
        raise TypeError(f"{norm_name} is not a LayerNorm module")
    _set_module(model, norm_name, _prune_layernorm_features(norm, keep))

    head_name = descriptor.get("head_name")
    if isinstance(head_name, str):
        head = _get_module(model, head_name)
        if isinstance(head, nn.Linear):
            _set_module(model, head_name, prune_linear_in_features(head, keep))

    blocks = descriptor["blocks"]
    if not isinstance(blocks, list):
        raise ValueError("vit_hidden_width requires metadata.blocks")
    qkv_keep = (
        keep
        + [old_embed_dim + index for index in keep]
        + [2 * old_embed_dim + index for index in keep]
    )
    for block in blocks:
        if not isinstance(block, Mapping):
            raise ValueError("metadata.blocks entries must be mappings")
        norm1_name = str(block["norm1_name"])
        norm2_name = str(block["norm2_name"])
        qkv_name = str(block["qkv_name"])
        proj_name = str(block["proj_name"])
        fc1_name = str(block["fc1_name"])
        fc2_name = str(block["fc2_name"])
        attn_name = str(block["attn_name"])

        norm1 = _get_module(model, norm1_name)
        norm2 = _get_module(model, norm2_name)
        qkv = _get_module(model, qkv_name)
        proj = _get_module(model, proj_name)
        fc1 = _get_module(model, fc1_name)
        fc2 = _get_module(model, fc2_name)
        attn = _get_module(model, attn_name)
        if not isinstance(norm1, nn.LayerNorm) or not isinstance(norm2, nn.LayerNorm):
            raise TypeError("vit_hidden_width requires LayerNorm modules")
        if not all(isinstance(module, nn.Linear) for module in (qkv, proj, fc1, fc2)):
            raise TypeError("vit_hidden_width requires qkv/proj/fc1/fc2 Linear modules")
        if not isinstance(qkv, nn.Linear) or not isinstance(proj, nn.Linear):
            raise TypeError("vit_hidden_width requires attention Linear modules")
        if not isinstance(fc1, nn.Linear) or not isinstance(fc2, nn.Linear):
            raise TypeError("vit_hidden_width requires MLP Linear modules")

        _set_module(model, norm1_name, _prune_layernorm_features(norm1, keep))
        _set_module(model, norm2_name, _prune_layernorm_features(norm2, keep))
        _set_module(model, qkv_name, prune_linear_in_out_features(qkv, keep, qkv_keep))
        _set_module(model, proj_name, prune_linear_in_out_features(proj, keep, keep))
        _set_module(model, fc1_name, prune_linear_in_features(fc1, keep))
        _set_module(model, fc2_name, prune_linear_out_features(fc2, keep))
        if hasattr(attn, "embed_dim"):
            setattr(attn, "embed_dim", new_embed_dim)
        if hasattr(attn, "head_dim"):
            setattr(attn, "head_dim", new_embed_dim // num_heads)
        if hasattr(attn, "scale"):
            setattr(attn, "scale", (new_embed_dim // num_heads) ** -0.5)


def _apply_drop_experts_action(
    model: nn.Module,
    action: StructuredPruningAction,
) -> None:
    module = _get_module(model, action.module_name)
    router_name = str(action.metadata["router_name"])
    experts_name = str(action.metadata["experts_name"])
    router = _get_module(model, router_name)
    experts = _get_module(model, experts_name)
    if not isinstance(router, nn.Linear):
        raise TypeError(f"{router_name} is not a Linear router")
    if not isinstance(experts, (nn.ModuleList, nn.Sequential)):
        raise TypeError(f"{experts_name} is not a ModuleList or Sequential expert container")
    if router.out_features != len(list(experts.children())):
        raise ValueError("router.out_features must match expert count before pruning")
    _set_module(model, router_name, prune_linear_out_features(router, action.keep_indices))
    _set_module(
        model,
        experts_name,
        _rewrite_indexed_container(experts, action.keep_indices),
    )
    kept_usage = [float(action.metadata["usage_scores"][index]) for index in action.keep_indices]
    for attr_name in ("expert_usage", "expert_usage_counts", "router_usage"):
        raw_value = getattr(module, attr_name, None)
        if raw_value is None:
            continue
        if isinstance(raw_value, torch.Tensor):
            keep = torch.tensor(
                action.keep_indices,
                dtype=torch.long,
                device=raw_value.device,
            )
            setattr(module, attr_name, raw_value.index_select(0, keep).clone())
        else:
            setattr(module, attr_name, kept_usage)
    if hasattr(module, "num_experts"):
        setattr(module, "num_experts", len(action.keep_indices))


def _validate_forward(model: nn.Module, example_input: Any) -> None:
    was_training = model.training
    model.eval()
    with torch.no_grad():
        if isinstance(example_input, Mapping):
            model(**example_input)
        elif isinstance(example_input, tuple):
            model(*example_input)
        else:
            model(example_input)
    model.train(was_training)


def apply_structured_pruning_plan(
    model: nn.Module,
    plan: StructuredPruningPlan,
    *,
    example_input: Any = None,
) -> StructuredPruningReport:
    """Apply a planned structured pruning rewrite in-place."""

    parameter_count_before = _count_parameters(model)
    concat_branch_keep_map = {
        item.module_name: list(item.keep_indices)
        for item in plan.actions
        if item.action_type == "concat_branch_channels"
    }
    rewritten_concat_consumers: set[str] = set()

    for action in plan.actions:
        if not action.prune_indices:
            continue
        if action.action_type == "residual_stage_channels":
            _apply_residual_stage_action(model, action)
            continue
        if action.action_type == "concat_branch_channels":
            _apply_concat_branch_action(
                model,
                action,
                concat_branch_keep_map,
                rewritten_concat_consumers,
            )
            continue
        if action.action_type == "mbconv_mid_channels":
            _apply_mbconv_action(model, action)
            continue
        if action.action_type == "vit_hidden_width":
            _apply_vit_hidden_width_action(model, action)
            continue
        if action.action_type == "drop_experts":
            _apply_drop_experts_action(model, action)
            continue
        if action.action_type == "conv_channel_group":
            producer = _get_module(model, action.module_name)
            if not isinstance(producer, nn.Conv2d):
                raise TypeError(f"{action.module_name} is not a Conv2d module")
            _set_module(
                model,
                action.module_name,
                prune_conv2d_out_channels(producer, action.keep_indices),
            )

            if action.normalization_name is not None:
                normalization = _get_module(model, action.normalization_name)
                if not isinstance(normalization, nn.modules.batchnorm._BatchNorm):
                    raise TypeError(f"{action.normalization_name} is not a BatchNorm module")
                _set_module(
                    model,
                    action.normalization_name,
                    prune_batchnorm_channels(normalization, action.keep_indices),
                )

            if action.consumer_name is None or action.consumer_type is None:
                continue
            consumer = _get_module(model, action.consumer_name)
            if action.consumer_type == "Conv2d":
                if not isinstance(consumer, nn.Conv2d):
                    raise TypeError(f"{action.consumer_name} is not a Conv2d module")
                _set_module(
                    model,
                    action.consumer_name,
                    prune_conv2d_in_channels(consumer, action.keep_indices),
                )
                continue
            if action.consumer_type == "Linear":
                if not isinstance(consumer, nn.Linear):
                    raise TypeError(f"{action.consumer_name} is not a Linear module")
                feature_keep_indices = _expand_linear_keep_indices(
                    action.keep_indices,
                    block_size=action.feature_block_size,
                )
                _set_module(
                    model,
                    action.consumer_name,
                    prune_linear_in_features(consumer, feature_keep_indices),
                )
                continue
            raise ValueError(f"Unsupported consumer type: {action.consumer_type}")

        if action.action_type == "mlp_neuron_group":
            fc1 = _get_module(model, action.module_name)
            if not isinstance(fc1, nn.Linear):
                raise TypeError(f"{action.module_name} is not a Linear module")
            if action.consumer_name is None:
                raise ValueError("mlp_neuron_group requires a consumer_name")
            fc2 = _get_module(model, action.consumer_name)
            if not isinstance(fc2, nn.Linear):
                raise TypeError(f"{action.consumer_name} is not a Linear module")
            _set_module(
                model,
                action.module_name,
                prune_linear_out_features(fc1, action.keep_indices),
            )
            _set_module(
                model,
                action.consumer_name,
                prune_linear_in_features(fc2, action.keep_indices),
            )
            continue

        if action.action_type == "gated_mlp_neuron_group":
            gate_proj_name = action.metadata.get("gate_proj_name")
            up_proj_name = action.metadata.get("up_proj_name")
            down_proj_name = action.metadata.get("down_proj_name")
            if not all(
                isinstance(name, str)
                for name in (gate_proj_name, up_proj_name, down_proj_name)
            ):
                raise ValueError(
                    "gated_mlp_neuron_group requires gate_proj_name/up_proj_name/down_proj_name"
                )
            gate_proj = _get_module(model, str(gate_proj_name))
            up_proj = _get_module(model, str(up_proj_name))
            down_proj = _get_module(model, str(down_proj_name))
            if not isinstance(gate_proj, nn.Linear):
                raise TypeError(f"{gate_proj_name} is not a Linear module")
            if not isinstance(up_proj, nn.Linear):
                raise TypeError(f"{up_proj_name} is not a Linear module")
            if not isinstance(down_proj, nn.Linear):
                raise TypeError(f"{down_proj_name} is not a Linear module")
            _set_module(
                model,
                str(gate_proj_name),
                prune_linear_out_features(gate_proj, action.keep_indices),
            )
            _set_module(
                model,
                str(up_proj_name),
                prune_linear_out_features(up_proj, action.keep_indices),
            )
            _set_module(
                model,
                str(down_proj_name),
                prune_linear_in_features(down_proj, action.keep_indices),
            )
            continue

        if action.action_type == "drop_blocks":
            container = _get_module(model, action.module_name)
            _set_module(
                model,
                action.module_name,
                _rewrite_indexed_container(container, action.keep_indices),
            )
            continue

        if action.action_type == "drop_stages":
            container = _get_module(model, action.module_name)
            _set_module(
                model,
                action.module_name,
                _rewrite_indexed_container(container, action.keep_indices),
            )
            continue

        if action.action_type == "attention_heads":
            attention = _get_module(model, action.module_name)
            head_dim = getattr(attention, "head_dim", None)
            if not isinstance(head_dim, int):
                raise TypeError(f"{action.module_name} is not a supported attention module")
            attention_kind = str(action.metadata.get("attention_kind", "fused_qkv"))
            attention_variant = str(action.metadata.get("attention_variant", "mha"))
            if attention_kind == "split_qkv":
                if attention_variant in {"gqa", "mqa"}:
                    _set_module(
                        model,
                        action.module_name,
                        PrunedGroupedQueryAttention(attention, action.keep_indices),
                    )
                    continue
                _set_module(
                    model,
                    action.module_name,
                    PrunedSplitProjectionAttention(attention, action.keep_indices),
                )
                continue
            _set_module(
                model,
                action.module_name,
                PrunedMultiHeadAttention(attention, action.keep_indices),
            )
            continue

        raise ValueError(f"Unsupported structured action type: {action.action_type}")

    forward_checked = example_input is not None
    if forward_checked:
        _validate_forward(model, example_input)

    return StructuredPruningReport(
        method=plan.method,
        granularity=plan.granularity,
        scope=plan.scope,
        target_sparsity=plan.target_sparsity,
        importance_metric=plan.importance_metric,
        parameter_count_before=parameter_count_before,
        parameter_count_after=_count_parameters(model),
        forward_checked=forward_checked,
        adapters=list(plan.adapters),
        structure_families=list(plan.structure_families),
        blocked_modules=list(plan.blocked_modules),
        dependency_graph=dict(plan.dependency_graph),
        topology_changes=_topology_changes_from_actions(plan.actions),
        export_status={
            "attempted": False,
            "passed": None,
            "artifacts": [],
        },
        benchmark_status={
            "attempted": False,
            "passed": None,
            "latency": None,
            "memory": None,
        },
        targets=plan.targets,
        actions=plan.actions,
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
) -> StructuredPruningReport:
    """Plan and apply simple structured pruning for chain-like CNNs."""

    plan = plan_structured_pruning(
        model,
        target_sparsity,
        granularity=granularity,
        scope=scope,
        importance=importance,
        selection=selection,
    )
    return apply_structured_pruning_plan(
        model,
        plan,
        example_input=example_input,
    )


__all__ = [
    "find_structured_pruning_targets",
    "BlockSparseLayerReport",
    "BlockSparsePruningReport",
    "PruningTarget",
    "SUPPORTED_GRANULARITIES",
    "SUPPORTED_IMPORTANCE_METRICS",
    "SUPPORTED_SCOPES",
    "StructuredPruningAction",
    "StructuredPruningPlan",
    "StructuredPruningReport",
    "StructuredPruningToyCNN",
    "apply_block_sparse_pruning",
    "apply_structured_pruning",
    "apply_structured_pruning_plan",
    "plan_structured_pruning",
]
