"""Structured pruning rewrite helpers."""

from __future__ import annotations

import inspect
from typing import Any, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .report import (
    StructuredPruningAction,
    StructuredPruningPlan,
    StructuredPruningReport,
)


def _as_index_tensor(indices: Sequence[int], device: torch.device) -> torch.LongTensor:
    if not indices:
        raise ValueError("indices must not be empty")
    index_tensor = torch.tensor(list(indices), dtype=torch.long, device=device)
    if torch.unique(index_tensor).numel() != index_tensor.numel():
        raise ValueError("indices must be unique")
    return index_tensor


def _clone_training_state(source: nn.Module, target: nn.Module) -> nn.Module:
    target.train(source.training)
    return target


def prune_linear_out_features(
    module: nn.Linear,
    keep_indices: Sequence[int],
) -> nn.Linear:
    """Return a new linear layer with selected output channels."""

    keep = _as_index_tensor(keep_indices, module.weight.device)
    new_module = nn.Linear(
        module.in_features,
        keep.numel(),
        bias=module.bias is not None,
        device=module.weight.device,
        dtype=module.weight.dtype,
    )
    new_module.weight.data.copy_(module.weight.data.index_select(0, keep))
    if module.bias is not None and new_module.bias is not None:
        new_module.bias.data.copy_(module.bias.data.index_select(0, keep))
    return _clone_training_state(module, new_module)


def prune_linear_in_features(
    module: nn.Linear,
    keep_indices: Sequence[int],
) -> nn.Linear:
    """Return a new linear layer with selected input channels."""

    keep = _as_index_tensor(keep_indices, module.weight.device)
    new_module = nn.Linear(
        keep.numel(),
        module.out_features,
        bias=module.bias is not None,
        device=module.weight.device,
        dtype=module.weight.dtype,
    )
    new_module.weight.data.copy_(module.weight.data.index_select(1, keep))
    if module.bias is not None and new_module.bias is not None:
        new_module.bias.data.copy_(module.bias.data)
    return _clone_training_state(module, new_module)


def prune_linear_in_out_features(
    module: nn.Linear,
    keep_in_indices: Sequence[int],
    keep_out_indices: Sequence[int],
) -> nn.Linear:
    """Return a new linear layer with selected input and output channels."""

    keep_in = _as_index_tensor(keep_in_indices, module.weight.device)
    keep_out = _as_index_tensor(keep_out_indices, module.weight.device)
    new_module = nn.Linear(
        keep_in.numel(),
        keep_out.numel(),
        bias=module.bias is not None,
        device=module.weight.device,
        dtype=module.weight.dtype,
    )
    new_module.weight.data.copy_(
        module.weight.data.index_select(0, keep_out).index_select(1, keep_in)
    )
    if module.bias is not None and new_module.bias is not None:
        new_module.bias.data.copy_(module.bias.data.index_select(0, keep_out))
    return _clone_training_state(module, new_module)


def _validate_conv_groups(module: nn.Conv2d) -> None:
    if module.groups == 1:
        return
    if module.groups == module.in_channels == module.out_channels:
        return
    if module.in_channels % module.groups != 0 or module.out_channels % module.groups != 0:
        raise ValueError("Grouped Conv2d must satisfy in_channels/out_channels divisibility")


def _group_keep_local_indices(
    *,
    total_channels: int,
    groups: int,
    keep_indices: Sequence[int],
) -> tuple[list[list[int]], int]:
    if total_channels % groups != 0:
        raise ValueError("total_channels must be divisible by groups")
    channels_per_group = total_channels // groups
    local_keep_by_group: list[list[int]] = [[] for _ in range(groups)]
    for index in keep_indices:
        if index < 0 or index >= total_channels:
            raise ValueError("keep_indices contains out-of-range channel index")
        group_index = index // channels_per_group
        local_keep_by_group[group_index].append(index % channels_per_group)
    keep_counts = {len(local_keep) for local_keep in local_keep_by_group}
    if len(keep_counts) != 1:
        raise ValueError(
            "Grouped Conv2d pruning requires keeping the same number of channels in every group"
        )
    kept_per_group = keep_counts.pop()
    if kept_per_group <= 0:
        raise ValueError("Grouped Conv2d pruning must keep at least one channel in every group")
    return [sorted(local_keep) for local_keep in local_keep_by_group], kept_per_group


def validate_conv2d_keep_indices(
    module: nn.Conv2d,
    keep_indices: Sequence[int],
    *,
    axis: str,
) -> None:
    """Validate keep indices for grouped Conv2d pruning."""

    if axis not in {"in", "out"}:
        raise ValueError("axis must be 'in' or 'out'")
    if module.groups == 1:
        return
    if module.groups == module.in_channels == module.out_channels:
        return
    total_channels = module.in_channels if axis == "in" else module.out_channels
    _group_keep_local_indices(
        total_channels=total_channels,
        groups=module.groups,
        keep_indices=keep_indices,
    )


def prune_conv2d_out_channels(
    module: nn.Conv2d,
    keep_indices: Sequence[int],
) -> nn.Conv2d:
    """Return a new Conv2d layer with selected output channels."""

    _validate_conv_groups(module)
    keep = _as_index_tensor(keep_indices, module.weight.device)
    if module.groups == 1:
        new_in_channels = module.in_channels
        new_groups = 1
        weight = module.weight.data.index_select(0, keep)
    elif module.groups == module.in_channels == module.out_channels:
        new_in_channels = keep.numel()
        new_groups = keep.numel()
        weight = module.weight.data.index_select(0, keep)
    else:
        _group_keep_local_indices(
            total_channels=module.out_channels,
            groups=module.groups,
            keep_indices=keep_indices,
        )
        new_in_channels = module.in_channels
        new_groups = module.groups
        weight = module.weight.data.index_select(0, keep)
    new_module = nn.Conv2d(
        in_channels=new_in_channels,
        out_channels=keep.numel(),
        kernel_size=module.kernel_size,
        stride=module.stride,
        padding=module.padding,
        dilation=module.dilation,
        groups=new_groups,
        bias=module.bias is not None,
        padding_mode=module.padding_mode,
        device=module.weight.device,
        dtype=module.weight.dtype,
    )
    new_module.weight.data.copy_(weight)
    if module.bias is not None and new_module.bias is not None:
        new_module.bias.data.copy_(module.bias.data.index_select(0, keep))
    return _clone_training_state(module, new_module)


def prune_conv2d_in_channels(
    module: nn.Conv2d,
    keep_indices: Sequence[int],
) -> nn.Conv2d:
    """Return a new Conv2d layer with selected input channels."""

    _validate_conv_groups(module)
    keep = _as_index_tensor(keep_indices, module.weight.device)
    if module.groups == 1:
        new_in_channels = keep.numel()
        new_groups = 1
        weight = module.weight.data.index_select(1, keep)
    elif module.groups == module.in_channels == module.out_channels:
        new_in_channels = keep.numel()
        new_groups = keep.numel()
        weight = module.weight.data.index_select(0, keep)
    else:
        local_keep_by_group, kept_per_group = _group_keep_local_indices(
            total_channels=module.in_channels,
            groups=module.groups,
            keep_indices=keep_indices,
        )
        out_channels_per_group = module.out_channels // module.groups
        grouped_weights: list[torch.Tensor] = []
        for group_index, local_keep in enumerate(local_keep_by_group):
            out_start = group_index * out_channels_per_group
            out_end = out_start + out_channels_per_group
            grouped_weights.append(
                module.weight.data[out_start:out_end].index_select(
                    1,
                    torch.tensor(
                        local_keep,
                        dtype=torch.long,
                        device=module.weight.device,
                    ),
                )
            )
        new_in_channels = keep.numel()
        new_groups = module.groups
        weight = torch.cat(grouped_weights, dim=0)
    new_module = nn.Conv2d(
        in_channels=new_in_channels,
        out_channels=(
            keep.numel()
            if module.groups == module.in_channels == module.out_channels
            else module.out_channels
        ),
        kernel_size=module.kernel_size,
        stride=module.stride,
        padding=module.padding,
        dilation=module.dilation,
        groups=new_groups,
        bias=module.bias is not None,
        padding_mode=module.padding_mode,
        device=module.weight.device,
        dtype=module.weight.dtype,
    )
    new_module.weight.data.copy_(weight)
    if module.bias is not None and new_module.bias is not None:
        if module.groups == 1:
            new_module.bias.data.copy_(module.bias.data)
        elif module.groups == module.in_channels == module.out_channels:
            new_module.bias.data.copy_(module.bias.data.index_select(0, keep))
        else:
            new_module.bias.data.copy_(module.bias.data)
    return _clone_training_state(module, new_module)


def prune_batchnorm_channels(
    module: nn.modules.batchnorm._BatchNorm,
    keep_indices: Sequence[int],
) -> nn.modules.batchnorm._BatchNorm:
    """Return a new BatchNorm layer with selected channels."""

    keep = _as_index_tensor(keep_indices, module.weight.device)
    new_module = type(module)(
        num_features=keep.numel(),
        eps=module.eps,
        momentum=module.momentum,
        affine=module.affine,
        track_running_stats=module.track_running_stats,
        device=module.weight.device if module.affine else None,
        dtype=module.weight.dtype if module.affine else None,
    )
    if module.affine and module.weight is not None and new_module.weight is not None:
        new_module.weight.data.copy_(module.weight.data.index_select(0, keep))
    if module.affine and module.bias is not None and new_module.bias is not None:
        new_module.bias.data.copy_(module.bias.data.index_select(0, keep))
    if module.track_running_stats:
        new_module.running_mean.data.copy_(module.running_mean.data.index_select(0, keep))
        new_module.running_var.data.copy_(module.running_var.data.index_select(0, keep))
        new_module.num_batches_tracked.data.copy_(module.num_batches_tracked.data)
    return _clone_training_state(module, new_module)


def _dropout_probability(module_or_probability: Any) -> float:
    if isinstance(module_or_probability, nn.Dropout):
        return float(module_or_probability.p)
    return float(module_or_probability)


def infer_attention_role(module: nn.Module) -> str:
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


def attention_variant(*, num_heads: int, num_kv_heads: int) -> str:
    if num_kv_heads == num_heads:
        return "mha"
    if num_kv_heads == 1:
        return "mqa"
    return "gqa"


class PrunedMultiHeadAttention(nn.Module):
    """Attention module with fewer heads but unchanged input/output embed dim."""

    def __init__(self, source: nn.Module, keep_head_indices: list[int]) -> None:
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
        return self.proj_dropout(x)


class PrunedSplitProjectionAttention(nn.Module):
    """Attention module rewritten from split q/k/v/out projections."""

    def __init__(self, source: nn.Module, keep_head_indices: list[int]) -> None:
        super().__init__()
        q_proj = getattr(source, "q_proj")
        k_proj = getattr(source, "k_proj")
        v_proj = getattr(source, "v_proj")
        out_proj = getattr(source, "out_proj")
        num_heads = int(getattr(source, "num_heads"))
        head_dim = int(getattr(source, "head_dim"))
        embed_dim = int(getattr(source, "embed_dim"))
        dropout = float(getattr(source, "dropout", 0.0))
        if not all(isinstance(module, nn.Linear) for module in (q_proj, k_proj, v_proj, out_proj)):
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
        self.attention_role = infer_attention_role(source)
        self.q_proj = prune_linear_out_features(q_proj, keep_features)
        self.k_proj = prune_linear_out_features(k_proj, keep_features)
        self.v_proj = prune_linear_out_features(v_proj, keep_features)
        self.out_proj = prune_linear_in_features(out_proj, keep_features)
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
        attn = self.attn_dropout(F.softmax(attn, dim=-1))
        x = (attn @ v).transpose(1, 2).reshape(batch_size, token_count, self.inner_dim)
        return self.proj_dropout(self.out_proj(x))


class PrunedGroupedQueryAttention(nn.Module):
    """Attention rewrite for GQA/MQA split q/k/v/out projections."""

    def __init__(self, source: nn.Module, keep_kv_head_indices: list[int]) -> None:
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
        if not all(isinstance(module, nn.Linear) for module in (q_proj, k_proj, v_proj, out_proj)):
            raise TypeError("grouped-query attention pruning requires q_proj/k_proj/v_proj/out_proj")
        if num_kv_heads <= 0 or num_heads <= 0:
            raise ValueError("grouped-query attention requires positive num_heads and num_kv_heads")
        if num_heads % num_kv_heads != 0:
            raise ValueError("grouped-query attention requires num_heads divisible by num_kv_heads")
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
        self.attention_role = infer_attention_role(source)
        self.attention_variant = attention_variant(
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
        attn = self.attn_dropout(F.softmax((q @ k.transpose(-2, -1)) * self.scale, dim=-1))
        x = (attn @ v).transpose(1, 2).reshape(batch_size, token_count, self.inner_dim)
        return self.proj_dropout(self.out_proj(x))


def _get_module(root: nn.Module, name: str) -> nn.Module:
    if name in {"", "<root>"}:
        return root
    module = root
    for part in name.split("."):
        module = module[int(part)] if part.isdigit() else getattr(module, part)
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
        rewritten: nn.Module = nn.ModuleList([container[index] for index in keep_indices])
    elif isinstance(container, nn.Sequential):
        rewritten = nn.Sequential(*(container[index] for index in keep_indices))
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


def topology_changes_from_actions(
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


def _apply_mbconv_action(model: nn.Module, action: StructuredPruningAction) -> None:
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
    if not all(isinstance(module, nn.Conv2d) for module in (expand_conv, depthwise_conv, project_conv)):
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


def _apply_drop_experts_action(model: nn.Module, action: StructuredPruningAction) -> None:
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
    _set_module(model, experts_name, _rewrite_indexed_container(experts, action.keep_indices))
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


def _count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


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

        if action.action_type in {"drop_blocks", "drop_stages"}:
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
        topology_changes=topology_changes_from_actions(plan.actions),
        export_status={"attempted": False, "passed": None, "artifacts": []},
        benchmark_status={
            "attempted": False,
            "passed": None,
            "latency": None,
            "memory": None,
        },
        targets=plan.targets,
        actions=plan.actions,
    )


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
