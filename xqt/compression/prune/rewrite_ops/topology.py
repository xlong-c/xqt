"""Module topology helpers: get/set, container rewrite, and topology change mapping."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from ..report import StructuredPruningAction


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
