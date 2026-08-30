"""ViT hidden-width and MoE expert-drop action appliers."""

from __future__ import annotations

from typing import Mapping

import torch
from torch import nn

from ..report import StructuredPruningAction
from .linear_conv import (
    prune_conv2d_out_channels,
    prune_linear_in_features,
    prune_linear_in_out_features,
    prune_linear_out_features,
)
from .topology import (
    _get_module,
    _prune_layernorm_features,
    _prune_parameter_last_dim,
    _rewrite_indexed_container,
    _set_module,
)


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
