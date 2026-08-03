"""Internal ViT width candidate discovery helpers for structured pruning."""

from __future__ import annotations

from typing import Any, Callable, Mapping, Optional

import torch
from torch import nn

from .candidates import _CandidateDiscoveryResult, _StructuredCandidate
from .graph import PruningDependencyGraph


def _layernorm_feature_count(module: nn.LayerNorm) -> Optional[int]:
    shape = module.normalized_shape
    if isinstance(shape, int):
        return int(shape)
    if isinstance(shape, tuple) and len(shape) == 1:
        return int(shape[0])
    return None


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
    *,
    get_module: Callable[[nn.Module, str], nn.Module],
) -> torch.Tensor:
    embed_dim = int(descriptor["embed_dim"])
    score = torch.zeros(embed_dim, dtype=torch.float32)
    patch_proj = get_module(model, str(descriptor["patch_proj_name"]))
    norm = get_module(model, str(descriptor["norm_name"]))
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
        head = get_module(model, head_name)
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
        qkv = get_module(model, str(block["qkv_name"]))
        proj = get_module(model, str(block["proj_name"]))
        fc1 = get_module(model, str(block["fc1_name"]))
        fc2 = get_module(model, str(block["fc2_name"]))
        norm1 = get_module(model, str(block["norm1_name"]))
        norm2 = get_module(model, str(block["norm2_name"]))
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


def collect_vit_hidden_width_candidates(
    model: nn.Module,
    *,
    importance_metric: str,
    get_module: Callable[[nn.Module, str], nn.Module],
) -> _CandidateDiscoveryResult:
    """Collect ViT hidden-width pruning candidates."""

    candidates: list[_StructuredCandidate] = []
    graph = PruningDependencyGraph()
    for module_name, module in model.named_modules():
        descriptor_module_name = module_name or "<root>"
        descriptor = _vit_hidden_descriptor(descriptor_module_name, module)
        if descriptor is None:
            continue
        scores = _vit_hidden_width_scores(
            descriptor,
            model,
            importance_metric,
            get_module=get_module,
        )
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


def collect_vit_embedding_width_candidates(
    model: nn.Module,
    *,
    importance_metric: str,
    get_module: Callable[[nn.Module, str], nn.Module],
) -> _CandidateDiscoveryResult:
    """Collect ViT embedding-width pruning candidates."""

    result = collect_vit_hidden_width_candidates(
        model,
        importance_metric=importance_metric,
        get_module=get_module,
    )
    for candidate in result.candidates:
        candidate.adapter = "vit_embedding_width_adapter"
        candidate.granularity = "embedding_width"
        candidate.metadata["width_kind"] = "embedding"
    return result
