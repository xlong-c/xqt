"""Internal attention-head candidate discovery helpers for structured pruning."""

from __future__ import annotations

from typing import Any, Callable, Mapping, Optional

import torch
from torch import nn

from .candidates import _CandidateDiscoveryResult, _StructuredCandidate
from .graph import PruningDependencyGraph


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
    *,
    infer_attention_role: Callable[[nn.Module], str],
    attention_variant: Callable[..., str],
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
    role = infer_attention_role(module)
    variant = attention_variant(num_heads=num_heads, num_kv_heads=num_kv_heads)

    qkv = getattr(module, "qkv", None)
    proj = getattr(module, "proj", None)
    if isinstance(qkv, nn.Linear) and isinstance(proj, nn.Linear):
        return {
            "module_name": module_name,
            "attention_kind": "fused_qkv",
            "attention_role": role,
            "attention_variant": variant,
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
            "attention_role": role,
            "attention_variant": variant,
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


def collect_head_candidates(
    model: nn.Module,
    *,
    importance_metric: str,
    infer_attention_role: Callable[[nn.Module], str],
    attention_variant: Callable[..., str],
) -> _CandidateDiscoveryResult:
    """Collect attention-head pruning candidates."""

    candidates: list[_StructuredCandidate] = []
    graph = PruningDependencyGraph()
    for module_name, module in model.named_modules():
        if not module_name:
            continue
        descriptor = _build_attention_descriptor(
            module_name,
            module,
            infer_attention_role=infer_attention_role,
            attention_variant=attention_variant,
        )
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
                    **{key: value for key, value in descriptor.items() if key.endswith("_name")},
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
