"""Internal mixture-of-experts candidate discovery helpers for structured pruning."""

from __future__ import annotations

import torch
from torch import nn

from .candidates import _CandidateDiscoveryResult, _StructuredCandidate
from .graph import PruningDependencyGraph


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


def collect_expert_candidates(
    model: nn.Module,
    *,
    importance_metric: str,
) -> _CandidateDiscoveryResult:
    """Collect MoE expert pruning candidates."""

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
