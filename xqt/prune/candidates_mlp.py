"""Internal MLP candidate discovery helpers for structured pruning."""

from __future__ import annotations

import torch
from torch import nn

from .candidates import _CandidateDiscoveryResult, _StructuredCandidate
from .graph import PruningDependencyGraph


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


def collect_mlp_candidates(
    model: nn.Module,
    *,
    importance_metric: str,
) -> _CandidateDiscoveryResult:
    """Collect plain and gated MLP neuron pruning candidates."""

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
