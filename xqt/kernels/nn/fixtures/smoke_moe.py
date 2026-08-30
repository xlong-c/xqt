"""Smoke-only MoE module and model-family metadata helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch
from torch import nn

from xqt.contracts.moe import classify_moe_module


class SmokeMoEBlock(nn.Module):
    """Tiny MoE block with router / experts / shared expert for CPU smoke tests."""

    def __init__(
        self,
        *,
        hidden_dim: int = 16,
        intermediate_dim: int = 32,
        num_experts: int = 2,
    ) -> None:
        super().__init__()
        self.router = nn.Linear(hidden_dim, num_experts, bias=False)
        self.experts = nn.ModuleList(
            [
                nn.Linear(hidden_dim, intermediate_dim, bias=True)
                for _ in range(num_experts)
            ]
        )
        self.shared_expert = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.num_experts = num_experts

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        scores = torch.softmax(self.router(inputs), dim=-1)
        output = self.shared_expert(inputs) * 0.0
        for index, expert in enumerate(self.experts):
            output = output + expert(inputs) * scores[..., index : index + 1]
        return output


class SmokeMoE(nn.Module):
    """Tiny MoE model with a routing layer and expert modules."""

    def __init__(
        self,
        *,
        hidden_dim: int = 16,
        intermediate_dim: int = 32,
        num_experts: int = 2,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                SmokeMoEBlock(
                    hidden_dim=hidden_dim,
                    intermediate_dim=intermediate_dim,
                    num_experts=num_experts,
                )
                for _ in range(num_layers)
            ]
        )
        self.head = nn.Linear(hidden_dim, 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = inputs
        for block in self.blocks:
            hidden = block(hidden)
        return self.head(hidden)


def build_smoke_moe(
    *,
    hidden_dim: int = 16,
    intermediate_dim: int = 32,
    num_experts: int = 2,
    num_layers: int = 1,
) -> SmokeMoE:
    """Build a deterministic small MoE smoke model."""

    return SmokeMoE(
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        num_experts=num_experts,
        num_layers=num_layers,
    )


@dataclass(frozen=True)
class MoEFamilyReport:
    """Model-side MoE metadata and readiness summary."""

    expert_count: int
    shared_expert_count: int
    router_module_count: int
    expert_parallel_readiness: str
    load_balance_verified: bool = False
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "expert_count": self.expert_count,
            "shared_expert_count": self.shared_expert_count,
            "router_module_count": self.router_module_count,
            "expert_parallel_readiness": self.expert_parallel_readiness,
            "load_balance_verified": self.load_balance_verified,
            "notes": list(self.notes),
        }


def moe_family_report(model: nn.Module) -> MoEFamilyReport:
    """Summarize MoE module roles and parallel readiness metadata.

    XQT does not implement expert-parallel runtime; the report only records
    model-side metadata and keeps load-balance claims unverified unless routing
    data is supplied by the caller.
    """

    expert_count = 0
    shared_expert_count = 0
    router_count = 0
    for name, _module in model.named_modules():
        kind = classify_moe_module(name)
        if kind == "expert":
            expert_count += 1
        elif kind == "shared_expert":
            shared_expert_count += 1
        elif kind == "router":
            router_count += 1
    return MoEFamilyReport(
        expert_count=expert_count,
        shared_expert_count=shared_expert_count,
        router_module_count=router_count,
        expert_parallel_readiness="metadata_only",
        load_balance_verified=False,
        notes=(
            "expert parallel readiness is metadata only; XQT 不实现分布式 runtime.",
            "load balance 需调用方提供真实 routing 分数后才可验证.",
        ),
    )


def suggest_expert_pruning(
    router_scores: torch.Tensor,
    *,
    prune_count: int,
) -> list[int]:
    """Suggest expert indices to prune from average router scores.

    This is a pure suggestion helper; it never mutates the model. Experts with
    the lowest mean routing weight are returned first.
    """

    if prune_count < 0:
        raise ValueError("prune_count must be non-negative")
    if router_scores.ndim != 2:
        raise ValueError("router_scores must be (batch, num_experts)")
    num_experts = int(router_scores.shape[1])
    if prune_count >= num_experts:
        raise ValueError(
            f"prune_count {prune_count} must be smaller than expert count {num_experts}"
        )
    mean_scores = router_scores.detach().float().mean(dim=0)
    order = torch.argsort(mean_scores, descending=False).tolist()
    return [int(index) for index in order[:prune_count]]


__all__ = [
    "MoEFamilyReport",
    "SmokeMoE",
    "SmokeMoEBlock",
    "build_smoke_moe",
    "moe_family_report",
    "suggest_expert_pruning",
]
