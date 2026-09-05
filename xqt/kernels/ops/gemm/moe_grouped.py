"""MoE Persistent Grouped GEMM contracts and layout scheduling.

Provides structured execution abstractions for Mixture-of-Experts (MoE) architectures
where token dispatch produces unbalanced, sparse token counts per expert. Consolidates
scattered expert launches into a unified persistent grouped scheduling plan to eliminate
kernel launch overhead during small-batch inference.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any, Sequence

import torch
from torch import nn

from xqt.core.errors import XQTBackendError


@dataclass(frozen=True)
class MoEGroupedGemmProblem:
    """Problem signature for grouped GEMM across multiple experts."""

    num_experts: int
    in_features: int
    out_features: int
    tokens_per_expert: tuple[int, ...]
    dtype: str = "bfloat16"
    device: str = "cuda"

    def __post_init__(self) -> None:
        if self.num_experts <= 0:
            raise ValueError(f"num_experts must be positive, got {self.num_experts}")
        if self.in_features <= 0 or self.out_features <= 0:
            raise ValueError("in_features and out_features must be positive")
        if len(self.tokens_per_expert) != self.num_experts:
            raise ValueError(
                f"tokens_per_expert length ({len(self.tokens_per_expert)}) must match num_experts ({self.num_experts})"
            )
        if any(c < 0 for c in self.tokens_per_expert):
            raise ValueError("token counts must be non-negative")

    @property
    def total_tokens(self) -> int:
        return sum(self.tokens_per_expert)

    @property
    def active_experts(self) -> int:
        return sum(1 for c in self.tokens_per_expert if c > 0)


@dataclass(frozen=True)
class MoEGroupedLayoutReport:
    """Diagnostic layout and workload summary for grouped expert execution."""

    num_experts: int
    total_tokens: int
    active_experts: int
    max_tokens_per_expert: int
    imbalance_ratio: float
    expert_offsets: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_experts": self.num_experts,
            "total_tokens": self.total_tokens,
            "active_experts": self.active_experts,
            "max_tokens_per_expert": self.max_tokens_per_expert,
            "imbalance_ratio": self.imbalance_ratio,
            "expert_offsets": list(self.expert_offsets),
        }


def analyze_moe_grouped_layout(problem: MoEGroupedGemmProblem) -> MoEGroupedLayoutReport:
    """Analyze token distribution, offsets, and load imbalance across experts."""
    counts = problem.tokens_per_expert
    total = sum(counts)
    active = sum(1 for c in counts if c > 0)
    max_c = max(counts) if counts else 0
    avg_active = (total / float(active)) if active > 0 else 1.0
    imbalance = (max_c / avg_active) if avg_active > 0 else 1.0

    offsets = [0]
    curr = 0
    for c in counts:
        curr += c
        offsets.append(curr)

    return MoEGroupedLayoutReport(
        num_experts=problem.num_experts,
        total_tokens=total,
        active_experts=active,
        max_tokens_per_expert=max_c,
        imbalance_ratio=float(imbalance),
        expert_offsets=tuple(offsets[:-1]),
    )


class MoEPersistentScheduler:
    """Scheduler generating persistent work units across unbalanced expert batches."""

    def __init__(self, cta_block_m: int = 64) -> None:
        self.cta_block_m = int(cta_block_m)

    def plan_launches(self, problem: MoEGroupedGemmProblem) -> list[dict[str, Any]]:
        """Partition non-empty expert work into persistent tile assignments."""
        report = analyze_moe_grouped_layout(problem)
        tasks: list[dict[str, Any]] = []

        for expert_id, count in enumerate(problem.tokens_per_expert):
            if count == 0:
                continue
            offset = report.expert_offsets[expert_id]
            num_tiles = (count + self.cta_block_m - 1) // self.cta_block_m
            for tile_idx in range(num_tiles):
                tile_start = offset + tile_idx * self.cta_block_m
                tile_count = min(self.cta_block_m, offset + count - tile_start)
                tasks.append(
                    {
                        "expert_id": expert_id,
                        "token_offset": tile_start,
                        "token_count": tile_count,
                        "tile_index": tile_idx,
                    }
                )
        return tasks


def moe_grouped_gemm_reference(
    packed_tokens: torch.Tensor,
    expert_weights: torch.Tensor,
    problem: MoEGroupedGemmProblem,
    *,
    expert_biases: torch.Tensor | None = None,
) -> torch.Tensor:
    """Numerical reference for MoE grouped GEMM.

    Args:
        packed_tokens: [total_tokens, in_features]
        expert_weights: [num_experts, out_features, in_features]
        problem: MoEGroupedGemmProblem definition
        expert_biases: optional [num_experts, out_features]
    """
    total = problem.total_tokens
    if packed_tokens.shape[0] != total:
        raise ValueError(f"packed_tokens shape[0] ({packed_tokens.shape[0]}) != total_tokens ({total})")

    report = analyze_moe_grouped_layout(problem)
    output = torch.zeros(
        (total, problem.out_features),
        dtype=packed_tokens.dtype,
        device=packed_tokens.device,
    )

    for expert_id, count in enumerate(problem.tokens_per_expert):
        if count == 0:
            continue
        start = report.expert_offsets[expert_id]
        end = start + count
        x_e = packed_tokens[start:end]
        w_e = expert_weights[expert_id].to(dtype=packed_tokens.dtype, device=packed_tokens.device)
        y_e = x_e @ w_e.t()
        if expert_biases is not None:
            b_e = expert_biases[expert_id].to(dtype=packed_tokens.dtype, device=packed_tokens.device)
            y_e = y_e + b_e
        output[start:end] = y_e

    return output


__all__ = [
    "MoEGroupedGemmProblem",
    "MoEGroupedLayoutReport",
    "MoEPersistentScheduler",
    "analyze_moe_grouped_layout",
    "moe_grouped_gemm_reference",
]
