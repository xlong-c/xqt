"""Tests for Hopper SM90 WGMMA contracts and MoE Persistent Grouped GEMM."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from xqt.kernels.ops._impl.tilelang.sm90_wgmma import (
    SM90WgmmaSchedule,
    check_sm90_execution_readiness,
    resolve_sm90_wgmma_schedule,
    sm90_wgmma_linear_reference,
)
from xqt.kernels.ops.gemm.moe_grouped import (
    MoEGroupedGemmProblem,
    MoEPersistentScheduler,
    analyze_moe_grouped_layout,
    moe_grouped_gemm_reference,
)


def test_resolve_sm90_wgmma_schedule() -> None:
    # Decode phase (m <= 4)
    sched_decode = resolve_sm90_wgmma_schedule(m=1, n=4096, k=4096)
    assert sched_decode.target_arch == "sm_90"
    assert sched_decode.block_m == 16
    assert sched_decode.block_n == 128
    assert sched_decode.block_k == 64
    assert sched_decode.threads == 128
    assert sched_decode.use_wgmma is True
    assert sched_decode.use_tma is True
    assert sched_decode.num_stages == 3
    assert sched_decode.preset == "sm90_wgmma_decode"

    # Medium batch (m <= 64)
    sched_medium = resolve_sm90_wgmma_schedule(m=32, n=4096, k=4096)
    assert sched_medium.block_m == 64
    assert sched_medium.num_stages == 3
    assert sched_medium.preset == "sm90_wgmma_medium"

    # Prefill phase (m > 64)
    sched_prefill = resolve_sm90_wgmma_schedule(m=128, n=4096, k=4096)
    assert sched_prefill.block_m == 128
    assert sched_prefill.num_stages == 4
    assert sched_prefill.preset == "sm90_wgmma_prefill"

    # Dict serialization
    d = sched_prefill.to_dict()
    assert d["use_wgmma"] is True
    assert d["threads"] == 128


def test_check_sm90_execution_readiness() -> None:
    # CPU tensor check
    cpu_t = torch.randn(2, 4)
    ready, reason = check_sm90_execution_readiness(cpu_t)
    assert ready is False
    assert reason is not None
    assert "not resident on a CUDA device" in reason

    # Current host check
    if torch.cuda.is_available():
        cuda_t = torch.randn(2, 4, device="cuda")
        major, minor = torch.cuda.get_device_capability()
        ready, reason = check_sm90_execution_readiness(cuda_t)
        if (major, minor) == (9, 0):
            assert ready is True
            assert reason is None
        else:
            assert ready is False
            assert reason is not None
            assert f"sm_{major}{minor}" in reason


def test_sm90_wgmma_linear_reference() -> None:
    x = torch.randn(4, 32)
    w = torch.randn(64, 32)
    b = torch.randn(64)

    out = sm90_wgmma_linear_reference(x, w, b, activation="silu")
    expected = F.silu(F.linear(x, w, b))
    assert torch.allclose(out, expected, atol=1e-5)


def test_moe_grouped_gemm_layout_and_scheduling() -> None:
    # 4 experts: tokens per expert = 10, 0 (idle), 25, 5
    problem = MoEGroupedGemmProblem(
        num_experts=4,
        in_features=32,
        out_features=64,
        tokens_per_expert=(10, 0, 25, 5),
    )

    assert problem.total_tokens == 40
    assert problem.active_experts == 3

    report = analyze_moe_grouped_layout(problem)
    assert report.total_tokens == 40
    assert report.active_experts == 3
    assert report.expert_offsets == (0, 10, 10, 35)
    assert report.max_tokens_per_expert == 25
    assert report.imbalance_ratio > 1.0

    # Scheduler with cta_block_m = 16
    scheduler = MoEPersistentScheduler(cta_block_m=16)
    tasks = scheduler.plan_launches(problem)

    # Expert 0 (10 tokens) -> 1 tile (10 tokens)
    # Expert 1 (0 tokens) -> 0 tiles (skipped)
    # Expert 2 (25 tokens) -> 2 tiles (16 tokens + 9 tokens)
    # Expert 3 (5 tokens) -> 1 tile (5 tokens)
    # Total tasks: 1 + 2 + 1 = 4
    assert len(tasks) == 4
    assert tasks[0]["expert_id"] == 0 and tasks[0]["token_count"] == 10
    assert tasks[1]["expert_id"] == 2 and tasks[1]["token_count"] == 16
    assert tasks[2]["expert_id"] == 2 and tasks[2]["token_count"] == 9
    assert tasks[3]["expert_id"] == 3 and tasks[3]["token_count"] == 5


def test_moe_grouped_gemm_reference_correctness() -> None:
    num_experts = 3
    in_feat = 16
    out_feat = 32
    tokens = (8, 0, 12)
    problem = MoEGroupedGemmProblem(
        num_experts=num_experts,
        in_features=in_feat,
        out_features=out_feat,
        tokens_per_expert=tokens,
    )

    packed_tokens = torch.randn(20, in_feat)
    expert_weights = torch.randn(num_experts, out_feat, in_feat)
    expert_biases = torch.randn(num_experts, out_feat)

    out = moe_grouped_gemm_reference(
        packed_tokens,
        expert_weights,
        problem,
        expert_biases=expert_biases,
    )

    assert out.shape == (20, out_feat)

    # Verify per-expert slices
    # Expert 0: [0:8]
    expected_0 = packed_tokens[0:8] @ expert_weights[0].t() + expert_biases[0]
    assert torch.allclose(out[0:8], expected_0, atol=1e-5)

    # Expert 2: [8:20]
    expected_2 = packed_tokens[8:20] @ expert_weights[2].t() + expert_biases[2]
    assert torch.allclose(out[8:20], expected_2, atol=1e-5)
