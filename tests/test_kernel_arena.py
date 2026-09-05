"""Tests for KernelArena multi-backend tournament and correctness verification."""

from __future__ import annotations

import time
from pathlib import Path

import pytest
import torch

from xqt.kernels.timing.arena import KernelArena
from xqt.kernels.timing.cache import UnifiedKernelTimingCache


def test_kernel_arena_tournament_cpu(tmp_path: Path) -> None:
    cache = UnifiedKernelTimingCache(cache_path=tmp_path / "arena_cache.json")
    arena = KernelArena(timing_cache=cache, warmup_runs=2, timed_runs=5)

    x = torch.randn(4, 16)
    w = torch.randn(16, 32)

    # 1. Define candidate implementations
    def fn_reference() -> torch.Tensor:
        return torch.matmul(x, w)

    def fn_fast() -> torch.Tensor:
        # Simulate faster execution
        return torch.matmul(x, w)

    def fn_slow() -> torch.Tensor:
        # Simulate slower execution
        time.sleep(0.001)
        return torch.matmul(x, w)

    def fn_broken() -> torch.Tensor:
        raise RuntimeError("simulated kernel compilation crash")

    candidates = {
        "fast_dsl": fn_fast,
        "slow_dsl": fn_slow,
        "broken_dsl": fn_broken,
    }

    report = arena.run_tournament(
        op="test.gemm",
        dtype="float32",
        shape=(4, 16, 32),
        candidates=candidates,
        reference_fn=fn_reference,
        sync_cuda=False,
        update_cache=True,
    )

    # Verification
    assert report.op == "test.gemm"
    assert report.fastest_backend == "fast_dsl"
    assert "fast_dsl" in report.results
    assert "slow_dsl" in report.results
    assert "broken_dsl" in report.results

    fast_res = report.results["fast_dsl"]
    assert fast_res.verified_correct is True
    assert fast_res.max_abs_error is not None
    assert fast_res.max_abs_error < 1e-4

    slow_res = report.results["slow_dsl"]
    assert slow_res.verified_correct is True
    assert slow_res.median_latency_us > fast_res.median_latency_us

    broken_res = report.results["broken_dsl"]
    assert broken_res.verified_correct is False
    assert broken_res.error_message == "simulated kernel compilation crash"

    # Speedups
    assert report.speedups["fast_dsl"] == pytest.approx(1.0)
    assert report.speedups["slow_dsl"] > 1.0

    # Summary text
    summary = report.summary()
    assert "FAST_DSL" in summary
    assert "BROKEN_DSL" in summary
    assert "ERROR" in summary

    # Verify cache was updated
    arch = report.arch
    cached_candidates = cache.query_candidates(
        op="test.gemm",
        arch=arch,
        dtype="float32",
        shape=(4, 16, 32),
    )
    assert "fast_dsl" in cached_candidates
    assert cached_candidates["fast_dsl"][1].verified_correct is True
