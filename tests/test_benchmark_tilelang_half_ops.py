from __future__ import annotations

import pytest
import torch

from tools.benchmark_tilelang_half_ops import (
    _cuda_graph_replay_callable,
    _per_call_latency,
    _repeat_callable,
)


def test_repeat_callable_returns_last_output_and_runs_exact_count() -> None:
    calls = 0

    def fn() -> torch.Tensor:
        nonlocal calls
        calls += 1
        return torch.tensor(calls)

    repeated = _repeat_callable(fn, 4)

    output = repeated()

    assert calls == 4
    assert output.item() == 4


def test_per_call_latency_scales_latency_fields_and_samples() -> None:
    report = {
        "iterations": 2,
        "warmup": 1,
        "mean_ms": 20.0,
        "p50_ms": 18.0,
        "p90_ms": 30.0,
        "p99_ms": 38.0,
        "samples_ms": [10.0, 30.0],
    }

    scaled = _per_call_latency(report, 10)

    assert scaled["iterations"] == 2
    assert scaled["warmup"] == 1
    assert scaled["inner_iterations"] == 10
    assert scaled["mean_ms"] == 2.0
    assert scaled["p50_ms"] == 1.8
    assert scaled["p90_ms"] == 3.0
    assert scaled["p99_ms"] == 3.8
    assert scaled["samples_ms"] == [1.0, 3.0]


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for CUDA Graph helper test",
)
def test_cuda_graph_replay_callable_returns_stable_tensor() -> None:
    source = torch.randn(8, device="cuda", dtype=torch.float16)

    def fn() -> torch.Tensor:
        return source + 1

    replay = _cuda_graph_replay_callable(fn, warmup=2)
    output = replay()

    assert output.is_cuda
    assert torch.allclose(output, source + 1)
