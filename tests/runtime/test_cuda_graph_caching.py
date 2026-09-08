"""Tests for Universal CUDA Graph caching, LRU eviction and parameter invalidation (XQT-013)."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from xqt.core.errors import XQTBackendError
from xqt.runtime.cuda_graph import (
    CUDAGraphBlockRunner,
    CUDAGraphCache,
    CUDAGraphCacheKey,
    CUDAGraphEntry,
    compute_model_param_fingerprint,
)


class _SimpleBlock(nn.Module):
    def __init__(self, dim: int = 16) -> None:
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.linear(x))


def test_cuda_graph_cache_key_binding() -> None:
    model = _SimpleBlock(dim=8)
    t1 = torch.randn(2, 4, 8)
    key1 = CUDAGraphCacheKey.from_inputs((t1,), model)

    # 1. 相同 shape/dtype/device/param -> key 相同
    key1_copy = CUDAGraphCacheKey.from_inputs((t1,), model)
    assert key1 == key1_copy

    # 2. 改变 shape -> key 不同
    t2 = torch.randn(2, 8, 8)
    key2 = CUDAGraphCacheKey.from_inputs((t2,), model)
    assert key1 != key2

    # 3. 改变 requires_grad -> key 不同
    t3 = torch.randn(2, 4, 8, requires_grad=True)
    key3 = CUDAGraphCacheKey.from_inputs((t3,), model)
    assert key1 != key3

    # 4. 改变模型参数 -> param_fingerprint 改变 -> key 不同
    fp_before = compute_model_param_fingerprint(model)
    with torch.no_grad():
        model.linear.weight.add_(1.0)
    fp_after = compute_model_param_fingerprint(model)
    assert fp_before != fp_after

    key4 = CUDAGraphCacheKey.from_inputs((t1,), model)
    assert key1 != key4


def test_cuda_graph_lru_budget_eviction_and_replay_rejection() -> None:
    # 模拟一个没有实际 CUDA device 的假 entry 验证 LRU 淘汰与失效
    cache = CUDAGraphCache(max_entries=2)
    model = _SimpleBlock(dim=4)

    t_a = torch.randn(1, 4)
    t_b = torch.randn(2, 4)
    t_c = torch.randn(3, 4)

    key_a = CUDAGraphCacheKey.from_inputs((t_a,), model)
    key_b = CUDAGraphCacheKey.from_inputs((t_b,), model)
    key_c = CUDAGraphCacheKey.from_inputs((t_c,), model)

    entry_a = CUDAGraphEntry(key=key_a, graph=None, static_inputs=(t_a,), static_output=t_a)
    entry_b = CUDAGraphEntry(key=key_b, graph=None, static_inputs=(t_b,), static_output=t_b)
    entry_c = CUDAGraphEntry(key=key_c, graph=None, static_inputs=(t_c,), static_output=t_c)

    # 存入 A 和 B
    evicted = cache.put(entry_a)
    assert evicted is None
    evicted = cache.put(entry_b)
    assert evicted is None
    assert len(cache) == 2

    # 存入 C 时，最老的 A 必须被淘汰
    evicted = cache.put(entry_c)
    assert evicted is entry_a
    assert entry_a.evicted is True
    assert entry_a.is_valid is False
    assert cache.get(key_a) is None
    assert cache.get(key_b) is entry_b
    assert cache.get(key_c) is entry_c

    # 断言被淘汰的 graph 无法再次 replay
    with pytest.raises(XQTBackendError, match="invalid or evicted"):
        entry_a.replay((t_a,))


def test_cuda_graph_parameter_invalidation() -> None:
    cache = CUDAGraphCache(max_entries=4)
    model = _SimpleBlock(dim=4)
    fp_orig = compute_model_param_fingerprint(model)

    t = torch.randn(1, 4)
    key = CUDAGraphCacheKey.from_inputs((t,), model)
    entry = CUDAGraphEntry(key=key, graph=None, static_inputs=(t,), static_output=t)
    cache.put(entry)
    assert cache.get(key) is entry

    # 更新模型参数
    with torch.no_grad():
        model.linear.bias.add_(0.5)
    fp_new = compute_model_param_fingerprint(model)
    assert fp_orig != fp_new

    # 执行失效操作
    invalidated = cache.invalidate_stale_parameters(fp_new)
    assert len(invalidated) == 1
    assert entry.is_valid is False
    assert cache.get(key) is None


def test_cuda_graph_runner_fallback_control() -> None:
    model = _SimpleBlock(dim=4)
    cpu_tensor = torch.randn(2, 4)

    # 1. 当 allow_eager_fallback=False 时，传入 CPU tensor 必须严密报错
    runner_strict = CUDAGraphBlockRunner(model, allow_eager_fallback=False)
    with pytest.raises(XQTBackendError, match="requires all inputs to be on CUDA device"):
        runner_strict(cpu_tensor)

    # 2. 当 allow_eager_fallback=True 时，优雅回退到 eager 并记录 report 原因
    runner_tolerant = CUDAGraphBlockRunner(model, allow_eager_fallback=True)
    out = runner_tolerant(cpu_tensor)
    assert runner_tolerant.last_execution_report["mode"] == "eager_fallback"
    assert runner_tolerant.last_execution_report["reason"] == "inputs_not_cuda"
    ref_out = model(cpu_tensor)
    assert torch.allclose(out, ref_out)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_graph_runner_continuous_replay_hardware() -> None:
    model = _SimpleBlock(dim=8).cuda().eval()
    runner = CUDAGraphBlockRunner(model, allow_eager_fallback=False, max_cache_size=2)

    x = torch.randn(2, 8, device="cuda")
    ref_out = model(x)

    # 第一次运行：触发 capture
    out1 = runner(x)
    assert runner.last_execution_report["status"] == "captured"
    assert torch.allclose(out1, ref_out, atol=1e-4)

    # 第二次运行相同 shape：触发 cache_hit 并连续 replay
    out2 = runner(x)
    assert runner.last_execution_report["status"] == "cache_hit"
    assert torch.allclose(out2, ref_out, atol=1e-4)
