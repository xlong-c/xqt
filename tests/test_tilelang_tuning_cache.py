"""Tests for TileLang timing cache and adaptive schedule resolution."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from xqt.kernels.ops._impl.tilelang.linear import (
    resolve_tilelang_linear_schedule,
)
from xqt.kernels.ops._impl.tilelang.tuning_cache import (
    TileLangTimingCache,
    TimingCacheEntry,
    TimingCacheKey,
    get_tilelang_timing_cache,
)


def test_timing_cache_key_roundtrip() -> None:
    key1 = TimingCacheKey(
        op_type="linear",
        arch="sm_89",
        dtype="bfloat16",
        shape=(1, 4096, 4096),
        extra="bias=False,act=none",
    )
    s1 = key1.to_key_string()
    assert s1 == "linear@sm_89@bfloat16@1x4096x4096#bias=False,act=none"
    parsed1 = TimingCacheKey.from_key_string(s1)
    assert parsed1 == key1

    key2 = TimingCacheKey(
        op_type="gemm",
        arch="sm_90",
        dtype="float16",
        shape=(128, 256, 512),
    )
    s2 = key2.to_key_string()
    assert s2 == "gemm@sm_90@float16@128x256x512"
    parsed2 = TimingCacheKey.from_key_string(s2)
    assert parsed2 == key2

    with pytest.raises(ValueError, match="invalid timing cache key string"):
        TimingCacheKey.from_key_string("invalid_key_string")


def test_timing_cache_entry_roundtrip() -> None:
    entry = TimingCacheEntry(
        schedule={"block_m": 32, "block_n": 128, "block_k": 64, "threads": 256},
        median_latency_us=45.2,
        preset_name="test_preset",
        notes="sample note",
    )
    d = entry.to_dict()
    restored = TimingCacheEntry.from_dict(d)
    assert restored.schedule == entry.schedule
    assert restored.median_latency_us == pytest.approx(45.2)
    assert restored.preset_name == "test_preset"
    assert restored.notes == "sample note"


def test_timing_cache_disk_persistence(tmp_path: Path) -> None:
    cache_file = tmp_path / "custom_cache.json"
    cache = TileLangTimingCache(cache_path=cache_file)

    key = TimingCacheKey(
        op_type="linear",
        arch="sm_90",
        dtype="bfloat16",
        shape=(1, 2048, 4096),
        extra="bias=False,act=none",
    )
    schedule = {
        "block_m": 16,
        "block_n": 128,
        "block_k": 32,
        "threads": 128,
        "num_stages": 3,
    }
    cache.record(
        key,
        schedule,
        median_latency_us=12.5,
        preset_name="sm90_test_preset",
        notes="persisted entry",
        persist=True,
    )

    assert cache_file.is_file()

    # Re-open and verify
    cache_reloaded = TileLangTimingCache(cache_path=cache_file)
    entry = cache_reloaded.lookup(key)
    assert entry is not None
    assert entry.schedule["block_n"] == 128
    assert entry.preset_name == "sm90_test_preset"
    assert entry.median_latency_us == pytest.approx(12.5)


def test_timing_cache_candidate_generation() -> None:
    cache = TileLangTimingCache(cache_path=None)

    # Decode candidates (m <= 4)
    candidates_decode = cache.get_candidate_linear_schedules(m=1, n=4096, k=4096)
    assert len(candidates_decode) >= 1
    assert any(c["block_m"] == 16 for c in candidates_decode)

    # Prefill candidates (m > 64)
    candidates_prefill = cache.get_candidate_linear_schedules(m=512, n=4096, k=4096)
    assert len(candidates_prefill) >= 1
    assert any(c["block_m"] >= 64 for c in candidates_prefill)

    # Divisibility filtering: if k % block_k != 0, it should filter out invalid ones
    candidates_k96 = cache.get_candidate_linear_schedules(m=1, n=4096, k=96)
    assert len(candidates_k96) >= 1
    assert all(96 % c["block_k"] == 0 for c in candidates_k96)
    assert all(c["block_k"] != 64 for c in candidates_k96)


def test_resolve_tilelang_linear_schedule_timing_cache_hit() -> None:
    cache = get_tilelang_timing_cache()
    custom_key = TimingCacheKey(
        op_type="linear",
        arch="sm_90",
        dtype="bfloat16",
        shape=(1, 8192, 4096),
        extra="bias=False,act=none",
    )
    cache.record(
        custom_key,
        {
            "block_m": 32,
            "block_n": 128,
            "block_k": 64,
            "threads": 256,
            "num_stages": 3,
        },
        preset_name="sm90_tuned_preset",
        persist=False,
    )

    x = torch.empty(1, 4096, dtype=torch.bfloat16)
    schedule = resolve_tilelang_linear_schedule(
        x,
        out_features=8192,
        activation=None,
        target_arch="sm_90",
    )

    assert schedule.preset == "sm90_tuned_preset"
    assert schedule.block_m == 32
    assert schedule.block_n == 128
    assert schedule.block_k == 64
    assert schedule.threads == 256
    assert schedule.num_stages == 3

    # Explicit override should take precedence over cached values
    schedule_overridden = resolve_tilelang_linear_schedule(
        x,
        out_features=8192,
        activation=None,
        block_m=16,
        threads=128,
        target_arch="sm_90",
    )
    assert schedule_overridden.block_m == 16
    assert schedule_overridden.threads == 128
    assert schedule_overridden.block_n == 128
    assert schedule_overridden.num_stages == 3
