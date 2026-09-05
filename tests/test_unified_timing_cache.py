"""Tests for UnifiedKernelTimingCache, key serialization, and multi-DSL arbitration."""

from __future__ import annotations

from pathlib import Path

import pytest

from xqt.kernels.timing.cache import UnifiedKernelTimingCache
from xqt.kernels.timing.schema import (
    TimingCacheKey,
    TimingCacheRecord,
    TimingResolutionResult,
)


def test_timing_cache_key_roundtrip() -> None:
    key1 = TimingCacheKey(
        op="attention",
        backend="tilelang",
        arch="sm_89",
        dtype="float16",
        shape=(1, 8, 1, 1024, 64),
        extra="causal=True",
    )
    s1 = key1.to_key_string()
    assert s1 == "attention@tilelang@sm_89@float16@1x8x1x1024x64#causal=True"
    assert key1.problem_signature == "attention@sm_89@float16@1x8x1x1024x64#causal=True"

    restored = TimingCacheKey.from_key_string(s1)
    assert restored == key1

    with pytest.raises(ValueError, match="invalid unified timing cache key string"):
        TimingCacheKey.from_key_string("invalid@key@string")


def test_timing_cache_record_roundtrip() -> None:
    record = TimingCacheRecord(
        median_latency_us=25.4,
        p95_latency_us=29.1,
        min_latency_us=24.0,
        max_latency_us=31.2,
        schedule={"block_m": 16, "block_n": 64},
        verified_correct=True,
        max_abs_error=1.2e-4,
        preset_name="test_preset",
        notes="sample test benchmark",
    )
    d = record.to_dict()
    restored = TimingCacheRecord.from_dict(d)
    assert restored.median_latency_us == pytest.approx(25.4)
    assert restored.p95_latency_us == pytest.approx(29.1)
    assert restored.schedule["block_m"] == 16
    assert restored.verified_correct is True
    assert restored.max_abs_error == pytest.approx(1.2e-4)


def test_timing_cache_disk_persistence(tmp_path: Path) -> None:
    cache_path = tmp_path / "unified_timing_cache.json"
    cache = UnifiedKernelTimingCache(cache_path=cache_path)

    key = TimingCacheKey(
        op="linear",
        backend="triton",
        arch="sm_90",
        dtype="float16",
        shape=(16, 4096, 4096),
        extra="bias=False",
    )
    cache.record_measurement(
        op="linear",
        backend="triton",
        arch="sm_90",
        dtype="float16",
        shape=(16, 4096, 4096),
        median_latency_us=35.0,
        extra="bias=False",
        save=True,
    )

    assert cache_path.is_file()

    # Reload from disk
    reloaded = UnifiedKernelTimingCache(cache_path=cache_path)
    entry = reloaded.get(key)
    assert entry is not None
    assert entry.median_latency_us == pytest.approx(35.0)


def test_timing_cache_arbitration(tmp_path: Path) -> None:
    cache = UnifiedKernelTimingCache(cache_path=tmp_path / "cache.json")

    # Record 3 competing backends for the same operator problem
    shape = (1, 8, 1, 1024, 64)
    cache.record_measurement(
        op="attention",
        backend="torch",
        arch="sm_89",
        dtype="float16",
        shape=shape,
        median_latency_us=85.0,
        verified_correct=True,
        save=False,
    )
    cache.record_measurement(
        op="attention",
        backend="triton",
        arch="sm_89",
        dtype="float16",
        shape=shape,
        median_latency_us=32.0,
        verified_correct=True,
        save=False,
    )
    cache.record_measurement(
        op="attention",
        backend="tilelang",
        arch="sm_89",
        dtype="float16",
        shape=shape,
        median_latency_us=22.5,
        verified_correct=True,
        save=False,
    )

    # 1. Query all candidates
    candidates = cache.query_candidates(
        op="attention",
        arch="sm_89",
        dtype="float16",
        shape=shape,
    )
    assert set(candidates.keys()) == {"torch", "triton", "tilelang"}

    # 2. Query fastest overall
    res = cache.query_fastest(
        op="attention",
        arch="sm_89",
        dtype="float16",
        shape=shape,
    )
    assert res is not None
    assert res.fastest_backend == "tilelang"
    assert res.winning_record.median_latency_us == pytest.approx(22.5)
    assert res.speedup_vs_baseline == pytest.approx(85.0 / 22.5)

    # 3. Query fastest with restricted eligibility (only torch and triton)
    res_restricted = cache.query_fastest(
        op="attention",
        arch="sm_89",
        dtype="float16",
        shape=shape,
        eligible_backends=["torch", "triton"],
    )
    assert res_restricted is not None
    assert res_restricted.fastest_backend == "triton"
    assert res_restricted.winning_record.median_latency_us == pytest.approx(32.0)
