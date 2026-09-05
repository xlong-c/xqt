"""Tests for TileLang Auto-Tuning and Schedule Cache Integration."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from xqt.kernels.ops._impl.tilelang.autotuner import (
    AutoTuneConfig,
    TileLangAutoTuner,
    tune_linear_schedule,
)
from xqt.kernels.ops._impl.tilelang.tuning_cache import (
    TileLangTimingCache,
    TimingCacheKey,
)


def test_tilelang_autotune_linear_schedule_dryrun() -> None:
    cache = TileLangTimingCache(cache_path=None)

    result = tune_linear_schedule(
        m=1,
        n=4096,
        k=4096,
        input_dtype="float16",
        target_arch="sm_89",
        cache=cache,
        config=AutoTuneConfig(persist_to_cache=False),
    )

    assert result.key.shape == (1, 4096, 4096)
    assert len(result.tested_candidates) >= 1
    assert result.best_latency_us > 0
    # Decode m=1 should favor smaller block_m
    assert result.best_schedule["block_m"] <= 32

    # Verify that cache now contains this tuned schedule
    cached_entry = cache.lookup(result.key)
    assert cached_entry is not None
    assert cached_entry.preset_name == "autotuned"
    assert cached_entry.schedule == result.best_schedule


def test_tilelang_autotune_custom_candidates() -> None:
    cache = TileLangTimingCache(cache_path=None)
    custom_candidates = [
        {"block_m": 16, "block_n": 64, "block_k": 32, "threads": 128, "num_stages": 2},
        {"block_m": 64, "block_n": 64, "block_k": 64, "threads": 128, "num_stages": 2},
    ]

    result = tune_linear_schedule(
        m=4,
        n=2048,
        k=4096,
        input_dtype="bfloat16",
        target_arch="sm_89",
        candidates=custom_candidates,
        cache=cache,
        config=AutoTuneConfig(persist_to_cache=False),
    )

    assert len(result.tested_candidates) == 2
    assert result.best_schedule in custom_candidates


def test_tilelang_autotuner_facade_and_persistence(tmp_path: Path) -> None:
    cache_path = tmp_path / "autotuned_cache.json"
    cache = TileLangTimingCache(cache_path=cache_path)
    tuner = TileLangAutoTuner(cache=cache)

    result = tuner.tune_linear(
        m=128,
        n=1024,
        k=1024,
        input_dtype="float16",
        target_arch="sm_89",
        config=AutoTuneConfig(persist_to_cache=True),
    )

    assert result.persisted is True
    assert cache_path.is_file()

    # Re-open cache from disk and verify persistence
    reloaded_cache = TileLangTimingCache(cache_path=cache_path)
    hit = reloaded_cache.lookup(result.key)
    assert hit is not None
    assert hit.preset_name == "autotuned"
    assert hit.schedule == result.best_schedule
