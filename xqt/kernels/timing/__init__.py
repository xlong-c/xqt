"""Unified timing cache and benchmark arena across DSL compiler backends."""

from __future__ import annotations

from xqt.kernels.timing.arena import (
    ArenaComparisonReport,
    BenchmarkResult,
    KernelArena,
)
from xqt.kernels.timing.cache import (
    UnifiedKernelTimingCache,
    get_timing_cache,
    set_timing_cache,
)
from xqt.kernels.timing.schema import (
    TimingCacheKey,
    TimingCacheRecord,
    TimingResolutionResult,
)

__all__ = [
    "ArenaComparisonReport",
    "BenchmarkResult",
    "KernelArena",
    "TimingCacheKey",
    "TimingCacheRecord",
    "TimingResolutionResult",
    "UnifiedKernelTimingCache",
    "get_timing_cache",
    "set_timing_cache",
]
