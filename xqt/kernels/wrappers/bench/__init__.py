"""Benchmark helpers for XQT."""

from .latency import LatencyReport, benchmark_callable, measure_callable_ms
from .memory import MemoryReport, benchmark_memory
from .phase_latency import (
    PhaseLatencyReport,
    benchmark_prefill_decode,
    merge_phase_into_metrics,
)
from .profiler import ProfiledOperatorRecord, ProfilerReport, profile_callable

__all__ = [
    "LatencyReport",
    "MemoryReport",
    "PhaseLatencyReport",
    "ProfiledOperatorRecord",
    "ProfilerReport",
    "benchmark_callable",
    "benchmark_memory",
    "benchmark_prefill_decode",
    "measure_callable_ms",
    "merge_phase_into_metrics",
    "profile_callable",
]
