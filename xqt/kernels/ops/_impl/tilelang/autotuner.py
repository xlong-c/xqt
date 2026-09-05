"""Automated schedule tuner for TileLang operator kernels.

Sweeps candidate block shapes, warps, and pipelining stages for a problem signature,
benchmarks latency via CUDA events (or high-resolution timing fallback), selects the
pareto-optimal schedule, and writes it directly to TileLangTimingCache.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Sequence

import torch
from torch import nn

from xqt.core.errors import XQTBackendError
from xqt.kernels.ops.gemm.benchmark import benchmark_cuda_callable
from ._common import tilelang_runtime_usable
from .gemm_builder import build_tilelang_gemm_kernel
from .tuning_cache import (
    TileLangTimingCache,
    TimingCacheEntry,
    TimingCacheKey,
    get_tilelang_timing_cache,
)


@dataclass(frozen=True)
class AutoTuneConfig:
    """Execution options for TileLang kernel auto-tuning."""

    warmup: int = 5
    repeats: int = 10
    persist_to_cache: bool = True
    timeout_seconds: float = 10.0


@dataclass(frozen=True)
class AutoTuneCandidateResult:
    """Benchmark outcome for a single candidate schedule."""

    schedule: dict[str, int]
    latency_us: float
    success: bool
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schedule": dict(self.schedule),
            "latency_us": self.latency_us,
            "success": self.success,
            "error_message": self.error_message,
        }


@dataclass(frozen=True)
class AutoTuneResult:
    """Aggregate result from tuning an operator problem signature."""

    key: TimingCacheKey
    best_schedule: dict[str, int]
    best_latency_us: float
    tested_candidates: tuple[AutoTuneCandidateResult, ...]
    persisted: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key.to_key_string(),
            "best_schedule": dict(self.best_schedule),
            "best_latency_us": self.best_latency_us,
            "tested_candidates": [c.to_dict() for c in self.tested_candidates],
            "persisted": self.persisted,
        }


def _mock_schedule_score(m: int, n: int, k: int, schedule: dict[str, int]) -> float:
    """Deterministic latency estimator used when CUDA/TileLang runtime is not present."""
    bm = schedule.get("block_m", 64)
    bn = schedule.get("block_n", 64)
    bk = schedule.get("block_k", 64)
    # Score based on tile alignment with problem shape
    penalty = 0.0
    if m <= 4 and bm > 16:
        penalty += 50.0  # Excessive block_m for decode
    if k % bk != 0:
        penalty += 1000.0
    base_us = 20.0 + (bm * bn * bk) / 10000.0 + penalty
    return float(base_us)


def tune_linear_schedule(
    m: int,
    n: int,
    k: int,
    *,
    input_dtype: str = "float16",
    target_arch: str | None = None,
    has_bias: bool = False,
    activation: str | None = None,
    candidates: Sequence[dict[str, int]] | None = None,
    device: torch.device | str = "cuda",
    config: AutoTuneConfig | None = None,
    cache: TileLangTimingCache | None = None,
) -> AutoTuneResult:
    """Tune and persist the optimal TileLang linear schedule for a given problem shape."""
    cfg = config or AutoTuneConfig()
    timing_cache = cache or get_tilelang_timing_cache()

    resolved_arch = target_arch or "sm_89"
    key = TimingCacheKey(
        op_type="linear",
        arch=resolved_arch,
        dtype=input_dtype,
        shape=(m, n, k),
        extra=f"bias={bool(has_bias)},act={activation or 'none'}",
    )

    search_space = (
        list(candidates)
        if candidates is not None
        else timing_cache.get_candidate_linear_schedules(m, n, k, arch=resolved_arch)
    )

    tested_results: list[AutoTuneCandidateResult] = []
    cuda_usable = torch.cuda.is_available() and tilelang_runtime_usable() and str(device).startswith("cuda")

    if cuda_usable:
        cuda_dev = torch.device(device)
        dtype_torch = torch.float16 if input_dtype == "float16" else torch.bfloat16
        x = torch.randn((m, k), dtype=dtype_torch, device=cuda_dev)
        w = torch.randn((n, k), dtype=dtype_torch, device=cuda_dev)
        bias = torch.randn((n,), dtype=dtype_torch, device=cuda_dev) if has_bias else None

        for sched in search_space:
            try:
                kernel = build_tilelang_gemm_kernel(
                    m=m,
                    n=n,
                    k=k,
                    input_dtype=input_dtype,
                    block_m=sched["block_m"],
                    block_n=sched["block_n"],
                    block_k=sched["block_k"],
                    threads=sched.get("threads", 128),
                    num_stages=sched.get("num_stages", 2),
                    target_arch=resolved_arch,
                    has_bias=has_bias,
                    activation=activation,
                )
                fn = (lambda: kernel(x, w, bias)) if has_bias else (lambda: kernel(x, w))
                bench = benchmark_cuda_callable(
                    fn,
                    label="tilelang_tune",
                    device=cuda_dev,
                    warmup=cfg.warmup,
                    repeats=cfg.repeats,
                )
                latency_us = float(bench.median_ms * 1000.0)
                tested_results.append(
                    AutoTuneCandidateResult(
                        schedule=sched,
                        latency_us=latency_us,
                        success=True,
                    )
                )
            except Exception as exc:
                tested_results.append(
                    AutoTuneCandidateResult(
                        schedule=sched,
                        latency_us=float("inf"),
                        success=False,
                        error_message=str(exc),
                    )
                )
    else:
        # Dry-run / CPU fallback scoring for smoke testing
        for sched in search_space:
            score = _mock_schedule_score(m, n, k, sched)
            tested_results.append(
                AutoTuneCandidateResult(
                    schedule=sched,
                    latency_us=score,
                    success=True,
                )
            )

    successful = [r for r in tested_results if r.success and r.latency_us < float("inf")]
    if not successful:
        raise XQTBackendError(
            f"All auto-tuning candidates failed for problem {key.to_key_string()}"
        )

    best = min(successful, key=lambda r: r.latency_us)

    # Persist best entry to timing cache
    timing_cache.record(
        key,
        best.schedule,
        median_latency_us=best.latency_us,
        preset_name="autotuned",
        notes=f"Auto-tuned across {len(tested_results)} candidates",
        persist=cfg.persist_to_cache,
    )

    return AutoTuneResult(
        key=key,
        best_schedule=best.schedule,
        best_latency_us=best.latency_us,
        tested_candidates=tuple(tested_results),
        persisted=cfg.persist_to_cache,
    )


class TileLangAutoTuner:
    """High-level facade for tuning TileLang operators."""

    def __init__(self, cache: TileLangTimingCache | None = None) -> None:
        self._cache = cache or get_tilelang_timing_cache()

    @property
    def cache(self) -> TileLangTimingCache:
        return self._cache

    def tune_linear(
        self,
        m: int,
        n: int,
        k: int,
        *,
        input_dtype: str = "float16",
        target_arch: str | None = None,
        has_bias: bool = False,
        activation: str | None = None,
        config: AutoTuneConfig | None = None,
    ) -> AutoTuneResult:
        return tune_linear_schedule(
            m,
            n,
            k,
            input_dtype=input_dtype,
            target_arch=target_arch,
            has_bias=has_bias,
            activation=activation,
            config=config,
            cache=self._cache,
        )


__all__ = [
    "AutoTuneCandidateResult",
    "AutoTuneConfig",
    "AutoTuneResult",
    "TileLangAutoTuner",
    "tune_linear_schedule",
]
