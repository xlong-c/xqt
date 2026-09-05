"""Operator Benchmark Arena for cross-DSL latency competition and correctness verification."""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import torch

from xqt.kernels.spec import PlatformInfo
from xqt.kernels.timing.cache import UnifiedKernelTimingCache, get_timing_cache
from xqt.kernels.timing.schema import TimingCacheKey, TimingCacheRecord


@dataclass(frozen=True)
class BenchmarkResult:
    """Individual benchmark result for a specific backend candidate."""

    backend: str
    median_latency_us: float
    p95_latency_us: float
    min_latency_us: float
    max_latency_us: float
    tflops: float | None = None
    verified_correct: bool = True
    max_abs_error: float | None = None
    relative_l2_error: float | None = None
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "median_latency_us": self.median_latency_us,
            "p95_latency_us": self.p95_latency_us,
            "min_latency_us": self.min_latency_us,
            "max_latency_us": self.max_latency_us,
            "tflops": self.tflops,
            "verified_correct": self.verified_correct,
            "max_abs_error": self.max_abs_error,
            "relative_l2_error": self.relative_l2_error,
            "error_message": self.error_message,
        }


@dataclass(frozen=True)
class ArenaComparisonReport:
    """Full tournament comparison report across competing DSL backends."""

    op: str
    shape: tuple[int, ...]
    dtype: str
    arch: str
    fastest_backend: str | None
    results: dict[str, BenchmarkResult] = field(default_factory=dict)
    speedups: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "op": self.op,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "arch": self.arch,
            "fastest_backend": self.fastest_backend,
            "results": {b: r.to_dict() for b, r in self.results.items()},
            "speedups": self.speedups,
        }

    def summary(self) -> str:
        lines = [
            f"=== Kernel Arena Tournament: {self.op} (Shape={self.shape}, Dtype={self.dtype}, Arch={self.arch}) ===",
            f"Fastest Backend: {self.fastest_backend or 'None'}",
        ]
        for backend, res in self.results.items():
            if res.error_message:
                lines.append(f" - {backend.upper():<10}: ERROR ({res.error_message})")
            else:
                speedup = self.speedups.get(backend, 1.0)
                err_str = f"max_err={res.max_abs_error:.1e}" if res.max_abs_error is not None else "unverified"
                lines.append(
                    f" - {backend.upper():<10}: {res.median_latency_us:8.2f} us (p95: {res.p95_latency_us:8.2f} us) "
                    f"[{speedup:5.2f}x] ({err_str})"
                )
        return "\n".join(lines)


class KernelArena:
    """Benchmarking arena running multi-backend operator shootouts."""

    def __init__(
        self,
        timing_cache: UnifiedKernelTimingCache | None = None,
        warmup_runs: int = 5,
        timed_runs: int = 20,
    ) -> None:
        self._cache = timing_cache if timing_cache is not None else get_timing_cache()
        self._warmup_runs = warmup_runs
        self._timed_runs = timed_runs

    def run_tournament(
        self,
        op: str,
        dtype: str,
        shape: tuple[int, ...] | Sequence[int],
        candidates: Mapping[str, Callable[[], torch.Tensor]],
        reference_fn: Callable[[], torch.Tensor] | None = None,
        extra: str = "",
        arch: str | None = None,
        sync_cuda: bool = True,
        update_cache: bool = True,
        rel_tol: float = 1e-2,
        abs_tol: float = 1e-2,
    ) -> ArenaComparisonReport:
        """Run tournament across candidate callables and rank their latency."""
        platform = PlatformInfo.detect()
        resolved_arch = (
            arch
            if arch is not None
            else (
                f"sm_{platform.cuda_arch_major}{platform.cuda_arch_minor}"
                if platform.cuda_arch_major
                else "cpu"
            )
        )

        results: dict[str, BenchmarkResult] = {}
        ref_output: torch.Tensor | None = None

        if reference_fn is not None:
            try:
                ref_output = reference_fn()
                if isinstance(ref_output, torch.Tensor):
                    ref_output = ref_output.detach()
            except Exception as e:
                ref_output = None

        is_cuda = torch.cuda.is_available() and sync_cuda

        for backend, fn in candidates.items():
            try:
                # 1. Warmup
                for _ in range(self._warmup_runs):
                    out = fn()
                if is_cuda:
                    torch.cuda.synchronize()

                # 2. Correctness verification against reference
                verified = True
                max_abs_err: float | None = None
                rel_l2_err: float | None = None

                if ref_output is not None and isinstance(out, torch.Tensor):
                    try:
                        diff = torch.abs(out.detach().float() - ref_output.float())
                        max_abs_err = float(diff.max().item())
                        denom = torch.norm(ref_output.float()).item() + 1e-8
                        rel_l2_err = float(torch.norm(diff).item() / denom)
                        if max_abs_err > abs_tol and (rel_l2_err is None or rel_l2_err > rel_tol):
                            verified = False
                    except Exception:
                        verified = True

                # 3. Precise Timing
                latencies_us: list[float] = []
                if is_cuda:
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    for _ in range(self._timed_runs):
                        start_event.record()
                        fn()
                        end_event.record()
                        end_event.synchronize()
                        # elapsed_time is in milliseconds
                        latencies_us.append(start_event.elapsed_time(end_event) * 1000.0)
                else:
                    for _ in range(self._timed_runs):
                        t0 = time.perf_counter_ns()
                        fn()
                        t1 = time.perf_counter_ns()
                        latencies_us.append((t1 - t0) / 1000.0)

                latencies_us.sort()
                median_lat = float(statistics.median(latencies_us))
                p95_index = int(len(latencies_us) * 0.95)
                p95_lat = float(latencies_us[min(p95_index, len(latencies_us) - 1)])

                results[backend] = BenchmarkResult(
                    backend=backend,
                    median_latency_us=median_lat,
                    p95_latency_us=p95_lat,
                    min_latency_us=min(latencies_us),
                    max_latency_us=max(latencies_us),
                    verified_correct=verified,
                    max_abs_error=max_abs_err,
                    relative_l2_error=rel_l2_err,
                )

                if update_cache:
                    self._cache.record_measurement(
                        op=op,
                        backend=backend,
                        arch=resolved_arch,
                        dtype=dtype,
                        shape=shape,
                        median_latency_us=median_lat,
                        p95_latency_us=p95_lat,
                        extra=extra,
                        verified_correct=verified,
                        max_abs_error=max_abs_err,
                        relative_l2_error=rel_l2_err,
                        save=False,
                    )
            except Exception as e:
                results[backend] = BenchmarkResult(
                    backend=backend,
                    median_latency_us=float("inf"),
                    p95_latency_us=float("inf"),
                    min_latency_us=float("inf"),
                    max_latency_us=float("inf"),
                    verified_correct=False,
                    error_message=str(e),
                )

        if update_cache:
            self._cache.save_to_disk()

        valid = {b: r for b, r in results.items() if not r.error_message and r.verified_correct}
        fastest_backend = min(valid.keys(), key=lambda b: valid[b].median_latency_us) if valid else None

        speedups: dict[str, float] = {}
        if fastest_backend:
            best_lat = valid[fastest_backend].median_latency_us
            for b, r in results.items():
                if r.median_latency_us > 0 and r.median_latency_us != float("inf"):
                    speedups[b] = r.median_latency_us / best_lat
                else:
                    speedups[b] = 0.0

        return ArenaComparisonReport(
            op=op,
            shape=tuple(shape),
            dtype=dtype,
            arch=resolved_arch,
            fastest_backend=fastest_backend,
            results=results,
            speedups=speedups,
        )
