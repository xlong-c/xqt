"""Latency benchmark helper."""

from dataclasses import dataclass
from time import perf_counter
from typing import Callable, List, Optional

import torch


@dataclass
class LatencyReport:
    """Latency summary in milliseconds."""

    iterations: int
    warmup: int
    mean_ms: float
    p50_ms: float
    p90_ms: float
    p99_ms: float
    samples_ms: List[float]

    def to_dict(self) -> dict[str, float | int | list[float]]:
        """Convert the report to a plain dictionary."""

        return {
            "iterations": self.iterations,
            "warmup": self.warmup,
            "mean_ms": self.mean_ms,
            "p50_ms": self.p50_ms,
            "p90_ms": self.p90_ms,
            "p99_ms": self.p99_ms,
            "samples_ms": self.samples_ms,
        }


def _sync_if_needed(sync_cuda: bool) -> None:
    if sync_cuda and torch.cuda.is_available():
        torch.cuda.synchronize()


def _percentile(sorted_values: List[float], percentile: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (percentile / 100.0) * (len(sorted_values) - 1)
    lower = int(rank)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = rank - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def benchmark_callable(
    fn: Callable[[], object],
    *,
    warmup: int = 10,
    iterations: int = 50,
    sync_cuda: bool = True,
    device: Optional[str] = None,
) -> LatencyReport:
    """Benchmark a zero-argument callable and return millisecond latency stats."""

    if warmup < 0:
        raise ValueError("warmup must be non-negative")
    if iterations <= 0:
        raise ValueError("iterations must be positive")

    torch_device = torch.device(device) if device is not None else None
    should_sync_cuda = sync_cuda and (
        torch_device is None or torch_device.type == "cuda"
    )

    with torch.no_grad():
        for _ in range(warmup):
            fn()
        _sync_if_needed(should_sync_cuda)

        samples_ms: List[float] = []
        for _ in range(iterations):
            _sync_if_needed(should_sync_cuda)
            start = perf_counter()
            fn()
            _sync_if_needed(should_sync_cuda)
            samples_ms.append((perf_counter() - start) * 1000.0)

    sorted_samples = sorted(samples_ms)
    mean_ms = sum(samples_ms) / len(samples_ms)
    return LatencyReport(
        iterations=iterations,
        warmup=warmup,
        mean_ms=mean_ms,
        p50_ms=_percentile(sorted_samples, 50),
        p90_ms=_percentile(sorted_samples, 90),
        p99_ms=_percentile(sorted_samples, 99),
        samples_ms=samples_ms,
    )


__all__ = ["LatencyReport", "benchmark_callable"]
