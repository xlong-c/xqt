"""Memory benchmark helper."""

from __future__ import annotations

import os
import resource
from dataclasses import dataclass
from typing import Callable, Optional

import torch


@dataclass
class MemoryReport:
    """Memory summary in bytes."""

    backend: str
    before_bytes: int
    after_bytes: int
    delta_bytes: int
    peak_bytes: Optional[int] = None
    cuda_peak_allocated_bytes: Optional[int] = None
    cuda_peak_reserved_bytes: Optional[int] = None

    def to_dict(self) -> dict[str, int | str | None]:
        """Convert the report to a plain dictionary."""

        return {
            "backend": self.backend,
            "before_bytes": self.before_bytes,
            "after_bytes": self.after_bytes,
            "delta_bytes": self.delta_bytes,
            "peak_bytes": self.peak_bytes,
            "cuda_peak_allocated_bytes": self.cuda_peak_allocated_bytes,
            "cuda_peak_reserved_bytes": self.cuda_peak_reserved_bytes,
        }


def _rss_bytes() -> int:
    try:
        import psutil
    except ImportError:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        return int(usage.ru_maxrss) * 1024
    process = psutil.Process(os.getpid())
    return int(process.memory_info().rss)


def benchmark_memory(
    fn: Callable[[], object],
    *,
    iterations: int = 1,
    device: Optional[str] = None,
    sync_cuda: bool = True,
) -> MemoryReport:
    """Run a callable and return process/CUDA memory stats."""

    if iterations <= 0:
        raise ValueError("iterations must be positive")

    use_cuda = device == "cuda" and torch.cuda.is_available()
    if use_cuda and sync_cuda:
        torch.cuda.synchronize()
    if use_cuda:
        torch.cuda.reset_peak_memory_stats()

    before = _rss_bytes()
    with torch.no_grad():
        for _ in range(iterations):
            fn()
    if use_cuda and sync_cuda:
        torch.cuda.synchronize()
    after = _rss_bytes()

    return MemoryReport(
        backend="cuda" if use_cuda else "process_rss",
        before_bytes=before,
        after_bytes=after,
        delta_bytes=after - before,
        peak_bytes=max(before, after),
        cuda_peak_allocated_bytes=(
            int(torch.cuda.max_memory_allocated()) if use_cuda else None
        ),
        cuda_peak_reserved_bytes=(
            int(torch.cuda.max_memory_reserved()) if use_cuda else None
        ),
    )


__all__ = ["MemoryReport", "benchmark_memory"]
