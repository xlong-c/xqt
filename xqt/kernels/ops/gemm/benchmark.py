"""CUDA event benchmarking primitives for the XQT GEMM module.

The benchmark boundary is deliberately lower level than a model benchmark:
callers provide already materialized tensors and callables, while this module
owns synchronization, warmup/repeat accounting and machine-readable timing.
"""

from __future__ import annotations

import statistics
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import torch

from xqt.core.errors import XQTBackendError


@dataclass(frozen=True, slots=True)
class CudaEventBenchmark:
    """Timing summary for one CUDA callable."""

    label: str
    device: str
    device_name: str
    capability: str
    warmup: int
    repeats: int
    iterations: int
    samples_ms: tuple[float, ...]
    output_shape: tuple[int, ...] | None
    output_dtype: str | None

    @property
    def median_ms(self) -> float:
        return float(statistics.median(self.samples_ms))

    @property
    def mean_ms(self) -> float:
        return float(statistics.fmean(self.samples_ms))

    @property
    def min_ms(self) -> float:
        return float(min(self.samples_ms))

    @property
    def max_ms(self) -> float:
        return float(max(self.samples_ms))

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "device": self.device,
            "device_name": self.device_name,
            "capability": self.capability,
            "warmup": self.warmup,
            "repeats": self.repeats,
            "iterations": self.iterations,
            "samples_ms": list(self.samples_ms),
            "median_ms": self.median_ms,
            "mean_ms": self.mean_ms,
            "min_ms": self.min_ms,
            "max_ms": self.max_ms,
            "output_shape": None if self.output_shape is None else list(self.output_shape),
            "output_dtype": self.output_dtype,
        }


def _validate_counts(warmup: int, repeats: int, iterations: int) -> None:
    for name, value in (("warmup", warmup), ("repeats", repeats), ("iterations", iterations)):
        if isinstance(value, bool) or int(value) != value or int(value) < 1:
            raise ValueError(f"{name} must be a positive integer")


def benchmark_cuda_callable(
    fn: Callable[[], Any],
    *,
    label: str,
    device: torch.device | str,
    warmup: int = 20,
    repeats: int = 15,
    iterations: int = 1,
) -> CudaEventBenchmark:
    """Measure a callable with CUDA events and explicit synchronization.

    ``fn`` must enqueue work on ``device``.  Host-only work is intentionally
    not rejected here, but it will produce a near-zero event duration and is
    therefore not a valid GEMM benchmark.
    """

    _validate_counts(warmup, repeats, iterations)
    if not torch.cuda.is_available():
        raise XQTBackendError("CUDA event benchmark requires torch.cuda.is_available()")
    cuda_device = torch.device(device)
    if cuda_device.type != "cuda":
        raise ValueError(f"CUDA event benchmark requires a CUDA device, got {cuda_device}")
    stream = torch.cuda.current_stream(cuda_device)
    last_output: Any = None
    for _ in range(warmup):
        last_output = fn()
    torch.cuda.synchronize(cuda_device)
    samples: list[float] = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(stream)
        for _ in range(iterations):
            last_output = fn()
        end.record(stream)
        end.synchronize()
        samples.append(float(start.elapsed_time(end)) / float(iterations))
    major, minor = torch.cuda.get_device_capability(cuda_device)
    output_shape: tuple[int, ...] | None = None
    output_dtype: str | None = None
    if isinstance(last_output, torch.Tensor):
        output_shape = tuple(int(item) for item in last_output.shape)
        output_dtype = str(last_output.dtype)
    return CudaEventBenchmark(
        label=label,
        device=str(cuda_device),
        device_name=str(torch.cuda.get_device_name(cuda_device)),
        capability=f"sm_{major}{minor}",
        warmup=int(warmup),
        repeats=int(repeats),
        iterations=int(iterations),
        samples_ms=tuple(samples),
        output_shape=output_shape,
        output_dtype=output_dtype,
    )


def benchmark_cuda_callables(
    callables: Mapping[str, Callable[[], Any]],
    *,
    device: torch.device | str,
    warmup: int = 20,
    repeats: int = 15,
    iterations: int = 1,
) -> dict[str, CudaEventBenchmark]:
    """Benchmark several paths with identical CUDA event parameters."""

    return {
        label: benchmark_cuda_callable(
            fn,
            label=label,
            device=device,
            warmup=warmup,
            repeats=repeats,
            iterations=iterations,
        )
        for label, fn in callables.items()
    }


__all__ = ["CudaEventBenchmark", "benchmark_cuda_callable", "benchmark_cuda_callables"]
