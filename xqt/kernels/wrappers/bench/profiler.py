"""torch.profiler helpers for XQT operator analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import torch


@dataclass
class ProfiledOperatorRecord:
    """Summarized profiler row for one operator or kernel entry."""

    name: str
    cpu_time_total_us: float
    self_cpu_time_total_us: float
    cuda_time_total_us: float
    self_cuda_time_total_us: float
    count: int
    cpu_memory_usage: int
    cuda_memory_usage: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "cpu_time_total_us": self.cpu_time_total_us,
            "self_cpu_time_total_us": self.self_cpu_time_total_us,
            "cuda_time_total_us": self.cuda_time_total_us,
            "self_cuda_time_total_us": self.self_cuda_time_total_us,
            "count": self.count,
            "cpu_memory_usage": self.cpu_memory_usage,
            "cuda_memory_usage": self.cuda_memory_usage,
        }


@dataclass
class ProfilerReport:
    """Stable torch.profiler summary for XQT analysis passes."""

    activities: list[str]
    record_count: int
    kernel_count: int
    operators: list[ProfiledOperatorRecord] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "activities": list(self.activities),
            "record_count": self.record_count,
            "kernel_count": self.kernel_count,
            "operators": [record.to_dict() for record in self.operators],
        }


def profile_callable(
    fn: Callable[[], object],
    *,
    warmup: int = 1,
    active: int = 1,
    repeat: int = 1,
    device: Optional[str] = None,
    top_k: Optional[int] = None,
) -> ProfilerReport:
    """Profile a zero-argument callable with torch.profiler."""

    if warmup < 0:
        raise ValueError("warmup must be non-negative")
    if active <= 0:
        raise ValueError("active must be positive")
    if repeat <= 0:
        raise ValueError("repeat must be positive")

    torch_device = torch.device(device) if device is not None else torch.device("cpu")
    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch_device.type == "cuda" and torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    schedule = torch.profiler.schedule(
        wait=0,
        warmup=warmup,
        active=active,
        repeat=repeat,
    )
    with torch.profiler.profile(
        activities=activities,
        schedule=schedule,
        profile_memory=True,
        record_shapes=True,
        with_stack=False,
    ) as profiler:
        total_steps = (warmup + active) * repeat
        for _ in range(total_steps):
            fn()
            profiler.step()

    events = profiler.key_averages()
    records = [
        ProfiledOperatorRecord(
            name=str(event.key),
            cpu_time_total_us=float(getattr(event, "cpu_time_total", 0.0)),
            self_cpu_time_total_us=float(getattr(event, "self_cpu_time_total", 0.0)),
            cuda_time_total_us=float(getattr(event, "device_time_total", 0.0)),
            self_cuda_time_total_us=float(getattr(event, "self_device_time_total", 0.0)),
            count=int(getattr(event, "count", 0)),
            cpu_memory_usage=int(getattr(event, "cpu_memory_usage", 0)),
            cuda_memory_usage=int(getattr(event, "device_memory_usage", 0)),
        )
        for event in events
    ]
    records.sort(key=lambda record: record.self_cpu_time_total_us + record.self_cuda_time_total_us, reverse=True)
    if top_k is not None:
        records = records[:top_k]
    return ProfilerReport(
        activities=[activity.name.lower() for activity in activities],
        record_count=len(records),
        kernel_count=sum(record.count for record in records),
        operators=records,
    )


__all__ = [
    "ProfiledOperatorRecord",
    "ProfilerReport",
    "profile_callable",
]
