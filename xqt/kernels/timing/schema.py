"""Unified schema for kernel timing cache and benchmark records across DSL backends."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class TimingCacheKey:
    """Canonical key identifying an operator problem and execution backend."""

    op: str
    backend: str
    arch: str
    dtype: str
    shape: tuple[int, ...]
    extra: str = ""

    def to_key_string(self) -> str:
        shape_str = "x".join(str(s) for s in self.shape)
        extra_part = f"#{self.extra}" if self.extra else ""
        return f"{self.op}@{self.backend}@{self.arch}@{self.dtype}@{shape_str}{extra_part}"

    @property
    def problem_signature(self) -> str:
        """Return canonical problem signature excluding the backend engine."""
        shape_str = "x".join(str(s) for s in self.shape)
        extra_part = f"#{self.extra}" if self.extra else ""
        return f"{self.op}@{self.arch}@{self.dtype}@{shape_str}{extra_part}"

    @classmethod
    def from_key_string(cls, key_str: str) -> "TimingCacheKey":
        extra = ""
        if "#" in key_str:
            key_str, extra = key_str.split("#", 1)
        parts = key_str.split("@")
        if len(parts) != 5:
            raise ValueError(
                f"invalid unified timing cache key string: {key_str!r} "
                f"(expected format: op@backend@arch@dtype@shape[#extra])"
            )
        op, backend, arch, dtype, shape_part = parts
        shape = tuple(int(s) for s in shape_part.split("x")) if shape_part else ()
        return cls(
            op=op,
            backend=backend,
            arch=arch,
            dtype=dtype,
            shape=shape,
            extra=extra,
        )


@dataclass(frozen=True)
class TimingCacheRecord:
    """Detailed benchmark record of a kernel execution schedule and latency."""

    median_latency_us: float
    p95_latency_us: float | None = None
    min_latency_us: float | None = None
    max_latency_us: float | None = None
    memory_allocated_bytes: int | None = None
    tflops: float | None = None
    schedule: dict[str, Any] = field(default_factory=dict)
    verified_correct: bool = True
    max_abs_error: float | None = None
    relative_l2_error: float | None = None
    preset_name: str = "custom"
    notes: str = ""
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "median_latency_us": self.median_latency_us,
            "p95_latency_us": self.p95_latency_us,
            "min_latency_us": self.min_latency_us,
            "max_latency_us": self.max_latency_us,
            "memory_allocated_bytes": self.memory_allocated_bytes,
            "tflops": self.tflops,
            "schedule": dict(self.schedule),
            "verified_correct": self.verified_correct,
            "max_abs_error": self.max_abs_error,
            "relative_l2_error": self.relative_l2_error,
            "preset_name": self.preset_name,
            "notes": self.notes,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TimingCacheRecord":
        return cls(
            median_latency_us=float(data["median_latency_us"]),
            p95_latency_us=(
                float(data["p95_latency_us"])
                if data.get("p95_latency_us") is not None
                else None
            ),
            min_latency_us=(
                float(data["min_latency_us"])
                if data.get("min_latency_us") is not None
                else None
            ),
            max_latency_us=(
                float(data["max_latency_us"])
                if data.get("max_latency_us") is not None
                else None
            ),
            memory_allocated_bytes=(
                int(data["memory_allocated_bytes"])
                if data.get("memory_allocated_bytes") is not None
                else None
            ),
            tflops=(
                float(data["tflops"])
                if data.get("tflops") is not None
                else None
            ),
            schedule=dict(data.get("schedule", {})),
            verified_correct=bool(data.get("verified_correct", True)),
            max_abs_error=(
                float(data["max_abs_error"])
                if data.get("max_abs_error") is not None
                else None
            ),
            relative_l2_error=(
                float(data["relative_l2_error"])
                if data.get("relative_l2_error") is not None
                else None
            ),
            preset_name=str(data.get("preset_name", "custom")),
            notes=str(data.get("notes", "")),
            timestamp=float(data.get("timestamp", time.time())),
        )


@dataclass(frozen=True)
class TimingResolutionResult:
    """Resolution decision comparing multiple eligible backend implementations."""

    problem_signature: str
    fastest_backend: str
    winning_record: TimingCacheRecord
    all_candidates: dict[str, TimingCacheRecord]
    speedup_vs_baseline: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "problem_signature": self.problem_signature,
            "fastest_backend": self.fastest_backend,
            "winning_record": self.winning_record.to_dict(),
            "all_candidates": {
                b: rec.to_dict() for b, rec in self.all_candidates.items()
            },
            "speedup_vs_baseline": self.speedup_vs_baseline,
        }
