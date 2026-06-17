"""Reporting helpers for few-step diffusion distillation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class ImageGridRecord:
    """Metadata for a generated image grid artifact."""

    path: str
    prompts: list[str]
    seed: int
    teacher_steps: int
    student_steps: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DiffusionSamplingReport:
    """Sampling comparison report for teacher and few-step student."""

    teacher_steps: int
    student_steps: int
    scheduler: Optional[str]
    guidance_scale: float
    latency_ms: dict[str, float] = field(default_factory=dict)
    memory_mb: dict[str, float] = field(default_factory=dict)
    image_grid: Optional[ImageGridRecord] = None
    metrics: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        return data


def build_image_grid_record(
    path: str | Path,
    prompts: list[str],
    *,
    seed: int,
    teacher_steps: int,
    student_steps: int,
    metadata: Optional[dict[str, Any]] = None,
) -> ImageGridRecord:
    """Build metadata for a fixed-seed image grid artifact."""

    if teacher_steps <= 0 or student_steps <= 0:
        raise ValueError("teacher_steps and student_steps must be positive")
    if student_steps > teacher_steps:
        raise ValueError("student_steps must be less than or equal to teacher_steps")
    return ImageGridRecord(
        path=str(path),
        prompts=list(prompts),
        seed=seed,
        teacher_steps=teacher_steps,
        student_steps=student_steps,
        metadata=dict(metadata or {}),
    )


__all__ = [
    "DiffusionSamplingReport",
    "ImageGridRecord",
    "build_image_grid_record",
]
