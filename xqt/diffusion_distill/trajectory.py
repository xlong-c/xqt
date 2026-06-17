"""Schedule helpers for diffusion step distillation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class DiffusionStepPair:
    """A teacher timestep paired with a student step index."""

    teacher_step: int
    student_step: int


def _validate_steps(teacher_steps: int, student_steps: int) -> None:
    if teacher_steps <= 0:
        raise ValueError("teacher_steps must be positive")
    if student_steps <= 0:
        raise ValueError("student_steps must be positive")
    if student_steps > teacher_steps:
        raise ValueError("student_steps must be less than or equal to teacher_steps")


def downsample_timesteps(teacher_steps: int, student_steps: int) -> list[int]:
    """Select evenly spaced teacher timesteps for a smaller student schedule."""

    _validate_steps(teacher_steps, student_steps)
    if student_steps == 1:
        return [teacher_steps - 1]

    denominator = student_steps - 1
    return [
        ((student_steps - 1 - index) * (teacher_steps - 1)) // denominator
        for index in range(student_steps)
    ]


def build_step_schedule(teacher_steps: int, student_steps: int) -> list[DiffusionStepPair]:
    """Build a monotonic teacher-to-student timestep schedule."""

    teacher_timesteps = downsample_timesteps(teacher_steps, student_steps)
    return [
        DiffusionStepPair(teacher_step=teacher_step, student_step=index)
        for index, teacher_step in enumerate(teacher_timesteps)
    ]


__all__ = ["DiffusionStepPair", "build_step_schedule", "downsample_timesteps"]
