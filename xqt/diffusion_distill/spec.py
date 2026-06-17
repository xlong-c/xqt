"""Diffusion distillation data specifications."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional


@dataclass
class DiffusionSpec:
    """Configuration needed to compare teacher and few-step student sampling."""

    teacher_id: str
    student_id: Optional[str] = None
    scheduler: Optional[str] = None
    prediction_type: str = "epsilon"
    teacher_steps: int = 20
    student_steps: int = 4
    guidance_scale: float = 1.0
    latent_shape: tuple[int, ...] = (1, 4, 64, 64)
    seed: int = 0
    params: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.teacher_steps <= 0:
            raise ValueError("teacher_steps must be positive")
        if self.student_steps <= 0:
            raise ValueError("student_steps must be positive")
        if self.student_steps > self.teacher_steps:
            raise ValueError("student_steps must be less than or equal to teacher_steps")
        if self.guidance_scale < 0:
            raise ValueError("guidance_scale must be non-negative")
        if not self.latent_shape or any(dim <= 0 for dim in self.latent_shape):
            raise ValueError("latent_shape must contain positive dimensions")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["latent_shape"] = list(self.latent_shape)
        return data


@dataclass
class PromptRecord:
    """Prompt or condition record used by diffusion distillation."""

    prompt: str
    negative_prompt: Optional[str] = None
    seed: Optional[int] = None
    metadata: dict[str, Any] = field(default_factory=dict)


__all__ = ["DiffusionSpec", "PromptRecord"]
