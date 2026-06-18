"""Prompt, latent, and trajectory cache helpers for diffusion distillation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch

from .spec import DiffusionSpec, PromptRecord
from .trajectory import DiffusionStepPair


@dataclass
class TrajectoryRecord:
    """Cached diffusion trajectory tensors for one prompt."""

    prompt_id: str
    timesteps: list[int]
    latents: list[torch.Tensor]
    metadata: dict[str, Any] = field(default_factory=dict)


class DiffusionCache:
    """Filesystem layout for diffusion prompts, latents, and trajectories."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def prompt_key(self, prompt: PromptRecord) -> str:
        payload = json.dumps(asdict(prompt), sort_keys=True, ensure_ascii=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def condition_key(self, prompt: PromptRecord) -> str:
        payload = {
            "prompt": prompt.prompt,
            "negative_prompt": prompt.negative_prompt,
            "seed": prompt.seed,
            "guidance_scale": prompt.guidance_scale,
            "steps": prompt.steps,
            "width": prompt.width,
            "height": prompt.height,
            "condition_image": prompt.condition_image,
            "condition_mask": prompt.condition_mask,
            "reference_image": prompt.reference_image,
            "latent_cache_key": prompt.latent_cache_key,
            "metadata": dict(prompt.metadata),
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
        ).hexdigest()

    def write_spec(self, spec: DiffusionSpec) -> Path:
        spec.validate()
        path = self.root / "spec.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(spec.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        return path

    def read_spec(self) -> DiffusionSpec:
        payload = json.loads((self.root / "spec.json").read_text(encoding="utf-8"))
        payload["latent_shape"] = tuple(payload["latent_shape"])
        spec = DiffusionSpec(**payload)
        spec.validate()
        return spec

    def write_prompt(self, prompt: PromptRecord) -> Path:
        key = self.prompt_key(prompt)
        path = self.root / "prompts" / f"{key}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(prompt), indent=2, sort_keys=True), encoding="utf-8")
        return path

    def read_prompt(self, key: str) -> PromptRecord:
        payload = json.loads((self.root / "prompts" / f"{key}.json").read_text(encoding="utf-8"))
        return PromptRecord(**payload)

    def write_latent(self, key: str, latent: torch.Tensor) -> Path:
        path = self.root / "latents" / f"{key}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(latent.detach().cpu(), path)
        return path

    def read_latent(self, key: str) -> torch.Tensor:
        return torch.load(self.root / "latents" / f"{key}.pt", map_location="cpu")

    def write_trajectory(self, record: TrajectoryRecord) -> Path:
        if len(record.timesteps) != len(record.latents):
            raise ValueError("timesteps and latents must have the same length")
        path = self.root / "trajectories" / f"{record.prompt_id}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "prompt_id": record.prompt_id,
                "timesteps": list(record.timesteps),
                "latents": [latent.detach().cpu() for latent in record.latents],
                "metadata": dict(record.metadata),
            },
            path,
        )
        return path

    def read_trajectory(self, prompt_id: str) -> TrajectoryRecord:
        payload = torch.load(
            self.root / "trajectories" / f"{prompt_id}.pt",
            map_location="cpu",
        )
        return TrajectoryRecord(
            prompt_id=payload["prompt_id"],
            timesteps=list(payload["timesteps"]),
            latents=list(payload["latents"]),
            metadata=dict(payload.get("metadata", {})),
        )


def trajectory_from_schedule(
    prompt_id: str,
    schedule: list[DiffusionStepPair],
    latents: list[torch.Tensor],
    *,
    metadata: Optional[dict[str, Any]] = None,
) -> TrajectoryRecord:
    """Build a trajectory record from a teacher/student schedule."""

    return TrajectoryRecord(
        prompt_id=prompt_id,
        timesteps=[pair.teacher_step for pair in schedule],
        latents=latents,
        metadata=dict(metadata or {}),
    )


__all__ = [
    "DiffusionCache",
    "TrajectoryRecord",
    "trajectory_from_schedule",
]
