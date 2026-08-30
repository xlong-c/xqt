"""Smoke-only deterministic diffusion denoiser for model-family recipes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn


class SmokeDiffusionDenoiser(nn.Module):
    """Tiny timestep-conditioned denoiser for synthetic CPU smoke tests."""

    def __init__(
        self,
        *,
        input_dim: int = 8,
        hidden_dim: int = 16,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.time_proj = nn.Linear(hidden_dim, input_dim)
        self.dit_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                for _ in range(num_layers)
            ]
        )
        self.out_proj = nn.Linear(hidden_dim, input_dim)

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        del args, kwargs
        time_embedding = self.time_proj(
            self.time_embed(timestep.reshape(-1, 1).float())
        )
        hidden = x + time_embedding
        for block in self.dit_blocks:
            hidden = block(hidden)
        return self.out_proj(hidden)


def build_smoke_diffusion_denoiser(
    *,
    input_dim: int = 8,
    hidden_dim: int = 16,
    num_layers: int = 2,
) -> SmokeDiffusionDenoiser:
    """Build a deterministic small diffusion smoke model."""

    return SmokeDiffusionDenoiser(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
    )


@dataclass(frozen=True)
class DiffusionSmokeReport:
    """Model-side diffusion smoke report with sampling context metadata."""

    model_family: str = "diffusion"
    sampling_steps: int | None = None
    component_counts: dict[str, int] = field(default_factory=dict)
    export_limitations: tuple[str, ...] = ()
    synthetic: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_family": self.model_family,
            "sampling_steps": self.sampling_steps,
            "component_counts": dict(self.component_counts),
            "export_limitations": list(self.export_limitations),
            "synthetic": self.synthetic,
        }
def diffusion_smoke_report(
    model: nn.Module,
    *,
    sampling_steps: int | None = None,
) -> DiffusionSmokeReport:
    """Build a diffusion smoke report with sampling-step context.

    Sampling steps are model-side metadata only; XQT 不实现采样循环, and the
    report never claims real generation quality.
    """

    counts: dict[str, int] = {}
    for name, _module in model.named_modules():
        lowered = name.lower()
        if any(marker in lowered for marker in ("unet", "dit")):
            key = "dit_or_unet"
        elif "vae" in lowered:
            key = "vae"
        elif "text_encoder" in lowered:
            key = "text_encoder"
        else:
            key = "other"
        counts[key] = counts.get(key, 0) + 1
    return DiffusionSmokeReport(
        sampling_steps=sampling_steps,
        component_counts=counts,
        export_limitations=(
            "diffusion 导出受采样循环与动态 shape 限制, 以组件级导出为准.",
        ),
        synthetic=True,
    )


__all__ = [
    "DiffusionSmokeReport",
    "SmokeDiffusionDenoiser",
    "build_smoke_diffusion_denoiser",
    "diffusion_smoke_report",
]
