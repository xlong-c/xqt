"""Loss helpers for few-step diffusion distillation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass
class DiffusionDistillationLoss:
    """Diffusion distillation loss components."""

    total: torch.Tensor
    prediction: torch.Tensor
    consistency: Optional[torch.Tensor] = None

    def to_dict(self) -> dict[str, float]:
        data = {
            "total": float(self.total.detach().cpu().item()),
            "prediction": float(self.prediction.detach().cpu().item()),
        }
        if self.consistency is not None:
            data["consistency"] = float(self.consistency.detach().cpu().item())
        return data


def prediction_target(
    *,
    clean_latent: torch.Tensor,
    noisy_latent: torch.Tensor,
    noise: torch.Tensor,
    alpha: torch.Tensor | float,
    sigma: torch.Tensor | float,
    prediction_type: str = "epsilon",
) -> torch.Tensor:
    """Return epsilon, x0, or v-prediction target for diffusion training."""

    if prediction_type == "epsilon":
        return noise
    if prediction_type == "x0":
        return clean_latent
    if prediction_type == "v_prediction":
        alpha_tensor = torch.as_tensor(alpha, dtype=noisy_latent.dtype, device=noisy_latent.device)
        sigma_tensor = torch.as_tensor(sigma, dtype=noisy_latent.dtype, device=noisy_latent.device)
        while alpha_tensor.ndim < noisy_latent.ndim:
            alpha_tensor = alpha_tensor.unsqueeze(-1)
        while sigma_tensor.ndim < noisy_latent.ndim:
            sigma_tensor = sigma_tensor.unsqueeze(-1)
        return alpha_tensor * noise - sigma_tensor * clean_latent
    raise ValueError("prediction_type must be epsilon, x0, or v_prediction")


def consistency_distillation_loss(
    student_prediction: torch.Tensor,
    teacher_prediction: torch.Tensor,
    *,
    student_consistency: Optional[torch.Tensor] = None,
    teacher_consistency: Optional[torch.Tensor] = None,
    prediction_weight: float = 1.0,
    consistency_weight: float = 0.0,
) -> DiffusionDistillationLoss:
    """LCM/consistency-style latent matching loss."""

    if student_prediction.shape != teacher_prediction.shape:
        raise ValueError("student_prediction and teacher_prediction must have the same shape")
    prediction = F.mse_loss(student_prediction.float(), teacher_prediction.float())

    consistency: Optional[torch.Tensor] = None
    if student_consistency is not None or teacher_consistency is not None:
        if student_consistency is None or teacher_consistency is None:
            raise ValueError("student_consistency and teacher_consistency must be provided together")
        if student_consistency.shape != teacher_consistency.shape:
            raise ValueError("student_consistency and teacher_consistency must have the same shape")
        consistency = F.mse_loss(student_consistency.float(), teacher_consistency.float())

    total = prediction_weight * prediction
    if consistency is not None:
        total = total + consistency_weight * consistency
    return DiffusionDistillationLoss(
        total=total,
        prediction=prediction,
        consistency=consistency,
    )


__all__ = [
    "DiffusionDistillationLoss",
    "consistency_distillation_loss",
    "prediction_target",
]
