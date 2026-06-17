"""PyTorch distillation losses used by XQT recipes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass
class DistillationLossBreakdown:
    """Loss components returned by the combined distillation helper."""

    total: torch.Tensor
    soft_target: torch.Tensor
    hard_target: Optional[torch.Tensor] = None
    feature: Optional[torch.Tensor] = None
    relation: Optional[torch.Tensor] = None

    def to_dict(self) -> dict[str, float]:
        """Convert scalar components into plain Python numbers."""

        data: dict[str, float] = {
            "total": float(self.total.detach().item()),
            "soft_target": float(self.soft_target.detach().item()),
        }
        if self.hard_target is not None:
            data["hard_target"] = float(self.hard_target.detach().item())
        if self.feature is not None:
            data["feature"] = float(self.feature.detach().item())
        if self.relation is not None:
            data["relation"] = float(self.relation.detach().item())
        return data


def kl_divergence_with_temperature(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    *,
    temperature: float = 2.0,
    reduction: str = "batchmean",
) -> torch.Tensor:
    """Compute softened KL divergence between student and teacher logits."""

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if student_logits.shape != teacher_logits.shape:
        raise ValueError(
            "student_logits and teacher_logits must have the same shape"
        )

    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    return F.kl_div(student_log_probs, teacher_probs, reduction=reduction) * (
        temperature**2
    )


def feature_distillation_loss(
    student_features: torch.Tensor,
    teacher_features: torch.Tensor,
    *,
    normalize: bool = False,
    reduction: str = "mean",
) -> torch.Tensor:
    """Match intermediate features with MSE or normalized cosine-style MSE."""

    if student_features.shape != teacher_features.shape:
        raise ValueError("student_features and teacher_features must have the same shape")

    student = student_features.float().reshape(student_features.shape[0], -1)
    teacher = teacher_features.float().reshape(teacher_features.shape[0], -1)
    if normalize:
        student = F.normalize(student, dim=-1)
        teacher = F.normalize(teacher, dim=-1)
    return F.mse_loss(student, teacher, reduction=reduction)


def relation_distillation_loss(
    student_features: torch.Tensor,
    teacher_features: torch.Tensor,
    *,
    reduction: str = "mean",
) -> torch.Tensor:
    """Match pairwise sample relations inside a batch."""

    if student_features.shape != teacher_features.shape:
        raise ValueError("student_features and teacher_features must have the same shape")

    student = F.normalize(
        student_features.float().reshape(student_features.shape[0], -1), dim=-1
    )
    teacher = F.normalize(
        teacher_features.float().reshape(teacher_features.shape[0], -1), dim=-1
    )
    student_relation = student @ student.T
    teacher_relation = teacher @ teacher.T
    return F.mse_loss(student_relation, teacher_relation, reduction=reduction)


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    *,
    targets: Optional[torch.Tensor] = None,
    temperature: float = 2.0,
    alpha: float = 0.5,
    feature_student: Optional[torch.Tensor] = None,
    feature_teacher: Optional[torch.Tensor] = None,
    feature_weight: float = 0.0,
    relation_student: Optional[torch.Tensor] = None,
    relation_teacher: Optional[torch.Tensor] = None,
    relation_weight: float = 0.0,
) -> DistillationLossBreakdown:
    """Combine logit KD, optional hard labels, and optional feature losses."""

    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")

    soft_target = kl_divergence_with_temperature(
        student_logits,
        teacher_logits,
        temperature=temperature,
    )

    hard_target: Optional[torch.Tensor] = None
    if targets is not None:
        hard_target = F.cross_entropy(student_logits, targets)

    feature_loss: Optional[torch.Tensor] = None
    if feature_student is not None or feature_teacher is not None:
        if feature_student is None or feature_teacher is None:
            raise ValueError("feature_student and feature_teacher must be provided together")
        feature_loss = feature_distillation_loss(feature_student, feature_teacher)

    relation_loss: Optional[torch.Tensor] = None
    if relation_student is not None or relation_teacher is not None:
        if relation_student is None or relation_teacher is None:
            raise ValueError(
                "relation_student and relation_teacher must be provided together"
            )
        relation_loss = relation_distillation_loss(relation_student, relation_teacher)

    total = alpha * soft_target
    if hard_target is not None:
        total = total + (1.0 - alpha) * hard_target
    if feature_loss is not None:
        total = total + feature_weight * feature_loss
    if relation_loss is not None:
        total = total + relation_weight * relation_loss

    return DistillationLossBreakdown(
        total=total,
        soft_target=soft_target,
        hard_target=hard_target,
        feature=feature_loss,
        relation=relation_loss,
    )


__all__ = [
    "DistillationLossBreakdown",
    "distillation_loss",
    "feature_distillation_loss",
    "kl_divergence_with_temperature",
    "relation_distillation_loss",
]
