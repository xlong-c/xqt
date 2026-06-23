"""Compatibility exports for distillation losses now owned by ``xdl.loss``."""

from xdl.loss.distillation_loss import (
    DistillationLossBreakdown,
    distillation_loss,
    feature_distillation_loss,
    kl_divergence_with_temperature,
    relation_distillation_loss,
)


__all__ = [
    "DistillationLossBreakdown",
    "distillation_loss",
    "feature_distillation_loss",
    "kl_divergence_with_temperature",
    "relation_distillation_loss",
]
