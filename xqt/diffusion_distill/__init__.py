"""Few-step diffusion distillation helpers for XQT."""

from .cache import DiffusionCache, TrajectoryRecord, trajectory_from_schedule
from .losses import (
    DiffusionDistillationLoss,
    consistency_distillation_loss,
    prediction_target,
)
from .report import (
    DiffusionSamplingReport,
    ImageGridRecord,
    build_image_grid_record,
)
from .spec import DiffusionSpec, PromptRecord
from .trajectory import DiffusionStepPair, build_step_schedule, downsample_timesteps

__all__ = [
    "DiffusionCache",
    "DiffusionDistillationLoss",
    "DiffusionSamplingReport",
    "DiffusionSpec",
    "DiffusionStepPair",
    "ImageGridRecord",
    "PromptRecord",
    "TrajectoryRecord",
    "build_step_schedule",
    "downsample_timesteps",
    "build_image_grid_record",
    "consistency_distillation_loss",
    "prediction_target",
    "trajectory_from_schedule",
]
