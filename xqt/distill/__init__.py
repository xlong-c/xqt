"""Distillation helpers for XQT."""

from .cache import (
    TeacherCacheRecord,
    TeacherOutput,
    TeacherOutputCache,
    cache_teacher_outputs,
)
from .hooks import ModuleOutputCapture, capture_module_outputs, collect_module_outputs
from .hf_text import (
    HFTextClassificationBundle,
    HFTextClassificationSpec,
    build_hf_text_classification_bundle,
    build_hf_text_classification_bundle_from_params,
    train_hf_text_classification_distillation,
)
from .losses import (
    DistillationLossBreakdown,
    distillation_loss,
    feature_distillation_loss,
    kl_divergence_with_temperature,
    relation_distillation_loss,
)
from .training import DistillationTrainReport, train_logit_distillation

__all__ = [
    "DistillationLossBreakdown",
    "DistillationTrainReport",
    "HFTextClassificationBundle",
    "HFTextClassificationSpec",
    "ModuleOutputCapture",
    "TeacherCacheRecord",
    "TeacherOutput",
    "TeacherOutputCache",
    "cache_teacher_outputs",
    "capture_module_outputs",
    "collect_module_outputs",
    "build_hf_text_classification_bundle",
    "build_hf_text_classification_bundle_from_params",
    "distillation_loss",
    "feature_distillation_loss",
    "kl_divergence_with_temperature",
    "relation_distillation_loss",
    "train_hf_text_classification_distillation",
    "train_logit_distillation",
]
