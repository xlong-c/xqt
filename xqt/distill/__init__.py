"""Distillation helpers for XQT."""

from .cache import (
    TeacherCacheRecord,
    TeacherOutput,
    TeacherOutputCache,
    batch_identity,
    cache_teacher_outputs,
    dataset_signature,
)
from .hooks import (
    FeatureAlignmentRecord,
    ModuleOutputCapture,
    analyze_feature_alignment,
    capture_module_outputs,
    collect_module_outputs,
)
from .hf_text import (
    HFTextClassificationDataBundle,
    HFTextClassificationDataSpec,
    HFTextClassificationBundle,
    HFTextClassificationSpec,
    build_hf_text_classification_data,
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
    "FeatureAlignmentRecord",
    "HFTextClassificationDataBundle",
    "HFTextClassificationDataSpec",
    "HFTextClassificationBundle",
    "HFTextClassificationSpec",
    "ModuleOutputCapture",
    "analyze_feature_alignment",
    "TeacherCacheRecord",
    "TeacherOutput",
    "TeacherOutputCache",
    "batch_identity",
    "cache_teacher_outputs",
    "dataset_signature",
    "capture_module_outputs",
    "collect_module_outputs",
    "build_hf_text_classification_data",
    "build_hf_text_classification_bundle",
    "build_hf_text_classification_bundle_from_params",
    "distillation_loss",
    "feature_distillation_loss",
    "kl_divergence_with_temperature",
    "relation_distillation_loss",
    "train_hf_text_classification_distillation",
    "train_logit_distillation",
]
