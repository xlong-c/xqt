"""Experimental model compression and deployment toolkit."""

import importlib
from typing import Any

from .workflows import (
    OptimizedModelResult,
    OptimizationConfig,
    OptimizationStageConfig,
    OptimizationStageResult,
    StageAcceptanceConfig,
    XQTOptimizationSession,
    load_optimization_config,
    optimize_model,
)
from .core.artifact import ArtifactManifest, ArtifactRecord, MetricRecord
from .readiness import (
    XQTReadinessReport,
    XQTReadinessScenario,
    assess_xqt_readiness,
)
from .xdl_adapter import (
    load_checkpoint_into_model,
    xdl_checkpoint_to_xqt_context,
    xdl_setup_to_xqt_context,
)

__all__ = [
    "XQTOptimizationSession",
    "OptimizedModelResult",
    "OptimizationConfig",
    "OptimizationStageConfig",
    "OptimizationStageResult",
    "StageAcceptanceConfig",
    "load_optimization_config",
    "optimize_model",
    "ArtifactManifest",
    "ArtifactRecord",
    "MetricRecord",
    "XQTReadinessReport",
    "XQTReadinessScenario",
    "assess_xqt_readiness",
    "load_checkpoint_into_model",
    "xdl_checkpoint_to_xqt_context",
    "xdl_setup_to_xqt_context",
]


def __getattr__(name: str) -> Any:
    if name == "convert":
        from .conversion import convert

        return convert
    if name in {
        "ConvertResult",
        "FeedForwardPrecisionPolicy",
        "MatmulPrecisionSpec",
        "PrecisionPolicy",
    }:
        from . import conversion

        return getattr(conversion, name)
    if name == "nn":
        return importlib.import_module(".nn", __name__)
    raise AttributeError(f"module 'xqt' has no attribute {name!r}")
