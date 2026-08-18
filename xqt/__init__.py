"""Experimental model compression and deployment toolkit."""

from __future__ import annotations

import importlib
from typing import Any


_LAZY_EXPORTS = {
    "ArtifactManifest": (".core.artifact", "ArtifactManifest"),
    "ArtifactRecord": (".core.artifact", "ArtifactRecord"),
    "MetricRecord": (".core.artifact", "MetricRecord"),
    "OptimizedModelResult": (".workflows", "OptimizedModelResult"),
    "OptimizationConfig": (".workflows", "OptimizationConfig"),
    "OptimizationStageConfig": (".workflows", "OptimizationStageConfig"),
    "OptimizationStageResult": (".workflows", "OptimizationStageResult"),
    "StageAcceptanceConfig": (".workflows", "StageAcceptanceConfig"),
    "XQTOptimizationSession": (".workflows", "XQTOptimizationSession"),
    "XQTReadinessReport": (".readiness", "XQTReadinessReport"),
    "XQTReadinessScenario": (".readiness", "XQTReadinessScenario"),
    "assess_xqt_readiness": (".readiness", "assess_xqt_readiness"),
    "load_checkpoint_into_model": (".xdl_adapter", "load_checkpoint_into_model"),
    "load_optimization_config": (".workflows", "load_optimization_config"),
    "optimize_model": (".workflows", "optimize_model"),
    "xdl_checkpoint_to_xqt_context": (
        ".xdl_adapter",
        "xdl_checkpoint_to_xqt_context",
    ),
    "xdl_setup_to_xqt_context": (".xdl_adapter", "xdl_setup_to_xqt_context"),
    "convert": (".conversion", "convert"),
    "ConvertResult": (".conversion", "ConvertResult"),
    "FeedForwardPrecisionPolicy": (
        ".conversion",
        "FeedForwardPrecisionPolicy",
    ),
    "MatmulPrecisionSpec": (".conversion", "MatmulPrecisionSpec"),
    "PrecisionPolicy": (".conversion", "PrecisionPolicy"),
}

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
    if name == "nn":
        module = importlib.import_module(".nn", __name__)
        globals()[name] = module
        return module
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module 'xqt' has no attribute {name!r}")
    module_name, attribute = target
    value = getattr(importlib.import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_LAZY_EXPORTS, "nn"})
