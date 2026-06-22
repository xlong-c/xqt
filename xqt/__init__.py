"""Experimental model compression and deployment toolkit."""

from .core.artifact import ArtifactManifest, ArtifactRecord, MetricRecord
from .core.config import load_xqt_config
from .core.registry import (
    EXPORTER_REGISTRY,
    PASS_REGISTRY,
    RECIPE_REGISTRY,
    XQTRegistry,
    register_exporter,
    register_pass,
    register_recipe,
)
from .core.schema import XQTConfig
from .operator_opt import (
    OperatorOptimizationExecutionPlan,
    OperatorOptimizationExecutionResult,
    OperatorOptimizationReport,
    OperatorOptimizationTargetPlan,
    build_operator_optimization_plan,
    describe_operator_backend_capability,
    execute_operator_optimization_plan,
)
from .pipeline.preflight import preflight_xqt_config
from .pipeline.runner import run_xqt_recipe
from .workflows import (
    OptimizedModelResult,
    OptimizationConfig,
    OptimizationStageConfig,
    OptimizationStageResult,
    StageAcceptanceConfig,
    load_optimization_config,
    optimize_model,
)
from .xdl_adapter import (
    load_checkpoint_into_model,
    xdl_checkpoint_to_xqt_context,
    xdl_setup_to_xqt_context,
)

__all__ = [
    "ArtifactManifest",
    "ArtifactRecord",
    "EXPORTER_REGISTRY",
    "MetricRecord",
    "OperatorOptimizationExecutionPlan",
    "OperatorOptimizationExecutionResult",
    "OperatorOptimizationReport",
    "OperatorOptimizationTargetPlan",
    "OptimizedModelResult",
    "OptimizationConfig",
    "OptimizationStageConfig",
    "OptimizationStageResult",
    "PASS_REGISTRY",
    "RECIPE_REGISTRY",
    "StageAcceptanceConfig",
    "XQTConfig",
    "XQTRegistry",
    "build_operator_optimization_plan",
    "describe_operator_backend_capability",
    "execute_operator_optimization_plan",
    "load_optimization_config",
    "load_xqt_config",
    "load_checkpoint_into_model",
    "preflight_xqt_config",
    "optimize_model",
    "register_exporter",
    "register_pass",
    "register_recipe",
    "run_xqt_recipe",
    "xdl_checkpoint_to_xqt_context",
    "xdl_setup_to_xqt_context",
]
