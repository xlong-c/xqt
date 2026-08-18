"""Structured schema for XQT optimization workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from .schema import BenchmarkConfig, ModelConfig, TaskConfig


@dataclass
class StageAcceptanceConfig:
    """Acceptance thresholds for one model-side stage."""

    min_speedup: Optional[float] = None
    max_mean_abs: Optional[float] = None
    max_max_abs: Optional[float] = None
    max_relative_error: Optional[float] = None
    max_memory_mb: Optional[float] = None
    max_accuracy_drop: Optional[float] = None


@dataclass
class OptimizationStageConfig:
    """One independent optimization stage."""

    name: str
    kind: str
    enabled: bool = True
    compare_to: Optional[str] = None
    from_stage: Optional[str] = None
    save_model: bool = True
    revert_on_reject: bool = False
    params: dict[str, Any] = field(default_factory=dict)
    accept: StageAcceptanceConfig = field(default_factory=StageAcceptanceConfig)


@dataclass
class OptimizationConfig:
    """User-facing config for XQT model optimization."""

    project: dict[str, Any] = field(
        default_factory=lambda: {
            "name": "xqt_optimization",
            "artifact_dir": "artifacts/xqt/optimization",
        }
    )
    model: ModelConfig = field(default_factory=ModelConfig)
    task: TaskConfig = field(default_factory=TaskConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    compression_axes: list[str] = field(default_factory=list)
    hardware: dict[str, Any] = field(default_factory=dict)
    stages: list[OptimizationStageConfig] = field(default_factory=list)
    device: Optional[str] = None


__all__ = [
    "OptimizationConfig",
    "OptimizationStageConfig",
    "StageAcceptanceConfig",
]
