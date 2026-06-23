"""Structured execution types for XQT quantization passes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from torch import nn


@dataclass
class QuantizationComponentPlan:
    """Resolved quantization plan for one model or submodule component."""

    name: str
    backend: str
    target_path: Optional[str] = None
    strategy: Optional[str] = None
    policy: dict[str, Any] = field(default_factory=dict)
    calibration_split: Optional[str] = None
    validation_split: Optional[str] = None
    keep_high_precision: list[str] = field(default_factory=list)
    skip_quantize: list[str] = field(default_factory=list)
    force_quantize: list[str] = field(default_factory=list)
    analysis_only: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "backend": self.backend,
            "target_path": self.target_path,
            "strategy": self.strategy,
            "policy": dict(self.policy),
            "calibration_split": self.calibration_split,
            "validation_split": self.validation_split,
            "keep_high_precision": list(self.keep_high_precision),
            "skip_quantize": list(self.skip_quantize),
            "force_quantize": list(self.force_quantize),
            "analysis_only": self.analysis_only,
        }


@dataclass
class QuantizationExecutionPlan:
    """Quantization pass execution plan."""

    components: list[QuantizationComponentPlan] = field(default_factory=list)
    analysis_enabled: bool = False
    artifact_prefix: str = "quant"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "components": [component.to_dict() for component in self.components],
            "analysis_enabled": self.analysis_enabled,
            "artifact_prefix": self.artifact_prefix,
            "metadata": dict(self.metadata),
        }


@dataclass
class QuantizationReport:
    """Unified backend-agnostic quantization result for one component."""

    component_name: str
    backend: str
    runtime: Optional[str] = None
    strategy: Optional[str] = None
    target_path: Optional[str] = None
    quantized_modules: list[str] = field(default_factory=list)
    skipped_modules: list[str] = field(default_factory=list)
    high_precision_modules: list[str] = field(default_factory=list)
    artifacts: dict[str, str] = field(default_factory=dict)
    calibration_samples: Optional[int] = None
    calibration_summary: Optional[dict[str, Any]] = None
    source_split: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "component_name": self.component_name,
            "backend": self.backend,
            "runtime": self.runtime,
            "strategy": self.strategy,
            "target_path": self.target_path,
            "quantized_modules": list(self.quantized_modules),
            "quantized_module_count": len(self.quantized_modules),
            "skipped_modules": list(self.skipped_modules),
            "high_precision_modules": list(self.high_precision_modules),
            "artifacts": dict(self.artifacts),
            "calibration_samples": self.calibration_samples,
            "calibration_summary": self.calibration_summary,
            "source_split": self.source_split,
            "metadata": dict(self.metadata),
        }


@dataclass
class QuantizationExecutionResult:
    """Executed quantization result across one or more components."""

    model: nn.Module | None
    reports: list[QuantizationReport] = field(default_factory=list)
    artifacts: dict[str, Any] = field(default_factory=dict)


__all__ = [
    "QuantizationComponentPlan",
    "QuantizationExecutionPlan",
    "QuantizationExecutionResult",
    "QuantizationReport",
]
