"""Structured execution types for XQT quantization passes."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Optional

from torch import nn


class QuantizationNature(str, enum.Enum):
    """Whether a quantization strategy reduces compute FLOPs or only saves memory bandwidth.

    ``TRUE`` : native low-precision tensor core MMA (e.g. W8A8 fp8 mma m16n8k32,
    W8A8 int8 mma m16n8k32). K dimension doubles relative to fp16, per-clock
    math throughput increases -- should measure *compute* speedup.

    ``PSEUDO`` : weight storage in low precision, dequantized to fp16/bf16 before
    compute (e.g. fp8_weight_only, weight_only_int4, W8A16). K dimension stays at
    16 (fp16 mma), per-clock math throughput unchanged -- can measure *memory
    bandwidth* savings but zero compute speedup.

    ``UNKNOWN`` : the backend or strategy has not been classified yet.
    """

    TRUE = "true"
    PSEUDO = "pseudo"
    UNKNOWN = "unknown"


@dataclass
class QuantizationComponentPlan:
    """Resolved quantization plan for one model or submodule component."""

    name: str
    backend: str
    target_path: Optional[str] = None
    method: Optional[str] = None
    strategy: Optional[str] = None
    policy: dict[str, Any] = field(default_factory=dict)
    keep_high_precision: list[str] = field(default_factory=list)
    skip_quantize: list[str] = field(default_factory=list)
    force_quantize: list[str] = field(default_factory=list)
    analysis_only: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "backend": self.backend,
            "target_path": self.target_path,
            "method": self.method,
            "strategy": self.strategy,
            "policy": dict(self.policy),
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
    method: Optional[str] = None
    strategy: Optional[str] = None
    target_path: Optional[str] = None
    quantized_modules: list[str] = field(default_factory=list)
    skipped_modules: list[str] = field(default_factory=list)
    high_precision_modules: list[str] = field(default_factory=list)
    artifacts: dict[str, str] = field(default_factory=dict)
    calibration_samples: Optional[int] = None
    calibration_summary: Optional[dict[str, Any]] = None
    nature: QuantizationNature = QuantizationNature.UNKNOWN
    algorithm_executable: Optional[bool] = None
    method_semantics: Optional[str] = None
    compute_speedup_expected: Optional[float] = None
    fusion_applied: list[str] = field(default_factory=list)
    dequant_nodes_eliminated: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "component_name": self.component_name,
            "backend": self.backend,
            "runtime": self.runtime,
            "method": self.method,
            "strategy": self.strategy,
            "target_path": self.target_path,
            "quantized_modules": list(self.quantized_modules),
            "quantized_module_count": len(self.quantized_modules),
            "skipped_modules": list(self.skipped_modules),
            "high_precision_modules": list(self.high_precision_modules),
            "artifacts": dict(self.artifacts),
            "calibration_samples": self.calibration_samples,
            "calibration_summary": self.calibration_summary,
            "nature": self.nature.value,
            "algorithm_executable": self.algorithm_executable,
            "method_semantics": self.method_semantics,
            "compute_speedup_expected": self.compute_speedup_expected,
            "fusion_applied": list(self.fusion_applied),
            "dequant_nodes_eliminated": self.dequant_nodes_eliminated,
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
    "QuantizationNature",
    "QuantizationReport",
]
