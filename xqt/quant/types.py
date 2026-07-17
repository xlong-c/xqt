"""Structured execution types for XQT quantization passes."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from torch import nn
from xqt.contracts.module import (
    CompositePrecisionGemmSpec,
    coerce_composite_precision_gemm_spec,
)


class QuantizationNature(str, enum.Enum):
    """Static classification of a quantization route's requested compute contract.

    This enum describes the route selected at quantization time. It is not proof
    that a particular forward used a native low-precision kernel. Runtime records
    must report the selected engine, operands, and any fallback separately.

    ``TRUE``: the configured compute contract requests native low-precision MMA,
    such as W8A8 INT8 MMA. Hardware, shape, and engine availability still decide
    whether a forward realizes that contract.

    ``PSEUDO``: the route's current XQT implementation executes a dequantized or
    reference floating-point compute path. It can still change memory traffic or
    benefit from fusion; it does not imply a zero measured speedup.

    ``UNKNOWN``: the strategy or backend alone does not determine runtime compute.
    """

    TRUE = "true"
    PSEUDO = "pseudo"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class CompositeQuantBranchArtifact:
    """Branch-level artifact references for one composite-precision quant payload."""

    name: str
    format: str
    weight_format: str | None = None
    scale_format: str | None = None
    weight_artifact: str | None = None
    scale_artifact: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "name": self.name,
            "format": self.format,
        }
        if self.weight_format is not None:
            payload["weight_format"] = self.weight_format
        if self.scale_format is not None:
            payload["scale_format"] = self.scale_format
        if self.weight_artifact is not None:
            payload["weight_artifact"] = self.weight_artifact
        if self.scale_artifact is not None:
            payload["scale_artifact"] = self.scale_artifact
        return payload


@dataclass(frozen=True)
class CompositeQuantizationArtifact:
    """Canonical composite-precision quant artifact for one Linear/GEMM payload."""

    component_name: str
    backend: str
    requested_mode: str
    actual_mode: str
    partition_group_count: int
    residual_group_count: int
    partition_map: list[dict[str, Any]] = field(default_factory=list)
    branch_formats: dict[str, str] = field(default_factory=dict)
    accumulation_dtype: str | None = None
    kernel_count: int | None = None
    workspace_bytes: int | None = None
    fallback_reason: str | None = None
    branches: list[CompositeQuantBranchArtifact] = field(default_factory=list)
    composite_precision: bool = field(default=True, init=False)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "component_name": self.component_name,
            "backend": self.backend,
            "composite_precision": self.composite_precision,
            "requested_mode": self.requested_mode,
            "actual_mode": self.actual_mode,
            "partition_group_count": self.partition_group_count,
            "residual_group_count": self.residual_group_count,
            "partition_map": [dict(item) for item in self.partition_map],
            "branch_formats": dict(self.branch_formats),
            "branches": [branch.to_dict() for branch in self.branches],
        }
        if self.accumulation_dtype is not None:
            payload["accumulation_dtype"] = self.accumulation_dtype
        if self.kernel_count is not None:
            payload["kernel_count"] = self.kernel_count
        if self.workspace_bytes is not None:
            payload["workspace_bytes"] = self.workspace_bytes
        if self.fallback_reason is not None:
            payload["fallback_reason"] = self.fallback_reason
        return payload


def build_composite_quantization_artifact(
    spec: CompositePrecisionGemmSpec,
    *,
    component_name: str,
    backend: str,
    available_modes: Sequence[str] | None = None,
) -> CompositeQuantizationArtifact:
    runtime_plan = spec.resolve_runtime_plan(
        backend=backend,
        available_modes=available_modes,
    )
    branches = [
        CompositeQuantBranchArtifact(
            name=spec.selected_branch.name,
            format=spec.selected_branch.format,
            weight_format=spec.selected_branch.weight_format
            or spec.selected_branch.format,
            scale_format=spec.selected_branch.scale_format,
            weight_artifact=f"{component_name}:{spec.selected_branch.name}:weight",
            scale_artifact=(
                f"{component_name}:{spec.selected_branch.name}:scale"
                if spec.selected_branch.scale_format is not None
                else None
            ),
        ),
        CompositeQuantBranchArtifact(
            name=spec.residual_branch.name,
            format=spec.residual_branch.format,
            weight_format=spec.residual_branch.weight_format
            or spec.residual_branch.format,
            scale_format=spec.residual_branch.scale_format,
            weight_artifact=f"{component_name}:{spec.residual_branch.name}:weight",
            scale_artifact=(
                f"{component_name}:{spec.residual_branch.name}:scale"
                if spec.residual_branch.scale_format is not None
                else None
            ),
        ),
    ]
    return CompositeQuantizationArtifact(
        component_name=component_name,
        backend=backend,
        requested_mode=str(runtime_plan["requested_mode"]),
        actual_mode=str(runtime_plan["actual_mode"]),
        partition_group_count=int(runtime_plan["partition_group_count"]),
        residual_group_count=int(runtime_plan["residual_group_count"]),
        partition_map=[
            dict(item) for item in runtime_plan.get("partition_map", []) or []
        ],
        branch_formats={
            str(key): str(value)
            for key, value in dict(runtime_plan.get("branch_formats", {})).items()
        },
        accumulation_dtype=(
            str(runtime_plan["accumulation_dtype"])
            if runtime_plan.get("accumulation_dtype") is not None
            else None
        ),
        kernel_count=(
            int(runtime_plan["kernel_count"])
            if runtime_plan.get("kernel_count") is not None
            else None
        ),
        workspace_bytes=(
            int(runtime_plan["workspace_bytes"])
            if runtime_plan.get("workspace_bytes") is not None
            else None
        ),
        fallback_reason=(
            str(runtime_plan["fallback_reason"])
            if runtime_plan.get("fallback_reason") is not None
            else None
        ),
        branches=branches,
    )


@dataclass
class QuantizationComponentPlan:
    """Resolved quantization plan for one model or submodule component."""

    name: str
    backend: str
    target_path: Optional[str] = None
    method: Optional[str] = None
    strategy: Optional[str] = None
    compute: Optional[str] = None
    policy: dict[str, Any] = field(default_factory=dict)
    composite_gemm: CompositePrecisionGemmSpec | None = None
    keep_high_precision: list[str] = field(default_factory=list)
    skip_quantize: list[str] = field(default_factory=list)
    force_quantize: list[str] = field(default_factory=list)
    analysis_only: bool = False

    def __post_init__(self) -> None:
        self.composite_gemm = coerce_composite_precision_gemm_spec(self.composite_gemm)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "name": self.name,
            "backend": self.backend,
            "target_path": self.target_path,
            "method": self.method,
            "strategy": self.strategy,
            "compute": self.compute,
            "policy": dict(self.policy),
            "keep_high_precision": list(self.keep_high_precision),
            "skip_quantize": list(self.skip_quantize),
            "force_quantize": list(self.force_quantize),
            "analysis_only": self.analysis_only,
        }
        if self.composite_gemm is not None:
            payload["composite_gemm"] = self.composite_gemm.to_dict()
        return payload


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
    composite_quant_artifact: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
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
        if self.composite_quant_artifact is not None:
            payload["composite_quant_artifact"] = dict(self.composite_quant_artifact)
        module_contract = self.metadata.get("module_contract")
        if isinstance(module_contract, dict):
            payload["module_contract"] = dict(module_contract)
        return payload


@dataclass
class QuantizationExecutionResult:
    """Executed quantization result across one or more components."""

    model: nn.Module | None
    reports: list[QuantizationReport] = field(default_factory=list)
    artifacts: dict[str, Any] = field(default_factory=dict)


__all__ = [
    "CompositeQuantBranchArtifact",
    "CompositeQuantizationArtifact",
    "QuantizationComponentPlan",
    "QuantizationExecutionPlan",
    "QuantizationExecutionResult",
    "QuantizationNature",
    "QuantizationReport",
    "build_composite_quantization_artifact",
]
