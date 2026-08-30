"""Structured execution types for XQT operator optimization passes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from torch import nn


@dataclass
class OperatorOptimizationTargetPlan:
    """Resolved runtime candidate with an explicit replacement and benchmark scope."""

    name: str
    engine: str
    target_path: Optional[str] = None
    candidate_kind: str = "single_kernel"
    benchmark_target_path: Optional[str] = None
    block_kernel: Optional[str] = None
    block_kernel_engine: Optional[str] = None
    fallback_for: Optional[str] = None
    mode: Optional[str] = None
    fullgraph: bool = False
    dynamic: Optional[bool] = None
    options: dict[str, Any] = field(default_factory=dict)
    patterns: list[str] = field(default_factory=list)
    fallback: str = "eager"
    fallback_policy: str = "prefer_fallback"
    min_speedup: float = 1.01
    validate: dict[str, float] = field(default_factory=dict)
    tilelang: dict[str, Any] = field(default_factory=dict)
    cutile: dict[str, Any] = field(default_factory=dict)
    cutlass: dict[str, Any] = field(default_factory=dict)
    cute_dsl: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "engine": self.engine,
            "target_path": self.target_path,
            "candidate_kind": self.candidate_kind,
            "benchmark_target_path": self.benchmark_target_path,
            "block_kernel": self.block_kernel,
            "block_kernel_engine": self.block_kernel_engine,
            "fallback_for": self.fallback_for,
            "mode": self.mode,
            "fullgraph": self.fullgraph,
            "dynamic": self.dynamic,
            "options": dict(self.options),
            "patterns": list(self.patterns),
            "fallback": self.fallback,
            "fallback_policy": self.fallback_policy,
            "min_speedup": self.min_speedup,
            "validate": dict(self.validate),
            "tilelang": dict(self.tilelang),
            "cutile": dict(self.cutile),
            "cutlass": dict(self.cutlass),
            "cute_dsl": dict(self.cute_dsl),
        }


@dataclass
class OperatorOptimizationExecutionPlan:
    """Operator optimization pass execution plan."""

    targets: list[OperatorOptimizationTargetPlan] = field(default_factory=list)
    default_engine: str = "torch_compile"
    stage: str = "after_compression"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "targets": [target.to_dict() for target in self.targets],
            "default_engine": self.default_engine,
            "stage": self.stage,
            "metadata": dict(self.metadata),
        }


@dataclass
class OperatorOptimizationReport:
    """Unified runtime result measured at the configured block boundary."""

    target_name: str
    module_path: Optional[str]
    engine: str
    runtime: str
    applied: bool
    fallback: str
    fallback_policy: str
    candidate_kind: str = "single_kernel"
    benchmark_target_path: Optional[str] = None
    skip_reason: Optional[str] = None
    compile_time_ms: Optional[float] = None
    latency_before: Optional[dict[str, Any]] = None
    latency_after: Optional[dict[str, Any]] = None
    speedup: Optional[float] = None
    numeric_diff: Optional[dict[str, Any]] = None
    device: Optional[str] = None
    dtype: Optional[str] = None
    shape_signature: Optional[dict[str, Any]] = None
    exportable: bool = True
    artifact_paths: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "target_name": self.target_name,
            "module_path": self.module_path,
            "engine": self.engine,
            "runtime": self.runtime,
            "applied": self.applied,
            "fallback": self.fallback,
            "fallback_policy": self.fallback_policy,
            "candidate_kind": self.candidate_kind,
            "benchmark_target_path": self.benchmark_target_path,
            "skip_reason": self.skip_reason,
            "compile_time_ms": self.compile_time_ms,
            "latency_before": dict(self.latency_before or {}),
            "latency_after": dict(self.latency_after or {}),
            "speedup": self.speedup,
            "numeric_diff": dict(self.numeric_diff or {}),
            "device": self.device,
            "dtype": self.dtype,
            "shape_signature": dict(self.shape_signature or {}),
            "exportable": self.exportable,
            "artifact_paths": dict(self.artifact_paths),
            "metadata": dict(self.metadata),
        }
        module_contract = self.metadata.get("module_contract")
        if isinstance(module_contract, dict):
            payload["module_contract"] = dict(module_contract)
        return payload


@dataclass
class OperatorOptimizationExecutionResult:
    """Executed operator optimization result across one or more targets."""

    model: nn.Module
    reports: list[OperatorOptimizationReport] = field(default_factory=list)
    artifacts: dict[str, Any] = field(default_factory=dict)


__all__ = [
    "OperatorOptimizationExecutionPlan",
    "OperatorOptimizationExecutionResult",
    "OperatorOptimizationReport",
    "OperatorOptimizationTargetPlan",
]
