"""Runtime artifact contracts shared by workflows and deployment adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .module import coerce_composite_precision_gemm_spec


def _json_safe_contract_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {
            str(key): _json_safe_contract_value(item) for key, item in value.items()
        }
    if isinstance(value, list):
        return [_json_safe_contract_value(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe_contract_value(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _find_first_mapping(value: Any, key: str) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        raw = value.get(key)
        if isinstance(raw, Mapping):
            return raw
        for item in value.values():
            found = _find_first_mapping(item, key)
            if found is not None:
                return found
    if isinstance(value, (list, tuple)):
        for item in value:
            found = _find_first_mapping(item, key)
            if found is not None:
                return found
    return None


def _resolve_runtime_plan_fields(
    metrics: Mapping[str, Any],
    *,
    module_contract: Mapping[str, Any] | None,
    backend: str,
) -> dict[str, Any] | None:
    direct_requested = metrics.get("requested_mode")
    direct_actual = metrics.get("actual_mode")
    direct_branch_formats = metrics.get("branch_formats")
    if direct_requested is not None or direct_actual is not None:
        payload: dict[str, Any] = {
            "composite_precision": bool(metrics.get("composite_precision", True)),
            "backend": str(metrics.get("backend") or backend),
            "requested_mode": (
                str(direct_requested) if direct_requested is not None else None
            ),
            "actual_mode": str(direct_actual) if direct_actual is not None else None,
            "partition_group_count": metrics.get("partition_group_count"),
            "residual_group_count": metrics.get("residual_group_count"),
            "branch_formats": (
                {
                    str(key): str(value)
                    for key, value in dict(direct_branch_formats).items()
                }
                if isinstance(direct_branch_formats, Mapping)
                else {}
            ),
            "accumulation_dtype": metrics.get("accumulation_dtype"),
            "kernel_count": metrics.get("kernel_count"),
            "workspace_bytes": metrics.get("workspace_bytes"),
            "fallback_reason": metrics.get("fallback_reason"),
        }
        return payload

    resolved_contract = module_contract
    if resolved_contract is None:
        nested_contract = _find_first_mapping(metrics, "module_contract")
        if nested_contract is not None:
            resolved_contract = nested_contract
    if not isinstance(resolved_contract, Mapping):
        return None
    raw_policy = resolved_contract.get("policy")
    if not isinstance(raw_policy, Mapping):
        return None
    spec = coerce_composite_precision_gemm_spec(raw_policy.get("composite_gemm"))
    if spec is None:
        return None
    return spec.resolve_runtime_plan(backend=backend)


@dataclass(kw_only=True)
class RuntimeArtifactPayload:
    """Shared runtime-side payload contract for non-model stage artifacts."""

    artifact_kind: str
    stage_name: str
    source_model_stage: str
    artifacts: dict[str, str] = field(default_factory=dict)

    def base_dict(self) -> dict[str, Any]:
        return {
            "artifact_kind": self.artifact_kind,
            "stage_name": self.stage_name,
            "source_model_stage": self.source_model_stage,
            "artifacts": dict(self.artifacts),
        }


@dataclass(kw_only=True)
class RuntimePlanPayload(RuntimeArtifactPayload):
    """Type-safe payload describing an operator-optimized runtime plan.

    ``required_capabilities`` is the Infer-facing primary constraint.
    ``engine`` is a materialize *result* (or ``"unresolved"``), not a hard
    required_engine schema key on the quant→infer handoff.
    """

    engine: str = "unresolved"
    target_count: int = 0
    targets: list[dict[str, Any]] = field(default_factory=list)
    required_capabilities: list[str] = field(default_factory=list)
    preferred_engines: list[str] = field(default_factory=list)
    backend: str | None = None
    requested_mode: str | None = None
    actual_mode: str | None = None
    partition_group_count: int | None = None
    residual_group_count: int | None = None
    branch_formats: dict[str, str] = field(default_factory=dict)
    accumulation_dtype: str | None = None
    kernel_count: int | None = None
    workspace_bytes: int | None = None
    fallback_reason: str | None = None
    composite_precision: bool = False
    module_contract: dict[str, Any] | None = None

    def __init__(
        self,
        *,
        stage_name: str,
        source_model_stage: str,
        engine: str = "unresolved",
        target_count: int = 0,
        targets: list[dict[str, Any]] | None = None,
        required_capabilities: list[str] | None = None,
        preferred_engines: list[str] | None = None,
        backend: str | None = None,
        requested_mode: str | None = None,
        actual_mode: str | None = None,
        partition_group_count: int | None = None,
        residual_group_count: int | None = None,
        branch_formats: dict[str, str] | None = None,
        accumulation_dtype: str | None = None,
        kernel_count: int | None = None,
        workspace_bytes: int | None = None,
        fallback_reason: str | None = None,
        composite_precision: bool = False,
        artifacts: dict[str, str] | None = None,
        module_contract: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            artifact_kind="runtime_plan",
            stage_name=stage_name,
            source_model_stage=source_model_stage,
            artifacts=dict(artifacts or {}),
        )
        self.engine = str(engine) if engine is not None else "unresolved"
        self.target_count = target_count
        self.targets = [dict(target) for target in (targets or [])]
        self.required_capabilities = [
            str(item) for item in (required_capabilities or []) if str(item)
        ]
        self.preferred_engines = [
            str(item).strip().lower()
            for item in (preferred_engines or [])
            if str(item).strip()
        ]
        self.backend = backend
        self.requested_mode = requested_mode
        self.actual_mode = actual_mode
        self.partition_group_count = partition_group_count
        self.residual_group_count = residual_group_count
        self.branch_formats = {
            str(key): str(value) for key, value in dict(branch_formats or {}).items()
        }
        self.accumulation_dtype = accumulation_dtype
        self.kernel_count = kernel_count
        self.workspace_bytes = workspace_bytes
        self.fallback_reason = fallback_reason
        self.composite_precision = bool(composite_precision)
        self.module_contract = (
            dict(module_contract) if module_contract is not None else None
        )

    @classmethod
    def from_stage_metrics(
        cls,
        *,
        stage_name: str,
        source_model_stage: str,
        metrics: Mapping[str, Any],
        artifacts: Mapping[str, str] | None = None,
        module_contract: Mapping[str, Any] | None = None,
    ) -> "RuntimePlanPayload":
        targets = metrics.get("targets", [])
        if not isinstance(targets, list):
            targets = []
        target_dicts = [dict(item) for item in targets if isinstance(item, Mapping)]
        engines = [
            str(target.get("engine"))
            for target in target_dicts
            if target.get("engine")
        ]
        resolved_contract = (
            dict(module_contract)
            if module_contract is not None
            else (
                dict(raw)
                if isinstance((raw := metrics.get("module_contract")), Mapping)
                else None
            )
        )
        backend_value = metrics.get("backend") or metrics.get("runtime")
        if backend_value is None and engines:
            backend_value = engines[0]
        backend = str(backend_value or "unknown")
        raw_caps = metrics.get("required_capabilities", [])
        if not isinstance(raw_caps, list):
            raw_caps = []
        required_capabilities = [str(item) for item in raw_caps if str(item)]
        raw_pref = metrics.get("preferred_engines", [])
        if not isinstance(raw_pref, list):
            raw_pref = []
        preferred_engines = [
            str(item).strip().lower() for item in raw_pref if str(item).strip()
        ]
        runtime_plan = _resolve_runtime_plan_fields(
            metrics,
            module_contract=resolved_contract,
            backend=backend,
        )
        return cls(
            stage_name=stage_name,
            source_model_stage=source_model_stage,
            engine=engines[0] if engines else "unresolved",
            target_count=len(target_dicts),
            targets=target_dicts,
            required_capabilities=required_capabilities,
            preferred_engines=preferred_engines,
            backend=(str(runtime_plan.get("backend")) if runtime_plan is not None else None),
            requested_mode=(
                str(runtime_plan.get("requested_mode"))
                if runtime_plan is not None and runtime_plan.get("requested_mode") is not None
                else None
            ),
            actual_mode=(
                str(runtime_plan.get("actual_mode"))
                if runtime_plan is not None and runtime_plan.get("actual_mode") is not None
                else None
            ),
            partition_group_count=(
                int(runtime_plan.get("partition_group_count"))
                if runtime_plan is not None and runtime_plan.get("partition_group_count") is not None
                else None
            ),
            residual_group_count=(
                int(runtime_plan.get("residual_group_count"))
                if runtime_plan is not None and runtime_plan.get("residual_group_count") is not None
                else None
            ),
            branch_formats=(
                {
                    str(key): str(value)
                    for key, value in dict(runtime_plan.get("branch_formats", {})).items()
                }
                if runtime_plan is not None
                and isinstance(runtime_plan.get("branch_formats"), Mapping)
                else None
            ),
            accumulation_dtype=(
                str(runtime_plan.get("accumulation_dtype"))
                if runtime_plan is not None and runtime_plan.get("accumulation_dtype") is not None
                else None
            ),
            kernel_count=(
                int(runtime_plan.get("kernel_count"))
                if runtime_plan is not None and runtime_plan.get("kernel_count") is not None
                else None
            ),
            workspace_bytes=(
                int(runtime_plan.get("workspace_bytes"))
                if runtime_plan is not None and runtime_plan.get("workspace_bytes") is not None
                else None
            ),
            fallback_reason=(
                str(runtime_plan.get("fallback_reason"))
                if runtime_plan is not None and runtime_plan.get("fallback_reason") is not None
                else None
            ),
            composite_precision=bool(
                runtime_plan is not None and runtime_plan.get("composite_precision")
            ),
            artifacts={str(key): str(value) for key, value in dict(artifacts or {}).items()},
            module_contract=resolved_contract,
        )

    def to_dict(self) -> dict[str, Any]:
        payload = {
            **self.base_dict(),
            "engine": self.engine,
            "target_count": self.target_count,
            "targets": [dict(target) for target in self.targets],
            "required_capabilities": list(self.required_capabilities),
            "preferred_engines": list(self.preferred_engines),
        }
        if self.composite_precision:
            payload.update(
                {
                    "composite_precision": True,
                    "backend": self.backend,
                    "requested_mode": self.requested_mode,
                    "actual_mode": self.actual_mode,
                    "partition_group_count": self.partition_group_count,
                    "residual_group_count": self.residual_group_count,
                    "branch_formats": dict(self.branch_formats),
                    "accumulation_dtype": self.accumulation_dtype,
                    "kernel_count": self.kernel_count,
                    "workspace_bytes": self.workspace_bytes,
                    "fallback_reason": self.fallback_reason,
                }
            )
        if self.module_contract is not None:
            payload["module_contract"] = _json_safe_contract_value(self.module_contract)
        return payload


@dataclass(kw_only=True)
class StageReportPayload(RuntimeArtifactPayload):
    """Type-safe payload describing one observation-only stage report."""

    report_kind: str
    metrics: dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        *,
        stage_name: str,
        source_model_stage: str,
        report_kind: str,
        metrics: dict[str, Any] | None = None,
        artifacts: dict[str, str] | None = None,
    ) -> None:
        super().__init__(
            artifact_kind="stage_report",
            stage_name=stage_name,
            source_model_stage=source_model_stage,
            artifacts=dict(artifacts or {}),
        )
        self.report_kind = report_kind
        self.metrics = dict(metrics or {})

    @classmethod
    def from_stage_metrics(
        cls,
        *,
        stage_name: str,
        source_model_stage: str,
        report_kind: str,
        metrics: Mapping[str, Any],
        artifacts: Mapping[str, str] | None = None,
    ) -> "StageReportPayload":
        return cls(
            stage_name=stage_name,
            source_model_stage=source_model_stage,
            report_kind=report_kind,
            metrics=dict(metrics),
            artifacts={str(key): str(value) for key, value in dict(artifacts or {}).items()},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.base_dict(),
            "report_kind": self.report_kind,
            "metrics": _json_safe_contract_value(self.metrics),
        }


@dataclass(kw_only=True)
class RuntimeHandlePayload(RuntimeArtifactPayload):
    """Type-safe payload describing a materialized executable runtime handle."""

    runtime: str
    handle_kind: str
    target_count: int = 0
    targets: list[dict[str, Any]] = field(default_factory=list)
    handle: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        *,
        stage_name: str,
        source_model_stage: str,
        runtime: str,
        handle_kind: str,
        target_count: int = 0,
        targets: list[dict[str, Any]] | None = None,
        handle: Any = None,
        artifacts: dict[str, str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            artifact_kind="runtime_handle",
            stage_name=stage_name,
            source_model_stage=source_model_stage,
            artifacts=dict(artifacts or {}),
        )
        self.runtime = runtime
        self.handle_kind = handle_kind
        self.target_count = target_count
        self.targets = [dict(target) for target in (targets or [])]
        self.handle = handle
        self.metadata = dict(metadata or {})

    @classmethod
    def from_stage_metrics(
        cls,
        *,
        stage_name: str,
        source_model_stage: str,
        runtime_handle: Mapping[str, Any],
    ) -> "RuntimeHandlePayload":
        targets = runtime_handle.get("targets", [])
        if not isinstance(targets, list):
            targets = []
        return cls(
            stage_name=stage_name,
            source_model_stage=source_model_stage,
            runtime=str(runtime_handle["runtime"]),
            handle_kind=str(runtime_handle["handle_kind"]),
            target_count=int(runtime_handle.get("target_count", 0)),
            targets=[dict(item) for item in targets if isinstance(item, Mapping)],
            handle=runtime_handle.get("handle"),
            artifacts={
                str(key): str(value)
                for key, value in dict(runtime_handle.get("artifacts", {})).items()
            },
            metadata=dict(runtime_handle.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.base_dict(),
            "runtime": self.runtime,
            "handle_kind": self.handle_kind,
            "target_count": self.target_count,
            "targets": [dict(target) for target in self.targets],
            "handle_materialized": self.handle is not None,
            "metadata": _json_safe_contract_value(self.metadata),
        }


@dataclass(kw_only=True)
class ExportBundlePayload(RuntimeArtifactPayload):
    """Type-safe payload describing exported model bundle outputs."""

    format: str
    target_count: int
    targets: list[dict[str, Any]] = field(default_factory=list)
    module_contract: dict[str, Any] | None = None

    def __init__(
        self,
        *,
        stage_name: str,
        source_model_stage: str,
        format: str,
        target_count: int,
        targets: list[dict[str, Any]] | None = None,
        artifacts: dict[str, str] | None = None,
        module_contract: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            artifact_kind="export_bundle",
            stage_name=stage_name,
            source_model_stage=source_model_stage,
            artifacts=dict(artifacts or {}),
        )
        self.format = format
        self.target_count = target_count
        self.targets = [dict(target) for target in (targets or [])]
        self.module_contract = (
            dict(module_contract) if module_contract is not None else None
        )

    @classmethod
    def from_stage_metrics(
        cls,
        *,
        stage_name: str,
        source_model_stage: str,
        metrics: Mapping[str, Any],
        artifacts: Mapping[str, str] | None = None,
        module_contract: Mapping[str, Any] | None = None,
    ) -> "ExportBundlePayload":
        targets = metrics.get("targets", [])
        if not isinstance(targets, list):
            targets = []
        target_dicts = [dict(item) for item in targets if isinstance(item, Mapping)]
        first_format = "unknown"
        if target_dicts:
            first_format = str(target_dicts[0].get("format", "unknown"))
        resolved_contract = (
            dict(module_contract)
            if module_contract is not None
            else (
                dict(raw)
                if isinstance((raw := metrics.get("module_contract")), Mapping)
                else None
            )
        )
        return cls(
            stage_name=stage_name,
            source_model_stage=source_model_stage,
            format=first_format,
            target_count=len(target_dicts),
            targets=target_dicts,
            artifacts={str(key): str(value) for key, value in dict(artifacts or {}).items()},
            module_contract=resolved_contract,
        )

    def to_dict(self) -> dict[str, Any]:
        payload = {
            **self.base_dict(),
            "format": self.format,
            "target_count": self.target_count,
            "targets": [dict(target) for target in self.targets],
        }
        if self.module_contract is not None:
            payload["module_contract"] = _json_safe_contract_value(self.module_contract)
        return payload


@dataclass(kw_only=True)
class ExecutionPolicyPayload(RuntimeArtifactPayload):
    """Type-safe payload describing one execution policy derived from a quant artifact.

    Carries precision overrides and optional required_capabilities for Infer.
    Does not use required_engine as a primary field.
    """

    policy_kind: str
    runtime: str
    module_count: int
    precision_overrides: list[dict[str, Any]] = field(default_factory=list)
    required_capabilities: list[str] = field(default_factory=list)
    preferred_engines: list[str] = field(default_factory=list)
    compute_config: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    module_contract: dict[str, Any] | None = None

    def __init__(
        self,
        *,
        stage_name: str,
        source_model_stage: str,
        policy_kind: str,
        runtime: str,
        module_count: int,
        precision_overrides: list[dict[str, Any]] | None = None,
        required_capabilities: list[str] | None = None,
        preferred_engines: list[str] | None = None,
        compute_config: dict[str, Any] | None = None,
        artifacts: dict[str, str] | None = None,
        metadata: dict[str, Any] | None = None,
        module_contract: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            artifact_kind="execution_policy",
            stage_name=stage_name,
            source_model_stage=source_model_stage,
            artifacts=dict(artifacts or {}),
        )
        self.policy_kind = policy_kind
        self.runtime = runtime
        self.module_count = int(module_count)
        self.precision_overrides = [
            dict(item) for item in (precision_overrides or [])
        ]
        self.required_capabilities = [
            str(item) for item in (required_capabilities or []) if str(item)
        ]
        self.preferred_engines = [
            str(item).strip().lower()
            for item in (preferred_engines or [])
            if str(item).strip()
        ]
        self.compute_config = (
            dict(compute_config) if isinstance(compute_config, Mapping) else None
        )
        self.metadata = dict(metadata or {})
        self.module_contract = (
            dict(module_contract) if module_contract is not None else None
        )

    def to_dict(self) -> dict[str, Any]:
        payload = {
            **self.base_dict(),
            "policy_kind": self.policy_kind,
            "runtime": self.runtime,
            "module_count": self.module_count,
            "precision_overrides": _json_safe_contract_value(self.precision_overrides),
            "required_capabilities": list(self.required_capabilities),
            "preferred_engines": list(self.preferred_engines),
            "metadata": _json_safe_contract_value(self.metadata),
        }
        if self.compute_config is not None:
            payload["compute_config"] = _json_safe_contract_value(self.compute_config)
        if self.module_contract is not None:
            payload["module_contract"] = _json_safe_contract_value(self.module_contract)
        return payload


__all__ = [
    "ExecutionPolicyPayload",
    "ExportBundlePayload",
    "RuntimeArtifactPayload",
    "RuntimeHandlePayload",
    "RuntimePlanPayload",
    "StageReportPayload",
]
