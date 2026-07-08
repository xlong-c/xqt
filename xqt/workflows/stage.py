"""Stage and payload protocol types for XQT optimization sessions."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Literal, Optional


StageKind = Literal[
    "baseline",
    "quantized",
    "optimized",
    "exported",
    "imported",
    "custom",
]

PayloadKind = Literal[
    "torch_module",
    "quantized_model",
    "runtime_plan",
    "export_bundle",
    "runtime_handle",
]

PersistenceState = Literal["transient", "materialized", "persisted"]


_STAGE_KIND_BY_TRANSFORM: dict[str, StageKind] = {
    "benchmark": "optimized",
    "prune": "optimized",
    "quant": "quantized",
    "operator": "optimized",
    "export": "exported",
    "deploy": "exported",
    "analyze": "optimized",
}

_TRANSFORM_FAMILY_BY_KIND: dict[str, str] = {
    "session_init": "import",
    "quant": "model_quantizer",
    "operator": "operator_optimizer",
    "export": "export",
    "deploy": "export",
    "prune": "model_transform",
    "benchmark": "evaluation",
    "analyze": "analysis",
}

_PAYLOAD_CAPABILITIES_BY_KIND: dict[PayloadKind, dict[str, bool]] = {
    "torch_module": {
        "can_evaluate": True,
        "can_export": True,
        "can_quantize": True,
        "can_optimize_ops": True,
        "can_restore_model": True,
    },
    "quantized_model": {
        "can_evaluate": True,
        "can_export": True,
        "can_quantize": False,
        "can_optimize_ops": True,
        "can_restore_model": True,
    },
    "runtime_plan": {
        "can_evaluate": False,
        "can_export": True,
        "can_quantize": False,
        "can_optimize_ops": False,
        "can_restore_model": False,
    },
    "export_bundle": {
        "can_evaluate": False,
        "can_export": False,
        "can_quantize": False,
        "can_optimize_ops": False,
        "can_restore_model": False,
    },
    "runtime_handle": {
        "can_evaluate": True,
        "can_export": False,
        "can_quantize": False,
        "can_optimize_ops": False,
        "can_restore_model": False,
    },
}


@dataclass(kw_only=True)
class QuantizedModelPayload:
    """Type-safe payload describing a quantized model-side stage."""

    stage_name: str
    source_model_stage: str
    model: Any
    backend: str
    method: str
    strategy: str
    quantized_module_count: int = 0
    quantized_modules: list[str] = field(default_factory=list)
    calibration_samples: int | None = None
    calibration_summary: dict[str, Any] | None = None
    components: list[dict[str, Any]] = field(default_factory=list)
    artifacts: dict[str, str] = field(default_factory=dict)
    capability: dict[str, Any] | None = None
    artifact_kind: str = field(default="quantized_model", init=False)

    def to_dict(self) -> dict[str, Any]:
        model_type: str | None
        if self.model is None:
            model_type = None
        else:
            model_type = f"{type(self.model).__module__}.{type(self.model).__qualname__}"
        return {
            "artifact_kind": self.artifact_kind,
            "stage_name": self.stage_name,
            "source_model_stage": self.source_model_stage,
            "model_type": model_type,
            "backend": self.backend,
            "method": self.method,
            "strategy": self.strategy,
            "quantized_module_count": self.quantized_module_count,
            "quantized_modules": list(self.quantized_modules),
            "calibration_samples": self.calibration_samples,
            "calibration_summary": _json_safe_stage_value(self.calibration_summary),
            "components": _json_safe_stage_value(self.components),
            "artifacts": dict(self.artifacts),
            "capability": _json_safe_stage_value(self.capability),
        }


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
    """Type-safe payload describing an operator-optimized runtime plan."""

    engine: str
    target_count: int
    targets: list[dict[str, Any]] = field(default_factory=list)

    def __init__(
        self,
        *,
        stage_name: str,
        source_model_stage: str,
        engine: str,
        target_count: int,
        targets: list[dict[str, Any]] | None = None,
        artifacts: dict[str, str] | None = None,
    ) -> None:
        super().__init__(
            artifact_kind="runtime_plan",
            stage_name=stage_name,
            source_model_stage=source_model_stage,
            artifacts=dict(artifacts or {}),
        )
        self.engine = engine
        self.target_count = target_count
        self.targets = [dict(target) for target in (targets or [])]

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.base_dict(),
            "engine": self.engine,
            "target_count": self.target_count,
            "targets": [dict(target) for target in self.targets],
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

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.base_dict(),
            "runtime": self.runtime,
            "handle_kind": self.handle_kind,
            "target_count": self.target_count,
            "targets": [dict(target) for target in self.targets],
            "handle_materialized": self.handle is not None,
            "metadata": _json_safe_stage_value(self.metadata),
        }


@dataclass(kw_only=True)
class ExportBundlePayload(RuntimeArtifactPayload):
    """Type-safe payload describing exported model bundle outputs."""

    format: str
    target_count: int
    targets: list[dict[str, Any]] = field(default_factory=list)

    def __init__(
        self,
        *,
        stage_name: str,
        source_model_stage: str,
        format: str,
        target_count: int,
        targets: list[dict[str, Any]] | None = None,
        artifacts: dict[str, str] | None = None,
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

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.base_dict(),
            "format": self.format,
            "target_count": self.target_count,
            "targets": [dict(target) for target in self.targets],
        }


def stage_kind_for_transform(transform_kind: str) -> StageKind:
    """Map one workflow transform kind to the resulting session stage kind."""

    return _STAGE_KIND_BY_TRANSFORM.get(transform_kind, "custom")


def transform_family_for_kind(transform_kind: str) -> str:
    """Map one workflow transform kind to its internal transform family."""

    return _TRANSFORM_FAMILY_BY_KIND.get(transform_kind, "transform")


def payload_kind_for_stage(
    stage_kind: StageKind | str,
    *,
    transform_kind: str | None = None,
) -> PayloadKind:
    """Resolve the canonical payload kind for one session stage."""

    if stage_kind == "exported":
        return "export_bundle"
    if stage_kind == "quantized" and transform_kind == "quant":
        return "quantized_model"
    if stage_kind == "optimized" and transform_kind == "operator":
        return "runtime_plan"
    return "torch_module"


def payload_capabilities_for_kind(payload_kind: PayloadKind | str) -> dict[str, bool]:
    """Return a copy of the capability contract for one payload kind."""

    if payload_kind not in _PAYLOAD_CAPABILITIES_BY_KIND:
        raise ValueError(f"unknown payload kind: {payload_kind}")
    return dict(_PAYLOAD_CAPABILITIES_BY_KIND[payload_kind])


def payload_can_restore_model(payload: "StagePayload" | PayloadKind | str) -> bool:
    """Whether this payload kind can restore the session model directly."""

    payload_kind = payload.payload_kind if isinstance(payload, StagePayload) else payload
    return bool(payload_capabilities_for_kind(payload_kind).get("can_restore_model"))


def _json_safe_stage_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "to_dict"):
        return _json_safe_stage_value(value.to_dict())
    if is_dataclass(value):
        return _json_safe_stage_value(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_safe_stage_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe_stage_value(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe_stage_value(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return None
    return None


@dataclass
class StagePayload:
    """Core payload carried by one session stage."""

    payload_kind: PayloadKind
    value: Any = None
    ref: str | Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    capabilities: dict[str, Any] = field(default_factory=dict)

    @property
    def materialized(self) -> bool:
        return self.value is not None

    def to_dict(self) -> dict[str, Any]:
        ref: str | None
        if self.ref is None:
            ref = None
        else:
            ref = str(self.ref)
        value = _json_safe_stage_value(self.value)
        return {
            "payload_kind": self.payload_kind,
            "ref": ref,
            "value": value,
            "metadata": _json_safe_stage_value(self.metadata),
            "capabilities": _json_safe_stage_value(self.capabilities),
            "materialized": self.materialized,
        }


@dataclass
class StagePersistence:
    """Persistence policy and storage state for one stage."""

    requested: bool = True
    state: PersistenceState = "transient"
    path: str | Path | None = None

    def to_dict(self) -> dict[str, Any]:
        path: str | None
        if self.path is None:
            path = None
        else:
            path = str(self.path)
        return {
            "requested": self.requested,
            "state": self.state,
            "path": path,
        }


@dataclass
class TransformLineage:
    """Structured metadata describing the transform that produced a stage."""

    kind: str
    transform: str
    transform_family: str
    transform_name: str
    params: dict[str, Any] = field(default_factory=dict)
    from_stage: Optional[str] = None
    compare_to: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "transform": self.transform,
            "transform_family": self.transform_family,
            "transform_name": self.transform_name,
            "params": _json_safe_stage_value(self.params),
            "from_stage": self.from_stage,
            "compare_to": self.compare_to,
        }


@dataclass
class SessionStage:
    """Managed state node inside one optimization session."""

    stage_id: str
    name: str
    stage_kind: StageKind
    payload: StagePayload
    parent_stage_ids: list[str] = field(default_factory=list)
    created_by: TransformLineage = field(default_factory=lambda: TransformLineage(kind="unknown", transform="unknown", transform_family="transform", transform_name="unknown"))
    metrics: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)
    persistence: StagePersistence = field(default_factory=StagePersistence)
    summary: str = ""
    tags: list[str] = field(default_factory=list)
    notes: str = ""
    compare_baseline: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage_id": self.stage_id,
            "name": self.name,
            "stage_kind": self.stage_kind,
            "payload": self.payload.to_dict(),
            "parent_stage_ids": list(self.parent_stage_ids),
            "created_by": self.created_by.to_dict(),
            "metrics": _json_safe_stage_value(self.metrics),
            "artifacts": _json_safe_stage_value(self.artifacts),
            "persistence": self.persistence.to_dict(),
            "summary": self.summary,
            "tags": list(self.tags),
            "notes": self.notes,
            "compare_baseline": self.compare_baseline,
        }


@dataclass
class StageComparison:
    """Structured comparison between two session stages."""

    source_stage: str
    target_stage: str
    source_stage_kind: str
    target_stage_kind: str
    source_payload_kind: str
    target_payload_kind: str
    payload_kind_changed: bool
    payload_capability_delta: dict[str, dict[str, Any]] = field(default_factory=dict)
    metrics_added: list[str] = field(default_factory=list)
    metrics_removed: list[str] = field(default_factory=list)
    metrics_changed: list[str] = field(default_factory=list)
    artifacts_added: list[str] = field(default_factory=list)
    artifacts_removed: list[str] = field(default_factory=list)
    artifacts_changed: list[str] = field(default_factory=list)
    source_can_restore_model: bool = False
    target_can_restore_model: bool = False
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_stage": self.source_stage,
            "target_stage": self.target_stage,
            "source_stage_kind": self.source_stage_kind,
            "target_stage_kind": self.target_stage_kind,
            "source_payload_kind": self.source_payload_kind,
            "target_payload_kind": self.target_payload_kind,
            "payload_kind_changed": self.payload_kind_changed,
            "payload_capability_delta": _json_safe_stage_value(self.payload_capability_delta),
            "metrics_added": list(self.metrics_added),
            "metrics_removed": list(self.metrics_removed),
            "metrics_changed": list(self.metrics_changed),
            "artifacts_added": list(self.artifacts_added),
            "artifacts_removed": list(self.artifacts_removed),
            "artifacts_changed": list(self.artifacts_changed),
            "source_can_restore_model": self.source_can_restore_model,
            "target_can_restore_model": self.target_can_restore_model,
            "summary": self.summary,
        }


def _values_differ(source: Any, target: Any) -> bool:
    source_safe = _json_safe_stage_value(source)
    target_safe = _json_safe_stage_value(target)
    try:
        return source_safe != target_safe
    except Exception:
        return repr(source_safe) != repr(target_safe)


def _mapping_key_delta(
    source: Mapping[str, Any],
    target: Mapping[str, Any],
) -> tuple[list[str], list[str], list[str]]:
    source_keys = set(source)
    target_keys = set(target)
    added = sorted(target_keys - source_keys)
    removed = sorted(source_keys - target_keys)
    changed = sorted(
        key for key in source_keys & target_keys if _values_differ(source[key], target[key])
    )
    return added, removed, changed


def _capability_delta(
    source: Mapping[str, Any],
    target: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    delta: dict[str, dict[str, Any]] = {}
    for key in sorted(set(source) | set(target)):
        source_value = source.get(key)
        target_value = target.get(key)
        if source_value != target_value:
            delta[key] = {"source": source_value, "target": target_value}
    return delta


def compare_session_stages(
    source: SessionStage,
    target: SessionStage,
) -> StageComparison:
    """Build a capability-aware and artifact-aware diff for two stages."""

    metrics_added, metrics_removed, metrics_changed = _mapping_key_delta(
        source.metrics,
        target.metrics,
    )
    artifacts_added, artifacts_removed, artifacts_changed = _mapping_key_delta(
        source.artifacts,
        target.artifacts,
    )
    payload_capability_delta = _capability_delta(
        source.payload.capabilities,
        target.payload.capabilities,
    )
    payload_kind_changed = source.payload.payload_kind != target.payload.payload_kind
    summary = (
        f"{source.name} -> {target.name}: "
        f"payload {source.payload.payload_kind} -> {target.payload.payload_kind}, "
        f"metrics +{len(metrics_added)}/-{len(metrics_removed)}/~{len(metrics_changed)}, "
        f"artifacts +{len(artifacts_added)}/-{len(artifacts_removed)}/~{len(artifacts_changed)}"
    )
    return StageComparison(
        source_stage=source.name,
        target_stage=target.name,
        source_stage_kind=source.stage_kind,
        target_stage_kind=target.stage_kind,
        source_payload_kind=source.payload.payload_kind,
        target_payload_kind=target.payload.payload_kind,
        payload_kind_changed=payload_kind_changed,
        payload_capability_delta=payload_capability_delta,
        metrics_added=metrics_added,
        metrics_removed=metrics_removed,
        metrics_changed=metrics_changed,
        artifacts_added=artifacts_added,
        artifacts_removed=artifacts_removed,
        artifacts_changed=artifacts_changed,
        source_can_restore_model=payload_can_restore_model(source.payload),
        target_can_restore_model=payload_can_restore_model(target.payload),
        summary=summary,
    )


__all__ = [
    "PayloadKind",
    "PersistenceState",
    "ExportBundlePayload",
    "QuantizedModelPayload",
    "RuntimeArtifactPayload",
    "RuntimeHandlePayload",
    "RuntimePlanPayload",
    "StageComparison",
    "SessionStage",
    "TransformLineage",
    "StageKind",
    "StagePayload",
    "StagePersistence",
    "compare_session_stages",
    "payload_can_restore_model",
    "payload_capabilities_for_kind",
    "payload_kind_for_stage",
    "stage_kind_for_transform",
    "transform_family_for_kind",
]
