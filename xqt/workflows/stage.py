"""Stage and payload protocol types for XQT optimization sessions."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional

from xqt.contracts.runtime import (
    ExportBundlePayload,
    RuntimeArtifactPayload,
    RuntimeHandlePayload,
    RuntimePlanPayload,
    StageReportPayload,
)
from xqt.contracts.quantized import QuantizedModelPayload
from xqt.contracts.pruned import PrunedModelPayload
from xqt.core.serialization import json_safe_value


StageKind = Literal[
    "baseline",
    "quantized",
    "optimized",
    "observed",
    "exported",
    "imported",
    "custom",
]

PayloadKind = Literal[
    "torch_module",
    "pruned_model",
    "quantized_model",
    "runtime_plan",
    "stage_report",
    "export_bundle",
    "runtime_handle",
]

PersistenceState = Literal["transient", "materialized", "persisted"]


_STAGE_KIND_BY_TRANSFORM: dict[str, StageKind] = {
    "benchmark": "observed",
    "prune": "optimized",
    "quant": "quantized",
    "operator": "optimized",
    "export": "exported",
    "deploy": "exported",
    "analyze": "observed",
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
    "pruned_model": {
        "can_evaluate": True,
        "can_export": True,
        "can_quantize": True,
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
    "stage_report": {
        "can_evaluate": False,
        "can_export": False,
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


def stage_kind_for_transform(transform_kind: str) -> StageKind:
    """Map one workflow transform kind to the resulting session stage kind."""

    return _STAGE_KIND_BY_TRANSFORM.get(transform_kind, "custom")


def transform_family_for_kind(transform_kind: str) -> str:
    """Map one workflow transform kind to its internal transform family."""

    return _TRANSFORM_FAMILY_BY_KIND.get(transform_kind, "transform")


def transform_mutates_model(transform_kind: str) -> bool:
    """Whether this workflow transform is expected to produce a new model state."""

    return transform_kind in {"quant", "prune", "operator"}


def payload_kind_for_stage(
    stage_kind: StageKind | str,
    *,
    transform_kind: str | None = None,
    payload_value: Any = None,
) -> PayloadKind:
    """Resolve the canonical payload kind for one session stage.

    A materialized runtime handle is more specific than the enclosing deploy
    stage. Preserve that distinction so its executable capability is not
    downgraded to an export-bundle capability.
    """

    if stage_kind == "exported" and isinstance(payload_value, RuntimeHandlePayload):
        return "runtime_handle"

    if stage_kind == "observed":
        return "stage_report"
    if stage_kind == "exported":
        return "export_bundle"
    if stage_kind == "quantized" and transform_kind == "quant":
        return "quantized_model"
    if stage_kind == "optimized" and transform_kind == "prune":
        return "pruned_model"
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

    payload_kind = (
        payload.payload_kind if isinstance(payload, StagePayload) else payload
    )
    return bool(payload_capabilities_for_kind(payload_kind).get("can_restore_model"))


def _json_safe_stage_value(value: Any) -> Any:
    """Return the shared JSON-safe representation for stage data."""

    return json_safe_value(value)


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
    created_by: TransformLineage = field(
        default_factory=lambda: TransformLineage(
            kind="unknown",
            transform="unknown",
            transform_family="transform",
            transform_name="unknown",
        )
    )
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
            "payload_capability_delta": _json_safe_stage_value(
                self.payload_capability_delta
            ),
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
        key
        for key in source_keys & target_keys
        if _values_differ(source[key], target[key])
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
    "PrunedModelPayload",
    "QuantizedModelPayload",
    "RuntimeArtifactPayload",
    "RuntimeHandlePayload",
    "RuntimePlanPayload",
    "StageReportPayload",
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
    "transform_mutates_model",
    "transform_family_for_kind",
]
