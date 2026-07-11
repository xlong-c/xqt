"""Runtime artifact contracts shared by workflows and deployment adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


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
    module_contract: dict[str, Any] | None = None

    def __init__(
        self,
        *,
        stage_name: str,
        source_model_stage: str,
        engine: str,
        target_count: int,
        targets: list[dict[str, Any]] | None = None,
        artifacts: dict[str, str] | None = None,
        module_contract: dict[str, Any] | None = None,
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
        return cls(
            stage_name=stage_name,
            source_model_stage=source_model_stage,
            engine=engines[0] if engines else "unknown",
            target_count=len(target_dicts),
            targets=target_dicts,
            artifacts={str(key): str(value) for key, value in dict(artifacts or {}).items()},
            module_contract=resolved_contract,
        )

    def to_dict(self) -> dict[str, Any]:
        payload = {
            **self.base_dict(),
            "engine": self.engine,
            "target_count": self.target_count,
            "targets": [dict(target) for target in self.targets],
        }
        if self.module_contract is not None:
            payload["module_contract"] = _json_safe_contract_value(self.module_contract)
        return payload


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


__all__ = [
    "ExportBundlePayload",
    "RuntimeArtifactPayload",
    "RuntimeHandlePayload",
    "RuntimePlanPayload",
]
