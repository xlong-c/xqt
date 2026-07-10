"""Internal transform-side stage providers for XQT optimization sessions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol

from xqt.contracts import (
    ExportBundlePayload,
    QuantizedModelPayload,
    RuntimeHandlePayload,
    RuntimePlanPayload,
)
from xqt.quant.capability import describe_quant_backend_capability

from .stage_specs import stage_params
from .stage import (
    TransformLineage,
    stage_kind_for_transform,
    transform_family_for_kind,
)


@dataclass(frozen=True)
class StagePayloadBuildContext:
    """Inputs needed to describe one accepted session stage."""

    state: Any
    stage: Any
    source_stage_name: str
    accepted: bool
    message: str
    metrics: Mapping[str, Any]
    new_artifacts: Mapping[str, Any]


@dataclass(frozen=True)
class StageProviderOutput:
    """Provider output consumed by optimization workflow registration."""

    session_stage_kind: str
    lineage: TransformLineage
    payload_value: Any
    payload_metadata: dict[str, Any]


class StageProvider(Protocol):
    """Internal provider role for transform-specific stage descriptions."""

    def build(self, context: StagePayloadBuildContext) -> StageProviderOutput:
        """Build a complete session-stage description."""


def build_stage_lineage(stage: Any) -> TransformLineage:
    """Build structured lineage for a workflow stage config."""

    params = stage_params(stage)
    return TransformLineage(
        kind=stage.kind,
        transform=stage.kind,
        transform_family=transform_family_for_kind(stage.kind),
        transform_name=stage.kind,
        params=params,
        from_stage=stage.from_stage,
        compare_to=stage.compare_to,
    )


def _base_payload_metadata(context: StagePayloadBuildContext) -> dict[str, Any]:
    return {
        "accepted": context.accepted,
        "message": context.message,
        "model_stage_name": context.stage.name,
        "source_model_stage": context.source_stage_name,
    }


def _artifact_paths(context: StagePayloadBuildContext) -> dict[str, str]:
    return {key: str(value) for key, value in context.new_artifacts.items()}


def _first_mapping_list(metrics: Mapping[str, Any], key: str) -> list[dict[str, Any]]:
    value = metrics.get(key, [])
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def _quant_capability(stage: Any, metrics: Mapping[str, Any]) -> dict[str, Any] | None:
    params = stage_params(stage)
    backend = str(metrics.get("backend") or params.get("backend") or "")
    if not backend:
        return None
    policy = params.get("policy")
    method = metrics.get("method") or params.get("method")
    strategy = metrics.get("strategy") or params.get("strategy")
    try:
        return describe_quant_backend_capability(
            backend,
            method=str(method) if method is not None else None,
            strategy=str(strategy) if strategy is not None else None,
            policy=policy if isinstance(policy, Mapping) else None,
        ).to_dict()
    except ValueError:
        return None


class DefaultStageProvider:
    """Provider for model-state transforms that do not need typed artifacts."""

    def build(self, context: StagePayloadBuildContext) -> StageProviderOutput:
        return StageProviderOutput(
            session_stage_kind=stage_kind_for_transform(context.stage.kind),
            lineage=build_stage_lineage(context.stage),
            payload_value=context.state.context.model,
            payload_metadata=_base_payload_metadata(context),
        )


class ModelQuantizerProvider:
    """Provider for quantization stages and quantized model payloads."""

    def build(self, context: StagePayloadBuildContext) -> StageProviderOutput:
        metrics = context.metrics
        params = stage_params(context.stage)
        components = _first_mapping_list(metrics, "components")
        quantized_modules = metrics.get("quantized_modules", [])
        if not isinstance(quantized_modules, list):
            quantized_modules = []
        payload_value = QuantizedModelPayload(
            stage_name=context.stage.name,
            source_model_stage=context.source_stage_name,
            model=context.state.context.model,
            backend=str(metrics.get("backend") or params.get("backend") or "unknown"),
            method=str(metrics.get("method") or params.get("method") or "unknown"),
            strategy=str(metrics.get("strategy") or params.get("strategy") or "unknown"),
            quantized_module_count=int(metrics.get("quantized_module_count") or 0),
            quantized_modules=[str(item) for item in quantized_modules],
            calibration_samples=metrics.get("calibration_samples")
            if isinstance(metrics.get("calibration_samples"), int)
            else None,
            calibration_summary=dict(metrics.get("calibration_summary"))
            if isinstance(metrics.get("calibration_summary"), Mapping)
            else None,
            components=components,
            artifacts=_artifact_paths(context),
            capability=_quant_capability(context.stage, metrics),
        )
        payload_metadata = _base_payload_metadata(context)
        payload_metadata["quantized_model"] = payload_value.to_dict()
        return StageProviderOutput(
            session_stage_kind="quantized",
            lineage=build_stage_lineage(context.stage),
            payload_value=payload_value,
            payload_metadata=payload_metadata,
        )


class OperatorOptimizerProvider:
    """Provider for operator optimization runtime-plan payloads."""

    def build(self, context: StagePayloadBuildContext) -> StageProviderOutput:
        targets = _first_mapping_list(context.metrics, "targets")
        target_engines = [
            str(target.get("engine"))
            for target in targets
            if target.get("engine")
        ]
        payload_value = RuntimePlanPayload(
            stage_name=context.stage.name,
            source_model_stage=context.source_stage_name,
            engine=target_engines[0] if target_engines else "unknown",
            target_count=len(targets),
            targets=targets,
            artifacts=_artifact_paths(context),
        )
        payload_metadata = _base_payload_metadata(context)
        payload_metadata["runtime_plan"] = payload_value.to_dict()
        return StageProviderOutput(
            session_stage_kind="optimized",
            lineage=build_stage_lineage(context.stage),
            payload_value=payload_value,
            payload_metadata=payload_metadata,
        )


class ExportProvider:
    """Provider for export and deploy bundle payloads."""

    def build(self, context: StagePayloadBuildContext) -> StageProviderOutput:
        targets = _first_mapping_list(context.metrics, "targets")
        runtime_handle = context.metrics.get("runtime_handle")
        if isinstance(runtime_handle, Mapping):
            payload_value = RuntimeHandlePayload(
                stage_name=context.stage.name,
                source_model_stage=context.source_stage_name,
                runtime=str(runtime_handle["runtime"]),
                handle_kind=str(runtime_handle["handle_kind"]),
                target_count=int(runtime_handle.get("target_count", 0)),
                targets=_first_mapping_list(runtime_handle, "targets"),
                handle=runtime_handle.get("handle"),
                artifacts={
                    str(key): str(value)
                    for key, value in dict(runtime_handle.get("artifacts", {})).items()
                },
                metadata=dict(runtime_handle.get("metadata", {})),
            )
            payload_metadata = _base_payload_metadata(context)
            payload_metadata["runtime_handle"] = payload_value.to_dict()
            return StageProviderOutput(
                session_stage_kind="exported",
                lineage=build_stage_lineage(context.stage),
                payload_value=payload_value,
                payload_metadata=payload_metadata,
            )
        first_format = "unknown"
        if targets:
            first_format = str(targets[0].get("format", "unknown"))
        payload_value = ExportBundlePayload(
            stage_name=context.stage.name,
            source_model_stage=context.source_stage_name,
            format=first_format,
            target_count=len(targets),
            targets=targets,
            artifacts=_artifact_paths(context),
        )
        payload_metadata = _base_payload_metadata(context)
        payload_metadata["export_bundle"] = payload_value.to_dict()
        return StageProviderOutput(
            session_stage_kind="exported",
            lineage=build_stage_lineage(context.stage),
            payload_value=payload_value,
            payload_metadata=payload_metadata,
        )


_DEFAULT_PROVIDER = DefaultStageProvider()
_STAGE_PROVIDERS: dict[str, StageProvider] = {
    "quant": ModelQuantizerProvider(),
    "operator": OperatorOptimizerProvider(),
    "export": ExportProvider(),
    "deploy": ExportProvider(),
}


def resolve_stage_provider(context: StagePayloadBuildContext) -> StageProviderOutput:
    """Resolve and run the provider for a workflow stage."""

    provider = _STAGE_PROVIDERS.get(context.stage.kind, _DEFAULT_PROVIDER)
    return provider.build(context)


__all__ = [
    "DefaultStageProvider",
    "ExportProvider",
    "ModelQuantizerProvider",
    "OperatorOptimizerProvider",
    "StagePayloadBuildContext",
    "StageProvider",
    "StageProviderOutput",
    "build_stage_lineage",
    "resolve_stage_provider",
]
