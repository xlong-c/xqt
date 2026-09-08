"""Internal transform-side stage providers for XQT optimization sessions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol

from xqt.contracts import (
    ExportBundlePayload,
    PrunedModelPayload,
    QuantizedModelPayload,
    RuntimeHandlePayload,
    RuntimePlanPayload,
    StageReportPayload,
)
from xqt.contracts.module import get_module_contract

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

        ...


def build_stage_lineage(stage: Any, *, source_stage_name: str) -> TransformLineage:
    """Build structured lineage for a workflow stage config."""

    params = stage_params(stage)
    return TransformLineage(
        kind=stage.kind,
        transform=stage.kind,
        transform_family=transform_family_for_kind(stage.kind),
        transform_name=stage.kind,
        params=params,
        from_stage=source_stage_name,
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


def _capability_from_metrics_value(value: Any) -> dict[str, Any] | None:
    if isinstance(value, Mapping):
        raw_capability = value.get("capability")
        if isinstance(raw_capability, Mapping):
            return dict(raw_capability)
        raw_optimization = value.get("optimization_capability")
        if isinstance(raw_optimization, Mapping):
            return dict(raw_optimization)
        for item in value.values():
            capability = _capability_from_metrics_value(item)
            if capability is not None:
                return capability
    if isinstance(value, (list, tuple)):
        for item in value:
            capability = _capability_from_metrics_value(item)
            if capability is not None:
                return capability
    return None


def _quant_capability_from_metrics(
    metrics: Mapping[str, Any],
) -> dict[str, Any] | None:
    return _capability_from_metrics_value(metrics)


def _prune_capability_from_metrics(
    metrics: Mapping[str, Any],
) -> dict[str, Any] | None:
    return _capability_from_metrics_value(metrics)


class DefaultStageProvider:
    """Provider for model-state transforms that do not need typed artifacts."""

    def build(self, context: StagePayloadBuildContext) -> StageProviderOutput:
        return StageProviderOutput(
            session_stage_kind=stage_kind_for_transform(context.stage.kind),
            lineage=build_stage_lineage(
                context.stage,
                source_stage_name=context.source_stage_name,
            ),
            payload_value=context.state.context.model,
            payload_metadata=_base_payload_metadata(context),
        )


def _model_module_contract(context: StagePayloadBuildContext) -> dict[str, Any] | None:
    model = getattr(context.state.context, "model", None)
    return get_module_contract(model)


class ModelQuantizerProvider:
    """Provider for quantization stages and quantized model payloads."""

    def build(self, context: StagePayloadBuildContext) -> StageProviderOutput:
        metrics = context.metrics
        payload_value = QuantizedModelPayload.from_stage_metrics(
            stage_name=context.stage.name,
            source_model_stage=context.source_stage_name,
            model=context.state.context.model,
            metrics=metrics,
            params=stage_params(context.stage),
            artifacts=_artifact_paths(context),
            capability=_quant_capability_from_metrics(metrics),
            module_contract=_model_module_contract(context),
        )
        payload_metadata = _base_payload_metadata(context)
        payload_metadata["quantized_model"] = payload_value.to_dict()
        return StageProviderOutput(
            session_stage_kind="quantized",
            lineage=build_stage_lineage(
                context.stage,
                source_stage_name=context.source_stage_name,
            ),
            payload_value=payload_value,
            payload_metadata=payload_metadata,
        )


class ModelPrunerProvider:
    """Provider for pruning stages and pruned model payloads."""

    def build(self, context: StagePayloadBuildContext) -> StageProviderOutput:
        metrics = context.metrics
        payload_value = PrunedModelPayload.from_stage_metrics(
            stage_name=context.stage.name,
            source_model_stage=context.source_stage_name,
            model=context.state.context.model,
            metrics=metrics,
            params=stage_params(context.stage),
            artifacts=_artifact_paths(context),
            capability=_prune_capability_from_metrics(metrics),
            module_contract=_model_module_contract(context),
        )
        payload_metadata = _base_payload_metadata(context)
        payload_metadata["pruned_model"] = payload_value.to_dict()
        return StageProviderOutput(
            session_stage_kind="optimized",
            lineage=build_stage_lineage(
                context.stage,
                source_stage_name=context.source_stage_name,
            ),
            payload_value=payload_value,
            payload_metadata=payload_metadata,
        )


class OperatorOptimizerProvider:
    """Provider for operator optimization runtime-plan payloads."""

    def build(self, context: StagePayloadBuildContext) -> StageProviderOutput:
        payload_value = RuntimePlanPayload.from_stage_metrics(
            stage_name=context.stage.name,
            source_model_stage=context.source_stage_name,
            metrics=context.metrics,
            artifacts=_artifact_paths(context),
            module_contract=_model_module_contract(context),
        )
        payload_metadata = _base_payload_metadata(context)
        payload_metadata["runtime_plan"] = payload_value.to_dict()
        return StageProviderOutput(
            session_stage_kind="optimized",
            lineage=build_stage_lineage(
                context.stage,
                source_stage_name=context.source_stage_name,
            ),
            payload_value=payload_value,
            payload_metadata=payload_metadata,
        )


class ExportProvider:
    """Provider for export and deploy bundle payloads."""

    def build(self, context: StagePayloadBuildContext) -> StageProviderOutput:
        runtime_handle = context.metrics.get("runtime_handle")
        if isinstance(runtime_handle, Mapping):
            payload_value = RuntimeHandlePayload.from_stage_metrics(
                stage_name=context.stage.name,
                source_model_stage=context.source_stage_name,
                runtime_handle=runtime_handle,
            )
            payload_metadata = _base_payload_metadata(context)
            payload_metadata["runtime_handle"] = payload_value.to_dict()
            return StageProviderOutput(
                session_stage_kind="exported",
                lineage=build_stage_lineage(
                    context.stage,
                    source_stage_name=context.source_stage_name,
                ),
                payload_value=payload_value,
                payload_metadata=payload_metadata,
            )
        payload_value = ExportBundlePayload.from_stage_metrics(
            stage_name=context.stage.name,
            source_model_stage=context.source_stage_name,
            metrics=context.metrics,
            artifacts=_artifact_paths(context),
            module_contract=_model_module_contract(context),
        )
        payload_metadata = _base_payload_metadata(context)
        payload_metadata["export_bundle"] = payload_value.to_dict()
        return StageProviderOutput(
            session_stage_kind="exported",
            lineage=build_stage_lineage(
                context.stage,
                source_stage_name=context.source_stage_name,
            ),
            payload_value=payload_value,
            payload_metadata=payload_metadata,
        )


class ObservationProvider:
    """Provider for benchmark/analyze observation-only stage payloads."""

    def build(self, context: StagePayloadBuildContext) -> StageProviderOutput:
        payload_value = StageReportPayload.from_stage_metrics(
            stage_name=context.stage.name,
            source_model_stage=context.source_stage_name,
            report_kind=context.stage.kind,
            metrics=context.metrics,
            artifacts=_artifact_paths(context),
        )
        payload_metadata = _base_payload_metadata(context)
        payload_metadata["stage_report"] = payload_value.to_dict()
        return StageProviderOutput(
            session_stage_kind="observed",
            lineage=build_stage_lineage(
                context.stage,
                source_stage_name=context.source_stage_name,
            ),
            payload_value=payload_value,
            payload_metadata=payload_metadata,
        )


_DEFAULT_PROVIDER = DefaultStageProvider()
_STAGE_PROVIDERS: dict[str, StageProvider] = {
    "benchmark": ObservationProvider(),
    "quant": ModelQuantizerProvider(),
    "prune": ModelPrunerProvider(),
    "operator": OperatorOptimizerProvider(),
    "export": ExportProvider(),
    "deploy": ExportProvider(),
    "analyze": ObservationProvider(),
}


def resolve_stage_provider(context: StagePayloadBuildContext) -> StageProviderOutput:
    """Resolve and run the provider for a workflow stage."""

    provider = _STAGE_PROVIDERS.get(context.stage.kind, _DEFAULT_PROVIDER)
    return provider.build(context)


__all__ = [
    "DefaultStageProvider",
    "ExportProvider",
    "ModelPrunerProvider",
    "ModelQuantizerProvider",
    "OperatorOptimizerProvider",
    "StagePayloadBuildContext",
    "StageProvider",
    "StageProviderOutput",
    "build_stage_lineage",
    "resolve_stage_provider",
]
