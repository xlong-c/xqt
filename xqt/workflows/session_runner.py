"""Internal workflow-session state and stage orchestration helpers."""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, field
from itertools import count
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Mapping, Optional, cast

from omegaconf import OmegaConf
from torch import nn

from xqt.core.reporting import add_stage_report_to_manifest, build_stage_report
from xqt.core.schema import (
    AnalysisConfig,
    BenchmarkConfig,
    OperatorOptimizationConfig,
    OutputDiffConfig,
    PruneConfig,
    QuantConfig,
)
from xqt.core.serialization import json_safe_value
from xqt.core.types import XQTContext
from xqt.pipeline.model_pass import LoadModelPass
from xqt.pipeline.runner import create_context

from .stage import (
    SessionStage,
    StagePayload,
    StagePersistence,
    TransformLineage,
    payload_can_restore_model,
    payload_capabilities_for_kind,
    payload_kind_for_stage,
    transform_family_for_kind,
    transform_mutates_model,
)
from .stage_provider import StagePayloadBuildContext, resolve_stage_provider

if TYPE_CHECKING:
    from .optimization import (
        OptimizationConfig,
        OptimizationStageConfig,
        OptimizationStageResult,
        OptimizedModelResult,
        StageAcceptanceConfig,
    )


StageRunner = Callable[["OptimizationConfig", "OptimizationStageConfig", XQTContext], None]


@dataclass(frozen=True)
class SessionStageRunners:
    """Bound stage runners injected from the public workflow module."""

    benchmark: StageRunner
    prune: StageRunner
    quant: StageRunner
    operator: StageRunner
    export: StageRunner
    analyze: StageRunner


@dataclass
class _OptimizationRunState:
    """Mutable execution state shared by workflow and session entrypoints."""

    config: "OptimizationConfig"
    context: XQTContext
    stage_results: list["OptimizationStageResult"] = field(default_factory=list)
    model_snapshots: dict[str, Any] = field(default_factory=dict)
    benchmark_results: dict[str, dict[str, Any]] = field(default_factory=dict)
    stages_by_name: dict[str, SessionStage] = field(default_factory=dict)
    stage_order: list[str] = field(default_factory=list)
    baseline_stage: Optional[str] = None
    current_stage: Optional[str] = None
    best_stage: Optional[str] = None
    stage_counter: Any = field(default_factory=lambda: count(1))


def _snapshot_model(model: Any) -> Any:
    return copy.deepcopy(model) if isinstance(model, nn.Module) else model


def _copy_attempt_value(value: Any) -> Any:
    """Copy mutable context metadata without requiring model duplication."""

    try:
        return copy.deepcopy(value)
    except (TypeError, RuntimeError):
        return value


def _snapshot_context_fields(context: XQTContext) -> dict[str, Any]:
    """Snapshot non-model context state before a candidate attempt."""

    return {
        name: _copy_attempt_value(value)
        for name, value in vars(context).items()
        if name != "model"
    }


def _restore_context_fields(
    context: XQTContext,
    fields: Mapping[str, Any],
) -> None:
    """Restore all non-model fields, discarding attempt-local additions."""

    for name in tuple(vars(context)):
        if name != "model" and name not in fields:
            delattr(context, name)
    for name, value in fields.items():
        setattr(context, name, _copy_attempt_value(value))


def _stage_id(state: _OptimizationRunState) -> str:
    return f"stage_{next(state.stage_counter):04d}"


def _register_stage(
    state: _OptimizationRunState,
    *,
    name: str,
    stage_kind: str,
    payload_value: Any,
    snapshot_value: Any = None,
    parent_names: list[str],
    created_by: TransformLineage,
    metrics: dict[str, Any] | None = None,
    artifacts: dict[str, Any] | None = None,
    save_requested: bool = True,
    save_model_snapshot: bool = True,
    compare_baseline: str | None = None,
    summary: str = "",
    payload_metadata: dict[str, Any] | None = None,
) -> SessionStage:
    if name in state.stages_by_name:
        raise ValueError(f"stage name is already registered: {name}")
    if any(parent not in state.stages_by_name for parent in parent_names):
        raise ValueError(f"stage parents must already exist: {parent_names}")
    payload_kind = payload_kind_for_stage(
        stage_kind,
        transform_kind=created_by.kind,
        payload_value=payload_value,
    )
    payload = StagePayload(
        payload_kind=payload_kind,
        value=_snapshot_model(payload_value),
        metadata=dict(payload_metadata or {}),
        capabilities=payload_capabilities_for_kind(payload_kind),
    )
    persistence = StagePersistence(
        requested=save_requested,
        state="materialized" if save_requested else "transient",
    )
    parent_stage_ids = [
        state.stages_by_name[parent].stage_id
        for parent in parent_names
        if parent in state.stages_by_name
    ]
    stage = SessionStage(
        stage_id=_stage_id(state),
        name=name,
        stage_kind=stage_kind,
        payload=payload,
        parent_stage_ids=parent_stage_ids,
        created_by=created_by,
        metrics=dict(metrics or {}),
        artifacts=dict(artifacts or {}),
        persistence=persistence,
        summary=summary,
        compare_baseline=compare_baseline,
    )
    state.stages_by_name[name] = stage
    state.stage_order.append(name)
    if save_model_snapshot:
        model_value = snapshot_value if snapshot_value is not None else payload_value
        state.model_snapshots[name] = _snapshot_model(model_value)
    return stage


def restore_stage_model(state: _OptimizationRunState, stage_name: str) -> Any:
    stage = state.stages_by_name.get(stage_name)
    if stage is None:
        raise ValueError(f"unknown stage: {stage_name}")
    if stage_name in state.model_snapshots:
        return _snapshot_model(state.model_snapshots[stage_name])
    if stage.payload.payload_kind in {"quantized_model", "pruned_model"}:
        payload_value = stage.payload.value
        if getattr(payload_value, "model", None) is not None:
            return _snapshot_model(payload_value.model)
    if payload_can_restore_model(stage.payload) and stage.payload.value is not None:
        return _snapshot_model(stage.payload.value)
    raise ValueError(f"stage does not carry a restorable model payload: {stage_name}")


def create_optimization_state(
    config: "OptimizationConfig",
    *,
    model: Any | None = None,
    example_inputs: Any = None,
    calibration_inputs: Any = None,
) -> _OptimizationRunState:
    """Build one initialized session/workflow state with formal baseline stage."""

    context = create_context(
        config,
        model=model,
        example_inputs=example_inputs,
        calibration_inputs=calibration_inputs,
    )
    if context.model is None and config.model.target:
        LoadModelPass().run(context)
    state = _OptimizationRunState(config=config, context=context)
    baseline_name = "baseline"
    if baseline_name in state.stages_by_name:
        raise ValueError("baseline stage name is reserved")
    _register_stage(
        state,
        name=baseline_name,
        stage_kind="baseline",
        payload_value=context.model,
        parent_names=[],
        created_by=TransformLineage(
            kind="session_init",
            transform="external_input",
            transform_family=transform_family_for_kind("session_init"),
            transform_name="external_input",
        ),
        metrics={},
        artifacts={},
        save_requested=True,
        save_model_snapshot=True,
        summary="Initial model state entering XQT.",
        payload_metadata={"source": "session_init"},
    )
    state.baseline_stage = baseline_name
    state.current_stage = baseline_name
    state.best_stage = baseline_name
    state.model_snapshots["initial"] = _snapshot_model(context.model)
    return state


def _accept_stage(
    stage: "OptimizationStageConfig",
    metrics: Mapping[str, Any],
    *,
    reference_benchmark: Mapping[str, Any] | None = None,
) -> tuple[bool, str]:
    from xqt.auto.acceptance import evaluate_stage_acceptance

    evaluation = evaluate_stage_acceptance(
        stage.accept,
        metrics,
        reference_benchmark=reference_benchmark,
    )
    return evaluation.accepted, evaluation.message


def _new_artifacts(
    before: Mapping[str, Any], after: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        key: value
        for key, value in after.items()
        if key not in before or before[key] != value
    }


def _extract_optimization_capability(value: Any) -> dict[str, Any] | None:
    if isinstance(value, Mapping):
        raw = value.get("optimization_capability")
        if isinstance(raw, Mapping):
            return dict(raw)
        raw_capability = value.get("capability")
        if isinstance(raw_capability, Mapping):
            nested = _extract_optimization_capability(raw_capability)
            if nested is not None:
                return nested
            if "status" in raw_capability and "runtime" in raw_capability:
                return dict(raw_capability)
        for item in value.values():
            nested = _extract_optimization_capability(item)
            if nested is not None:
                return nested
    elif isinstance(value, (list, tuple)):
        for item in value:
            nested = _extract_optimization_capability(item)
            if nested is not None:
                return nested
    return None


def _record_stage_report(
    state: _OptimizationRunState,
    stage: "OptimizationStageConfig",
    result: "OptimizationStageResult",
    *,
    source_stage_name: str,
) -> None:
    capability = _extract_optimization_capability(result.metrics)
    benchmark_config = state.context.benchmark_config
    report = build_stage_report(
        stage_name=result.name,
        stage_kind=result.kind,
        accepted=result.accepted,
        message=result.message,
        metrics=result.metrics,
        artifacts=result.artifacts,
        capability=capability,
        lineage={
            "from_stage": source_stage_name,
            "compare_to": stage.compare_to,
            "baseline_stage": state.baseline_stage,
            "current_stage": state.current_stage,
            "best_stage": state.best_stage,
        },
        metadata={
            "save_model": stage.save_model,
            "revert_on_reject": stage.revert_on_reject,
        },
        device=state.context.device or None,
        shape=state.context.example_inputs,
        warmup=benchmark_config.warmup if benchmark_config is not None else None,
        iterations=benchmark_config.iterations
        if benchmark_config is not None
        else None,
    )
    stage_reports = state.context.metrics.setdefault("stage_reports", {})
    if isinstance(stage_reports, dict):
        stage_reports[result.name] = report.to_dict()
    if state.context.manifest is not None:
        stage_record = f"{result.kind}:{result.name}"
        if stage_record not in state.context.manifest.passes:
            state.context.manifest.passes.append(stage_record)
        add_stage_report_to_manifest(state.context.manifest, report)


def run_optimization_stage(
    state: _OptimizationRunState,
    stage: "OptimizationStageConfig",
    *,
    runners: SessionStageRunners,
) -> "OptimizationStageResult" | None:
    """Execute one stage and register its accepted session-stage payload."""

    if not stage.enabled:
        return None
    source_stage_name = (
        stage.from_stage or state.current_stage or state.baseline_stage or "baseline"
    )
    if source_stage_name not in state.stages_by_name:
        raise ValueError(f"unknown source stage: {source_stage_name}")
    state.context.model = restore_stage_model(state, source_stage_name)
    before_context = _snapshot_context_fields(state.context)
    before_artifacts = dict(state.context.artifacts)
    before_benchmarks = _copy_attempt_value(state.benchmark_results)

    try:
        if stage.kind == "benchmark":
            runners.benchmark(state.config, stage, state.context)
            metrics = dict(state.context.metrics.get("benchmark", {}))
        elif stage.kind == "prune":
            runners.prune(state.config, stage, state.context)
            metrics = dict(state.context.metrics.get("prune", {}))
        elif stage.kind == "quant":
            runners.quant(state.config, stage, state.context)
            metrics = dict(state.context.metrics.get("quant", {}))
        elif stage.kind == "operator":
            runners.operator(state.config, stage, state.context)
            metrics = dict(state.context.metrics.get("operator_optimization", {}))
        elif stage.kind in {"export", "deploy"}:
            runners.export(state.config, stage, state.context)
            metrics = dict(state.context.metrics.get("export", {}))
        elif stage.kind == "analyze":
            runners.analyze(state.config, stage, state.context)
            metrics = dict(state.context.metrics.get("analysis", {}))
        else:
            raise ValueError(f"unsupported stage kind {stage.kind}")
    except Exception:
        _restore_context_fields(state.context, before_context)
        state.context.model = restore_stage_model(state, source_stage_name)
        state.benchmark_results = before_benchmarks
        raise

    reference_benchmark = (
        state.benchmark_results.get(stage.compare_to)
        if stage.compare_to is not None
        else None
    )
    accepted, message = _accept_stage(
        stage,
        metrics,
        reference_benchmark=reference_benchmark,
    )
    new_artifacts = _new_artifacts(before_artifacts, state.context.artifacts)
    if not accepted:
        _restore_context_fields(state.context, before_context)
        state.context.model = restore_stage_model(state, source_stage_name)
        state.benchmark_results = before_benchmarks
    from .optimization import OptimizationStageResult

    result = OptimizationStageResult(
        name=stage.name,
        kind=stage.kind,
        accepted=accepted,
        metrics=metrics,
        artifacts=new_artifacts,
        message=message,
    )
    state.stage_results.append(result)
    if accepted:
        provider_output = resolve_stage_provider(
            StagePayloadBuildContext(
                state=state,
                stage=stage,
                source_stage_name=source_stage_name,
                accepted=accepted,
                message=message,
                metrics=metrics,
                new_artifacts=new_artifacts,
            ),
        )
        _register_stage(
            state,
            name=stage.name,
            stage_kind=provider_output.session_stage_kind,
            payload_value=provider_output.payload_value,
            snapshot_value=state.context.model,
            parent_names=[source_stage_name] if source_stage_name else [],
            created_by=provider_output.lineage,
            metrics=metrics,
            artifacts=new_artifacts,
            save_requested=stage.save_model,
            save_model_snapshot=transform_mutates_model(stage.kind),
            compare_baseline=stage.compare_to or state.baseline_stage,
            summary=f"{stage.kind} stage '{stage.name}'",
            payload_metadata=provider_output.payload_metadata,
        )
        if stage.kind == "benchmark":
            state.benchmark_results[stage.name] = metrics
        if transform_mutates_model(stage.kind):
            state.current_stage = stage.name
            # The default explicit ranking rule is latest accepted model stage.
            state.best_stage = stage.name
    _record_stage_report(
        state,
        stage,
        result,
        source_stage_name=source_stage_name,
    )
    return result


def result_from_state(state: _OptimizationRunState) -> "OptimizedModelResult":
    """Return the current public result view for one workflow/session state."""

    from .optimization import OptimizedModelResult

    return OptimizedModelResult(
        model=state.context.model,
        context=state.context,
        stages=list(state.stage_results),
        session_stages=[state.stages_by_name[name] for name in state.stage_order],
        best_stage=state.best_stage,
        baseline_stage=state.baseline_stage,
        current_stage=state.current_stage,
        models=dict(state.model_snapshots),
    )


def write_workflow_outputs(result: "OptimizedModelResult") -> None:
    """Persist the shared workflow/session result payload."""

    artifact_dir = Path(result.context.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "stages": [json_safe_value(asdict(stage)) for stage in result.stages],
        "session_stages": [stage.to_dict() for stage in result.session_stages],
        "metrics": json_safe_value(result.metrics),
        "artifacts": json_safe_value(result.artifacts),
        "best_stage": result.best_stage,
        "baseline_stage": result.baseline_stage,
        "current_stage": result.current_stage,
    }
    path = artifact_dir / "workflow_result.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    result.context.artifacts["workflow_result"] = path
    if result.context.manifest is not None:
        manifest_path = artifact_dir / "manifest.json"
        result.context.manifest.write_json(manifest_path)
        result.context.artifacts["manifest"] = manifest_path
        result.context.artifacts["workflow_manifest"] = manifest_path


def acceptance_from_mapping(
    accept: Mapping[str, Any] | "StageAcceptanceConfig" | None,
) -> "StageAcceptanceConfig":
    """Normalize one public acceptance mapping into the structured config."""

    from .optimization import StageAcceptanceConfig

    if accept is None:
        return StageAcceptanceConfig()
    if isinstance(accept, StageAcceptanceConfig):
        return accept
    try:
        merged = OmegaConf.merge(
            OmegaConf.structured(StageAcceptanceConfig), dict(accept)
        )
        return cast(StageAcceptanceConfig, OmegaConf.to_object(merged))
    except Exception as exc:
        raise ValueError(f"failed to load stage acceptance config: {exc}") from exc


__all__ = [
    "SessionStageRunners",
    "_OptimizationRunState",
    "acceptance_from_mapping",
    "create_optimization_state",
    "restore_stage_model",
    "result_from_state",
    "run_optimization_stage",
    "write_workflow_outputs",
]
