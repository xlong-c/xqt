"""Stage-based model optimization workflow for XQT."""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, field, is_dataclass
from itertools import count
from pathlib import Path
from typing import Any, Mapping, Optional, cast

from omegaconf import OmegaConf
from torch import nn

from xdl.config.resolver import register_default_resolvers
from xqt.core.config import ConfigInput, load_xqt_config
from xqt.core.reporting import add_stage_report_to_manifest, build_stage_report
from xqt.core.schema import (
    BenchmarkConfig,
    ModelConfig,
    PruneConfig,
    TaskConfig,
)
from xqt.core.types import XQTContext
from xqt.pipeline.passes import (
    AnalyzePass,
    BenchmarkPass,
    ExportPass,
    LoadModelPass,
    OperatorOptimizationPass,
    PrunePass,
    QuantPass,
)
from xqt.pipeline.runner import create_context
from xqt.readiness import XQTReadinessReport, assess_xqt_readiness
from .stage import (
    SessionStage,
    StageComparison,
    StagePayload,
    StagePersistence,
    TransformLineage,
    compare_session_stages,
    payload_can_restore_model,
    payload_capabilities_for_kind,
    payload_kind_for_stage,
    transform_family_for_kind,
)
from .stage_provider import StagePayloadBuildContext, resolve_stage_provider


STAGE_KINDS = {
    "benchmark",
    "prune",
    "quant",
    "operator",
    "export",
    "deploy",
    "analyze",
}


@dataclass
class StageAcceptanceConfig:
    """Acceptance thresholds for one model-side stage."""

    min_speedup: Optional[float] = None
    max_mean_abs: Optional[float] = None
    max_max_abs: Optional[float] = None


@dataclass
class OptimizationStageConfig:
    """One independent optimization stage."""

    name: str
    kind: str
    enabled: bool = True
    compare_to: Optional[str] = None
    from_stage: Optional[str] = None
    save_model: bool = True
    revert_on_reject: bool = False
    params: dict[str, Any] = field(default_factory=dict)
    accept: StageAcceptanceConfig = field(default_factory=StageAcceptanceConfig)


@dataclass
class OptimizationConfig:
    """User-facing config for XQT model optimization."""

    project: dict[str, Any] = field(
        default_factory=lambda: {
            "name": "xqt_optimization",
            "artifact_dir": "artifacts/xqt/optimization",
        }
    )
    model: ModelConfig = field(default_factory=ModelConfig)
    task: TaskConfig = field(default_factory=TaskConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    compression_axes: list[str] = field(default_factory=list)
    hardware: dict[str, Any] = field(default_factory=dict)
    stages: list[OptimizationStageConfig] = field(default_factory=list)
    device: Optional[str] = None


@dataclass
class OptimizationStageResult:
    """Runtime result for one stage."""

    name: str
    kind: str
    accepted: bool
    metrics: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)
    message: str = "ok"


@dataclass
class OptimizedModelResult:
    """Final result returned by ``optimize_model``."""

    model: Any
    context: XQTContext
    stages: list[OptimizationStageResult]
    session_stages: list[SessionStage] = field(default_factory=list)
    best_stage: Optional[str] = None
    baseline_stage: Optional[str] = None
    models: dict[str, Any] = field(default_factory=dict)

    @property
    def metrics(self) -> dict[str, Any]:
        return self.context.metrics

    @property
    def artifacts(self) -> dict[str, Any]:
        return self.context.artifacts

    @property
    def best_model(self) -> Any:
        if self.best_stage is None:
            return self.model
        return self.models.get(self.best_stage, self.model)


@dataclass
class _OptimizationRunState:
    """Mutable execution state shared by workflow and session entrypoints."""

    config: OptimizationConfig
    context: XQTContext
    stage_results: list[OptimizationStageResult] = field(default_factory=list)
    model_snapshots: dict[str, Any] = field(default_factory=dict)
    benchmark_results: dict[str, dict[str, Any]] = field(default_factory=dict)
    stages_by_name: dict[str, SessionStage] = field(default_factory=dict)
    stage_order: list[str] = field(default_factory=list)
    baseline_stage: Optional[str] = None
    best_stage: Optional[str] = None
    stage_counter: Any = field(default_factory=lambda: count(1))


def load_optimization_config(config: ConfigInput | OptimizationConfig) -> OptimizationConfig:
    """Load a stage workflow config using OmegaConf structured defaults."""

    if is_dataclass(config) and isinstance(config, OptimizationConfig):
        return config

    register_default_resolvers()
    raw = OmegaConf.load(config) if isinstance(config, (str, Path)) else OmegaConf.create(config)
    try:
        merged = OmegaConf.merge(OmegaConf.structured(OptimizationConfig), raw)
        OmegaConf.resolve(merged)
        return cast(OptimizationConfig, OmegaConf.to_object(merged))
    except Exception as exc:
        raise ValueError(f"failed to load XQT optimization config: {exc}") from exc


def _project(config: OptimizationConfig, *, stage_name: str | None = None) -> dict[str, Any]:
    project = dict(config.project)
    if stage_name is not None:
        artifact_dir = Path(str(project.get("artifact_dir", "artifacts/xqt/optimization")))
        project["artifact_dir"] = str(artifact_dir / stage_name)
    return project


def _model_config(config: OptimizationConfig) -> dict[str, Any]:
    model = asdict(config.model) if is_dataclass(config.model) else dict(config.model)
    if config.device is not None:
        model["device"] = config.device
    return model


def _task_config(config: OptimizationConfig) -> dict[str, Any]:
    return asdict(config.task) if is_dataclass(config.task) else dict(config.task)


def _base_xqt_config(
    config: OptimizationConfig,
    *,
    stage_name: str | None = None,
) -> Any:
    return load_xqt_config(
        {
            "project": _project(config, stage_name=stage_name),
            "model": _model_config(config),
            "task": _task_config(config),
            "benchmark": asdict(config.benchmark) if is_dataclass(config.benchmark) else dict(config.benchmark),
        }
    )


def _snapshot_model(model: Any) -> Any:
    return copy.deepcopy(model) if isinstance(model, nn.Module) else model


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
    compare_baseline: str | None = None,
    summary: str = "",
    payload_metadata: dict[str, Any] | None = None,
) -> SessionStage:
    payload_kind = payload_kind_for_stage(
        stage_kind,
        transform_kind=created_by.kind,
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
    parent_stage_ids = [state.stages_by_name[parent].stage_id for parent in parent_names if parent in state.stages_by_name]
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
    if save_requested:
        model_value = snapshot_value if snapshot_value is not None else payload_value
        state.model_snapshots[name] = _snapshot_model(model_value)
    return stage


def _restore_stage_model(state: _OptimizationRunState, stage_name: str) -> Any:
    stage = state.stages_by_name.get(stage_name)
    if stage is None:
        raise ValueError(f"unknown stage: {stage_name}")
    if stage_name in state.model_snapshots:
        return _snapshot_model(state.model_snapshots[stage_name])
    if stage.payload.payload_kind == "quantized_model":
        payload_value = stage.payload.value
        if getattr(payload_value, "model", None) is not None:
            return _snapshot_model(payload_value.model)
    if payload_can_restore_model(stage.payload) and stage.payload.value is not None:
        return _snapshot_model(stage.payload.value)
    raise ValueError(f"stage does not carry a restorable model payload: {stage_name}")


def _create_optimization_state(
    config: OptimizationConfig,
    *,
    model: Any | None = None,
    example_inputs: Any = None,
    calibration_inputs: Any = None,
) -> _OptimizationRunState:
    context = create_context(
        _base_xqt_config(config),
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
        summary="Initial model state entering XQT.",
        payload_metadata={"source": "session_init"},
    )
    state.baseline_stage = baseline_name
    state.best_stage = baseline_name
    state.model_snapshots["initial"] = _snapshot_model(context.model)
    return state


def _stage_context(
    config: OptimizationConfig,
    stage: OptimizationStageConfig,
    *,
    context: XQTContext,
) -> XQTContext:
    stage_config = _base_xqt_config(config, stage_name=stage.name)
    context.config = stage_config
    context.device = stage_config.model.device
    return context


def _max_nested_numeric(value: Any, key: str) -> Optional[float]:
    values: list[float] = []
    if isinstance(value, Mapping):
        raw = value.get(key)
        if isinstance(raw, (float, int)):
            values.append(float(raw))
        for item in value.values():
            nested = _max_nested_numeric(item, key)
            if nested is not None:
                values.append(nested)
    elif isinstance(value, (list, tuple)):
        for item in value:
            nested = _max_nested_numeric(item, key)
            if nested is not None:
                values.append(nested)
    return max(values) if values else None


def _benchmark_speedup(reference: Mapping[str, Any] | None, metrics: Mapping[str, Any]) -> Optional[float]:
    if reference is None:
        return None
    reference_latency = _max_nested_numeric(reference, "p50_ms")
    current_latency = _max_nested_numeric(metrics, "p50_ms")
    if reference_latency is None or current_latency is None or current_latency <= 0:
        return None
    return reference_latency / current_latency


def _accept_stage(
    stage: OptimizationStageConfig,
    metrics: Mapping[str, Any],
    *,
    reference_benchmark: Mapping[str, Any] | None = None,
) -> tuple[bool, str]:
    accept = stage.accept
    accepted = True

    if accept.min_speedup is not None:
        speedup = _benchmark_speedup(reference_benchmark, metrics)
        if speedup is None:
            speedup = _max_nested_numeric(metrics, "speedup")
        if speedup is None or speedup < accept.min_speedup:
            accepted = False

    if accept.max_mean_abs is not None:
        mean_abs = _max_nested_numeric(metrics, "mean_abs")
        if mean_abs is None or mean_abs > accept.max_mean_abs:
            accepted = False

    if accept.max_max_abs is not None:
        max_abs = _max_nested_numeric(metrics, "max_abs")
        if max_abs is None or max_abs > accept.max_max_abs:
            accepted = False

    return accepted, "ok" if accepted else "rejected by acceptance thresholds"


def _benchmark_config(params: Mapping[str, Any]) -> BenchmarkConfig:
    try:
        merged = OmegaConf.merge(OmegaConf.structured(BenchmarkConfig), dict(params))
        return cast(BenchmarkConfig, OmegaConf.to_object(merged))
    except Exception as exc:
        raise ValueError(f"failed to load benchmark stage params: {exc}") from exc


def _new_artifacts(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
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
    stage: OptimizationStageConfig,
    result: OptimizationStageResult,
) -> None:
    capability = _extract_optimization_capability(result.metrics)
    report = build_stage_report(
        stage_name=result.name,
        stage_kind=result.kind,
        accepted=result.accepted,
        message=result.message,
        metrics=result.metrics,
        artifacts=result.artifacts,
        capability=capability,
        lineage={
            "from_stage": stage.from_stage,
            "compare_to": stage.compare_to,
            "baseline_stage": state.baseline_stage,
            "best_stage": state.best_stage,
        },
        metadata={
            "save_model": stage.save_model,
            "revert_on_reject": stage.revert_on_reject,
        },
    )
    stage_reports = state.context.metrics.setdefault("stage_reports", {})
    if isinstance(stage_reports, dict):
        stage_reports[result.name] = report.to_dict()
    if state.context.manifest is not None:
        stage_record = f"{result.kind}:{result.name}"
        if stage_record not in state.context.manifest.passes:
            state.context.manifest.passes.append(stage_record)
        add_stage_report_to_manifest(state.context.manifest, report)


def _run_prune(
    config: OptimizationConfig,
    stage: OptimizationStageConfig,
    context: XQTContext,
) -> None:
    _stage_context(config, stage, context=context)
    params = dict(stage.params)
    params.pop("enabled", None)
    context.config.compression.prune = PruneConfig(enabled=True, **params)
    PrunePass().run(context)


def _run_quant(
    config: OptimizationConfig,
    stage: OptimizationStageConfig,
    context: XQTContext,
) -> None:
    _stage_context(config, stage, context=context)
    params = dict(stage.params)
    params.pop("enabled", None)
    context.config.compression.quant = load_xqt_config(
        {"compression": {"quant": {"enabled": True, **params}}}
    ).compression.quant
    QuantPass().run(context)


def _run_operator(
    config: OptimizationConfig,
    stage: OptimizationStageConfig,
    context: XQTContext,
) -> None:
    _stage_context(config, stage, context=context)
    params = dict(stage.params)
    params.pop("enabled", None)
    benchmark_params = params.pop("benchmark", None)
    if isinstance(benchmark_params, Mapping):
        context.config.benchmark = _benchmark_config(benchmark_params)
    context.config.operator_optimization = load_xqt_config(
        {"operator_optimization": {"enabled": True, **params}}
    ).operator_optimization
    OperatorOptimizationPass().run(context)


def _run_export(
    config: OptimizationConfig,
    stage: OptimizationStageConfig,
    context: XQTContext,
) -> None:
    _stage_context(config, stage, context=context)
    params = dict(stage.params)
    params.pop("enabled", None)
    targets = params.pop("targets", [])
    context.config.export = load_xqt_config({"export": {"targets": targets}}).export
    ExportPass().run(context)


def _run_analyze(
    config: OptimizationConfig,
    stage: OptimizationStageConfig,
    context: XQTContext,
) -> None:
    _stage_context(config, stage, context=context)
    params = dict(stage.params)
    params.pop("enabled", None)
    context.config.analysis = load_xqt_config(
        {"analysis": {"enabled": True, **params}}
    ).analysis
    AnalyzePass().run(context)


def _run_benchmark(
    config: OptimizationConfig,
    stage: OptimizationStageConfig,
    context: XQTContext,
) -> None:
    _stage_context(config, stage, context=context)
    context.config.benchmark = _benchmark_config(stage.params)
    BenchmarkPass().run(context)


def _run_optimization_stage(
    state: _OptimizationRunState,
    stage: OptimizationStageConfig,
) -> OptimizationStageResult | None:
    if not stage.enabled:
        return None
    if stage.kind not in STAGE_KINDS:
        allowed = ", ".join(sorted(STAGE_KINDS))
        raise ValueError(f"unsupported stage kind {stage.kind}. Allowed: {allowed}")
    if stage.from_stage is not None:
        if stage.from_stage not in state.model_snapshots and stage.from_stage not in state.stages_by_name:
            raise ValueError(f"unknown from_stage: {stage.from_stage}")
        state.context.model = _restore_stage_model(state, stage.from_stage)

    before_artifacts = dict(state.context.artifacts)
    if stage.kind == "benchmark":
        _run_benchmark(state.config, stage, state.context)
        metrics = dict(state.context.metrics.get("benchmark", {}))
        state.benchmark_results[stage.name] = metrics
    elif stage.kind == "prune":
        _run_prune(state.config, stage, state.context)
        metrics = dict(state.context.metrics.get("prune", {}))
    elif stage.kind == "quant":
        _run_quant(state.config, stage, state.context)
        metrics = dict(state.context.metrics.get("quant", {}))
    elif stage.kind == "operator":
        _run_operator(state.config, stage, state.context)
        metrics = dict(state.context.metrics.get("operator_optimization", {}))
    elif stage.kind in {"export", "deploy"}:
        _run_export(state.config, stage, state.context)
        metrics = dict(state.context.metrics.get("export", {}))
    elif stage.kind == "analyze":
        _run_analyze(state.config, stage, state.context)
        metrics = dict(state.context.metrics.get("analysis", {}))
    else:
        raise ValueError(f"unsupported stage kind {stage.kind}")

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
    source_stage_name = stage.from_stage or state.best_stage or state.baseline_stage or "baseline"
    if accepted and stage.save_model:
        state.model_snapshots[stage.name] = _snapshot_model(state.context.model)
        state.best_stage = stage.name
    elif not accepted and stage.revert_on_reject and stage.from_stage is not None:
        state.context.model = _restore_stage_model(state, stage.from_stage)

    new_artifacts = _new_artifacts(before_artifacts, state.context.artifacts)
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
            compare_baseline=stage.compare_to or state.baseline_stage,
            summary=f"{stage.kind} stage '{stage.name}'",
            payload_metadata=provider_output.payload_metadata,
        )
    _record_stage_report(state, stage, result)
    return result


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    return value


def _write_workflow_outputs(result: OptimizedModelResult) -> None:
    artifact_dir = Path(result.context.config.project.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "stages": [_json_safe(asdict(stage)) for stage in result.stages],
        "session_stages": [stage.to_dict() for stage in result.session_stages],
        "metrics": _json_safe(result.metrics),
        "artifacts": _json_safe(result.artifacts),
        "best_stage": result.best_stage,
        "baseline_stage": result.baseline_stage,
    }
    path = artifact_dir / "workflow_result.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    result.context.artifacts["workflow_result"] = path


def _result_from_state(state: _OptimizationRunState) -> OptimizedModelResult:
    return OptimizedModelResult(
        model=state.context.model,
        context=state.context,
        stages=list(state.stage_results),
        session_stages=[state.stages_by_name[name] for name in state.stage_order],
        best_stage=state.best_stage,
        baseline_stage=state.baseline_stage,
        models=dict(state.model_snapshots),
    )


def optimize_model(
    config: ConfigInput | OptimizationConfig,
    *,
    model: Any | None = None,
    example_inputs: Any = None,
    calibration_inputs: Any = None,
    write_outputs: bool = True,
) -> OptimizedModelResult:
    """Run the configured XQT model optimization stages."""

    loaded = load_optimization_config(config)
    state = _create_optimization_state(
        loaded,
        model=model,
        example_inputs=example_inputs,
        calibration_inputs=calibration_inputs,
    )
    for stage in loaded.stages:
        _run_optimization_stage(state, stage)
    result = _result_from_state(state)
    if write_outputs:
        _write_workflow_outputs(result)
    return result


def _acceptance_from_mapping(
    accept: Mapping[str, Any] | StageAcceptanceConfig | None,
) -> StageAcceptanceConfig:
    if accept is None:
        return StageAcceptanceConfig()
    if isinstance(accept, StageAcceptanceConfig):
        return accept
    try:
        merged = OmegaConf.merge(OmegaConf.structured(StageAcceptanceConfig), dict(accept))
        return cast(StageAcceptanceConfig, OmegaConf.to_object(merged))
    except Exception as exc:
        raise ValueError(f"failed to load stage acceptance config: {exc}") from exc


class XQTOptimizationSession:
    """Pythonic step-by-step interface for XQT stage workflows."""

    def __init__(
        self,
        config: ConfigInput | OptimizationConfig | None = None,
        *,
        model: Any | None = None,
        example_inputs: Any = None,
        calibration_inputs: Any = None,
        project: Mapping[str, Any] | None = None,
        model_config: Mapping[str, Any] | ModelConfig | None = None,
        task: Mapping[str, Any] | TaskConfig | None = None,
        device: str | None = None,
    ) -> None:
        if config is None:
            raw_config: dict[str, Any] = {"stages": []}
            if project is not None:
                raw_config["project"] = dict(project)
            if model_config is not None:
                raw_config["model"] = (
                    asdict(model_config)
                    if is_dataclass(model_config)
                    else dict(model_config)
                )
            if task is not None:
                raw_config["task"] = asdict(task) if is_dataclass(task) else dict(task)
            if device is not None:
                raw_config["device"] = device
            loaded = load_optimization_config(raw_config)
        else:
            loaded = load_optimization_config(config)
            if loaded.stages:
                loaded = copy.deepcopy(loaded)
                loaded.stages = []
        self._state = _create_optimization_state(
            loaded,
            model=model,
            example_inputs=example_inputs,
            calibration_inputs=calibration_inputs,
        )
        self._outputs_written = False

    @property
    def config(self) -> OptimizationConfig:
        return self._state.config

    @property
    def context(self) -> XQTContext:
        return self._state.context

    @property
    def model(self) -> Any:
        return self._state.context.model

    @property
    def stages(self) -> list[OptimizationStageResult]:
        return list(self._state.stage_results)

    @property
    def session_stages(self) -> list[SessionStage]:
        return [self._state.stages_by_name[name] for name in self._state.stage_order]

    @property
    def baseline_stage(self) -> Optional[str]:
        return self._state.baseline_stage

    @property
    def best_stage(self) -> Optional[str]:
        return self._state.best_stage

    def compare_stages(self, source_stage: str, target_stage: str) -> StageComparison:
        """Compare two managed session stages without mutating session state."""

        if source_stage not in self._state.stages_by_name:
            raise ValueError(f"unknown source stage: {source_stage}")
        if target_stage not in self._state.stages_by_name:
            raise ValueError(f"unknown target stage: {target_stage}")
        return compare_session_stages(
            self._state.stages_by_name[source_stage],
            self._state.stages_by_name[target_stage],
        )

    def compare_to_baseline(self, stage_name: str) -> StageComparison:
        """Compare one managed stage against the formal baseline stage."""

        if self._state.baseline_stage is None:
            raise ValueError("baseline stage is not initialized")
        return self.compare_stages(self._state.baseline_stage, stage_name)

    def set_example_inputs(self, example_inputs: Any) -> None:
        self._state.context.example_inputs = example_inputs

    def set_calibration_inputs(self, calibration_inputs: Any) -> None:
        self._state.context.calibration_inputs = calibration_inputs

    def readiness(
        self,
        *,
        name: str = "readiness",
        write_artifacts: bool = True,
        run_tilelang_probe: bool = False,
        tilelang_compile_only: bool = True,
        tilelang_target_arch: str | None = "sm_80",
        tilelang_warmup: int = 1,
        tilelang_iterations: int = 2,
        tensorrt_plugin_libraries: list[str | Path] | None = None,
        validate_tensorrt_plugin_loadability: bool = False,
        trtexec_path: str = "trtexec",
    ) -> XQTReadinessReport:
        """Assess XQT scenario readiness and attach the report to this session."""

        report = assess_xqt_readiness(
            run_tilelang_probe=run_tilelang_probe,
            tilelang_compile_only=tilelang_compile_only,
            tilelang_target_arch=tilelang_target_arch,
            tilelang_warmup=tilelang_warmup,
            tilelang_iterations=tilelang_iterations,
            tensorrt_plugin_libraries=tensorrt_plugin_libraries,
            validate_tensorrt_plugin_loadability=validate_tensorrt_plugin_loadability,
            trtexec_path=trtexec_path,
        )
        artifact_paths: dict[str, Path] = {}
        if write_artifacts:
            artifact_dir = Path(
                str(
                    self._state.config.project.get(
                        "artifact_dir",
                        "artifacts/xqt/optimization",
                    )
                )
            )
            artifact_paths = report.write_artifacts(
                artifact_dir / name,
                stem=name,
            )
            self._state.context.artifacts[f"{name}_json"] = artifact_paths["json"]
            self._state.context.artifacts[f"{name}_markdown"] = artifact_paths["markdown"]
        self._state.context.metrics[name] = report.to_dict()
        if self._state.context.manifest is not None:
            report.add_to_manifest(
                self._state.context.manifest,
                artifact_paths=artifact_paths,
            )
            if write_artifacts:
                manifest_path = Path(
                    str(
                        self._state.config.project.get(
                            "artifact_dir",
                            "artifacts/xqt/optimization",
                        )
                    )
                ) / "manifest.json"
                self._state.context.manifest.write_json(manifest_path)
                self._state.context.artifacts["manifest"] = manifest_path
        self._outputs_written = False
        return report

    def revert_to(self, stage_name: str) -> None:
        if stage_name not in self._state.model_snapshots and stage_name not in self._state.stages_by_name:
            raise ValueError(f"unknown model snapshot: {stage_name}")
        self._state.context.model = _restore_stage_model(self._state, stage_name)

    def use(self, stage_name: str) -> None:
        self.revert_to(stage_name)

    def run_stage(self, stage: OptimizationStageConfig) -> OptimizationStageResult:
        self._validate_stage(stage)
        self._state.config.stages.append(stage)
        result = _run_optimization_stage(self._state, stage)
        if result is None:
            raise ValueError(f"stage is disabled: {stage.name}")
        self._outputs_written = False
        return result

    def benchmark(
        self,
        *,
        name: str,
        compare_to: str | None = None,
        accept: Mapping[str, Any] | StageAcceptanceConfig | None = None,
        save_model: bool = True,
        **params: Any,
    ) -> OptimizationStageResult:
        return self._run(
            name=name,
            kind="benchmark",
            compare_to=compare_to,
            save_model=save_model,
            params=params,
            accept=accept,
        )

    def prune(
        self,
        *,
        name: str,
        method: str | None = None,
        target_sparsity: float | None = None,
        granularity: str | None = None,
        from_stage: str | None = None,
        compare_to: str | None = None,
        accept: Mapping[str, Any] | StageAcceptanceConfig | None = None,
        save_model: bool = True,
        revert_on_reject: bool = False,
        **params: Any,
    ) -> OptimizationStageResult:
        if method is not None:
            params["method"] = method
        if target_sparsity is not None:
            params["target_sparsity"] = target_sparsity
        if granularity is not None:
            params["granularity"] = granularity
        return self._run(
            name=name,
            kind="prune",
            from_stage=from_stage,
            compare_to=compare_to,
            save_model=save_model,
            revert_on_reject=revert_on_reject,
            params=params,
            accept=accept,
        )

    def quant(
        self,
        *,
        name: str,
        backend: str | None = None,
        strategy: str | None = None,
        policy: Mapping[str, Any] | None = None,
        from_stage: str | None = None,
        compare_to: str | None = None,
        accept: Mapping[str, Any] | StageAcceptanceConfig | None = None,
        save_model: bool = True,
        revert_on_reject: bool = False,
        **params: Any,
    ) -> OptimizationStageResult:
        if backend is not None:
            params["backend"] = backend
        if strategy is not None:
            params["strategy"] = strategy
        if policy is not None:
            params["policy"] = dict(policy)
        return self._run(
            name=name,
            kind="quant",
            from_stage=from_stage,
            compare_to=compare_to,
            save_model=save_model,
            revert_on_reject=revert_on_reject,
            params=params,
            accept=accept,
        )

    def operator(
        self,
        *,
        name: str,
        from_stage: str | None = None,
        compare_to: str | None = None,
        accept: Mapping[str, Any] | StageAcceptanceConfig | None = None,
        save_model: bool = True,
        **params: Any,
    ) -> OptimizationStageResult:
        return self._run(
            name=name,
            kind="operator",
            from_stage=from_stage,
            compare_to=compare_to,
            save_model=save_model,
            params=params,
            accept=accept,
        )

    def export(
        self,
        *,
        name: str,
        format: str | None = None,
        output_path: str | Path | None = None,
        targets: list[Mapping[str, Any]] | None = None,
        target_params: Mapping[str, Any] | None = None,
        opset: int | None = None,
        from_stage: str | None = None,
        compare_to: str | None = None,
        accept: Mapping[str, Any] | StageAcceptanceConfig | None = None,
        save_model: bool = False,
        **params: Any,
    ) -> OptimizationStageResult:
        if targets is None:
            if format is None or output_path is None:
                raise ValueError("export requires targets or format + output_path")
            target: dict[str, Any] = {
                "format": format,
                "output_path": str(output_path),
            }
            if opset is not None:
                target["opset"] = opset
            if target_params is not None:
                target["params"] = dict(target_params)
            targets = [target]
        params["targets"] = [dict(target) for target in targets]
        return self._run(
            name=name,
            kind="export",
            from_stage=from_stage,
            compare_to=compare_to,
            save_model=save_model,
            params=params,
            accept=accept,
        )

    def analyze(
        self,
        *,
        name: str,
        from_stage: str | None = None,
        compare_to: str | None = None,
        accept: Mapping[str, Any] | StageAcceptanceConfig | None = None,
        save_model: bool = False,
        **params: Any,
    ) -> OptimizationStageResult:
        return self._run(
            name=name,
            kind="analyze",
            from_stage=from_stage,
            compare_to=compare_to,
            save_model=save_model,
            params=params,
            accept=accept,
        )

    def result(self, *, write_outputs: bool = True) -> OptimizedModelResult:
        result = _result_from_state(self._state)
        if write_outputs and not self._outputs_written:
            _write_workflow_outputs(result)
            self._outputs_written = True
        return result

    def write_outputs(self) -> OptimizedModelResult:
        return self.result(write_outputs=True)

    def _run(
        self,
        *,
        name: str,
        kind: str,
        compare_to: str | None = None,
        from_stage: str | None = None,
        save_model: bool = True,
        revert_on_reject: bool = False,
        params: Mapping[str, Any] | None = None,
        accept: Mapping[str, Any] | StageAcceptanceConfig | None = None,
    ) -> OptimizationStageResult:
        return self.run_stage(
            OptimizationStageConfig(
                name=name,
                kind=kind,
                compare_to=compare_to,
                from_stage=from_stage,
                save_model=save_model,
                revert_on_reject=revert_on_reject,
                params=dict(params or {}),
                accept=_acceptance_from_mapping(accept),
            )
        )

    def _validate_stage(self, stage: OptimizationStageConfig) -> None:
        if not stage.name:
            raise ValueError("stage.name is required")
        if stage.kind not in STAGE_KINDS:
            allowed = ", ".join(sorted(STAGE_KINDS))
            raise ValueError(f"unsupported stage kind {stage.kind}. Allowed: {allowed}")
        seen = {"initial", *(item.name for item in self._state.config.stages), *self._state.stages_by_name.keys()}
        if stage.name in seen:
            raise ValueError(f"stage names must be unique: {stage.name}")
        if stage.from_stage is not None and stage.from_stage not in self._state.model_snapshots and stage.from_stage not in self._state.stages_by_name:
            raise ValueError(f"unknown from_stage: {stage.from_stage}")


__all__ = [
    "OptimizedModelResult",
    "OptimizationConfig",
    "OptimizationStageConfig",
    "OptimizationStageResult",
    "StageAcceptanceConfig",
    "XQTOptimizationSession",
    "load_optimization_config",
    "optimize_model",
]
