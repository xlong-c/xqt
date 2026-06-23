"""Stage-based model optimization workflow for XQT."""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, cast

from omegaconf import OmegaConf
from torch import nn

from xdl.config.resolver import register_default_resolvers
from xqt.core.config import ConfigInput, load_xqt_config
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
    baseline_stage: Optional[str] = None
    best_stage: Optional[str] = None


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
        }
    )


def _snapshot_model(model: Any) -> Any:
    return copy.deepcopy(model) if isinstance(model, nn.Module) else model


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
        if stage.from_stage not in state.model_snapshots:
            raise ValueError(f"unknown from_stage: {stage.from_stage}")
        state.context.model = _snapshot_model(state.model_snapshots[stage.from_stage])

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
    if accepted and stage.save_model:
        state.model_snapshots[stage.name] = _snapshot_model(state.context.model)
        state.best_stage = stage.name
    elif not accepted and stage.revert_on_reject and stage.from_stage is not None:
        state.context.model = _snapshot_model(state.model_snapshots[stage.from_stage])

    result = OptimizationStageResult(
        name=stage.name,
        kind=stage.kind,
        accepted=accepted,
        metrics=metrics,
        artifacts=_new_artifacts(before_artifacts, state.context.artifacts),
        message=message,
    )
    state.stage_results.append(result)
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
        "stages": [asdict(stage) for stage in result.stages],
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
    def baseline_stage(self) -> Optional[str]:
        return self._state.baseline_stage

    @property
    def best_stage(self) -> Optional[str]:
        return self._state.best_stage

    def set_example_inputs(self, example_inputs: Any) -> None:
        self._state.context.example_inputs = example_inputs

    def set_calibration_inputs(self, calibration_inputs: Any) -> None:
        self._state.context.calibration_inputs = calibration_inputs

    def revert_to(self, stage_name: str) -> None:
        if stage_name not in self._state.model_snapshots:
            raise ValueError(f"unknown model snapshot: {stage_name}")
        self._state.context.model = _snapshot_model(self._state.model_snapshots[stage_name])

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
        seen = {"initial", *(item.name for item in self._state.config.stages)}
        if stage.name in seen:
            raise ValueError(f"stage names must be unique: {stage.name}")
        if stage.from_stage is not None and stage.from_stage not in self._state.model_snapshots:
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
