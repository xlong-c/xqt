"""Stage-based model optimization workflow for XQT."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, TypeVar, cast

from omegaconf import OmegaConf

from xqt.core.config import register_default_resolvers
from xqt.core.config import ConfigInput
from xqt.core.errors import XQTConfigError
from xqt.core.schema import (
    AnalysisConfig,
    BenchmarkConfig,
    ModelConfig,
    OperatorOptimizationConfig,
    OutputDiffConfig,
    PruneConfig,
    QuantConfig,
    TaskConfig,
)
from xqt.core.types import XQTContext
from xqt.core.workflow_schema import (
    OptimizationConfig,
    OptimizationStageConfig,
    StageAcceptanceConfig,
)
from xqt.pipeline.passes import (
    run_analyze_stage,
    run_benchmark_stage,
    run_export_stage,
    run_operator_stage,
    run_prune_stage,
    run_quant_stage,
)
from xqt.readiness import XQTReadinessReport, assess_xqt_readiness
from .session_runner import (
    SessionStageRunners,
    _OptimizationRunState,
    acceptance_from_mapping,
    create_optimization_state,
    restore_stage_model,
    result_from_state,
    run_optimization_stage,
    write_workflow_outputs,
)
from .session_targets import build_session_export_targets
from .stage_specs import (
    AnalyzeStageSpec,
    BenchmarkStageSpec,
    DeployStageSpec,
    ExportStageSpec,
    OperatorStageSpec,
    PruneStageSpec,
    QuantStageSpec,
    StageSpec,
    ensure_stage_spec,
)
from .stage import (
    SessionStage,
    StageComparison,
    compare_session_stages,
)


STAGE_KINDS = {
    "benchmark",
    "prune",
    "quant",
    "operator",
    "export",
    "deploy",
    "analyze",
}

_REMOVED_WORKFLOW_TOP_LEVEL_KEYS = {
    "analysis",
    "compression",
    "config_version",
    "export",
    "operator_optimization",
    "validation",
}

_REMOVED_KEY_MIGRATIONS = {
    "compression": "split into stages[*].params for quant / prune stages",
    "operator_optimization": "move under stages[*].params for operator stages",
    "export": "move under stages[*].params.targets for export / deploy stages",
    "analysis": "move under stages[*].params for analyze stages",
    "validation": "move output_diff thresholds under stages[*].params.validate",
    "config_version": "drop it; OptimizationConfig has no public version field",
}

_StageSpecT = TypeVar("_StageSpecT", bound=StageSpec)


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


def load_optimization_config(
    config: ConfigInput | OptimizationConfig,
) -> OptimizationConfig:
    """Load a stage workflow config using OmegaConf structured defaults."""

    if is_dataclass(config) and isinstance(config, OptimizationConfig):
        _attach_stage_specs(config)
        return config

    register_default_resolvers()
    raw = (
        OmegaConf.load(config)
        if isinstance(config, (str, Path))
        else OmegaConf.create(config)
    )
    try:
        _validate_raw_optimization_config(raw)
        merged = OmegaConf.merge(OmegaConf.structured(OptimizationConfig), raw)
        OmegaConf.resolve(merged)
        loaded = cast(OptimizationConfig, OmegaConf.to_object(merged))
        _attach_stage_specs(loaded)
        return loaded
    except Exception as exc:
        if isinstance(exc, XQTConfigError):
            raise
        raise XQTConfigError(f"failed to load XQT optimization config: {exc}") from exc


def _validate_raw_optimization_config(raw_config: Any) -> None:
    raw = OmegaConf.to_container(raw_config, resolve=False, enum_to_str=True)
    if not isinstance(raw, Mapping):
        return
    removed = sorted(set(raw) & _REMOVED_WORKFLOW_TOP_LEVEL_KEYS)
    if removed:
        migration = ", ".join(
            f"{key} -> {_REMOVED_KEY_MIGRATIONS[key]}" for key in removed
        )
        raise XQTConfigError(
            "OptimizationConfig does not accept removed recipe top-level keys: "
            f"{removed}. Migration: {migration}."
        )


def _attach_stage_specs(config: OptimizationConfig) -> None:
    seen: set[str] = set()
    for index, stage in enumerate(config.stages):
        if not stage.name:
            raise XQTConfigError(f"stages.{index}.name is required")
        if stage.name in seen:
            raise XQTConfigError(f"stage names must be unique: {stage.name}")
        seen.add(stage.name)
        if stage.kind not in STAGE_KINDS:
            allowed = ", ".join(sorted(STAGE_KINDS))
            raise XQTConfigError(
                f"unsupported stage kind {stage.kind}. Allowed: {allowed}"
            )
        ensure_stage_spec(stage, rebuild=True)


def _typed_stage_spec(
    stage: OptimizationStageConfig, spec_type: type[_StageSpecT]
) -> _StageSpecT:
    spec = ensure_stage_spec(stage)
    if not isinstance(spec, spec_type):
        raise XQTConfigError(
            f"stage {stage.name!r} expected {spec_type.__name__}, "
            f"got {type(spec).__name__}"
        )
    return spec


def _project(
    config: OptimizationConfig, *, stage_name: str | None = None
) -> dict[str, Any]:
    project = dict(config.project)
    if stage_name is not None:
        artifact_dir = Path(
            str(project.get("artifact_dir", "artifacts/xqt/optimization"))
        )
        project["artifact_dir"] = str(artifact_dir / stage_name)
    return project


def _model_config(config: OptimizationConfig) -> dict[str, Any]:
    model = asdict(config.model) if is_dataclass(config.model) else dict(config.model)
    if config.device is not None:
        model["device"] = config.device
    return model


def _task_config(config: OptimizationConfig) -> dict[str, Any]:
    return asdict(config.task) if is_dataclass(config.task) else dict(config.task)


def _stage_context(
    config: OptimizationConfig,
    stage: OptimizationStageConfig,
    *,
    context: XQTContext,
) -> XQTContext:
    project = _project(config, stage_name=stage.name)
    model_config = _model_config(config)
    task_config = _task_config(config)
    context.device = str(model_config.get("device", ""))
    context.artifact_dir = str(project.get("artifact_dir", ""))
    context.project_name = str(project.get("name", ""))
    context.task_type = str(task_config.get("type", "classification"))
    context.compression_axes = list(config.compression_axes)
    context.model_target = model_config.get("target")
    context.model_params = copy.deepcopy(model_config.get("params", {}))
    context.quant_config = QuantConfig()
    context.prune_config = PruneConfig()
    context.analysis_config = AnalysisConfig()
    context.benchmark_config = copy.deepcopy(config.benchmark)
    context.operator_config = OperatorOptimizationConfig()
    context.output_diff_config = OutputDiffConfig()
    context.export_targets = []
    return context


def _run_prune(
    config: OptimizationConfig,
    stage: OptimizationStageConfig,
    context: XQTContext,
) -> None:
    _stage_context(config, stage, context=context)
    run_prune_stage(context, _typed_stage_spec(stage, PruneStageSpec))


def _run_quant(
    config: OptimizationConfig,
    stage: OptimizationStageConfig,
    context: XQTContext,
) -> None:
    _stage_context(config, stage, context=context)
    run_quant_stage(context, _typed_stage_spec(stage, QuantStageSpec))


def _run_operator(
    config: OptimizationConfig,
    stage: OptimizationStageConfig,
    context: XQTContext,
) -> None:
    _stage_context(config, stage, context=context)
    run_operator_stage(
        context,
        _typed_stage_spec(stage, OperatorStageSpec),
        base_benchmark_config=config.benchmark,
    )


def _run_export(
    config: OptimizationConfig,
    stage: OptimizationStageConfig,
    context: XQTContext,
) -> None:
    _stage_context(config, stage, context=context)
    if stage.kind == "deploy":
        spec = _typed_stage_spec(stage, DeployStageSpec)
    else:
        spec = _typed_stage_spec(stage, ExportStageSpec)
    run_export_stage(context, spec, stage_kind=stage.kind)


def _run_analyze(
    config: OptimizationConfig,
    stage: OptimizationStageConfig,
    context: XQTContext,
) -> None:
    _stage_context(config, stage, context=context)
    run_analyze_stage(context, _typed_stage_spec(stage, AnalyzeStageSpec))


def _run_benchmark(
    config: OptimizationConfig,
    stage: OptimizationStageConfig,
    context: XQTContext,
) -> None:
    _stage_context(config, stage, context=context)
    run_benchmark_stage(
        context,
        _typed_stage_spec(stage, BenchmarkStageSpec),
        base_benchmark_config=config.benchmark,
    )


def _stage_runners() -> SessionStageRunners:
    return SessionStageRunners(
        benchmark=_run_benchmark,
        prune=_run_prune,
        quant=_run_quant,
        operator=_run_operator,
        export=_run_export,
        analyze=_run_analyze,
    )


def _create_optimization_state(
    config: OptimizationConfig,
    *,
    model: Any | None = None,
    example_inputs: Any = None,
    calibration_inputs: Any = None,
) -> _OptimizationRunState:
    return create_optimization_state(
        config,
        model=model,
        example_inputs=example_inputs,
        calibration_inputs=calibration_inputs,
    )


def _run_optimization_stage(
    state: _OptimizationRunState,
    stage: OptimizationStageConfig,
) -> OptimizationStageResult | None:
    return run_optimization_stage(state, stage, runners=_stage_runners())


def _result_from_state(state: _OptimizationRunState) -> OptimizedModelResult:
    return result_from_state(state)


def _write_workflow_outputs(result: OptimizedModelResult) -> None:
    write_workflow_outputs(result)


def _acceptance_from_mapping(
    accept: Mapping[str, Any] | StageAcceptanceConfig | None,
) -> StageAcceptanceConfig:
    return acceptance_from_mapping(accept)


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

    def benchmark_leaderboard(
        self,
        *,
        metric: str = "speedup",
    ) -> Any:
        """Rank stage benchmark history into best / rejected stage view.

        The returned leaderboard is a pure view of ``stage_results``; it does
        not mutate session state. Supported metrics: ``speedup``, ``memory``,
        ``numeric``.
        """

        from xqt.auto.stage_history import rank_stage_benchmark_history

        return rank_stage_benchmark_history(
            self._state.stage_results,
            metric=metric,
        )

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
            self._state.context.artifacts[f"{name}_markdown"] = artifact_paths[
                "markdown"
            ]
        self._state.context.metrics[name] = report.to_dict()
        if self._state.context.manifest is not None:
            report.add_to_manifest(
                self._state.context.manifest,
                artifact_paths=artifact_paths,
            )
            if write_artifacts:
                manifest_path = (
                    Path(
                        str(
                            self._state.config.project.get(
                                "artifact_dir",
                                "artifacts/xqt/optimization",
                            )
                        )
                    )
                    / "manifest.json"
                )
                self._state.context.manifest.write_json(manifest_path)
                self._state.context.artifacts["manifest"] = manifest_path
        self._outputs_written = False
        return report

    def revert_to(self, stage_name: str) -> None:
        if (
            stage_name not in self._state.model_snapshots
            and stage_name not in self._state.stages_by_name
        ):
            raise ValueError(f"unknown model snapshot: {stage_name}")
        self._state.context.model = restore_stage_model(self._state, stage_name)

    def use(self, stage_name: str) -> None:
        self.revert_to(stage_name)

    def run_stage(self, stage: OptimizationStageConfig) -> OptimizationStageResult:
        self._validate_stage(stage)
        ensure_stage_spec(stage, rebuild=True)
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
        inference: Mapping[str, Any] | None = None,
        onnx: Mapping[str, Any] | None = None,
        openvino: Mapping[str, Any] | None = None,
        tensorrt: Mapping[str, Any] | None = None,
        torch_export: Mapping[str, Any] | None = None,
        torchscript: Mapping[str, Any] | None = None,
        executorch: Mapping[str, Any] | None = None,
        ncnn: Mapping[str, Any] | None = None,
        mnn: Mapping[str, Any] | None = None,
        qnn: Mapping[str, Any] | None = None,
        opset: int | None = None,
        from_stage: str | None = None,
        compare_to: str | None = None,
        accept: Mapping[str, Any] | StageAcceptanceConfig | None = None,
        save_model: bool = False,
        **params: Any,
    ) -> OptimizationStageResult:
        params["targets"] = build_session_export_targets(
            "export",
            format=format,
            output_path=output_path,
            targets=targets,
            target_params=target_params,
            opset=opset,
            inference=inference,
            onnx=onnx,
            openvino=openvino,
            tensorrt=tensorrt,
            torch_export=torch_export,
            torchscript=torchscript,
            executorch=executorch,
            ncnn=ncnn,
            mnn=mnn,
            qnn=qnn,
        )
        return self._run(
            name=name,
            kind="export",
            from_stage=from_stage,
            compare_to=compare_to,
            save_model=save_model,
            params=params,
            accept=accept,
        )

    def deploy(
        self,
        *,
        name: str,
        format: str | None = None,
        output_path: str | Path | None = None,
        targets: list[Mapping[str, Any]] | None = None,
        target_params: Mapping[str, Any] | None = None,
        inference: Mapping[str, Any] | None = None,
        onnx: Mapping[str, Any] | None = None,
        openvino: Mapping[str, Any] | None = None,
        tensorrt: Mapping[str, Any] | None = None,
        torch_export: Mapping[str, Any] | None = None,
        torchscript: Mapping[str, Any] | None = None,
        executorch: Mapping[str, Any] | None = None,
        ncnn: Mapping[str, Any] | None = None,
        mnn: Mapping[str, Any] | None = None,
        qnn: Mapping[str, Any] | None = None,
        runtime_handle: Mapping[str, Any] | None = None,
        opset: int | None = None,
        from_stage: str | None = None,
        compare_to: str | None = None,
        accept: Mapping[str, Any] | StageAcceptanceConfig | None = None,
        save_model: bool = False,
        **params: Any,
    ) -> OptimizationStageResult:
        params["targets"] = build_session_export_targets(
            "deploy",
            format=format,
            output_path=output_path,
            targets=targets,
            target_params=target_params,
            opset=opset,
            inference=inference,
            onnx=onnx,
            openvino=openvino,
            tensorrt=tensorrt,
            torch_export=torch_export,
            torchscript=torchscript,
            executorch=executorch,
            ncnn=ncnn,
            mnn=mnn,
            qnn=qnn,
        )
        if runtime_handle is not None:
            params["runtime_handle"] = dict(runtime_handle)
        return self._run(
            name=name,
            kind="deploy",
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
        seen = {
            "initial",
            *(item.name for item in self._state.config.stages),
            *self._state.stages_by_name.keys(),
        }
        if stage.name in seen:
            raise ValueError(f"stage names must be unique: {stage.name}")
        if (
            stage.from_stage is not None
            and stage.from_stage not in self._state.model_snapshots
            and stage.from_stage not in self._state.stages_by_name
        ):
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
