"""Stage-based optimization workflow for XQT."""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, cast

import torch
from omegaconf import OmegaConf
from torch import nn

from xdl.config.resolver import register_default_resolvers
from xqt.benchmark import benchmark_callable
from xqt.core.artifact import ArtifactManifest, ArtifactRecord, MetricRecord, file_sha256
from xqt.core.config import ConfigInput, load_xqt_config
from xqt.core.errors import XQTBackendError
from xqt.core.imports import build_target
from xqt.core.schema import (
    BenchmarkConfig,
    DataSplitConfig,
    ModelConfig,
    PruneConfig,
    TaskConfig,
)
from xqt.core.types import XQTContext
from xqt.data import (
    build_data_split,
    extract_model_inputs,
    infer_model_input_count,
    split_batch,
)
from xqt.eval import (
    compare_tensors,
    evaluate_detection_model,
    evaluate_onnx_detection_model,
    evaluate_pytorch_model,
    evaluate_tensorrt_detection_model,
    topk_accuracy,
)
from xqt.export.input_utils import (
    build_onnx_feed,
    call_model_with_example_input,
    default_input_names,
    first_tensor_output,
)
from xqt.integrations import (
    EvaluationJob,
    resolve_evaluation_provider,
    run_evaluation_job,
    TrainingJob,
    resolve_training_provider,
    run_training_job,
)
from xqt.pipeline.passes import (
    AnalyzePass,
    ExportPass,
    OperatorOptimizationPass,
    PrunePass,
    QuantPass,
)


STAGE_KINDS = {
    "eval",
    "benchmark",
    "prune",
    "quant",
    "finetune",
    "distill",
    "operator",
    "export",
    "deploy",
    "analyze",
    "runtime_eval",
}


@dataclass
class StageAcceptanceConfig:
    """Acceptance thresholds for one stage."""

    metric: str = ""
    max_drop: Optional[float] = None
    min_speedup: Optional[float] = None
    max_mean_abs: Optional[float] = None
    max_max_abs: Optional[float] = None


@dataclass
class OptimizationStageConfig:
    """One independent optimization stage."""

    name: str
    kind: str
    enabled: bool = True
    split: Optional[str] = None
    train_split: Optional[str] = None
    calibration_split: Optional[str] = None
    validation_split: Optional[str] = None
    compare_to: Optional[str] = None
    from_stage: Optional[str] = None
    save_model: bool = True
    revert_on_reject: bool = False
    params: dict[str, Any] = field(default_factory=dict)
    accept: StageAcceptanceConfig = field(default_factory=StageAcceptanceConfig)


@dataclass
class OptimizationConfig:
    """Simple user-facing config for stage-based optimization."""

    project: dict[str, Any] = field(
        default_factory=lambda: {
            "name": "xqt_optimization",
            "artifact_dir": "artifacts/xqt/optimization",
        }
    )
    model: ModelConfig = field(default_factory=ModelConfig)
    task: TaskConfig = field(default_factory=TaskConfig)
    data_splits: dict[str, DataSplitConfig] = field(default_factory=dict)
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
    splits: dict[str, Any]
    teacher: Any | None = None
    stage_results: list[OptimizationStageResult] = field(default_factory=list)
    model_snapshots: dict[str, Any] = field(default_factory=dict)
    eval_results: dict[str, dict[str, Any]] = field(default_factory=dict)
    benchmark_results: dict[str, dict[str, Any]] = field(default_factory=dict)
    baseline_stage: Optional[str] = None
    best_stage: Optional[str] = None


def load_optimization_config(config: ConfigInput | OptimizationConfig) -> OptimizationConfig:
    """Load a stage workflow config using OmegaConf structured defaults."""

    if is_dataclass(config) and isinstance(config, OptimizationConfig):
        loaded = config
    else:
        register_default_resolvers()
        raw = OmegaConf.load(config) if isinstance(config, (str, Path)) else OmegaConf.create(config)
        try:
            merged = OmegaConf.merge(OmegaConf.structured(OptimizationConfig), raw)
            OmegaConf.resolve(merged)
            loaded = cast(OptimizationConfig, OmegaConf.to_object(merged))
        except Exception as exc:
            raise ValueError(f"failed to load optimization config: {exc}") from exc
    _validate_optimization_config(loaded)
    return loaded


def _validate_optimization_config(config: OptimizationConfig) -> None:
    seen: set[str] = {"initial"}
    for stage in config.stages:
        if not stage.name:
            raise ValueError("stage.name is required")
        if stage.name in seen:
            raise ValueError(f"stage names must be unique: {stage.name}")
        if stage.kind not in STAGE_KINDS:
            allowed = ", ".join(sorted(STAGE_KINDS))
            raise ValueError(f"unsupported stage kind {stage.kind}. Allowed: {allowed}")
        if stage.from_stage is not None and stage.from_stage not in seen:
            raise ValueError(
                f"stage {stage.name} references unknown previous from_stage {stage.from_stage}"
            )
        seen.add(stage.name)


def _project_name(config: OptimizationConfig) -> str:
    return str(config.project.get("name", "xqt_optimization"))


def _artifact_dir(config: OptimizationConfig) -> str:
    return str(config.project.get("artifact_dir", "artifacts/xqt/optimization"))


def _device(config: OptimizationConfig) -> str:
    return str(config.device or config.model.device or "cpu")


def _stage_requires_model(stage: OptimizationStageConfig) -> bool:
    if stage.kind in {"eval", "benchmark", "prune", "finetune", "distill", "operator", "analyze"}:
        return True
    if stage.kind == "quant":
        params = stage.params
        backend = str(params.get("backend", ""))
        policy = params.get("policy", {})
        if not isinstance(policy, Mapping):
            policy = {}
        if backend == "onnxruntime_qdq" and policy.get("onnx_path") is not None:
            return False
        return True
    if stage.kind == "runtime_eval":
        return False
    if stage.kind not in {"export", "deploy"}:
        return False
    targets = stage.params.get("targets", [])
    if not isinstance(targets, list) or not targets:
        return True
    for target in targets:
        if not isinstance(target, Mapping):
            return True
        fmt = str(target.get("format", ""))
        params = target.get("params", {})
        if not isinstance(params, Mapping):
            params = {}
        if fmt in {"tensorrt", "openvino", "ncnn", "mnn"} and params.get("onnx_path") is not None:
            continue
        return True
    return False


def _workflow_requires_model(config: OptimizationConfig) -> bool:
    return any(_stage_requires_model(stage) for stage in config.stages)


def _base_xqt_config(config: OptimizationConfig, *, stage_name: str) -> Any:
    return load_xqt_config(
        {
            "project": {
                "name": f"{_project_name(config)}_{stage_name}",
                "artifact_dir": str(Path(_artifact_dir(config)) / stage_name),
            },
            "model": {
                **asdict(config.model),
                "device": _device(config),
            },
            "task": asdict(config.task),
        }
    )


def _create_workflow_manifest(config: OptimizationConfig) -> ArtifactManifest:
    source_checksum: Optional[str] = None
    if config.model.checkpoint:
        checkpoint_path = Path(config.model.checkpoint).expanduser()
        if checkpoint_path.is_file():
            source_checksum = file_sha256(checkpoint_path)
    return ArtifactManifest(
        project_name=_project_name(config),
        source_checkpoint=config.model.checkpoint,
        source_checksum=source_checksum,
        task={
            "type": config.task.type,
            "class_names": list(config.task.class_names),
            "detection_postprocess": asdict(config.task.detection_postprocess),
            "detection_metric": asdict(config.task.detection_metric),
            "params": dict(config.task.params),
        },
        config_snapshot=asdict(config),
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _write_workflow_outputs(
    result: OptimizedModelResult,
    *,
    summary_name: str = "workflow_result.json",
    manifest_name: str = "workflow_manifest.json",
) -> None:
    artifact_dir = Path(result.context.config.project.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    if result.context.manifest is not None:
        seen_metric_names = {metric.name for metric in result.context.manifest.metrics}
        seen_artifact_paths = {artifact.path for artifact in result.context.manifest.artifacts}
        for stage in result.stages:
            result.context.manifest.passes.append(stage.name)
            result.context.manifest.add_metric(
                MetricRecord(
                    name=f"workflow.stage.{stage.name}.accepted",
                    value=stage.accepted,
                    metadata={"kind": stage.kind, "message": stage.message},
                )
            )
            for top_key, payload in stage.metrics.items():
                metric_name = f"workflow.stage.{stage.name}.{top_key}"
                if metric_name in seen_metric_names:
                    continue
                result.context.manifest.add_metric(
                    MetricRecord(
                        name=metric_name,
                        value=_json_safe(payload),
                    )
                )
                seen_metric_names.add(metric_name)
            detection_metadata = _stage_detection_manifest_metadata(stage.metrics)
            if detection_metadata is not None:
                dataset_metric_name = f"workflow.stage.{stage.name}.dataset_metadata"
                if dataset_metric_name not in seen_metric_names:
                    result.context.manifest.add_metric(
                        MetricRecord(
                            name=dataset_metric_name,
                            value=_json_safe(detection_metadata.get("dataset")),
                            metadata={"kind": stage.kind, "source": "detection_report"},
                        )
                    )
                    seen_metric_names.add(dataset_metric_name)
                batch_metric_name = f"workflow.stage.{stage.name}.batch_metadata"
                if batch_metric_name not in seen_metric_names:
                    result.context.manifest.add_metric(
                        MetricRecord(
                            name=batch_metric_name,
                            value=_json_safe(detection_metadata.get("batches")),
                            metadata={"kind": stage.kind, "source": "detection_report"},
                        )
                    )
                    seen_metric_names.add(batch_metric_name)
            for artifact_key, artifact_value in stage.artifacts.items():
                if isinstance(artifact_value, (str, Path)):
                    path = Path(artifact_value)
                    if path.is_file() and str(path) not in seen_artifact_paths:
                        result.context.manifest.add_artifact(
                            ArtifactRecord.from_file(
                                path,
                                format=stage.kind,
                                runtime=artifact_key,
                                metadata={"stage": stage.name, "artifact_key": artifact_key},
                            )
                        )
                        seen_artifact_paths.add(str(path))
        manifest_path = artifact_dir / manifest_name
        result.context.manifest.write_json(manifest_path)
        result.context.artifacts["workflow_manifest"] = manifest_path

    summary = {
        "project": result.context.config.project.name,
        "artifact_dir": str(artifact_dir),
        "baseline_stage": result.baseline_stage,
        "best_stage": result.best_stage,
        "stages": [
            {
                "name": stage.name,
                "kind": stage.kind,
                "accepted": stage.accepted,
                "message": stage.message,
                "metrics": _json_safe(stage.metrics),
                "artifacts": _json_safe(stage.artifacts),
            }
            for stage in result.stages
        ],
        "artifacts": _json_safe(result.context.artifacts),
    }
    summary_path = artifact_dir / summary_name
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    result.context.artifacts["workflow_result"] = summary_path


def _stage_detection_manifest_metadata(stage_metrics: Mapping[str, Any]) -> dict[str, Any] | None:
    for key in ("eval", "runtime_eval"):
        payload = stage_metrics.get(key)
        if not isinstance(payload, Mapping):
            continue
        metadata = payload.get("metadata")
        if not isinstance(metadata, Mapping):
            continue
        dataset = metadata.get("dataset")
        batches = metadata.get("batches")
        if dataset is None and batches is None:
            continue
        return {
            "dataset": dict(dataset) if isinstance(dataset, Mapping) else dataset,
            "batches": list(batches) if isinstance(batches, list) else batches,
        }
    return None


def _build_model(config: OptimizationConfig, model: Any | None) -> Any:
    if model is not None:
        return model
    if not _workflow_requires_model(config):
        return None
    if not config.model.target:
        raise ValueError("model.target is required when model is not provided")
    built = build_target(config.model.target, config.model.params)
    if isinstance(built, nn.Module):
        built.eval()
    return built


def _build_splits(config: OptimizationConfig, data: Mapping[str, Any] | None) -> dict[str, Any]:
    splits = dict(data or {})
    for name, split_config in config.data_splits.items():
        if name in splits:
            continue
        splits[name] = build_data_split(
            name,
            split_config,
            model_params=config.model.params,
            default_seed=len(splits),
        )
    return splits


def _snapshot_model(model: Any) -> Any:
    return copy.deepcopy(model) if isinstance(model, nn.Module) else model


def _create_optimization_state(
    config: OptimizationConfig,
    *,
    model: Any | None = None,
    teacher: Any | None = None,
    data: Mapping[str, Any] | None = None,
    training_provider: Any | None = None,
    evaluation_provider: Any | None = None,
) -> _OptimizationRunState:
    current_model = _build_model(config, model)
    splits = _build_splits(config, data)
    context = XQTContext(
        config=_base_xqt_config(config, stage_name="init"),
        model=current_model,
        reference_model=_snapshot_model(current_model),
        teacher=teacher,
        data=dict(splits),
        device=_device(config),
        manifest=_create_workflow_manifest(config),
        training_provider=training_provider,
        evaluation_provider=evaluation_provider,
    )
    return _OptimizationRunState(
        config=config,
        context=context,
        splits=splits,
        teacher=teacher,
        model_snapshots={"initial": _snapshot_model(current_model)},
    )


def _result_from_state(state: _OptimizationRunState) -> OptimizedModelResult:
    if state.context.manifest is not None:
        state.context.manifest.config_snapshot = asdict(state.config)
    return OptimizedModelResult(
        model=state.context.model,
        context=state.context,
        stages=list(state.stage_results),
        best_stage=state.best_stage,
        baseline_stage=state.baseline_stage,
        models=dict(state.model_snapshots),
    )


def _split_name(stage: OptimizationStageConfig, default: str = "validation") -> str:
    return str(stage.split or stage.validation_split or default)


def _require_split(splits: Mapping[str, Any], name: str) -> Any:
    if name not in splits:
        raise ValueError(f"data split not found: {name}")
    return splits[name]


def _move_to_device(data: Any, device: torch.device) -> Any:
    if isinstance(data, torch.Tensor):
        return data.to(device)
    if isinstance(data, Mapping):
        return {key: _move_to_device(value, device) for key, value in data.items()}
    if isinstance(data, tuple):
        return tuple(_move_to_device(value, device) for value in data)
    if isinstance(data, list):
        return [_move_to_device(value, device) for value in data]
    return data


def _first_batch_inputs(model: nn.Module, dataloader: Any, device: str) -> Any:
    batch = next(iter(dataloader))
    expected = infer_model_input_count(model)
    inputs = extract_model_inputs(batch, expected_input_count=expected)
    return _move_to_device(inputs, torch.device(device))


def _evaluate_current(
    context: XQTContext,
    *,
    dataloader: Any,
    stage_name: str,
    params: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    model = context.require_model()
    eval_params = dict(params or {})
    provider = resolve_evaluation_provider(context, eval_params)
    if provider is not None:
        report = run_evaluation_job(
            provider,
            EvaluationJob(
                name=stage_name,
                mode="eval",
                model=model,
                data=dataloader,
                reference_model=context.reference_model,
                params=eval_params,
                device=context.config.model.device,
                task_type=context.config.task.type,
                context=context,
            ),
        )
        return report.to_dict()
    if context.config.task.type == "detection":
        report = evaluate_detection_model(
            model,
            dataloader,
            postprocess=context.config.task.detection_postprocess,
            metric_config=context.config.task.detection_metric,
            device=context.config.model.device,
        ).to_dict()
    else:
        report = evaluate_pytorch_model(
            model,
            dataloader,
            device=context.config.model.device,
        ).to_dict()
    report.setdefault("provider", "xqt_internal_transition")
    metadata = report.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    metadata.setdefault("evaluation_source", "xqt.eval")
    report["metadata"] = metadata
    return report


def _benchmark_model(
    model: nn.Module,
    *,
    dataloader: Any,
    config: BenchmarkConfig,
    device: str,
) -> dict[str, Any]:
    model = model.to(device)
    inputs = _first_batch_inputs(model, dataloader, device)

    def call() -> object:
        return call_model_with_example_input(model, inputs)

    return benchmark_callable(
        call,
        warmup=config.warmup,
        iterations=config.iterations,
        sync_cuda=config.sync_cuda,
        device=device,
    ).to_dict()


def _benchmark_against_reference(
    reference: Mapping[str, Any] | None,
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    reference_mean = reference.get("mean_ms") if reference else None
    candidate_mean = candidate.get("mean_ms")
    if reference_mean is None and reference is not None:
        reference_mean = _max_nested_numeric(reference, "mean_ms")
    if candidate_mean is None:
        candidate_mean = _max_nested_numeric(candidate, "mean_ms")
    speedup = None
    if isinstance(reference_mean, (float, int)) and isinstance(candidate_mean, (float, int)):
        speedup = float(reference_mean) / float(candidate_mean) if candidate_mean else None
    return {"speedup": speedup}


def _metric_drop(
    reference: Mapping[str, Any] | None,
    candidate: Mapping[str, Any],
    metric: str,
) -> Optional[float]:
    if reference is None:
        return None
    ref_metrics = reference.get("metrics")
    cand_metrics = candidate.get("metrics")
    if not isinstance(ref_metrics, Mapping) or not isinstance(cand_metrics, Mapping):
        return None
    ref_value = ref_metrics.get(metric)
    cand_value = cand_metrics.get(metric)
    if not isinstance(ref_value, (float, int)) or not isinstance(cand_value, (float, int)):
        return None
    return float(ref_value) - float(cand_value)


def _accept_stage(
    stage: OptimizationStageConfig,
    metrics: Mapping[str, Any],
    *,
    reference_eval: Mapping[str, Any] | None,
    reference_benchmark: Mapping[str, Any] | None,
) -> tuple[bool, str, dict[str, Any]]:
    accept = stage.accept
    checks: dict[str, Any] = {}
    accepted = True

    if accept.metric and accept.max_drop is not None:
        drop = _metric_drop(reference_eval, metrics, accept.metric)
        checks["metric_drop"] = drop
        checks["metric"] = accept.metric
        checks["max_drop"] = accept.max_drop
        if drop is None:
            accepted = False
        elif drop > accept.max_drop:
            accepted = False

    if accept.min_speedup is not None:
        speed = _benchmark_against_reference(reference_benchmark, metrics).get("speedup")
        if speed is None:
            speed = _max_nested_numeric(metrics, "speedup")
        checks["speedup"] = speed
        checks["min_speedup"] = accept.min_speedup
        if speed is None:
            accepted = False
        elif speed < accept.min_speedup:
            accepted = False

    if accept.max_mean_abs is not None:
        mean_abs = _max_nested_numeric(metrics, "mean_abs")
        checks["mean_abs"] = mean_abs
        checks["max_mean_abs"] = accept.max_mean_abs
        if mean_abs is None:
            accepted = False
        elif mean_abs > accept.max_mean_abs:
            accepted = False

    if accept.max_max_abs is not None:
        max_abs = _max_nested_numeric(metrics, "max_abs")
        checks["max_abs"] = max_abs
        checks["max_max_abs"] = accept.max_max_abs
        if max_abs is None:
            accepted = False
        elif max_abs > accept.max_max_abs:
            accepted = False

    return accepted, "ok" if accepted else "rejected by acceptance thresholds", checks


def _stage_context(
    config: OptimizationConfig,
    stage: OptimizationStageConfig,
    *,
    context: XQTContext,
    splits: Mapping[str, Any],
) -> XQTContext:
    stage_config = _base_xqt_config(config, stage_name=stage.name)
    context.config = stage_config
    context.device = stage_config.model.device
    context.data.update(splits)
    return context


def _has_acceptance_thresholds(stage: OptimizationStageConfig) -> bool:
    accept = stage.accept
    return any(
        (
            bool(accept.metric and accept.max_drop is not None),
            accept.min_speedup is not None,
            accept.max_mean_abs is not None,
            accept.max_max_abs is not None,
        )
    )


def _benchmark_config(params: Mapping[str, Any]) -> BenchmarkConfig:
    try:
        merged = OmegaConf.merge(OmegaConf.structured(BenchmarkConfig), dict(params))
        return cast(BenchmarkConfig, OmegaConf.to_object(merged))
    except Exception as exc:
        raise ValueError(f"failed to load benchmark stage params: {exc}") from exc


def _new_artifacts(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        key: value
        for key, value in after.items()
        if key not in before or before[key] != value
    }


def _collect_numeric_key(value: Any, key: str) -> list[float]:
    values: list[float] = []
    if isinstance(value, Mapping):
        raw = value.get(key)
        if isinstance(raw, (float, int)):
            values.append(float(raw))
        for item in value.values():
            values.extend(_collect_numeric_key(item, key))
    elif isinstance(value, (list, tuple)):
        for item in value:
            values.extend(_collect_numeric_key(item, key))
    return values


def _max_nested_numeric(value: Any, key: str) -> Optional[float]:
    values = _collect_numeric_key(value, key)
    return max(values) if values else None


def _run_prune(
    config: OptimizationConfig,
    stage: OptimizationStageConfig,
    context: XQTContext,
    splits: Mapping[str, Any],
) -> None:
    _stage_context(config, stage, context=context, splits=splits)
    params = dict(stage.params)
    params.pop("enabled", None)
    train_split = stage.train_split or params.pop("train_split", None)
    validation_split = stage.validation_split or stage.split or params.pop(
        "validation_split",
        None,
    )
    context.config.compression.prune = PruneConfig(
        enabled=True,
        **params,
    )
    if train_split is not None:
        context.data["train"] = _require_split(splits, str(train_split))
    if validation_split is not None:
        context.data["validation"] = _require_split(splits, str(validation_split))
    PrunePass().run(context)


def _run_quant(
    config: OptimizationConfig,
    stage: OptimizationStageConfig,
    context: XQTContext,
    splits: Mapping[str, Any],
) -> None:
    _stage_context(config, stage, context=context, splits=splits)
    params = dict(stage.params)
    params.pop("enabled", None)
    calibration_split = stage.calibration_split or params.pop("calibration_split", None)
    validation_split = (
        stage.validation_split
        or stage.split
        or params.pop("validation_split", None)
    )
    if calibration_split is not None:
        context.data["calibration"] = _require_split(splits, str(calibration_split))
        params.setdefault("calibration_split", "calibration")
    if validation_split is not None:
        context.data["validation"] = _require_split(splits, str(validation_split))
        params.setdefault("validation_split", "validation")
    context.config.compression.quant = load_xqt_config(
        {"compression": {"quant": {"enabled": True, **params}}}
    ).compression.quant
    QuantPass().run(context)


def _run_finetune(
    config: OptimizationConfig,
    stage: OptimizationStageConfig,
    context: XQTContext,
    splits: Mapping[str, Any],
    teacher: Any | None,
) -> None:
    _stage_context(config, stage, context=context, splits=splits)
    params = dict(stage.params)
    params.pop("enabled", None)
    train_split = stage.train_split or stage.split or params.pop("train_split", None)
    validation_split = stage.validation_split or params.pop("validation_split", None)
    if train_split is None:
        raise ValueError(f"stage {stage.name} requires train_split")
    context.data["train"] = _require_split(splits, str(train_split))
    if validation_split is not None:
        context.data["validation"] = _require_split(splits, str(validation_split))
    context.teacher = teacher or context.teacher
    provider = resolve_training_provider(context, params)
    job_teacher = params.pop("teacher", context.teacher)
    if job_teacher is None and stage.kind == "distill":
        raise ValueError(
            f"stage {stage.name} requires a teacher model for distill finetune"
        )
    metric_key = "distill" if stage.kind == "distill" else "finetune"
    report = run_training_job(
        provider,
        TrainingJob(
            name=stage.name,
            mode=stage.kind,
            model=context.require_model(),
            teacher=job_teacher,
            train_data=context.data["train"],
            validation_data=context.data.get("validation"),
            params=params,
            device=context.config.model.device,
            task_type=context.config.task.type,
            context=context,
        ),
    )
    context.metrics[metric_key] = report.to_dict()
    if context.manifest is not None:
        mean_loss = report.metrics.get("mean_loss")
        context.manifest.add_metric(
            MetricRecord(
                name=f"{metric_key}.provider",
                value=float(mean_loss) if isinstance(mean_loss, (float, int)) else 0.0,
                metadata={
                    "provider": report.provider,
                    "mode": report.mode,
                    "steps": report.steps,
                    "samples": report.samples,
                    "message": report.message,
                },
            )
        )


def _run_operator(
    config: OptimizationConfig,
    stage: OptimizationStageConfig,
    context: XQTContext,
    splits: Mapping[str, Any],
) -> None:
    _stage_context(config, stage, context=context, splits=splits)
    params = dict(stage.params)
    params.pop("enabled", None)
    validation_split = stage.validation_split or stage.split or params.pop(
        "validation_split",
        None,
    )
    context.config.operator_optimization = load_xqt_config(
        {"operator_optimization": {"enabled": True, **params}}
    ).operator_optimization
    if validation_split is not None:
        context.data["validation"] = _require_split(splits, str(validation_split))
    OperatorOptimizationPass().run(context)


def _run_export(
    config: OptimizationConfig,
    stage: OptimizationStageConfig,
    context: XQTContext,
    splits: Mapping[str, Any],
) -> None:
    _stage_context(config, stage, context=context, splits=splits)
    validation_split = stage.validation_split or stage.split or "validation"
    context.data["validation"] = _require_split(splits, str(validation_split))
    params = dict(stage.params)
    params.pop("enabled", None)
    targets = params.pop("targets", [])
    context.config.export = load_xqt_config(
        {"export": {"targets": targets}}
    ).export
    ExportPass().run(context)


def _run_analyze(
    config: OptimizationConfig,
    stage: OptimizationStageConfig,
    context: XQTContext,
    splits: Mapping[str, Any],
) -> None:
    _stage_context(config, stage, context=context, splits=splits)
    params = dict(stage.params)
    validation_split = (
        stage.validation_split
        or stage.split
        or params.pop("validation_split", None)
        or "validation"
    )
    context.data["validation"] = _require_split(splits, str(validation_split))
    params.pop("enabled", None)
    context.config.analysis = load_xqt_config(
        {"analysis": {"enabled": True, **params}}
    ).analysis
    AnalyzePass().run(context)


def _resolve_runtime_artifact(
    context: XQTContext,
    params: Mapping[str, Any],
    *,
    default_key: str = "last_onnx",
) -> Path:
    path_value = params.get("path") or params.get("artifact_path")
    artifact_key = params.get("artifact") or params.get("artifact_key") or default_key
    if path_value is None and artifact_key is not None:
        path_value = context.artifacts.get(str(artifact_key))
    if isinstance(path_value, (list, tuple)):
        path_value = path_value[0] if path_value else None
    if path_value is None:
        raise ValueError(f"runtime_eval requires params.path or artifact key {artifact_key}")
    return Path(path_value)


def _runtime_eval_artifact_metadata(
    context: XQTContext,
    params: Mapping[str, Any],
    resolved_path: Path,
    *,
    default_key: str = "last_onnx",
) -> dict[str, Any]:
    artifact_key = str(params.get("artifact") or params.get("artifact_key") or default_key)
    explicit_path = params.get("path") or params.get("artifact_path")
    source = "explicit_path" if explicit_path is not None else "artifact"
    if source == "artifact":
        context_value = context.artifacts.get(artifact_key)
        source = "artifact" if context_value is not None else "artifact_missing"
    return {
        "artifact_key": artifact_key,
        "source": source,
        "resolved_path": str(resolved_path),
    }


def _detection_task_metadata(context: XQTContext) -> dict[str, Any]:
    task = context.config.task
    return {
        "task_type": task.type,
        "class_names": list(task.class_names),
        "detection_postprocess": asdict(task.detection_postprocess),
        "detection_metric": asdict(task.detection_metric),
    }


def _detection_runtime_boundary_metadata(
    context: XQTContext,
    *,
    runtime: str,
) -> dict[str, Any]:
    quant_metrics = context.metrics.get("quant", {})
    if not isinstance(quant_metrics, Mapping):
        quant_metrics = {}
    quant_metadata = quant_metrics.get("metadata", {})
    if not isinstance(quant_metadata, Mapping):
        quant_metadata = {}
    qdq_graph = quant_metadata.get("qdq_graph", {})
    if not isinstance(qdq_graph, Mapping):
        qdq_graph = {}
    return {
        "runtime": runtime,
        "quantized_runtime_scope": (
            "onnx_graph_only" if quant_metrics.get("backend") == "onnxruntime_qdq" else "unknown"
        ),
        "decode_stage": "runtime_eval_postprocess",
        "decode_execution": "outside_quantized_graph",
        "postprocess_format": context.config.task.detection_postprocess.format,
        "box_format": context.config.task.detection_postprocess.box_format,
        "score_activation": context.config.task.detection_postprocess.score_activation,
        "rescale_to_original": bool(context.config.task.detection_postprocess.rescale_to_original),
        "quant_backend": quant_metrics.get("backend"),
        "quantized_op_types": list(quant_metadata.get("quantized_op_types", [])),
        "qdq_node_count": qdq_graph.get("qdq_node_count"),
        "quantize_linear_count": qdq_graph.get("quantize_linear_count"),
        "dequantize_linear_count": qdq_graph.get("dequantize_linear_count"),
        "raw_model_output_contract": (
            "logits+pred_boxes" if context.config.task.detection_postprocess.format == "auto" else "generic_detection"
        ),
    }


def _onnx_input_type_map(session: Any) -> dict[str, str]:
    return {
        str(item.name): str(getattr(item, "type", ""))
        for item in session.get_inputs()
    }


def _cast_onnx_feed(feed: Mapping[str, Any], input_types: Mapping[str, str]) -> dict[str, Any]:
    casted: dict[str, Any] = {}
    for name, value in feed.items():
        if input_types.get(name) == "tensor(float16)":
            casted[name] = value.astype("float16")
        else:
            casted[name] = value
    return casted


def _evaluate_onnx_classification_model(
    context: XQTContext,
    *,
    onnx_path: Path,
    dataloader: Any,
    input_names: list[str] | None,
    reference_model: nn.Module | None,
    max_batches: int | None,
    benchmark_config: BenchmarkConfig,
) -> dict[str, Any]:
    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError as exc:
        raise XQTBackendError("onnxruntime is required for runtime_eval") from exc

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_types = _onnx_input_type_map(session)
    device = torch.device(context.config.model.device)
    resolved_input_names = list(input_names or [])
    total_samples = 0
    top1_sum = 0.0
    raw_reference_tensors: list[torch.Tensor] = []
    raw_candidate_tensors: list[torch.Tensor] = []
    benchmark_inputs: Any = None

    if reference_model is not None:
        reference_model.to(device)
        reference_model.eval()

    for batch_index, batch in enumerate(dataloader):
        if max_batches is not None and batch_index >= max_batches:
            break
        split = split_batch(batch)
        inputs = split.inputs
        names = resolved_input_names or default_input_names(inputs)
        benchmark_inputs = inputs if benchmark_inputs is None else benchmark_inputs
        feed = _cast_onnx_feed(
            build_onnx_feed(inputs, input_names=names),
            input_types,
        )
        ort_outputs = session.run(None, feed)
        candidate = torch.from_numpy(np.asarray(ort_outputs[0]))
        raw_candidate_tensors.append(candidate.detach().cpu())
        if isinstance(split.targets, torch.Tensor):
            targets = split.targets.detach().cpu()
            batch_size = int(targets.shape[0]) if targets.ndim > 0 else 1
            total_samples += batch_size
            top1_sum += topk_accuracy(candidate, targets) * batch_size

        if reference_model is not None:
            moved_inputs = _move_to_device(inputs, device)
            with torch.no_grad():
                reference_output = call_model_with_example_input(
                    reference_model,
                    moved_inputs,
                )
            raw_reference_tensors.append(first_tensor_output(reference_output).detach().cpu())

    raw_output_diff = None
    if raw_reference_tensors and raw_candidate_tensors:
        raw_output_diff = compare_tensors(
            torch.cat([tensor.reshape(-1) for tensor in raw_reference_tensors]),
            torch.cat([tensor.reshape(-1) for tensor in raw_candidate_tensors]),
            atol=context.config.validation.output_diff.atol,
            rtol=context.config.validation.output_diff.rtol,
        )

    latency = None
    if benchmark_inputs is not None:
        names = resolved_input_names or default_input_names(benchmark_inputs)
        latency = benchmark_callable(
            lambda: session.run(
                None,
                _cast_onnx_feed(
                    build_onnx_feed(benchmark_inputs, input_names=names),
                    input_types,
                ),
            ),
            warmup=benchmark_config.warmup,
            iterations=benchmark_config.iterations,
            sync_cuda=benchmark_config.sync_cuda,
            device="cpu",
        ).to_dict()

    metrics = {"top1": top1_sum / total_samples if total_samples else 0.0}
    return {
        "runtime": "onnxruntime",
        "path": str(onnx_path),
        "samples": total_samples,
        "metrics": metrics,
        "raw_output_diff": (
            raw_output_diff.to_dict() if raw_output_diff is not None else None
        ),
        "latency": dict(latency or {}),
    }


def _run_runtime_eval(
    config: OptimizationConfig,
    stage: OptimizationStageConfig,
    context: XQTContext,
    splits: Mapping[str, Any],
) -> dict[str, Any]:
    _stage_context(config, stage, context=context, splits=splits)
    params = dict(stage.params)
    runtime = str(params.pop("runtime", "onnxruntime"))
    split_name = _split_name(stage)
    dataloader = _require_split(splits, split_name)
    benchmark_params = {
        key: params.pop(key)
        for key in ("warmup", "iterations", "sync_cuda", "measure_memory", "percentiles")
        if key in params
    }
    benchmark_config = _benchmark_config(benchmark_params)
    max_batches = params.pop("max_batches", None)
    max_batches = int(max_batches) if max_batches is not None else None
    input_names_value = params.pop("input_names", None)
    input_names = [str(name) for name in input_names_value] if input_names_value else None
    output_names_value = params.pop("output_names", None)
    output_names = [str(name) for name in output_names_value] if output_names_value else None
    reference_model = context.reference_model if params.pop("compare_reference", True) else None

    if context.config.task.type == "detection":
        runtime_default_key = "last_engine" if runtime == "tensorrt" else "last_onnx"
        runtime_path = _resolve_runtime_artifact(
            context,
            params,
            default_key=runtime_default_key,
        )
        artifact_metadata = _runtime_eval_artifact_metadata(
            context,
            stage.params,
            runtime_path,
            default_key=runtime_default_key,
        )
        if runtime == "tensorrt":
            report = evaluate_tensorrt_detection_model(
                str(runtime_path),
                dataloader,
                postprocess=context.config.task.detection_postprocess,
                metric_config=context.config.task.detection_metric,
                input_names=input_names,
                output_names=output_names,
                reference_model=reference_model if isinstance(reference_model, nn.Module) else None,
                device=context.config.model.device,
                max_batches=max_batches,
                atol=context.config.validation.output_diff.atol,
                rtol=context.config.validation.output_diff.rtol,
                benchmark_warmup=benchmark_config.warmup,
                benchmark_iterations=benchmark_config.iterations,
                benchmark_sync_cuda=benchmark_config.sync_cuda,
            ).to_dict()
            report["path"] = str(runtime_path)
            report["artifact"] = artifact_metadata
            report["task"] = _detection_task_metadata(context)
            report["quantization"] = dict(context.metrics.get("quant", {}))
            report["runtime_boundary"] = _detection_runtime_boundary_metadata(
                context,
                runtime=runtime,
            )
            return report
        if runtime != "onnxruntime":
            raise ValueError("detection runtime_eval supports runtime=onnxruntime|tensorrt")
        report = evaluate_onnx_detection_model(
            str(runtime_path),
            dataloader,
            postprocess=context.config.task.detection_postprocess,
            metric_config=context.config.task.detection_metric,
            input_names=input_names,
            reference_model=reference_model if isinstance(reference_model, nn.Module) else None,
            device=context.config.model.device,
            max_batches=max_batches,
            atol=context.config.validation.output_diff.atol,
            rtol=context.config.validation.output_diff.rtol,
            benchmark_warmup=benchmark_config.warmup,
            benchmark_iterations=benchmark_config.iterations,
            benchmark_sync_cuda=benchmark_config.sync_cuda,
        ).to_dict()
        report["path"] = str(runtime_path)
        report["artifact"] = artifact_metadata
        report["task"] = _detection_task_metadata(context)
        report["quantization"] = dict(context.metrics.get("quant", {}))
        report["runtime_boundary"] = _detection_runtime_boundary_metadata(
            context,
            runtime=runtime,
        )
        return report

    if runtime != "onnxruntime":
        raise ValueError("classification runtime_eval currently supports runtime=onnxruntime")
    onnx_path = _resolve_runtime_artifact(context, params)
    artifact_metadata = _runtime_eval_artifact_metadata(context, stage.params, onnx_path)
    report = _evaluate_onnx_classification_model(
        context,
        onnx_path=onnx_path,
        dataloader=dataloader,
        input_names=input_names,
        reference_model=reference_model if isinstance(reference_model, nn.Module) else None,
        max_batches=max_batches,
        benchmark_config=benchmark_config,
    )
    report["artifact"] = artifact_metadata
    return report


def _run_optimization_stage(
    state: _OptimizationRunState,
    stage: OptimizationStageConfig,
) -> OptimizationStageResult | None:
    if not stage.enabled:
        return None

    loaded = state.config
    context = state.context
    splits = state.splits
    if stage.from_stage is not None:
        source = state.model_snapshots[stage.from_stage]
        context.model = _snapshot_model(source)

    before_model = _snapshot_model(context.model)
    metrics_before = dict(context.metrics)
    artifacts_before = dict(context.artifacts)
    accepted = True
    message = "ok"
    stage_metrics: dict[str, Any] = {}

    if stage.kind == "eval":
        stage_context = _stage_context(loaded, stage, context=context, splits=splits)
        split_name = _split_name(stage)
        report = _evaluate_current(
            stage_context,
            dataloader=_require_split(splits, split_name),
            stage_name=stage.name,
            params=stage.params,
        )
        compare_to = stage.compare_to or state.baseline_stage
        reference = state.eval_results.get(compare_to) if compare_to else None
        accepted, message, checks = _accept_stage(
            stage,
            report,
            reference_eval=reference,
            reference_benchmark=None,
        )
        stage_metrics = {"eval": report, "acceptance": checks}
        state.eval_results[stage.name] = report
        context.metrics[stage.name] = stage_metrics
        if state.baseline_stage is None or bool(stage.params.get("baseline", False)):
            state.baseline_stage = stage.name
            context.reference_model = _snapshot_model(context.model)
    elif stage.kind == "benchmark":
        stage_context = _stage_context(loaded, stage, context=context, splits=splits)
        split_name = _split_name(stage)
        bench_cfg = _benchmark_config(stage.params)
        report = _benchmark_model(
            cast(nn.Module, stage_context.require_model()),
            dataloader=_require_split(splits, split_name),
            config=bench_cfg,
            device=stage_context.config.model.device,
        )
        compare_to = stage.compare_to
        reference = state.benchmark_results.get(compare_to) if compare_to else None
        accepted, message, checks = _accept_stage(
            stage,
            report,
            reference_eval=None,
            reference_benchmark=reference,
        )
        stage_metrics = {"benchmark": report, "acceptance": checks}
        state.benchmark_results[stage.name] = report
        context.metrics[stage.name] = stage_metrics
    elif stage.kind == "prune":
        _run_prune(loaded, stage, context, splits)
        stage_metrics = {"prune": context.metrics.get("prune", {})}
    elif stage.kind == "quant":
        _run_quant(loaded, stage, context, splits)
        stage_metrics = {"quant": context.metrics.get("quant", {})}
    elif stage.kind in {"finetune", "distill"}:
        _run_finetune(loaded, stage, context, splits, state.teacher)
        metric_key = "distill" if stage.kind == "distill" else "finetune"
        stage_metrics = {
            metric_key: context.metrics.get(metric_key, context.metrics.get("distill", {}))
        }
    elif stage.kind == "operator":
        _run_operator(loaded, stage, context, splits)
        stage_metrics = {
            "operator": context.metrics.get("operator_optimization", {})
        }
    elif stage.kind in {"export", "deploy"}:
        _run_export(loaded, stage, context, splits)
        stage_metrics = {"export": context.metrics.get("export", {})}
    elif stage.kind == "analyze":
        _run_analyze(loaded, stage, context, splits)
        stage_metrics = {"analysis": context.metrics.get("analysis", {})}
    elif stage.kind == "runtime_eval":
        report = _run_runtime_eval(loaded, stage, context, splits)
        context.metrics["runtime_eval"] = report
        stage_metrics = {
            "runtime_eval": report,
            "metrics": report.get("metrics", {}),
            "latency": report.get("latency", {}),
            "raw_output_diff": report.get("raw_output_diff"),
            "decoded_diff": report.get("decoded_diff"),
        }
        state.benchmark_results[stage.name] = report

    if stage.kind not in {"eval", "benchmark"} and _has_acceptance_thresholds(stage):
        compare_to = stage.compare_to or state.baseline_stage
        reference_eval = state.eval_results.get(compare_to) if compare_to else None
        reference_benchmark = (
            state.benchmark_results.get(compare_to) if compare_to else None
        )
        accepted, message, checks = _accept_stage(
            stage,
            stage_metrics,
            reference_eval=reference_eval,
            reference_benchmark=reference_benchmark,
        )
        stage_metrics["acceptance"] = checks

    stage_artifacts = _new_artifacts(artifacts_before, context.artifacts)
    context.metrics[stage.name] = stage_metrics

    if not accepted and stage.revert_on_reject:
        context.model = before_model
        context.metrics = metrics_before
        context.artifacts = artifacts_before
    elif stage.save_model:
        state.model_snapshots[stage.name] = _snapshot_model(context.model)
        if accepted and stage.kind not in {
            "eval",
            "benchmark",
            "export",
            "deploy",
            "analyze",
            "runtime_eval",
        }:
            state.best_stage = stage.name

    result = OptimizationStageResult(
        name=stage.name,
        kind=stage.kind,
        accepted=accepted,
        metrics=stage_metrics,
        artifacts=stage_artifacts,
        message=message,
    )
    state.stage_results.append(result)
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
        teacher: Any | None = None,
        data_splits: Mapping[str, Any] | None = None,
        training_provider: Any | None = None,
        evaluation_provider: Any | None = None,
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
                raw_config["task"] = (
                    asdict(task) if is_dataclass(task) else dict(task)
                )
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
            teacher=teacher,
            data=data_splits,
            training_provider=training_provider,
            evaluation_provider=evaluation_provider,
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

    def set_data(self, name: str, data: Any) -> None:
        self._state.splits[name] = data
        self._state.context.data[name] = data

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

    def eval(
        self,
        *,
        name: str,
        split: str = "validation",
        compare_to: str | None = None,
        baseline: bool = False,
        accept: Mapping[str, Any] | StageAcceptanceConfig | None = None,
        save_model: bool = True,
        **params: Any,
    ) -> OptimizationStageResult:
        if baseline:
            params = {**params, "baseline": True}
        return self._run(
            name=name,
            kind="eval",
            split=split,
            compare_to=compare_to,
            save_model=save_model,
            params=params,
            accept=accept,
        )

    def benchmark(
        self,
        *,
        name: str,
        split: str = "validation",
        compare_to: str | None = None,
        accept: Mapping[str, Any] | StageAcceptanceConfig | None = None,
        save_model: bool = True,
        **params: Any,
    ) -> OptimizationStageResult:
        return self._run(
            name=name,
            kind="benchmark",
            split=split,
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
        split: str | None = None,
        train_split: str | None = None,
        validation_split: str | None = None,
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
            split=split,
            train_split=train_split,
            validation_split=validation_split,
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
        calibration_split: str | None = None,
        validation_split: str | None = None,
        split: str | None = None,
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
            split=split,
            calibration_split=calibration_split,
            validation_split=validation_split,
            from_stage=from_stage,
            compare_to=compare_to,
            save_model=save_model,
            revert_on_reject=revert_on_reject,
            params=params,
            accept=accept,
        )

    def finetune(
        self,
        *,
        name: str,
        train_split: str = "train",
        compare_to: str | None = None,
        accept: Mapping[str, Any] | StageAcceptanceConfig | None = None,
        save_model: bool = True,
        **params: Any,
    ) -> OptimizationStageResult:
        return self._run(
            name=name,
            kind="finetune",
            train_split=train_split,
            compare_to=compare_to,
            save_model=save_model,
            params=params,
            accept=accept,
        )

    def distill(
        self,
        *,
        name: str,
        train_split: str = "train",
        compare_to: str | None = None,
        accept: Mapping[str, Any] | StageAcceptanceConfig | None = None,
        save_model: bool = True,
        **params: Any,
    ) -> OptimizationStageResult:
        return self._run(
            name=name,
            kind="distill",
            train_split=train_split,
            compare_to=compare_to,
            save_model=save_model,
            params=params,
            accept=accept,
        )

    def operator(
        self,
        *,
        name: str,
        split: str | None = None,
        validation_split: str | None = None,
        from_stage: str | None = None,
        compare_to: str | None = None,
        accept: Mapping[str, Any] | StageAcceptanceConfig | None = None,
        save_model: bool = True,
        **params: Any,
    ) -> OptimizationStageResult:
        return self._run(
            name=name,
            kind="operator",
            split=split,
            validation_split=validation_split,
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
        split: str = "validation",
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
            split=split,
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
        split: str | None = None,
        validation_split: str | None = None,
        from_stage: str | None = None,
        compare_to: str | None = None,
        accept: Mapping[str, Any] | StageAcceptanceConfig | None = None,
        save_model: bool = False,
        **params: Any,
    ) -> OptimizationStageResult:
        return self._run(
            name=name,
            kind="analyze",
            split=split,
            validation_split=validation_split,
            from_stage=from_stage,
            compare_to=compare_to,
            save_model=save_model,
            params=params,
            accept=accept,
        )

    def runtime_eval(
        self,
        *,
        name: str,
        split: str = "validation",
        artifact: str | None = None,
        path: str | Path | None = None,
        input_names: list[str] | None = None,
        output_names: list[str] | None = None,
        runtime: str | None = None,
        from_stage: str | None = None,
        compare_to: str | None = None,
        accept: Mapping[str, Any] | StageAcceptanceConfig | None = None,
        save_model: bool = False,
        **params: Any,
    ) -> OptimizationStageResult:
        if artifact is not None:
            params["artifact"] = artifact
        if path is not None:
            params["path"] = str(path)
        if input_names is not None:
            params["input_names"] = list(input_names)
        if output_names is not None:
            params["output_names"] = list(output_names)
        if runtime is not None:
            params["runtime"] = runtime
        return self._run(
            name=name,
            kind="runtime_eval",
            split=split,
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
        split: str | None = None,
        train_split: str | None = None,
        calibration_split: str | None = None,
        validation_split: str | None = None,
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
                split=split,
                train_split=train_split,
                calibration_split=calibration_split,
                validation_split=validation_split,
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
            raise ValueError(
                f"stage {stage.name} references unknown previous from_stage {stage.from_stage}"
            )


def optimize_model(
    config: ConfigInput | OptimizationConfig,
    *,
    model: Any | None = None,
    teacher: Any | None = None,
    data: Mapping[str, Any] | None = None,
    training_provider: Any | None = None,
    evaluation_provider: Any | None = None,
) -> OptimizedModelResult:
    """Run a decoupled stage workflow and return the optimized model/result."""

    loaded = load_optimization_config(config)
    state = _create_optimization_state(
        loaded,
        model=model,
        teacher=teacher,
        data=data,
        training_provider=training_provider,
        evaluation_provider=evaluation_provider,
    )
    for stage in loaded.stages:
        _run_optimization_stage(state, stage)
    result = _result_from_state(state)
    _write_workflow_outputs(result)
    return result


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
