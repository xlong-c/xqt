"""Stage-based optimization workflow for XQT."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, cast

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch import nn

from xdl.config.resolver import register_default_resolvers
from xqt.benchmark import benchmark_callable
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
    topk_accuracy,
)
from xqt.export.input_utils import (
    build_onnx_feed,
    call_model_with_example_input,
    default_input_names,
    first_tensor_output,
)
from xqt.pipeline.passes import (
    AnalyzePass,
    DistillPass,
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


def _build_model(config: OptimizationConfig, model: Any | None) -> Any:
    if model is not None:
        return model
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
) -> dict[str, Any]:
    model = context.require_model()
    if context.config.task.type == "detection":
        return evaluate_detection_model(
            model,
            dataloader,
            postprocess=context.config.task.detection_postprocess,
            metric_config=context.config.task.detection_metric,
            device=context.config.model.device,
        ).to_dict()
    return evaluate_pytorch_model(
        model,
        dataloader,
        device=context.config.model.device,
    ).to_dict()


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


def _run_supervised_finetune(
    context: XQTContext,
    *,
    dataloader: Any,
    params: Mapping[str, Any],
) -> None:
    model = cast(nn.Module, context.require_model())
    device = torch.device(context.config.model.device)
    model.to(device)
    was_training = model.training
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(params.get("lr", 1e-3)),
        weight_decay=float(params.get("weight_decay", 0.0)),
    )
    max_steps = params.get("max_steps")
    max_steps = int(max_steps) if max_steps is not None else None
    history: list[float] = []
    sample_count = 0

    for step, batch in enumerate(dataloader):
        if max_steps is not None and step >= max_steps:
            break
        split = split_batch(
            batch,
            expected_input_count=infer_model_input_count(model),
        )
        if split.targets is None:
            raise ValueError("supervised finetune requires labeled train data")
        inputs = _move_to_device(split.inputs, device)
        targets = _move_to_device(split.targets, device)
        if not isinstance(targets, torch.Tensor):
            raise TypeError("supervised finetune targets must be a tensor")

        logits = first_tensor_output(call_model_with_example_input(model, inputs))
        loss = F.cross_entropy(logits, targets.reshape(-1).long())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        history.append(float(loss.detach().cpu().item()))
        sample_count += int(targets.shape[0]) if targets.ndim > 0 else 1

    if not was_training:
        model.eval()
    steps = len(history)
    context.metrics["finetune"] = {
        "mode": "supervised",
        "steps": steps,
        "samples": sample_count,
        "mean_loss": sum(history) / steps if steps else 0.0,
        "last_loss": history[-1] if history else 0.0,
        "loss_history": history,
    }


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
    if train_split is None:
        raise ValueError(f"stage {stage.name} requires train_split")
    context.data["train"] = _require_split(splits, str(train_split))
    context.teacher = teacher or context.teacher
    if context.teacher is None and stage.kind == "distill":
        raise ValueError(
            f"stage {stage.name} requires a teacher model for distill finetune"
        )
    if context.teacher is None:
        _run_supervised_finetune(
            context,
            dataloader=context.data["train"],
            params=params,
        )
        return
    context.config.compression.distill.enabled = True
    context.config.compression.distill.temperature = float(params.pop("temperature", 2.0))
    context.config.compression.distill.alpha = float(params.pop("alpha", 0.5))
    context.config.compression.distill.params = params
    DistillPass().run(context)


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
    if runtime != "onnxruntime":
        raise ValueError("runtime_eval currently supports runtime=onnxruntime")
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
    reference_model = context.reference_model if params.pop("compare_reference", True) else None
    onnx_path = _resolve_runtime_artifact(context, params)

    if context.config.task.type == "detection":
        return evaluate_onnx_detection_model(
            str(onnx_path),
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
        ).to_dict() | {"path": str(onnx_path)}

    return _evaluate_onnx_classification_model(
        context,
        onnx_path=onnx_path,
        dataloader=dataloader,
        input_names=input_names,
        reference_model=reference_model if isinstance(reference_model, nn.Module) else None,
        max_batches=max_batches,
        benchmark_config=benchmark_config,
    )


def optimize_model(
    config: ConfigInput | OptimizationConfig,
    *,
    model: Any | None = None,
    teacher: Any | None = None,
    data: Mapping[str, Any] | None = None,
) -> OptimizedModelResult:
    """Run a decoupled stage workflow and return the optimized model/result."""

    loaded = load_optimization_config(config)
    current_model = _build_model(loaded, model)
    splits = _build_splits(loaded, data)
    context = XQTContext(
        config=_base_xqt_config(loaded, stage_name="init"),
        model=current_model,
        reference_model=copy.deepcopy(current_model)
        if isinstance(current_model, nn.Module)
        else current_model,
        teacher=teacher,
        data=dict(splits),
        device=_device(loaded),
    )

    stage_results: list[OptimizationStageResult] = []
    model_snapshots: dict[str, Any] = {
        "initial": copy.deepcopy(current_model) if isinstance(current_model, nn.Module) else current_model
    }
    eval_results: dict[str, dict[str, Any]] = {}
    benchmark_results: dict[str, dict[str, Any]] = {}
    baseline_stage: Optional[str] = None
    best_stage: Optional[str] = None

    for stage in loaded.stages:
        if not stage.enabled:
            continue
        if stage.from_stage is not None:
            source = model_snapshots[stage.from_stage]
            context.model = copy.deepcopy(source) if isinstance(source, nn.Module) else source
        before_model = (
            copy.deepcopy(context.model) if isinstance(context.model, nn.Module) else context.model
        )
        metrics_before = dict(context.metrics)
        artifacts_before = dict(context.artifacts)
        accepted = True
        message = "ok"
        stage_metrics: dict[str, Any] = {}
        stage_artifacts: dict[str, Any] = {}

        if stage.kind == "eval":
            stage_context = _stage_context(loaded, stage, context=context, splits=splits)
            split_name = _split_name(stage)
            report = _evaluate_current(
                stage_context,
                dataloader=_require_split(splits, split_name),
            )
            compare_to = stage.compare_to or baseline_stage
            reference = eval_results.get(compare_to) if compare_to else None
            accepted, message, checks = _accept_stage(
                stage,
                report,
                reference_eval=reference,
                reference_benchmark=None,
            )
            stage_metrics = {"eval": report, "acceptance": checks}
            eval_results[stage.name] = report
            context.metrics[stage.name] = stage_metrics
            if baseline_stage is None or bool(stage.params.get("baseline", False)):
                baseline_stage = stage.name
                context.reference_model = copy.deepcopy(context.model)
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
            reference = benchmark_results.get(compare_to) if compare_to else None
            accepted, message, checks = _accept_stage(
                stage,
                report,
                reference_eval=None,
                reference_benchmark=reference,
            )
            stage_metrics = {"benchmark": report, "acceptance": checks}
            benchmark_results[stage.name] = report
            context.metrics[stage.name] = stage_metrics
        elif stage.kind == "prune":
            _run_prune(loaded, stage, context, splits)
            stage_metrics = {"prune": context.metrics.get("prune", {})}
        elif stage.kind == "quant":
            _run_quant(loaded, stage, context, splits)
            stage_metrics = {"quant": context.metrics.get("quant", {})}
        elif stage.kind in {"finetune", "distill"}:
            _run_finetune(loaded, stage, context, splits, teacher)
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
            benchmark_results[stage.name] = report

        if stage.kind not in {"eval", "benchmark"} and _has_acceptance_thresholds(stage):
            compare_to = stage.compare_to or baseline_stage
            reference_eval = eval_results.get(compare_to) if compare_to else None
            reference_benchmark = (
                benchmark_results.get(compare_to) if compare_to else None
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
            model_snapshots[stage.name] = (
                copy.deepcopy(context.model)
                if isinstance(context.model, nn.Module)
                else context.model
            )
            if accepted and stage.kind not in {
                "eval",
                "benchmark",
                "export",
                "deploy",
                "analyze",
                "runtime_eval",
            }:
                best_stage = stage.name

        stage_results.append(
            OptimizationStageResult(
                name=stage.name,
                kind=stage.kind,
                accepted=accepted,
                metrics=stage_metrics,
                artifacts=stage_artifacts,
                message=message,
            )
        )

    return OptimizedModelResult(
        model=context.require_model(),
        context=context,
        stages=stage_results,
        best_stage=best_stage,
        baseline_stage=baseline_stage,
        models=model_snapshots,
    )


__all__ = [
    "OptimizedModelResult",
    "OptimizationConfig",
    "OptimizationStageConfig",
    "OptimizationStageResult",
    "StageAcceptanceConfig",
    "load_optimization_config",
    "optimize_model",
]
