"""Built-in lightweight XQT passes."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from xqt.benchmark import benchmark_callable, benchmark_memory
from xqt.core.artifact import ArtifactRecord, MetricRecord
from xqt.core.imports import build_target
from xqt.core.registry import register_pass
from xqt.core.types import XQTContext
from xqt.data import (
    SyntheticClassificationSpec,
    TorchvisionImageClassificationSpec,
    build_synthetic_classification_loader,
    build_torchvision_image_classification_loader,
)
from xqt.distill import HFTextClassificationBundle, train_logit_distillation
from xqt.eval import (
    evaluate_pytorch_model,
    write_json_report,
    write_markdown_report,
)
from xqt.export import (
    build_tensorrt_engine,
    compare_onnxruntime_outputs,
    export_executorch_program,
    export_mnn_from_onnx,
    export_ncnn_from_onnx,
    export_onnx,
    export_torch_program,
    export_torchscript,
)
from xqt.prune import (
    PruningSchedule,
    apply_global_l1_unstructured_pruning,
    remove_pruning_reparameterization,
    run_prune_kd_loop,
    summarize_pruning,
)
from xqt.quant import quantize_with_torchao
from xqt.quant.onnx_qdq import quantize_onnx_qdq_static


def _split_inputs_from_batch(batch: Any) -> Any:
    if isinstance(batch, Mapping):
        inputs = batch.get("inputs", batch.get("input", batch.get("x")))
        if inputs is not None:
            return inputs
        return {
            key: value
            for key, value in batch.items()
            if key not in {"targets", "target", "y", "labels", "label"}
        }
    if isinstance(batch, (tuple, list)) and len(batch) >= 2:
        return batch[0] if len(batch) == 2 else tuple(batch[:-1])
    return batch


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


def _call_model(model: nn.Module, inputs: Any) -> Any:
    if isinstance(inputs, Mapping):
        return model(**inputs)
    if isinstance(inputs, tuple):
        return model(*inputs)
    return model(inputs)


@register_pass("load_model")
class LoadModelPass:
    """Build the configured PyTorch model."""

    name = "load_model"

    def run(self, context: XQTContext) -> XQTContext:
        if context.model is not None:
            return context
        target = context.config.model.target
        if not target:
            raise ValueError("model.target is required when context.model is not set")
        model = build_target(target, context.config.model.params)
        if isinstance(model, HFTextClassificationBundle):
            context.model = model.student
            context.teacher = model.teacher
            context.artifacts["tokenizer"] = model.tokenizer
            context.metrics["hf_text_bundle"] = model.metadata
            context.data.setdefault("train", model.train_loader)
            if model.validation_loader is not None:
                context.data.setdefault("validation", model.validation_loader)
            return context
        if not isinstance(model, nn.Module):
            raise TypeError("model.target must build a torch.nn.Module")
        model.eval()
        context.model = model
        return context


@register_pass("load_data")
class LoadDataPass:
    """Build configured calibration and validation data."""

    name = "load_data"

    def _build_data_split(self, split: Any, context: XQTContext, *, default_seed: int) -> Any:
        if split.target == "synthetic_classification":
            input_dim = int(context.config.model.params.get("in_features", 4))
            output_dim = int(context.config.model.params.get("out_features", 2))
            input_shape = split.params.get("input_shape")
            num_classes = int(split.params.get("num_classes", output_dim))
            spec = SyntheticClassificationSpec(
                sample_limit=split.sample_limit or 8,
                batch_size=split.batch_size,
                input_dim=input_dim,
                input_shape=input_shape,
                num_classes=num_classes,
                seed=int(split.params.get("seed", default_seed)),
            )
            return build_synthetic_classification_loader(spec)
        if split.target == "torchvision_image_classification":
            spec = TorchvisionImageClassificationSpec(
                dataset_name=str(split.params.get("dataset_name", "CIFAR100")),
                root=split.root or str(split.params.get("root", "./data")),
                train=bool(split.params.get("train", False)),
                download=bool(split.params.get("download", False)),
                batch_size=split.batch_size,
                sample_limit=split.sample_limit,
                shuffle=bool(split.params.get("shuffle", False)),
                num_workers=int(split.params.get("num_workers", 0)),
                transform_params=dict(split.params.get("transform_params", {})),
            )
            return build_torchvision_image_classification_loader(spec)
        raise ValueError(
            "Only target=synthetic_classification or "
            "target=torchvision_image_classification is supported by the built-in "
            "data pass"
        )

    def run(self, context: XQTContext) -> XQTContext:
        calibration = context.config.data.calibration
        validation = context.config.data.validation
        train = context.config.data.train
        if calibration is not None and "calibration" not in context.data:
            context.data["calibration"] = self._build_data_split(
                calibration,
                context,
                default_seed=2,
            )
        if train is not None and "train" not in context.data:
            context.data["train"] = self._build_data_split(
                train,
                context,
                default_seed=1,
            )
        if validation is not None and "validation" not in context.data:
            context.data["validation"] = self._build_data_split(
                validation,
                context,
                default_seed=0,
            )
        return context


@register_pass("distill")
class DistillPass:
    """Run a small teacher-to-student logit distillation loop."""

    name = "distill"

    def run(self, context: XQTContext) -> XQTContext:
        model = context.require_model()
        distill_config = context.config.compression.distill
        if not distill_config.enabled:
            return context
        if context.teacher is None:
            raise ValueError("context.teacher is required for distill pass")
        dataloader = context.data.get("train")
        if dataloader is None:
            raise ValueError("train data is required for distill pass")

        params = distill_config.params
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(params.get("lr", 1e-3)),
            weight_decay=float(params.get("weight_decay", 0.0)),
        )
        report = train_logit_distillation(
            model,
            context.teacher,
            dataloader,
            optimizer,
            temperature=distill_config.temperature,
            alpha=distill_config.alpha,
            device=context.config.model.device,
            max_steps=params.get("max_steps"),
        )
        context.metrics["distill"] = report.to_dict()
        if context.manifest is not None:
            context.manifest.add_metric(
                MetricRecord(
                    name="distill.mean_loss",
                    value=report.mean_loss,
                    metadata={"steps": report.steps},
                )
            )
        return context


@register_pass("baseline_eval")
class BaselineEvalPass:
    """Evaluate the current PyTorch model on validation data."""

    name = "baseline_eval"

    def run(self, context: XQTContext) -> XQTContext:
        model = context.require_model()
        dataloader = context.data.get("validation")
        if dataloader is None:
            raise ValueError("validation data is required for baseline_eval")
        report = evaluate_pytorch_model(
            model,
            dataloader,
            device=context.config.model.device,
        )
        context.metrics["baseline"] = report.to_dict()
        if context.manifest is not None:
            for metric_name, value in report.metrics.items():
                context.manifest.add_metric(
                    MetricRecord(
                        name=f"baseline.{metric_name}",
                        value=value,
                    )
                )
        return context


@register_pass("prune")
class PrunePass:
    """Apply global L1 unstructured pruning when enabled."""

    name = "prune"

    def run(self, context: XQTContext) -> XQTContext:
        model = context.require_model()
        prune_config = context.config.compression.prune
        if not prune_config.enabled:
            return context
        if prune_config.method != "global_l1_unstructured":
            raise ValueError(
                "Only compression.prune.method=global_l1_unstructured is supported "
                "by the built-in prune pass"
            )
        if prune_config.schedule in {"linear", "one_shot"} and (
            int(prune_config.params.get("steps", 1)) > 1
            or bool(prune_config.params.get("finetune", False))
        ):
            dataloader = context.data.get("train")
            optimizer = None
            if dataloader is not None and bool(prune_config.params.get("finetune", False)):
                optimizer = torch.optim.AdamW(
                    model.parameters(),
                    lr=float(prune_config.params.get("lr", 1e-3)),
                    weight_decay=float(prune_config.params.get("weight_decay", 0.0)),
                )
            report = run_prune_kd_loop(
                model,
                context.teacher,
                dataloader,
                schedule=PruningSchedule(
                    target_sparsity=prune_config.target_sparsity,
                    steps=int(prune_config.params.get("steps", 1)),
                    start_sparsity=float(prune_config.params.get("start_sparsity", 0.0)),
                    schedule=prune_config.schedule,
                ),
                optimizer=optimizer,
                temperature=context.config.compression.distill.temperature,
                alpha=context.config.compression.distill.alpha,
                device=context.config.model.device,
                kd_steps_per_prune=prune_config.params.get("kd_steps_per_prune"),
            )
            context.metrics["prune"] = report.to_dict()
            if context.manifest is not None:
                context.manifest.add_metric(
                    MetricRecord(
                        name="prune.sparsity",
                        value=report.final_sparsity,
                        threshold=prune_config.target_sparsity,
                        passed=report.final_sparsity >= prune_config.target_sparsity,
                        metadata={
                            "schedule": prune_config.schedule,
                            "steps": len(report.steps),
                            "finetune": optimizer is not None,
                        },
                    )
                )
            return context
        report = apply_global_l1_unstructured_pruning(
            model,
            prune_config.target_sparsity,
        )
        remove_pruning_reparameterization(model)
        report = summarize_pruning(model)
        context.metrics["prune"] = report.to_dict()
        if context.manifest is not None:
            context.manifest.add_metric(
                MetricRecord(
                    name="prune.sparsity",
                    value=report.sparsity,
                    threshold=prune_config.target_sparsity,
                    passed=report.sparsity >= prune_config.target_sparsity,
                )
            )
        return context


@register_pass("quant")
class QuantPass:
    """Apply configured quantization backend."""

    name = "quant"

    def run(self, context: XQTContext) -> XQTContext:
        model = context.require_model()
        quant_config = context.config.compression.quant
        if not quant_config.enabled:
            return context
        if quant_config.backend == "onnxruntime_qdq":
            dataloader = context.data.get("calibration") or context.data.get("validation")
            if dataloader is None:
                raise ValueError("calibration or validation data is required for ONNX QDQ")
            policy = quant_config.policy
            onnx_path = policy.get("onnx_path") or context.artifacts.get("last_onnx")
            if onnx_path is None:
                batch = next(iter(dataloader))
                example_input = _split_inputs_from_batch(batch)
                if not isinstance(example_input, torch.Tensor):
                    raise TypeError(
                        "ONNX QDQ auto-export currently supports one Tensor input"
                    )
                onnx_path = (
                    Path(context.config.project.artifact_dir)
                    / str(policy.get("source_name", "quant_source.onnx"))
                )
                export_onnx(
                    model,
                    example_input,
                    onnx_path,
                    opset=policy.get("opset"),
                    input_names=policy.get("input_names"),
                    output_names=policy.get("output_names"),
                    dynamo=bool(policy.get("dynamo", True)),
                    validate=bool(policy.get("validate", True)),
                )
                context.artifacts["last_onnx"] = Path(onnx_path)
            output_path = policy.get("output_path")
            if output_path is None:
                output_path = str(
                    Path(context.config.project.artifact_dir) / "model_qdq.onnx"
                )
            result = quantize_onnx_qdq_static(
                onnx_path,
                output_path,
                dataloader,
                input_names=policy.get("input_names") or ("input",),
                sample_limit=policy.get("sample_limit"),
                activation_type=str(policy.get("activation_type", "QUInt8")),
                weight_type=str(policy.get("weight_type", "QInt8")),
                per_channel=bool(policy.get("per_channel", False)),
                reduce_range=bool(policy.get("reduce_range", False)),
                op_types_to_quantize=policy.get("op_types_to_quantize"),
                extra_options=policy.get("extra_options"),
            )
            context.artifacts["quant_onnx"] = result.path
            context.artifacts["last_onnx"] = result.path
            context.metrics["quant"] = {
                "backend": quant_config.backend,
                "path": str(result.path),
                "checksum": result.checksum,
                "calibration_samples": result.calibration_samples,
                "metadata": result.metadata,
            }
            if context.manifest is not None:
                context.manifest.add_artifact(
                    ArtifactRecord(
                        path=str(result.path),
                        format="onnx",
                        runtime="onnxruntime",
                        checksum=result.checksum,
                        metadata={"quantization": "qdq", **result.metadata},
                    )
                )
                context.manifest.add_metric(
                    MetricRecord(
                        name="quant.calibration_samples",
                        value=result.calibration_samples,
                        metadata={"backend": quant_config.backend},
                    )
                )
            return context
        if quant_config.backend != "torchao":
            raise ValueError(
                "compression.quant.backend must be torchao or onnxruntime_qdq"
            )
        result = quantize_with_torchao(
            model,
            policy=quant_config.policy,
            strategy=quant_config.policy.get("strategy"),
            inplace=True,
        )
        context.model = result.model
        context.metrics["quant"] = {
            "backend": result.backend,
            "strategy": result.strategy,
            "quantized_modules": result.quantized_modules,
            "quantized_module_count": len(result.quantized_modules),
        }
        if context.manifest is not None:
            context.manifest.add_metric(
                MetricRecord(
                    name="quant.quantized_module_count",
                    value=len(result.quantized_modules),
                    metadata={"strategy": result.strategy},
                )
            )
        return context


@register_pass("export")
class ExportPass:
    """Export configured artifacts."""

    name = "export"

    def run(self, context: XQTContext) -> XQTContext:
        model = context.require_model()
        if not context.config.export.targets:
            return context

        dataloader = context.data.get("validation")
        if dataloader is None:
            raise ValueError("validation data is required for export")
        batch = next(iter(dataloader))
        example_input = batch[0] if isinstance(batch, (tuple, list)) else batch
        if not isinstance(example_input, torch.Tensor):
            raise TypeError("built-in export pass currently supports one Tensor input")

        exported: list[dict[str, object]] = []
        with torch.no_grad():
            reference_output = model(example_input)
            if isinstance(reference_output, (tuple, list)):
                reference_output = reference_output[0]
            if not isinstance(reference_output, torch.Tensor):
                raise TypeError("model output must be a Tensor for export diff")

        for index, target in enumerate(context.config.export.targets):
            if target.format == "torch_export":
                output_path = target.output_path
                if output_path is None:
                    output_path = str(
                        Path(context.config.project.artifact_dir)
                        / f"model_{index}.pt2"
                    )
                result = export_torch_program(
                    model,
                    example_input,
                    output_path,
                    dynamic_shapes=target.dynamic_shapes,
                    strict=bool(target.params.get("strict", False)),
                    validate=bool(target.params.get("validate", True)),
                    compare_output=bool(target.params.get("runtime_diff", True)),
                    atol=context.config.validation.output_diff.atol,
                    rtol=context.config.validation.output_diff.rtol,
                )
                record = ArtifactRecord.from_file(
                    result.path,
                    format="torch_export",
                    runtime="pytorch",
                    metadata={
                        "checked": result.checked,
                        "output_diff": (
                            result.output_diff.to_dict()
                            if result.output_diff is not None
                            else None
                        ),
                        **result.metadata,
                    },
                )
                context.artifacts[f"export_{index}"] = result.path
                if context.manifest is not None:
                    context.manifest.add_artifact(record)
                exported.append(
                    {
                        "path": str(result.path),
                        "format": "torch_export",
                        "checked": result.checked,
                        "checksum": result.checksum,
                        "output_diff": (
                            result.output_diff.to_dict()
                            if result.output_diff is not None
                            else None
                        ),
                    }
                )
                continue

            if target.format == "torchscript":
                output_path = target.output_path
                if output_path is None:
                    output_path = str(
                        Path(context.config.project.artifact_dir)
                        / f"model_{index}.pt"
                    )
                result = export_torchscript(
                    model,
                    example_input,
                    output_path,
                    method=str(target.params.get("method", "trace")),
                    check_trace=bool(target.params.get("check_trace", True)),
                    compare_output=bool(target.params.get("runtime_diff", True)),
                    atol=context.config.validation.output_diff.atol,
                    rtol=context.config.validation.output_diff.rtol,
                )
                record = ArtifactRecord.from_file(
                    result.path,
                    format="torchscript",
                    runtime="pytorch",
                    metadata={
                        "output_diff": (
                            result.output_diff.to_dict()
                            if result.output_diff is not None
                            else None
                        ),
                        **result.metadata,
                    },
                )
                context.artifacts[f"export_{index}"] = result.path
                context.artifacts.setdefault("last_torchscript", result.path)
                if context.manifest is not None:
                    context.manifest.add_artifact(record)
                exported.append(
                    {
                        "path": str(result.path),
                        "format": "torchscript",
                        "checksum": result.checksum,
                        "output_diff": (
                            result.output_diff.to_dict()
                            if result.output_diff is not None
                            else None
                        ),
                    }
                )
                continue

            if target.format == "onnx":
                output_path = target.output_path
                if output_path is None:
                    output_path = str(
                        Path(context.config.project.artifact_dir)
                        / f"model_{index}.onnx"
                    )
                result = export_onnx(
                    model,
                    example_input,
                    output_path,
                    opset=target.opset,
                    dynamic_shapes=target.dynamic_shapes,
                    input_names=target.params.get("input_names"),
                    output_names=target.params.get("output_names"),
                    dynamo=bool(target.params.get("dynamo", True)),
                    validate=bool(target.params.get("validate", True)),
                )
                diff = None
                if bool(target.params.get("runtime_diff", True)):
                    diff = compare_onnxruntime_outputs(
                        result.path,
                        reference_output,
                        example_input,
                        input_name=(target.params.get("input_names") or ["input"])[0],
                        atol=context.config.validation.output_diff.atol,
                        rtol=context.config.validation.output_diff.rtol,
                    )
                    result.output_diff = diff
                record = ArtifactRecord.from_file(
                    result.path,
                    format="onnx",
                    runtime="onnxruntime" if diff is not None else None,
                    metadata={
                        "opset": result.opset,
                        "checked": result.checked,
                        "output_diff": diff.to_dict() if diff is not None else None,
                    },
                )
                context.artifacts[f"export_{index}"] = result.path
                context.artifacts.setdefault("last_onnx", result.path)
                if context.manifest is not None:
                    context.manifest.add_artifact(record)
                exported.append(
                    {
                        "path": str(result.path),
                        "format": "onnx",
                        "checked": result.checked,
                        "checksum": result.checksum,
                        "output_diff": diff.to_dict() if diff is not None else None,
                    }
                )
                continue

            if target.format == "tensorrt":
                onnx_path = target.params.get("onnx_path") or context.artifacts.get(
                    "last_onnx"
                )
                if onnx_path is None:
                    raise ValueError("TensorRT export requires params.onnx_path or a prior ONNX export")
                output_path = target.output_path
                if output_path is None:
                    output_path = str(
                        Path(context.config.project.artifact_dir)
                        / f"model_{index}.engine"
                    )
                result = build_tensorrt_engine(
                    onnx_path,
                    output_path,
                    precision=target.precision,
                    profiles=target.profiles,
                    trtexec_path=str(target.params.get("trtexec_path", "trtexec")),
                    extra_args=target.params.get("extra_args"),
                    timeout=target.params.get("timeout"),
                    dry_run=bool(target.params.get("dry_run", False)),
                    performance_thresholds=target.params.get("performance_thresholds"),
                )
                context.artifacts[f"export_{index}"] = result.engine_path
                if context.manifest is not None and result.checksum is not None:
                    context.manifest.add_artifact(
                        ArtifactRecord(
                            path=str(result.engine_path),
                            format="tensorrt",
                            runtime="tensorrt",
                            checksum=result.checksum,
                            metadata={
                                "precision": target.precision,
                                "dry_run": result.dry_run,
                                "command": result.command,
                                "performance": result.metadata.get("performance"),
                                "performance_threshold_report": result.metadata.get(
                                    "performance_threshold_report"
                                ),
                            },
                        )
                    )
                    threshold_report = result.metadata.get("performance_threshold_report")
                    if isinstance(threshold_report, Mapping):
                        context.manifest.add_metric(
                            MetricRecord(
                                name=f"export.{index}.tensorrt.performance",
                                value=result.metadata.get("performance"),
                                threshold=result.metadata.get("performance_thresholds"),
                                passed=bool(threshold_report.get("passed")),
                                metadata={"checks": threshold_report.get("checks", [])},
                            )
                        )
                exported.append(
                    {
                        "path": str(result.engine_path),
                        "format": "tensorrt",
                        "dry_run": result.dry_run,
                        "command": result.command,
                        "checksum": result.checksum,
                        "performance": result.metadata.get("performance"),
                        "performance_threshold_report": result.metadata.get(
                            "performance_threshold_report"
                        ),
                    }
                )
                continue

            if target.format == "executorch":
                output_path = target.output_path
                if output_path is None:
                    output_path = str(
                        Path(context.config.project.artifact_dir)
                        / f"model_{index}.pte"
                    )
                result = export_executorch_program(
                    model,
                    example_input,
                    output_path,
                    dry_run=bool(target.params.get("dry_run", False)),
                    metadata={"precision": target.precision},
                )
                context.artifacts[f"export_{index}"] = result.pte_path
                if context.manifest is not None and result.checksum is not None:
                    context.manifest.add_artifact(
                        ArtifactRecord(
                            path=str(result.pte_path),
                            format="executorch",
                            runtime="executorch",
                            checksum=result.checksum,
                            metadata=result.metadata | {"dry_run": result.dry_run},
                        )
                    )
                exported.append(
                    {
                        "path": str(result.pte_path),
                        "format": "executorch",
                        "dry_run": result.dry_run,
                        "checksum": result.checksum,
                    }
                )
                continue

            if target.format == "ncnn":
                onnx_path = target.params.get("onnx_path") or context.artifacts.get(
                    "last_onnx"
                )
                if onnx_path is None:
                    raise ValueError("ncnn export requires params.onnx_path or a prior ONNX export")
                param_path = target.output_path
                if param_path is None:
                    param_path = str(
                        Path(context.config.project.artifact_dir)
                        / f"model_{index}.param"
                    )
                bin_path = target.params.get("bin_path")
                if bin_path is None:
                    bin_path = str(Path(param_path).with_suffix(".bin"))
                result = export_ncnn_from_onnx(
                    onnx_path,
                    param_path,
                    bin_path,
                    onnx2ncnn_path=str(target.params.get("onnx2ncnn_path", "onnx2ncnn")),
                    extra_args=target.params.get("extra_args"),
                    timeout=target.params.get("timeout"),
                    dry_run=bool(target.params.get("dry_run", False)),
                )
                context.artifacts[f"export_{index}"] = result.output_paths
                if context.manifest is not None and result.checksums:
                    for output in result.output_paths:
                        context.manifest.add_artifact(
                            ArtifactRecord(
                                path=str(output),
                                format="ncnn",
                                runtime="ncnn",
                                checksum=result.checksums.get(str(output)),
                                metadata={
                                    "dry_run": result.dry_run,
                                    "command": result.command,
                                },
                            )
                        )
                exported.append(
                    {
                        "paths": [str(path) for path in result.output_paths],
                        "format": "ncnn",
                        "dry_run": result.dry_run,
                        "command": result.command,
                        "checksums": result.checksums,
                    }
                )
                continue

            if target.format == "mnn":
                onnx_path = target.params.get("onnx_path") or context.artifacts.get(
                    "last_onnx"
                )
                if onnx_path is None:
                    raise ValueError("MNN export requires params.onnx_path or a prior ONNX export")
                output_path = target.output_path
                if output_path is None:
                    output_path = str(
                        Path(context.config.project.artifact_dir)
                        / f"model_{index}.mnn"
                    )
                result = export_mnn_from_onnx(
                    onnx_path,
                    output_path,
                    converter_path=str(target.params.get("converter_path", "MNNConvert")),
                    framework=str(target.params.get("framework", "ONNX")),
                    extra_args=target.params.get("extra_args"),
                    timeout=target.params.get("timeout"),
                    dry_run=bool(target.params.get("dry_run", False)),
                )
                context.artifacts[f"export_{index}"] = result.output_paths[0]
                if context.manifest is not None and result.checksums:
                    output = result.output_paths[0]
                    context.manifest.add_artifact(
                        ArtifactRecord(
                            path=str(output),
                            format="mnn",
                            runtime="mnn",
                            checksum=result.checksums.get(str(output)),
                            metadata={
                                "dry_run": result.dry_run,
                                "command": result.command,
                            },
                        )
                    )
                exported.append(
                    {
                        "path": str(result.output_paths[0]),
                        "format": "mnn",
                        "dry_run": result.dry_run,
                        "command": result.command,
                        "checksums": result.checksums,
                    }
                )
                continue

            raise ValueError(f"Unsupported export format: {target.format}")
        context.metrics["export"] = {"artifacts": exported}
        return context


@register_pass("benchmark")
class BenchmarkPass:
    """Benchmark current model latency on the first validation batch."""

    name = "benchmark"

    def run(self, context: XQTContext) -> XQTContext:
        model = context.require_model()
        dataloader = context.data.get("validation")
        if dataloader is None:
            raise ValueError("validation data is required for benchmark")
        batch = next(iter(dataloader))
        inputs = _move_to_device(
            _split_inputs_from_batch(batch),
            torch.device(context.config.model.device),
        )

        def fn() -> object:
            return _call_model(model, inputs)

        report = benchmark_callable(
            fn,
            warmup=context.config.benchmark.warmup,
            iterations=context.config.benchmark.iterations,
            sync_cuda=context.config.benchmark.sync_cuda,
            device=context.config.model.device,
        )
        benchmark_metrics = report.to_dict()
        benchmark_metrics["latency"] = report.to_dict()
        memory_report = None
        if context.config.benchmark.measure_memory:
            memory_report = benchmark_memory(
                fn,
                iterations=1,
                sync_cuda=context.config.benchmark.sync_cuda,
                device=context.config.model.device,
            )
            benchmark_metrics["memory"] = memory_report.to_dict()
        context.metrics["benchmark"] = benchmark_metrics
        if context.manifest is not None:
            context.manifest.add_metric(
                MetricRecord(
                    name="benchmark.p50_ms",
                    value=report.p50_ms,
                )
            )
            if memory_report is not None:
                context.manifest.add_metric(
                    MetricRecord(
                        name="benchmark.memory.delta_bytes",
                        value=memory_report.delta_bytes,
                        metadata={"backend": memory_report.backend},
                    )
                )
        return context


@register_pass("write_reports")
class WriteReportsPass:
    """Write JSON and Markdown reports for collected metrics."""

    name = "write_reports"

    def run(self, context: XQTContext) -> XQTContext:
        artifact_dir = Path(context.config.project.artifact_dir)
        json_path = write_json_report(context.metrics, artifact_dir / "metrics.json")
        markdown_path = write_markdown_report(
            "XQT Metrics",
            {
                key: value if isinstance(value, dict) else {"value": value}
                for key, value in context.metrics.items()
            },
            artifact_dir / "metrics.md",
        )
        context.artifacts["metrics_json"] = json_path
        context.artifacts["metrics_markdown"] = markdown_path
        return context


__all__ = [
    "BaselineEvalPass",
    "BenchmarkPass",
    "DistillPass",
    "ExportPass",
    "LoadDataPass",
    "LoadModelPass",
    "PrunePass",
    "QuantPass",
    "WriteReportsPass",
]
