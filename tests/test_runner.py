from dataclasses import dataclass
from pathlib import Path

import torch

from xqt.core.artifact import load_manifest
from xqt.core.config import load_xqt_config
from xqt.core.registry import XQTRegistry
from xqt.core.types import XQTContext
from xqt.export import TensorRTBuildResult
from xqt.pipeline.runner import (
    build_pipeline_from_config,
    create_context,
    create_manifest,
    default_pass_names,
    enabled_pass_names,
    run_xqt_recipe,
)


@dataclass
class MarkPass:
    name: str
    key: str

    def run(self, context: XQTContext) -> XQTContext:
        context.metrics[self.key] = True
        return context


def test_enabled_pass_names_follow_default_order() -> None:
    config = load_xqt_config(
        {
            "compression": {
                "axes": ["precision", "sparsity", "steps"],
                "quant": {"enabled": True},
                "prune": {"enabled": True},
                "diffusion_distill": {"enabled": True},
            }
        }
    )

    assert enabled_pass_names(config) == ["prune", "quant", "diffusion_distill"]


def test_default_pass_names_include_core_pipeline_and_enabled_compression() -> None:
    config = load_xqt_config(
        {
            "compression": {
                "quant": {"enabled": True},
                "prune": {"enabled": False},
            }
        }
    )

    assert default_pass_names(config) == [
        "load_model",
        "load_data",
        "baseline_eval",
        "quant",
        "benchmark",
        "write_reports",
    ]


def test_default_pass_names_include_export_when_targets_exist() -> None:
    config = load_xqt_config(
        {
            "export": {
                "targets": [
                    {
                        "format": "onnx",
                        "output_path": "model.onnx",
                    }
                ]
            }
        }
    )

    assert "export" in default_pass_names(config)


def test_create_context_initializes_manifest_from_config(tmp_path) -> None:
    checkpoint = tmp_path / "source.pt"
    checkpoint.write_bytes(b"checkpoint")
    config = {
        "project": {
            "name": "runner_context",
            "artifact_dir": str(tmp_path / "artifacts"),
        },
        "model": {
            "checkpoint": str(checkpoint),
            "device": "cpu",
        },
        "compression": {
            "axes": ["precision"],
        },
    }

    context = create_context(config, model="model")

    assert context.config.project.name == "runner_context"
    assert context.model == "model"
    assert context.device == "cpu"
    assert context.manifest is not None
    assert context.manifest.project_name == "runner_context"
    assert context.manifest.source_checkpoint == str(checkpoint)
    assert context.manifest.source_checksum is not None
    assert context.manifest.compression_axes == ["precision"]


def test_build_pipeline_from_config_uses_enabled_passes_and_registry() -> None:
    registry = XQTRegistry("TEST_PASS")
    registry.register("prune")(lambda: MarkPass(name="prune", key="ran_prune"))
    registry.register("quant")(lambda: MarkPass(name="quant", key="ran_quant"))
    config = load_xqt_config(
        {
            "compression": {
                "prune": {"enabled": True},
                "quant": {"enabled": True},
            }
        }
    )
    context = XQTContext(config=config, manifest=create_manifest(config))

    pipeline = build_pipeline_from_config(
        config,
        pass_names=["prune", "quant"],
        pass_registry=registry,
    )
    output = pipeline.run(context)

    assert output.metrics == {"ran_prune": True, "ran_quant": True}
    assert output.manifest is not None
    assert output.manifest.passes == ["prune", "quant"]


def test_run_xqt_recipe_runs_passes_and_writes_manifest(tmp_path) -> None:
    registry = XQTRegistry("TEST_PASS")
    registry.register("quant")(lambda: MarkPass(name="quant", key="ran_quant"))
    config_path = tmp_path / "recipe.yaml"
    artifact_dir = tmp_path / "artifacts"
    config_path.write_text(
        f"""
project:
  name: runner_recipe
  artifact_dir: {artifact_dir}
compression:
  axes: [precision]
  quant:
    enabled: true
benchmark:
  warmup: 0
  iterations: 1
""",
        encoding="utf-8",
    )

    context = run_xqt_recipe(
        config_path,
        pass_names=["quant"],
        pass_registry=registry,
    )

    assert context.metrics == {"ran_quant": True}
    assert context.artifacts["manifest"] == artifact_dir / "manifest.json"
    manifest = load_manifest(context.artifacts["manifest"])
    assert manifest["project_name"] == "runner_recipe"
    assert manifest["passes"] == ["quant"]
    assert manifest["compression_axes"] == ["precision"]


def test_smoke_cpu_recipe_loads_and_runs_empty_pipeline(tmp_path) -> None:
    recipe_path = "xqt/recipes/smoke_cpu.yaml"

    context = run_xqt_recipe(
        recipe_path,
        pass_names=[],
        write_manifest=False,
    )

    assert context.config.project.name == "xqt_smoke_cpu"
    assert context.config.model.device == "cpu"
    assert context.config.compression.axes == ["precision", "sparsity"]
    assert context.manifest is not None
    assert context.manifest.passes == []


def test_smoke_cpu_recipe_runs_builtin_default_pipeline(tmp_path) -> None:
    config = load_xqt_config(
        "xqt/recipes/smoke_cpu.yaml",
        overrides={
            "project": {
                "artifact_dir": str(tmp_path / "smoke_artifacts"),
            }
        },
    )

    context = run_xqt_recipe(config)

    assert context.model is not None
    assert "validation" in context.data
    assert "baseline" in context.metrics
    assert "prune" in context.metrics
    assert "benchmark" in context.metrics
    assert context.metrics["benchmark"]["latency"]["iterations"] == 1
    assert context.metrics["benchmark"]["memory"]["backend"] == "process_rss"
    assert context.metrics["prune"]["sparsity"] == 0.5
    assert context.artifacts["metrics_json"].is_file()
    assert context.artifacts["metrics_markdown"].is_file()
    assert context.artifacts["manifest"].is_file()
    assert context.manifest is not None
    assert context.manifest.passes == [
        "load_model",
        "load_data",
        "baseline_eval",
        "prune",
        "export",
        "benchmark",
        "write_reports",
    ]
    assert context.manifest.artifacts[0].format == "torch_export"
    assert any(
        metric.name == "benchmark.memory.delta_bytes"
        for metric in context.manifest.metrics
    )


def test_builtin_pipeline_exports_onnx_when_configured(tmp_path) -> None:
    output_path = tmp_path / "model.onnx"
    config = load_xqt_config(
        "xqt/recipes/smoke_cpu.yaml",
        overrides={
            "project": {
                "artifact_dir": str(tmp_path / "export_artifacts"),
            },
            "compression": {
                "prune": {"enabled": False},
            },
            "export": {
                "targets": [
                    {
                        "format": "onnx",
                        "output_path": str(output_path),
                        "opset": 18,
                        "params": {
                            "dynamo": True,
                            "runtime_diff": True,
                        },
                    }
                ]
            },
        },
    )

    context = run_xqt_recipe(config)

    assert output_path.is_file()
    assert "export" in context.metrics
    assert context.metrics["export"]["artifacts"][0]["checked"] is True
    assert context.metrics["export"]["artifacts"][0]["output_diff"]["allclose"] is True
    assert context.manifest is not None
    assert "export" in context.manifest.passes
    assert context.manifest.artifacts[0].format == "onnx"


def test_builtin_pipeline_supports_tensorrt_dry_run_after_onnx(tmp_path) -> None:
    onnx_path = tmp_path / "model.onnx"
    engine_path = tmp_path / "model.engine"
    config = load_xqt_config(
        "xqt/recipes/smoke_cpu.yaml",
        overrides={
            "project": {
                "artifact_dir": str(tmp_path / "trt_artifacts"),
            },
            "compression": {
                "prune": {"enabled": False},
            },
            "export": {
                "targets": [
                    {
                        "format": "onnx",
                        "output_path": str(onnx_path),
                        "params": {
                            "runtime_diff": False,
                        },
                    },
                    {
                        "format": "tensorrt",
                        "output_path": str(engine_path),
                        "precision": "fp16",
                        "params": {
                            "dry_run": True,
                        },
                    },
                ]
            },
        },
    )

    context = run_xqt_recipe(config)

    artifacts = context.metrics["export"]["artifacts"]
    assert artifacts[0]["format"] == "onnx"
    assert artifacts[1]["format"] == "tensorrt"
    assert artifacts[1]["dry_run"] is True
    assert "--fp16" in artifacts[1]["command"]


def test_builtin_pipeline_exports_torch_native_formats(tmp_path) -> None:
    config = load_xqt_config(
        "xqt/recipes/smoke_cpu.yaml",
        overrides={
            "project": {
                "artifact_dir": str(tmp_path / "torch_native_artifacts"),
            },
            "compression": {
                "prune": {"enabled": False},
            },
            "export": {
                "targets": [
                    {
                        "format": "torch_export",
                        "output_path": str(tmp_path / "model.pt2"),
                    },
                    {
                        "format": "torchscript",
                        "output_path": str(tmp_path / "model.pt"),
                        "params": {"method": "trace"},
                    },
                ]
            },
        },
    )

    context = run_xqt_recipe(config)

    artifacts = context.metrics["export"]["artifacts"]
    assert artifacts[0]["format"] == "torch_export"
    assert artifacts[0]["checked"] is True
    assert artifacts[0]["output_diff"]["allclose"] is True
    assert artifacts[1]["format"] == "torchscript"
    assert artifacts[1]["output_diff"]["allclose"] is True
    assert context.manifest is not None
    assert [item.format for item in context.manifest.artifacts] == [
        "torch_export",
        "torchscript",
    ]


def test_builtin_pipeline_records_tensorrt_performance_thresholds(
    tmp_path,
    monkeypatch,
) -> None:
    onnx_path = tmp_path / "model.onnx"
    engine_path = tmp_path / "model.engine"
    engine_path.write_bytes(b"engine")

    def fake_export_onnx(*args, **kwargs):
        onnx_path.write_bytes(b"onnx")

        class Result:
            path = onnx_path
            opset = 18
            checked = True
            checksum = "onnx_checksum"
            output_diff = None

        return Result()

    def fake_build_tensorrt_engine(*args, **kwargs):
        assert kwargs["performance_thresholds"] == {
            "throughput_qps_min": 100.0,
            "latency_p99_ms_max": 5.0,
        }
        return TensorRTBuildResult(
            engine_path=engine_path,
            command=["trtexec", "--onnx=model.onnx"],
            returncode=0,
            checksum="engine_checksum",
            metadata={
                "performance": {
                    "throughput_qps": 200.0,
                    "latency_ms": {"p99": 4.0},
                },
                "performance_thresholds": kwargs["performance_thresholds"],
                "performance_threshold_report": {
                    "passed": True,
                    "checks": [
                        {
                            "name": "throughput_qps_min",
                            "metric_path": "throughput_qps",
                            "value": 200.0,
                            "threshold": 100.0,
                            "direction": ">=",
                            "passed": True,
                        }
                    ],
                },
            },
        )

    monkeypatch.setattr("xqt.pipeline.passes.export_onnx", fake_export_onnx)
    monkeypatch.setattr(
        "xqt.pipeline.passes.build_tensorrt_engine",
        fake_build_tensorrt_engine,
    )
    config = load_xqt_config(
        "xqt/recipes/smoke_cpu.yaml",
        overrides={
            "project": {"artifact_dir": str(tmp_path / "trt_metrics")},
            "compression": {"prune": {"enabled": False}},
            "export": {
                "targets": [
                    {
                        "format": "onnx",
                        "output_path": str(onnx_path),
                        "params": {"runtime_diff": False},
                    },
                    {
                        "format": "tensorrt",
                        "output_path": str(engine_path),
                        "precision": "fp16",
                        "params": {
                            "performance_thresholds": {
                                "throughput_qps_min": 100.0,
                                "latency_p99_ms_max": 5.0,
                            },
                        },
                    },
                ]
            },
        },
    )

    context = run_xqt_recipe(config)

    tensorrt_artifact = context.metrics["export"]["artifacts"][1]
    assert tensorrt_artifact["performance"]["throughput_qps"] == 200.0
    assert tensorrt_artifact["performance_threshold_report"]["passed"] is True
    assert context.manifest is not None
    metric = next(
        item
        for item in context.manifest.metrics
        if item.name == "export.1.tensorrt.performance"
    )
    assert metric.name == "export.1.tensorrt.performance"
    assert metric.passed is True


def test_builtin_distill_pass_runs_with_injected_teacher(tmp_path) -> None:
    teacher = torch.nn.Linear(4, 2)
    config = load_xqt_config(
        "xqt/recipes/smoke_cpu.yaml",
        overrides={
            "project": {
                "artifact_dir": str(tmp_path / "distill_artifacts"),
            },
            "data": {
                "train": {
                    "target": "synthetic_classification",
                    "sample_limit": 4,
                    "batch_size": 2,
                }
            },
            "compression": {
                "distill": {
                    "enabled": True,
                    "temperature": 2.0,
                    "alpha": 0.5,
                    "params": {
                        "max_steps": 1,
                        "lr": 0.01,
                    },
                },
                "prune": {"enabled": False},
            },
        },
    )

    context = run_xqt_recipe(config, teacher=teacher)

    assert "distill" in context.metrics
    assert context.metrics["distill"]["steps"] == 1
    assert context.metrics["distill"]["samples"] == 2
    assert context.manifest is not None
    assert "distill" in context.manifest.passes


def test_builtin_quant_pass_supports_onnxruntime_qdq(monkeypatch, tmp_path) -> None:
    output_path = tmp_path / "model_qdq.onnx"

    def fake_quantize_onnx_qdq_static(
        onnx_path,
        output_path_arg,
        calibration_data,
        **kwargs,
    ):
        from xqt.quant.onnx_qdq import ONNXQDQQuantizationResult

        del onnx_path, calibration_data
        output = Path(output_path_arg)
        output.write_bytes(b"qdq")
        return ONNXQDQQuantizationResult(
            path=output,
            source_path=output,
            checksum="checksum",
            calibration_samples=1,
            metadata={"input_names": list(kwargs["input_names"])},
        )

    monkeypatch.setattr(
        "xqt.pipeline.passes.quantize_onnx_qdq_static",
        fake_quantize_onnx_qdq_static,
    )
    config = load_xqt_config(
        "xqt/recipes/smoke_cpu.yaml",
        overrides={
            "project": {
                "artifact_dir": str(tmp_path / "qdq_artifacts"),
            },
            "compression": {
                "prune": {"enabled": False},
                "quant": {
                    "enabled": True,
                    "backend": "onnxruntime_qdq",
                    "policy": {
                        "output_path": str(output_path),
                        "input_names": ["input"],
                        "runtime_diff": False,
                    },
                },
            },
        },
    )

    context = run_xqt_recipe(config)

    assert context.artifacts["quant_onnx"] == output_path
    assert context.artifacts["last_onnx"] == output_path
    assert context.metrics["quant"]["backend"] == "onnxruntime_qdq"
    assert context.metrics["quant"]["calibration_samples"] == 1


def test_image_recipes_load_and_resnet_qdq_smoke_runs(monkeypatch, tmp_path) -> None:
    vit_config = load_xqt_config("xqt/recipes/image_vit_torchao_fp8.yaml")
    assert vit_config.project.name == "image_vit_torchao_fp8"
    assert vit_config.data.validation is not None
    assert vit_config.data.validation.params["input_shape"] == [3, 224, 224]
    assert vit_config.model.device == "cuda:0"

    def fake_quantize_onnx_qdq_static(
        onnx_path,
        output_path_arg,
        calibration_data,
        **kwargs,
    ):
        from xqt.quant.onnx_qdq import ONNXQDQQuantizationResult

        del onnx_path, calibration_data
        output = Path(output_path_arg)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"qdq")
        return ONNXQDQQuantizationResult(
            path=output,
            source_path=output,
            checksum="checksum",
            calibration_samples=1,
            metadata={"input_names": list(kwargs["input_names"])},
        )

    monkeypatch.setattr(
        "xqt.pipeline.passes.quantize_onnx_qdq_static",
        fake_quantize_onnx_qdq_static,
    )
    resnet_config = load_xqt_config(
        "xqt/recipes/image_resnet_onnx_qdq_int8.yaml",
        overrides={
            "project": {"artifact_dir": str(tmp_path / "resnet_qdq")},
            "compression": {
                "quant": {
                    "policy": {
                        "output_path": str(tmp_path / "resnet_qdq" / "model_qdq.onnx"),
                    }
                }
            },
            "export": {
                "targets": [
                    {
                        "format": "tensorrt",
                        "output_path": str(tmp_path / "resnet_qdq" / "model.engine"),
                        "precision": "int8",
                        "params": {"dry_run": True},
                    }
                ]
            },
            "benchmark": {"warmup": 0, "iterations": 1},
        },
    )

    context = run_xqt_recipe(resnet_config)

    assert context.config.project.name == "image_resnet_onnx_qdq_int8"
    assert context.metrics["quant"]["backend"] == "onnxruntime_qdq"
    assert context.metrics["export"]["artifacts"][0]["format"] == "tensorrt"
    assert context.metrics["export"]["artifacts"][0]["dry_run"] is True


def test_image_vit_torchao_fp8_recipe_runs_on_cuda_when_available(tmp_path) -> None:
    if not torch.cuda.is_available():
        return

    config = load_xqt_config(
        "xqt/recipes/image_vit_torchao_fp8.yaml",
        overrides={
            "project": {"artifact_dir": str(tmp_path / "vit_fp8_cuda")},
            "benchmark": {"warmup": 1, "iterations": 2},
        },
    )

    context = run_xqt_recipe(config)

    assert context.metrics["quant"]["backend"] == "torchao"
    assert context.metrics["quant"]["strategy"] == "fp8_dynamic"
    assert context.metrics["quant"]["quantized_module_count"] > 0
    assert context.metrics["benchmark"]["memory"]["backend"] == "cuda"
    assert context.artifacts["manifest"].is_file()


def test_prune_finetune_recipe_runs_schedule_with_teacher(tmp_path) -> None:
    teacher = torch.nn.Linear(4, 2)
    config = load_xqt_config(
        "xqt/recipes/prune_finetune_cpu.yaml",
        overrides={
            "project": {"artifact_dir": str(tmp_path / "prune_finetune")},
        },
    )

    context = run_xqt_recipe(config, teacher=teacher)

    assert context.metrics["prune"]["final_sparsity"] == 0.5
    assert len(context.metrics["prune"]["steps"]) == 2
    assert context.metrics["prune"]["steps"][0]["distillation"]["steps"] == 1
    assert context.metrics["prune"]["steps"][1]["distillation"]["steps"] == 1
    assert context.manifest is not None
    assert "prune" in context.manifest.passes


def test_cifar100_qdq_recipe_loads_real_local_data_and_quantizes(monkeypatch, tmp_path) -> None:
    def fake_build_torchvision_image_classification_loader(spec):
        del spec
        inputs = torch.randn(2, 3, 224, 224)
        targets = torch.tensor([0, 1], dtype=torch.long)
        return torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(inputs, targets),
            batch_size=1,
        )

    def fake_quantize_onnx_qdq_static(
        onnx_path,
        output_path_arg,
        calibration_data,
        **kwargs,
    ):
        from xqt.quant.onnx_qdq import ONNXQDQQuantizationResult

        first_batch = next(iter(calibration_data))
        assert first_batch[0].shape == (1, 3, 224, 224)
        del onnx_path
        output = Path(output_path_arg)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"qdq")
        return ONNXQDQQuantizationResult(
            path=output,
            source_path=output,
            checksum="checksum",
            calibration_samples=1,
            metadata={"input_names": list(kwargs["input_names"])},
        )

    monkeypatch.setattr(
        "xqt.pipeline.passes.build_torchvision_image_classification_loader",
        fake_build_torchvision_image_classification_loader,
    )
    monkeypatch.setattr(
        "xqt.pipeline.passes.quantize_onnx_qdq_static",
        fake_quantize_onnx_qdq_static,
    )
    config = load_xqt_config(
        "xqt/recipes/image_resnet_cifar100_qdq_cpu.yaml",
        overrides={
            "project": {"artifact_dir": str(tmp_path / "cifar_qdq")},
            "compression": {
                "quant": {
                    "policy": {
                        "output_path": str(tmp_path / "cifar_qdq" / "model_qdq.onnx"),
                    }
                }
            },
        },
    )

    context = run_xqt_recipe(config)

    assert "calibration" in context.data
    assert context.metrics["baseline"]["samples"] == 2
    assert context.metrics["quant"]["backend"] == "onnxruntime_qdq"
