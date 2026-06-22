from dataclasses import dataclass
from pathlib import Path

import onnx
import pytest
import torch

from xqt.core.artifact import load_manifest
from xqt.core.config import load_xqt_config
from xqt.core.errors import XQTPipelineError
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
            },
            "operator_optimization": {
                "enabled": True,
                "targets": [
                    {
                        "name": "model",
                        "backend": "torch_compile",
                    }
                ],
            },
        }
    )

    assert enabled_pass_names(config) == [
        "prune",
        "quant",
        "operator_optimization",
        "diffusion_distill",
    ]


def test_default_pass_names_include_core_pipeline_and_enabled_compression() -> None:
    config = load_xqt_config(
        {
            "compression": {
                "quant": {"enabled": True},
                "prune": {"enabled": False},
            },
            "operator_optimization": {
                "enabled": True,
                "targets": [
                    {
                        "name": "model",
                        "backend": "torch_compile",
                    }
                ],
            },
        }
    )

    assert default_pass_names(config) == [
        "load_model",
        "load_data",
        "baseline_eval",
        "quant",
        "operator_optimization",
        "benchmark",
        "write_reports",
    ]


def test_default_pass_names_include_analyze_when_enabled() -> None:
    config = load_xqt_config(
        {
            "analysis": {"enabled": True},
            "compression": {
                "quant": {"enabled": True},
            },
        }
    )

    assert default_pass_names(config) == [
        "load_model",
        "load_data",
        "baseline_eval",
        "quant",
        "analyze",
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
    assert context.reference_model == "model"
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
    recipe_path = "xqt/recipes/smoke/smoke_cpu.yaml"

    context = run_xqt_recipe(
        recipe_path,
        pass_names=[],
        write_manifest=False,
    )

    assert context.config.project.name == "smoke_cpu"
    assert context.config.model.device == "cpu"
    assert context.config.compression.axes == ["precision", "sparsity"]
    assert context.manifest is not None
    assert context.manifest.passes == []


def test_smoke_cpu_recipe_runs_builtin_default_pipeline(tmp_path) -> None:
    config = load_xqt_config(
        "xqt/recipes/smoke/smoke_cpu.yaml",
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


def test_yolo_detection_smoke_recipe_runs_builtin_default_pipeline(tmp_path) -> None:
    config = load_xqt_config(
        "xqt/recipes/detection/yolo_detection_smoke.yaml",
        overrides={
            "project": {"artifact_dir": str(tmp_path / "yolo_detection_smoke")},
        },
    )

    context = run_xqt_recipe(config)

    assert context.metrics["baseline"]["samples"] == 2
    assert context.metrics["baseline"]["metrics"]["map50"] == pytest.approx(1.0)
    assert context.metrics["prune"]["sparsity"] >= 0.0
    assert context.metrics["export"]["artifacts"][0]["format"] == "onnx"
    assert context.metrics["benchmark"]["latency"]["iterations"] == 1
    assert context.artifacts["manifest"].is_file()
    assert context.manifest is not None
    assert context.manifest.task is not None
    assert context.manifest.task["type"] == "detection"


def test_operator_compile_smoke_recipe_records_manifest_metrics(tmp_path) -> None:
    config = load_xqt_config(
        "xqt/recipes/operator/torch_compile/operator_compile_smoke_cpu.yaml",
        overrides={
            "project": {"artifact_dir": str(tmp_path / "operator_compile_smoke")},
        },
    )

    context = run_xqt_recipe(config)

    assert context.manifest is not None
    assert "operator_optimization" in context.manifest.passes
    assert any(
        metric.name == "operator_optimization.model.applied"
        for metric in context.manifest.metrics
    )
    assert any(
        artifact.metadata.get("kind") == "operator_optimization_report"
        for artifact in context.manifest.artifacts
    )
    report_path = context.artifacts["operator_optimization_report"]
    report_data = load_manifest(report_path)
    assert "candidates" in report_data
    assert "fx" in report_data["candidates"]


def test_operator_quant_torchao_compile_recipe_runs(tmp_path) -> None:
    config = load_xqt_config(
        "xqt/recipes/quant/int8/operator_quant_torchao_compile.yaml",
        overrides={
            "project": {"artifact_dir": str(tmp_path / "operator_quant_torchao_compile")},
        },
    )

    context = run_xqt_recipe(config)

    assert context.metrics["quant"]["backend"] == "torchao"
    assert "operator_optimization" in context.metrics
    assert context.metrics["operator_optimization"]["target_count"] == 1
    assert context.metrics["operator_optimization"]["targets"][0]["backend"] == "torch_compile"


def test_operator_qdq_export_guard_recipe_skips_rewrite_and_preserves_export(
    monkeypatch,
    tmp_path,
) -> None:
    def fake_quantize_onnx_qdq_static(
        onnx_path,
        output_path_arg,
        calibration_data,
        **kwargs,
    ):
        from xqt.quant.onnx_qdq import ONNXQDQQuantizationResult

        del onnx_path, calibration_data, kwargs
        output = Path(output_path_arg)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"qdq")
        return ONNXQDQQuantizationResult(
            path=output,
            source_path=output,
            checksum="checksum",
            calibration_samples=1,
            metadata={},
        )

    def fake_export_onnx(*args, **kwargs):
        from xqt.export.onnx_exporter import ONNXExportResult

        output = Path(args[2])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"onnx")
        return ONNXExportResult(
            path=output,
            opset=kwargs.get("opset"),
            checksum="onnx_checksum",
            checked=True,
            metadata={"input_names": ["input"]},
        )

    monkeypatch.setattr(
        "xqt.pipeline.passes.quantize_onnx_qdq_static",
        fake_quantize_onnx_qdq_static,
    )
    monkeypatch.setattr("xqt.pipeline.passes.export_onnx", fake_export_onnx)

    config = load_xqt_config(
        "xqt/recipes/quant/int8/operator_qdq_export_guard.yaml",
        overrides={
            "project": {"artifact_dir": str(tmp_path / "operator_qdq_export_guard")},
            "export": {
                "targets": [
                    {
                        "format": "onnx",
                        "output_path": str(tmp_path / "operator_qdq_export_guard" / "model.onnx"),
                        "params": {"runtime_diff": False},
                    }
                ]
            },
        },
    )

    context = run_xqt_recipe(config)

    assert context.metrics["quant"]["backend"] == "onnxruntime_qdq"
    target = context.metrics["operator_optimization"]["targets"][0]
    assert target["applied"] is False
    assert "onnxruntime_qdq" in target["skip_reason"]
    assert context.metrics["export"]["artifacts"][0]["format"] == "onnx"


def test_cnn_structured_prune_recipe_runs_default_pipeline(tmp_path) -> None:
    config = load_xqt_config(
        "xqt/recipes/prune/structured/cnn_structured_prune.yaml",
        overrides={
            "project": {
                "artifact_dir": str(tmp_path / "cnn_structured_prune"),
            }
        },
    )

    context = run_xqt_recipe(config)

    assert context.model is not None
    assert "validation" in context.data
    assert "baseline" in context.metrics
    assert "prune" in context.metrics
    assert "benchmark" in context.metrics
    assert context.metrics["prune"]["method"] == "structured"
    assert context.metrics["prune"]["granularity"] == "channel"
    assert context.metrics["prune"]["sparsity"] == pytest.approx(0.25)
    assert context.metrics["prune"]["parameter_count_after"] < context.metrics["prune"]["parameter_count_before"]
    assert len(context.metrics["prune"]["actions"]) == 2
    assert context.metrics["prune"]["benchmark_status"]["attempted"] is True
    assert context.metrics["prune"]["benchmark_status"]["latency"]["iterations"] == 1
    assert context.artifacts["metrics_json"].is_file()
    assert context.artifacts["metrics_markdown"].is_file()
    assert context.artifacts["manifest"].is_file()
    assert context.manifest is not None
    assert "prune" in context.manifest.passes
    assert any(metric.name == "prune.sparsity" for metric in context.manifest.metrics)


def test_cnn_structured_prune_recipe_exports_onnx_with_reduced_channel_shapes(tmp_path) -> None:
    output_path = tmp_path / "cnn_structured_prune" / "model.onnx"
    config = load_xqt_config(
        "xqt/recipes/prune/structured/cnn_structured_prune.yaml",
        overrides={
            "project": {"artifact_dir": str(tmp_path / "cnn_structured_prune")},
            "export": {
                "targets": [
                    {
                        "format": "onnx",
                        "output_path": str(output_path),
                        "params": {"runtime_diff": False},
                    }
                ]
            },
        },
    )

    context = run_xqt_recipe(config)

    assert context.metrics["export"]["artifacts"][0]["format"] == "onnx"
    assert context.metrics["prune"]["export_status"]["attempted"] is True
    assert context.metrics["prune"]["export_status"]["formats"] == ["onnx"]
    assert context.metrics["prune"]["export_status"]["artifact_count"] == 1
    model = onnx.load(output_path)
    initializers = {initializer.name: list(initializer.dims) for initializer in model.graph.initializer}
    assert initializers["features.0.weight"][0] < 8
    assert initializers["features.3.weight"][1] < 8


def test_load_data_builds_prompts_and_records_summary(tmp_path) -> None:
    config = load_xqt_config(
        "xqt/recipes/smoke/smoke_cpu.yaml",
        overrides={
            "project": {
                "artifact_dir": str(tmp_path / "prompt_artifacts"),
            },
            "data": {
                "prompts": {
                    "target": "prompt_list",
                    "sample_limit": 2,
                    "params": {
                        "prompts": [
                            {
                                "prompt": "a castle",
                                "seed": 7,
                                "image": "inputs/castle.png",
                                "mask": "inputs/castle-mask.png",
                            },
                            {
                                "prompt": "a forest",
                                "negative_prompt": "fog",
                                "reference_image": "refs/forest.png",
                                "latent_cache_key": "forest-latent",
                            },
                            {"prompt": "unused"},
                        ]
                    },
                }
            },
        },
    )

    context = run_xqt_recipe(config, pass_names=["load_model", "load_data"])

    assert "prompts" in context.data
    assert len(context.data["prompts"]) == 2
    assert context.data["prompts"][0].prompt == "a castle"
    assert context.data["prompts"][0].condition_image == "inputs/castle.png"
    assert context.data["prompts"][0].condition_mask == "inputs/castle-mask.png"
    assert context.metrics["prompts"]["count"] == 2
    assert context.metrics["prompts"]["has_seed"] is True
    assert context.metrics["prompts"]["has_condition_image"] is True
    assert context.metrics["prompts"]["has_condition_mask"] is True
    assert context.metrics["prompts"]["has_reference_image"] is True
    assert context.metrics["prompts"]["has_latent_cache_key"] is True
    assert context.manifest is not None
    prompt_metric = next(
        metric for metric in context.manifest.metrics if metric.name == "prompts.count"
    )
    assert prompt_metric.value == 2


def test_smoke_cpu_recipe_runs_analysis_when_enabled(tmp_path) -> None:
    config = load_xqt_config(
        "xqt/recipes/quant/fp8/image_vit_torchao_fp8.yaml",
        overrides={
            "project": {
                "artifact_dir": str(tmp_path / "analysis_artifacts"),
            },
            "model": {
                "device": "cpu",
            },
            "compression": {
                "quant": {
                    "strategy": "dynamic_int8",
                    "policy": {
                        "strategy": "dynamic_int8",
                    },
                },
            },
            "benchmark": {
                "warmup": 0,
                "iterations": 1,
            },
            "analysis": {
                "enabled": True,
                "top_k": 2,
            },
        },
    )

    context = run_xqt_recipe(config)

    assert "analysis" in context.metrics
    assert context.metrics["analysis"]["metrics"] == [
        "max_abs",
        "mean_abs",
        "cosine_similarity",
    ]
    assert context.metrics["analysis"]["record_count"] >= 1
    assert len(context.metrics["analysis"]["records"]) == 2
    assert len(context.metrics["analysis"]["activation_drift"]) >= 1
    assert len(context.metrics["analysis"]["importance"]) >= 1
    assert len(context.metrics["analysis"]["prune_candidates"]) >= 1
    assert len(context.metrics["analysis"]["recommended_high_precision_modules"]) >= 1
    assert len(context.metrics["analysis"]["pareto_points"]) == 1
    assert context.artifacts["analysis_json"].is_file()
    assert context.artifacts["analysis_csv"].is_file()
    assert context.artifacts["analysis_markdown"].is_file()
    assert context.manifest is not None
    assert "analyze" in context.manifest.passes
    assert any(metric.name == "analysis.record_count" for metric in context.manifest.metrics)


def test_smoke_cpu_recipe_analysis_reports_teacher_student_alignment(tmp_path) -> None:
    teacher = torch.nn.Linear(4, 2)
    model = torch.nn.Linear(4, 2)
    model.load_state_dict(teacher.state_dict())
    with torch.no_grad():
        model.bias.add_(0.25)
    config = load_xqt_config(
        "xqt/recipes/smoke/smoke_cpu.yaml",
        overrides={
            "project": {
                "artifact_dir": str(tmp_path / "analysis_teacher_artifacts"),
            },
            "compression": {
                "prune": {"enabled": False},
            },
            "analysis": {
                "enabled": True,
                "top_k": 1,
            },
        },
    )

    context = run_xqt_recipe(config, model=model, teacher=teacher)

    assert len(context.metrics["analysis"]["teacher_student_alignment"]) == 1


def test_builtin_pipeline_exports_onnx_when_configured(tmp_path) -> None:
    output_path = tmp_path / "model.onnx"
    config = load_xqt_config(
        "xqt/recipes/smoke/smoke_cpu.yaml",
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


def test_builtin_pipeline_exports_tuple_inputs_to_onnx(monkeypatch, tmp_path) -> None:
    class PairModel(torch.nn.Module):
        def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
            return left + right

    export_calls = {}
    diff_calls = {}

    def fake_export_onnx(*args, **kwargs):
        from xqt.export.onnx_exporter import ONNXExportResult

        export_calls["example_input"] = args[1]
        export_calls["input_names"] = kwargs.get("input_names")
        output = Path(args[2])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"onnx")
        return ONNXExportResult(
            path=output,
            opset=kwargs.get("opset"),
            checksum="onnx_checksum",
            checked=True,
            metadata={"input_names": list(kwargs.get("input_names") or [])},
        )

    def fake_compare_onnxruntime_outputs(
        onnx_path,
        reference_output,
        example_input,
        **kwargs,
    ):
        from xqt.eval.compare import compare_tensors

        del onnx_path
        diff_calls["example_input"] = example_input
        diff_calls["input_names"] = kwargs.get("input_names")
        assert isinstance(example_input, tuple)
        return compare_tensors(reference_output, reference_output.clone())

    monkeypatch.setattr("xqt.pipeline.passes.export_onnx", fake_export_onnx)
    monkeypatch.setattr(
        "xqt.pipeline.passes.compare_onnxruntime_outputs",
        fake_compare_onnxruntime_outputs,
    )
    validation_loader = torch.utils.data.DataLoader(
        [
            (
                torch.randn(2, 4),
                torch.randn(2, 4),
                torch.tensor([0, 1]),
            )
        ],
        batch_size=None,
    )
    config = load_xqt_config(
        {
            "project": {
                "name": "tuple_onnx_export",
                "artifact_dir": str(tmp_path / "artifacts"),
            },
            "model": {"device": "cpu"},
            "export": {
                "targets": [
                    {
                        "format": "onnx",
                        "output_path": str(tmp_path / "artifacts" / "model.onnx"),
                        "params": {"runtime_diff": True},
                    }
                ]
            },
        }
    )

    context = run_xqt_recipe(
        config,
        model=PairModel(),
        data={"validation": validation_loader},
        pass_names=["export"],
        write_manifest=False,
    )

    assert isinstance(export_calls["example_input"], tuple)
    assert export_calls["input_names"] == ["input_0", "input_1"]
    assert diff_calls["input_names"] == ["input_0", "input_1"]
    assert context.metrics["export"]["artifacts"][0]["output_diff"]["allclose"] is True


def test_builtin_pipeline_supports_tensorrt_dry_run_after_onnx(tmp_path) -> None:
    onnx_path = tmp_path / "model.onnx"
    engine_path = tmp_path / "model.engine"
    config = load_xqt_config(
        "xqt/recipes/smoke/smoke_cpu.yaml",
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


def test_builtin_pipeline_supports_openvino_dry_run_after_onnx(tmp_path) -> None:
    onnx_path = tmp_path / "model.onnx"
    xml_path = tmp_path / "model.xml"
    artifact_dir = tmp_path / "openvino_artifacts"
    config = load_xqt_config(
        "xqt/recipes/smoke/smoke_cpu.yaml",
        overrides={
            "project": {
                "artifact_dir": str(artifact_dir),
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
                        "format": "openvino",
                        "output_path": str(xml_path),
                        "precision": "fp32",
                        "params": {
                            "dry_run": True,
                            "input_shape": [1, 4],
                        },
                    },
                ]
            },
        },
    )

    context = run_xqt_recipe(config)

    artifacts = context.metrics["export"]["artifacts"]
    assert artifacts[0]["format"] == "onnx"
    assert artifacts[1]["format"] == "openvino"
    assert artifacts[1]["dry_run"] is True
    assert artifacts[1]["input_shape"] == [1, 4]
    assert artifacts[1]["command"][0] == "openvino.convert_model"

    manifest = load_manifest(artifact_dir / "manifest.json")
    openvino_artifacts = [
        item for item in manifest["artifacts"] if item["format"] == "openvino"
    ]
    assert openvino_artifacts
    assert openvino_artifacts[0]["metadata"]["dry_run"] is True
    assert openvino_artifacts[0]["metadata"]["input_shape"] == [1, 4]


def test_builtin_pipeline_exports_torch_native_formats(tmp_path) -> None:
    config = load_xqt_config(
        "xqt/recipes/smoke/smoke_cpu.yaml",
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


def test_builtin_pipeline_exports_mapping_validation_inputs(tmp_path) -> None:
    class MappingClassifier(torch.nn.Module):
        def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
        ) -> dict[str, torch.Tensor]:
            return {"logits": input_ids.float() + attention_mask.float()}

    batch = {
        "input_ids": torch.randn(2, 4),
        "attention_mask": torch.randn(2, 4),
        "labels": torch.tensor([0, 1]),
    }
    validation_loader = torch.utils.data.DataLoader([batch], batch_size=None)
    config = load_xqt_config(
        {
            "project": {
                "name": "mapping_export",
                "artifact_dir": str(tmp_path / "artifacts"),
            },
            "model": {"device": "cpu"},
            "export": {
                "targets": [
                    {
                        "format": "torch_export",
                        "output_path": str(tmp_path / "artifacts" / "model.pt2"),
                    },
                    {
                        "format": "torchscript",
                        "output_path": str(tmp_path / "artifacts" / "model.pt"),
                        "params": {"method": "trace"},
                    },
                ]
            },
        }
    )

    context = run_xqt_recipe(
        config,
        model=MappingClassifier(),
        data={"validation": validation_loader},
        pass_names=["export"],
        write_manifest=False,
    )

    artifacts = context.metrics["export"]["artifacts"]
    assert artifacts[0]["format"] == "torch_export"
    assert artifacts[0]["output_diff"]["allclose"] is True
    assert artifacts[1]["format"] == "torchscript"
    assert artifacts[1]["output_diff"]["allclose"] is True


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
        "xqt/recipes/smoke/smoke_cpu.yaml",
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
        "xqt/recipes/smoke/smoke_cpu.yaml",
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


def test_builtin_quant_pass_rejects_planned_backend_until_adapter_exists(tmp_path) -> None:
    config = load_xqt_config(
        {
            "project": {
                "name": "planned_quant_backend",
                "artifact_dir": str(tmp_path / "planned_quant_backend"),
            },
            "compression": {
                "quant": {
                    "enabled": True,
                    "backend": "gptq",
                },
                "prune": {"enabled": False},
            },
        }
    )

    with pytest.raises(XQTPipelineError, match="planned but not executable"):
        run_xqt_recipe(
            config,
            model=torch.nn.Linear(4, 2),
            pass_names=["quant"],
            write_manifest=False,
        )


def test_builtin_quant_pass_supports_onnxruntime_qdq(monkeypatch, tmp_path) -> None:
    output_path = tmp_path / "model_qdq.onnx"
    export_calls = {}

    def fake_quantize_onnx_qdq_static(
        onnx_path,
        output_path_arg,
        calibration_data,
        **kwargs,
    ):
        from xqt.quant.onnx_qdq import ONNXQDQQuantizationResult

        del onnx_path, calibration_data
        output = Path(output_path_arg)
        graph = onnx.helper.make_graph(
            [
                onnx.helper.make_node(
                    "QuantizeLinear",
                    ["input", "scale", "zero_point"],
                    ["input_q"],
                ),
                onnx.helper.make_node(
                    "DequantizeLinear",
                    ["input_q", "scale", "zero_point"],
                    ["input_dq"],
                ),
                onnx.helper.make_node("Conv", ["input_dq", "weight"], ["output"]),
            ],
            "qdq_test_graph",
            [
                onnx.helper.make_tensor_value_info(
                    "input",
                    onnx.TensorProto.FLOAT,
                    [1, 1, 4, 4],
                )
            ],
            [
                onnx.helper.make_tensor_value_info(
                    "output",
                    onnx.TensorProto.FLOAT,
                    [1, 1, 2, 2],
                )
            ],
            [
                onnx.helper.make_tensor("scale", onnx.TensorProto.FLOAT, [], [0.1]),
                onnx.helper.make_tensor("zero_point", onnx.TensorProto.UINT8, [], [0]),
                onnx.helper.make_tensor(
                    "weight",
                    onnx.TensorProto.FLOAT,
                    [1, 1, 3, 3],
                    [1.0] * 9,
                ),
            ],
        )
        onnx.save(onnx.helper.make_model(graph), output)
        return ONNXQDQQuantizationResult(
            path=output,
            source_path=output,
            checksum="checksum",
            calibration_samples=1,
            metadata={
                "input_names": list(kwargs["input_names"]),
                "calibration_summary": {
                    "input_names": list(kwargs["input_names"]),
                    "batch_count": 1,
                    "shapes": {"input": [[1, 4]]},
                    "dtypes": {"input": ["float32"]},
                },
            },
        )

    def fake_export_onnx(*args, **kwargs):
        from xqt.export.onnx_exporter import ONNXExportResult

        output = Path(args[2])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"onnx")
        export_calls["pre_export_fusion"] = kwargs.get("pre_export_fusion")
        return ONNXExportResult(
            path=output,
            opset=kwargs.get("opset"),
            checksum="onnx_checksum",
            checked=True,
            metadata={
                "pre_export_fusion": {
                    "enabled": True,
                    "mode": "eager",
                    "inplace": False,
                    "fused_groups": [["features.0", "features.1"]],
                }
            },
        )

    monkeypatch.setattr(
        "xqt.pipeline.passes.quantize_onnx_qdq_static",
        fake_quantize_onnx_qdq_static,
    )
    monkeypatch.setattr("xqt.pipeline.passes.export_onnx", fake_export_onnx)
    config = load_xqt_config(
        "xqt/recipes/smoke/smoke_cpu.yaml",
        overrides={
            "project": {
                "artifact_dir": str(tmp_path / "qdq_artifacts"),
            },
            "data": {
                "calibration": {
                    "target": "synthetic_classification",
                    "sample_limit": 2,
                    "batch_size": 1,
                }
            },
            "compression": {
                "prune": {"enabled": False},
                "quant": {
                    "enabled": True,
                    "backend": "onnxruntime_qdq",
                    "policy": {
                        "output_path": str(output_path),
                        "input_names": ["input"],
                        "op_types_to_quantize": ["Conv", "Gemm"],
                        "runtime_diff": False,
                        "pre_export_fusion": {
                            "enabled": True,
                            "mode": "eager",
                            "modules_to_fuse": [["features.0", "features.1"]],
                        },
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
    assert context.metrics["quant"]["calibration_summary"]["batch_count"] == 1
    assert context.metrics["quant"]["quantized_modules"] == ["onnx::Conv"]
    assert context.metrics["quant"]["metadata"]["quantized_op_types"] == ["Conv"]
    assert context.metrics["quant"]["metadata"]["qdq_node_count"] == 2
    assert context.metrics["quant"]["metadata"]["qdq_graph"]["op_type_counts"] == {
        "Conv": 1,
        "DequantizeLinear": 1,
        "QuantizeLinear": 1,
    }
    assert export_calls["pre_export_fusion"]["mode"] == "eager"
    assert context.metrics["quant"]["metadata"]["pre_export_fusion"]["mode"] == "eager"
    assert (
        context.metrics["quant"]["metadata"]["pre_export_fusion"]["fused_groups"]
        == [["features.0", "features.1"]]
    )


def test_builtin_quant_pass_auto_exports_multi_input_onnx(monkeypatch, tmp_path) -> None:
    class PairModel(torch.nn.Module):
        def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
            return left + right

    export_calls = {}

    def fake_quantize_onnx_qdq_static(
        onnx_path,
        output_path_arg,
        calibration_data,
        **kwargs,
    ):
        from xqt.quant.onnx_qdq import ONNXQDQQuantizationResult

        del onnx_path
        first_batch = next(iter(calibration_data))
        assert first_batch[0].shape == (1, 4)
        assert first_batch[1].shape == (1, 4)
        output = Path(output_path_arg)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"qdq")
        return ONNXQDQQuantizationResult(
            path=output,
            source_path=output,
            checksum="checksum",
            calibration_samples=1,
            metadata={
                "input_names": list(kwargs["input_names"]),
                "calibration_summary": {
                    "input_names": list(kwargs["input_names"]),
                    "batch_count": 1,
                    "shapes": {"input_0": [[1, 4]], "input_1": [[1, 4]]},
                    "dtypes": {"input_0": ["float32"], "input_1": ["float32"]},
                },
            },
        )

    def fake_export_onnx(*args, **kwargs):
        from xqt.export.onnx_exporter import ONNXExportResult

        export_calls["example_input"] = args[1]
        export_calls["input_names"] = kwargs.get("input_names")
        output = Path(args[2])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"onnx")
        return ONNXExportResult(
            path=output,
            opset=kwargs.get("opset"),
            checksum="onnx_checksum",
            checked=True,
            metadata={"input_names": list(kwargs.get("input_names") or [])},
        )

    monkeypatch.setattr("xqt.pipeline.passes.export_onnx", fake_export_onnx)
    monkeypatch.setattr(
        "xqt.pipeline.passes.quantize_onnx_qdq_static",
        fake_quantize_onnx_qdq_static,
    )
    calibration_loader = torch.utils.data.DataLoader(
        [
            (
                torch.ones(1, 4),
                torch.zeros(1, 4),
                torch.tensor([1]),
            )
        ],
        batch_size=None,
    )
    config = load_xqt_config(
        {
            "project": {
                "name": "multi_input_qdq",
                "artifact_dir": str(tmp_path / "artifacts"),
            },
            "model": {"device": "cpu"},
            "compression": {
                "quant": {
                    "enabled": True,
                    "backend": "onnxruntime_qdq",
                    "policy": {
                        "output_path": str(tmp_path / "artifacts" / "model_qdq.onnx"),
                    },
                }
            },
        }
    )

    context = run_xqt_recipe(
        config,
        model=PairModel(),
        data={"calibration": calibration_loader},
        pass_names=["quant"],
        write_manifest=False,
    )

    assert export_calls["input_names"] == ["input_0", "input_1"]
    assert isinstance(export_calls["example_input"], tuple)
    assert context.metrics["quant"]["calibration_summary"]["input_names"] == [
        "input_0",
        "input_1",
    ]


def test_builtin_quant_pass_auto_exports_unlabeled_multi_input_onnx(
    monkeypatch,
    tmp_path,
) -> None:
    class PairModel(torch.nn.Module):
        def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
            return left + right

    export_calls = {}

    def fake_quantize_onnx_qdq_static(
        onnx_path,
        output_path_arg,
        calibration_data,
        **kwargs,
    ):
        from xqt.quant.onnx_qdq import ONNXQDQQuantizationResult

        del onnx_path
        first_batch = next(iter(calibration_data))
        assert len(first_batch) == 2
        assert first_batch[0].shape == (1, 4)
        assert first_batch[1].shape == (1, 4)
        output = Path(output_path_arg)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"qdq")
        return ONNXQDQQuantizationResult(
            path=output,
            source_path=output,
            checksum="checksum",
            calibration_samples=1,
            metadata={
                "input_names": list(kwargs["input_names"]),
                "calibration_summary": {
                    "input_names": list(kwargs["input_names"]),
                    "batch_count": 1,
                    "shapes": {"input_0": [[1, 4]], "input_1": [[1, 4]]},
                    "dtypes": {"input_0": ["float32"], "input_1": ["float32"]},
                },
            },
        )

    def fake_export_onnx(*args, **kwargs):
        from xqt.export.onnx_exporter import ONNXExportResult

        export_calls["example_input"] = args[1]
        export_calls["input_names"] = kwargs.get("input_names")
        output = Path(args[2])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"onnx")
        return ONNXExportResult(
            path=output,
            opset=kwargs.get("opset"),
            checksum="onnx_checksum",
            checked=True,
            metadata={"input_names": list(kwargs.get("input_names") or [])},
        )

    monkeypatch.setattr("xqt.pipeline.passes.export_onnx", fake_export_onnx)
    monkeypatch.setattr(
        "xqt.pipeline.passes.quantize_onnx_qdq_static",
        fake_quantize_onnx_qdq_static,
    )
    calibration_loader = torch.utils.data.DataLoader(
        [
            (
                torch.ones(1, 4),
                torch.zeros(1, 4),
            )
        ],
        batch_size=None,
    )
    config = load_xqt_config(
        {
            "project": {
                "name": "multi_input_qdq_unlabeled",
                "artifact_dir": str(tmp_path / "artifacts_unlabeled"),
            },
            "model": {"device": "cpu"},
            "compression": {
                "quant": {
                    "enabled": True,
                    "backend": "onnxruntime_qdq",
                    "policy": {
                        "output_path": str(
                            tmp_path / "artifacts_unlabeled" / "model_qdq.onnx"
                        ),
                    },
                }
            },
        }
    )

    context = run_xqt_recipe(
        config,
        model=PairModel(),
        data={"calibration": calibration_loader},
        pass_names=["quant"],
        write_manifest=False,
    )

    assert export_calls["input_names"] == ["input_0", "input_1"]
    assert isinstance(export_calls["example_input"], tuple)
    assert len(export_calls["example_input"]) == 2
    assert context.metrics["quant"]["calibration_summary"]["input_names"] == [
        "input_0",
        "input_1",
    ]


def test_builtin_quant_pass_requires_explicit_calibration_split_for_onnx_qdq(tmp_path) -> None:
    config = load_xqt_config(
        {
            "project": {
                "name": "qdq_requires_calibration",
                "artifact_dir": str(tmp_path / "qdq_requires_calibration"),
            },
            "model": {"device": "cpu"},
            "compression": {
                "quant": {
                    "enabled": True,
                    "backend": "onnxruntime_qdq",
                    "policy": {
                        "output_path": str(tmp_path / "qdq_requires_calibration" / "model_qdq.onnx"),
                    },
                }
            },
        }
    )

    with pytest.raises(XQTPipelineError, match="calibration data is required"):
        run_xqt_recipe(
            config,
            model=torch.nn.Linear(4, 2),
            data={
                "validation": torch.utils.data.DataLoader(
                    [(torch.ones(1, 4), torch.tensor([1]))],
                    batch_size=None,
                )
            },
            pass_names=["quant"],
            write_manifest=False,
        )


def test_image_recipes_load_and_resnet_qdq_smoke_runs(monkeypatch, tmp_path) -> None:
    vit_config = load_xqt_config("xqt/recipes/quant/fp8/image_vit_torchao_fp8.yaml")
    assert vit_config.project.name == "quant_fp8_vit_torchao"
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
            metadata={
                "input_names": list(kwargs["input_names"]),
                "calibration_summary": {
                    "input_names": list(kwargs["input_names"]),
                    "batch_count": 1,
                    "shapes": {"input": [[1, 3, 224, 224]]},
                    "dtypes": {"input": ["float32"]},
                },
            },
        )

    monkeypatch.setattr(
        "xqt.pipeline.passes.quantize_onnx_qdq_static",
        fake_quantize_onnx_qdq_static,
    )
    resnet_config = load_xqt_config(
        "xqt/recipes/quant/int8/image_resnet_onnx_qdq_int8.yaml",
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

    assert context.config.project.name == "quant_int8_resnet_onnx_qdq"
    assert context.metrics["quant"]["backend"] == "onnxruntime_qdq"
    assert context.metrics["export"]["artifacts"][0]["format"] == "tensorrt"
    assert context.metrics["export"]["artifacts"][0]["dry_run"] is True


def test_image_vit_torchao_fp8_recipe_runs_on_cuda_when_available(tmp_path) -> None:
    if not torch.cuda.is_available():
        return

    config = load_xqt_config(
        "xqt/recipes/quant/fp8/image_vit_torchao_fp8.yaml",
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


def test_builtin_quant_pass_supports_component_policies_for_torchao(tmp_path) -> None:
    config = load_xqt_config(
        {
            "project": {
                "name": "component_torchao_quant",
                "artifact_dir": str(tmp_path / "component_torchao_quant"),
            },
            "model": {
                "target": "torch.nn.Sequential",
                "params": {},
                "device": "cpu",
            },
            "compression": {
                "quant": {
                    "enabled": True,
                    "backend": "torchao",
                    "strategy": "dynamic_int8",
                    "component_policies": [
                        {
                            "name": "encoder",
                            "target": "0",
                            "backend": "torchao",
                            "force_quantize": ["0"],
                        },
                        {
                            "name": "decoder",
                            "target": "2",
                            "backend": "torchao",
                            "skip_quantize": ["2"],
                            "analysis_only": True,
                        },
                    ],
                },
                "prune": {"enabled": False},
            },
        }
    )
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 4),
        torch.nn.ReLU(),
        torch.nn.Linear(4, 2),
    )

    context = run_xqt_recipe(
        config,
        model=model,
        pass_names=["quant"],
        write_manifest=False,
    )

    assert context.metrics["quant"]["mode"] == "multi_component"
    assert context.metrics["quant"]["summary"]["component_count"] == 2
    assert context.metrics["quant"]["backend"] == "torchao"
    assert len(context.metrics["quant"]["components"]) == 2
    by_name = {
        component["component_name"]: component
        for component in context.metrics["quant"]["components"]
    }
    assert by_name["encoder"]["backend"] == "torchao"
    assert by_name["encoder"]["quantized_module_count"] >= 0
    assert by_name["decoder"]["metadata"]["analysis_only"] is True
    assert by_name["decoder"]["metadata"]["executed"] is False


def test_multi_component_quant_smoke_recipe_runs_with_hetero_backends(
    monkeypatch,
    tmp_path,
) -> None:
    def fake_quantize_onnx_qdq_static(
        onnx_path,
        output_path_arg,
        calibration_data,
        **kwargs,
    ):
        from xqt.quant.onnx_qdq import ONNXQDQQuantizationResult

        del onnx_path
        first_batch = next(iter(calibration_data))
        assert first_batch[0].shape == (1, 4)
        output = Path(output_path_arg)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"qdq")
        return ONNXQDQQuantizationResult(
            path=output,
            source_path=output,
            checksum="vision_checksum",
            calibration_samples=1,
            metadata={
                "input_names": list(kwargs["input_names"]),
                "calibration_summary": {
                    "input_names": list(kwargs["input_names"]),
                    "batch_count": 1,
                    "shapes": {"input": [[1, 4]]},
                    "dtypes": {"input": ["float32"]},
                },
            },
        )

    def fake_export_onnx(*args, **kwargs):
        from xqt.export.onnx_exporter import ONNXExportResult

        output = Path(args[2])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"onnx")
        return ONNXExportResult(
            path=output,
            opset=kwargs.get("opset"),
            checksum="onnx_checksum",
            checked=True,
            metadata={"input_names": list(kwargs.get("input_names") or [])},
        )

    def fake_quantize_with_torchao(model, **kwargs):
        from xqt.quant.torchao_backend import TorchAOQuantizationResult

        del kwargs
        return TorchAOQuantizationResult(
            model=model,
            backend="torchao",
            strategy="dynamic_int8",
            quantized_modules=[""],
            metadata={"policy": {"include_module_types": ["Linear"]}},
        )

    monkeypatch.setattr("xqt.pipeline.passes.export_onnx", fake_export_onnx)
    monkeypatch.setattr(
        "xqt.pipeline.passes.quantize_onnx_qdq_static",
        fake_quantize_onnx_qdq_static,
    )
    monkeypatch.setattr("xqt.quant.executor.quantize_with_torchao", fake_quantize_with_torchao)

    config = load_xqt_config(
        "xqt/recipes/quant/int8/multi_component_quant_smoke.yaml",
        overrides={
            "project": {
                "artifact_dir": str(tmp_path / "multi_component_quant_smoke"),
            }
        },
    )

    context = run_xqt_recipe(config)

    assert context.metrics["quant"]["mode"] == "multi_component"
    assert context.metrics["quant"]["summary"]["component_count"] == 3
    assert context.metrics["quant"]["summary"]["backends"] == [
        "onnxruntime_qdq",
        "torchao",
        "torchao",
    ]
    components = {
        component["component_name"]: component
        for component in context.metrics["quant"]["components"]
    }
    assert components["vision_encoder"]["runtime"] == "onnxruntime"
    assert components["vision_encoder"]["artifacts"]["onnx"].endswith("vision_encoder_qdq.onnx")
    assert components["projector"]["metadata"]["analysis_only"] is True
    assert components["decoder"]["runtime"] == "pytorch"
    assert "quant_onnx_vision_encoder" in context.artifacts
    assert context.artifacts["manifest"].is_file()
    assert context.manifest is not None
    assert any(
        metric.name == "quant.vision_encoder.calibration_samples"
        for metric in context.manifest.metrics
    )
    assert any(
        metric.name == "quant.decoder.quantized_module_count"
        for metric in context.manifest.metrics
    )
    assert any(
        artifact.metadata.get("component_name") == "vision_encoder"
        for artifact in context.manifest.artifacts
    )


def test_prune_finetune_recipe_runs_schedule_with_teacher(tmp_path) -> None:
    teacher = torch.nn.Linear(4, 2)
    config = load_xqt_config(
        "xqt/recipes/prune/unstructured/prune_finetune_cpu.yaml",
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


def test_structured_prune_kd_recipe_runs_with_teacher_and_exports(tmp_path) -> None:
    teacher = torch.nn.Linear(5, 5)
    config = load_xqt_config(
        "xqt/recipes/prune/structured/structured_prune_kd_cpu.yaml",
        overrides={
            "project": {"artifact_dir": str(tmp_path / "structured_prune_kd")},
            "export": {
                "targets": [
                    {
                        "format": "onnx",
                        "output_path": str(tmp_path / "structured_prune_kd" / "model.onnx"),
                        "params": {"runtime_diff": False},
                    }
                ]
            },
        },
    )

    class TeacherWrapper(torch.nn.Module):
        def __init__(self, num_classes: int = 5) -> None:
            super().__init__()
            self.num_classes = num_classes

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            batch_size = x.shape[0]
            return torch.zeros(batch_size, self.num_classes, device=x.device)

    context = run_xqt_recipe(config, teacher=TeacherWrapper())

    assert context.metrics["prune"]["method"] == "structured"
    assert context.metrics["prune"]["final_sparsity"] == pytest.approx(0.25)
    assert len(context.metrics["prune"]["steps"]) == 1
    assert context.metrics["prune"]["steps"][0]["distillation"]["steps"] == 1
    assert context.metrics["prune"]["steps"][0]["pruning"]["granularity"] == "mlp_neuron"
    assert context.metrics["export"]["artifacts"][0]["format"] == "onnx"
    assert context.metrics["benchmark"]["latency"]["iterations"] == 1
    assert context.artifacts["manifest"].is_file()
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
            metadata={
                "input_names": list(kwargs["input_names"]),
                "calibration_summary": {
                    "input_names": list(kwargs["input_names"]),
                    "batch_count": 1,
                    "shapes": {"input": [[1, 3, 224, 224]]},
                    "dtypes": {"input": ["float32"]},
                },
            },
        )

    monkeypatch.setattr(
        "xqt.data.builders.build_torchvision_image_classification_loader",
        fake_build_torchvision_image_classification_loader,
    )
    monkeypatch.setattr(
        "xqt.pipeline.passes.quantize_onnx_qdq_static",
        fake_quantize_onnx_qdq_static,
    )
    config = load_xqt_config(
        "xqt/recipes/quant/int8/image_resnet_cifar100_qdq_cpu.yaml",
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
    assert context.metrics["quant"]["calibration_summary"]["batch_count"] == 1


def test_vit_mnist_prune_recipe_runs_with_small_torchvision_split(
    monkeypatch,
    tmp_path,
) -> None:
    calls = []

    def fake_build_torchvision_image_classification_loader(spec):
        calls.append(spec)
        inputs = torch.randn(2, 1, 224, 224)
        targets = torch.tensor([0, 1], dtype=torch.long)
        return torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(inputs, targets),
            batch_size=1,
        )

    monkeypatch.setattr(
        "xqt.data.builders.build_torchvision_image_classification_loader",
        fake_build_torchvision_image_classification_loader,
    )
    config = load_xqt_config(
        "xqt/recipes/prune/unstructured/image_vit_mnist_prune_cpu.yaml",
        overrides={
            "project": {"artifact_dir": str(tmp_path / "vit_mnist_prune")},
        },
    )

    context = run_xqt_recipe(config)

    assert len(calls) == 2
    assert calls[0].dataset_name == "MNIST"
    assert calls[0].download is True
    assert calls[0].sample_limit == 32
    assert calls[1].sample_limit == 16
    assert "calibration" in context.data
    assert "validation" in context.data
    assert context.metrics["baseline"]["samples"] == 2
    assert context.metrics["prune"]["sparsity"] == 0.5
    assert "benchmark" in context.metrics
    assert context.manifest is not None
    assert "prune" in context.manifest.passes


def test_vit_structured_prune_recipe_runs_with_mlp_neuron_pruning(tmp_path) -> None:
    config = load_xqt_config(
        "xqt/recipes/prune/structured/vit_structured_prune.yaml",
        overrides={
            "project": {"artifact_dir": str(tmp_path / "vit_structured_prune")},
        },
    )

    context = run_xqt_recipe(config)

    assert context.model is not None
    assert "validation" in context.data
    assert "baseline" in context.metrics
    assert "prune" in context.metrics
    assert "benchmark" in context.metrics
    assert context.metrics["prune"]["method"] == "structured"
    assert context.metrics["prune"]["granularity"] == "mlp_neuron"
    assert context.metrics["prune"]["importance_metric"] == "l2"
    assert context.metrics["prune"]["parameter_count_after"] < context.metrics["prune"]["parameter_count_before"]
    assert len(context.metrics["prune"]["targets"]) == 4
    assert len(context.metrics["prune"]["actions"]) == 4
    assert context.artifacts["metrics_json"].is_file()
    assert context.artifacts["metrics_markdown"].is_file()
    assert context.artifacts["manifest"].is_file()
    assert context.manifest is not None
    assert any(metric.name == "prune.sparsity" for metric in context.manifest.metrics)


def test_vit_structured_prune_recipe_supports_block_pruning_via_overrides(tmp_path) -> None:
    config = load_xqt_config(
        "xqt/recipes/prune/structured/vit_structured_prune.yaml",
        overrides={
            "project": {"artifact_dir": str(tmp_path / "vit_block_prune")},
            "compression": {
                "prune": {
                    "granularity": "block",
                    "scope": "per_layer",
                    "target_sparsity": 0.5,
                    "importance": {"metric": "l1"},
                    "selection": {"min_keep": 1},
                }
            },
        },
    )

    context = run_xqt_recipe(config)

    assert context.model is not None
    assert context.metrics["prune"]["granularity"] == "block"
    assert len(context.metrics["prune"]["targets"]) == 1
    assert len(context.metrics["prune"]["actions"]) == 1
    assert context.metrics["prune"]["actions"][0]["module_name"] == "blocks"
    assert context.metrics["prune"]["parameter_count_after"] < context.metrics["prune"]["parameter_count_before"]
    assert context.manifest is not None
    assert "prune" in context.manifest.passes


def test_vit_structured_prune_recipe_supports_head_pruning_via_overrides(tmp_path) -> None:
    config = load_xqt_config(
        "xqt/recipes/prune/structured/vit_structured_prune.yaml",
        overrides={
            "project": {"artifact_dir": str(tmp_path / "vit_head_prune")},
            "compression": {
                "prune": {
                    "granularity": "head",
                    "scope": "per_layer",
                    "target_sparsity": 0.5,
                    "importance": {"metric": "l2"},
                    "selection": {"min_keep": 1},
                }
            },
        },
    )

    context = run_xqt_recipe(config)

    assert context.model is not None
    assert context.metrics["prune"]["granularity"] == "head"
    assert len(context.metrics["prune"]["targets"]) == 4
    assert len(context.metrics["prune"]["actions"]) == 4
    assert context.metrics["prune"]["actions"][0]["action_type"] == "attention_heads"
    assert context.metrics["prune"]["parameter_count_after"] < context.metrics["prune"]["parameter_count_before"]
    assert context.manifest is not None
    assert "prune" in context.manifest.passes


def test_smoke_cpu_recipe_supports_nm_structured_pruning_via_overrides(tmp_path) -> None:
    config = load_xqt_config(
        "xqt/recipes/smoke/smoke_cpu.yaml",
        overrides={
            "project": {"artifact_dir": str(tmp_path / "nm_structured_prune")},
            "compression": {
                "prune": {
                    "enabled": True,
                    "method": "nm_structured",
                    "selection": {"pattern": [2, 4]},
                    "params": {"include_module_types": ["Linear"]},
                }
            },
            "export": {"targets": []},
        },
    )

    context = run_xqt_recipe(config)

    assert context.model is not None
    assert context.metrics["prune"]["method"] == "nm_structured"
    assert context.metrics["prune"]["granularity"] == "nm"
    assert context.metrics["prune"]["pattern_n"] == 2
    assert context.metrics["prune"]["pattern_m"] == 4
    assert context.metrics["prune"]["sparsity"] == pytest.approx(0.4)
    assert context.metrics["prune"]["compliance_ratio"] == pytest.approx(1.0)
    assert context.metrics["prune"]["layers"][0]["sparsity"] == pytest.approx(0.5)
    assert context.metrics["benchmark"]["capability"]["prune"]["supported"] is False
    assert context.metrics["benchmark"]["capability"]["prune"]["runtime"] == "pytorch_eager"
    assert context.metrics["benchmark"]["capability"]["prune"]["pattern_present"] is True
    assert context.metrics["benchmark"]["capability"]["prune"]["speedup_verified"] is False
    assert context.manifest is not None
    assert any(metric.name == "prune.sparsity" for metric in context.manifest.metrics)


def test_smoke_cpu_recipe_supports_block_sparse_pruning_via_overrides(tmp_path) -> None:
    config = load_xqt_config(
        "xqt/recipes/smoke/smoke_cpu.yaml",
        overrides={
            "project": {"artifact_dir": str(tmp_path / "block_sparse_prune")},
            "compression": {
                "prune": {
                    "enabled": True,
                    "method": "block_sparse",
                    "target_sparsity": 0.5,
                    "selection": {"block_shape": [2, 2]},
                    "params": {"include_module_types": ["Linear"]},
                }
            },
            "export": {"targets": []},
        },
    )

    context = run_xqt_recipe(config)

    assert context.model is not None
    assert context.metrics["prune"]["method"] == "block_sparse"
    assert context.metrics["prune"]["granularity"] == "block_sparse"
    assert context.metrics["prune"]["block_shape"] == [2, 2]
    assert context.metrics["prune"]["sparsity"] >= 0.5
    assert context.metrics["prune"]["parameter_sparsity"] > 0.0
    assert context.metrics["prune"]["layers"][0]["block_shape"] == [2, 2]
    assert context.metrics["benchmark"]["capability"]["prune"]["supported"] is False
    assert context.metrics["benchmark"]["capability"]["prune"]["runtime"] == "pytorch_eager"
    assert context.metrics["benchmark"]["capability"]["prune"]["pattern_present"] is True
    assert context.metrics["benchmark"]["capability"]["prune"]["speedup_verified"] is False
    assert context.manifest is not None
    assert any(metric.name == "prune.block_sparsity" for metric in context.manifest.metrics)


def test_sparse_benchmark_recipe_runs_and_reports_capability(tmp_path) -> None:
    config = load_xqt_config(
        "xqt/recipes/prune/block_sparse/sparse_benchmark_cpu.yaml",
        overrides={
            "project": {"artifact_dir": str(tmp_path / "sparse_benchmark_cpu")},
        },
    )

    context = run_xqt_recipe(config)

    assert context.model is not None
    assert "prune" in context.metrics
    assert "benchmark" in context.metrics
    assert context.metrics["prune"]["method"] == "block_sparse"
    assert context.metrics["prune"]["block_shape"] == [2, 2]
    assert context.metrics["benchmark"]["latency"]["iterations"] == 2
    assert context.metrics["benchmark"]["capability"]["prune"]["method"] == "block_sparse"
    assert context.metrics["benchmark"]["capability"]["prune"]["pattern_present"] is True
    assert context.metrics["benchmark"]["capability"]["prune"]["speedup_verified"] is False
    assert context.artifacts["metrics_json"].is_file()
    assert context.artifacts["metrics_markdown"].is_file()
    assert context.manifest is not None
    assert any(metric.name == "prune.block_sparsity" for metric in context.manifest.metrics)
