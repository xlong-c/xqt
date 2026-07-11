from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest
import torch

from xqt.export.tensorrt import execute_tensorrt_session
from xqt.workflows import load_optimization_config, optimize_model
from xqt.workflows.stage_specs import (
    AnalyzeStageSpec,
    BenchmarkStageSpec,
    DeployStageSpec,
    ExportStageSpec,
    OperatorStageSpec,
    QuantStageSpec,
)


requires_golden_hardware = pytest.mark.skipif(
    not (
        os.environ.get("XQT_RUN_GOLDEN_HARDWARE_TESTS") == "1"
        and torch.cuda.is_available()
        and importlib.util.find_spec("onnx") is not None
        and importlib.util.find_spec("tilelang") is not None
    ),
    reason=(
        "XQT_RUN_GOLDEN_HARDWARE_TESTS=1, CUDA, ONNX, and TileLang are required "
        "for the golden workflow hardware smoke test"
    ),
)

requires_golden_tensorrt_hardware = pytest.mark.skipif(
    not (
        os.environ.get("XQT_RUN_GOLDEN_HARDWARE_TESTS") == "1"
        and torch.cuda.is_available()
        and importlib.util.find_spec("onnx") is not None
        and importlib.util.find_spec("tilelang") is not None
        and importlib.util.find_spec("tensorrt") is not None
    ),
    reason=(
        "XQT_RUN_GOLDEN_HARDWARE_TESTS=1, CUDA, ONNX, TileLang, and TensorRT "
        "are required for the golden TensorRT runtime test"
    ),
)


def _golden_recipe_path() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "xqt"
        / "recipes"
        / "smoke"
        / "weight_only_tilelang_onnx_tensorrt_golden.yaml"
    )


def test_weight_only_tilelang_onnx_tensorrt_recipe_declares_golden_stage_order() -> (
    None
):
    config = load_optimization_config(_golden_recipe_path())

    assert [stage.kind for stage in config.stages] == [
        "quant",
        "operator",
        "benchmark",
        "analyze",
        "export",
        "deploy",
    ]
    quant, operator, benchmark, analyze, export, deploy = config.stages
    assert isinstance(quant.spec, QuantStageSpec)
    assert quant.spec.strategy == "fp4_weight_only"
    assert isinstance(operator.spec, OperatorStageSpec)
    assert operator.spec.targets[0].engine == "tilelang"
    assert operator.spec.targets[0].patterns == ["dequant_gemm_epilogue"]
    assert isinstance(benchmark.spec, BenchmarkStageSpec)
    assert benchmark.spec.warmup == 10
    assert benchmark.spec.iterations == 50
    assert isinstance(analyze.spec, AnalyzeStageSpec)
    assert isinstance(export.spec, ExportStageSpec)
    assert export.spec.targets[0].format == "onnx"
    assert export.spec.targets[0].dynamic_shapes == {"input": {0: "batch"}}
    assert export.spec.targets[0].onnx.dynamo is False
    assert export.spec.targets[0].onnx.runtime_diff is False
    assert export.spec.targets[0].onnx.pre_export_lowering.enabled is True
    assert (
        export.spec.targets[0].onnx.pre_export_lowering.mode
        == "fp4_weight_only_to_dense_linear"
    )
    assert isinstance(deploy.spec, DeployStageSpec)
    assert deploy.spec.targets[0].format == "tensorrt"
    assert deploy.spec.targets[0].tensorrt.dry_run is True
    assert deploy.spec.targets[0].tensorrt.runtime_benchmark.enabled is True
    assert deploy.spec.targets[0].tensorrt.runtime_benchmark.input_shapes == {
        "input": [8, 64]
    }
    assert deploy.spec.targets[0].tensorrt.runtime_benchmark.warmup == 2
    assert deploy.spec.targets[0].tensorrt.runtime_benchmark.iterations == 5
    assert deploy.spec.targets[0].tensorrt.runtime_benchmark.device == "cuda:0"
    assert deploy.spec.runtime_handle is not None
    assert deploy.spec.runtime_handle.materialize is False


@requires_golden_hardware
def test_golden_recipe_runs_cuda_dry_deploy_without_promoting_fallback(
    tmp_path: Path,
) -> None:
    """Exercise the declarative golden path without materializing a TensorRT engine."""

    config = load_optimization_config(_golden_recipe_path())
    config.project["artifact_dir"] = str(tmp_path)
    export_stage = next(stage for stage in config.stages if stage.kind == "export")
    deploy_stage = next(stage for stage in config.stages if stage.kind == "deploy")
    export_stage.params["targets"][0]["output_path"] = str(tmp_path / "model.onnx")
    deploy_stage.params["targets"][0]["output_path"] = str(tmp_path / "model.engine")

    result = optimize_model(
        config,
        example_inputs=torch.randn(8, 64, device="cuda"),
        write_outputs=False,
    )

    assert [stage.name for stage in result.stages] == [
        "fp4_weight_only",
        "tilelang_dequant_gemm",
        "benchmark_optimized_model",
        "analyze_optimized_model",
        "export_onnx",
        "deploy_tensorrt",
    ]
    operator = next(stage for stage in result.stages if stage.kind == "operator")
    operator_target = operator.metrics["targets"][0]
    execution_state = operator_target["metadata"]["execution_state"]
    assert operator_target["fallback_policy"] == "strict"
    assert execution_state in {"executed", "fallback"}
    assert operator_target["applied"] is (execution_state == "executed")
    if execution_state == "fallback":
        assert operator_target["skip_reason"]

    export = next(stage for stage in result.stages if stage.kind == "export")
    assert export.metrics["artifacts"][0]["format"] == "onnx"
    assert export.metrics["artifacts"][0]["checked"] is True
    assert export.metrics["artifacts"][0]["pre_export_lowering"]["mode"] == (
        "fp4_weight_only_to_dense_linear"
    )
    assert (tmp_path / "model.onnx").is_file()

    deploy = next(stage for stage in result.stages if stage.kind == "deploy")
    deploy_target = deploy.metrics["targets"][0]
    assert deploy_target["format"] == "tensorrt"
    assert deploy_target["dry_run"] is True
    assert deploy_target["artifact_status"] == "command_only"
    assert deploy_target["backend_execution"] == "dry_run"
    assert deploy.metrics["runtime_handle_request"]["materialize"] is False


@requires_golden_tensorrt_hardware
def test_golden_recipe_materializes_tensorrt_runtime_without_promoting_fallback(
    tmp_path: Path,
) -> None:
    """Validate the dense deploy lowering through a real TensorRT runtime handle."""

    config = load_optimization_config(_golden_recipe_path())
    config.project["artifact_dir"] = str(tmp_path)
    export_stage = next(stage for stage in config.stages if stage.kind == "export")
    deploy_stage = next(stage for stage in config.stages if stage.kind == "deploy")
    export_stage.params["targets"][0]["output_path"] = str(tmp_path / "model.onnx")
    deploy_stage.params["targets"][0]["output_path"] = str(tmp_path / "model.engine")
    deploy_stage.params["targets"][0]["tensorrt"]["dry_run"] = False
    deploy_stage.params["targets"][0]["tensorrt"]["workspace_mib"] = 256
    deploy_stage.params["runtime_handle"]["materialize"] = True
    deploy_stage.params["runtime_handle"]["tensorrt"] = {"device": "cuda:0"}

    inputs = torch.randn(8, 64, device="cuda")
    result = optimize_model(config, example_inputs=inputs, write_outputs=False)

    operator = next(stage for stage in result.stages if stage.kind == "operator")
    operator_target = operator.metrics["targets"][0]
    execution_state = operator_target["metadata"]["execution_state"]
    assert operator_target["applied"] is (execution_state == "executed")

    deploy = next(stage for stage in result.stages if stage.kind == "deploy")
    deploy_target = deploy.metrics["targets"][0]
    runtime_handle = deploy.metrics["runtime_handle"]
    assert deploy_target["dry_run"] is False
    assert deploy_target["artifact_status"] == "materialized"
    assert deploy_target["backend_execution"] == "executed"
    assert (tmp_path / "model.engine").is_file()
    assert runtime_handle["metadata"]["runtime_validation"] == {
        "status": "session_created",
        "engine_deserialized": True,
        "execution_context_created": True,
    }
    benchmark = deploy_target["runtime_benchmark"]
    assert benchmark is not None
    assert benchmark["backend"] == "python_api"
    assert benchmark["input_shapes"] == {"input": [8, 64]}
    assert benchmark["latency"]["warmup"] == 2
    assert benchmark["latency"]["iterations"] == 5
    assert benchmark["latency"]["mean_ms"] > 0.0

    with torch.no_grad():
        reference = result.context.require_model()(inputs)
    execution = execute_tensorrt_session(
        runtime_handle["handle"],
        inputs={"input": inputs},
    )
    assert execution.input_shapes == {"input": [8, 64]}
    assert execution.output_shapes == {"output": [8, 64]}
    torch.testing.assert_close(
        execution.output_tensors["output"],
        reference,
        atol=1e-3,
        rtol=1e-3,
    )
