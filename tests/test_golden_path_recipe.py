from __future__ import annotations

from pathlib import Path

from xqt.workflows import load_optimization_config
from xqt.workflows.stage_specs import (
    AnalyzeStageSpec,
    BenchmarkStageSpec,
    DeployStageSpec,
    ExportStageSpec,
    OperatorStageSpec,
    QuantStageSpec,
)


def test_weight_only_tilelang_onnx_tensorrt_recipe_declares_golden_stage_order() -> None:
    recipe = (
        Path(__file__).resolve().parents[2]
        / "xqt"
        / "recipes"
        / "smoke"
        / "weight_only_tilelang_onnx_tensorrt_golden.yaml"
    )

    config = load_optimization_config(recipe)

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
    assert isinstance(deploy.spec, DeployStageSpec)
    assert deploy.spec.targets[0].format == "tensorrt"
    assert deploy.spec.targets[0].params["dry_run"] is True
    assert deploy.spec.runtime_handle is not None
    assert deploy.spec.runtime_handle.materialize is False
