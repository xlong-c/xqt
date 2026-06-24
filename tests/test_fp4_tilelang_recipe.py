from __future__ import annotations

from pathlib import Path

import torch

from xqt.workflows import optimize_model


def test_fp4_tilelang_workflow_recipe_runs_via_optimize_model() -> None:
    recipe = (
        Path(__file__).resolve().parents[2]
        / "xqt"
        / "recipes"
        / "operator"
        / "tilelang"
        / "fp4_tilelang_workflow.yaml"
    )

    result = optimize_model(
        recipe,
        example_inputs=torch.randn(64, 64, dtype=torch.float32),
        write_outputs=False,
    )

    assert [stage.name for stage in result.stages] == [
        "fp4_quant",
        "tilelang_fp4_fc1",
    ]
    assert result.stages[0].accepted is True

    operator_metrics = result.stages[1].metrics
    assert operator_metrics["target_count"] == 1
    target = operator_metrics["targets"][0]
    assert target["backend"] == "tilelang"
    assert target["module_path"] == "fc1"
    assert target["metadata"]["execution_mode"] == "reference_fallback"
    assert target["metadata"]["kernel_constraints"]["supported_patterns"] == [
        "dequant_gemm_epilogue"
    ]
    assert target["metadata"]["kernel_pattern"] == "fp4_packed_dequant_gemm_epilogue"
    assert target["metadata"]["weight_source"] == "reference_fp4_linear_packed_bridge"
    assert target["metadata"]["weight_representation"] == "packed_signed_int4_plus_group_scale"
    assert target["metadata"]["consumes_packed_weight"] is True
    assert target["metadata"]["unpack_stage"] == "eager_reference_fallback"
