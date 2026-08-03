from pathlib import Path

import torch

from xqt.workflows import optimize_model


def test_convrot_w8a8_workflow_runs_via_optimize_model() -> None:
    recipe = (
        Path(__file__).resolve().parents[2]
        / "xqt"
        / "recipes"
        / "quant"
        / "int8"
        / "convrot_w8a8_smoke.yaml"
    )

    result = optimize_model(
        recipe,
        example_inputs=torch.randn(2, 16, dtype=torch.float32),
        calibration_inputs=[torch.randn(2, 16, dtype=torch.float32)],
        write_outputs=False,
    )

    assert [stage.name for stage in result.stages] == [
        "convrot_w8a8_quant",
        "benchmark_model",
    ]
    quant_stage = result.stages[0]
    assert quant_stage.accepted is True
    assert quant_stage.metrics["strategy"] == "w8a8_int8"
    assert quant_stage.metrics["metadata"]["execution_state"] == "convrot_int8"
    assert "linear" in quant_stage.metrics["quantized_modules"]
