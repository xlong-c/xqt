from pathlib import Path

import torch

from xqt.workflows import optimize_model


def test_convrot_w4a4_workflow_runs_via_optimize_model() -> None:
    recipe = (
        Path(__file__).resolve().parents[2]
        / "xqt"
        / "recipes"
        / "quant"
        / "int4"
        / "convrot_w4a4_smoke.yaml"
    )

    result = optimize_model(
        recipe,
        example_inputs=torch.randn(2, 16, dtype=torch.float32),
        calibration_inputs=[torch.randn(2, 16, dtype=torch.float32)],
        write_outputs=False,
    )

    assert [stage.name for stage in result.stages] == [
        "convrot_quant",
        "benchmark_model",
    ]
    quant_stage = result.stages[0]
    assert quant_stage.accepted is True
    assert quant_stage.metrics["strategy"] == "convrot_w4a4"
    assert quant_stage.metrics["metadata"]["execution_state"] == "convrot_4bit"
    assert quant_stage.metrics["quantized_modules"] == ["linear", "proj"]
    assert quant_stage.metrics["metadata"]["recommended_high_precision_modules"] == [
        "linear",
        "proj",
    ]
