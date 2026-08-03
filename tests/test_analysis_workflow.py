from __future__ import annotations

from pathlib import Path

import torch

from xqt.workflows import optimize_model


def test_quant_layer_statistics_workflow_runs_via_optimize_model() -> None:
    recipe = (
        Path(__file__).resolve().parents[2]
        / "xqt"
        / "recipes"
        / "analysis"
        / "quant_layer_statistics_workflow.yaml"
    )

    result = optimize_model(
        recipe,
        example_inputs=torch.randn(64, 64, dtype=torch.float32),
        write_outputs=False,
    )

    assert [stage.name for stage in result.stages] == [
        "fp4_quant",
        "analyze_quant_error",
    ]
    assert result.stages[0].accepted is True
    assert result.stages[1].accepted is True

    analysis_metrics = result.stages[1].metrics
    assert analysis_metrics["record_count"] >= 1
    assert analysis_metrics["activation_drift"]
    assert analysis_metrics["importance"]
    assert analysis_metrics["prune_candidates"]
    assert analysis_metrics["recommended_high_precision_modules"]

    stats = analysis_metrics["layer_statistics"]
    assert isinstance(stats, list)
    variable_pairs = {(row["layer"], row["variable"]) for row in stats}
    assert ("fc1", "output") in variable_pairs
    assert ("fc1", "weight") in variable_pairs
    assert ("fc2", "output") in variable_pairs
    assert ("fc2", "weight") in variable_pairs
    for row in stats:
        assert "error" in row
        assert "quantized" in row
        assert "float" in row
