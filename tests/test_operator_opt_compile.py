import pytest
import torch

from xqt.core.artifact import ArtifactManifest
from xqt.core.config import load_xqt_config
from xqt.core.types import XQTContext
from xqt.pipeline.passes import OperatorOptimizationPass
from xqt.pipeline.runner import run_xqt_recipe


@pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile unavailable")
def test_operator_optimization_compile_smoke_runs(tmp_path) -> None:
    model = torch.nn.Linear(4, 2)
    config = load_xqt_config(
        {
            "project": {"artifact_dir": str(tmp_path / "compile_smoke")},
            "model": {"device": "cpu"},
            "benchmark": {"warmup": 0, "iterations": 1},
            "operator_optimization": {
                "enabled": True,
                "targets": [
                    {
                        "name": "model",
                        "backend": "torch_compile",
                        "min_speedup": 1.000001,
                        "mode": "reduce-overhead",
                    }
                ],
            },
        }
    )
    context = XQTContext(
        config=config,
        model=model,
        data={"validation": [(torch.randn(2, 4), torch.zeros(2, dtype=torch.long))]},
        manifest=ArtifactManifest(project_name=config.project.name),
    )

    output = OperatorOptimizationPass().run(context)

    target = output.metrics["operator_optimization"]["targets"][0]
    assert target["compile_time_ms"] is not None
    assert target["latency_before"]["iterations"] == 1
    assert target["latency_after"]["iterations"] == 1
    assert target["numeric_diff"]["allclose"] is True
    assert target["shape_signature"]["tensor_shapes"] == [[2, 4]]
    graph_break_report = target["metadata"]["graph_break_report"]
    assert graph_break_report["status"] in {"ok", "error", "unavailable"}
    assert "graph_break_count" in graph_break_report
    assert target["metadata"]["fallback_detail"]["graph_break_count"] == graph_break_report[
        "graph_break_count"
    ]


def test_operator_compile_smoke_recipe_runs_in_runner(tmp_path) -> None:
    config = load_xqt_config(
        "xqt/recipes/operator/torch_compile/operator_compile_smoke_cpu.yaml",
        overrides={
            "project": {"artifact_dir": str(tmp_path / "recipe_operator_compile")},
        },
    )

    context = run_xqt_recipe(config)

    assert "operator_optimization" in context.metrics
    assert context.metrics["operator_optimization"]["target_count"] == 1
    assert "operator_optimization" in context.manifest.passes
    assert context.artifacts["operator_optimization_report"].is_file()
