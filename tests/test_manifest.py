from pathlib import Path

from xqt.core.artifact import load_manifest
from xqt.core.config import load_xqt_config
from xqt.pipeline.runner import run_xqt_recipe


def test_manifest_records_operator_optimization_section_for_recipe(tmp_path) -> None:
    config = load_xqt_config(
        "xqt/recipes/operator_compile_smoke_cpu.yaml",
        overrides={
            "project": {"artifact_dir": str(tmp_path / "operator_manifest_recipe")},
        },
    )

    context = run_xqt_recipe(config)

    assert context.manifest is not None
    assert context.manifest.operator_optimization is not None
    section = context.manifest.operator_optimization
    assert section["target_count"] == 1
    assert isinstance(section["targets"], list)
    assert isinstance(section["candidates"], dict)
    assert "fx" in section["candidates"]

    report_artifact = next(
        artifact
        for artifact in context.manifest.artifacts
        if artifact.metadata.get("kind") == "operator_optimization_report"
    )
    assert report_artifact.format == "json"
    assert Path(report_artifact.path).is_file()


def test_operator_optimization_report_contains_expected_target_fields(tmp_path) -> None:
    config = load_xqt_config(
        {
            "project": {"artifact_dir": str(tmp_path / "operator_manifest_fields")},
            "model": {
                "target": "torch.nn.Linear",
                "params": {"in_features": 4, "out_features": 2},
                "device": "cpu",
            },
            "data": {
                "validation": {
                    "target": "synthetic_classification",
                    "sample_limit": 2,
                    "batch_size": 2,
                    "params": {"input_shape": [4], "num_classes": 2},
                }
            },
            "benchmark": {"warmup": 0, "iterations": 1},
            "operator_optimization": {
                "enabled": True,
                "targets": [
                    {
                        "name": "model",
                        "backend": "torch_compile",
                        "min_speedup": 10.0,
                    }
                ],
            },
        }
    )

    context = run_xqt_recipe(config)

    assert context.manifest is not None
    section = context.manifest.operator_optimization
    assert section is not None
    target = section["targets"][0]
    expected_keys = {
        "target_name",
        "module_path",
        "backend",
        "runtime",
        "applied",
        "fallback",
        "skip_reason",
        "compile_time_ms",
        "latency_before",
        "latency_after",
        "speedup",
        "numeric_diff",
        "device",
        "dtype",
        "shape_signature",
        "exportable",
        "artifact_paths",
    }
    assert expected_keys.issubset(target.keys())
    assert target["target_name"] == "model"
    assert target["backend"] == "torch_compile"
    assert target["fallback"] == "eager"
    assert target["runtime"] == "pytorch"
    assert isinstance(target["latency_before"], dict)
    assert isinstance(target["latency_after"], dict)
    assert isinstance(target["numeric_diff"], dict)
    assert isinstance(target["shape_signature"], dict)
    assert isinstance(target["artifact_paths"], dict)

    report_path = context.artifacts["operator_optimization_report"]
    report_data = load_manifest(report_path)
    assert "targets" in report_data
    assert "candidates" in report_data
    assert report_data["targets"][0]["target_name"] == "model"
