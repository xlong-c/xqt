import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from examples.yolo_detection_practice import main


def test_yolo_detection_practice_example_emits_scenario_matrix(
    monkeypatch,
    tmp_path,
    capsys,
) -> None:
    recipe = tmp_path / "recipe.yaml"
    artifact_dir = tmp_path / "artifacts"
    recipe.write_text(
        f"""
project:
  name: yolo_detection_practice
  artifact_dir: {artifact_dir}
model:
  target: xqt.model.build_ultralytics_detection_module
  params:
    weights: yolo11n.pt
task:
  type: detection
  params:
    dataset_yaml: coco8.yaml
data:
  calibration:
    target: synthetic_detection
    sample_limit: 1
    batch_size: 1
    params:
      image_shape: [3, 32, 32]
  validation:
    target: synthetic_detection
    sample_limit: 1
    batch_size: 1
    params:
      image_shape: [3, 32, 32]
compression:
  quant:
    enabled: false
operator_optimization:
  enabled: false
benchmark:
  warmup: 0
  iterations: 1
""",
        encoding="utf-8",
    )

    monkeypatch.setenv("XQT_YOLO_PRACTICE_CONFIG", str(recipe))
    monkeypatch.setenv("XQT_YOLO_PRACTICE_SCENARIOS", "baseline,quant_only")

    def fake_resolve_ultralytics_dataset(dataset: str, *, autodownload: bool = True):
        del autodownload
        return SimpleNamespace(
            to_dict=lambda: {
                "yaml_path": dataset,
                "root": str(tmp_path / "datasets" / "coco8"),
                "train": "train",
                "val": "val",
                "names": ["a", "b"],
                "nc": 2,
            },
            names=["a", "b"],
        )

    def fake_run_xqt_recipe(config):
        artifact_dir_for_config = Path(config.project.artifact_dir)
        if config.export.targets:
            for index, target in enumerate(config.export.targets):
                if target.format == "onnx":
                    output_path = Path(
                        target.output_path
                        or artifact_dir_for_config / f"model_{index}.onnx"
                    )
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    output_path.write_bytes(b"onnx")
        return SimpleNamespace(
            config=config,
            metrics={
                "baseline": {"metrics": {"map50_95": 0.5}},
                "quant": (
                    {
                        "backend": "onnxruntime_qdq",
                        "path": str(
                            Path(config.project.artifact_dir) / "model_qdq.onnx"
                        ),
                        "metadata": {
                            "op_types_to_quantize": ["Conv", "Gemm"],
                            "quantized_op_types": ["Conv"],
                            "qdq_node_count": 2,
                            "calibration_summary": {"batch_count": 1},
                        },
                    }
                    if "quant_only" in config.project.name
                    else None
                ),
                "export": (
                    {
                        "artifacts": [
                            {
                                "format": "onnx",
                                "path": str(
                                    Path(config.project.artifact_dir)
                                    / "model_fp32.onnx"
                                ),
                                "checked": True,
                            },
                            {
                                "format": "tensorrt",
                                "path": str(
                                    Path(config.project.artifact_dir) / "model.engine"
                                ),
                                "dry_run": True,
                                "precision": "int8",
                                "profiles": {"images": {"opt": [1, 3, 32, 32]}},
                            },
                            {
                                "format": "openvino",
                                "path": str(
                                    Path(config.project.artifact_dir) / "model.xml"
                                ),
                                "dry_run": True,
                                "precision": "int8",
                            },
                        ]
                    }
                    if config.export.targets
                    else None
                ),
                "benchmark": {"mean_ms": 1.0},
                "operator_optimization": {
                    "target_count": len(config.operator_optimization.targets),
                    "targets": [
                        {
                            "target_name": target.name,
                            "backend": target.backend,
                            "applied": False,
                            "skip_reason": "metadata-only",
                            "metadata": {
                                "deployment_target": {"semantic_target": target.name}
                            },
                        }
                        for target in config.operator_optimization.targets
                    ],
                },
                "prune": None,
            },
            manifest=SimpleNamespace(passes=["load_data", "baseline_eval"]),
            data={"validation": [object()]},
            artifacts={},
            reference_model="baseline_model",
            model="current_model",
        )

    def fake_evaluate_detection_runtime_model(*args, **kwargs):
        del args, kwargs
        return SimpleNamespace(
            to_dict=lambda: {
                "runtime": "pytorch",
                "metrics": {"map50_95": 0.5},
                "raw_output_diff": {"max_abs": 0.0, "mean_abs": 0.0},
                "decoded_diff": {"box_mae": 0.0},
                "latency": {"iterations": 1, "mean_ms": 2.0},
            }
        )

    def fake_evaluate_onnx_detection_model(*args, **kwargs):
        onnx_path = str(args[0])
        del kwargs
        metric = 0.45 if "fp16" in onnx_path else 0.4
        return SimpleNamespace(
            to_dict=lambda: {
                "runtime": "onnxruntime",
                "metrics": {"map50_95": metric},
                "raw_output_diff": {"max_abs": 0.2, "mean_abs": 0.1},
                "decoded_diff": {"box_mae": 0.1, "label_match_rate": 1.0},
                "latency": {"iterations": 1, "mean_ms": 1.0},
            }
        )

    def fake_convert_onnx_to_fp16(onnx_path, output_path, **kwargs):
        del onnx_path, kwargs
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"fp16")
        return SimpleNamespace(
            path=output,
            checksum="abc",
            metadata={"precision": "fp16", "keep_io_types": False},
        )

    monkeypatch.setattr(
        "examples.yolo_detection_practice.resolve_ultralytics_dataset",
        fake_resolve_ultralytics_dataset,
    )
    monkeypatch.setattr(
        "examples.yolo_detection_practice.run_xqt_recipe",
        fake_run_xqt_recipe,
    )
    monkeypatch.setattr(
        "examples.yolo_detection_practice.evaluate_detection_runtime_model",
        fake_evaluate_detection_runtime_model,
    )
    monkeypatch.setattr(
        "examples.yolo_detection_practice.evaluate_onnx_detection_model",
        fake_evaluate_onnx_detection_model,
    )
    monkeypatch.setattr(
        "examples.yolo_detection_practice.convert_onnx_to_fp16",
        fake_convert_onnx_to_fp16,
    )

    assert main() == 0

    console_output = json.loads(capsys.readouterr().out)
    assert console_output["project"] == "yolo_detection_practice"
    assert console_output["scenario_count"] == 2
    assert console_output["scenario_order"] == ["baseline", "quant_only"]
    assert set(console_output["outputs"]) == {"matrix_json", "text_report", "log"}
    assert "scenarios" not in console_output

    matrix_path = Path(console_output["outputs"]["matrix_json"])
    report_path = Path(console_output["outputs"]["text_report"])
    log_path = Path(console_output["outputs"]["log"])
    output = json.loads(matrix_path.read_text(encoding="utf-8"))
    assert output["project"] == "yolo_detection_practice"
    assert output["dataset"]["yaml_path"] == "coco8.yaml"
    assert output["scenario_order"] == ["baseline", "quant_only"]
    assert output["outputs"]["matrix_json"] == str(matrix_path)
    assert output["outputs"]["text_report"] == str(report_path)
    assert output["outputs"]["log"] == str(log_path)
    assert matrix_path.is_file()
    assert report_path.is_file()
    assert log_path.is_file()
    assert "YOLO Detection Practice Report" in report_path.read_text(encoding="utf-8")
    assert "Selected scenarios: baseline, quant_only" in log_path.read_text(
        encoding="utf-8"
    )

    baseline = output["scenarios"]["baseline"]
    assert baseline["baseline"]["metrics"]["map50_95"] == 0.5
    assert baseline["benchmark"]["mean_ms"] == 1.0
    assert baseline["operator_optimization"]["target_count"] == 0
    assert (
        baseline["runtime_scenarios"]["baseline_pytorch"]["metrics"]["map50_95"] == 0.5
    )
    assert baseline["runtime_scenarios"]["fp32_onnx"]["metrics"]["map50_95"] == 0.4
    assert baseline["runtime_scenarios"]["fp16_onnx"]["metrics"]["map50_95"] == 0.45
    assert baseline["metrics"]["drops"]["fp16_onnx"]["map50_95_drop"] == pytest.approx(
        0.05
    )
    assert (
        baseline["metrics"]["latency_speedups"]["fp32_onnx"][
            "speedup_vs_baseline_pytorch"
        ]
        == 2.0
    )
    assert Path(baseline["runtime_scenarios_path"]).is_file()

    quant_only = output["scenarios"]["quant_only"]
    assert quant_only["quant"]["backend"] == "onnxruntime_qdq"
    assert (
        quant_only["runtime_scenarios"]["quant_onnx_qdq"]["metrics"]["map50_95"] == 0.4
    )
    assert output["summary"]["runtime_coverage"]["baseline"] == [
        "baseline_pytorch",
        "current_pytorch",
        "fp16_onnx",
        "fp32_onnx",
    ]
    assert output["summary"]["quantized_op_types"] == ["Conv"]
    assert output["summary"]["calibration"]["quant_only"]["batch_count"] == 1
