from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from xqt.core.artifact import load_manifest
from xqt.model import build_toy_detection_module
from xqt.workflows import load_optimization_config, optimize_model


class TinyClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def _classification_loader(*, seed: int = 0, samples: int = 8) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    inputs = torch.randn(samples, 4, generator=generator)
    targets = torch.randint(0, 2, (samples,), generator=generator)
    return DataLoader(TensorDataset(inputs, targets), batch_size=4)


def test_stage_workflow_prunes_evaluates_benchmarks_and_exports(tmp_path: Path) -> None:
    model = TinyClassifier()
    export_path = tmp_path / "model.pt"
    config = {
        "project": {
            "name": "stage_workflow",
            "artifact_dir": str(tmp_path / "artifacts"),
        },
        "task": {"type": "classification"},
        "stages": [
            {
                "name": "baseline_eval",
                "kind": "eval",
                "split": "validation",
                "params": {"baseline": True},
            },
            {
                "name": "prune_sparse",
                "kind": "prune",
                "split": "validation",
                "params": {
                    "method": "global_l1_unstructured",
                    "target_sparsity": 0.5,
                },
            },
            {
                "name": "prune_eval",
                "kind": "eval",
                "split": "validation",
                "compare_to": "baseline_eval",
                "accept": {"metric": "top1", "max_drop": 1.0},
            },
            {
                "name": "latency",
                "kind": "benchmark",
                "split": "validation",
                "params": {"warmup": 0, "iterations": 1},
            },
            {
                "name": "export_torchscript",
                "kind": "export",
                "split": "validation",
                "save_model": False,
                "params": {
                    "targets": [
                        {
                            "format": "torchscript",
                            "output_path": str(export_path),
                            "params": {
                                "method": "trace",
                                "check_trace": False,
                            },
                        }
                    ]
                },
            },
        ],
    }

    result = optimize_model(
        config,
        model=model,
        data={"validation": _classification_loader()},
    )

    assert [stage.name for stage in result.stages] == [
        "baseline_eval",
        "prune_sparse",
        "prune_eval",
        "latency",
        "export_torchscript",
    ]
    assert result.baseline_stage == "baseline_eval"
    assert result.best_stage == "prune_sparse"
    assert result.best_model is result.models["prune_sparse"]
    assert result.stages[1].metrics["prune"]["sparsity"] >= 0.0
    assert result.stages[2].accepted is True
    assert result.stages[3].metrics["benchmark"]["iterations"] == 1
    assert export_path.is_file()
    assert result.stages[4].metrics["export"]["artifacts"][0]["format"] == "torchscript"


def test_stage_workflow_runtime_eval_uses_artifact_metrics_and_latency(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    onnx_path = tmp_path / "toy.onnx"
    onnx_path.write_bytes(b"fake")
    calls: list[str] = []

    def fake_evaluate_onnx_detection_model(
        path: str,
        *_args: Any,
        **_kwargs: Any,
    ) -> SimpleNamespace:
        calls.append(path)
        return SimpleNamespace(
            to_dict=lambda: {
                "runtime": "onnxruntime",
                "samples": 1,
                "metrics": {"map50_95": 0.95},
                "raw_output_diff": {"mean_abs": 0.01, "max_abs": 0.02},
                "decoded_diff": {"box_mae": 0.0},
                "latency": {"mean_ms": 0.5, "iterations": 1},
            }
        )

    monkeypatch.setattr(
        "xqt.workflows.optimization.evaluate_onnx_detection_model",
        fake_evaluate_onnx_detection_model,
    )
    config = {
        "project": {
            "name": "runtime_eval_case",
            "artifact_dir": str(tmp_path / "artifacts"),
        },
        "task": {
            "type": "detection",
            "class_names": ["a", "b", "c"],
            "detection_postprocess": {
                "format": "yolo_raw",
                "box_format": "xyxy",
                "score_activation": "sigmoid",
            },
        },
        "data_splits": {
            "validation": {
                "target": "synthetic_detection",
                "sample_limit": 1,
                "batch_size": 1,
                "params": {
                    "image_shape": [3, 32, 32],
                    "num_classes": 3,
                    "boxes_per_image": 2,
                },
            }
        },
        "stages": [
            {
                "name": "baseline_eval",
                "kind": "eval",
                "split": "validation",
                "params": {"baseline": True},
            },
            {
                "name": "runtime_eval",
                "kind": "runtime_eval",
                "split": "validation",
                "compare_to": "baseline_eval",
                "params": {
                    "path": str(onnx_path),
                    "input_names": ["image"],
                    "warmup": 0,
                    "iterations": 1,
                },
                "accept": {
                    "metric": "map50_95",
                    "max_drop": 1.0,
                    "max_mean_abs": 0.1,
                    "max_max_abs": 0.1,
                },
            },
        ],
    }

    result = optimize_model(config, model=build_toy_detection_module())

    assert calls == [str(onnx_path)]
    runtime_stage = result.stages[1]
    assert runtime_stage.accepted is True
    assert runtime_stage.metrics["metrics"]["map50_95"] == 0.95
    assert runtime_stage.metrics["latency"]["mean_ms"] == 0.5
    assert runtime_stage.metrics["acceptance"]["mean_abs"] == 0.01
    assert runtime_stage.metrics["runtime_eval"]["artifact"]["artifact_key"] == "last_onnx"
    assert runtime_stage.metrics["runtime_eval"]["artifact"]["source"] == "explicit_path"
    assert runtime_stage.metrics["runtime_eval"]["task"]["task_type"] == "detection"
    assert (
        runtime_stage.metrics["runtime_eval"]["task"]["detection_postprocess"]["format"]
        == "yolo_raw"
    )


def test_detection_runtime_eval_resolves_export_artifact_and_rejects_acceptance(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    onnx_path = tmp_path / "exported.onnx"
    calls: list[str] = []

    def fake_export_onnx(*args: Any, **kwargs: Any) -> Any:
        from xqt.export.onnx_exporter import ONNXExportResult

        output = Path(args[2])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"onnx")
        return ONNXExportResult(
            path=output,
            opset=kwargs.get("opset"),
            checksum="onnx_checksum",
            checked=True,
            metadata={
                "input_names": list(kwargs.get("input_names") or []),
                "output_names": list(kwargs.get("output_names") or []),
            },
        )

    def fake_evaluate_onnx_detection_model(
        path: str,
        *_args: Any,
        **_kwargs: Any,
    ) -> SimpleNamespace:
        calls.append(path)
        return SimpleNamespace(
            to_dict=lambda: {
                "runtime": "onnxruntime",
                "samples": 1,
                "metrics": {"map50_95": 0.25},
                "raw_output_diff": {"mean_abs": 0.20, "max_abs": 0.30},
                "decoded_diff": {"box_mae": 0.10},
                "latency": {"mean_ms": 1.5, "iterations": 1},
            }
        )

    monkeypatch.setattr("xqt.pipeline.passes.export_onnx", fake_export_onnx)
    monkeypatch.setattr(
        "xqt.workflows.optimization.evaluate_onnx_detection_model",
        fake_evaluate_onnx_detection_model,
    )

    result = optimize_model(
        {
            "project": {
                "name": "runtime_eval_artifact_reject_case",
                "artifact_dir": str(tmp_path / "artifacts"),
            },
            "task": {
                "type": "detection",
                "class_names": ["a", "b", "c"],
                "detection_postprocess": {
                    "format": "yolo_raw",
                    "box_format": "xyxy",
                    "score_activation": "sigmoid",
                },
            },
            "data_splits": {
                "validation": {
                    "target": "synthetic_detection",
                    "sample_limit": 1,
                    "batch_size": 1,
                    "params": {
                        "image_shape": [3, 32, 32],
                        "num_classes": 3,
                        "boxes_per_image": 2,
                    },
                }
            },
            "stages": [
                {
                    "name": "export_fp32_onnx",
                    "kind": "export",
                    "split": "validation",
                    "save_model": False,
                    "params": {
                        "targets": [
                            {
                                "format": "onnx",
                                "output_path": str(onnx_path),
                                "opset": 18,
                                "params": {
                                    "input_names": ["images"],
                                    "output_names": ["predictions"],
                                    "dynamo": False,
                                    "runtime_diff": False,
                                },
                            }
                        ]
                    },
                },
                {
                    "name": "runtime_eval_from_artifact",
                    "kind": "runtime_eval",
                    "split": "validation",
                    "save_model": False,
                    "params": {
                        "artifact": "last_onnx",
                        "input_names": ["images"],
                        "warmup": 0,
                        "iterations": 1,
                    },
                    "accept": {
                        "max_mean_abs": 0.05,
                        "max_max_abs": 0.50,
                    },
                },
            ],
        },
        model=build_toy_detection_module(),
    )

    assert calls == [str(onnx_path)]
    assert result.context.artifacts["last_onnx"] == onnx_path
    export_stage, runtime_stage = result.stages
    assert export_stage.metrics["export"]["artifacts"][0]["path"] == str(onnx_path)
    assert runtime_stage.accepted is False
    assert runtime_stage.message == "rejected by acceptance thresholds"
    artifact = runtime_stage.metrics["runtime_eval"]["artifact"]
    assert artifact["artifact_key"] == "last_onnx"
    assert artifact["source"] == "artifact"
    assert artifact["resolved_path"] == str(onnx_path)
    assert runtime_stage.metrics["acceptance"]["mean_abs"] == 0.20
    assert runtime_stage.metrics["acceptance"]["max_mean_abs"] == 0.05
    assert runtime_stage.metrics["acceptance"]["max_abs"] == 0.30
    assert runtime_stage.metrics["acceptance"]["max_max_abs"] == 0.50


def test_detection_runtime_eval_writes_dataset_metadata_into_workflow_manifest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    onnx_path = tmp_path / "toy.onnx"
    onnx_path.write_bytes(b"fake")

    def fake_evaluate_onnx_detection_model(
        path: str,
        *_args: Any,
        **_kwargs: Any,
    ) -> SimpleNamespace:
        assert path == str(onnx_path)
        return SimpleNamespace(
            to_dict=lambda: {
                "runtime": "onnxruntime",
                "samples": 1,
                "metrics": {"map50_95": 0.95},
                "raw_output_diff": {"mean_abs": 0.01, "max_abs": 0.02},
                "decoded_diff": {"box_mae": 0.0},
                "latency": {"mean_ms": 0.5, "iterations": 1},
                "metadata": {
                    "dataset": {
                        "target": "synthetic_detection",
                        "sample_limit": 1,
                        "batch_size": 1,
                        "image_shape": [3, 32, 32],
                        "resize": [32, 32],
                        "resize_mode": "direct_resize",
                        "letterbox": False,
                    },
                    "batches": [
                        {
                            "batch_size": 1,
                            "input_image_sizes": [[32, 32]],
                            "orig_sizes": [[32, 32]],
                        }
                    ],
                },
            }
        )

    monkeypatch.setattr(
        "xqt.workflows.optimization.evaluate_onnx_detection_model",
        fake_evaluate_onnx_detection_model,
    )

    result = optimize_model(
        {
            "project": {
                "name": "runtime_eval_manifest_case",
                "artifact_dir": str(tmp_path / "artifacts"),
            },
            "task": {
                "type": "detection",
                "class_names": ["a", "b", "c"],
                "detection_postprocess": {
                    "format": "yolo_raw",
                    "box_format": "xyxy",
                    "score_activation": "sigmoid",
                },
            },
            "data_splits": {
                "validation": {
                    "target": "synthetic_detection",
                    "sample_limit": 1,
                    "batch_size": 1,
                    "params": {
                        "image_shape": [3, 32, 32],
                        "num_classes": 3,
                        "boxes_per_image": 2,
                    },
                }
            },
            "stages": [
                {
                    "name": "runtime_eval",
                    "kind": "runtime_eval",
                    "split": "validation",
                    "params": {
                        "path": str(onnx_path),
                        "input_names": ["image"],
                        "warmup": 0,
                        "iterations": 1,
                    },
                }
            ],
        },
        model=build_toy_detection_module(),
    )

    manifest = load_manifest(result.context.artifacts["workflow_manifest"])
    metrics = {item["name"]: item for item in manifest["metrics"]}
    assert metrics["workflow.stage.runtime_eval.dataset_metadata"]["value"]["target"] == (
        "synthetic_detection"
    )
    assert metrics["workflow.stage.runtime_eval.dataset_metadata"]["value"]["resize_mode"] == (
        "direct_resize"
    )
    assert metrics["workflow.stage.runtime_eval.dataset_metadata"]["value"]["letterbox"] is False
    assert metrics["workflow.stage.runtime_eval.dataset_metadata"]["metadata"]["source"] == (
        "detection_report"
    )
    assert metrics["workflow.stage.runtime_eval.batch_metadata"]["value"][0]["batch_size"] == 1
    assert metrics["workflow.stage.runtime_eval.batch_metadata"]["value"][0][
        "input_image_sizes"
    ] == [[32, 32]]


def test_detection_quant_stage_reports_calibration_and_qdq_graph(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_onnx = tmp_path / "source_detection.onnx"
    output_path = tmp_path / "source_detection_qdq.onnx"
    source_onnx.write_bytes(b"onnx")

    def fake_quantize_onnx_qdq_static(
        onnx_path: str | Path,
        output_path_arg: str | Path,
        calibration_data: Any,
        **kwargs: Any,
    ) -> Any:
        import onnx

        from xqt.quant.onnx_qdq import ONNXQDQQuantizationResult

        assert Path(onnx_path) == source_onnx
        assert kwargs["input_names"] == ["images"]
        first_batch = next(iter(calibration_data))
        assert isinstance(first_batch, dict)
        assert first_batch["images"].shape == (1, 3, 32, 32)

        output = Path(output_path_arg)
        output.parent.mkdir(parents=True, exist_ok=True)
        graph = onnx.helper.make_graph(
            [
                onnx.helper.make_node(
                    "QuantizeLinear",
                    ["images", "scale", "zero_point"],
                    ["images_q"],
                ),
                onnx.helper.make_node(
                    "DequantizeLinear",
                    ["images_q", "scale", "zero_point"],
                    ["images_dq"],
                ),
                onnx.helper.make_node("Conv", ["images_dq", "weight"], ["predictions"]),
            ],
            "detection_qdq_test_graph",
            [
                onnx.helper.make_tensor_value_info(
                    "images",
                    onnx.TensorProto.FLOAT,
                    [1, 3, 32, 32],
                )
            ],
            [
                onnx.helper.make_tensor_value_info(
                    "predictions",
                    onnx.TensorProto.FLOAT,
                    [1, 2, 8],
                )
            ],
            [
                onnx.helper.make_tensor("scale", onnx.TensorProto.FLOAT, [], [0.1]),
                onnx.helper.make_tensor("zero_point", onnx.TensorProto.UINT8, [], [0]),
                onnx.helper.make_tensor(
                    "weight",
                    onnx.TensorProto.FLOAT,
                    [1, 3, 1, 1],
                    [1.0, 1.0, 1.0],
                ),
            ],
        )
        onnx.save(onnx.helper.make_model(graph), output)
        return ONNXQDQQuantizationResult(
            path=output,
            source_path=Path(onnx_path),
            checksum="qdq_checksum",
            calibration_samples=1,
            metadata={
                "input_names": list(kwargs["input_names"]),
                "activation_type": str(kwargs["activation_type"]),
                "weight_type": str(kwargs["weight_type"]),
                "op_types_to_quantize": list(kwargs["op_types_to_quantize"]),
                "calibration_summary": {
                    "input_names": list(kwargs["input_names"]),
                    "batch_count": 1,
                    "sample_count": 1,
                    "shapes": {"images": [[1, 3, 32, 32]]},
                    "dtypes": {"images": ["float32"]},
                },
            },
        )

    monkeypatch.setattr(
        "xqt.pipeline.passes.quantize_onnx_qdq_static",
        fake_quantize_onnx_qdq_static,
    )

    result = optimize_model(
        {
            "project": {
                "name": "detection_quant_report_case",
                "artifact_dir": str(tmp_path / "artifacts"),
            },
            "task": {"type": "detection"},
            "data_splits": {
                "calibration": {
                    "target": "synthetic_detection",
                    "sample_limit": 1,
                    "batch_size": 1,
                    "params": {
                        "image_shape": [3, 32, 32],
                        "num_classes": 3,
                        "boxes_per_image": 2,
                        "include_model_inputs": True,
                        "input_image_key": "images",
                    },
                },
                "validation": {
                    "target": "synthetic_detection",
                    "sample_limit": 1,
                    "batch_size": 1,
                    "params": {
                        "image_shape": [3, 32, 32],
                        "num_classes": 3,
                        "boxes_per_image": 2,
                        "include_model_inputs": True,
                        "input_image_key": "images",
                    },
                },
            },
            "stages": [
                {
                    "name": "quant_qdq",
                    "kind": "quant",
                    "calibration_split": "calibration",
                    "validation_split": "validation",
                    "save_model": False,
                    "params": {
                        "backend": "onnxruntime_qdq",
                        "strategy": "static_int8",
                        "policy": {
                            "onnx_path": str(source_onnx),
                            "output_path": str(output_path),
                            "input_names": ["images"],
                            "sample_limit": 1,
                            "activation_type": "QUInt8",
                            "weight_type": "QInt8",
                            "op_types_to_quantize": ["Conv"],
                        },
                    },
                }
            ],
        }
    )

    quant = result.stages[0].metrics["quant"]
    metadata = quant["metadata"]
    assert result.model is None
    assert result.context.artifacts["quant_onnx"] == output_path
    assert result.context.artifacts["last_onnx"] == output_path
    assert quant["backend"] == "onnxruntime_qdq"
    assert quant["calibration_samples"] == 1
    assert quant["calibration_summary"]["batch_count"] == 1
    assert quant["calibration_summary"]["sample_count"] == 1
    assert quant["quantized_modules"] == ["onnx::Conv"]
    assert metadata["source_split"] == "calibration"
    assert metadata["quantized_op_types"] == ["Conv"]
    assert metadata["qdq_node_count"] == 2
    assert metadata["qdq_graph"]["op_type_counts"] == {
        "Conv": 1,
        "DequantizeLinear": 1,
        "QuantizeLinear": 1,
    }

    manifest = load_manifest(result.context.artifacts["workflow_manifest"])
    metrics = {item["name"]: item for item in manifest["metrics"]}
    assert metrics["quant.model.calibration_samples"]["value"] == 1
    assert metrics["workflow.stage.quant_qdq.quant"]["value"]["metadata"][
        "quantized_op_types"
    ] == ["Conv"]
    assert metrics["workflow.stage.quant_qdq.quant"]["value"]["calibration_summary"][
        "batch_count"
    ] == 1


@pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile unavailable")
def test_detection_operator_stage_records_graph_break_and_fallback(
    tmp_path: Path,
) -> None:
    config = {
        "project": {
            "name": "detection_operator_compile_case",
            "artifact_dir": str(tmp_path / "artifacts"),
        },
        "task": {
            "type": "detection",
            "class_names": ["a", "b", "c"],
            "detection_postprocess": {
                "format": "yolo_raw",
                "box_format": "xyxy",
                "score_activation": "sigmoid",
            },
        },
        "model": {"device": "cpu"},
        "data_splits": {
            "validation": {
                "target": "synthetic_detection",
                "sample_limit": 1,
                "batch_size": 1,
                "params": {
                    "image_shape": [3, 32, 32],
                    "num_classes": 3,
                    "boxes_per_image": 2,
                },
            }
        },
        "stages": [
            {
                "name": "operator_compile",
                "kind": "operator",
                "split": "validation",
                "save_model": False,
                "params": {
                    "targets": [
                        {
                            "name": "model",
                            "backend": "torch_compile",
                            "min_speedup": 999.0,
                            "mode": "default",
                        }
                    ]
                },
            }
        ],
    }

    result = optimize_model(config, model=build_toy_detection_module())

    operator_metrics = result.stages[0].metrics["operator"]
    target = operator_metrics["targets"][0]
    graph_break_report = target["metadata"]["graph_break_report"]
    fallback_detail = target["metadata"]["fallback_detail"]

    assert operator_metrics["target_count"] == 1
    assert target["backend"] == "torch_compile"
    assert target["metadata"]["execution_state"] in {"fallback", "executed", "skipped"}
    assert graph_break_report["status"] in {"ok", "error", "unavailable"}
    assert "graph_break_count" in graph_break_report
    assert fallback_detail["graph_break_count"] == graph_break_report["graph_break_count"]
    assert fallback_detail["graph_breaks"] == graph_break_report["break_reasons"]
    assert fallback_detail["compiled_regions"] == graph_break_report["graph_count"]
    assert fallback_detail["explain"] == graph_break_report
    assert fallback_detail["backend"] == "torch_compile"
    assert fallback_detail["fallback"] == "eager"


def test_detection_structured_prune_stage_is_guarded_and_reports_skips(
    tmp_path: Path,
) -> None:
    result = optimize_model(
        {
            "project": {
                "name": "detection_structured_prune_guard_case",
                "artifact_dir": str(tmp_path / "artifacts"),
            },
            "task": {"type": "detection"},
            "model": {"device": "cpu"},
            "data_splits": {
                "validation": {
                    "target": "synthetic_detection",
                    "sample_limit": 1,
                    "batch_size": 1,
                    "params": {
                        "image_shape": [3, 32, 32],
                        "num_classes": 3,
                        "boxes_per_image": 2,
                    },
                }
            },
            "stages": [
                {
                    "name": "structured_prune_guard",
                    "kind": "prune",
                    "split": "validation",
                    "params": {
                        "method": "structured",
                        "granularity": "channel",
                        "target_sparsity": 0.5,
                    },
                    "accept": {"min_speedup": 1.01},
                }
            ],
        },
        model=build_toy_detection_module(),
    )

    stage = result.stages[0]
    prune = stage.metrics["prune"]
    assert stage.accepted is False
    assert stage.message == "rejected by acceptance thresholds"
    assert result.best_stage is None
    assert prune["method"] == "structured"
    assert prune["task_type"] == "detection"
    assert prune["applied"] is False
    assert prune["execution_state"] == "skipped"
    assert prune["speedup_claimed"] is False
    assert prune["sparsity"] >= 0.0
    assert prune["zero_parameters"] <= prune["total_parameters"]
    assert prune["skipped_modules"]
    assert prune["skipped_modules"][0]["module_name"] == "stem"
    assert "dependency rewrite is not implemented" in prune["skip_reason"]
    assert prune["skipped_modules"][0]["reason"] == prune["skip_reason"]
    assert stage.metrics["acceptance"]["speedup"] is None

    manifest = load_manifest(result.context.artifacts["workflow_manifest"])
    prune_metric = next(
        item for item in manifest["metrics"] if item["name"] == "prune.sparsity"
    )
    assert prune_metric["passed"] is False
    assert prune_metric["metadata"]["execution_state"] == "skipped"
    assert prune_metric["metadata"]["skipped_module_count"] == 1
    assert prune_metric["metadata"]["speedup_claimed"] is False


def test_detection_unstructured_prune_stage_is_sparsity_baseline(
    tmp_path: Path,
) -> None:
    result = optimize_model(
        {
            "project": {
                "name": "detection_unstructured_prune_baseline_case",
                "artifact_dir": str(tmp_path / "artifacts"),
            },
            "task": {"type": "detection"},
            "model": {"device": "cpu"},
            "data_splits": {
                "validation": {
                    "target": "synthetic_detection",
                    "sample_limit": 1,
                    "batch_size": 1,
                    "params": {
                        "image_shape": [3, 32, 32],
                        "num_classes": 3,
                        "boxes_per_image": 2,
                    },
                }
            },
            "stages": [
                {
                    "name": "prune_sparse",
                    "kind": "prune",
                    "split": "validation",
                    "params": {
                        "method": "global_l1_unstructured",
                        "target_sparsity": 0.2,
                    },
                }
            ],
        },
        model=build_toy_detection_module(),
    )

    prune = result.stages[0].metrics["prune"]
    assert result.stages[0].accepted is True
    assert result.best_stage == "prune_sparse"
    assert prune["method"] == "global_l1_unstructured"
    assert prune["task_type"] == "detection"
    assert prune["baseline_kind"] == "unstructured_sparsity_report"
    assert prune["applied"] is True
    assert prune["execution_state"] == "applied"
    assert prune["speedup_claimed"] is False
    assert prune["skip_reason"] is None
    assert prune["skipped_modules"] == []

    manifest = load_manifest(result.context.artifacts["workflow_manifest"])
    prune_metric = next(
        item for item in manifest["metrics"] if item["name"] == "prune.sparsity"
    )
    assert prune_metric["metadata"]["method"] == "global_l1_unstructured"
    assert prune_metric["metadata"]["task_type"] == "detection"
    assert prune_metric["metadata"]["baseline_kind"] == "unstructured_sparsity_report"
    assert prune_metric["metadata"]["speedup_claimed"] is False


def test_stage_workflow_finetune_and_distill_use_train_split(tmp_path: Path) -> None:
    config = {
        "project": {
            "name": "train_stage_case",
            "artifact_dir": str(tmp_path / "artifacts"),
        },
        "task": {"type": "classification"},
        "stages": [
            {
                "name": "finetune",
                "kind": "finetune",
                "train_split": "train",
                "params": {"max_steps": 1, "lr": 0.01},
            },
            {
                "name": "distill",
                "kind": "distill",
                "from_stage": "finetune",
                "train_split": "train",
                "params": {"max_steps": 1, "temperature": 2.0, "alpha": 0.5},
            },
        ],
    }
    teacher = TinyClassifier()

    result = optimize_model(
        config,
        model=TinyClassifier(),
        teacher=teacher,
        data={"train": _classification_loader(seed=3)},
    )

    assert result.stages[0].metrics["finetune"]["steps"] == 1
    assert result.stages[1].metrics["distill"]["steps"] == 1
    assert result.best_stage == "distill"


def test_stage_workflow_supervised_finetune_without_teacher(tmp_path: Path) -> None:
    config = {
        "project": {
            "name": "supervised_finetune_case",
            "artifact_dir": str(tmp_path / "artifacts"),
        },
        "task": {"type": "classification"},
        "stages": [
            {
                "name": "finetune",
                "kind": "finetune",
                "train_split": "train",
                "params": {"max_steps": 1, "lr": 0.01},
            }
        ],
    }

    result = optimize_model(
        config,
        model=TinyClassifier(),
        data={"train": _classification_loader(seed=5)},
    )

    assert result.stages[0].metrics["finetune"]["mode"] == "supervised"
    assert result.stages[0].metrics["finetune"]["steps"] == 1
    assert result.best_stage == "finetune"


def test_load_optimization_config_rejects_forward_from_stage() -> None:
    with pytest.raises(ValueError, match="unknown previous from_stage future"):
        load_optimization_config(
            {
                "stages": [
                    {
                        "name": "prune",
                        "kind": "prune",
                        "from_stage": "future",
                    },
                    {"name": "future", "kind": "eval"},
                ]
            }
        )


def test_yolo_detection_practice_recipe_uses_stage_schema() -> None:
    config = load_optimization_config("xqt/recipes/detection/yolo_detection_practice.yaml")

    assert config.project["name"] == "yolo_detection_practice"
    assert config.model.target == "xqt.model.build_toy_detection_module"
    assert config.task.type == "detection"
    assert "ultralytics" not in repr(config).lower()
    assert "validation" in config.data_splits
    assert "calibration" in config.data_splits
    assert config.data_splits["validation"].target == "synthetic_detection"
    assert config.data_splits["calibration"].target == "synthetic_detection"
    assert [stage.kind for stage in config.stages] == [
        "eval",
        "benchmark",
        "export",
        "runtime_eval",
        "quant",
        "runtime_eval",
        "prune",
        "eval",
        "benchmark",
    ]
    assert [stage.name for stage in config.stages] == [
        "baseline_eval",
        "baseline_latency",
        "export_fp32_onnx",
        "fp32_onnx_runtime",
        "quant_qdq",
        "qdq_onnx_runtime",
        "prune_sparse",
        "prune_eval",
        "prune_latency",
    ]
    assert all(stage.kind not in {"operator", "deploy"} for stage in config.stages)
    assert config.stages[3].name == "fp32_onnx_runtime"
    assert config.stages[3].params["artifact"] == "last_onnx"
    assert config.stages[3].accept.max_mean_abs == pytest.approx(5.0e-2)
    assert config.stages[4].name == "quant_qdq"
    assert config.stages[4].calibration_split == "calibration"
    assert config.stages[4].validation_split == "validation"
    assert config.stages[4].params["backend"] == "onnxruntime_qdq"
    assert config.stages[4].params["policy"]["op_types_to_quantize"] == [
        "Conv",
        "MatMul",
        "Gemm",
    ]
    assert config.stages[5].name == "qdq_onnx_runtime"
    assert config.stages[5].params["artifact"] == "quant_onnx"
    assert config.stages[6].params["method"] == "global_l1_unstructured"


def test_yolo_detection_trt_qdq_practice_recipe_uses_python_api_deploy() -> None:
    config = load_optimization_config(
        "xqt/recipes/detection/yolo_detection_trt_qdq_practice.yaml"
    )

    assert [stage.kind for stage in config.stages] == [
        "eval",
        "export",
        "quant",
        "runtime_eval",
        "deploy",
    ]
    deploy_target = config.stages[-1].params["targets"][0]
    assert deploy_target["format"] == "tensorrt"
    assert deploy_target["params"]["backend"] == "python_api"
    assert deploy_target["params"]["dry_run"] is True


def test_external_detection_trt_deploy_recipe_uses_external_onnx() -> None:
    config = load_optimization_config(
        "xqt/recipes/detection/external_detection_trt_deploy.yaml"
    )

    assert [stage.kind for stage in config.stages] == ["deploy"]
    deploy_target = config.stages[0].params["targets"][0]
    assert deploy_target["format"] == "tensorrt"
    assert deploy_target["params"]["backend"] == "python_api"
    assert deploy_target["params"]["dry_run"] is True
    assert deploy_target["params"]["onnx_path"] == "/tmp/xqt-external-detection.onnx"


def test_external_detection_qdq_trt_practice_recipe_uses_external_onnx() -> None:
    config = load_optimization_config(
        "xqt/recipes/detection/external_detection_qdq_trt_practice.yaml"
    )

    assert [stage.kind for stage in config.stages] == ["quant", "deploy"]
    quant_policy = config.stages[0].params["policy"]
    deploy_target = config.stages[1].params["targets"][0]
    assert quant_policy["onnx_path"] == "/tmp/xqt-external-detection.onnx"
    assert quant_policy["input_names"] == ["images", "orig_target_sizes"]
    assert deploy_target["format"] == "tensorrt"
    assert deploy_target["params"]["backend"] == "python_api"
    assert deploy_target["params"]["dry_run"] is True


def test_hf_rtdetr_r18vd_qdq_trt_practice_recipe_uses_single_input_external_onnx() -> None:
    config = load_optimization_config(
        "xqt/recipes/detection/hf_rtdetr_r18vd_qdq_trt_practice.yaml"
    )

    assert [stage.kind for stage in config.stages] == ["quant", "deploy"]
    quant_policy = config.stages[0].params["policy"]
    deploy_target = config.stages[1].params["targets"][0]
    assert (
        quant_policy["onnx_path"]
        == "/root/others/rtdetr-artifacts/onnx-community-rtdetr_r18vd-direct.onnx"
    )
    assert quant_policy["input_names"] == ["pixel_values"]
    assert deploy_target["format"] == "tensorrt"
    assert deploy_target["params"]["backend"] == "python_api"
    assert deploy_target["params"]["dry_run"] is True
    assert list(deploy_target["profiles"].keys()) == ["pixel_values"]


def test_hf_rtdetr_r18vd_qdq_trt_tensorrt_friendly_recipe_uses_symmetric_int8() -> None:
    config = load_optimization_config(
        "xqt/recipes/detection/hf_rtdetr_r18vd_qdq_trt_tensorrt_friendly.yaml"
    )

    assert [stage.kind for stage in config.stages] == ["quant", "deploy"]
    quant_policy = config.stages[0].params["policy"]
    deploy_target = config.stages[1].params["targets"][0]
    assert quant_policy["input_names"] == ["pixel_values"]
    assert quant_policy["activation_type"] == "QInt8"
    assert quant_policy["weight_type"] == "QInt8"
    assert quant_policy["extra_options"]["ActivationSymmetric"] is True
    assert quant_policy["extra_options"]["WeightSymmetric"] is True
    assert deploy_target["params"]["backend"] == "python_api"
    assert deploy_target["params"]["dry_run"] is True


def test_hf_rtdetr_r18vd_qdq_trt_tensorrt_friendly_eval_recipe_adds_runtime_eval() -> None:
    config = load_optimization_config(
        "xqt/recipes/detection/hf_rtdetr_r18vd_qdq_trt_tensorrt_friendly_eval.yaml"
    )

    assert [stage.kind for stage in config.stages] == [
        "quant",
        "runtime_eval",
        "deploy",
        "runtime_eval",
    ]
    assert config.device == "cuda:0"
    assert config.stages[1].params["runtime"] == "onnxruntime"
    assert config.stages[1].params["artifact"] == "quant_onnx"
    assert config.stages[3].params["runtime"] == "tensorrt"
    assert config.stages[3].params["artifact"] == "last_engine"
    assert config.stages[3].params["output_names"] == ["logits", "pred_boxes"]
    deploy_target = config.stages[2].params["targets"][0]
    assert deploy_target["params"]["backend"] == "python_api"
    assert deploy_target["params"]["dry_run"] is False
    assert deploy_target["params"]["runtime_benchmark"]["enabled"] is True


def test_build_toy_detection_module_accepts_input_channels() -> None:
    model = build_toy_detection_module(
        num_classes=5,
        boxes_per_image=3,
        input_channels=3,
    )

    assert model.input_channels == 3
    assert model.stem.in_channels == 3
    assert model.stem.out_channels == 3
