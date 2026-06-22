from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

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

    assert "validation" in config.data_splits
    assert "calibration" in config.data_splits
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
        "operator",
        "benchmark",
        "deploy",
    ]
    assert config.stages[3].name == "fp32_onnx_runtime"
    assert config.stages[5].name == "qdq_onnx_runtime"
