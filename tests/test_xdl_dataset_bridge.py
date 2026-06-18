import json

import torch

from xqt.data import build_data_split
from xqt.pipeline.runner import run_xqt_recipe


class PairLinear(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        with torch.no_grad():
            self.linear.weight.copy_(torch.eye(2))
            self.linear.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def test_build_data_split_supports_xdl_dataset_bridge(tmp_path) -> None:
    manifest_path = tmp_path / "text.jsonl"
    manifest_path.write_text(
        json.dumps({"text": "hello", "target_text": "world", "sample_id": "txt-1"}) + "\n"
        + json.dumps({"text": "goodbye", "target_text": "moon", "sample_id": "txt-2"}) + "\n",
        encoding="utf-8",
    )
    split = type(
        "Split",
        (),
        {
            "target": "xdl_dataset",
            "params": {
                "dataset": {
                    "target": "registry:RecordTextDataset",
                    "params": {
                        "manifest_path": str(manifest_path),
                    },
                },
                "dataloader": {
                    "batch_size": 2,
                },
            },
            "sample_limit": 1,
            "batch_size": 4,
            "root": None,
        },
    )()

    loader = build_data_split("validation", split)
    batch = next(iter(loader))

    assert len(loader.dataset) == 1
    assert batch["text"] == ["hello"]
    assert batch["target_text"] == ["world"]
    assert batch["sample_id"] == ["txt-1"]


def test_builtin_pipeline_consumes_xdl_dataset_validation_split(tmp_path) -> None:
    recipe = {
        "project": {
            "name": "xdl_bridge_eval",
            "artifact_dir": str(tmp_path / "artifacts"),
        },
        "model": {
            "device": "cpu",
        },
        "data": {
            "validation": {
                "target": "xdl_dataset",
                "batch_size": 2,
                "params": {
                    "dataset": {
                        "target": "registry:SyntheticClassificationDataset",
                        "params": {
                            "num_samples": 2,
                            "input_shape": [2],
                            "num_classes": 2,
                            "seed": 0,
                        },
                    },
                    "dataloader": {
                        "batch_size": 2,
                    },
                },
            }
        },
    }

    context = run_xqt_recipe(
        recipe,
        model=PairLinear(),
        pass_names=["load_data", "baseline_eval"],
        write_manifest=False,
    )

    assert "validation" in context.data
    assert context.metrics["baseline"]["samples"] == 2
    assert "top1" in context.metrics["baseline"]["metrics"]
