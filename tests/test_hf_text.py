from dataclasses import dataclass
from types import ModuleType
import sys

import torch
from torch import nn
from torch.utils.data import DataLoader

from xqt.core.config import load_xqt_config
from xqt.data.hf_text import build_hf_text_classification_loader
from xqt.distill.hf_text import HFTextClassificationBundle
from xqt.distill.hf_text import HFTextClassificationDataBundle
from xqt.pipeline.runner import run_xqt_recipe


class MappingClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(3, 2)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        del attention_mask
        pooled = input_ids.float()
        return {"logits": self.proj(pooled)}


@dataclass
class FakeTokenizer:
    name: str = "fake"


def build_fake_hf_bundle() -> HFTextClassificationBundle:
    batch = {
        "input_ids": torch.randn(4, 3),
        "attention_mask": torch.ones(4, 3),
        "labels": torch.tensor([0, 1, 0, 1]),
    }
    loader = DataLoader([batch], batch_size=None)
    return HFTextClassificationBundle(
        teacher=MappingClassifier(),
        student=MappingClassifier(),
        tokenizer=FakeTokenizer(),
        train_loader=loader,
        validation_loader=loader,
        metadata={"dataset_name": "fake"},
    )


def test_hf_text_bundle_target_populates_context_and_runs_distill(tmp_path) -> None:
    module = ModuleType("fake_hf_text_target")
    module.build_fake_hf_bundle = build_fake_hf_bundle
    sys.modules[module.__name__] = module

    config = load_xqt_config(
        {
            "project": {
                "name": "hf_text_fake",
                "artifact_dir": str(tmp_path / "artifacts"),
            },
            "model": {
                "target": "fake_hf_text_target.build_fake_hf_bundle",
                "device": "cpu",
            },
            "compression": {
                "axes": ["sparsity", "depth"],
                "distill": {
                    "enabled": True,
                    "temperature": 2.0,
                    "alpha": 0.5,
                    "params": {"max_steps": 1, "lr": 0.01},
                },
            },
            "benchmark": {"warmup": 0, "iterations": 1},
        }
    )

    context = run_xqt_recipe(config)

    assert isinstance(context.model, MappingClassifier)
    assert isinstance(context.teacher, MappingClassifier)
    assert "train" in context.data
    assert "validation" in context.data
    assert context.metrics["hf_text_bundle"]["dataset_name"] == "fake"
    assert context.metrics["distill"]["steps"] == 1
    assert context.metrics["baseline"]["samples"] == 4


def test_hf_text_recipe_schema_loads() -> None:
    config = load_xqt_config("xqt/recipes/distill/hf_text_kd_prune.yaml")

    assert config.project.name == "distill_hf_text_kd_prune"
    assert config.model.target == "xqt.distill.build_hf_text_classification_bundle_from_params"
    assert config.compression.distill.enabled is True
    assert config.compression.prune.enabled is True
    assert config.compression.axes == ["sparsity", "depth"]


def test_build_hf_text_classification_loader_uses_role_specific_limits(
    monkeypatch,
) -> None:
    captured = {}
    batch = {
        "input_ids": torch.randn(2, 3),
        "attention_mask": torch.ones(2, 3),
        "labels": torch.tensor([0, 1]),
    }

    def fake_build_hf_text_classification_data(
        spec,
        *,
        train_batch_size,
        validation_batch_size=None,
    ):
        captured["spec"] = spec
        captured["train_batch_size"] = train_batch_size
        captured["validation_batch_size"] = validation_batch_size
        loader = DataLoader([batch], batch_size=None)
        return HFTextClassificationDataBundle(
            tokenizer=FakeTokenizer(),
            train_loader=loader,
            validation_loader=loader,
            metadata={"dataset_name": spec.dataset_name},
        )

    monkeypatch.setattr(
        "xqt.data.hf_text.build_hf_text_classification_data",
        fake_build_hf_text_classification_data,
    )

    loader = build_hf_text_classification_loader(
        "train",
        model_params={
            "teacher_name_or_path": "teacher",
            "student_name_or_path": "student",
            "dataset_name": "glue",
            "dataset_config_name": "sst2",
            "text_column": "sentence",
            "label_column": "label",
        },
        split_params={"max_length": 256},
        batch_size=4,
        sample_limit=12,
    )

    first_batch = next(iter(loader))
    assert first_batch["labels"].shape[0] == 2
    assert captured["spec"].dataset_name == "glue"
    assert captured["spec"].train_sample_limit == 12
    assert captured["spec"].validation_sample_limit is None
    assert captured["spec"].max_length == 256
    assert captured["train_batch_size"] == 4
    assert captured["validation_batch_size"] == 4


def test_load_data_supports_hf_text_classification_target(monkeypatch, tmp_path) -> None:
    calls = []
    batch = {
        "input_ids": torch.randn(2, 3),
        "attention_mask": torch.ones(2, 3),
        "labels": torch.tensor([0, 1]),
    }

    def fake_build_hf_text_classification_loader(
        split_name,
        *,
        model_params=None,
        split_params=None,
        batch_size=1,
        sample_limit=None,
    ):
        calls.append(
            {
                "split_name": split_name,
                "model_params": dict(model_params or {}),
                "split_params": dict(split_params or {}),
                "batch_size": batch_size,
                "sample_limit": sample_limit,
            }
        )
        return DataLoader([batch], batch_size=None)

    monkeypatch.setattr(
        "xqt.data.builders.build_hf_text_classification_loader",
        fake_build_hf_text_classification_loader,
    )

    config = load_xqt_config(
        {
            "project": {
                "name": "hf_text_data_only",
                "artifact_dir": str(tmp_path / "artifacts"),
            },
            "model": {
                "params": {
                    "teacher_name_or_path": "teacher",
                    "student_name_or_path": "student",
                    "dataset_name": "glue",
                    "dataset_config_name": "sst2",
                    "text_column": "sentence",
                    "label_column": "label",
                }
            },
            "data": {
                "train": {
                    "target": "hf_text_classification",
                    "sample_limit": 4,
                    "batch_size": 2,
                },
                "validation": {
                    "target": "hf_text_classification",
                    "sample_limit": 6,
                    "batch_size": 3,
                },
            },
        }
    )

    context = run_xqt_recipe(config, pass_names=["load_data"], write_manifest=False)

    assert "train" in context.data
    assert "validation" in context.data
    assert [call["split_name"] for call in calls] == ["train", "validation"]
    assert calls[0]["model_params"]["dataset_name"] == "glue"
    assert calls[0]["batch_size"] == 2
    assert calls[0]["sample_limit"] == 4
    assert calls[1]["batch_size"] == 3
    assert calls[1]["sample_limit"] == 6
