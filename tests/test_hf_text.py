from dataclasses import dataclass
from types import ModuleType
import sys

import torch
from torch import nn
from torch.utils.data import DataLoader

from xqt.core.config import load_xqt_config
from xqt.distill.hf_text import HFTextClassificationBundle
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
    config = load_xqt_config("xqt/recipes/hf_text_kd_prune.yaml")

    assert config.project.name == "hf_text_kd_prune"
    assert config.model.target == "xqt.distill.build_hf_text_classification_bundle_from_params"
    assert config.compression.distill.enabled is True
    assert config.compression.prune.enabled is True
    assert config.compression.axes == ["sparsity", "depth"]
