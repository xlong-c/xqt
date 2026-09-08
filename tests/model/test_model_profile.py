from __future__ import annotations

import pytest
from torch import nn

from xqt.core.schema import ModelConfig
from xqt.core.workflow_loader import load_optimization_config
from xqt.model import (
    ModelProfile,
    ModelAdapter,
    model_profile_names,
    profile_from_model_config,
    register_model_profile,
)


class DemoModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(2, 2)


class DemoAdapter(ModelAdapter):
    def load(self, checkpoint: str | None, **params: object) -> nn.Module:
        del params
        assert checkpoint == "custom.pt"
        return DemoModel()

from xqt.pipeline.runner import create_context


def test_registered_profile_resolves_loader_and_serializes_in_context() -> None:
    profile = ModelProfile(
        profile_id="test.llama",
        family="llm",
        loader_target="tests.conftest.tiny_model",
        structure_contract="llama_decoder_v1",
        inference_adapter="text.generation",
        requirements={"transformers": ">=4.0"},
    )
    register_model_profile(profile, replace=True)

    config = load_optimization_config(
        {
            "model": {"profile": "test.llama", "checkpoint": "weights.safetensors"}
        }
    )
    context = create_context(config)

    assert context.model_profile is not None
    assert context.model_profile.profile_id == "test.llama"
    assert context.model_target == "tests.conftest.tiny_model"
    assert context.manifest is not None
    assert context.manifest.model_profile["family"] == "llm"
    assert "layers" not in context.manifest.model_profile
    assert "operators" not in context.manifest.model_profile


def test_builtin_profiles_cover_shipped_model_loaders() -> None:
    names = model_profile_names()
    assert {
        "hf.hunyuan-ocr",
        "hf.unlimited-ocr",
        "hf.ovisocr2",
        "diffusers.wan21-vae",
        "diffusers.flux2-klein",
    }.issubset(names)


def test_explicit_model_fields_override_registered_profile() -> None:
    register_model_profile(
        ModelProfile(
            profile_id="test.override",
            family="transformer",
            loader_target="tests.conftest.tiny_model",
        ),
        replace=True,
    )
    profile = profile_from_model_config(
        ModelConfig(
            profile="test.override",
            family="llm",
            target="tests.conftest.tiny_model",
        )
    )

    assert profile is not None
    assert profile.family == "llm"
    assert profile.loader_target == "tests.conftest.tiny_model"


def test_unknown_profile_fails_explicitly() -> None:
    with pytest.raises(Exception, match="unknown model profile"):
        profile_from_model_config(ModelConfig(profile="missing.profile"))


def test_external_model_instance_can_use_profile_without_loader() -> None:
    register_model_profile(
        ModelProfile(
            profile_id="external.custom",
            family="transformer",
            structure_contract="custom_transformer_v1",
        ),
        replace=True,
    )
    config = load_optimization_config(
        {"model": {"profile": "external.custom", "checkpoint": "external.pt"}}
    )
    context = create_context(config, model=object())

    assert context.model is not None
    assert context.model_profile is not None
    assert context.model_profile.loader_target is None


def test_profile_adapter_can_construct_model_without_loader_target() -> None:
    register_model_profile(
        ModelProfile(
            profile_id="external.adapter",
            family="transformer",
            adapter_target="tests.xqt.model.test_model_profile.DemoAdapter",
        ),
        replace=True,
    )
    config = load_optimization_config(
        {"model": {"profile": "external.adapter", "checkpoint": "custom.pt"}}
    )
    context = create_context(config)

    from xqt.pipeline.passes import LoadModelPass

    LoadModelPass().run(context)
    assert isinstance(context.model, nn.Module)
    assert context.model.__class__.__name__ == "DemoModel"
