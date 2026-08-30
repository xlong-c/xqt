from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from xqt.kernels.nn.fixtures import (
    build_smoke_diffusion_denoiser,
    build_smoke_detection_module,
    build_smoke_llm,
    build_smoke_moe,
    build_smoke_multimodal_classifier,
    build_smoke_vit_classifier,
    classify_model_family,
    component_grouping,
    diffusion_smoke_report,
    encoder_cache_metadata,
    family_smoke_report,
    moe_family_report,
    model_family_names,
    multimodal_input_signature,
    suggest_expert_pruning,
    visual_token_compression_metadata,
)
import torch
from xqt.kernels.nn.fixtures.toy_models import (
    ToyAttentionClassifier,
    ToyConvBlock,
    ToyTransformerClassifier,
)


REPO_ROOT = Path(__file__).resolve().parents[3]


def test_model_family_names_are_stable() -> None:
    assert model_family_names() == (
        "transformer",
        "vit",
        "detection",
        "llm",
        "diffusion",
        "moe",
        "multimodal",
        "convnet",
        "unknown",
    )


def test_classify_vit_and_detection_families() -> None:
    vit = build_smoke_vit_classifier()
    detection = build_smoke_detection_module()

    assert classify_model_family(vit) == "vit"
    assert classify_model_family(detection, task_type="detection") == "detection"
    assert classify_model_family(detection) == "detection"


def test_classify_transformer_and_convnet_heuristics() -> None:
    transformer = ToyTransformerClassifier()
    conv = ToyConvBlock()
    attention = ToyAttentionClassifier()

    assert classify_model_family(transformer) == "transformer"
    assert classify_model_family(conv) == "convnet"
    assert classify_model_family(attention) in {"transformer", "unknown"}


def test_classify_llm_moe_diffusion_multimodal_families() -> None:
    llm = build_smoke_llm()
    moe = build_smoke_moe()
    diffusion = build_smoke_diffusion_denoiser()
    multimodal = build_smoke_multimodal_classifier()

    assert classify_model_family(llm) == "llm"
    assert classify_model_family(moe) == "moe"
    assert classify_model_family(diffusion) == "diffusion"
    assert classify_model_family(multimodal) == "multimodal"


def test_component_grouping_groups_vit_attention_and_head() -> None:
    vit = build_smoke_vit_classifier()
    groups = component_grouping(vit)

    assert any(name.startswith("blocks.0.self_attn") for name in groups["attention"])
    assert "head" in groups["head"]
    assert any(name.startswith("blocks.0") for name in groups["norm"])


def test_component_grouping_llm_attention_projections() -> None:
    llm = build_smoke_llm()
    groups = component_grouping(llm)

    assert any(name == "blocks.0.q_proj" for name in groups["attention"])
    assert any(name == "blocks.0.k_proj" for name in groups["attention"])
    assert any(name == "blocks.0.v_proj" for name in groups["attention"])
    assert any(name.startswith("blocks.0.mlp_") for name in groups["ffn"])


def test_component_grouping_groups_detection_head() -> None:
    detection = build_smoke_detection_module()
    groups = component_grouping(detection, family="detection")

    assert any("head" in name for name in groups["head"])
    assert any(name == "stem" for name in groups["backbone"])


def test_family_smoke_report_is_honest_about_synthetic_status() -> None:
    vit = build_smoke_vit_classifier()
    report = family_smoke_report(vit)

    payload = report.to_dict()
    assert payload["model_family"] == "vit"
    assert payload["task_type"] is None
    assert payload["synthetic"] is True
    assert payload["speedup_claimed"] is False
    assert payload["performance_verified"] is False
    assert payload["module_counts"]["Linear"] >= 3
    assert any("smoke" in note for note in payload["notes"])


def test_moe_family_report_and_expert_pruning_suggestion() -> None:
    moe = build_smoke_moe()
    report = moe_family_report(moe)

    assert report.expert_count == 2
    assert report.shared_expert_count == 1
    assert report.router_module_count == 1
    assert report.expert_parallel_readiness == "metadata_only"
    assert report.load_balance_verified is False

    scores = torch.tensor([[0.9, 0.1], [0.7, 0.3], [0.8, 0.2]])
    assert suggest_expert_pruning(scores, prune_count=1) == [1]


def test_diffusion_and_multimodal_metadata_helpers() -> None:
    diffusion = build_smoke_diffusion_denoiser()
    report = diffusion_smoke_report(diffusion, sampling_steps=4)

    assert report.sampling_steps == 4
    assert report.synthetic is True
    assert report.to_dict()["export_limitations"]

    compression = visual_token_compression_metadata(ratio=0.25, method="pool")
    assert compression.ratio == 0.25
    assert "metadata only" in compression.to_dict()["notes"][0]

    cache = encoder_cache_metadata(cacheable=True)
    assert cache.owner == "external_runtime"
    assert cache.to_dict()["cacheable"] is True

    signature = multimodal_input_signature(
        [
            {"name": "image", "modality": "vision", "shape": [1, 3, 16, 16]},
            {"name": "text", "modality": "text", "shape": [1, 8]},
        ]
    )
    assert signature.to_dict()["modalities"][0]["modality"] == "vision"


def test_all_model_family_smoke_recipes_load() -> None:
    names = [
        "llm_smoke.yaml",
        "moe_smoke.yaml",
        "diffusion_smoke.yaml",
        "multimodal_smoke.yaml",
    ]
    for name in names:
        recipe: dict[str, Any] = yaml.safe_load(
            (REPO_ROOT / f"xqt/recipes/smoke/{name}").read_text(encoding="utf-8")
        )
        assert [stage["kind"] for stage in recipe["stages"]][:1] == ["prune"]


def test_transformer_vit_smoke_recipe_loads() -> None:
    recipe: dict[str, Any] = yaml.safe_load(
        (REPO_ROOT / "xqt/recipes/smoke/transformer_vit_smoke.yaml").read_text(
            encoding="utf-8"
        )
    )

    assert recipe["project"]["name"] == "transformer_vit_smoke"
    assert [stage["kind"] for stage in recipe["stages"]] == [
        "prune",
        "quant",
        "export",
        "benchmark",
    ]
    quant_stage = recipe["stages"][1]
    assert quant_stage["params"]["backend"] == "torchao"
    assert quant_stage["params"]["strategy"] == "w8a8_int8"
