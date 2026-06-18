import json

from xqt.data import (
    PromptBatch,
    build_prompt_list,
    build_prompt_list_from_file,
    prompt_summary,
)


def test_build_prompt_list_supports_string_mapping_and_prompt_record() -> None:
    prompts = build_prompt_list(
        [
            "a cat",
            {
                "prompt": "a dog",
                "negative_prompt": "low quality",
                "seed": 7,
                "guidance_scale": 6.5,
            },
        ]
    )

    assert len(prompts) == 2
    assert prompts[0].prompt == "a cat"
    assert prompts[1].negative_prompt == "low quality"
    assert prompts[1].guidance_scale == 6.5
    record = prompts[1].to_prompt_record()
    assert record.prompt == "a dog"
    assert record.metadata["guidance_scale"] == 6.5


def test_build_prompt_list_supports_image_condition_schema() -> None:
    prompts = build_prompt_list(
        [
            {
                "prompt": "edit portrait",
                "image": "inputs/source.png",
                "mask": "inputs/mask.png",
                "reference_image": "refs/style.png",
                "latent_cache_key": "latent-001",
                "width": 768,
                "height": 1024,
            }
        ]
    )

    assert len(prompts) == 1
    prompt = prompts[0]
    assert prompt.condition_image == "inputs/source.png"
    assert prompt.condition_mask == "inputs/mask.png"
    assert prompt.reference_image == "refs/style.png"
    assert prompt.latent_cache_key == "latent-001"
    record = prompt.to_prompt_record()
    assert record.condition_image == "inputs/source.png"
    assert record.condition_mask == "inputs/mask.png"
    assert record.reference_image == "refs/style.png"
    assert record.latent_cache_key == "latent-001"
    assert record.metadata["condition_image"] == "inputs/source.png"


def test_build_prompt_list_supports_nested_condition_schema() -> None:
    prompts = build_prompt_list(
        [
            {
                "prompt": "nested prompt",
                "condition": {
                    "image": "inputs/nested.png",
                    "mask": "inputs/nested-mask.png",
                    "reference_image": "refs/nested.png",
                    "latent_cache_key": "latent-nested",
                },
            }
        ]
    )

    prompt = prompts[0]
    assert prompt.condition_image == "inputs/nested.png"
    assert prompt.condition_mask == "inputs/nested-mask.png"
    assert prompt.reference_image == "refs/nested.png"
    assert prompt.latent_cache_key == "latent-nested"


def test_build_prompt_list_from_json_and_summary(tmp_path) -> None:
    prompt_file = tmp_path / "prompts.json"
    prompt_file.write_text(
        json.dumps(
            {
                "prompts": [
                    {"prompt": "castle", "seed": 1},
                    {"prompt": "forest", "negative_prompt": "fog"},
                ]
            }
        ),
        encoding="utf-8",
    )

    prompts = build_prompt_list_from_file(prompt_file)
    summary = prompt_summary(prompts)

    assert [item.prompt for item in prompts] == ["castle", "forest"]
    assert summary["count"] == 2
    assert summary["has_seed"] is True
    assert summary["has_negative_prompt"] is True
    assert summary["sample_prompts"] == ["castle", "forest"]


def test_prompt_summary_reports_condition_fields() -> None:
    prompts = build_prompt_list(
        [
            {
                "prompt": "masked edit",
                "condition_image": "images/source.png",
                "condition_mask": "images/mask.png",
                "reference_image": "images/ref.png",
                "latent_cache_key": "latent-42",
            }
        ]
    )

    summary = prompt_summary(prompts)

    assert summary["has_condition_image"] is True
    assert summary["has_condition_mask"] is True
    assert summary["has_reference_image"] is True
    assert summary["has_latent_cache_key"] is True


def test_prompt_summary_limits_in_memory_prompt_count() -> None:
    prompts = build_prompt_list(
        [PromptBatch(prompt="a"), PromptBatch(prompt="b"), PromptBatch(prompt="c")],
        sample_limit=2,
    )

    assert len(prompts) == 2
    assert prompt_summary(prompts)["sample_prompts"] == ["a", "b"]
