import pytest
import torch

from xqt.diffusion_distill import (
    DiffusionCache,
    DiffusionSpec,
    PromptRecord,
    build_step_schedule,
    trajectory_from_schedule,
)


def test_diffusion_spec_validates_and_round_trips(tmp_path) -> None:
    spec = DiffusionSpec(
        teacher_id="teacher",
        student_id="student",
        teacher_steps=20,
        student_steps=4,
        latent_shape=(1, 4, 8, 8),
    )
    cache = DiffusionCache(tmp_path)

    path = cache.write_spec(spec)
    loaded = cache.read_spec()

    assert path.is_file()
    assert loaded.teacher_id == "teacher"
    assert loaded.student_steps == 4
    assert loaded.latent_shape == (1, 4, 8, 8)


@pytest.mark.parametrize(
    "spec",
    [
        DiffusionSpec(teacher_id="x", teacher_steps=0),
        DiffusionSpec(teacher_id="x", teacher_steps=2, student_steps=4),
        DiffusionSpec(teacher_id="x", guidance_scale=-1.0),
        DiffusionSpec(teacher_id="x", latent_shape=(1, 0)),
    ],
)
def test_diffusion_spec_rejects_invalid_values(spec: DiffusionSpec) -> None:
    with pytest.raises(ValueError):
        spec.validate()


def test_diffusion_cache_round_trips_prompt_latent_and_trajectory(tmp_path) -> None:
    cache = DiffusionCache(tmp_path)
    prompt = PromptRecord(
        prompt="a small castle",
        seed=7,
        guidance_scale=6.5,
        condition_image="images/source.png",
        condition_mask="images/mask.png",
        latent_cache_key="latent-007",
    )
    key = cache.prompt_key(prompt)
    latent = torch.randn(1, 4, 8, 8)
    schedule = build_step_schedule(10, 3)
    trajectory = trajectory_from_schedule(
        key,
        schedule,
        [latent, latent + 1, latent + 2],
        metadata={"kind": "teacher"},
    )

    prompt_path = cache.write_prompt(prompt)
    latent_path = cache.write_latent(key, latent)
    trajectory_path = cache.write_trajectory(trajectory)

    assert prompt_path.is_file()
    assert latent_path.is_file()
    assert trajectory_path.is_file()
    loaded_prompt = cache.read_prompt(key)
    assert loaded_prompt.prompt == "a small castle"
    assert loaded_prompt.guidance_scale == 6.5
    assert loaded_prompt.condition_image == "images/source.png"
    assert loaded_prompt.condition_mask == "images/mask.png"
    assert loaded_prompt.latent_cache_key == "latent-007"
    assert torch.equal(cache.read_latent(key), latent)
    loaded_trajectory = cache.read_trajectory(key)
    assert loaded_trajectory.timesteps == [9, 4, 0]
    assert torch.equal(loaded_trajectory.latents[1], latent + 1)
    assert loaded_trajectory.metadata == {"kind": "teacher"}

    assert cache.condition_key(prompt) == cache.condition_key(loaded_prompt)


def test_diffusion_cache_rejects_mismatched_trajectory_lengths(tmp_path) -> None:
    cache = DiffusionCache(tmp_path)
    record = trajectory_from_schedule(
        "prompt",
        build_step_schedule(4, 2),
        [torch.zeros(1)],
    )

    with pytest.raises(ValueError, match="same length"):
        cache.write_trajectory(record)
