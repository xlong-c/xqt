import pytest
import torch

from xqt.diffusion_distill import (
    DiffusionSamplingReport,
    build_image_grid_record,
    consistency_distillation_loss,
    prediction_target,
)


def test_prediction_target_supports_epsilon_x0_and_v_prediction() -> None:
    clean = torch.tensor([[1.0, 2.0]])
    noisy = torch.tensor([[0.5, 0.25]])
    noise = torch.tensor([[0.1, 0.2]])

    assert torch.equal(
        prediction_target(
            clean_latent=clean,
            noisy_latent=noisy,
            noise=noise,
            alpha=0.9,
            sigma=0.1,
            prediction_type="epsilon",
        ),
        noise,
    )
    assert torch.equal(
        prediction_target(
            clean_latent=clean,
            noisy_latent=noisy,
            noise=noise,
            alpha=0.9,
            sigma=0.1,
            prediction_type="x0",
        ),
        clean,
    )
    v = prediction_target(
        clean_latent=clean,
        noisy_latent=noisy,
        noise=noise,
        alpha=0.9,
        sigma=0.1,
        prediction_type="v_prediction",
    )
    assert torch.allclose(v, 0.9 * noise - 0.1 * clean)

    with pytest.raises(ValueError, match="prediction_type"):
        prediction_target(
            clean_latent=clean,
            noisy_latent=noisy,
            noise=noise,
            alpha=0.9,
            sigma=0.1,
            prediction_type="bad",
        )


def test_consistency_distillation_loss_reports_components() -> None:
    student = torch.tensor([[1.0, 2.0]], requires_grad=True)
    teacher = torch.tensor([[1.5, 1.5]])

    loss = consistency_distillation_loss(
        student,
        teacher,
        student_consistency=student + 1,
        teacher_consistency=teacher + 1,
        consistency_weight=0.5,
    )

    assert loss.total.requires_grad is True
    assert loss.prediction.item() > 0.0
    assert loss.consistency is not None
    assert set(loss.to_dict()) == {"total", "prediction", "consistency"}


def test_consistency_distillation_loss_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="same shape"):
        consistency_distillation_loss(torch.zeros(1, 2), torch.zeros(1, 3))

    with pytest.raises(ValueError, match="provided together"):
        consistency_distillation_loss(
            torch.zeros(1, 2),
            torch.zeros(1, 2),
            student_consistency=torch.zeros(1, 2),
        )


def test_diffusion_sampling_report_and_image_grid_metadata() -> None:
    grid = build_image_grid_record(
        "grid.png",
        ["a cat", "a dog"],
        seed=3,
        teacher_steps=20,
        student_steps=4,
        metadata={"cols": 2},
    )
    report = DiffusionSamplingReport(
        teacher_steps=20,
        student_steps=4,
        scheduler="lcm",
        guidance_scale=1.0,
        latency_ms={"teacher": 100.0, "student": 20.0},
        image_grid=grid,
        metrics={"clip_score": 0.3},
    )

    data = report.to_dict()
    assert data["image_grid"]["path"] == "grid.png"
    assert data["image_grid"]["prompts"] == ["a cat", "a dog"]
    assert data["latency_ms"]["student"] == 20.0

    with pytest.raises(ValueError, match="student_steps"):
        build_image_grid_record(
            "bad.png",
            [],
            seed=0,
            teacher_steps=2,
            student_steps=4,
        )
