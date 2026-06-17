import pytest
import torch
from torch import nn

from xqt.distill.hooks import ModuleOutputCapture, capture_module_outputs
from xqt.distill.losses import (
    distillation_loss,
    feature_distillation_loss,
    kl_divergence_with_temperature,
    relation_distillation_loss,
)


def test_kl_divergence_with_temperature_is_small_for_matching_logits() -> None:
    logits = torch.tensor([[1.0, 2.0, 3.0], [2.0, 1.0, 0.5]])

    loss = kl_divergence_with_temperature(logits, logits, temperature=2.0)

    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_distillation_loss_combines_soft_hard_and_feature_terms() -> None:
    student_logits = torch.tensor([[1.0, 0.0], [0.2, 0.8]], requires_grad=True)
    teacher_logits = torch.tensor([[2.0, -1.0], [0.1, 1.4]])
    targets = torch.tensor([0, 1])
    student_features = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    teacher_features = torch.tensor([[1.0, 2.5], [3.5, 4.0]])

    breakdown = distillation_loss(
        student_logits,
        teacher_logits,
        targets=targets,
        temperature=2.0,
        alpha=0.7,
        feature_student=student_features,
        feature_teacher=teacher_features,
        feature_weight=0.2,
        relation_student=student_features,
        relation_teacher=teacher_features,
        relation_weight=0.1,
    )

    assert breakdown.total.requires_grad is True
    assert breakdown.soft_target.item() >= 0.0
    assert breakdown.hard_target is not None
    assert breakdown.feature is not None
    assert breakdown.relation is not None
    assert set(breakdown.to_dict()) == {
        "total",
        "soft_target",
        "hard_target",
        "feature",
        "relation",
    }


def test_distillation_loss_rejects_invalid_inputs() -> None:
    logits = torch.zeros(2, 3)

    with pytest.raises(ValueError, match="temperature must be positive"):
        kl_divergence_with_temperature(logits, logits, temperature=0)

    with pytest.raises(ValueError, match="same shape"):
        kl_divergence_with_temperature(logits, torch.zeros(2, 2))

    with pytest.raises(ValueError, match="alpha must be"):
        distillation_loss(logits, logits, alpha=1.5)

    with pytest.raises(ValueError, match="provided together"):
        distillation_loss(logits, logits, feature_student=logits)


def test_feature_and_relation_losses_match_identical_features() -> None:
    features = torch.randn(4, 3, 2)

    assert feature_distillation_loss(features, features).item() == pytest.approx(0.0)
    assert relation_distillation_loss(features, features).item() == pytest.approx(0.0)


def test_module_output_capture_records_named_outputs_and_removes_hooks() -> None:
    model = nn.Sequential(
        nn.Linear(3, 4),
        nn.ReLU(),
        nn.Linear(4, 2),
    )
    inputs = torch.randn(2, 3)

    outputs = capture_module_outputs(model, ["0", "2"], inputs)

    assert set(outputs) == {"0", "2"}
    assert outputs["0"].shape == (2, 4)
    assert outputs["2"].shape == (2, 2)

    with pytest.raises(KeyError, match="Modules not found"):
        with ModuleOutputCapture(model, ["missing"]):
            model(inputs)
