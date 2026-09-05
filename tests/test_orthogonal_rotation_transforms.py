"""Tests for Orthogonal Rotation Transforms and Transactional Graph Rewrite Engine."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from xqt.compression.quant.transforms.orthogonal import (
    OrthogonalRotationTransform,
    build_random_orthogonal_matrix,
)
from xqt.compression.quant.transforms.engine import (
    RewriteTransactionReport,
    SpeculativeRewriteConfig,
    speculative_graph_rewrite,
)


def test_build_random_orthogonal_matrix() -> None:
    dim = 32
    q1 = build_random_orthogonal_matrix(dim, seed=42)
    assert q1.shape == (dim, dim)

    # Orthogonality: Q @ Q.T == I
    eye = torch.eye(dim, dtype=torch.float32)
    assert torch.allclose(q1 @ q1.t(), eye, atol=1e-5)
    assert torch.allclose(q1.t() @ q1, eye, atol=1e-5)

    # Deterministic with same seed
    q2 = build_random_orthogonal_matrix(dim, seed=42)
    assert torch.equal(q1, q2)

    # Different with different seed
    q3 = build_random_orthogonal_matrix(dim, seed=100)
    assert not torch.equal(q1, q3)


def test_orthogonal_rotation_transform_mathematical_identity() -> None:
    class TwoLayerLinear(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc1 = nn.Linear(16, 32)
            self.fc2 = nn.Linear(32, 8)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.fc2(self.fc1(x))

    torch.manual_seed(42)
    model = TwoLayerLinear()
    x = torch.randn(4, 16)

    # Reference output before transform
    with torch.no_grad():
        y_ref = model(x)

    transform = OrthogonalRotationTransform(seed=123)
    plan = transform.match(model)
    assert plan is not None
    assert "fc1->fc2" in plan.absorbed_ops

    report = transform.apply(model, plan)
    assert report.applied is True
    assert "fc1->fc2" in report.absorbed_ops

    # Output after transform must be numerically identical
    with torch.no_grad():
        y_after = model(x)

    assert torch.allclose(y_ref, y_after, atol=1e-5)


def test_speculative_graph_rewrite_commit_success() -> None:
    class MLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layer1 = nn.Linear(16, 32)
            self.layer2 = nn.Linear(32, 16)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.layer2(self.layer1(x))

    model = MLP()
    calib = [torch.randn(2, 16) for _ in range(4)]

    cfg = SpeculativeRewriteConfig(
        verify_numerics=True,
        max_mean_abs_tolerance=1e-4,
        max_max_abs_tolerance=1e-3,
        min_cosine_similarity=0.9999,
    )

    model_rewritten, report = speculative_graph_rewrite(
        model,
        [OrthogonalRotationTransform(seed=42)],
        calibration_inputs=calib,
        config=cfg,
    )

    assert report.status == "committed"
    assert report.numeric_diff is not None
    assert report.numeric_diff["max_mean_abs"] < 1e-4
    assert report.numeric_diff["cosine_similarity"] >= 0.9999
    assert report.rollback_reason is None


def test_speculative_graph_rewrite_automatic_rollback_on_strict_tolerance() -> None:
    class MLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layer1 = nn.Linear(16, 32)
            self.layer2 = nn.Linear(32, 16)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.layer2(self.layer1(x))

    model = MLP()
    w1_original = model.layer1.weight.clone()
    calib = [torch.randn(2, 16) for _ in range(2)]

    # Set an impossibly strict tolerance to force rollback
    cfg = SpeculativeRewriteConfig(
        verify_numerics=True,
        max_mean_abs_tolerance=1e-15,
        min_cosine_similarity=0.999999999,
    )

    model_after, report = speculative_graph_rewrite(
        model,
        [OrthogonalRotationTransform(seed=42)],
        calibration_inputs=calib,
        config=cfg,
    )

    assert report.status == "rolled_back"
    assert report.rollback_reason is not None
    # Verify model weights are restored to pristine original state
    assert torch.equal(model.layer1.weight, w1_original)
