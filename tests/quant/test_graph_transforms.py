"""Tests for C6 graph-level quant transforms."""

from __future__ import annotations

import torch
from torch import nn

from xqt.compression.quant.transforms import (
    RotationAbsorbTransform,
    TransformPlan,
    apply_graph_transforms,
    preflight_hadamard_kernel,
)


class _SequentialMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(32, 32, bias=True)
        self.fc2 = nn.Linear(32, 16, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.fc1(inputs))


class _ResidualBranchBlock(nn.Module):
    """Toy residual: branch Linear + residual Linear (U7 non-sequential)."""

    def __init__(self) -> None:
        super().__init__()
        self.branch = nn.Linear(32, 32, bias=True)
        self.residual = nn.Linear(32, 32, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.branch(inputs) + self.residual(inputs)


class _ToyResidualModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block = _ResidualBranchBlock()
        self.head = nn.Linear(32, 8, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.head(self.block(inputs))


def test_convrot_from_linear_preserves_input_already_rotated() -> None:
    from xqt.compression.quant.quantizers.convrot_4bit import ConvRotMixedPrecisionLinear

    linear = nn.Linear(32, 32, bias=False)
    linear.input_already_rotated = True
    conv = ConvRotMixedPrecisionLinear.from_linear(linear, rot_size=16, group_size=16)
    assert conv.input_already_rotated is True
    x = torch.randn(2, 32)
    rotated = conv._rotate_inputs(x)
    assert torch.allclose(rotated, x.to(torch.float32))


def test_rotation_absorb_match_and_apply_on_sequential_linears() -> None:
    torch.manual_seed(0)
    model = _SequentialMLP().eval()

    transform = RotationAbsorbTransform(rot_size=16)
    plan = transform.match(model)
    assert plan is not None
    assert isinstance(plan, TransformPlan)
    assert plan.transform_name == "rotation_absorb"
    assert any("fc1->fc2" in target for target in plan.targets)

    weight_before = model.fc1.weight.detach().clone()
    x = torch.randn(4, 32)
    y_before = model(x).detach().clone()
    report = transform.apply(model, plan)
    assert report.applied is True
    assert report.absorbed_ops
    assert isinstance(report.online_ops, tuple)
    assert report.online_ops == ()
    assert report.required_kernels == ()
    assert not torch.allclose(model.fc1.weight, weight_before)
    flag = getattr(model.fc2, "_xqt_input_already_rotated", None)
    assert flag is not None
    assert bool(flag.item()) is True
    y_after = model(x)
    assert y_after.shape == y_before.shape
    assert torch.isfinite(y_after).all()


def test_rotation_absorb_no_match_on_single_linear() -> None:
    model = nn.Sequential(nn.Linear(16, 8)).eval()
    transform = RotationAbsorbTransform(rot_size=16)
    assert transform.match(model) is None
    reports = apply_graph_transforms(model, [transform])
    assert reports[0].applied is False
    assert "no_match" in reports[0].notes


def test_apply_graph_transforms_reports_required_kernels_when_online() -> None:
    transform = RotationAbsorbTransform(rot_size=4)
    assert "hadamard_groupwise" in transform.required_kernels


def test_rotation_absorb_residual_branch_declares_online_hadamard() -> None:
    """U7: residual/branch toy keeps online hadamard + preflight note."""

    torch.manual_seed(1)
    model = _ToyResidualModel().eval()
    transform = RotationAbsorbTransform(rot_size=16)
    plan = transform.match(model)
    assert plan is not None
    assert plan.online_ops or any(
        p.get("kind") == "residual_branch"
        for p in plan.metadata.get("online_pairs", [])
    )
    preflight = plan.metadata.get("hadamard_preflight") or preflight_hadamard_kernel()
    assert preflight["kernel"] == "hadamard_groupwise"
    assert preflight["status"] in {"reference_only", "engine_registered"}

    x = torch.randn(3, 32)
    y0 = model(x).detach().clone()
    report = transform.apply(model, plan)
    assert report.applied is True
    assert report.online_ops
    assert "hadamard_groupwise" in report.required_kernels
    assert any("hadamard_preflight" in n or "online_" in n for n in report.notes)
    y1 = model(x)
    assert y1.shape == y0.shape
    assert torch.isfinite(y1).all()


def test_preflight_hadamard_kernel_honest() -> None:
    info = preflight_hadamard_kernel()
    assert info["kernel"] == "hadamard_groupwise"
    assert info["status"] == "engine_registered"
    assert "tilelang" in info["providers"]
    assert info["performance_note"]


def test_residual_branch_whole_model_finite_and_online_declared() -> None:
    """V7: residual/branch toy keeps online hadamard; whole-model outputs finite."""

    torch.manual_seed(2)
    model = _ToyResidualModel().eval()
    transform = RotationAbsorbTransform(rot_size=16)
    plan = transform.match(model)
    assert plan is not None
    report = transform.apply(model, plan)
    assert report.online_ops
    assert "hadamard_groupwise" in report.required_kernels
    x = torch.randn(5, 32)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (5, 8)
    assert torch.isfinite(y).all()
    assert any("online_" in n or "hadamard_preflight" in n for n in report.notes)
