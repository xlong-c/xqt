"""Tests for Typed Graph Rewrite, Builtin Patterns, Dry-Run and Transaction Rollback (XQT-012)."""

from __future__ import annotations

import copy
import pytest
import torch
from torch import nn

from xqt.compression.quant.transforms import (
    ActivationQuantTransform,
    DequantGemmTransform,
    FusedActivationQuant,
    FusedDequantGemmLinear,
    FusedNormQuant,
    GraphTransformConfig,
    NormQuantTransform,
    SingleTransformConfig,
    apply_graph_transforms,
    build_graph_transform,
    parse_graph_transform_config,
)
from xqt.core.base import XQTConfigError
from xqt.core.types import XQTContext
from xqt.pipeline.pass_helpers.quant_stage import _maybe_run_graph_transforms


class _ToyTransformerBlock(nn.Module):
    def __init__(self, dim: int = 16) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, dim * 2)
        self.act = nn.GELU()
        self.linear2 = nn.Linear(dim * 2, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = self.linear1(h)
        h = self.act(h)
        return self.linear2(h)


def test_unknown_transform_name_raises_config_error() -> None:
    # 传入未知 transform 必须立即抛出 XQTConfigError，严禁静默吞掉
    with pytest.raises(XQTConfigError, match="Unknown graph transform 'unknown_fusion'"):
        parse_graph_transform_config(["unknown_fusion"])

    with pytest.raises(XQTConfigError, match="Unknown graph transform 'fake_rot'"):
        parse_graph_transform_config([{"name": "fake_rot"}])


def test_dequant_gemm_pattern_dry_run_and_execution() -> None:
    model = _ToyTransformerBlock(dim=8)
    x = torch.randn(2, 4, 8)
    ref_out = model(x)

    transform = DequantGemmTransform()

    # 1. Dry-run 预检：生成 plan，不修改模型
    plan = transform.match(model)
    assert plan is not None
    assert plan.metadata["matched"] is True
    assert "linear1" in plan.targets
    assert "linear2" in plan.targets
    assert isinstance(model.linear1, nn.Linear)

    reports = apply_graph_transforms(model, [transform], dry_run=True)
    assert len(reports) == 1
    assert reports[0].notes == ("dry_run",)
    assert isinstance(model.linear1, nn.Linear)  # 模型未被修改

    # 2. 真实执行：替换为 FusedDequantGemmLinear
    reports_real = apply_graph_transforms(model, [transform], dry_run=False)
    assert reports_real[0].applied is True
    assert isinstance(model.linear1, FusedDequantGemmLinear)
    assert isinstance(model.linear2, FusedDequantGemmLinear)

    # 3. 数值等价性验证
    fused_out = model(x)
    assert torch.allclose(ref_out, fused_out, atol=1e-5, rtol=1e-5)


def test_norm_quant_pattern_dry_run_and_execution() -> None:
    model = _ToyTransformerBlock(dim=8)
    x = torch.randn(2, 4, 8)
    ref_norm_out = model.norm(x)

    transform = NormQuantTransform()

    # 1. Dry-run
    plan = transform.match(model)
    assert plan is not None
    assert plan.metadata["matched"] is True
    assert "norm" in plan.targets

    apply_graph_transforms(model, [transform], dry_run=True)
    assert isinstance(model.norm, nn.LayerNorm)  # 未修改

    # 2. 真实应用
    apply_graph_transforms(model, [transform], dry_run=False)
    assert isinstance(model.norm, FusedNormQuant)

    # 3. 数值等价性
    fused_norm_out = model.norm(x)
    assert torch.allclose(ref_norm_out, fused_norm_out, atol=1e-5, rtol=1e-5)


def test_activation_quant_pattern_dry_run_and_execution() -> None:
    model = _ToyTransformerBlock(dim=8)
    x = torch.randn(2, 4, 8)
    ref_act_out = model.act(x)

    transform = ActivationQuantTransform()

    # 1. Dry-run
    plan = transform.match(model)
    assert plan is not None
    assert plan.metadata["matched"] is True
    assert "act" in plan.targets

    apply_graph_transforms(model, [transform], dry_run=True)
    assert isinstance(model.act, nn.GELU)

    # 2. 真实应用
    apply_graph_transforms(model, [transform], dry_run=False)
    assert isinstance(model.act, FusedActivationQuant)

    # 3. 数值等价性
    fused_act_out = model.act(x)
    assert torch.allclose(ref_act_out, fused_act_out, atol=1e-5, rtol=1e-5)


def test_graph_transform_transaction_rollback_on_failure() -> None:
    model = _ToyTransformerBlock(dim=8)
    orig_linear1 = model.linear1
    orig_norm = model.norm

    class _FaultyTransform:
        name = "faulty_tx"
        required_kernels = ("torch",)

        def match(self, m: nn.Module):
            from xqt.compression.quant.transforms.base import TransformPlan
            return TransformPlan(transform_name="faulty_tx", targets=("linear1",))

        def apply(self, m: nn.Module, plan):
            # 故意修改部分模块后抛出异常
            m.linear1 = nn.Identity()
            raise RuntimeError("Simulated graph rewrite failure during transaction")

    from xqt.compression.quant.transforms.registry import register_graph_transform
    register_graph_transform("faulty_tx", lambda **kwargs: _FaultyTransform())

    ctx = XQTContext()
    ctx.model = model

    class _MockQuant:
        policy = {"graph_transforms": ["faulty_tx"]}

    # 执行时必须抛出异常，并且将 model 完整回滚到原状态
    with pytest.raises(RuntimeError, match="Simulated graph rewrite failure"):
        _maybe_run_graph_transforms(ctx, _MockQuant())

    # 断言模型结构未被破坏（完整回滚）
    assert isinstance(ctx.model.linear1, nn.Linear)
    assert isinstance(ctx.model.norm, nn.LayerNorm)
