"""Tests for C8 MoE expert weight-only quantization."""

from __future__ import annotations

import torch
from torch import nn

from xqt.core.types import XQTContext
from xqt.quant.policy import classify_moe_module, is_moe_expert_module, is_moe_router_module
from xqt.quant.quantizers.awq_gptq_weight_only import AWQGPTQWeightOnlyLinear
from xqt.quant.quantizers.moe_weight_only import (
    execute_moe_weight_only_component,
    list_moe_module_roles,
    quantize_moe_experts_weight_only,
)
from xqt.quant.types import QuantizationComponentPlan


class _ToyMoEBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.router = nn.Linear(16, 2, bias=False)
        self.experts = nn.ModuleList(
            [
                nn.Linear(16, 32, bias=True),
                nn.Linear(16, 32, bias=True),
            ]
        )
        self.shared_expert = nn.Linear(16, 32, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        scores = self.router(inputs)
        weights = torch.softmax(scores, dim=-1)
        out = self.shared_expert(inputs) * 0.0
        for index, expert in enumerate(self.experts):
            out = out + expert(inputs) * weights[..., index : index + 1]
        return out


def test_moe_policy_patterns() -> None:
    assert is_moe_expert_module("model.layers.0.mlp.experts.0")
    assert is_moe_expert_module("block.shared_expert")
    assert is_moe_router_module("model.layers.0.mlp.router")
    assert classify_moe_module("block.shared_expert") == "shared_expert"
    assert classify_moe_module("block.router") == "router"
    assert classify_moe_module("block.experts.1") == "expert"


def test_quantize_moe_experts_keeps_router() -> None:
    model = _ToyMoEBlock().eval()
    roles = list_moe_module_roles(model)
    assert "experts.0" in roles["expert"]
    assert "experts.1" in roles["expert"]
    assert "router" in roles["router"]
    assert "shared_expert" in roles["shared_expert"]

    result = quantize_moe_experts_weight_only(
        model,
        policy={"bits": 4, "group_size": 16},
        strategy="w4a16_int4",
        method="awq",
        inplace=False,
    )
    assert isinstance(result.model.experts[0], AWQGPTQWeightOnlyLinear)
    assert isinstance(result.model.experts[1], AWQGPTQWeightOnlyLinear)
    assert isinstance(result.model.shared_expert, AWQGPTQWeightOnlyLinear)
    assert isinstance(result.model.router, nn.Linear)
    assert not isinstance(result.model.router, AWQGPTQWeightOnlyLinear)
    assert result.metadata["smoke_only"] is True
    assert result.metadata["no_ep_dispatcher"] is True
    assert "router" in result.router_modules
    layouts = result.metadata.get("expert_layout_reports") or []
    assert len(layouts) >= 2
    for item in layouts:
        assert "bits" in item
        assert "storage_layout" in item
        assert "selected_kernel" in item
        assert item["bits"] == 4
    output = result.model(torch.randn(2, 16))
    assert output.shape == (2, 32)


def test_execute_moe_component() -> None:
    model = _ToyMoEBlock().eval()
    context = XQTContext(model=model, example_inputs=torch.randn(2, 16))
    component = QuantizationComponentPlan(
        name="moe",
        backend="pytorch",
        method="moe_weight_only",
        strategy="w4a16_int4",
        policy={"bits": 4, "group_size": 16, "base_method": "awq"},
    )
    updated, report = execute_moe_weight_only_component(context, model, component)
    assert report.method == "moe_weight_only"
    assert report.metadata["router_modules"]
    assert any("experts" in name for name in report.quantized_modules)
    assert isinstance(updated.experts[0], AWQGPTQWeightOnlyLinear)
    assert isinstance(updated.router, nn.Linear)
