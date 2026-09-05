"""Tests for Layer-Sequential Quantization Pipeline."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from xqt.contracts.model_structure import (
    ComponentSpec,
    ModelStructureContract,
)
from xqt.contracts.weight_only import AWQGPTQWeightOnlyLinear
from xqt.compression.quant.sequential import (
    LayerSequentialConfig,
    discover_sequential_partition,
    quantize_layer_sequential,
    replace_submodule_in_block,
)
from xqt.kernels.nn.fixtures.smoke_llm import build_smoke_llm


def test_replace_submodule_in_block() -> None:
    class Sub(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(8, 8)

    class Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.sub = Sub()

    block = Block()
    new_lin = nn.Linear(8, 16)
    replace_submodule_in_block(block, "sub.linear", new_lin)
    assert block.sub.linear is new_lin
    assert block.sub.linear.out_features == 16


def test_discover_sequential_partition_smoke_llm() -> None:
    model = build_smoke_llm(
        vocab_size=32,
        hidden_dim=16,
        num_heads=2,
        num_layers=2,
        dim_feedforward=32,
    )

    partition = discover_sequential_partition(model)

    assert "embed" in partition.prefix_module_names
    assert len(partition.blocks) == 2
    assert partition.blocks[0].name == "blocks.0"
    assert partition.blocks[1].name == "blocks.1"
    assert "norm" in partition.postfix_module_names or "lm_head" in partition.postfix_module_names

    # Check discovered linear modules within each block
    block0_linears = partition.blocks[0].quantizable_linears
    assert "q_proj" in block0_linears
    assert "k_proj" in block0_linears
    assert "v_proj" in block0_linears
    assert "out_proj" in block0_linears
    assert "mlp_gate" in block0_linears
    assert "mlp_up" in block0_linears
    assert "mlp_down" in block0_linears
    assert len(block0_linears) == 7


def test_quantize_layer_sequential_rtn() -> None:
    model = build_smoke_llm(
        vocab_size=32,
        hidden_dim=16,
        num_heads=2,
        num_layers=2,
        dim_feedforward=32,
    )
    calibration_inputs = [torch.randint(0, 32, (2, 8)) for _ in range(2)]

    cfg = LayerSequentialConfig(
        target_device="cpu",
        offload_device="cpu",
        bits=4,
        group_size=16,
        method="rtn",
    )

    quantized_model, report = quantize_layer_sequential(
        model,
        calibration_inputs,
        config=cfg,
    )

    assert report.total_blocks == 2
    assert report.quantized_blocks == 2
    assert report.quantized_linear_count == 14

    # Verify linear modules in blocks are replaced
    for block in quantized_model.blocks:
        assert isinstance(block.q_proj, AWQGPTQWeightOnlyLinear)
        assert isinstance(block.k_proj, AWQGPTQWeightOnlyLinear)
        assert isinstance(block.v_proj, AWQGPTQWeightOnlyLinear)
        assert isinstance(block.out_proj, AWQGPTQWeightOnlyLinear)
        assert isinstance(block.mlp_gate, AWQGPTQWeightOnlyLinear)
        assert isinstance(block.mlp_up, AWQGPTQWeightOnlyLinear)
        assert isinstance(block.mlp_down, AWQGPTQWeightOnlyLinear)

    # Postfix lm_head is excluded by default exclude_name_patterns=("head", "classifier")
    assert isinstance(quantized_model.lm_head, nn.Linear)

    # Test forward pass with quantized model
    test_input = torch.randint(0, 32, (2, 8))
    output = quantized_model(test_input)
    assert output.shape == (2, 8, 32)
    assert not torch.isnan(output).any()


def test_quantize_layer_sequential_awq_with_calibration() -> None:
    model = build_smoke_llm(
        vocab_size=32,
        hidden_dim=16,
        num_heads=2,
        num_layers=2,
        dim_feedforward=32,
    )
    calibration_inputs = [torch.randint(0, 32, (2, 8)) for _ in range(4)]

    cfg = LayerSequentialConfig(
        target_device="cpu",
        offload_device="cpu",
        bits=4,
        group_size=16,
        method="awq",
        sample_limit=4,
    )

    quantized_model, report = quantize_layer_sequential(
        model,
        calibration_inputs,
        config=cfg,
    )

    assert report.total_blocks == 2
    assert report.quantized_blocks == 2
    assert report.quantized_linear_count == 14

    test_input = torch.randint(0, 32, (2, 8))
    output = quantized_model(test_input)
    assert output.shape == (2, 8, 32)
    assert not torch.isnan(output).any()


def test_quantize_layer_sequential_respects_contract_keep_high_precision() -> None:
    model = build_smoke_llm(
        vocab_size=32,
        hidden_dim=16,
        num_heads=2,
        num_layers=2,
        dim_feedforward=32,
    )
    calibration_inputs = [torch.randint(0, 32, (2, 8)) for _ in range(2)]

    # Declare out_proj as keep_high_precision in contract
    contract = ModelStructureContract(
        family="llm",
        components=(
            ComponentSpec(role="embedding", paths=("embed",)),
            ComponentSpec(
                role="attention",
                paths=("blocks.0.out_proj", "blocks.1.out_proj"),
                precision_hint="keep_high_precision",
            ),
            ComponentSpec(role="head", paths=("lm_head",)),
        ),
    )

    cfg = LayerSequentialConfig(
        target_device="cpu",
        offload_device="cpu",
        bits=4,
        group_size=16,
        method="rtn",
    )

    quantized_model, report = quantize_layer_sequential(
        model,
        calibration_inputs,
        config=cfg,
        contract=contract,
    )

    # 14 linears total - 2 out_proj kept in high precision = 12 quantized
    assert report.quantized_linear_count == 12
    assert isinstance(quantized_model.blocks[0].out_proj, nn.Linear)
    assert isinstance(quantized_model.blocks[1].out_proj, nn.Linear)
    assert isinstance(quantized_model.blocks[0].q_proj, AWQGPTQWeightOnlyLinear)


def test_awq_weight_only_3bit_and_2bit() -> None:
    torch.manual_seed(42)
    linear = nn.Linear(32, 64)
    x = torch.randn(2, 32)

    # 3-bit test
    linear_w3 = AWQGPTQWeightOnlyLinear.from_linear(
        linear, bits=3, group_size=16, method="rtn"
    )
    assert linear_w3.bits == 3
    assert linear_w3.quantized_weight.dtype == torch.uint8
    out_w3 = linear_w3(x)
    assert out_w3.shape == (2, 64)
    assert not torch.isnan(out_w3).any()

    # 2-bit test
    linear_w2 = AWQGPTQWeightOnlyLinear.from_linear(
        linear, bits=2, group_size=16, method="rtn"
    )
    assert linear_w2.bits == 2
    assert linear_w2.quantized_weight.dtype == torch.uint8
    out_w2 = linear_w2(x)
    assert out_w2.shape == (2, 64)
    assert not torch.isnan(out_w2).any()


def test_quantize_layer_sequential_w3a16_and_w2a16() -> None:
    torch.manual_seed(42)
    model = build_smoke_llm(
        vocab_size=32,
        hidden_dim=16,
        num_heads=2,
        num_layers=2,
        dim_feedforward=32,
    )
    calibration_inputs = [torch.randint(0, 32, (2, 8)) for _ in range(2)]

    # Test W3A16 layer-sequential quantization
    cfg_w3 = LayerSequentialConfig(
        target_device="cpu",
        offload_device="cpu",
        bits=3,
        group_size=16,
        method="rtn",
    )
    model_w3, report_w3 = quantize_layer_sequential(
        model,
        calibration_inputs,
        config=cfg_w3,
    )
    assert report_w3.quantized_blocks == 2
    assert report_w3.quantized_linear_count == 14
    assert model_w3.blocks[0].q_proj.bits == 3

    test_input = torch.randint(0, 32, (2, 4))
    output_w3 = model_w3(test_input)
    assert output_w3.shape == (2, 4, 32)
    assert not torch.isnan(output_w3).any()

    # Test W2A16 layer-sequential quantization
    model2 = build_smoke_llm(
        vocab_size=32,
        hidden_dim=16,
        num_heads=2,
        num_layers=2,
        dim_feedforward=32,
    )
    cfg_w2 = LayerSequentialConfig(
        target_device="cpu",
        offload_device="cpu",
        bits=2,
        group_size=16,
        method="rtn",
    )
    model_w2, report_w2 = quantize_layer_sequential(
        model2,
        calibration_inputs,
        config=cfg_w2,
    )
    assert report_w2.quantized_blocks == 2
    assert report_w2.quantized_linear_count == 14
    assert model_w2.blocks[0].q_proj.bits == 2

    output_w2 = model_w2(test_input)
    assert output_w2.shape == (2, 4, 32)
    assert not torch.isnan(output_w2).any()

