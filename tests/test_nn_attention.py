from __future__ import annotations

import pytest
import torch

from xqt import nn as xqt_nn
from xqt.conversion import convert
from xqt.contracts import ModuleContract


def test_attention_torch_forward_matches_shape() -> None:
    module = xqt_nn.Attention(64, heads=4, engine="torch")
    x = torch.randn(2, 8, 64)
    out = module(x)
    assert out.shape == (2, 8, 64)
    assert module.runtime_config()["engine"] == "torch"


def test_attention_configure_runtime_updates_precision() -> None:
    module = xqt_nn.Attention(32, heads=4, engine="torch")
    module.configure_runtime(engine="torch", activation_dtype="fp16", weight_dtype="fp16")
    cfg = module.runtime_config()
    assert cfg["activation"] == "fp16"
    assert cfg["weight"] == "fp16"
    assert cfg["heads"] == 4


def test_transformer_block_torch_forward_preserves_shape() -> None:
    block = xqt_nn.TransformerBlock(64, heads=4, ffn_mult=2, engine="torch")
    x = torch.randn(2, 8, 64)
    out = block(x)
    assert out.shape == (2, 8, 64)
    assert block.runtime_config()["attention"]["engine"] == "torch"


def test_convert_attention_torch_attaches_module_contract() -> None:
    module = xqt_nn.Attention(32, heads=4, engine="torch")
    result = convert(module, engine="torch", return_result=True)
    assert isinstance(result.contract, ModuleContract)
    assert result.contract.operator_kind == "attention"
    assert result.converted is False
    assert getattr(result.model, "_xqt_module_contract") == result.contract.to_dict()


def test_convert_transformer_block_torch_attaches_module_contract() -> None:
    module = xqt_nn.TransformerBlock(32, heads=4, ffn_mult=2, engine="torch")
    result = convert(module, engine="torch", return_result=True)
    assert result.contract.operator_kind == "transformer_block"
    assert result.converted is False
    assert getattr(result.model, "_xqt_module_contract") == result.contract.to_dict()


def test_convert_linear_torch_path_attaches_module_contract() -> None:
    module = torch.nn.Linear(16, 8)
    result = convert(module, engine="torch", return_result=True)
    assert result.contract.operator_kind == "linear"
    assert getattr(result.model, "_xqt_module_contract") == result.contract.to_dict()


def test_convert_attention_tilelang_materialize_attaches_contract() -> None:
    module = xqt_nn.Attention(64, heads=4, engine="torch")
    result = convert(module, engine="tilelang", return_result=True)
    assert result.converted is True
    assert isinstance(result.contract, ModuleContract)
    assert result.contract.operator_kind == "attention"
    assert getattr(result.model, "_xqt_module_contract") == result.contract.to_dict()


def test_convert_attention_tilelang_forward_preserves_shape() -> None:
    module = xqt_nn.Attention(32, heads=4, engine="torch")
    converted = convert(module, engine="tilelang")
    x = torch.randn(2, 8, 32)
    out = converted(x)
    assert out.shape == (2, 8, 32)


def test_convert_attention_tilelang_cpu_matches_eager_reference() -> None:
    torch.manual_seed(0)
    module = xqt_nn.Attention(32, heads=4, engine="torch")
    with torch.no_grad():
        for param in module.parameters():
            param.normal_(0.0, 0.02)
    converted = convert(module, engine="tilelang")
    x = torch.randn(2, 8, 32)
    with torch.no_grad():
        reference = module(x)
        actual = converted(x)
    torch.testing.assert_close(actual, reference, atol=1e-4, rtol=1e-4)


def test_convert_transformer_block_tilelang_materializes_internal_attention() -> None:
    from xqt.operator_opt.tilelang_wrappers import _TileLangXqtAttentionWrapper

    module = xqt_nn.TransformerBlock(32, heads=4, ffn_mult=2, engine="torch")
    result = convert(module, engine="tilelang", return_result=True)
    assert result.contract.operator_kind == "transformer_block"
    assert result.converted is True
    assert isinstance(result.model.attn, _TileLangXqtAttentionWrapper)
    assert "block-level" in str(result.report.get("reason", "")).lower()
    x = torch.randn(2, 8, 32)
    assert result.model(x).shape == (2, 8, 32)


@pytest.mark.parametrize(
    ("batch", "seq", "dim", "heads"),
    [
        (1, 8, 32, 4),
        (2, 16, 64, 8),
        (4, 32, 64, 4),
        (8, 64, 128, 8),
    ],
)
def test_convert_attention_tilelang_multi_shape(
    batch: int,
    seq: int,
    dim: int,
    heads: int,
) -> None:
    module = xqt_nn.Attention(dim, heads=heads, engine="torch")
    converted = convert(module, engine="tilelang")
    x = torch.randn(batch, seq, dim)
    out = converted(x)
    assert out.shape == (batch, seq, dim)


@pytest.mark.parametrize(
    ("batch", "seq", "dim", "heads"),
    [
        (1, 8, 32, 4),
        (2, 16, 64, 4),
        (4, 32, 64, 8),
    ],
)
def test_convert_transformer_block_tilelang_multi_shape(
    batch: int,
    seq: int,
    dim: int,
    heads: int,
) -> None:
    module = xqt_nn.TransformerBlock(dim, heads=heads, ffn_mult=2, engine="torch")
    converted = convert(module, engine="tilelang")
    x = torch.randn(batch, seq, dim)
    assert converted(x).shape == (batch, seq, dim)
