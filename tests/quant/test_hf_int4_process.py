"""T3: HF GPTQ/AWQ int32 process_weights_after_loading fixtures."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch import nn

from xqt.core.errors import XQTArtifactError
from xqt.quant.quantizers.awq_gptq_weight_only import AWQGPTQWeightOnlyLinear
from xqt.runtime.bridges.hf_int4_layout import process_weights_after_loading
from xqt.runtime.bridges.hf_int4_pack import awq_reverse_pack_order
from xqt.runtime.bridges.weight_io import load_weight_state_dict


def _pack_gptq_int32(codes_out_in: torch.Tensor, *, bits: int = 4) -> torch.Tensor:
    """Pack signed/unsigned codes [out, in] to GPTQ [K//pack, N] int32."""

    pack = 32 // bits
    out_features, in_features = codes_out_in.shape
    assert in_features % pack == 0
    # use unsigned 0..15 style for packing then process centers
    unsigned = codes_out_in.to(torch.int32)
    if unsigned.min() < 0:
        unsigned = unsigned + (1 << (bits - 1))
    mat = unsigned.T.contiguous()  # K, N
    reshaped = mat.reshape(in_features // pack, pack, out_features)
    shifts = torch.arange(0, 32, bits, dtype=torch.int32)
    return (reshaped << shifts.view(1, -1, 1)).sum(dim=1).to(torch.int32)


def _pack_awq_int32(codes_out_in: torch.Tensor, *, bits: int = 4) -> torch.Tensor:
    """Pack [out, in] unsigned codes to AWQ [K, N//pack] int32 with reverse order."""

    pack = 32 // bits
    out_features, in_features = codes_out_in.shape
    assert out_features % pack == 0
    order = awq_reverse_pack_order(bits)
    inv = [0] * pack
    for src, dst in enumerate(order):
        inv[dst] = src
    shifts = torch.arange(0, 32, bits, dtype=torch.int32)
    mat = codes_out_in.to(torch.int32).T.contiguous()  # K, N
    n_pack = out_features // pack
    rows = []
    for k in range(in_features):
        row = mat[k]
        words = []
        for p in range(n_pack):
            chunk = row[p * pack : (p + 1) * pack]
            reordered = torch.zeros(pack, dtype=torch.int32)
            for i in range(pack):
                reordered[order[i]] = chunk[i]
            words.append(int((reordered << shifts).sum().item()))
        rows.append(torch.tensor(words, dtype=torch.int32))
    return torch.stack(rows, dim=0)


def test_process_gptq_int32_k_pack_layout() -> None:
    bits = 4
    group_size = 32
    in_features = 64
    out_features = 32
    # unsigned codes 0..15
    codes = torch.randint(0, 16, (out_features, in_features), dtype=torch.int32)
    qweight = _pack_gptq_int32(codes, bits=bits)
    assert qweight.shape == (in_features // (32 // bits), out_features)
    groups = in_features // group_size
    scales = torch.rand(groups, out_features) * 0.05 + 0.01
    module = process_weights_after_loading(
        method="gptq",
        bits=bits,
        group_size=group_size,
        in_features=in_features,
        out_features=out_features,
        qweight=qweight,
        scales=scales,
        qzeros=None,
        bias=None,
        g_idx=None,
    )
    assert isinstance(module, AWQGPTQWeightOnlyLinear)
    assert module.dequantize_weight().shape == (out_features, in_features)
    x = torch.randn(2, in_features)
    y = module(x)
    assert y.shape == (2, out_features)
    assert torch.isfinite(y).all()


def test_process_awq_int32_out_pack_layout() -> None:
    bits = 4
    group_size = 32
    in_features = 64
    out_features = 32
    codes = torch.randint(0, 16, (out_features, in_features), dtype=torch.int32)
    qweight = _pack_awq_int32(codes, bits=bits)
    assert qweight.shape == (in_features, out_features // (32 // bits))
    groups = in_features // group_size
    scales = torch.rand(out_features, groups) * 0.05 + 0.01
    module = process_weights_after_loading(
        method="awq",
        bits=bits,
        group_size=group_size,
        in_features=in_features,
        out_features=out_features,
        qweight=qweight,
        scales=scales,
        qzeros=None,
        bias=torch.zeros(out_features),
        g_idx=None,
    )
    assert isinstance(module, AWQGPTQWeightOnlyLinear)
    assert module.bias is not None
    y = module(torch.randn(3, in_features))
    assert y.shape == (3, out_features)


def test_process_desc_act_g_idx_absorbed_matches_reference() -> None:
    """U3: non-sequential g_idx is absorbed; dequant ≈ g_idx reference path."""

    from xqt.runtime.bridges.hf_int4_layout import ProcessWeightsResult
    from xqt.runtime.bridges.hf_int4_pack import normalize_scales

    bits = 4
    group_size = 32
    in_features = 64
    out_features = 16
    torch.manual_seed(0)
    codes = torch.randint(0, 16, (out_features, in_features), dtype=torch.int32)
    qweight = _pack_gptq_int32(codes, bits=bits)
    scales = torch.rand(in_features // group_size, out_features) * 0.02 + 0.01
    g_idx = torch.randperm(in_features) // group_size

    sequential = process_weights_after_loading(
        method="gptq",
        bits=bits,
        group_size=group_size,
        in_features=in_features,
        out_features=out_features,
        qweight=qweight,
        scales=scales,
        qzeros=None,
        bias=None,
        g_idx=None,
        require_g_idx=True,
    )
    assert isinstance(sequential, ProcessWeightsResult)
    assert sequential.g_idx_applied is None

    result = process_weights_after_loading(
        method="gptq",
        bits=bits,
        group_size=group_size,
        in_features=in_features,
        out_features=out_features,
        qweight=qweight,
        scales=scales,
        qzeros=None,
        bias=None,
        g_idx=g_idx,
        require_g_idx=True,
    )
    assert isinstance(result, ProcessWeightsResult)
    assert result.g_idx_applied is True
    module = result.module
    assert isinstance(module, AWQGPTQWeightOnlyLinear)

    from xqt.runtime.bridges.hf_int4_pack import gptq_qweight_to_signed_matrix

    signed = gptq_qweight_to_signed_matrix(
        qweight, bits=bits, in_features=in_features, out_features=out_features
    )
    scale = normalize_scales(
        scales,
        out_features=out_features,
        in_features=in_features,
        group_size=group_size,
    )
    g = g_idx.to(torch.long)
    ref = signed * scale[:, g, 0]
    got = module.dequantize_weight()
    assert torch.allclose(got, ref, atol=0.15, rtol=0.05)


def test_process_sequential_g_idx_path() -> None:
    from xqt.runtime.bridges.hf_int4_layout import ProcessWeightsResult

    bits = 4
    group_size = 32
    in_features = 64
    out_features = 16
    codes = torch.randint(0, 16, (out_features, in_features), dtype=torch.int32)
    qweight = _pack_gptq_int32(codes, bits=bits)
    scales = torch.rand(in_features // group_size, out_features) * 0.02 + 0.01
    g_idx = torch.arange(in_features) // group_size
    result = process_weights_after_loading(
        method="gptq",
        bits=bits,
        group_size=group_size,
        in_features=in_features,
        out_features=out_features,
        qweight=qweight,
        scales=scales,
        qzeros=None,
        bias=None,
        g_idx=g_idx,
        require_g_idx=True,
    )
    assert isinstance(result, ProcessWeightsResult)
    assert result.g_idx_applied is True
    y = result.module(torch.randn(2, in_features))
    assert y.shape == (2, out_features)


def test_sharded_safetensors_empty_weight_map_fails(tmp_path: Path) -> None:
    index = tmp_path / "model.safetensors.index.json"
    index.write_text('{"weight_map": {}}', encoding="utf-8")
    with pytest.raises(XQTArtifactError, match="weight_map"):
        load_weight_state_dict(index)


def test_sharded_safetensors_two_shard_merge(tmp_path: Path) -> None:
    """U5: index.json weight_map merges multiple safetensors shards."""

    from safetensors.torch import save_file

    shard_a = tmp_path / "model-00001-of-00002.safetensors"
    shard_b = tmp_path / "model-00002-of-00002.safetensors"
    t_a = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    t_b = torch.arange(4, dtype=torch.float32).reshape(2, 2)
    save_file({"fc.weight": t_a}, str(shard_a))
    save_file({"fc.bias": t_b.reshape(-1)[:2]}, str(shard_b))
    index = tmp_path / "model.safetensors.index.json"
    index.write_text(
        json.dumps(
            {
                "metadata": {"total_size": 1},
                "weight_map": {
                    "fc.weight": shard_a.name,
                    "fc.bias": shard_b.name,
                },
            }
        ),
        encoding="utf-8",
    )
    state = load_weight_state_dict(index)
    assert set(state) == {"fc.weight", "fc.bias"}
    assert torch.equal(state["fc.weight"], t_a)
    assert state["fc.bias"].shape == (2,)


def test_sharded_safetensors_missing_shard_reports_name(tmp_path: Path) -> None:
    index = tmp_path / "model.safetensors.index.json"
    index.write_text(
        json.dumps(
            {
                "weight_map": {
                    "fc.weight": "missing-shard.safetensors",
                }
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(XQTArtifactError, match="missing-shard.safetensors"):
        load_weight_state_dict(index)


def test_process_awq_with_float_qzeros() -> None:
    bits = 4
    group_size = 32
    in_features = 64
    out_features = 32
    codes = torch.randint(0, 16, (out_features, in_features), dtype=torch.int32)
    qweight = _pack_awq_int32(codes, bits=bits)
    groups = in_features // group_size
    scales = torch.rand(out_features, groups) * 0.05 + 0.01
    qzeros = torch.full((out_features, groups), 8.0)
    module = process_weights_after_loading(
        method="awq",
        bits=bits,
        group_size=group_size,
        in_features=in_features,
        out_features=out_features,
        qweight=qweight,
        scales=scales,
        qzeros=qzeros,
        bias=None,
        g_idx=None,
    )
    assert module.dequantize_weight().shape == (out_features, in_features)
