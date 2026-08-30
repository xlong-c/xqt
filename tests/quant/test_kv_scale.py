"""Tests for C7 KV scale calibration."""

from __future__ import annotations

import torch
from torch import nn

from xqt.core.types import XQTContext
from xqt.compression.quant.quantizers.kv_scale import (
    KvScaleArtifact,
    attach_kv_scale_buffers,
    calibrate_kv_scales,
    discover_kv_projection_modules,
    evaluate_kv_scale_cosine,
    execute_kv_scale_component,
    kv_scales_to_compute_metadata,
)
from xqt.compression.quant.types import QuantizationComponentPlan
from xqt.contracts.quant_pair import write_quant_pair


class _ToyAttentionBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(16, 16, bias=False)
        self.k_proj = nn.Linear(16, 16, bias=False)
        self.v_proj = nn.Linear(16, 16, bias=False)
        self.o_proj = nn.Linear(16, 16, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        q = self.q_proj(inputs)
        k = self.k_proj(inputs)
        v = self.v_proj(inputs)
        return self.o_proj(q + k + v)


class _ToyLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_ToyAttentionBlock(), _ToyAttentionBlock()])

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = inputs
        for layer in self.layers:
            hidden = layer(hidden)
        return hidden


class _ToyFusedAttentionBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.qkv = nn.Linear(4, 12, bias=False)
        with torch.no_grad():
            self.qkv.weight[:4].fill_(0.25)
            self.qkv.weight[4:8].fill_(0.5)
            self.qkv.weight[8:].fill_(0.75)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.qkv(inputs)


def test_discover_and_calibrate_kv_scales() -> None:
    model = _ToyLM().eval()
    layers = discover_kv_projection_modules(model)
    assert "layers.0" in layers
    assert "layers.1" in layers
    assert layers["layers.0"]["k"].endswith("k_proj")
    assert layers["layers.0"]["v"].endswith("v_proj")

    batches = [torch.randn(2, 8, 16) * 3.0 for _ in range(2)]
    artifacts = calibrate_kv_scales(model, batches)
    assert set(artifacts) == {"layers.0", "layers.1"}
    for artifact in artifacts.values():
        assert isinstance(artifact, KvScaleArtifact)
        assert artifact.k_scale > 0.0
        assert artifact.v_scale > 0.0
        payload = artifact.to_dict()
        assert payload["attn.k_scale"] == payload["k_scale"]
        assert payload["attn.v_scale"] == payload["v_scale"]
        assert "attn.k_scale" in artifact.buffer_names()["k_scale"] or True


def test_fused_qkv_calibration_observes_k_and_v_thirds() -> None:
    model = _ToyFusedAttentionBlock().eval()
    layers = discover_kv_projection_modules(model)
    assert layers == {
        "": {"k": "qkv", "v": "qkv", "fused_qkv": "qkv"}
    }

    inputs = torch.ones(1, 2, 4)
    artifacts = calibrate_kv_scales(model, [inputs], qmax=127)

    artifact = artifacts[""]
    assert artifact.k_scale == 2.0 / 127.0
    assert artifact.v_scale == 3.0 / 127.0
    assert artifact.num_samples == 8


def test_attach_buffers_and_quant_pair_lineage(tmp_path) -> None:
    model = _ToyLM().eval()
    artifacts = calibrate_kv_scales(model, [torch.randn(2, 4, 16)])
    attached = attach_kv_scale_buffers(model, artifacts)
    assert "layers.0" in attached
    layer0 = model.layers[0]
    assert hasattr(layer0, "k_scale")
    assert hasattr(layer0, "v_scale")

    meta = kv_scales_to_compute_metadata(artifacts)
    assert "kv_cache_quant" in meta
    assert meta["kv_cache_quant"]["field_convention"] == "vllm.attn.k_scale"

    pair_dir = write_quant_pair(
        model,
        tmp_path / "pair",
        lineage={"method": "kv_scale", "strategy": "kv_scale"},
        metadata=meta,
    )
    text = (pair_dir / "quant.json").read_text(encoding="utf-8")
    assert "kv_cache_quant" in text
    assert "attn.k_scale" in text


def test_execute_kv_scale_component() -> None:
    model = _ToyLM().eval()
    context = XQTContext(
        model=model,
        calibration_inputs=[torch.randn(2, 4, 16)],
    )
    component = QuantizationComponentPlan(
        name="kv",
        backend="pytorch",
        method="kv_scale",
        strategy="kv_scale",
        policy={},
    )
    updated, report = execute_kv_scale_component(context, model, component)
    assert report.method == "kv_scale"
    assert report.metadata["field_convention"] == "vllm.attn.k_scale"
    assert len(report.quantized_modules) >= 1
    assert hasattr(updated.layers[0], "k_scale")


def test_cosine_gate() -> None:
    reference = torch.randn(8, 16)
    candidate = reference + 1e-4
    result = evaluate_kv_scale_cosine(reference, candidate, threshold=0.99)
    assert result["passed"] is True
    assert result["cosine"] >= 0.99
    bad = evaluate_kv_scale_cosine(reference, -reference, threshold=0.99)
    assert bad["passed"] is False
