"""Tests for weights + quant.json Infer sidecar helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch import nn

from xqt.contracts import QuantizedModel
from xqt.core.errors import XQTArtifactError
from xqt.runtime import (
    HybridInferenceEngine,
    load_quant_pair,
    load_quant_pair_into_model,
    write_quant_pair,
    write_quant_pair_from_quantized,
)


class _TinyLinear(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(8, 4, bias=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.fc(inputs)


def test_write_and_load_quant_pair_roundtrip(tmp_path: Path) -> None:
    model = _TinyLinear().eval()
    with torch.no_grad():
        model.fc.weight.fill_(0.25)
        model.fc.bias.fill_(0.5)

    compute_config = {
        "schema_version": "1.0",
        "default_precision": "w8a8",
        "modules": [
            {
                "name": "fc",
                "compute_contract": "int8_mma",
                "precision": "w8a8",
                "required_capabilities": ["int8_mma"],
                "preferred_engines": ["tilelang"],
            }
        ],
    }
    pair_dir = write_quant_pair(
        model,
        tmp_path / "pair",
        compute_config=compute_config,
        lineage={"backend": "pytorch", "method": "int8_mma", "strategy": "weight_act"},
    )

    assert (pair_dir / "model.pt").is_file()
    assert (pair_dir / "quant.json").is_file()

    sidecar = json.loads((pair_dir / "quant.json").read_text(encoding="utf-8"))
    assert sidecar["artifact_type"] == "xqt_quant_sidecar"
    assert sidecar["weights"]["format"] == "torch_state_dict"
    assert sidecar["compute_config"]["modules"][0]["compute_contract"] == "int8_mma"
    assert "required_engine" not in sidecar["compute_config"]
    assert sidecar["lineage"]["method"] == "int8_mma"

    loaded = load_quant_pair(pair_dir)
    assert loaded.weights_path == (pair_dir / "model.pt").resolve()
    assert loaded.compute_config is not None
    assert loaded.compute_config.default_precision == "w8a8"
    assert loaded.compute_config.modules[0].name == "fc"

    shell = _TinyLinear().eval()
    quantized = load_quant_pair_into_model(shell, pair_dir)
    handoff = quantized.infer_handoff()
    assert handoff["compute_config"] is not None
    assert handoff["compute_config"]["default_precision"] == "w8a8"
    assert torch.allclose(shell.fc.weight, model.fc.weight)
    assert torch.allclose(shell.fc.bias, model.fc.bias)
    assert quantized.method == "int8_mma"
    assert quantized.backend == "pytorch"


def test_write_quant_pair_from_quantized_uses_infer_handoff(tmp_path: Path) -> None:
    model = _TinyLinear().eval()
    quantized = QuantizedModel(
        model=model,
        backend="pytorch",
        method="fp4_weight_only",
        strategy="weight_only",
        compute_config={
            "schema_version": "1.0",
            "default_precision": "w4a16",
            "modules": [
                {
                    "name": "fc",
                    "compute_contract": "fp4_mma",
                    "required_capabilities": ["fp4_mma"],
                }
            ],
        },
    )
    pair_dir = write_quant_pair_from_quantized(quantized, tmp_path / "from_qm")
    restored = load_quant_pair_into_model(_TinyLinear().eval(), pair_dir)
    handoff = restored.infer_handoff()
    assert set(handoff.keys()) == {"model", "compute_config"}
    assert handoff["compute_config"] is not None
    assert handoff["compute_config"]["modules"][0]["compute_contract"] == "fp4_mma"
    assert restored.method == "fp4_weight_only"


def test_load_quant_pair_rejects_path_escape(tmp_path: Path) -> None:
    pair_dir = tmp_path / "bad_pair"
    pair_dir.mkdir()
    (pair_dir / "quant.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "artifact_type": "xqt_quant_sidecar",
                "weights": {
                    "path": "../escape.pt",
                    "format": "torch_state_dict",
                },
                "lineage": {},
                "metadata": {},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(XQTArtifactError, match="escapes the quant pair root"):
        load_quant_pair(pair_dir)


def test_load_quant_pair_rejects_checksum_mismatch(tmp_path: Path) -> None:
    model = _TinyLinear().eval()
    pair_dir = write_quant_pair(model, tmp_path / "checksum_pair")
    (pair_dir / "model.pt").write_bytes(b"corrupted")
    with pytest.raises(XQTArtifactError, match="checksum mismatch"):
        load_quant_pair(pair_dir)


def test_hybrid_engine_from_loaded_quant_pair_without_requant(tmp_path: Path) -> None:
    model = _TinyLinear().eval()
    pair_dir = write_quant_pair(
        model,
        tmp_path / "engine_pair",
        compute_config={
            "schema_version": "1.0",
            "default_precision": "bf16",
            "modules": [
                {
                    "name": "fc",
                    "precision": "bf16",
                    "compute_contract": "fp16_mma",
                    "required_capabilities": ["fp16_mma"],
                }
            ],
        },
        lineage={"backend": "pytorch", "method": "none"},
    )
    shell = _TinyLinear().eval()
    quantized = load_quant_pair_into_model(shell, pair_dir)
    engine = HybridInferenceEngine.from_quantized_model(
        quantized,
        default_precision="bf16",
        apply_policy_on_init=True,
    )
    output = engine(torch.randn(2, 8))
    assert output.shape == (2, 4)
    assert engine.policy is not None
    assert engine.policy.policy_kind == "compute_config"
