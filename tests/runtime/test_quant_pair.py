"""Tests for weights + quant.json Infer sidecar helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch import nn

from xqt.contracts import QuantizedModel
from xqt.contracts.quant_pair import (
    load_quant_pair,
    load_quant_pair_into_model,
    write_quant_pair,
    write_quant_pair_from_quantized,
)
from xqt.core.errors import XQTArtifactError
from xqt.runtime import HybridInferenceEngine


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


def test_quant_pair_roundtrip_preserves_runtime_quant_contract(tmp_path: Path) -> None:
    """U1: contract writes into quant.json and resolves after load."""

    from xqt.contracts import RuntimeQuantContract
    from xqt.compression.quant.types import QuantScheme

    model = _TinyLinear().eval()
    contract = RuntimeQuantContract(
        quant_spec=QuantScheme(
            weight_dtype="int4",
            weight_granularity="groupwise",
            group_size=32,
            activation_dtype=None,
            activation_mode="none",
            sym=True,
        ),
        storage_layout="xqt_awq_gptq_int4_v1",
        required_kernels=("dequant_fp16",),
        global_shape=(4, 8),
        local_shape=(4, 8),
        prefill_supported=True,
        decode_supported=True,
    )
    pair_dir = write_quant_pair(
        model,
        tmp_path / "contract_pair",
        runtime_quant_contract=contract,
        lineage={"backend": "pytorch", "method": "gptq", "strategy": "w4a16_int4"},
    )
    sidecar = json.loads((pair_dir / "quant.json").read_text(encoding="utf-8"))
    assert "runtime_quant_contract" in sidecar["metadata"]
    assert sidecar["metadata"]["runtime_quant_contract"]["storage_layout"] == (
        "xqt_awq_gptq_int4_v1"
    )

    loaded = load_quant_pair(pair_dir)
    restored_contract = loaded.resolve_runtime_quant_contract()
    assert restored_contract is not None
    assert restored_contract == contract

    shell = _TinyLinear().eval()
    quantized = load_quant_pair_into_model(shell, pair_dir)
    assert quantized.resolve_runtime_quant_contract() == contract


def test_write_from_quantized_carries_contract(tmp_path: Path) -> None:
    from xqt.contracts import RuntimeQuantContract
    from xqt.compression.quant.types import QuantScheme

    model = _TinyLinear().eval()
    contract = RuntimeQuantContract(
        quant_spec=QuantScheme(
            weight_dtype="int8",
            weight_granularity="per_channel",
            group_size=None,
            activation_dtype="int8",
            activation_mode="static",
            sym=True,
        ),
        storage_layout="xqt_int8_mma_v1",
        required_kernels=("w8a8_int8_mma",),
        global_shape=(4, 8),
        local_shape=(4, 8),
        prefill_supported=True,
        decode_supported=False,
    )
    quantized = QuantizedModel(
        model=model,
        backend="pytorch",
        method="int8_mma",
        strategy="w8a8_int8",
        compute_config={
            "schema_version": "1.0",
            "default_precision": "w8a8",
            "modules": [{"name": "fc", "compute_contract": "int8_mma"}],
        },
    ).with_runtime_quant_contract(contract)
    pair_dir = write_quant_pair_from_quantized(quantized, tmp_path / "qm_contract")
    restored = load_quant_pair_into_model(_TinyLinear().eval(), pair_dir)
    assert restored.resolve_runtime_quant_contract() == contract
    assert restored.resolve_runtime_quant_contract().decode_supported is False


def test_quant_pair_writes_runtime_manifest_with_contract(tmp_path: Path) -> None:
    """V1: quant.json metadata includes runtime_manifest when contract is set."""

    from xqt.contracts import RuntimeQuantContract, build_runtime_manifest
    from xqt.compression.quant.types import QuantScheme

    model = _TinyLinear().eval()
    contract = RuntimeQuantContract(
        quant_spec=QuantScheme(
            weight_dtype="int4",
            weight_granularity="groupwise",
            group_size=32,
            activation_dtype=None,
            activation_mode="none",
            sym=True,
        ),
        storage_layout="xqt_awq_gptq_int4_v1",
        required_kernels=("dequant_fp16",),
        global_shape=(4, 8),
        local_shape=(4, 8),
        prefill_supported=True,
        decode_supported=True,
    )
    pair_dir = write_quant_pair(
        model,
        tmp_path / "manifest_pair",
        runtime_quant_contract=contract,
        metadata={
            "layout_kernel": {
                "storage_layout": "xqt_awq_gptq_int4_v1",
                "selected_kernel": "dequant_fp16_reference",
                "bits": 4,
                "group_size": 32,
            }
        },
    )
    sidecar = json.loads((pair_dir / "quant.json").read_text(encoding="utf-8"))
    assert "runtime_manifest" in sidecar["metadata"]
    assert sidecar["metadata"]["runtime_manifest"]["contract"]["storage_layout"] == (
        "xqt_awq_gptq_int4_v1"
    )
    assert "dequant_fp16_reference" in sidecar["metadata"]["runtime_manifest"][
        "selected_kernels"
    ]
    manifest = build_runtime_manifest(pair_dir)
    assert manifest.contract is not None
    assert manifest.contract == contract


def test_load_quant_pair_absent_contract_is_none(tmp_path: Path) -> None:
    model = _TinyLinear().eval()
    pair_dir = write_quant_pair(model, tmp_path / "no_contract")
    loaded = load_quant_pair(pair_dir)
    assert loaded.resolve_runtime_quant_contract() is None
    quantized = load_quant_pair_into_model(_TinyLinear().eval(), pair_dir)
    assert quantized.resolve_runtime_quant_contract() is None


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


def test_write_and_load_quant_pair_safetensors_roundtrip(tmp_path: Path) -> None:
    model = _TinyLinear().eval()
    with torch.no_grad():
        model.fc.weight.fill_(0.125)
        model.fc.bias.fill_(0.375)

    compute_config = {
        "schema_version": "1.0",
        "default_precision": "w8a8",
        "modules": [
            {
                "name": "fc",
                "compute_contract": "int8_mma",
                "precision": "w8a8",
                "required_capabilities": ["int8_mma"],
            }
        ],
    }
    pair_dir = write_quant_pair(
        model,
        tmp_path / "safetensors_pair",
        compute_config=compute_config,
        lineage={"backend": "pytorch", "method": "int8_mma", "strategy": "weight_act"},
        weights_format="safetensors",
    )

    assert (pair_dir / "model.safetensors").is_file()
    assert (pair_dir / "quant.json").is_file()

    sidecar = json.loads((pair_dir / "quant.json").read_text(encoding="utf-8"))
    assert sidecar["artifact_type"] == "xqt_quant_sidecar"
    assert sidecar["weights"]["format"] == "safetensors"
    assert sidecar["weights"]["path"] == "model.safetensors"

    loaded = load_quant_pair(pair_dir)
    assert loaded.weights_path == (pair_dir / "model.safetensors").resolve()

    # Also test loading by passing the safetensors file directly
    loaded_by_file = load_quant_pair(pair_dir / "model.safetensors")
    assert loaded_by_file.weights_path == loaded.weights_path

    shell = _TinyLinear().eval()
    quantized = load_quant_pair_into_model(shell, pair_dir)
    assert torch.allclose(shell.fc.weight, model.fc.weight)
    assert torch.allclose(shell.fc.bias, model.fc.bias)
    assert quantized.method == "int8_mma"


def test_write_quant_pair_from_quantized_safetensors(tmp_path: Path) -> None:
    model = _TinyLinear().eval()
    quantized = QuantizedModel(
        model=model,
        backend="pytorch",
        method="fp4_weight_only",
        strategy="weight_only",
    )
    pair_dir = tmp_path / "from_quantized_safetensors"
    write_quant_pair_from_quantized(
        quantized,
        pair_dir,
        weights_format="safetensors",
    )

    assert (pair_dir / "model.safetensors").is_file()
    loaded = load_quant_pair(pair_dir)
    assert loaded.manifest.weights["format"] == "safetensors"

