"""Tests for C4 external quant probe / override / materialize (vLLM-aligned)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch import nn

from xqt.core.errors import XQTArtifactError
from xqt.contracts.external import (
    ExternalQuantInfo,
    iter_config_filenames,
    list_supported_external_formats,
    override_external_format,
    probe_external_quant_config,
    resolve_external_quantization,
)
from xqt.quant.quantizers.awq_gptq_weight_only import AWQGPTQWeightOnlyLinear
from xqt.runtime.bridges.external_weight_only import load_external_quantized_model
from xqt.runtime.bridges.hf_int4_layout import process_weights_after_loading


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_probe_gptq_from_config_json(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "config.json",
        {
            "model_type": "qwen2",
            "quantization_config": {
                "quant_method": "gptq",
                "bits": 4,
                "group_size": 128,
                "sym": True,
            },
        },
    )
    info = probe_external_quant_config(tmp_path)
    assert info is not None
    assert info.format == "gptq"
    assert info.bits == 4
    assert info.group_size == 128
    assert info.source_file == "config.json#quantization_config"


def test_probe_awq_from_quantize_config_json(tmp_path: Path) -> None:
    """vLLM get_config_filenames includes quantize_config.json for GPTQ/AWQ."""

    _write_json(
        tmp_path / "quantize_config.json",
        {
            "quant_method": "awq",
            "bits": 4,
            "group_size": 64,
            "zero_point": True,
        },
    )
    info = probe_external_quant_config(tmp_path)
    assert info is not None
    assert info.format == "awq"
    assert info.group_size == 64
    assert info.sym is False
    assert "quantize_config.json" in iter_config_filenames()


def test_probe_awq_from_hf_quant_config(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "hf_quant_config.json",
        {"quant_method": "awq", "bits": 4, "group_size": 64, "zero_point": True},
    )
    info = probe_external_quant_config(tmp_path)
    assert info is not None
    assert info.format == "awq"
    assert info.group_size == 64
    assert info.sym is False


def test_probe_compressed_tensors(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "config.json",
        {
            "quantization_config": {
                "format": "compressed-tensors",
                "config_groups": {
                    "group_0": {
                        "weights": {"num_bits": 4, "group_size": 128, "type": "int"}
                    }
                },
            }
        },
    )
    info = probe_external_quant_config(tmp_path)
    assert info is not None
    assert info.format == "compressed_tensors"
    assert info.bits == 4
    assert info.group_size == 128


def test_probe_conflict_raises(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "config.json",
        {"quantization_config": {"quant_method": "gptq", "bits": 4}},
    )
    _write_json(
        tmp_path / "hf_quant_config.json",
        {"quant_method": "awq", "bits": 4},
    )
    with pytest.raises(XQTArtifactError, match="conflicting"):
        probe_external_quant_config(tmp_path)


def test_override_compatible_marlin_keeps_checkpoint_method() -> None:
    info = ExternalQuantInfo(format="gptq", source_file="x", raw_config={})
    assert override_external_format(info, "gptq_marlin") == "gptq"
    assert override_external_format(info, "auto_gptq") == "gptq"
    assert "compressed_tensors" in list_supported_external_formats()


def test_override_incompatible_method_raises() -> None:
    info = ExternalQuantInfo(format="gptq", source_file="x", raw_config={})
    with pytest.raises(XQTArtifactError, match="incompatible"):
        override_external_format(info, "awq")
    with pytest.raises(XQTArtifactError):
        override_external_format(info, "gguf")


def test_resolve_external_quantization_applies_override(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "config.json",
        {"quantization_config": {"quant_method": "gptq", "bits": 4, "group_size": 32}},
    )
    resolved = resolve_external_quantization(tmp_path, user_quant="gptq_marlin")
    assert resolved.format == "gptq"
    assert resolved.bits == 4


def test_load_external_probe_only(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "config.json",
        {"quantization_config": {"quant_method": "gptq", "bits": 4, "group_size": 128}},
    )
    model, report = load_external_quantized_model(tmp_path)
    assert model is None
    assert report.format == "gptq"
    assert report.loaded is False
    assert any("probe_only" in note for note in report.notes)
    assert report.metadata["lifecycle"][0] == "resolve"


def test_process_weights_xqt_native_roundtrip() -> None:
    """XQT-native packed layout materializes and dequants stably."""

    linear = nn.Linear(64, 32, bias=True)
    ref = AWQGPTQWeightOnlyLinear.from_linear(
        linear, bits=4, group_size=32, method="gptq"
    )
    rebuilt = process_weights_after_loading(
        method="gptq",
        bits=4,
        group_size=32,
        in_features=64,
        out_features=32,
        qweight=ref.quantized_weight,
        scales=ref.weight_scale,
        qzeros=None,
        bias=ref.bias,
        g_idx=None,
    )
    x = torch.randn(2, 64)
    assert torch.allclose(ref(x), rebuilt(x), atol=1e-5, rtol=1e-5)


def test_load_external_materialize_xqt_export(tmp_path: Path) -> None:
    """Export-like XQT packed checkpoint loads into a dense base model."""

    class Tiny(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = nn.Linear(64, 32, bias=True)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.fc(x)

    base = Tiny()
    quant_mod = AWQGPTQWeightOnlyLinear.from_linear(
        base.fc, bits=4, group_size=32, method="gptq"
    )
    state = {
        "fc.qweight": quant_mod.quantized_weight.detach().cpu(),
        "fc.scales": quant_mod.weight_scale.detach().cpu(),
        "fc.bias": quant_mod.bias.detach().cpu(),
    }
    torch.save(state, tmp_path / "model.pt")
    _write_json(
        tmp_path / "config.json",
        {
            "quantization_config": {
                "quant_method": "gptq",
                "bits": 4,
                "group_size": 32,
                "sym": True,
            }
        },
    )

    loaded_base = Tiny()
    quantized, report = load_external_quantized_model(
        tmp_path, base_model=loaded_base, override="gptq_marlin"
    )
    assert report.loaded is True
    assert quantized is not None
    contract = quantized.resolve_runtime_quant_contract()
    assert contract is not None
    assert contract.storage_layout.startswith("xqt_awq_gptq")
    assert contract.quant_spec.weight_dtype in {"int4", "int8"}
    assert "layout_kernel" in quantized.metadata
    assert report.module_count == 1
    assert quantized is not None
    assert isinstance(loaded_base.fc, AWQGPTQWeightOnlyLinear)
    x = torch.randn(3, 64)
    assert torch.allclose(quant_mod(x), loaded_base.fc(x), atol=1e-5, rtol=1e-5)
