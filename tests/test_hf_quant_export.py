"""Tests for C9 HF quant export and serving config generation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch import nn

from xqt.core.errors import XQTArtifactError
from xqt.export.hf_quant import export_compressed_tensors
from xqt.contracts.external import probe_external_quant_config
from xqt.compression.quant.quantizers.awq_gptq_weight_only import quantize_with_awq_weight_only
from xqt.runtime.serving_config import generate_serving_config, write_serving_config


class _Tiny(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(16, 32, bias=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.fc(inputs)


def test_export_compressed_tensors_round_trip_probe(tmp_path: Path) -> None:
    model = _Tiny().eval()
    quantized = quantize_with_awq_weight_only(
        model,
        policy={
            "include_module_types": ["Linear"],
            "bits": 4,
            "group_size": 16,
        },
        strategy="w4a16_int4",
        inplace=False,
    )
    out = tmp_path / "hf_export"
    report = export_compressed_tensors(quantized, out)
    assert report.format == "compressed_tensors"
    assert (out / "config.json").is_file()
    assert (out / "model.pt").is_file()
    info = probe_external_quant_config(out)
    assert info is not None
    assert info.format == "compressed_tensors"
    assert report.metadata["probe_format"] == "compressed_tensors"


def test_generate_serving_config_from_export(tmp_path: Path) -> None:
    model = _Tiny().eval()
    quantized = quantize_with_awq_weight_only(
        model,
        policy={"include_module_types": ["Linear"], "bits": 4, "group_size": 16},
        strategy="w4a16_int4",
        inplace=False,
    )
    out = tmp_path / "hf_export"
    export_compressed_tensors(quantized, out, format_name="compressed_tensors")
    config = generate_serving_config(out, engine="vllm")
    assert config["engine"] == "vllm"
    assert config["schema"] == "vllm_quantization"
    assert config["quantization"] == "compressed-tensors"
    assert "--quantization compressed-tensors" in config["cli_args"]

    sglang = generate_serving_config(
        None,
        engine="sglang",
        method="awq",
        strategy="w4a16_int4",
    )
    assert sglang["schema"] == "vllm_quantization"
    assert sglang["quantization"] == "awq"

    with_kv = generate_serving_config(
        None,
        engine="vllm",
        method="gptq",
        metadata={"kv_cache_quant": {"dtype": "fp8"}},
    )
    assert with_kv["kv_cache_dtype"] == "fp8_e4m3"

    path = write_serving_config(
        tmp_path / "serve.json",
        quant_pair_path=out,
        engine="vllm",
    )
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["quantization"] == "compressed-tensors"


def test_generate_serving_config_rejects_unknown_engine() -> None:
    with pytest.raises(XQTArtifactError, match="unsupported serving engine"):
        generate_serving_config(None, engine="tensorrt-llm")  # type: ignore[arg-type]


def test_generate_serving_config_rejects_unknown_combo() -> None:
    with pytest.raises(XQTArtifactError, match="cannot derive"):
        generate_serving_config(None, engine="vllm", method="unknown_method_xyz")
