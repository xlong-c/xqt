"""Tests for XQT industrial model package exporter (.xqtpkg)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch import nn

from xqt.compression.quant.quantizers.awq_gptq_weight_only import quantize_with_awq_weight_only
from xqt.export.package import export_model_package, load_model_package_manifest


class _MiniMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(32, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.relu(self.fc1(x)))


def test_export_model_package_directory(tmp_path: Path) -> None:
    model = _MiniMLP().eval()
    quantized = quantize_with_awq_weight_only(
        model,
        policy={"include_module_types": ["Linear"], "bits": 4, "group_size": 16},
        strategy="w4a16_int4",
        inplace=False,
    )
    out_dir = tmp_path / "mini_mlp_pkg"
    report = export_model_package(
        quantized,
        out_dir,
        package_name="mini_mlp_awq",
        version="0.1.0",
        backend="tilelang",
        runtime_config={"max_batch_size": 8},
        compute_config={"block_m": 64, "block_n": 64},
    )

    assert report.succeeded
    assert report.package_name == "mini_mlp_awq"
    assert report.module_count == 2
    assert report.tensor_count >= 4
    assert (out_dir / "manifest.json").is_file()
    assert (out_dir / "config.json").is_file()
    assert (out_dir / "compute.json").is_file()
    assert (out_dir / "weights" / "model.safetensors").is_file()

    # Verify manifest integrity
    manifest = load_model_package_manifest(out_dir)
    assert manifest["package_name"] == "mini_mlp_awq"
    assert manifest["version"] == "0.1.0"
    assert manifest["backend"] == "tilelang"
    assert "checksums" in manifest
    assert len(manifest["checksums"]) == 3


def test_export_model_package_archive(tmp_path: Path) -> None:
    model = _MiniMLP().eval()
    archive_path = tmp_path / "model.xqtpkg"

    report = export_model_package(
        model,
        archive_path,
        package_name="raw_mlp",
        version="1.0.0",
        archive=True,
    )

    assert report.succeeded
    assert archive_path.is_file()
    assert report.archive_path == str(archive_path)

    # Verify load_model_package_manifest can read directly from .xqtpkg archive
    manifest = load_model_package_manifest(archive_path)
    assert manifest["package_name"] == "raw_mlp"
    assert manifest["version"] == "1.0.0"
    assert "checksums" in manifest
