"""Tests for profile-guided deployment export in export_model_package."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch import nn

from xqt.export.package import export_model_package
from xqt.kernels.timing.cache import UnifiedKernelTimingCache


class SimpleLinearModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(32, 64)
        self.fc2 = nn.Linear(64, 16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.fc1(x))


def test_export_package_with_optimal_kernel_routes(tmp_path: Path) -> None:
    model = SimpleLinearModel()
    pkg_dir = tmp_path / "model_with_routes"

    custom_routes = {
        "attention@triton@sm_89@float16@1x8x1x1024x64": {
            "median_latency_us": 28.5,
            "preset_name": "sm89_decode_fast",
        },
        "linear@tilelang@sm_89@float16@1x32x64": {
            "median_latency_us": 14.2,
            "preset_name": "tilelang_linear_tuned",
        },
    }

    report = export_model_package(
        model_or_quant=model,
        output_path=pkg_dir,
        package_name="test_profile_pkg",
        optimal_kernel_routes=custom_routes,
    )

    compute_file = Path(report.compute_path)
    assert compute_file.is_file()

    compute_data = json.loads(compute_file.read_text(encoding="utf-8"))
    assert "optimal_kernel_routes" in compute_data
    assert compute_data["optimal_kernel_routes"] == custom_routes


def test_export_package_with_timing_cache_auto_extraction(tmp_path: Path) -> None:
    model = SimpleLinearModel()
    pkg_dir = tmp_path / "model_with_cache"

    cache = UnifiedKernelTimingCache(cache_path=tmp_path / "cache.json")
    cache.record_measurement(
        op="linear",
        backend="tilelang",
        arch="sm_89",
        dtype="float16",
        shape=(1, 32, 64),
        median_latency_us=15.0,
        preset_name="auto_preset",
        verified_correct=True,
    )

    report = export_model_package(
        model_or_quant=model,
        output_path=pkg_dir,
        package_name="test_cache_pkg",
        timing_cache=cache,
    )

    compute_file = Path(report.compute_path)
    assert compute_file.is_file()

    compute_data = json.loads(compute_file.read_text(encoding="utf-8"))
    assert "optimal_kernel_routes" in compute_data
    routes = compute_data["optimal_kernel_routes"]
    matched_key = [k for k in routes if "linear@tilelang@sm_89@float16@1x32x64" in k]
    assert len(matched_key) == 1
    assert routes[matched_key[0]]["median_latency_us"] == pytest.approx(15.0)
    assert routes[matched_key[0]]["preset_name"] == "auto_preset"
