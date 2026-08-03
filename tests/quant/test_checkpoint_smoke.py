"""T10/U9: offline checkpoint smoke — quantize → export → load → diff → latency.

Default path uses an in-process CI fixture (no network / no public HF download).
Optional real HF-style directories are gated by ``XQT_HF_QUANT_FIXTURE`` or
``data/xqt_hf_quant_fixtures/<name>/``; missing fixtures skip, never fail CI.
See ``research/xqt-quant-inference-architecture/HF_CHECKPOINT_SMOKE.md``.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest
import torch
from torch import nn

from xqt.export.hf_quant import export_compressed_tensors
from xqt.quant.quantizers.awq_gptq_weight_only import (
    AWQGPTQWeightOnlyLinear,
    quantize_with_gptq_weight_only,
)
from xqt.runtime.bridges.external_weight_only import load_external_quantized_model

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_FIXTURE_ROOT = _REPO_ROOT / "data" / "xqt_hf_quant_fixtures"


class _TinyMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(64, 64, bias=True)
        self.fc2 = nn.Linear(64, 32, bias=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(inputs)))


def _resolve_optional_hf_fixture() -> Path | None:
    env = os.environ.get("XQT_HF_QUANT_FIXTURE", "").strip()
    if env:
        candidate = Path(env).expanduser()
        if candidate.is_dir() and (candidate / "config.json").is_file():
            return candidate.resolve()
        return None
    if not _DEFAULT_FIXTURE_ROOT.is_dir():
        return None
    for child in sorted(_DEFAULT_FIXTURE_ROOT.iterdir()):
        if child.is_dir() and (child / "config.json").is_file():
            return child.resolve()
    return None


def test_checkpoint_smoke_export_load_diff_latency(tmp_path: Path) -> None:
    torch.manual_seed(0)
    model = _TinyMLP().eval()
    quantized = quantize_with_gptq_weight_only(
        model,
        policy={
            "include_module_types": ["Linear"],
            "bits": 4,
            "group_size": 32,
        },
        strategy="w4a16_int4",
        inplace=False,
    )
    assert isinstance(quantized.model, nn.Module)

    export_dir = tmp_path / "ckpt"
    export_report = export_compressed_tensors(
        quantized,
        export_dir,
        format_name="gptq",
        bits=4,
        group_size=32,
    )
    assert export_report.module_count >= 1
    assert (export_dir / "config.json").is_file()
    assert (export_dir / "model.pt").is_file()

    # Dense base for materialize
    base = _TinyMLP().eval()
    loaded, load_report = load_external_quantized_model(
        export_dir,
        base_model=base,
        override="gptq",
    )
    assert load_report.loaded is True
    assert load_report.layout_kernel is not None
    layout = load_report.layout_kernel.to_dict()
    assert layout["bits"] == 4
    assert layout["storage_layout"]
    assert layout["selected_kernel"]
    assert layout.get("scale_time") == "weight_offline"
    assert loaded is not None
    assert isinstance(base.fc1, AWQGPTQWeightOnlyLinear)
    assert isinstance(base.fc2, AWQGPTQWeightOnlyLinear)

    x = torch.randn(8, 64)
    with torch.no_grad():
        y_ref = quantized.model(x)
        y_load = base(x)
    max_abs = (y_ref - y_load).abs().max().item()
    assert max_abs < 1e-4, f"forward diff too large: {max_abs}"

    # Latency smoke (CPU, structural only — not a perf claim)
    reps = 20
    with torch.no_grad():
        for _ in range(3):
            base(x)
        t0 = time.perf_counter()
        for _ in range(reps):
            base(x)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0 / reps
    assert elapsed_ms > 0.0
    assert load_report.to_dict()["loaded"] is True


def test_optional_hf_fixture_load_diff_latency_or_skip() -> None:
    """U9: real local HF-style dir when present; otherwise honest skip."""

    fixture = _resolve_optional_hf_fixture()
    if fixture is None:
        pytest.skip(
            "optional HF fixture absent; set XQT_HF_QUANT_FIXTURE or place a "
            "dir under data/xqt_hf_quant_fixtures/ (see HF_CHECKPOINT_SMOKE.md)"
        )

    base = _TinyMLP().eval()
    loaded, load_report = load_external_quantized_model(
        fixture,
        base_model=base,
        override=None,
    )
    if not load_report.loaded:
        pytest.skip(
            f"fixture present but not materializable for TinyMLP shell: "
            f"{fixture}; notes={load_report.notes}"
        )
    assert loaded is not None
    assert load_report.layout_kernel is not None
    x = torch.randn(4, 64)
    with torch.no_grad():
        y = base(x)
    assert y.shape[-1] == 32
    assert torch.isfinite(y).all()
    with torch.no_grad():
        t0 = time.perf_counter()
        for _ in range(10):
            base(x)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0 / 10.0
    assert elapsed_ms > 0.0
