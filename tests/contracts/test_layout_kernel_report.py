"""T2: LayoutKernelReport required diagnostic fields."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from xqt.contracts.layout_kernel_report import (
    LAYOUT_KERNEL_REPORT_KEYS,
    LayoutKernelReport,
    empty_layout_kernel_report,
)
from xqt.core.errors import XQTConfigError
from xqt.quant.quantizers.awq_gptq_weight_only import AWQGPTQWeightOnlyLinear
from xqt.runtime.bridges.external_weight_only import load_external_quantized_model


def test_layout_kernel_report_keys_are_stable() -> None:
    required = {
        "bits",
        "group_size",
        "symmetric",
        "zero_point",
        "desc_act",
        "g_idx_applied",
        "global_shape",
        "local_shape",
        "padding_ratio",
        "storage_layout",
        "selected_kernel",
        "fallback_reason",
        "sm",
        "min_capability",
        "scale_time",
        "activation_granularity",
    }
    assert set(LAYOUT_KERNEL_REPORT_KEYS) == required


def test_layout_kernel_report_round_trip() -> None:
    report = LayoutKernelReport(
        bits=4,
        group_size=128,
        symmetric=True,
        zero_point=False,
        desc_act=False,
        g_idx_applied=False,
        global_shape=(4096, 4096),
        local_shape=(4096, 1024),
        padding_ratio=0.0,
        storage_layout="xqt_awq_gptq_int4_v1",
        selected_kernel="dequant_fp16_reference",
        fallback_reason=None,
        sm=89,
        min_capability=80,
    )
    payload = report.to_dict()
    restored = LayoutKernelReport.from_dict(payload)
    assert restored == report
    for key in LAYOUT_KERNEL_REPORT_KEYS:
        assert key in payload


def test_empty_layout_kernel_report_has_all_keys() -> None:
    payload = empty_layout_kernel_report().to_dict()
    for key in LAYOUT_KERNEL_REPORT_KEYS:
        assert key in payload


def test_from_dict_rejects_partial_payload() -> None:
    with pytest.raises(XQTConfigError, match="storage_layout"):
        LayoutKernelReport.from_dict({"bits": 4})


def _write_json(path: Path, payload: dict) -> None:
    import json

    path.write_text(json.dumps(payload), encoding="utf-8")


def test_external_load_success_includes_layout_kernel_report(tmp_path: Path) -> None:
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
    loaded = Tiny()
    _, report = load_external_quantized_model(tmp_path, base_model=loaded)
    assert report.loaded is True
    layout = report.layout_kernel
    assert layout is not None
    payload = layout.to_dict()
    for key in LAYOUT_KERNEL_REPORT_KEYS:
        assert key in payload
    assert payload["bits"] == 4
    assert payload["group_size"] == 32
    assert payload["storage_layout"]
    assert payload["scale_time"] == "weight_offline"


def test_external_load_fallback_includes_layout_kernel_report(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "config.json",
        {"quantization_config": {"quant_method": "gptq", "bits": 4, "group_size": 128}},
    )
    _, report = load_external_quantized_model(tmp_path)
    assert report.loaded is False
    assert report.layout_kernel is not None
    payload = report.layout_kernel.to_dict()
    for key in LAYOUT_KERNEL_REPORT_KEYS:
        assert key in payload
    assert payload["fallback_reason"] is not None or any(
        "probe_only" in n for n in report.notes
    )
