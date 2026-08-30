"""U6: apply-path LayoutKernelReport selected_kernel wiring."""

from __future__ import annotations

import torch
from torch import nn

from xqt.contracts import build_runtime_manifest
from xqt.compression.quant.layout_apply_report import (
    layout_report_for_awq_gptq_model,
    layout_report_for_fp4_dynamic,
    layout_report_for_int8_mma,
    resolve_selected_kernel_name,
)
from xqt.compression.quant.quantizers.awq_gptq_weight_only import quantize_with_awq_weight_only
from xqt.compression.quant.quantizers.fp4_dynamic import quantize_with_dynamic_fp4
from xqt.compression.quant.quantizers.int8_mma import quantize_with_int8_mma


def test_resolve_selected_kernel_maps_primary_tokens() -> None:
    selected, reason = resolve_selected_kernel_name("w8a8_int8_mma", fallback="torch")
    assert selected == "tilelang"
    assert reason is None
    selected_fp16, reason_fp16 = resolve_selected_kernel_name(
        "dequant_fp16", fallback="torch"
    )
    assert selected_fp16 == "torch"
    assert reason_fp16 is None


def test_int8_mma_attach_layout_kernel_selected() -> None:
    model = nn.Linear(16, 8).eval()
    result = quantize_with_int8_mma(
        model,
        policy={"include_module_types": ["Linear"]},
        engine="torch_int_mm",
        inplace=False,
    )
    layout = result.metadata.get("layout_kernel")
    assert isinstance(layout, dict)
    assert layout["selected_kernel"] == "torch_int_mm"
    assert layout["storage_layout"] == "xqt_int8_mma_v1"
    manifest = build_runtime_manifest(result)
    assert "torch_int_mm" in manifest.selected_kernels


def test_awq_attach_layout_kernel_dequant_reference() -> None:
    model = nn.Sequential(nn.Linear(32, 16), nn.Linear(16, 8)).eval()
    result = quantize_with_awq_weight_only(
        model,
        policy={"include_module_types": ["Linear"], "group_size": 8, "bits": 4},
        strategy="w4a16_int4",
        inplace=False,
    )
    layout = result.metadata.get("layout_kernel")
    assert isinstance(layout, dict)
    assert layout["selected_kernel"] == "dequant_fp16_reference"
    assert layout["bits"] == 4
    report = layout_report_for_awq_gptq_model(
        result.model, bits=4, group_size=8
    )
    assert report.selected_kernel == "dequant_fp16_reference"
    assert report.desc_act is False
    assert report.g_idx_applied is False
    assert layout["desc_act"] is False
    assert layout["g_idx_applied"] is False


def test_fp4_dynamic_attach_layout_kernel() -> None:
    model = nn.Linear(32, 16).eval()
    result = quantize_with_dynamic_fp4(
        model,
        fp4_format="nvfp4",
        policy={"include_module_types": ["Linear"]},
        engine="torch",
        inplace=False,
    )
    layout = result.metadata.get("layout_kernel")
    assert isinstance(layout, dict)
    assert layout["selected_kernel"] == "torch"
    assert "fp4" in str(layout["storage_layout"])
    report = layout_report_for_fp4_dynamic(
        result.model, engine_preference="auto", fp4_format="nvfp4"
    )
    assert report.selected_kernel is not None


def test_layout_report_for_int8_mma_helper() -> None:
    model = nn.Linear(8, 4)
    report = layout_report_for_int8_mma(
        model,
        engine_preference="auto",
        fallback_engine="torch_int_mm",
        activation_scale_mode="dynamic",
        scale_time="activation_dynamic",
    )
    assert report.selected_kernel in {"tilelang", "triton", "torch_int_mm", "torch"}
    assert report.storage_layout == "xqt_int8_mma_v1"


def test_refresh_layout_kernel_after_forward_uses_realized_engine() -> None:
    """V2: after int8 forward, selected_kernel reflects execution_metadata engine."""

    from xqt.compression.quant.layout_apply_report import refresh_layout_kernel_after_forward
    from xqt.compression.quant.quantizers.int8_mma import Int8MmaLinear, quantize_with_int8_mma

    class _Tiny(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = nn.Linear(16, 8)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.fc(x)

    result = quantize_with_int8_mma(
        _Tiny().eval(),
        policy={"include_module_types": ["Linear"]},
        engine="torch_int_mm",
        inplace=False,
    )
    before = result.metadata["layout_kernel"]["selected_kernel"]
    assert before == "torch_int_mm"
    assert isinstance(result.model.fc, Int8MmaLinear)
    x = torch.randn(4, 16)
    with torch.no_grad():
        _ = result.model(x)
    meta = result.model.fc.execution_metadata()
    assert meta.get("engine") not in {None, "not_run"}
    refreshed = refresh_layout_kernel_after_forward(result.metadata, result.model)
    realized = refreshed["layout_kernel"]["selected_kernel"]
    assert realized == meta["engine"]
