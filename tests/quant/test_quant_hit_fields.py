"""Z3: quant hit/skip fields must not be empty unknowns."""

from __future__ import annotations

from torch import nn

from xqt.compression.quant.quantizers.awq_gptq_weight_only import quantize_with_awq_weight_only
from xqt.compression.quant.quantizers.int8_mma import quantize_with_int8_mma
from xqt.compression.quant.selection import summarize_quant_hit_fields


def test_int8_component_path_exposes_hit_fields() -> None:
    class Shell(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = nn.Linear(16, 8)

        def forward(self, x):  # type: ignore[no-untyped-def]
            return self.fc(x)

    from xqt.core.types import XQTContext
    from xqt.compression.quant.types import QuantizationComponentPlan
    from xqt.compression.quant.quantizers.int8_mma import execute_int8_mma_component

    model = Shell().eval()
    ctx = XQTContext(model=model, device="cpu")
    component = QuantizationComponentPlan(
        name="main",
        backend="pytorch",
        target_path="",
        method="dynamic_int8_mma",
        strategy="w8a8_int8",
        compute="w8a8_int8_mma",
        policy={"include_module_types": ["Linear"], "engine": "torch_int_mm"},
    )
    _model, report = execute_int8_mma_component(ctx, model, component)
    hit = summarize_quant_hit_fields(report)
    assert hit.ok is True
    assert hit.quantized_modules
    assert hit.has_selection_policy or hit.has_module_selection_reasons


def test_awq_direct_path_exposes_quantized_modules() -> None:
    model = nn.Sequential(nn.Linear(32, 16), nn.Linear(16, 8)).eval()
    result = quantize_with_awq_weight_only(
        model,
        policy={"include_module_types": ["Linear"], "group_size": 8, "bits": 4},
        strategy="w4a16_int4",
        inplace=False,
    )
    hit = summarize_quant_hit_fields(result)
    assert hit.ok is True
    assert hit.quantized_modules


def test_empty_result_is_not_ok() -> None:
    hit = summarize_quant_hit_fields({"quantized_modules": [], "metadata": {}})
    assert hit.ok is False
    assert "quantized_modules_empty" in hit.gaps
