"""W4: quant-after export readiness report."""

from __future__ import annotations

from torch import nn

from xqt.export import assess_export_readiness
from xqt.compression.quant.quantizers.awq_gptq_weight_only import quantize_with_awq_weight_only
from xqt.compression.quant.quantizers.int8_mma import quantize_with_int8_mma


def test_dense_model_can_export() -> None:
    model = nn.Sequential(nn.Linear(8, 4), nn.ReLU(), nn.Linear(4, 2)).eval()
    report = assess_export_readiness(model)
    assert report.can_export is True
    assert report.blockers == ()
    assert report.suggested_lowering is None


def test_awq_packed_suggests_lowering() -> None:
    model = nn.Sequential(nn.Linear(32, 16), nn.Linear(16, 8)).eval()
    result = quantize_with_awq_weight_only(
        model,
        policy={"include_module_types": ["Linear"], "group_size": 8, "bits": 4},
        strategy="w4a16_int4",
        inplace=False,
    )
    report = assess_export_readiness(result.model)
    assert report.can_export is False
    assert report.suggested_lowering == "fp4_weight_only_to_dense_linear"
    assert report.packed_modules
    assert any("packed_weight_modules" in b for b in report.blockers)


def test_awq_pre_export_lowering_applies() -> None:
    from xqt.export.lowering import apply_pre_export_lowering

    model = nn.Sequential(nn.Linear(32, 16), nn.Linear(16, 8)).eval()
    result = quantize_with_awq_weight_only(
        model,
        policy={"include_module_types": ["Linear"], "group_size": 8, "bits": 4},
        strategy="w4a16_int4",
        inplace=False,
    )
    lowered = apply_pre_export_lowering(
        result.model,
        {"enabled": True, "mode": "fp4_weight_only_to_dense_linear", "inplace": False},
    )
    assert lowered.applied is True
    assert lowered.lowered_modules
    for name, module in lowered.model.named_modules():
        if name and isinstance(module, nn.Linear):
            assert type(module).__name__ == "Linear"


def test_int8_mma_specialized_blocker() -> None:
    class Shell(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = nn.Linear(16, 8)

        def forward(self, x):  # type: ignore[no-untyped-def]
            return self.fc(x)

    result = quantize_with_int8_mma(
        Shell().eval(),
        policy={"include_module_types": ["Linear"]},
        engine="torch_int_mm",
        inplace=False,
    )
    report = assess_export_readiness(result.model)
    assert report.can_export is False
    assert report.specialized_modules
    assert "fc" in report.specialized_modules
