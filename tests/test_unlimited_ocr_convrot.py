from __future__ import annotations

import sys
import types

import pytest
import torch
from torch import nn

from xqt.model import (
    UNLIMITED_OCR_CONVROT_INT8_STRATEGY,
    UNLIMITED_OCR_REPO_ID,
    calibrate_unlimited_ocr_convrot_activation_scales,
    load_unlimited_ocr,
    quantize_unlimited_ocr_convrot_int8,
    select_unlimited_ocr_convrot_modules,
)
from xqt.quant.quantizers.convrot_int8 import ConvRotInt8Linear


class _TinyUnlimitedOcr(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.vision_model = nn.Linear(16, 16)
        self.model.decoder = nn.Linear(16, 16)
        self.lm_head = nn.Linear(16, 16, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.model.vision_model(inputs)
        return self.lm_head(self.model.decoder(hidden))


def test_unlimited_ocr_convrot_calibrates_real_call_and_quantizes() -> None:
    torch.manual_seed(71)
    model = _TinyUnlimitedOcr().eval()
    inputs = torch.randn(4, 16)
    policy = {
        "include_module_types": ["Linear"],
        "exclude_name_patterns": [r"^model\.vision_model(?:\.|$)", r"lm_head$"],
        "min_parameters": 0,
    }

    calibration = calibrate_unlimited_ocr_convrot_activation_scales(
        model,
        calibration_call=lambda current: current(inputs),
        policy=policy,
        rot_size=4,
    )
    result = quantize_unlimited_ocr_convrot_int8(
        model,
        calibration_call=lambda current: current(inputs),
        policy=policy,
        rot_size=4,
        engine="torch_int_mm",
        min_int8_rows=0,
    )

    assert UNLIMITED_OCR_REPO_ID == "baidu/Unlimited-OCR"
    assert UNLIMITED_OCR_CONVROT_INT8_STRATEGY == "w8a8_int8"
    assert calibration.target_module_names == ("model.decoder",)
    assert calibration.observed_module_names == ("model.decoder",)
    assert calibration.activation_scales["model.decoder"] > 0.0
    assert calibration.int8_eligible_module_names(4) == ("model.decoder",)
    assert calibration.int8_eligible_module_names(5) == ()
    assert isinstance(result.model.model.decoder, ConvRotInt8Linear)
    assert isinstance(result.model.model.vision_model, nn.Linear)
    assert isinstance(result.model.lm_head, nn.Linear)
    assert result.quantized_modules == ["model.decoder"]
    assert result.calibration is not None
    assert result.model(inputs).shape == (4, 16)


def test_unlimited_ocr_convrot_rejects_empty_int8_eligible_selection() -> None:
    model = _TinyUnlimitedOcr().eval()
    inputs = torch.randn(4, 16)
    policy = {
        "include_module_types": ["Linear"],
        "exclude_name_patterns": [r"^model\.vision_model(?:\.|$)", r"lm_head$"],
        "min_parameters": 0,
    }

    with pytest.raises(ValueError, match="no module reaching min_int8_rows"):
        quantize_unlimited_ocr_convrot_int8(
            model,
            calibration_call=lambda current: current(inputs),
            policy=policy,
            rot_size=4,
            engine="torch_int_mm",
            min_int8_rows=5,
            only_static_int8_eligible_modules=True,
        )


def test_unlimited_ocr_module_selection_uses_default_exclusions() -> None:
    model = _TinyUnlimitedOcr().eval()

    selected = select_unlimited_ocr_convrot_modules(
        model,
        policy={"min_parameters": 0},
    )

    assert selected == ("model.decoder",)


def test_unlimited_ocr_loader_preserves_explicit_float32(monkeypatch) -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    class _FakeAutoModel:
        @classmethod
        def from_pretrained(cls, repo_id: str, **kwargs: object) -> nn.Module:
            calls.append((repo_id, kwargs))
            return _TinyUnlimitedOcr()

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(AutoModel=_FakeAutoModel),
    )

    model = load_unlimited_ocr(
        revision="test-revision",
        dtype=torch.float32,
        device="cpu",
        local_files_only=True,
    )

    assert isinstance(model, _TinyUnlimitedOcr)
    assert calls == [
        (
            UNLIMITED_OCR_REPO_ID,
            {
                "trust_remote_code": True,
                "use_safetensors": True,
                "torch_dtype": torch.float32,
                "low_cpu_mem_usage": True,
                "local_files_only": True,
                "revision": "test-revision",
            },
        )
    ]
