from __future__ import annotations

import sys
import types

import torch
from torch import nn

from examples.xqt_models.ovisocr2 import (
    OVISOCR2_REPO_ID,
    calibrate_ovisocr2_convrot_activation_scales,
    load_ovisocr2,
    materialize_ovisocr2_convrot_int8_runtime,
    quantize_ovisocr2_convrot_int8,
    select_ovisocr2_convrot_modules,
)
from xqt.compression.quant.quantizers.convrot_int8 import ConvRotInt8Linear
from xqt.runtime.modules.convrot import ConvRotInt8ExecutionView


class _TinyOvisOcr2(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.visual_tokenizer = nn.Linear(16, 16)
        self.vte = nn.Linear(16, 16)
        self.llm = nn.Module()
        self.llm.proj = nn.Linear(16, 16)
        self.llm.lm_head = nn.Linear(16, 16, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.visual_tokenizer(inputs) + self.vte(inputs)
        return self.llm.lm_head(self.llm.proj(hidden))


def test_ovisocr2_convrot_limits_quantization_to_llm_projections() -> None:
    torch.manual_seed(97)
    model = _TinyOvisOcr2().eval()
    inputs = torch.randn(32, 16)
    policy = {"min_parameters": 0}

    calibration = calibrate_ovisocr2_convrot_activation_scales(
        model,
        calibration_call=lambda current: current(inputs),
        policy=policy,
        rot_size=4,
    )
    selected = select_ovisocr2_convrot_modules(model, policy=policy)
    result = quantize_ovisocr2_convrot_int8(
        model,
        policy=policy,
        activation_scale_mode="dynamic",
        rot_size=4,
        engine="torch_int_mm",
        min_int8_rows=0,
        inplace=False,
    )

    assert OVISOCR2_REPO_ID == "AIDC-AI/Ovis2.5-9B"
    assert selected == ("llm.proj",)
    assert calibration.target_module_names == ("llm.proj",)
    assert calibration.int8_eligible_module_names(32) == ("llm.proj",)
    assert isinstance(result.model.llm.proj, ConvRotInt8Linear)
    assert isinstance(result.model.visual_tokenizer, nn.Linear)
    assert isinstance(result.model.vte, nn.Linear)
    assert isinstance(result.model.llm.lm_head, nn.Linear)
    assert result.model(inputs).shape == (32, 16)


def test_ovisocr2_runtime_materialization_is_explicit() -> None:
    model = _TinyOvisOcr2().eval()
    result = quantize_ovisocr2_convrot_int8(
        model,
        policy={"min_parameters": 0},
        rot_size=4,
        engine="torch_int_mm",
        min_int8_rows=0,
    )

    runtime = materialize_ovisocr2_convrot_int8_runtime(result.model, inplace=False)

    assert isinstance(result.model.llm.proj, ConvRotInt8Linear)
    assert isinstance(runtime.llm.proj, ConvRotInt8ExecutionView)
    assert runtime.llm.proj.execution_metadata()["artifact_view"] == "runtime_execution"


def test_ovisocr2_defaults_to_small_m_float_fallback_threshold() -> None:
    model = _TinyOvisOcr2().eval()
    result = quantize_ovisocr2_convrot_int8(
        model,
        policy={"min_parameters": 0},
        rot_size=4,
        engine="torch_int_mm",
        inplace=False,
    )

    assert isinstance(result.model.llm.proj, ConvRotInt8Linear)
    assert result.model.llm.proj.int8_compute.min_int8_rows == 256


def test_ovisocr2_small_m_metadata_names_dense_fallback() -> None:
    model = _TinyOvisOcr2().eval()
    result = quantize_ovisocr2_convrot_int8(
        model,
        policy={"min_parameters": 0},
        rot_size=4,
        engine="torch_int_mm",
        inplace=False,
    )
    runtime = materialize_ovisocr2_convrot_int8_runtime(result.model, inplace=False)
    runtime(torch.randn(1, 16))
    metadata = runtime.llm.proj.execution_metadata()

    assert metadata["engine"] == "float_fallback"
    assert metadata["implementation"] == "bf16_dense_small_m_fallback"


def test_ovisocr2_loader_uses_remote_causal_lm(monkeypatch: object) -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    class _FakeAutoModelForCausalLM:
        @classmethod
        def from_pretrained(cls, repo_id: str, **kwargs: object) -> nn.Module:
            calls.append((repo_id, kwargs))
            return _TinyOvisOcr2()

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(AutoModelForCausalLM=_FakeAutoModelForCausalLM),
    )
    model = load_ovisocr2(
        revision="test-revision",
        dtype=torch.float32,
        device="cpu",
        local_files_only=True,
    )

    assert isinstance(model, _TinyOvisOcr2)
    assert calls == [
        (
            OVISOCR2_REPO_ID,
            {
                "trust_remote_code": True,
                "torch_dtype": torch.float32,
                "low_cpu_mem_usage": True,
                "local_files_only": True,
                "revision": "test-revision",
            },
        )
    ]
