from __future__ import annotations

import sys
import types

import torch
from torch import nn

from xqt.model import (
    HUNYUAN_OCR_DFLASH_SUBFOLDER,
    HUNYUAN_OCR_REPO_ID,
    HUNYUAN_OCR_SVD_FP4_INT8_MMA_STRATEGY,
    load_hunyuan_ocr_dflash,
    optimize_hunyuan_ocr_dflash_svd_fp4_int8_mma,
    optimize_hunyuan_ocr_svd_fp4_int8_mma,
)
from xqt.quant.quantizers.svd import SVDQuantInt8MmaLinear


class _TinyHunyuanOCR(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.vision_encoder = nn.Linear(16, 16)
        self.text_decoder = nn.Sequential(nn.Linear(16, 16), nn.GELU())

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.text_decoder(self.vision_encoder(inputs))


def test_hunyuan_ocr_helper_quantizes_linear_modules_with_int8_mma(tmp_path) -> None:
    model = _TinyHunyuanOCR().eval()

    result = optimize_hunyuan_ocr_svd_fp4_int8_mma(
        model,
        artifact_dir=tmp_path / "hunyuan_ocr",
        rank=4,
        group_size=8,
        engine="torch_int_mm",
        calibration_inputs=[torch.randn(2, 16)],
    )

    assert HUNYUAN_OCR_REPO_ID == "tencent/HunyuanOCR"
    assert result.stage.name == HUNYUAN_OCR_SVD_FP4_INT8_MMA_STRATEGY
    assert result.stage.accepted is True
    assert isinstance(result.model.vision_encoder, SVDQuantInt8MmaLinear)
    assert isinstance(result.model.text_decoder[0], SVDQuantInt8MmaLinear)
    assert result.model.vision_encoder.residual_int8.activation_scale_mode == "static"
    assert result.model.text_decoder[0].residual_int8.activation_scale_mode == "static"
    assert result.compute_config is not None
    assert len(result.compute_config["modules"]) == 2
    output = result.model(torch.randn(2, 16))
    assert output.shape == (2, 16)
    assert torch.isfinite(output).all()


def test_hunyuan_ocr_dflash_loader_requests_dflash_subfolder(monkeypatch) -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    class _FakeAutoModel:
        @classmethod
        def from_pretrained(cls, repo_id: str, **kwargs: object) -> nn.Module:
            calls.append((repo_id, kwargs))
            return _TinyHunyuanOCR()

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(AutoModel=_FakeAutoModel),
    )

    model = load_hunyuan_ocr_dflash(
        revision="test-revision",
        dtype=torch.float32,
        device="cpu",
        local_files_only=True,
    )

    assert isinstance(model, _TinyHunyuanOCR)
    assert calls == [
        (
            HUNYUAN_OCR_REPO_ID,
            {
                "trust_remote_code": True,
                "torch_dtype": torch.float32,
                "low_cpu_mem_usage": True,
                "local_files_only": True,
                "revision": "test-revision",
                "subfolder": HUNYUAN_OCR_DFLASH_SUBFOLDER,
            },
        )
    ]


def test_hunyuan_ocr_dflash_helper_runs_svd_fp4_int8_mma(tmp_path) -> None:
    result = optimize_hunyuan_ocr_dflash_svd_fp4_int8_mma(
        _TinyHunyuanOCR().eval(),
        artifact_dir=tmp_path / "hunyuan_ocr_dflash",
        rank=4,
        group_size=8,
        engine="torch_int_mm",
        calibration_inputs=[torch.randn(2, 16)],
    )

    assert isinstance(result.model.vision_encoder, SVDQuantInt8MmaLinear)
    assert isinstance(result.model.text_decoder[0], SVDQuantInt8MmaLinear)
    assert result.stage.accepted is True
    assert result.compute_config is not None
