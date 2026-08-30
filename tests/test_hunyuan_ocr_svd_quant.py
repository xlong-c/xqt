from __future__ import annotations

import sys
import types

import pytest
import torch
from torch import nn

import examples.xqt_models.hunyuan_ocr as hunyuan_ocr
from xqt.core.errors import XQTBackendError
from examples.xqt_models.hunyuan_ocr import (
    HUNYUAN_OCR_DFLASH_SUBFOLDER,
    HUNYUAN_OCR_REPO_ID,
    HUNYUAN_OCR_SVD_INT4_STRATEGY,
    load_hunyuan_ocr_dflash,
    optimize_hunyuan_ocr_dflash_svd_int4_blocks,
    optimize_hunyuan_ocr_svd_int4_blocks,
)
from xqt.kernels.wrappers.types import OperatorOptimizationTargetPlan
from xqt.runtime.modules import SVDQuantInt8MmaLinear


class _TinyHunyuanBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attention = nn.Linear(16, 16)
        self.mlp = nn.Linear(16, 16)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.mlp(torch.nn.functional.gelu(self.attention(inputs)))


class _TinyHunyuanOCR(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.vision_encoder = nn.Linear(16, 16)
        self.blocks = nn.ModuleList([_TinyHunyuanBlock(), _TinyHunyuanBlock()])

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.vision_encoder(inputs)
        for block in self.blocks:
            hidden = block(hidden)
        return hidden


class _LayerOnlyHunyuanOCR(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.first = nn.Linear(16, 16)
        self.second = nn.Linear(16, 16)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.second(self.first(inputs))


def _record_block_compilation(
    monkeypatch: pytest.MonkeyPatch,
) -> list[OperatorOptimizationTargetPlan]:
    plans: list[OperatorOptimizationTargetPlan] = []

    def fake_compile(
        module: nn.Module,
        plan: OperatorOptimizationTargetPlan,
    ) -> tuple[nn.Module, float]:
        plans.append(plan)
        return module, 0.25

    monkeypatch.setattr(hunyuan_ocr, "compile_with_torch", fake_compile)
    return plans


def test_hunyuan_ocr_helper_quantizes_int4_and_compiles_blocks(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plans = _record_block_compilation(monkeypatch)
    model = _TinyHunyuanOCR().eval()

    result = optimize_hunyuan_ocr_svd_int4_blocks(
        model,
        artifact_dir=tmp_path / "hunyuan_ocr",
        rank=4,
        group_size=8,
        engine="torch_int_mm",
        calibration_inputs=[torch.randn(2, 16)],
    )

    assert HUNYUAN_OCR_REPO_ID == "tencent/HunyuanOCR"
    assert result.stage.name == HUNYUAN_OCR_SVD_INT4_STRATEGY
    assert result.stage.accepted is True
    assert isinstance(result.model.vision_encoder, SVDQuantInt8MmaLinear)
    assert isinstance(result.model.blocks[0].attention, SVDQuantInt8MmaLinear)
    assert (
        result.model.blocks[0].attention.residual_int8.activation_scale_mode == "static"
    )
    assert result.compute_config is not None
    assert {
        module["storage"]["quant_dtype"] for module in result.compute_config["modules"]
    } == {"int4"}
    assert result.block_optimization.block_paths == ("blocks.0", "blocks.1")
    assert result.block_optimization.compiled_block_count == 2
    assert result.block_optimization.warmup_input_source == "calibration_inputs[0]"
    assert [plan.target_path for plan in plans] == ["blocks.0", "blocks.1"]
    assert all(plan.engine == "torch_compile" for plan in plans)
    assert all(plan.patterns == ["transformer_block"] for plan in plans)
    assert result.stage.metrics["hunyuan_ocr_block_optimization"]["level"] == "block"
    output = result.model(torch.randn(2, 16))
    assert output.shape == (2, 16)
    assert torch.isfinite(output).all()


def test_hunyuan_ocr_helper_rejects_layer_only_optimization(tmp_path) -> None:
    with pytest.raises(XQTBackendError, match="block optimization requires"):
        optimize_hunyuan_ocr_svd_int4_blocks(
            _LayerOnlyHunyuanOCR().eval(),
            artifact_dir=tmp_path / "layer_only",
            rank=4,
            group_size=8,
            engine="torch_int_mm",
            calibration_inputs=[torch.randn(2, 16)],
        )


def test_hunyuan_ocr_helper_materializes_real_compiled_blocks(tmp_path) -> None:
    result = optimize_hunyuan_ocr_svd_int4_blocks(
        _TinyHunyuanOCR().eval(),
        artifact_dir=tmp_path / "compiled_blocks",
        rank=4,
        group_size=8,
        engine="torch_int_mm",
        block_engine="eager",
        example_inputs=torch.randn(2, 16),
    )

    assert result.block_optimization.compiled_block_count == 2
    assert hasattr(result.model.blocks[0], "_orig_mod")
    assert result.model(torch.randn(2, 16)).shape == (2, 16)


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


def test_hunyuan_ocr_dflash_helper_quantizes_int4_and_compiles_blocks(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plans = _record_block_compilation(monkeypatch)

    result = optimize_hunyuan_ocr_dflash_svd_int4_blocks(
        _TinyHunyuanOCR().eval(),
        artifact_dir=tmp_path / "hunyuan_ocr_dflash",
        rank=4,
        group_size=8,
        engine="torch_int_mm",
        calibration_inputs=[torch.randn(2, 16)],
    )

    assert isinstance(result.model.blocks[0].attention, SVDQuantInt8MmaLinear)
    assert result.stage.accepted is True
    assert result.compute_config is not None
    assert {
        module["storage"]["quant_dtype"] for module in result.compute_config["modules"]
    } == {"int4"}
    assert result.block_optimization.compiled_block_count == 2
    assert len(plans) == 2
