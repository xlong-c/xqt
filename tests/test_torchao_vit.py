from pathlib import Path

import pytest
import torch

from xqt.eval.compare import TensorDiff
from xqt.quant import LayerAnalysisRecord
from xqt.torchao_vit import (
    DEFAULT_ARTIFACT_DIR,
    TorchAOViTExperimentConfig,
    analysis_records_to_rows,
    build_analysis_loader,
    config_from_env,
)


def _diff(mean_abs: float, *, max_abs: float | None = None) -> TensorDiff:
    return TensorDiff(
        max_abs=max_abs if max_abs is not None else mean_abs,
        mean_abs=mean_abs,
        mean_squared=mean_abs * mean_abs,
        sqnr_db=20.0,
        relative_error=None,
        cosine_similarity=0.99,
        correlation=None,
        argmax_mismatch_rate=None,
        allclose=False,
        atol=1e-3,
        rtol=1e-3,
        valid=True,
        message="ok",
    )


def test_config_from_env_uses_explicit_xqt_torchao_vit_vars() -> None:
    config = config_from_env(
        {
            "XQT_TORCHAO_VIT_MODEL": "vit_tiny_patch16_224",
            "XQT_TORCHAO_VIT_ARTIFACT_DIR": "/tmp/xqt-vit",
            "XQT_TORCHAO_VIT_IMAGE_ROOT": "/tmp/images",
            "XQT_TORCHAO_VIT_BATCH_SIZE": "4",
            "XQT_TORCHAO_VIT_ANALYSIS_SAMPLES": "8",
            "XQT_TORCHAO_VIT_WARMUP": "1",
            "XQT_TORCHAO_VIT_ITERATIONS": "2",
            "XQT_TORCHAO_VIT_COMPILE": "0",
        }
    )

    assert config.model_name == "vit_tiny_patch16_224"
    assert config.artifact_dir == Path("/tmp/xqt-vit")
    assert config.image_root == Path("/tmp/images")
    assert config.batch_size == 4
    assert config.analysis_sample_limit == 8
    assert config.benchmark_warmup == 1
    assert config.benchmark_iterations == 2
    assert config.compile_quantized is False


def test_config_from_env_uses_defaults_for_missing_values() -> None:
    config = config_from_env({})

    assert config.artifact_dir == DEFAULT_ARTIFACT_DIR
    assert config.image_root is None
    assert config.compile_quantized is True


def test_build_analysis_loader_uses_synthetic_images_without_dataset_root() -> None:
    config = TorchAOViTExperimentConfig(
        image_root=None,
        batch_size=2,
        analysis_sample_limit=3,
        image_size=16,
        device="cpu",
        compile_quantized=False,
        pretrained=False,
    )

    loader = build_analysis_loader(config)
    inputs, targets = next(iter(loader))

    assert inputs.shape == (2, 3, 16, 16)
    assert targets.shape == (2,)


def test_analysis_records_to_rows_keeps_legacy_columns_and_sorting() -> None:
    records = [
        LayerAnalysisRecord(
            name="blocks.1.mlp.fc1",
            module_type="Linear",
            diff=_diff(0.2, max_abs=0.5),
            parameter_count=16,
            reference_summary={},
            candidate_summary={},
            weight_diff=None,
            recommendation="consider_higher_precision",
            tags=("high_error",),
        ),
        LayerAnalysisRecord(
            name="blocks.0.mlp.fc1",
            module_type="Linear",
            diff=_diff(0.4, max_abs=0.7),
            parameter_count=16,
            reference_summary={},
            candidate_summary={},
            weight_diff=_diff(0.1),
            recommendation=None,
            tags=("weight_shift",),
        ),
    ]

    rows = analysis_records_to_rows(records)

    assert [row["layer_name"] for row in rows] == [
        "blocks.0.mlp.fc1",
        "blocks.1.mlp.fc1",
    ]
    assert rows[0]["act_mae"] == 0.4
    assert rows[0]["act_mse"] == pytest.approx(0.16)
    assert rows[0]["weight_mae"] == 0.1
    assert rows[0]["tags"] == "weight_shift"
    assert rows[1]["weight_mae"] is None
