import pytest
import torch
import torch.nn.functional as F

from xqt.core.artifact import ArtifactManifest
from xqt.core.config import load_xqt_config
from xqt.core.errors import XQTBackendError
from xqt.core.types import XQTContext
from xqt.operator_opt.backends.cutile import (
    CuTileCompileSettings,
    build_cutile_artifact_metadata,
    list_cutile_kernel_specs,
    run_cutile_kernel,
)
from xqt.operator_opt.backends.cutlass import (
    CutlassCompileSettings,
    build_cutlass_artifact_metadata,
    list_cutlass_kernel_specs,
    run_cutlass_kernel,
)
from xqt.operator_opt.kernels.cutile import (
    fused_bias_silu_cutile,
    fused_bias_silu_reference,
)
from xqt.operator_opt.kernels.cutlass import (
    gemm_epilogue_cutlass,
    gemm_epilogue_reference,
)
from xqt.operator_opt.kernels.tilelang import build_tilelang_attention_design
from xqt.pipeline.passes import OperatorOptimizationPass
from xqt.pipeline.preflight import preflight_xqt_config


def test_cutile_reference_and_cpu_fallback_match_pytorch() -> None:
    x = torch.randn(2, 4)
    bias = torch.randn(4)

    expected = F.silu(x + bias)

    assert torch.allclose(fused_bias_silu_reference(x, bias), expected)
    assert torch.allclose(run_cutile_kernel("bias_silu", x, bias), expected)
    with pytest.raises(XQTBackendError, match="requires CUDA tensors"):
        run_cutile_kernel("bias_silu", x, bias, fallback="raise")
    with pytest.raises(XQTBackendError, match="CUDA tensors"):
        fused_bias_silu_cutile(x, bias)


def test_cutlass_reference_and_cpu_fallback_match_pytorch() -> None:
    x = torch.randn(2, 4)
    weight = torch.randn(3, 4)
    bias = torch.randn(3)

    expected = F.gelu(x.matmul(weight.t()) + bias)

    assert torch.allclose(
        gemm_epilogue_reference(x, weight, bias, activation="gelu"),
        expected,
    )
    assert torch.allclose(
        run_cutlass_kernel("gemm_epilogue", x, weight, bias, activation="gelu"),
        expected,
    )
    with pytest.raises(XQTBackendError, match="requires CUDA tensors"):
        run_cutlass_kernel("gemm_epilogue", x, weight, bias, fallback="raise")
    with pytest.raises(XQTBackendError, match="CUDA tensors"):
        gemm_epilogue_cutlass(x, weight, bias)


def test_cutile_and_cutlass_artifact_metadata_are_stable(tmp_path) -> None:
    cutile_settings = CuTileCompileSettings(
        target="cuda",
        target_arch="sm_89",
        cache_dir=str(tmp_path / "cutile-cache"),
        threads=128,
        pass_configs={"CUTILE_ENABLE_FAST_MATH": True},
    )
    cutlass_settings = CutlassCompileSettings(
        target_arch="sm_89",
        cache_dir=str(tmp_path / "cutlass-cache"),
        tile_shape=(128, 128, 64),
        pass_configs={"CUTLASS_ENABLE_EPILOGUE_FUSION": True},
    )

    cutile = build_cutile_artifact_metadata("bias_silu", cutile_settings)
    cutlass = build_cutlass_artifact_metadata("gemm_epilogue", cutlass_settings)

    assert set(list_cutile_kernel_specs()) == {"bias_silu"}
    assert set(list_cutlass_kernel_specs()) == {"gemm_epilogue", "grouped_gemm"}
    assert cutile["artifact_path"] == str(tmp_path / "cutile-cache" / "bias_silu.cutile.json")
    assert cutile["compile"]["target_arch"] == "sm_89"
    assert cutile["compile_status"] == "metadata_only"
    assert cutlass["artifact_path"] == str(tmp_path / "cutlass-cache" / "gemm_epilogue.cutlass.json")
    assert cutlass["compile"]["tile_shape"] == [128, 128, 64]
    assert cutlass["compile_status"] == "metadata_only"


def test_tilelang_attention_design_records_learning_kernel_extraction() -> None:
    design = build_tilelang_attention_design()
    data = design.to_dict()

    assert data["source"] == "learn/tilelang/flashatt.py"
    assert data["softmax"] == "online softmax with running max/logsum"
    assert data["default_block_m"] == 64
    assert data["default_block_n"] == 64
    assert data["production_status"] == "design_extracted_reference_guarded"


def test_preflight_reports_cutile_and_cutlass_backend_metadata(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    monkeypatch.setattr("torch.cuda.device_count", lambda: 0)

    report = preflight_xqt_config(
        {
            "model": {
                "target": "torch.nn.Linear",
                "params": {"in_features": 4, "out_features": 2},
            },
            "data": {
                "validation": {
                    "target": "synthetic_classification",
                    "sample_limit": 1,
                    "batch_size": 1,
                }
            },
            "operator_optimization": {
                "enabled": True,
                "targets": [
                    {
                        "name": "cutile_bias_silu",
                        "backend": "cutile",
                        "target": "model",
                        "patterns": ["bias_silu"],
                        "cutile": {
                            "target_arch": "sm_89",
                            "cache_dir": str(tmp_path / "cutile-cache"),
                        },
                    },
                    {
                        "name": "cutlass_gemm",
                        "backend": "cutlass",
                        "target": "model",
                        "patterns": ["gemm_epilogue"],
                        "cutlass": {
                            "target_arch": "sm_89",
                            "cache_dir": str(tmp_path / "cutlass-cache"),
                        },
                    },
                ],
            },
        }
    )
    checks = {check.name: check for check in report.checks}

    assert checks["operator_optimization.targets.0.capability"].metadata["backend"] == "cutile"
    assert checks["operator_optimization.targets.0.cutile.config"].metadata["target_arch"] == "sm_89"
    assert checks["operator_optimization.targets.1.capability"].metadata["backend"] == "cutlass"
    assert checks["operator_optimization.targets.1.cutlass.config"].metadata["tile_shape"] == [128, 128, 64]


def test_operator_optimization_pass_records_cutile_cutlass_metadata(tmp_path) -> None:
    model = torch.nn.Linear(4, 2)
    config = load_xqt_config(
        {
            "project": {"artifact_dir": str(tmp_path / "operator_new_backends")},
            "model": {"device": "cpu"},
            "benchmark": {"warmup": 0, "iterations": 1},
            "operator_optimization": {
                "enabled": True,
                "targets": [
                    {
                        "name": "cutile_bias_silu",
                        "target": "",
                        "backend": "cutile",
                        "patterns": ["bias_silu"],
                        "cutile": {
                            "target_arch": "sm_89",
                            "cache_dir": str(tmp_path / "cutile-cache"),
                        },
                    },
                    {
                        "name": "cutlass_gemm",
                        "target": "",
                        "backend": "cutlass",
                        "patterns": ["gemm_epilogue"],
                        "cutlass": {
                            "target_arch": "sm_89",
                            "cache_dir": str(tmp_path / "cutlass-cache"),
                        },
                    },
                ],
            },
        }
    )
    context = XQTContext(
        config=config,
        model=model,
        data={"validation": [(torch.randn(2, 4), torch.zeros(2, dtype=torch.long))]},
        manifest=ArtifactManifest(project_name=config.project.name),
    )

    output = OperatorOptimizationPass().run(context)
    targets = {
        target["target_name"]: target
        for target in output.metrics["operator_optimization"]["targets"]
    }

    cutile = targets["cutile_bias_silu"]
    assert cutile["applied"] is False
    assert (
        "requires CUDA-capable hardware" in cutile["skip_reason"]
        or "not implemented in the built-in executor" in cutile["skip_reason"]
    )
    assert "bias_silu" in cutile["metadata"]["kernel_registry"]
    assert cutile["artifact_paths"]["cutile.bias_silu"] == str(
        tmp_path / "cutile-cache" / "bias_silu.cutile.json"
    )

    cutlass = targets["cutlass_gemm"]
    assert cutlass["applied"] is False
    assert (
        "requires CUDA-capable hardware" in cutlass["skip_reason"]
        or "not implemented in the built-in executor" in cutlass["skip_reason"]
    )
    assert "gemm_epilogue" in cutlass["metadata"]["kernel_registry"]
    assert cutlass["artifact_paths"]["cutlass.gemm_epilogue"] == str(
        tmp_path / "cutlass-cache" / "gemm_epilogue.cutlass.json"
    )
