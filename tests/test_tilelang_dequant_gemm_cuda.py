from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path

import pytest
import torch

from xqt.operator_opt.backends.tilelang import list_tilelang_kernel_specs
from xqt.operator_opt.backends.tilelang_validation import (
    TileLangFP4ValidationResult,
    validate_tilelang_packed_fp4_fused_gemm,
)
from xqt.operator_opt.kernels.tilelang import build_tilelang_fp4_fused_dequant_gemm_kernel
from xqt.operator_opt.kernels.tilelang.attention import (
    dequant_gemm_epilogue_reference,
    dequant_gemm_epilogue_tilelang,
    fp4_packed_dequant_gemm_epilogue_reference,
    fp4_packed_dequant_gemm_epilogue_tilelang,
)
from xqt.quant.fp4_backend import ReferenceFP4Linear


requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for TileLang dequant GEMM CUDA test",
)

requires_tilelang = pytest.mark.skipif(
    importlib.util.find_spec("tilelang") is None,
    reason="tilelang package is required for TileLang dequant GEMM CUDA test",
)

requires_nvcc = pytest.mark.skipif(
    shutil.which("nvcc") is None and not Path("/usr/local/cuda/bin/nvcc").exists(),
    reason="nvcc is required for TileLang compile-only tests",
)


def test_tilelang_fp4_registry_reports_tilelang_unpack_stage() -> None:
    specs = list_tilelang_kernel_specs()
    metadata = specs["fp4_packed_dequant_gemm_epilogue"]["metadata"]

    assert metadata["unpack_stage"] == "tilelang_fused_gemm_kernel"
    assert metadata["fusion_status"] == "single_tilelang_kernel_for_unpack_dequant_gemm_epilogue"
    assert metadata["epilogue_stage"] == "tilelang_fused_bias_activation"


def test_tilelang_packed_fp4_validation_reports_skipped_without_cuda() -> None:
    result = validate_tilelang_packed_fp4_fused_gemm(
        target_arch="sm_80",
        compile_only=False,
        warmup=0,
        iterations=1,
    )

    assert isinstance(result, TileLangFP4ValidationResult)
    assert result.compile_only is False
    assert result.shape["m"] == 64
    assert result.shape["out_features"] == 64
    if torch.cuda.is_available():
        assert result.status in {"ok", "error"}
        assert result.device == "cuda"
        assert result.compile_status in {"ok", "error"}
    else:
        assert result.status == "skipped"
        assert result.compile_status == "not_requested"
        assert "CUDA runtime is not available" in str(result.reason)


@requires_tilelang
@requires_nvcc
def test_tilelang_packed_fp4_fused_epilogue_kernel_compiles_with_explicit_arch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("TILELANG_CACHE_DIR", str(tmp_path / "tilelang-cache"))

    kernel = build_tilelang_fp4_fused_dequant_gemm_kernel(
        64,
        64,
        32,
        16,
        target_arch="sm_80",
        has_bias=True,
        activation="silu",
    )

    assert callable(kernel)


@requires_tilelang
@requires_nvcc
def test_tilelang_packed_fp4_validation_compile_only(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("TILELANG_CACHE_DIR", str(tmp_path / "tilelang-cache"))

    result = validate_tilelang_packed_fp4_fused_gemm(
        target_arch="sm_80",
        compile_only=True,
        warmup=0,
        iterations=1,
    )

    assert result.status == "ok"
    assert result.compile_only is True
    assert result.compile_status == "ok"
    assert result.target_arch == "sm_80"
    assert result.allclose is None


@requires_cuda
@requires_tilelang
def test_tilelang_dequant_gemm_cuda_matches_reference() -> None:
    torch.manual_seed(0)
    x = torch.randn(64, 32, device="cuda", dtype=torch.float16)
    qweight = torch.randn(64, 32, device="cuda", dtype=torch.float16)
    scale = torch.randn(64, device="cuda", dtype=torch.float16).abs() + 0.01
    bias = torch.randn(64, device="cuda", dtype=torch.float16)

    output = dequant_gemm_epilogue_tilelang(
        x,
        qweight,
        scale,
        bias,
        activation="silu",
        block_m=64,
        block_n=64,
        threads=128,
        num_stages=2,
    )
    reference = dequant_gemm_epilogue_reference(
        x,
        qweight,
        scale,
        bias,
        activation="silu",
    )

    assert output.shape == reference.shape
    assert output.dtype == torch.float16
    assert torch.allclose(output.float(), reference.float(), atol=1e-2, rtol=1e-2)


@requires_cuda
@requires_tilelang
def test_tilelang_packed_fp4_validation_cuda_runtime() -> None:
    result = validate_tilelang_packed_fp4_fused_gemm(
        target_arch=None,
        compile_only=False,
        warmup=1,
        iterations=2,
    )

    assert result.status == "ok"
    assert result.device == "cuda"
    assert result.compile_status == "ok"
    assert result.allclose is True
    assert result.max_abs_error is not None
    assert result.mean_abs_error is not None
    assert result.latency_ms_tilelang is not None
    assert result.latency_ms_reference is not None


@requires_cuda
@requires_tilelang
def test_tilelang_packed_fp4_dequant_gemm_cuda_matches_reference() -> None:
    torch.manual_seed(1)
    linear = torch.nn.Linear(32, 64)
    fp4_linear = ReferenceFP4Linear.from_linear(linear, group_size=16).to("cuda")
    x = torch.randn(64, 32, device="cuda", dtype=torch.float16)
    packed_weight = fp4_linear.packed_weight
    scale = fp4_linear.weight_scale.to(dtype=torch.float16)
    bias = fp4_linear.bias.to(dtype=torch.float16) if fp4_linear.bias is not None else None

    output = fp4_packed_dequant_gemm_epilogue_tilelang(
        x,
        packed_weight,
        scale,
        bias,
        input_features=fp4_linear.input_features,
        group_size=fp4_linear.group_size,
        block_m=64,
        block_n=64,
        threads=128,
        num_stages=2,
    )
    reference = fp4_packed_dequant_gemm_epilogue_reference(
        x,
        packed_weight,
        scale,
        bias,
        input_features=fp4_linear.input_features,
        group_size=fp4_linear.group_size,
    )

    assert output.shape == reference.shape
    assert output.dtype == torch.float16
    assert torch.allclose(output.float(), reference.float(), atol=1e-2, rtol=1e-2)
