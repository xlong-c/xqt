from __future__ import annotations

import importlib
from pathlib import Path


def test_migrated_cuda_sources_have_canonical_locations() -> None:
    root = Path(__file__).resolve().parents[2]
    gemm = root / "xqt/kernels/jit/csrc/gemm"
    quantization = root / "xqt/kernels/jit/csrc/quantization"

    for name in (
        "dense_sm89.cu",
        "fp8_cutlass_sm89.cu",
        "sm90_fp8_wgmma.cu",
        "sm120_gemm.cu",
    ):
        assert (gemm / name).is_file()
    for name in (
        "awq_w4a16_sm89_kernel.cu",
        "convrot_w4a4_rowwise_sm89_kernel.cu",
        "svdq_w4a4_sm89_kernel.cu",
        "svdq_w8a8_sm89_kernel.cu",
    ):
        assert (quantization / name).is_file()


def test_canonical_python_kernel_modules_are_importable() -> None:
    names = (
        "xqt.kernels.ops._impl.triton.gemm",
        "xqt.kernels.ops._impl.tilelang.linear",
        "xqt.kernels.ops._impl.cute.int8mma_binding",
        "xqt.kernels.ops._impl.engines.triton",
        "xqt.kernels.ops._impl.engines.tilelang",
        "xqt.kernels.ops._impl.guidance",
    )
    for name in names:
        module = importlib.import_module(name)
        assert module.__file__ is not None
        assert Path(module.__file__).is_file()


def test_kernel_guidance_table_is_exported() -> None:
    guidance = importlib.import_module("xqt.kernels.ops._impl.guidance")

    assert guidance.KERNEL_GUIDANCE_TABLE


def test_canonical_cuda_sources_keep_runtime_probe_abi() -> None:
    root = Path(__file__).resolve().parents[2]
    sm90 = (root / "xqt/kernels/jit/csrc/gemm/sm90_fp8_wgmma.cu").read_text(encoding="utf-8")
    sm120 = (root / "xqt/kernels/jit/csrc/gemm/sm120_gemm.cu").read_text(encoding="utf-8")
    assert "xqt_sm90_fp8_e4m3_fp16_wgmma_run" in sm90
    assert "xqt_sm120_fp8_e4m3_blockwise_bf16_run" in sm120
