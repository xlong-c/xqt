from __future__ import annotations

import importlib.util

import pytest
import torch

from xqt.operator_opt.kernels.tilelang.attention import (
    dequant_gemm_epilogue_reference,
    dequant_gemm_epilogue_tilelang,
)


requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for TileLang dequant GEMM CUDA test",
)

requires_tilelang = pytest.mark.skipif(
    importlib.util.find_spec("tilelang") is None,
    reason="tilelang package is required for TileLang dequant GEMM CUDA test",
)


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
