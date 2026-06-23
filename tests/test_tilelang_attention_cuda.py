from __future__ import annotations

import importlib.util

import pytest
import torch

from xqt.operator_opt.kernels.tilelang.attention import (
    fused_attention_forward_reference,
    fused_attention_forward_tilelang,
)


requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for TileLang attention CUDA test",
)

requires_tilelang = pytest.mark.skipif(
    importlib.util.find_spec("tilelang") is None,
    reason="tilelang package is required for TileLang attention CUDA test",
)


@requires_cuda
@requires_tilelang
def test_tilelang_attention_cuda_kernel_matches_reference() -> None:
    torch.manual_seed(0)
    q = torch.randn(1, 1, 64, 32, device="cuda", dtype=torch.float16)
    k = torch.randn(1, 1, 64, 32, device="cuda", dtype=torch.float16)
    v = torch.randn(1, 1, 64, 32, device="cuda", dtype=torch.float16)

    output = fused_attention_forward_tilelang(
        q,
        k,
        v,
        causal=False,
        dropout_p=0.0,
        block_m=64,
        block_n=64,
        threads=128,
        num_stages=2,
    )
    reference = fused_attention_forward_reference(
        q,
        k,
        v,
        causal=False,
        dropout_p=0.0,
    )

    assert output.shape == reference.shape
    assert output.dtype == torch.float16
    assert torch.allclose(output.float(), reference.float(), atol=1e-2, rtol=1e-2)
