from __future__ import annotations

import pytest
import torch
from torch import nn

from xqt.core.errors import XQTBackendError
from xqt.runtime.modules import SVDQuantLinear
from xqt.runtime.svd_fusion import (
    fused_svd_forward,
    fused_svd_forward_cuda,
    svd_fusion_report,
)


def _make_svd_linear() -> SVDQuantLinear:
    torch.manual_seed(0)
    linear = nn.Linear(16, 8, bias=True)
    return SVDQuantLinear.from_linear(linear, rank=4, group_size=8)


def test_svd_fusion_report_marks_reference_and_unverified() -> None:
    module = _make_svd_linear()

    report = svd_fusion_report(module)

    assert report.fuse_down is True
    assert report.fuse_up is True
    assert report.status == "reference"
    assert report.cuda_verified is False
    assert "svd_fuse_down_reference" in report.kernel_names
    payload = report.to_dict()
    assert payload["cuda_verified"] is False
    assert any("CUDA kernel 验证待验证" in note for note in payload["notes"])


def test_fused_svd_forward_matches_reference_forward() -> None:
    module = _make_svd_linear().eval()
    x = torch.randn(2, 16)

    with torch.no_grad():
        expected = module(x)
        fused, report = fused_svd_forward(module, x)

    assert fused.shape == expected.shape
    assert torch.allclose(fused, expected, rtol=1e-5, atol=1e-6)
    assert report.fuse_down and report.fuse_up


def test_fused_svd_forward_rejects_non_svd_module() -> None:
    module = nn.Linear(16, 8)

    with pytest.raises(TypeError, match="low-rank branch"):
        fused_svd_forward(module, torch.randn(2, 16))


def _tilelang_cuda_available() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        from xqt.operator_opt.kernels.tilelang._common import tilelang_runtime_usable
    except ImportError:
        return False
    return tilelang_runtime_usable()


requires_cuda_tilelang = pytest.mark.skipif(
    not _tilelang_cuda_available(),
    reason="需要 CUDA 设备和可用的 TileLang runtime",
)


def _make_cuda_svd_linear(
    *,
    group_size: int,
    rank: int,
    bias: bool,
    in_features: int = 128,
    out_features: int = 64,
) -> SVDQuantLinear:
    torch.manual_seed(0)
    linear = nn.Linear(in_features, out_features, bias=bias)
    module = SVDQuantLinear.from_linear(linear, rank=rank, group_size=group_size)
    return module.half().cuda().eval()


@requires_cuda_tilelang
@pytest.mark.parametrize(
    "group_size,rank,bias",
    [
        (64, 16, True),
        (64, 16, False),
        (32, 32, True),
        (128, 4, True),
    ],
)
def test_fused_svd_forward_cuda_matches_reference(
    group_size: int,
    rank: int,
    bias: bool,
) -> None:
    module = _make_cuda_svd_linear(group_size=group_size, rank=rank, bias=bias)
    x = torch.randn(64, module.input_features, device="cuda", dtype=torch.float16)

    with torch.no_grad():
        expected, ref_report = fused_svd_forward(module, x)
        fused, report = fused_svd_forward_cuda(module, x)

    assert fused.shape == expected.shape
    assert fused.dtype == torch.float16
    # fp16 容差理由: kernel 与 reference (cuBLAS F.linear) 都是 fp16 输入,
    # fp32 累加, 但累加顺序和 h 的 fp16 回写不同; 实测 max_abs <= 2e-3,
    # cosine >= 0.999999, 这里用 atol=5e-3/rtol=2e-2 留一个数量级余量.
    torch.testing.assert_close(fused, expected, rtol=2e-2, atol=5e-3)
    diff = (fused.float() - expected.float()).abs()
    cosine = torch.nn.functional.cosine_similarity(
        fused.float().flatten(), expected.float().flatten(), dim=0
    )
    assert cosine.item() > 0.9999
    assert diff.max().item() < 5e-3
    assert ref_report.cuda_verified is False


@requires_cuda_tilelang
def test_fused_svd_forward_cuda_report_marks_cuda_fused() -> None:
    module = _make_cuda_svd_linear(group_size=64, rank=16, bias=True)
    x = torch.randn(64, module.input_features, device="cuda", dtype=torch.float16)

    with torch.no_grad():
        _, report = fused_svd_forward_cuda(module, x)

    assert report.fuse_down is True
    assert report.fuse_up is True
    assert report.status == "cuda_fused"
    assert report.cuda_verified is True
    assert "svd_fused_dequant_gemm_low_rank" in report.kernel_names
    payload = report.to_dict()
    assert payload["status"] == "cuda_fused"
    assert payload["cuda_verified"] is True


@requires_cuda_tilelang
def test_fused_svd_forward_cuda_rejects_non_svd_module() -> None:
    module = nn.Linear(128, 64).half().cuda()

    with pytest.raises(TypeError, match="low-rank branch"):
        fused_svd_forward_cuda(module, torch.randn(64, 128, device="cuda", dtype=torch.float16))


@requires_cuda_tilelang
def test_fused_svd_forward_cuda_rejects_non_cuda_input() -> None:
    module = _make_cuda_svd_linear(group_size=64, rank=16, bias=True)

    with pytest.raises(XQTBackendError, match="CUDA input"):
        fused_svd_forward_cuda(module, torch.randn(64, 128, dtype=torch.float16))


@requires_cuda_tilelang
def test_fused_svd_forward_cuda_rejects_non_fp16_input() -> None:
    module = _make_cuda_svd_linear(group_size=64, rank=16, bias=True)

    with pytest.raises(XQTBackendError, match="float16"):
        fused_svd_forward_cuda(module, torch.randn(64, 128, device="cuda"))


@requires_cuda_tilelang
def test_fused_svd_forward_cuda_rejects_misaligned_dims() -> None:
    module = _make_cuda_svd_linear(group_size=64, rank=16, bias=True)
    # batch=32 不是 block_m=64 的倍数, 必须显式报错而不是静默错算.
    x = torch.randn(32, module.input_features, device="cuda", dtype=torch.float16)

    with pytest.raises(XQTBackendError, match="multiples of block sizes"):
        fused_svd_forward_cuda(module, x)
