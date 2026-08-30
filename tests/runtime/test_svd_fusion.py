from __future__ import annotations

import pytest
import torch
from torch import nn

from tests.xqt.svd_test_helpers import (
    make_legacy_svd_int8,
    make_legacy_svd_linear,
)
from xqt.core.errors import XQTBackendError
from xqt.runtime.modules import SVDQuantInt8MmaLinear, SVDQuantLinear
from xqt.runtime.svd_fusion import (
    fused_svd_forward,
    fused_svd_forward_cuda,
    svd_fusion_report,
)
from xqt.kernels.ops._impl.tilelang.svd_fused import resolve_svd_fused_schedule


def _make_svd_linear() -> SVDQuantLinear:
    torch.manual_seed(0)
    linear = nn.Linear(16, 8, bias=True)
    return make_legacy_svd_linear(linear, rank=4, group_size=8)


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


def test_svd_fused_sm89_schedule_promotes_short_prefill_only() -> None:
    schedule, reason = resolve_svd_fused_schedule(
        64,
        1024,
        1024,
        32,
        target_arch="sm_89",
    )
    assert schedule is not None
    assert schedule.to_dict() == {
        "block_m": 64,
        "block_n": 64,
        "block_k": 64,
        "threads": 128,
        "num_stages": 2,
    }
    assert reason == "promoted_svd_fused_schedule"

    large_schedule, large_reason = resolve_svd_fused_schedule(
        1024,
        2048,
        2048,
        64,
        target_arch="sm_89",
    )
    assert large_schedule is None
    assert "cached-dequant reference" in large_reason


def _tilelang_cuda_available() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        from xqt.kernels.ops._impl.tilelang._common import tilelang_runtime_usable
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
    module = make_legacy_svd_linear(
        linear,
        rank=rank,
        group_size=group_size,
    )
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


def _native_svdq_w8a8_test_available() -> bool:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 9):
        return False
    try:
        from xqt.kernels.ops._impl.cute.svdq_w8a8_sm89 import (
            native_svdq_w8a8_available,
        )
    except Exception:
        return False
    return native_svdq_w8a8_available(build=False)


def test_svdq_w4a4_native_metadata_reports_real_backend() -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 9):
        pytest.skip("native sm_89 W4A4 backend unavailable")
    try:
        from xqt.kernels.ops._impl.cute.svdq_w4a4_sm89 import (
            native_w4a4_available,
        )
    except Exception:
        pytest.skip("native sm_89 W4A4 backend unavailable")
    if not native_w4a4_available(build=False):
        pytest.skip("native sm_89 W4A4 backend unavailable")

    module = make_legacy_svd_linear(
        nn.Linear(128, 128, bias=True).eval(),
        rank=16,
        group_size=128,
        quant_dtype="int4",
    ).half().cuda().eval()
    assert module.enable_fusion() is True

    with torch.no_grad():
        output = module(torch.randn(13, 128, device="cuda", dtype=torch.float16))
    torch.cuda.synchronize()
    metadata = module.execution_metadata()

    assert output.shape == (13, 128)
    assert metadata["implementation"] == "native_svdq_w4a4_dynamic_lora"
    assert metadata["cuda_fused_backend"] == "native_w4a4_dynamic"
    assert metadata["cuda_fused_used"] is True


def test_svdq_w4a4_hot_cache_rebuilds_after_residual_scale_mutation() -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 9):
        pytest.skip("native sm_89 W4A4 backend unavailable")
    try:
        from xqt.kernels.ops._impl.cute.svdq_w4a4_sm89 import (
            native_w4a4_available,
        )
    except Exception:
        pytest.skip("native sm_89 W4A4 backend unavailable")
    if not native_w4a4_available(build=False):
        pytest.skip("native sm_89 W4A4 backend unavailable")

    module = make_legacy_svd_linear(
        nn.Linear(128, 128, bias=True).eval(),
        rank=16,
        group_size=128,
        quant_dtype="int4",
    ).half().cuda().eval()
    assert module.enable_fusion() is True
    inputs = torch.randn(13, 128, device="cuda", dtype=torch.float16)

    with torch.no_grad():
        before = module(inputs)
        first_cache = module._native_w4a4_packed_cache
        module(inputs)
    assert first_cache is not None

    with torch.no_grad():
        module.residual_scale.mul_(2.0)
    with torch.no_grad():
        after = module(inputs)
    torch.cuda.synchronize()
    second_cache = module._native_w4a4_packed_cache

    assert second_cache is not None
    assert second_cache[1] is not first_cache[1]
    assert not torch.equal(before, after)


def _make_native_w8_svd(rank: int) -> SVDQuantInt8MmaLinear:
    torch.manual_seed(53 + int(rank))
    source = nn.Linear(128, 128, bias=True).eval()
    return make_legacy_svd_int8(
        source,
        rank=rank,
        group_size=128,
        quant_dtype="int4",
        engine="torch_int_mm",
        activation_scale_mode="dynamic",
    ).bfloat16().cuda().eval()


def test_svdq_w8a8_native_route_cache_stream_and_fair_split() -> None:
    if not _native_svdq_w8a8_test_available():
        pytest.skip("native sm_89 SVDQuant W8A8 backend unavailable")

    from xqt.kernels.ops._impl.cute.svdq_w8a8_sm89 import (
        allocate_svdq_w8a8_workspace,
        w8a8_linear,
    )

    module = _make_native_w8_svd(16)
    assert module.enable_fusion() is True
    inputs = torch.randn(37, 128, device="cuda", dtype=torch.bfloat16)

    first = module(inputs)
    packed_first = module._native_w8a8_packed_cache
    second = module(inputs)
    packed_second = module._native_w8a8_packed_cache
    assert packed_first is not None and packed_second is not None

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        stream_output = module(inputs)
    stream.synchronize()
    after_stream = module(inputs)

    packed = packed_first[1]
    split_workspace = allocate_svdq_w8a8_workspace(int(inputs.shape[0]), packed)
    residual = w8a8_linear(inputs, packed, workspace=split_workspace)
    split = residual + module._low_rank(inputs, residual)
    torch.cuda.synchronize()

    metadata = module.execution_metadata()
    assert first.shape == (37, 128)
    assert first.dtype == torch.bfloat16
    assert torch.equal(first, second)
    assert torch.equal(first, stream_output)
    assert torch.equal(first, after_stream)
    torch.testing.assert_close(first, split, rtol=4e-2, atol=4e-2)
    assert packed_first[1] is packed_second[1]
    assert len(module._native_w8a8_workspace_cache) == 2
    assert metadata["implementation"] == "native_svdq_w8a8_dynamic_lora"
    assert metadata["native_w8a8_fusion_enabled"] is True
    assert metadata["native_w8a8_used"] is True
    assert metadata["native_w8a8_fallback_reason"] is None


def test_svdq_w8a8_native_hot_cache_invalidates_after_scale_mutation() -> None:
    if not _native_svdq_w8a8_test_available():
        pytest.skip("native sm_89 SVDQuant W8A8 backend unavailable")

    module = _make_native_w8_svd(16)
    assert module.enable_fusion() is True
    inputs = torch.randn(19, 128, device="cuda", dtype=torch.bfloat16)

    before = module(inputs)
    module(inputs)
    first_cache = module._native_w8a8_packed_cache
    assert first_cache is not None

    with torch.no_grad():
        module.residual_int8.group_scale.mul_(2.0)
    after = module(inputs)
    torch.cuda.synchronize()
    second_cache = module._native_w8a8_packed_cache

    assert second_cache is not None
    assert second_cache[1] is not first_cache[1]
    assert not torch.equal(before, after)


@pytest.mark.parametrize("rank", [16, 32, 48, 64, 80])
def test_svdq_w8a8_native_specialized_and_generic_rank_reset(rank: int) -> None:
    if not _native_svdq_w8a8_test_available():
        pytest.skip("native sm_89 SVDQuant W8A8 backend unavailable")

    module = _make_native_w8_svd(rank)
    assert module.enable_fusion() is True
    inputs = torch.randn(13, 128, device="cuda", dtype=torch.bfloat16)

    first = module(inputs)
    second = module(inputs)
    torch.cuda.synchronize()

    assert torch.equal(first, second)
    assert module.execution_metadata()["native_w8a8_used"] is True


def test_svdq_w8a8_native_pads_mnk_and_rank() -> None:
    if not _native_svdq_w8a8_test_available():
        pytest.skip("native sm_89 SVDQuant W8A8 backend unavailable")

    module = make_legacy_svd_int8(
        nn.Linear(132, 132, bias=True).eval(),
        rank=17,
        group_size=132,
        quant_dtype="int4",
        engine="torch_int_mm",
        activation_scale_mode="dynamic",
    ).bfloat16().cuda().eval()
    assert module.enable_fusion() is True

    output = module(torch.randn(19, 132, device="cuda", dtype=torch.bfloat16))
    torch.cuda.synchronize()
    cached = module._native_w8a8_packed_cache
    assert cached is not None
    packed = cached[1]

    assert output.shape == (19, 132)
    assert packed.padded_input_features == 256
    assert packed.padded_output_features == 256
    assert packed.padded_rank == 32
    assert tuple(packed.qweight.shape) == (256, 256)
    workspace = next(iter(module._native_w8a8_workspace_cache.values()))
    assert workspace.padded_rows == 256


def test_svdq_w8a8_native_rejects_non_vector_aligned_features() -> None:
    from xqt.kernels.ops._impl.cute.svdq_w8a8_sm89 import (
        native_svdq_w8a8_shape_supported,
    )

    assert native_svdq_w8a8_shape_supported(128, 128, 16) is True
    assert native_svdq_w8a8_shape_supported(130, 128, 16) is False
    assert native_svdq_w8a8_shape_supported(128, 130, 16) is False
    assert native_svdq_w8a8_shape_supported(128, 128, 1025) is False


def test_svdq_w8a8_native_contract_falls_back_explicitly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    monkeypatch.setattr(torch, "compile", lambda function, mode: function)
    fp16_module = make_legacy_svd_int8(
        nn.Linear(128, 128, bias=False).eval(),
        rank=16,
        group_size=128,
        engine="torch_int_mm",
        activation_scale_mode="dynamic",
    ).half().cuda().eval()
    static_bf16_module = make_legacy_svd_int8(
        nn.Linear(128, 128, bias=False).eval(),
        rank=16,
        group_size=128,
        engine="torch_int_mm",
        activation_scale_mode="static",
        activation_scale=0.02,
    ).bfloat16().cuda().eval()
    misaligned_module = make_legacy_svd_int8(
        nn.Linear(130, 126, bias=False).eval(),
        rank=16,
        group_size=130,
        engine="torch_int_mm",
        activation_scale_mode="dynamic",
    ).bfloat16().cuda().eval()

    assert fp16_module.enable_fusion() is True
    assert static_bf16_module.enable_fusion() is True
    assert misaligned_module.enable_fusion() is True
    assert fp16_module._native_w8a8_fusion_enabled is False
    assert static_bf16_module._native_w8a8_fusion_enabled is False
    assert misaligned_module._native_w8a8_fusion_enabled is False
    assert fp16_module._fused_forward is not None
    assert static_bf16_module._fused_forward is not None
    assert misaligned_module._fused_forward is not None

    native_module = _make_native_w8_svd(16)
    assert native_module.enable_fusion() is True
    native_module.residual_int8.min_int8_rows = 64
    allowed, reason = native_module._native_w8a8_gate(
        torch.randn(13, 128, device="cuda", dtype=torch.bfloat16)
    )
    assert allowed is False
    assert reason == "input rows are below min_int8_rows"
