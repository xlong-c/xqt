from __future__ import annotations

import importlib.util

import pytest
import torch

from xqt.core.errors import XQTBackendError
from xqt.operator_opt.kernels.tilelang.conv import (
    conv2d_reference,
    conv2d_tilelang,
)
from xqt.operator_opt.kernels.tilelang.linear import (
    dense_linear_epilogue_reference,
    dense_linear_epilogue_tilelang,
    half_linear_reference,
    half_linear_tilelang,
    resolve_tilelang_linear_schedule,
)
from xqt.operator_opt.kernels.tilelang.norm import (
    layer_norm_reference,
    layer_norm_tilelang,
)
from xqt.operator_opt.kernels.tilelang._common import tilelang_runtime_usable


requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for TileLang half-op CUDA test",
)

requires_tilelang = pytest.mark.skipif(
    not tilelang_runtime_usable(),
    reason="a runtime-compatible TileLang adapter is required for TileLang half-op CUDA test",
)


@requires_cuda
@requires_tilelang
@pytest.mark.parametrize(
    "dtype",
    [torch.float16, torch.bfloat16],
    ids=["fp16", "bf16"],
)
def test_tilelang_half_linear_cuda_kernel_matches_reference(
    dtype: torch.dtype,
) -> None:
    torch.manual_seed(0)
    x = torch.randn(64, 64, device="cuda", dtype=dtype)
    weight = torch.randn(64, 64, device="cuda", dtype=dtype)
    bias = torch.randn(64, device="cuda", dtype=dtype)

    output = half_linear_tilelang(
        x,
        weight,
        bias,
        block_m=64,
        block_n=64,
        block_k=64,
        threads=128,
        num_stages=2,
    )
    reference = half_linear_reference(x, weight, bias)

    assert output.shape == reference.shape
    assert output.dtype == dtype
    assert torch.allclose(output.float(), reference.float(), atol=1e-2, rtol=1e-2)


@requires_cuda
@requires_tilelang
def test_tilelang_bf16_linear_supports_partial_m_n_and_fused_epilogue() -> None:
    torch.manual_seed(1)
    x = torch.randn(1, 64, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(96, 64, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(96, device="cuda", dtype=torch.bfloat16)

    output = dense_linear_epilogue_tilelang(
        x,
        weight,
        bias,
        activation="silu",
        target_arch="sm_89",
    )
    reference = dense_linear_epilogue_reference(
        x,
        weight,
        bias,
        activation="silu",
    )

    assert output.shape == (1, 96)
    assert output.dtype == torch.bfloat16
    assert torch.allclose(output.float(), reference.float(), atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize(
    ("m", "expected_blocks", "expected_preset"),
    [
        (1, (16, 64, 32), "sm89_bf16_decode_m_le_4"),
        (4, (16, 64, 32), "sm89_bf16_decode_m_le_4"),
        (8, (64, 64, 64), "default"),
    ],
)
def test_tilelang_bf16_sm89_schedule_promotion_is_decode_scoped(
    m: int,
    expected_blocks: tuple[int, int, int],
    expected_preset: str,
) -> None:
    x = torch.empty(m, 64, dtype=torch.bfloat16)

    schedule = resolve_tilelang_linear_schedule(x, target_arch="sm_89")

    assert (schedule.block_m, schedule.block_n, schedule.block_k) == expected_blocks
    assert schedule.preset == expected_preset


@pytest.mark.parametrize(
    ("m", "k", "out_features", "activation", "expected_preset"),
    [
        (1, 4096, 4096, None, "sm89_fp16_decode_m_le_4_n4096"),
        (4, 4096, 4096, None, "sm89_fp16_decode_m_le_4_n4096"),
        (8, 4096, 4096, None, "default"),
        (1, 4096, 11008, None, "default"),
        (1, 4096, 4096, "silu", "default"),
        (1, 2048, 4096, None, "default"),
    ],
)
def test_tilelang_fp16_sm89_schedule_promotion_is_exact_signature_scoped(
    m: int,
    k: int,
    out_features: int,
    activation: str | None,
    expected_preset: str,
) -> None:
    x = torch.empty(m, k, dtype=torch.float16)

    schedule = resolve_tilelang_linear_schedule(
        x,
        out_features=out_features,
        activation=activation,
        target_arch="sm_89",
    )

    expected_blocks = (16, 64, 32) if expected_preset != "default" else (64, 64, 64)
    assert (schedule.block_m, schedule.block_n, schedule.block_k) == expected_blocks
    assert schedule.preset == expected_preset


def test_tilelang_linear_schedule_preserves_explicit_overrides() -> None:
    x = torch.empty(1, 64, dtype=torch.bfloat16)

    schedule = resolve_tilelang_linear_schedule(
        x,
        block_m=32,
        block_n=128,
        block_k=16,
        target_arch="sm_89",
    )

    assert (schedule.block_m, schedule.block_n, schedule.block_k) == (32, 128, 16)
    assert schedule.preset == "sm89_bf16_decode_m_le_4"


@requires_cuda
def test_tilelang_linear_rejects_mixed_fp16_bf16_inputs() -> None:
    x = torch.randn(16, 64, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(64, 64, device="cuda", dtype=torch.float16)

    with pytest.raises(XQTBackendError, match="matching float16 or bfloat16"):
        half_linear_tilelang(x, weight, block_m=16, block_n=64, block_k=32)


@requires_cuda
def test_tilelang_bf16_linear_rejects_non_mma_block_size() -> None:
    x = torch.randn(16, 64, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)

    with pytest.raises(XQTBackendError, match="must be multiples of 16"):
        half_linear_tilelang(x, weight, block_m=16, block_n=64, block_k=8)


@requires_cuda
@requires_tilelang
def test_tilelang_half_conv_cuda_kernel_matches_reference() -> None:
    torch.manual_seed(0)
    x = torch.randn(1, 64, 8, 8, device="cuda", dtype=torch.float16)
    weight = torch.randn(64, 64, 1, 1, device="cuda", dtype=torch.float16)
    bias = torch.randn(64, device="cuda", dtype=torch.float16)

    output = conv2d_tilelang(
        x,
        weight,
        bias,
        stride=(1, 1),
        padding=(0, 0),
        block_m=64,
        block_n=64,
        block_k=64,
        threads=128,
        num_stages=2,
    )
    reference = conv2d_reference(
        x,
        weight,
        bias,
        stride=(1, 1),
        padding=(0, 0),
    )

    assert output.shape == reference.shape
    assert output.dtype == torch.float16
    assert torch.allclose(output.float(), reference.float(), atol=1e-2, rtol=1e-2)


@requires_cuda
@requires_tilelang
def test_tilelang_half_conv_cuda_kernel_matches_reference_for_spatial_kernel() -> None:
    torch.manual_seed(0)
    x = torch.randn(1, 32, 8, 8, device="cuda", dtype=torch.float16)
    weight = torch.randn(64, 32, 3, 3, device="cuda", dtype=torch.float16)
    bias = torch.randn(64, device="cuda", dtype=torch.float16)

    output = conv2d_tilelang(
        x,
        weight,
        bias,
        stride=(1, 1),
        padding=(1, 1),
        block_m=64,
        block_n=64,
        block_k=64,
        threads=128,
        num_stages=2,
    )
    reference = conv2d_reference(
        x,
        weight,
        bias,
        stride=(1, 1),
        padding=(1, 1),
    )

    assert output.shape == reference.shape
    assert output.dtype == torch.float16
    assert torch.allclose(output.float(), reference.float(), atol=1e-2, rtol=1e-2)


@requires_cuda
@requires_tilelang
def test_tilelang_half_norm_cuda_kernel_matches_reference() -> None:
    torch.manual_seed(0)
    x = torch.randn(8, 64, device="cuda", dtype=torch.float16)
    weight = torch.randn(64, device="cuda", dtype=torch.float16)
    bias = torch.randn(64, device="cuda", dtype=torch.float16)

    output = layer_norm_tilelang(x, weight, bias, eps=1e-5)
    reference = layer_norm_reference(x, weight, bias, eps=1e-5)

    assert output.shape == reference.shape
    assert output.dtype == torch.float16
    assert torch.allclose(output.float(), reference.float(), atol=1e-2, rtol=1e-2)
