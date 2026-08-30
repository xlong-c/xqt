from __future__ import annotations

import pytest
import torch

from xqt.kernels.ops._impl.tilelang.hunyuan_block import (
    gqa_decode_attention_reference,
    gqa_decode_attention_tilelang,
    residual_add_reference,
    residual_add_tilelang,
    residual_rmsnorm_reference,
    residual_rmsnorm_tilelang,
    rmsnorm_reference,
    rmsnorm_tilelang,
    swiglu_reference,
    swiglu_tilelang,
)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="TileLang Hunyuan block kernels require CUDA",
)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_tilelang_hunyuan_rmsnorm_matches_reference(dtype: torch.dtype) -> None:
    torch.manual_seed(0)
    hidden_states = torch.randn(4, 1024, device="cuda", dtype=dtype)
    weight = torch.randn(1024, device="cuda", dtype=dtype)

    actual = rmsnorm_tilelang(hidden_states, weight, eps=1e-5, target_arch="sm_89")
    expected = rmsnorm_reference(hidden_states, weight, eps=1e-5)

    torch.testing.assert_close(actual.float(), expected.float(), rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_tilelang_hunyuan_residual_rmsnorm_matches_reference(
    dtype: torch.dtype,
) -> None:
    torch.manual_seed(1)
    hidden_states = torch.randn(4, 1024, device="cuda", dtype=dtype)
    residual = torch.randn_like(hidden_states)
    weight = torch.randn(1024, device="cuda", dtype=dtype)

    actual_residual, actual_normed = residual_rmsnorm_tilelang(
        hidden_states,
        residual,
        weight,
        eps=1e-5,
        target_arch="sm_89",
    )
    expected_residual, expected_normed = residual_rmsnorm_reference(
        hidden_states,
        residual,
        weight,
        eps=1e-5,
    )

    torch.testing.assert_close(
        actual_residual.float(), expected_residual.float(), rtol=2e-2, atol=2e-2
    )
    torch.testing.assert_close(
        actual_normed.float(), expected_normed.float(), rtol=2e-2, atol=2e-2
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_tilelang_hunyuan_swiglu_matches_reference(dtype: torch.dtype) -> None:
    torch.manual_seed(2)
    gate = torch.randn(4, 3584, device="cuda", dtype=dtype)
    up = torch.randn_like(gate)

    actual = swiglu_tilelang(gate, up, target_arch="sm_89")
    expected = swiglu_reference(gate, up)

    torch.testing.assert_close(actual.float(), expected.float(), rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_tilelang_hunyuan_residual_add_matches_reference(dtype: torch.dtype) -> None:
    torch.manual_seed(4)
    hidden_states = torch.randn(4, 1024, device="cuda", dtype=dtype)
    residual = torch.randn_like(hidden_states)

    actual = residual_add_tilelang(hidden_states, residual, target_arch="sm_89")
    expected = residual_add_reference(hidden_states, residual)

    torch.testing.assert_close(actual.float(), expected.float(), rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_tilelang_hunyuan_gqa_decode_matches_reference(dtype: torch.dtype) -> None:
    torch.manual_seed(3)
    query = torch.randn(1, 16, 128, device="cuda", dtype=dtype)
    key_cache = torch.randn(1, 8, 128, 128, device="cuda", dtype=dtype)
    value_cache = torch.randn_like(key_cache)
    actual = gqa_decode_attention_tilelang(
        query,
        key_cache,
        value_cache,
        query_tile_rows=1,
        target_arch="sm_89",
    )
    expected = gqa_decode_attention_reference(
        query,
        key_cache,
        value_cache,
    )

    torch.testing.assert_close(actual.float(), expected.float(), rtol=3e-2, atol=3e-2)
