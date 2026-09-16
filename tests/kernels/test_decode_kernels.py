"""Decode-kernel tests: GQA attention with a runtime length, fused RoPE + KV scatter."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from xqt.core.errors import XQTBackendError
from xqt.kernels.ops._impl.triton.decode_kernels import (
    decode_attention_forward_triton,
    decode_attention_forward_triton_tc,
    rope_write_qkv_triton,
)

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="decode kernels require CUDA"
)

HEADS = 8
KV_HEADS = 2
HEAD_DIM = 64
MAX_LEN = 256


@requires_cuda
@pytest.mark.parametrize("valid_len", [1, 5, 37, 128, 256])
def test_decode_attention_matches_gqa_sdpa(valid_len: int) -> None:
    torch.manual_seed(0)
    q = torch.randn(1, HEADS, 1, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(1, KV_HEADS, MAX_LEN, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(1, KV_HEADS, MAX_LEN, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    length = torch.tensor([valid_len], dtype=torch.int32, device="cuda")

    got = decode_attention_forward_triton(q, k, v, length, splits=4)
    ref = F.scaled_dot_product_attention(
        q, k[:, :, :valid_len], v[:, :, :valid_len], enable_gqa=True
    )
    torch.testing.assert_close(got.float(), ref.float(), atol=2e-2, rtol=2e-2)


@requires_cuda
def test_decode_attention_reads_length_at_runtime() -> None:
    torch.manual_seed(3)
    q = torch.randn(1, HEADS, 1, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(1, KV_HEADS, MAX_LEN, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(1, KV_HEADS, MAX_LEN, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    length = torch.tensor([16], dtype=torch.int32, device="cuda")

    short = decode_attention_forward_triton(q, k, v, length, splits=4)
    length.fill_(256)
    full = decode_attention_forward_triton(q, k, v, length, splits=4)

    ref_full = F.scaled_dot_product_attention(q, k, v, enable_gqa=True)
    torch.testing.assert_close(full.float(), ref_full.float(), atol=2e-2, rtol=2e-2)
    assert not torch.allclose(short, full, atol=1e-3)


@requires_cuda
def test_decode_attention_rejects_bad_split_geometry() -> None:
    torch.manual_seed(7)
    q = torch.randn(1, HEADS, 1, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(1, KV_HEADS, MAX_LEN, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(1, KV_HEADS, MAX_LEN, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    length = torch.tensor([MAX_LEN], dtype=torch.int32, device="cuda")

    with pytest.raises(XQTBackendError, match="power of two"):
        decode_attention_forward_triton(q, k, v, length, splits=4, block_l=96)


@requires_cuda
def test_rope_write_qkv_matches_hf_rotation() -> None:
    torch.manual_seed(5)
    q = torch.randn(1, HEADS, 1, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(1, KV_HEADS, 1, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(1, KV_HEADS, 1, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    cos = torch.randn(MAX_LEN, HEAD_DIM, dtype=torch.bfloat16, device="cuda") * 0.5
    sin = torch.randn(MAX_LEN, HEAD_DIM, dtype=torch.bfloat16, device="cuda") * 0.5
    position = torch.tensor([7], dtype=torch.long, device="cuda")
    k_cache = torch.zeros(
        1, KV_HEADS, MAX_LEN, HEAD_DIM, dtype=torch.bfloat16, device="cuda"
    )
    v_cache = torch.zeros_like(k_cache)

    def rotate(x: torch.Tensor) -> torch.Tensor:
        half = x.shape[-1] // 2
        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat((-x2, x1), dim=-1)

    row_cos = cos[7].view(1, 1, 1, HEAD_DIM)
    row_sin = sin[7].view(1, 1, 1, HEAD_DIM)
    want_q = q * row_cos + rotate(q) * row_sin
    want_k = k * row_cos + rotate(k) * row_sin
    q_out = torch.empty_like(q)

    rope_write_qkv_triton(q, k, v, cos, sin, q_out, k_cache, v_cache, position)
    torch.testing.assert_close(q_out, want_q, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(k_cache[:, :, 7], want_k[:, :, 0], atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(v_cache[:, :, 7], v[:, :, 0])


@requires_cuda
def test_int8_activation_kernels_quantize_rows() -> None:
    from xqt.kernels.ops._impl.triton.decode_kernels import (
        rmsnorm_int8_triton,
        swiglu_int8_triton,
    )

    torch.manual_seed(9)
    hidden = torch.randn(1, 1, 64, dtype=torch.bfloat16, device="cuda") * 2.0
    weight = torch.rand(64, dtype=torch.bfloat16, device="cuda") + 0.5
    normed = rmsnorm_int8_triton(hidden, weight, eps=1e-5)

    fp = F.rms_norm(hidden, (64,), weight, 1e-5)
    scale = fp.float().abs().amax(dim=-1, keepdim=True) / 127.0
    reference = (fp.float() / scale).round().clamp(-127, 127) * scale
    torch.testing.assert_close(normed.float(), reference, atol=2e-2, rtol=2e-2)

    gate = torch.randn(1, 4, 96, dtype=torch.bfloat16, device="cuda")
    up = torch.randn(1, 4, 96, dtype=torch.bfloat16, device="cuda")
    fused = swiglu_int8_triton(gate, up)
    silu = F.silu(gate.float()) * up.float()
    scale2 = silu.abs().amax(dim=-1, keepdim=True) / 127.0
    reference2 = (silu / scale2).round().clamp(-127, 127) * scale2
    torch.testing.assert_close(fused.float(), reference2, atol=2e-2, rtol=2e-2)


@requires_cuda
@pytest.mark.parametrize("valid_len", [1, 5, 37, 128, 256])
def test_decode_attention_tc_matches_gqa_sdpa(valid_len: int) -> None:
    torch.manual_seed(0)
    q = torch.randn(1, HEADS, 1, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(1, KV_HEADS, MAX_LEN, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(1, KV_HEADS, MAX_LEN, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    length = torch.tensor([valid_len], dtype=torch.int32, device="cuda")

    got = decode_attention_forward_triton_tc(q, k, v, length, splits=4)
    ref = F.scaled_dot_product_attention(
        q, k[:, :, :valid_len], v[:, :, :valid_len], enable_gqa=True
    )
    torch.testing.assert_close(got.float(), ref.float(), atol=2e-2, rtol=2e-2)


@requires_cuda
def test_decode_attention_tc_matches_simt_minicpm5_shape() -> None:
    """The tensor-core group tile agrees with the SIMT kernel on 16/2 heads."""

    torch.manual_seed(4)
    heads, kv_heads, head_dim, max_len, valid_len = 16, 2, 128, 512, 301
    q = torch.randn(1, heads, 1, head_dim, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(1, kv_heads, max_len, head_dim, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(1, kv_heads, max_len, head_dim, dtype=torch.bfloat16, device="cuda")
    length = torch.tensor([valid_len], dtype=torch.int32, device="cuda")

    tc = decode_attention_forward_triton_tc(q, k, v, length, splits=16)
    simt = decode_attention_forward_triton(q, k, v, length, splits=16)
    torch.testing.assert_close(tc.float(), simt.float(), atol=2e-2, rtol=2e-2)


@requires_cuda
def test_decode_attention_tc_reads_length_at_runtime() -> None:
    torch.manual_seed(3)
    q = torch.randn(1, HEADS, 1, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(1, KV_HEADS, MAX_LEN, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(1, KV_HEADS, MAX_LEN, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    length = torch.tensor([16], dtype=torch.int32, device="cuda")

    short = decode_attention_forward_triton_tc(q, k, v, length, splits=4)
    length.fill_(256)
    full = decode_attention_forward_triton_tc(q, k, v, length, splits=4)

    ref_full = F.scaled_dot_product_attention(q, k, v, enable_gqa=True)
    torch.testing.assert_close(full.float(), ref_full.float(), atol=2e-2, rtol=2e-2)
    assert not torch.allclose(short, full, atol=1e-3)


@requires_cuda
def test_decode_attention_tc_rejects_small_head_dim() -> None:
    torch.manual_seed(7)
    q = torch.randn(1, HEADS, 1, 8, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(1, KV_HEADS, 32, 8, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(1, KV_HEADS, 32, 8, dtype=torch.bfloat16, device="cuda")
    length = torch.tensor([16], dtype=torch.int32, device="cuda")

    with pytest.raises(XQTBackendError, match="head_dim"):
        decode_attention_forward_triton_tc(q, k, v, length)
