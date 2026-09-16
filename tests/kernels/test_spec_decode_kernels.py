"""Spec-decode kernel tests: multi-row verify attention and RoPE/KV scatter."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from xqt.kernels.ops._impl.triton.spec_decode_kernels import (
    decode_attention_rows_forward_triton,
    rope_write_qkv_rows_triton,
)

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="spec decode kernels require CUDA"
)

HEADS = 8
KV_HEADS = 2
HEAD_DIM = 64
MAX_LEN = 256


@requires_cuda
@pytest.mark.parametrize("rows", [1, 2, 4, 8])
def test_rows_attention_matches_per_row_sdpa(rows: int) -> None:
    torch.manual_seed(0)
    valid = torch.tensor(
        [5, 17, 33, 129, 200, 64, 7, 256][:rows],
        dtype=torch.int32,
        device="cuda",
    )
    q = torch.randn(1, HEADS, rows, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(1, KV_HEADS, MAX_LEN, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(1, KV_HEADS, MAX_LEN, HEAD_DIM, dtype=torch.bfloat16, device="cuda")

    got = decode_attention_rows_forward_triton(q, k, v, valid, splits=4)
    for row in range(rows):
        length = int(valid[row])
        ref = F.scaled_dot_product_attention(
            q[:, :, row : row + 1],
            k[:, :, :length],
            v[:, :, :length],
            enable_gqa=True,
        )
        torch.testing.assert_close(
            got[:, row : row + 1].float(), ref[0].float(), atol=2e-2, rtol=2e-2
        )


@requires_cuda
def test_rope_write_qkv_rows_matches_hf_rotation() -> None:
    torch.manual_seed(3)
    rows = 4
    start = 17
    qkv = torch.randn(1, HEADS + 2 * KV_HEADS, rows, HEAD_DIM, device="cuda")
    cos = torch.randn(MAX_LEN, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    sin = torch.randn(MAX_LEN, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    k_cache = torch.zeros(
        1, KV_HEADS, MAX_LEN, HEAD_DIM, dtype=torch.bfloat16, device="cuda"
    )
    v_cache = torch.zeros(
        1, KV_HEADS, MAX_LEN, HEAD_DIM, dtype=torch.bfloat16, device="cuda"
    )
    q_out = torch.empty_like(qkv[:, :HEADS])
    position = torch.tensor([start], dtype=torch.long, device="cuda")

    rope_write_qkv_rows_triton(
        qkv[:, :HEADS],
        qkv[:, HEADS : HEADS + KV_HEADS],
        qkv[:, HEADS + KV_HEADS :],
        cos,
        sin,
        q_out,
        k_cache,
        v_cache,
        position,
    )

    def rotate(x: torch.Tensor) -> torch.Tensor:
        half = x.shape[-1] // 2
        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat((-x2, x1), dim=-1)

    for row in range(rows):
        pos = start + row
        for head in range(HEADS):
            ref = (
                qkv[0, head, row].float() * cos[pos].float()
                + rotate(qkv[0, head, row]).float() * sin[pos].float()
            )
            torch.testing.assert_close(
                q_out[0, head, row].float(), ref, rtol=2e-2, atol=2e-2
            )
        for head in range(KV_HEADS):
            ref_k = (
                qkv[0, HEADS + head, row].float() * cos[pos].float()
                + rotate(qkv[0, HEADS + head, row]).float() * sin[pos].float()
            )
            torch.testing.assert_close(
                k_cache[0, head, pos].float(), ref_k, rtol=2e-2, atol=2e-2
            )
            torch.testing.assert_close(
                v_cache[0, head, pos].float(),
                qkv[0, HEADS + KV_HEADS + head, row].float(),
                rtol=2e-2,
                atol=2e-2,
            )
