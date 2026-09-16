"""Tests for attention backend registration and length-aware routing policy."""

from __future__ import annotations

import pytest
import torch

import xqt.kernels.ops.attention as attention_ops
from xqt.kernels.ops.attention import (
    AttentionRoutingDecision,
    fused_attention_sage,
    recommend_attention_backend,
)
from xqt.kernels.registry import registry
from xqt.kernels.selector import clear_cache, select_kernel
from xqt.kernels.spec import KernelBackend, KernelSpec

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for the Sage attention kernel",
)


def test_sage_backend_is_registered_and_resolvable() -> None:
    specs = registry.get("attention.fused_attention")
    sage_specs = [spec for spec in specs if spec.backend == KernelBackend.SAGE]
    assert len(sage_specs) == 1

    resolved = select_kernel("attention.fused_attention", backend=KernelBackend.SAGE)
    assert isinstance(resolved, KernelSpec)
    assert resolved.backend == KernelBackend.SAGE
    assert resolved.target == (
        "xqt.kernels.ops._impl.triton.sage_attention:sage_attention_forward_triton"
    )
    assert resolved.capabilities  # CUDA-only
    clear_cache()


def test_sage_backend_enum_value() -> None:
    assert KernelBackend.SAGE.value == "sage"


def test_sage_absent_from_default_precedence() -> None:
    from xqt.kernels.selector import _DEFAULT_BACKEND_PRECEDENCE

    assert KernelBackend.SAGE not in _DEFAULT_BACKEND_PRECEDENCE


def test_sage_wrapper_is_exported() -> None:
    assert callable(fused_attention_sage)
    assert "fused_attention_sage" in attention_ops.__all__
    assert "recommend_attention_backend" in attention_ops.__all__


@pytest.mark.parametrize(
    ("seq_q", "seq_kv", "gqa", "expected_backend", "expected_bucket"),
    [
        pytest.param(1, 1024, False, "triton", "decode", id="decode"),
        pytest.param(1, 8192, False, "triton", "decode", id="long-decode"),
        pytest.param(512, 512, False, "triton", "short", id="short-prefill"),
        pytest.param(512, 512, True, "triton", "short", id="gqa-short"),
        pytest.param(2048, 2048, False, "tilelang", "medium", id="medium-prefill"),
        pytest.param(2048, 2048, True, "triton", "medium", id="gqa-medium"),
        pytest.param(8192, 8192, False, "tilelang", "long", id="long-prefill"),
        pytest.param(8192, 8192, True, "triton", "long", id="gqa-long"),
    ],
)
def test_recommend_attention_backend_shapes(
    seq_q: int,
    seq_kv: int,
    gqa: bool,
    expected_backend: str,
    expected_bucket: str,
) -> None:
    decision = recommend_attention_backend(
        seq_q=seq_q,
        seq_kv=seq_kv,
        head_dim=64,
        dtype="float16",
        causal=True,
        gqa=gqa,
    )
    assert isinstance(decision, AttentionRoutingDecision)
    assert decision.backend == expected_backend
    assert decision.bucket == expected_bucket
    assert decision.reason
    assert decision.to_dict() == {
        "backend": expected_backend,
        "reason": decision.reason,
        "bucket": expected_bucket,
    }


def test_recommend_attention_backend_never_returns_sage() -> None:
    for seq_q in (1, 64, 512, 2048, 8192):
        for seq_kv in (1, 128, 512, 2048, 8192):
            for gqa in (False, True):
                decision = recommend_attention_backend(
                    seq_q=seq_q,
                    seq_kv=seq_kv,
                    head_dim=64,
                    dtype="float16",
                    causal=True,
                    gqa=gqa,
                )
                assert decision.backend != "sage"


def test_recommend_attention_backend_falls_back_when_tilelang_missing() -> None:
    decision = recommend_attention_backend(
        seq_q=8192,
        seq_kv=8192,
        head_dim=64,
        gqa=False,
        available_backends=["triton", "sdpa"],
    )
    assert decision.backend == "sdpa"
    assert "unavailable" in decision.reason
    assert decision.bucket == "long"


def test_recommend_attention_backend_falls_back_when_triton_missing() -> None:
    decision = recommend_attention_backend(
        seq_q=1,
        seq_kv=1024,
        head_dim=64,
        gqa=False,
        available_backends=["tilelang", "sdpa"],
    )
    assert decision.backend == "sdpa"
    assert decision.bucket == "decode"


def test_recommend_attention_backend_keeps_preferred_when_available() -> None:
    decision = recommend_attention_backend(
        seq_q=2048,
        seq_kv=2048,
        head_dim=64,
        gqa=False,
        available_backends=["tilelang", "triton", "sdpa"],
    )
    assert decision.backend == "tilelang"


@requires_cuda
def test_sage_wrapper_runs_on_cuda() -> None:
    torch.manual_seed(3)
    q = torch.randn(1, 2, 8, 64, device="cuda", dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    output = fused_attention_sage(q, k, v, causal=False)
    reference = torch.nn.functional.scaled_dot_product_attention(
        q,
        k,
        v,
        is_causal=False,
    )
    assert output.shape == q.shape
    assert torch.allclose(output.float(), reference.float(), atol=5e-2, rtol=5e-2)
