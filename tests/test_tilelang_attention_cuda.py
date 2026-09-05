from __future__ import annotations

import pytest
import torch

import xqt.kernels.nn as xqt_nn
from xqt.core.errors import XQTBackendError
from xqt.kernels.ops._impl.engines.tilelang import get_tilelang_kernel_spec
from xqt.kernels.ops._impl.tilelang.attention import (
    fused_attention_forward_reference,
    fused_attention_forward_tilelang,
)
from xqt.kernels.ops._impl.tilelang._common import tilelang_runtime_usable


requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for TileLang attention CUDA test",
)

requires_tilelang = pytest.mark.skipif(
    not tilelang_runtime_usable(),
    reason="a runtime-compatible TileLang adapter is required for TileLang attention CUDA test",
)


def test_tilelang_attention_metadata_declares_bf16_constraints() -> None:
    metadata = get_tilelang_kernel_spec("attention").metadata

    assert metadata["supported_dtypes"] == ["float16", "bfloat16"]
    assert metadata["bfloat16_head_dim_multiple"] == 16
    assert metadata["design"]["production_status"] == "runtime_kernel"


@requires_cuda
@requires_tilelang
@pytest.mark.parametrize(
    ("dtype", "atol", "rtol"),
    [
        pytest.param(torch.float16, 1e-2, 1e-2, id="fp16"),
        pytest.param(torch.bfloat16, 2e-2, 2e-2, id="bf16"),
    ],
)
@pytest.mark.parametrize(
    ("seq_q", "seq_kv", "causal"),
    [
        pytest.param(64, 64, False, id="square-noncausal"),
        pytest.param(64, 64, True, id="square-causal"),
        pytest.param(1, 64, True, id="decode-lower-right-causal"),
    ],
)
def test_tilelang_attention_cuda_kernel_matches_reference(
    dtype: torch.dtype,
    atol: float,
    rtol: float,
    seq_q: int,
    seq_kv: int,
    causal: bool,
) -> None:
    torch.manual_seed(0)
    q = torch.randn(1, 1, seq_q, 32, device="cuda", dtype=dtype)
    k = torch.randn(1, 1, seq_kv, 32, device="cuda", dtype=dtype)
    v = torch.randn(1, 1, seq_kv, 32, device="cuda", dtype=dtype)

    output = fused_attention_forward_tilelang(
        q,
        k,
        v,
        causal=causal,
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
        causal=causal,
        dropout_p=0.0,
    )

    assert output.shape == reference.shape
    assert output.dtype == dtype
    assert torch.allclose(output.float(), reference.float(), atol=atol, rtol=rtol)


@requires_cuda
def test_tilelang_attention_cuda_rejects_mixed_fp16_bf16_inputs() -> None:
    q = torch.randn(1, 1, 8, 16, device="cuda", dtype=torch.float16)
    k = torch.randn(1, 1, 8, 16, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(1, 1, 8, 16, device="cuda", dtype=torch.bfloat16)

    with pytest.raises(
        XQTBackendError,
        match="requires matching float16 or bfloat16 tensors",
    ):
        fused_attention_forward_tilelang(q, k, v)


@requires_cuda
def test_tilelang_attention_cuda_rejects_float32_inputs() -> None:
    q = torch.randn(1, 1, 8, 16, device="cuda", dtype=torch.float32)
    k = torch.randn(1, 1, 8, 16, device="cuda", dtype=torch.float32)
    v = torch.randn(1, 1, 8, 16, device="cuda", dtype=torch.float32)

    with pytest.raises(
        XQTBackendError,
        match="requires matching float16 or bfloat16 tensors",
    ):
        fused_attention_forward_tilelang(q, k, v)


@requires_cuda
def test_tilelang_attention_cuda_rejects_unaligned_bf16_head_dim() -> None:
    q = torch.randn(1, 1, 8, 8, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 1, 8, 8, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(1, 1, 8, 8, device="cuda", dtype=torch.bfloat16)

    with pytest.raises(
        XQTBackendError,
        match="bfloat16 attention requires head_dim divisible by 16",
    ):
        fused_attention_forward_tilelang(q, k, v)


@requires_cuda
def test_xqt_attention_facade_preserves_bf16_for_tilelang(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_dtypes: list[tuple[torch.dtype, torch.dtype, torch.dtype]] = []

    def fake_tilelang_attention(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        causal: bool = False,
        dropout_p: float = 0.0,
        **_: object,
    ) -> torch.Tensor:
        observed_dtypes.append((q.dtype, k.dtype, v.dtype))
        return fused_attention_forward_reference(
            q,
            k,
            v,
            causal=causal,
            dropout_p=dropout_p,
        )

    from xqt.kernels.ops._impl.tilelang import attention as tilelang_attention
    from xqt.kernels.selector import clear_cache

    clear_cache()
    monkeypatch.setattr(
        tilelang_attention,
        "fused_attention_forward_tilelang",
        fake_tilelang_attention,
    )
    attention = xqt_nn.Attention(64, heads=4, engine="tilelang").to(
        device="cuda",
        dtype=torch.bfloat16,
    )
    x = torch.randn(1, 16, 64, device="cuda", dtype=torch.bfloat16)

    output = attention(x)

    assert output.dtype == torch.bfloat16
    assert observed_dtypes == [
        (torch.bfloat16, torch.bfloat16, torch.bfloat16)
    ]
    assert attention.runtime_fallback is None


def test_resolve_tilelang_attention_schedule() -> None:
    from xqt.kernels.ops._impl.tilelang.attention import resolve_tilelang_attention_schedule
    from xqt.kernels.ops._impl.tilelang.tuning_cache import (
        TimingCacheKey,
        get_tilelang_timing_cache,
    )

    # 1. Test decode adaptive heuristic (seq_q <= 4)
    q_decode = torch.randn(1, 8, 1, 64, dtype=torch.float16)
    k_decode = torch.randn(1, 8, 128, 64, dtype=torch.float16)
    sched_dec = resolve_tilelang_attention_schedule(q_decode, k_decode, causal=False)
    assert sched_dec.block_m == 16
    assert sched_dec.preset == "decode_adaptive"

    # 2. Test prefill default (seq_q > 4)
    q_prefill = torch.randn(1, 8, 64, 64, dtype=torch.float16)
    k_prefill = torch.randn(1, 8, 64, 64, dtype=torch.float16)
    sched_pref = resolve_tilelang_attention_schedule(q_prefill, k_prefill, causal=True)
    assert sched_pref.block_m == 64
    assert sched_pref.preset == "prefill_default"

    # 3. Test explicit overrides take precedence
    sched_override = resolve_tilelang_attention_schedule(
        q_decode, k_decode, block_m=32, block_n=128
    )
    assert sched_override.block_m == 32
    assert sched_override.block_n == 128

    # 4. Test TimingCache lookup hit
    cache = get_tilelang_timing_cache()
    custom_key = TimingCacheKey(
        op_type="attention",
        arch="cuda",
        dtype="float16",
        shape=(1, 8, 32, 32, 64),
        extra="causal=False",
    )
    cache.record(
        custom_key,
        {"block_m": 32, "block_n": 32, "threads": 256, "num_stages": 3},
        preset_name="test_tuned_attn",
        persist=False,
    )
    q_custom = torch.randn(1, 8, 32, 64, dtype=torch.float16)
    k_custom = torch.randn(1, 8, 32, 64, dtype=torch.float16)
    sched_cached = resolve_tilelang_attention_schedule(
        q_custom, k_custom, causal=False, target_arch="cuda"
    )
    assert sched_cached.preset == "test_tuned_attn"
    assert sched_cached.block_m == 32
    assert sched_cached.threads == 256

    # 5. Test candidate attention schedule generation
    candidates = cache.get_candidate_attention_schedules(seq_q=1, seq_kv=128, head_dim=64)
    assert len(candidates) > 0
    assert any(c["block_m"] == 16 for c in candidates)

