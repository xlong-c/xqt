from __future__ import annotations

import pytest
import torch

from xqt.core.errors import XQTBackendError
from xqt.kernels.ops._impl.engines.triton import (
    get_triton_kernel_spec,
    run_triton_kernel,
)
from xqt.kernels.ops._impl.triton.attention import (
    TritonAttentionSchedule,
    fused_attention_forward_reference,
    fused_attention_forward_triton,
    resolve_triton_attention_schedule,
)


requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for Triton attention tests",
)


def test_triton_attention_registry_declares_execution_contract() -> None:
    spec = get_triton_kernel_spec("attention")
    metadata = spec.metadata

    assert spec.kernel is fused_attention_forward_triton
    assert spec.reference is fused_attention_forward_reference
    assert metadata["supported_dtypes"] == ["float16", "bfloat16"]
    assert metadata["supported_head_dims"] == [16, 32, 64, 128]
    assert metadata["causal_semantics"] == "lower-right when seq_kv >= seq_q"
    assert metadata["supports_dropout"] is False
    assert metadata["supports_backward"] is False
    assert metadata["requires_contiguous"] is True


@pytest.mark.parametrize(
    ("seq_q", "seq_kv", "causal"),
    [
        pytest.param(8, 8, False, id="square-noncausal"),
        pytest.param(8, 8, True, id="square-causal"),
        pytest.param(1, 8, True, id="decode-lower-right-causal"),
    ],
)
def test_triton_attention_cpu_registry_fallback_matches_reference(
    seq_q: int,
    seq_kv: int,
    causal: bool,
) -> None:
    torch.manual_seed(3)
    q = torch.randn(1, 2, seq_q, 16)
    k = torch.randn(1, 2, seq_kv, 16)
    v = torch.randn(1, 2, seq_kv, 16)

    output = run_triton_kernel(
        "attention",
        q,
        k,
        v,
        causal=causal,
        dropout_p=0.0,
    )
    reference = fused_attention_forward_reference(
        q,
        k,
        v,
        causal=causal,
        dropout_p=0.0,
    )

    assert torch.equal(output, reference)


def test_triton_attention_schedule_preserves_explicit_overrides() -> None:
    schedule = resolve_triton_attention_schedule(
        batch=1,
        heads=8,
        seq_q=256,
        seq_kv=256,
        head_dim=64,
        input_dtype="float16",
        causal=False,
        block_m=32,
        block_n=128,
        num_warps=8,
        num_stages=3,
        target_arch="sm_89",
    )

    assert isinstance(schedule, TritonAttentionSchedule)
    assert schedule.to_dict() == {
        "block_m": 32,
        "block_n": 128,
        "num_warps": 8,
        "num_stages": 3,
        "target_arch": "sm_89",
        "preset": "default",
    }


@pytest.mark.parametrize(
    ("seq_q", "seq_kv", "input_dtype", "causal", "expected", "preset"),
    [
        pytest.param(
            1,
            1024,
            "float16",
            True,
            (16, 64, 4, 2),
            "sm89_fp16_decode_q1_kv1024_d64",
            id="fp16-decode",
        ),
        pytest.param(
            1,
            1024,
            "bfloat16",
            True,
            (16, 128, 4, 2),
            "sm89_bf16_decode_q1_kv1024_d64",
            id="bf16-decode",
        ),
    ],
)
def test_triton_attention_sm89_presets_are_exact_signature_scoped(
    seq_q: int,
    seq_kv: int,
    input_dtype: str,
    causal: bool,
    expected: tuple[int, int, int, int],
    preset: str,
) -> None:
    schedule = resolve_triton_attention_schedule(
        batch=1,
        heads=8,
        seq_q=seq_q,
        seq_kv=seq_kv,
        head_dim=64,
        input_dtype=input_dtype,
        causal=causal,
        target_arch="sm_89",
    )

    assert (
        schedule.block_m,
        schedule.block_n,
        schedule.num_warps,
        schedule.num_stages,
    ) == expected
    assert schedule.preset == preset


@pytest.mark.parametrize(
    ("target_arch", "seq_q", "seq_kv", "input_dtype", "causal"),
    [
        pytest.param("sm_90", 1, 1024, "float16", True, id="other-sm"),
        pytest.param("sm_89", 2, 1024, "float16", True, id="other-q"),
        pytest.param("sm_89", 1, 512, "float16", True, id="other-kv"),
        pytest.param("sm_89", 1, 1024, "float16", False, id="noncausal"),
        pytest.param(
            "sm_89",
            1024,
            1024,
            "float16",
            False,
            id="long-prefill-audit-keeps-default",
        ),
    ],
)
def test_triton_attention_schedule_keeps_default_outside_exact_presets(
    target_arch: str,
    seq_q: int,
    seq_kv: int,
    input_dtype: str,
    causal: bool,
) -> None:
    schedule = resolve_triton_attention_schedule(
        batch=1,
        heads=8,
        seq_q=seq_q,
        seq_kv=seq_kv,
        head_dim=64,
        input_dtype=input_dtype,
        causal=causal,
        target_arch=target_arch,
    )

    assert (
        schedule.block_m,
        schedule.block_n,
        schedule.num_warps,
        schedule.num_stages,
    ) == (64, 64, 4, 2)
    assert schedule.preset == "default"


@requires_cuda
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
def test_triton_attention_cuda_matches_reference(
    dtype: torch.dtype,
    atol: float,
    rtol: float,
    seq_q: int,
    seq_kv: int,
    causal: bool,
) -> None:
    torch.manual_seed(5)
    q = torch.randn(1, 2, seq_q, 32, device="cuda", dtype=dtype)
    k = torch.randn(1, 2, seq_kv, 32, device="cuda", dtype=dtype)
    v = torch.randn(1, 2, seq_kv, 32, device="cuda", dtype=dtype)

    output = fused_attention_forward_triton(q, k, v, causal=causal)
    reference = fused_attention_forward_reference(q, k, v, causal=causal)

    assert output.shape == q.shape
    assert output.dtype == dtype
    assert torch.allclose(output.float(), reference.float(), atol=atol, rtol=rtol)


@requires_cuda
def test_triton_attention_cuda_forwards_explicit_scale() -> None:
    torch.manual_seed(7)
    q = torch.randn(1, 2, 64, 32, device="cuda", dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    output = fused_attention_forward_triton(q, k, v, scale=0.5)
    reference = fused_attention_forward_reference(q, k, v, scale=0.5)

    assert torch.allclose(output.float(), reference.float(), atol=1e-2, rtol=1e-2)


@requires_cuda
def test_triton_attention_rejects_mixed_half_dtypes() -> None:
    q = torch.randn(1, 1, 8, 16, device="cuda", dtype=torch.float16)
    k = torch.randn(1, 1, 8, 16, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(1, 1, 8, 16, device="cuda", dtype=torch.bfloat16)

    with pytest.raises(
        XQTBackendError,
        match="requires matching float16 or bfloat16 tensors",
    ):
        fused_attention_forward_triton(q, k, v)


@requires_cuda
def test_triton_attention_rejects_noncontiguous_bhsd() -> None:
    q = torch.randn(1, 1, 8, 16, device="cuda", dtype=torch.float16).transpose(
        2,
        3,
    )
    k = torch.randn(1, 1, 16, 8, device="cuda", dtype=torch.float16)
    v = torch.randn_like(k)

    with pytest.raises(XQTBackendError, match="requires contiguous BHSD"):
        fused_attention_forward_triton(q, k, v)


@requires_cuda
def test_triton_attention_rejects_dropout() -> None:
    q = torch.randn(1, 1, 8, 16, device="cuda", dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    with pytest.raises(XQTBackendError, match="does not support dropout_p"):
        fused_attention_forward_triton(q, k, v, dropout_p=0.1)


@requires_cuda
def test_triton_attention_rejects_unsupported_head_dim() -> None:
    q = torch.randn(1, 1, 8, 24, device="cuda", dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    with pytest.raises(XQTBackendError, match="head_dim must be one of"):
        fused_attention_forward_triton(q, k, v)


@requires_cuda
def test_triton_attention_rejects_autograd_inputs() -> None:
    q = torch.randn(
        1,
        1,
        8,
        16,
        device="cuda",
        dtype=torch.float16,
        requires_grad=True,
    )
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    with pytest.raises(XQTBackendError, match="inference-only"):
        fused_attention_forward_triton(q, k, v)
