"""CUDA tests for the KvScaleAttention TileLang KV-int8 fused attention path."""

from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any

import pytest
import torch

from xqt.operator_opt.kernels.tilelang._common import (
    tilelang_runtime_unavailability_reason,
    tilelang_runtime_usable,
)
from xqt.operator_opt.kernels.tilelang import kv_int8_attention as kv_int8_kernel
from xqt.operator_opt.kernels.tilelang.kv_int8_attention import (
    KV_INT8_PACKED_QKV_ATTENTION_KERNEL_NAME,
    KV_INT8_PACKED_QKV_QUANTIZE_LAYOUT_KERNEL_NAME,
    KV_INT8_PROJECTION_IO_KERNEL_NAME,
    KV_INT8_QUANTIZE_LAYOUT_KERNEL_NAME,
    fused_kv_int8_attention_forward_tilelang,
    fused_kv_int8_attention_packed_qkv_forward_tilelang,
    fused_kv_int8_attention_projection_forward_tilelang,
    quantize_kv_int8_layout_tilelang,
    quantize_packed_qkv_int8_layout_tilelang,
)
from xqt.contracts.runtime_quant import (
    RuntimeQuantContract,
    build_runtime_quant_contract,
)
from xqt.core.errors import XQTBackendError
from xqt.runtime.modules import KvScaleAttention


def _kv_contract() -> RuntimeQuantContract:
    """与 tests/xqt/runtime/test_runtime_features.py 一致的 KV contract."""

    return build_runtime_quant_contract(
        quant_spec={
            "weight_dtype": "int8",
            "weight_granularity": "per_tensor",
            "activation_mode": "none",
        },
        storage_layout="kv_scale",
        required_kernels=("torch_sdpa_kv_scale_reference",),
        global_shape=(8, 8),
        kv_cache_dtype="int8",
    )

_CUDA_AVAILABLE = torch.cuda.is_available()
_TILELANG_USABLE = _CUDA_AVAILABLE and tilelang_runtime_usable()

requires_cuda = pytest.mark.skipif(
    not _CUDA_AVAILABLE,
    reason="CUDA is not available",
)
requires_cuda_tilelang = pytest.mark.skipif(
    not _TILELANG_USABLE,
    reason=tilelang_runtime_unavailability_reason() or "CUDA is not available",
)

_DIM = 128
_HEADS = 4
_BATCH = 2


def _build_cuda_entity(
    *,
    seq: int,
    causal: bool,
    dtype: torch.dtype = torch.float16,
) -> tuple[KvScaleAttention, torch.Tensor]:
    """构造 CUDA 实体并用投影输出校准 per-tensor K/V scale."""

    torch.manual_seed(0)
    entity = KvScaleAttention(
        _DIM,
        heads=_HEADS,
        k_scale=1.0,
        v_scale=1.0,
        causal=causal,
        layer_path="model.layers.0.self_attn",
    )
    entity.to(device="cuda", dtype=dtype)
    # scale buffer 保持 fp32, 与 CPU reference 语义一致.
    entity.k_scale = entity.k_scale.float()
    entity.v_scale = entity.v_scale.float()
    entity.attn_k_scale = entity.attn_k_scale.float()
    entity.attn_v_scale = entity.attn_v_scale.float()
    x = torch.randn(_BATCH, seq, _DIM, device="cuda", dtype=dtype) * 0.5
    with torch.no_grad():
        _, k, v = entity._split_qkv(entity.qkv(x))
        k_scale = float(k.abs().max().item()) / 127.0
        v_scale = float(v.abs().max().item()) / 127.0
    entity.k_scale.fill_(k_scale)
    entity.attn_k_scale.fill_(k_scale)
    entity.v_scale.fill_(v_scale)
    entity.attn_v_scale.fill_(v_scale)
    return entity, x


def _max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().max().item())


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(
        torch.nn.functional.cosine_similarity(
            a.float().flatten(),
            b.float().flatten(),
            dim=0,
        ).item()
    )


@requires_cuda_tilelang
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("seq", [128, 256])
def test_fused_matches_reference_on_cuda(seq: int, causal: bool) -> None:
    """fused 与 reference 消费同一份 int8 K/V, 输出应在 fp16 舍入级别一致."""

    reference_entity, x = _build_cuda_entity(seq=seq, causal=causal)
    fused_entity = copy.deepcopy(reference_entity)
    fused_entity.preferred_kernel = "tilelang"

    with torch.no_grad():
        out_reference = reference_entity(x)
        out_fused = fused_entity(x)

    assert out_fused.shape == out_reference.shape
    max_abs = _max_abs_diff(out_fused, out_reference)
    cosine = _cosine(out_fused, out_reference)
    # 两条路径共享同一 int8 量化结果, 差异仅来自 SDPA 与 TileLang kernel 的
    # fp16/fp32 累加顺序, 实测 max_abs 约 5e-4 量级.
    assert max_abs < 1e-2
    assert cosine > 0.99999

    report = fused_entity.report()
    assert report["selected_kernel"] == KV_INT8_PACKED_QKV_ATTENTION_KERNEL_NAME
    assert report["projection_mode"] == "packed_qkv"
    assert report["cuda_fused_verified"] is True
    assert report["fallback_reason"] is None
    assert report["preferred_kernel"] == "tilelang"


@requires_cuda_tilelang
def test_auto_kernel_selects_fused_when_available() -> None:
    entity, x = _build_cuda_entity(seq=128, causal=True)
    entity.preferred_kernel = "auto"
    with torch.no_grad():
        entity(x)
    report = entity.report()
    assert report["selected_kernel"] == KV_INT8_PACKED_QKV_ATTENTION_KERNEL_NAME
    assert report["cuda_fused_verified"] is True


@requires_cuda
def test_fallback_to_reference_on_non_fp16_cuda() -> None:
    entity, x = _build_cuda_entity(seq=128, causal=False, dtype=torch.float32)
    entity.preferred_kernel = "tilelang"
    entity.attention_fastpath = "graph"
    with torch.no_grad():
        output = entity(x)
    assert output.shape == x.shape
    report = entity.report()
    assert report["selected_kernel"] == "torch_sdpa_kv_scale_reference"
    assert report["cuda_fused_verified"] is False
    assert report["fallback_reason"] == "dtype_not_fp16"
    assert report["cuda_graph"]["state"] == "fallback_eager"
    assert report["cuda_graph"]["reason"] == "dtype_not_fp16"
    assert report["cuda_graph"]["cache_size"] == 0


@requires_cuda
def test_graph_mode_with_dropout_falls_back_without_capture() -> None:
    entity = KvScaleAttention(
        _DIM,
        heads=_HEADS,
        k_scale=0.01,
        v_scale=0.02,
        dropout=0.1,
        preferred_kernel="tilelang",
        attention_fastpath="graph",
    ).to(device="cuda", dtype=torch.float16)
    x = torch.randn(
        _BATCH,
        32,
        _DIM,
        device="cuda",
        dtype=torch.float16,
    )

    with torch.no_grad():
        output = entity(x)
    report = entity.report()

    assert output.shape == x.shape
    assert report["selected_fastpath"] == "torch_sdpa_reference_fallback"
    assert report["fallback_reason"] == "dropout_unsupported"
    assert report["cuda_graph"] == {
        "state": "fallback_eager",
        "reason": "dropout_unsupported",
        "cache_size": 0,
        "output_storage": None,
    }


def test_fallback_to_reference_on_cpu() -> None:
    """无 CUDA 时 preferred_kernel="tilelang" 必须诚实 fallback 而非报错."""

    torch.manual_seed(0)
    entity = KvScaleAttention(
        8,
        heads=2,
        k_scale=1.0,
        v_scale=1.0,
        preferred_kernel="tilelang",
    )
    x = torch.randn(2, 5, 8) * 0.1
    output = entity(x)
    assert output.shape == (2, 5, 8)
    report = entity.report()
    assert report["selected_kernel"] == "torch_sdpa_kv_scale_reference"
    assert report["cuda_fused_verified"] is False
    assert report["fallback_reason"] == "cuda_unavailable"


def test_reference_preferred_kernel_keeps_default_report() -> None:
    entity = KvScaleAttention(
        8,
        heads=2,
        k_scale=1.0,
        v_scale=1.0,
        contract=_kv_contract(),
    )
    report = entity.report()
    assert report["preferred_kernel"] == "reference"
    assert report["selected_kernel"] == "torch_sdpa_kv_scale_reference"
    assert report["cuda_fused_verified"] is False
    assert report["fallback_reason"] is None


def test_invalid_preferred_kernel_rejected() -> None:
    with pytest.raises(ValueError, match="preferred_kernel"):
        KvScaleAttention(8, heads=2, k_scale=1.0, v_scale=1.0, preferred_kernel="magic")


def test_invalid_attention_fastpath_rejected() -> None:
    with pytest.raises(ValueError, match="attention_fastpath"):
        KvScaleAttention(
            8,
            heads=2,
            k_scale=1.0,
            v_scale=1.0,
            attention_fastpath="magic",
        )


def test_invalid_cuda_graph_warmup_rejected() -> None:
    with pytest.raises(ValueError, match="cuda_graph_warmup"):
        KvScaleAttention(
            8,
            heads=2,
            k_scale=1.0,
            v_scale=1.0,
            cuda_graph_warmup=-1,
        )


def test_graph_mode_on_cpu_falls_back_without_capture() -> None:
    entity = KvScaleAttention(
        8,
        heads=2,
        k_scale=1.0,
        v_scale=1.0,
        preferred_kernel="tilelang",
        attention_fastpath="graph",
    )
    x = torch.randn(2, 5, 8) * 0.1

    output = entity(x)
    report = entity.report()

    assert output.shape == x.shape
    assert report["selected_fastpath"] == "torch_sdpa_reference_fallback"
    assert report["fallback_reason"] == "cuda_unavailable"
    assert report["cuda_graph"] == {
        "state": "fallback_eager",
        "reason": "cuda_unavailable",
        "cache_size": 0,
        "output_storage": None,
    }


def test_cuda_graph_capture_error_falls_back_to_eager_packed_forward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entity = KvScaleAttention(
        8,
        heads=2,
        k_scale=1.0,
        v_scale=1.0,
        preferred_kernel="tilelang",
        attention_fastpath="graph",
    )
    x = torch.randn(2, 5, 8) * 0.1

    def capture_fails(_x: torch.Tensor) -> dict[str, Any]:
        raise RuntimeError("synthetic capture failure")

    def fake_packed_attention(qkv: torch.Tensor) -> torch.Tensor:
        return qkv[..., : entity.inner_dim]

    monkeypatch.setattr(entity, "_fused_unavailability_reason", lambda _x: None)
    monkeypatch.setattr(entity, "_capture_graph", capture_fails)
    monkeypatch.setattr(entity, "_run_packed_tilelang_attention", fake_packed_attention)

    with torch.no_grad():
        expected = entity.out_proj(entity.qkv(x)[..., : entity.inner_dim])
        output = entity(x)
    report = entity.report()

    assert torch.equal(output, expected)
    assert report["selected_fastpath"] == "packed_qkv_tilelang_eager"
    assert report["fallback_reason"] is None
    assert report["cuda_graph"]["state"] == "fallback_eager"
    assert "synthetic capture failure" in report["cuda_graph"]["reason"]
    assert report["cuda_graph"]["cache_size"] == 0


@pytest.mark.parametrize("block_size", [0, -1])
def test_invalid_fused_quant_block_size_rejected(block_size: int) -> None:
    with pytest.raises(ValueError, match="fused_quant_block_size"):
        KvScaleAttention(
            8,
            heads=2,
            k_scale=1.0,
            v_scale=1.0,
            fused_quant_block_size=block_size,
        )


@pytest.mark.parametrize("qkv_bias", [False, True])
def test_packed_qkv_is_the_only_projection_parameter_source(
    qkv_bias: bool,
) -> None:
    entity = KvScaleAttention(
        32,
        heads=2,
        k_scale=0.01,
        v_scale=0.02,
        qkv_bias=qkv_bias,
    )

    assert entity.qkv.in_features == 32
    assert entity.qkv.out_features == 96
    assert (entity.qkv.bias is not None) is qkv_bias
    assert not hasattr(entity, "q_proj")
    assert not hasattr(entity, "k_proj")
    assert not hasattr(entity, "v_proj")
    state_keys = set(entity.state_dict())
    assert "qkv.weight" in state_keys
    assert ("qkv.bias" in state_keys) is qkv_bias
    assert not any(
        key.startswith(("q_proj.", "k_proj.", "v_proj."))
        for key in state_keys
    )


@requires_cuda_tilelang
@pytest.mark.parametrize("qmax", [63, 127])
def test_tilelang_kv_quantize_layout_matches_eager_codes(qmax: int) -> None:
    torch.manual_seed(11 + qmax)
    batch, heads, seq, head_dim = 2, 4, 100, 32
    inner_dim = heads * head_dim
    k = torch.randn(
        batch,
        seq,
        inner_dim,
        device="cuda",
        dtype=torch.float16,
    ) * 0.5
    v = torch.randn_like(k)
    k_scale = torch.tensor(0.017, device="cuda", dtype=torch.float32)
    v_scale = torch.tensor(0.021, device="cuda", dtype=torch.float32)

    k_int8, v_int8 = quantize_kv_int8_layout_tilelang(
        k,
        v,
        k_scale,
        v_scale,
        heads=heads,
        head_dim=head_dim,
        qmax=qmax,
    )
    k_reference = (
        torch.round(k / k_scale)
        .clamp(-qmax, qmax)
        .to(torch.int8)
        .reshape(batch, seq, heads, head_dim)
        .permute(0, 2, 1, 3)
        .contiguous()
    )
    v_reference = (
        torch.round(v / v_scale)
        .clamp(-qmax, qmax)
        .to(torch.int8)
        .reshape(batch, seq, heads, head_dim)
        .permute(0, 2, 1, 3)
        .contiguous()
    )
    torch.cuda.synchronize()

    assert torch.equal(k_int8, k_reference)
    assert torch.equal(v_int8, v_reference)


@requires_cuda_tilelang
@pytest.mark.parametrize("qmax", [63, 127])
def test_tilelang_packed_qkv_quantize_layout_matches_eager_codes(
    qmax: int,
) -> None:
    torch.manual_seed(101 + qmax)
    batch, heads, seq, head_dim = 2, 4, 100, 32
    inner_dim = heads * head_dim
    qkv = torch.randn(
        batch,
        seq,
        3 * inner_dim,
        device="cuda",
        dtype=torch.float16,
    ) * 0.5
    k_scale = torch.tensor(0.017, device="cuda", dtype=torch.float32)
    v_scale = torch.tensor(0.021, device="cuda", dtype=torch.float32)

    k_int8, v_int8 = quantize_packed_qkv_int8_layout_tilelang(
        qkv,
        k_scale,
        v_scale,
        heads=heads,
        head_dim=head_dim,
        qmax=qmax,
    )
    _, k, v = torch.split(qkv, inner_dim, dim=-1)
    k_reference = (
        torch.round(k / k_scale)
        .clamp(-qmax, qmax)
        .to(torch.int8)
        .reshape(batch, seq, heads, head_dim)
        .permute(0, 2, 1, 3)
        .contiguous()
    )
    v_reference = (
        torch.round(v / v_scale)
        .clamp(-qmax, qmax)
        .to(torch.int8)
        .reshape(batch, seq, heads, head_dim)
        .permute(0, 2, 1, 3)
        .contiguous()
    )
    torch.cuda.synchronize()

    assert torch.equal(k_int8, k_reference)
    assert torch.equal(v_int8, v_reference)


@requires_cuda
def test_tilelang_kv_quantize_layout_rejects_noncontiguous_projection() -> None:
    batch, heads, seq, head_dim = 2, 4, 5, 16
    inner_dim = heads * head_dim
    k = torch.randn(
        batch,
        inner_dim,
        seq,
        device="cuda",
        dtype=torch.float16,
    ).transpose(1, 2)
    v = torch.randn_like(k)
    k_scale = torch.tensor(0.017, device="cuda", dtype=torch.float32)
    v_scale = torch.tensor(0.021, device="cuda", dtype=torch.float32)

    assert not k.is_contiguous()
    assert not v.is_contiguous()
    with pytest.raises(
        XQTBackendError,
        match="requires contiguous K/V projections",
    ):
        quantize_kv_int8_layout_tilelang(
            k,
            v,
            k_scale,
            v_scale,
            heads=heads,
            head_dim=head_dim,
        )


@requires_cuda
def test_tilelang_packed_qkv_rejects_noncontiguous_projection() -> None:
    batch, heads, seq, head_dim = 2, 4, 5, 16
    inner_dim = heads * head_dim
    qkv = torch.randn(
        batch,
        3 * inner_dim,
        seq,
        device="cuda",
        dtype=torch.float16,
    ).transpose(1, 2)
    k_scale = torch.tensor(0.017, device="cuda", dtype=torch.float32)
    v_scale = torch.tensor(0.021, device="cuda", dtype=torch.float32)

    assert not qkv.is_contiguous()
    with pytest.raises(
        XQTBackendError,
        match="requires a contiguous projection tensor",
    ):
        quantize_packed_qkv_int8_layout_tilelang(
            qkv,
            k_scale,
            v_scale,
            heads=heads,
            head_dim=head_dim,
        )


@requires_cuda_tilelang
@pytest.mark.parametrize("causal", [False, True])
def test_projection_io_attention_matches_bhsd_kernel_bitwise(causal: bool) -> None:
    torch.manual_seed(9100 + int(causal))
    batch, heads, seq, head_dim = 2, 4, 100, 32
    inner_dim = heads * head_dim
    q = torch.randn(
        batch,
        seq,
        inner_dim,
        device="cuda",
        dtype=torch.float16,
    )
    k_int8 = torch.randint(
        -127,
        128,
        (batch, heads, seq, head_dim),
        device="cuda",
        dtype=torch.int8,
    )
    v_int8 = torch.randint(
        -127,
        128,
        (batch, heads, seq, head_dim),
        device="cuda",
        dtype=torch.int8,
    )
    k_scale = torch.tensor(0.017, device="cuda", dtype=torch.float32)
    v_scale = torch.tensor(0.021, device="cuda", dtype=torch.float32)
    q_bhsd = (
        q.reshape(batch, seq, heads, head_dim)
        .permute(0, 2, 1, 3)
        .contiguous()
    )

    bhsd_output = fused_kv_int8_attention_forward_tilelang(
        q_bhsd,
        k_int8,
        v_int8,
        k_scale,
        v_scale,
        causal=causal,
    )
    projection_output = fused_kv_int8_attention_projection_forward_tilelang(
        q,
        k_int8,
        v_int8,
        k_scale,
        v_scale,
        heads=heads,
        head_dim=head_dim,
        causal=causal,
    )
    bsi_output = (
        bhsd_output.permute(0, 2, 1, 3)
        .contiguous()
        .reshape(batch, seq, inner_dim)
    )
    torch.cuda.synchronize()

    assert torch.equal(projection_output, bsi_output)


@requires_cuda_tilelang
@pytest.mark.parametrize("causal", [False, True])
def test_packed_qkv_attention_matches_projection_io_bitwise(causal: bool) -> None:
    torch.manual_seed(9200 + int(causal))
    batch, heads, seq, head_dim = 2, 4, 100, 32
    inner_dim = heads * head_dim
    qkv = torch.randn(
        batch,
        seq,
        3 * inner_dim,
        device="cuda",
        dtype=torch.float16,
    )
    q = qkv[..., :inner_dim].contiguous()
    k_int8 = torch.randint(
        -127,
        128,
        (batch, heads, seq, head_dim),
        device="cuda",
        dtype=torch.int8,
    )
    v_int8 = torch.randint(
        -127,
        128,
        (batch, heads, seq, head_dim),
        device="cuda",
        dtype=torch.int8,
    )
    k_scale = torch.tensor(0.017, device="cuda", dtype=torch.float32)
    v_scale = torch.tensor(0.021, device="cuda", dtype=torch.float32)

    projection_output = fused_kv_int8_attention_projection_forward_tilelang(
        q,
        k_int8,
        v_int8,
        k_scale,
        v_scale,
        heads=heads,
        head_dim=head_dim,
        causal=causal,
    )
    packed_output = fused_kv_int8_attention_packed_qkv_forward_tilelang(
        qkv,
        k_int8,
        v_int8,
        k_scale,
        v_scale,
        heads=heads,
        head_dim=head_dim,
        causal=causal,
    )
    torch.cuda.synchronize()

    assert torch.equal(packed_output, projection_output)


@requires_cuda
def test_projection_io_attention_rejects_noncontiguous_q() -> None:
    batch, heads, seq, head_dim = 2, 4, 5, 16
    inner_dim = heads * head_dim
    q = torch.randn(
        batch,
        inner_dim,
        seq,
        device="cuda",
        dtype=torch.float16,
    ).transpose(1, 2)
    k_int8 = torch.zeros(
        batch,
        heads,
        seq,
        head_dim,
        device="cuda",
        dtype=torch.int8,
    )
    v_int8 = torch.zeros_like(k_int8)
    k_scale = torch.tensor(0.017, device="cuda", dtype=torch.float32)
    v_scale = torch.tensor(0.021, device="cuda", dtype=torch.float32)

    assert not q.is_contiguous()
    with pytest.raises(
        XQTBackendError,
        match="requires contiguous Q/K/V tensors",
    ):
        fused_kv_int8_attention_projection_forward_tilelang(
            q,
            k_int8,
            v_int8,
            k_scale,
            v_scale,
            heads=heads,
            head_dim=head_dim,
        )


def test_fused_dispatch_passes_scale_buffers_without_scalar_readback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """runtime dispatch 必须传原始 tensor buffer, 不能调用 `.item()`."""

    entity = KvScaleAttention(
        32,
        heads=2,
        k_scale=0.01,
        v_scale=0.02,
        preferred_kernel="tilelang",
    )
    qkv = torch.randn(2, 5, 96, dtype=torch.float16)
    k_int8 = torch.zeros(2, 2, 5, 16, dtype=torch.int8)
    v_int8 = torch.zeros(2, 2, 5, 16, dtype=torch.int8)
    attention_output = torch.empty(2, 5, 32, dtype=torch.float16)
    captured: dict[str, Any] = {}

    def fused_available(_q: torch.Tensor) -> None:
        return None

    def fake_fused(
        qkv_arg: torch.Tensor,
        k_arg: torch.Tensor,
        v_arg: torch.Tensor,
        k_scale_arg: torch.Tensor,
        v_scale_arg: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        captured.update(
            {
                "qkv": qkv_arg,
                "k": k_arg,
                "v": v_arg,
                "k_scale": k_scale_arg,
                "v_scale": v_scale_arg,
                "kwargs": kwargs,
            }
        )
        return attention_output

    def fake_quantize_layout(
        qkv_arg: torch.Tensor,
        k_scale_arg: torch.Tensor,
        v_scale_arg: torch.Tensor,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        captured.update(
            {
                "qkv_projection": qkv_arg,
                "quant_k_scale": k_scale_arg,
                "quant_v_scale": v_scale_arg,
                "quant_kwargs": kwargs,
            }
        )
        return k_int8, v_int8

    monkeypatch.setattr(entity, "_fused_unavailability_reason", fused_available)
    monkeypatch.setattr(
        kv_int8_kernel,
        "fused_kv_int8_attention_packed_qkv_forward_tilelang",
        fake_fused,
    )
    monkeypatch.setattr(
        kv_int8_kernel,
        "quantize_packed_qkv_int8_layout_tilelang",
        fake_quantize_layout,
    )

    output = entity._try_fused_attention(qkv)

    assert output is attention_output
    assert captured["qkv_projection"] is qkv
    assert captured["qkv"] is qkv
    assert captured["quant_k_scale"] is entity.k_scale
    assert captured["quant_v_scale"] is entity.v_scale
    assert captured["quant_kwargs"] == {
        "heads": 2,
        "head_dim": 16,
        "qmax": 127,
        "block_size": 256,
    }
    assert captured["k"] is k_int8
    assert captured["v"] is v_int8
    assert captured["k_scale"] is entity.k_scale
    assert captured["v_scale"] is entity.v_scale
    assert isinstance(captured["k_scale"], torch.Tensor)
    assert isinstance(captured["v_scale"], torch.Tensor)
    assert captured["kwargs"] == {
        "heads": 2,
        "head_dim": 16,
        "causal": False,
        "block_m": 64,
        "block_n": 64,
    }
    report = entity.report()
    assert report["selected_kernel"] == KV_INT8_PACKED_QKV_ATTENTION_KERNEL_NAME
    assert report["selected_kernels"] == [
        KV_INT8_PACKED_QKV_QUANTIZE_LAYOUT_KERNEL_NAME,
        KV_INT8_PACKED_QKV_ATTENTION_KERNEL_NAME,
    ]


@requires_cuda
def test_cuda_graph_cache_key_tracks_tensor_and_attention_contract() -> None:
    entity, x = _build_cuda_entity(seq=16, causal=False)
    same_contract = x.clone()
    different_shape = x[:, :-1].contiguous()
    strided_storage = torch.empty(
        _BATCH,
        16,
        2 * _DIM,
        device=x.device,
        dtype=x.dtype,
    )
    different_stride = strided_storage[..., ::2]
    different_dtype = x.float()

    base_key = entity._graph_cache_key(x)
    assert entity._graph_cache_key(same_contract) == base_key
    assert entity._graph_cache_key(different_shape) != base_key
    assert entity._graph_cache_key(different_stride) != base_key
    assert entity._graph_cache_key(different_dtype) != base_key

    entity.causal = True
    assert entity._graph_cache_key(x) != base_key


@requires_cuda_tilelang
@pytest.mark.parametrize("causal", [False, True])
def test_cuda_graph_full_forward_captures_and_replays_dynamic_inputs_bitwise(
    causal: bool,
) -> None:
    eager_entity, x = _build_cuda_entity(seq=128, causal=causal)
    eager_entity.preferred_kernel = "tilelang"
    graph_entity = copy.deepcopy(eager_entity)
    graph_entity.attention_fastpath = "graph"
    x_next = (x + 0.25).contiguous()

    with torch.inference_mode():
        expected_first = eager_entity(x).clone()
        first_output = graph_entity(x)
        first_snapshot = first_output.clone()
        first_report = graph_entity.report()
        expected_next = eager_entity(x_next).clone()
        second_output = graph_entity(x_next)
        second_snapshot = second_output.clone()
        second_report = graph_entity.report()
    torch.cuda.synchronize()

    assert torch.equal(first_snapshot, expected_first)
    assert torch.equal(second_snapshot, expected_next)
    assert first_output.data_ptr() == second_output.data_ptr()
    assert torch.equal(first_output, second_output)
    assert not torch.equal(first_snapshot, first_output)
    assert first_report["cuda_graph"] == {
        "state": "captured",
        "reason": None,
        "cache_size": 1,
        "output_storage": "graph_owned",
    }
    assert second_report["cuda_graph"] == {
        "state": "replayed",
        "reason": None,
        "cache_size": 1,
        "output_storage": "graph_owned",
    }
    assert second_report["attention_fastpath"] == "graph"
    assert second_report["selected_fastpath"] == (
        "packed_qkv_tilelang_cuda_graph"
    )
    assert second_report["selected_kernels"] == [
        KV_INT8_PACKED_QKV_QUANTIZE_LAYOUT_KERNEL_NAME,
        KV_INT8_PACKED_QKV_ATTENTION_KERNEL_NAME,
    ]
    assert set(graph_entity.state_dict()) == set(eager_entity.state_dict())


@requires_cuda_tilelang
def test_cuda_graph_cache_is_cleared_when_module_dtype_changes() -> None:
    entity, x = _build_cuda_entity(seq=128, causal=False)
    entity.preferred_kernel = "tilelang"
    entity.attention_fastpath = "graph"

    with torch.inference_mode():
        entity(x)
    assert entity.report()["cuda_graph"]["cache_size"] == 1

    entity.float()
    report = entity.report()

    assert report["cuda_graph"] == {
        "state": "disabled",
        "reason": "module device or dtype changed",
        "cache_size": 0,
        "output_storage": None,
    }


@requires_cuda_tilelang
def test_cuda_graph_causal_mode_uses_distinct_cache_entries() -> None:
    graph_entity, x = _build_cuda_entity(seq=128, causal=False)
    graph_entity.preferred_kernel = "tilelang"
    graph_entity.attention_fastpath = "graph"
    eager_entity = copy.deepcopy(graph_entity)
    eager_entity.attention_fastpath = "eager"

    with torch.inference_mode():
        noncausal_expected = eager_entity(x).clone()
        noncausal_output = graph_entity(x).clone()
    assert torch.equal(noncausal_output, noncausal_expected)
    assert graph_entity.report()["cuda_graph"]["cache_size"] == 1

    graph_entity.causal = True
    eager_entity.causal = True
    with torch.inference_mode():
        causal_expected = eager_entity(x).clone()
        causal_output = graph_entity(x).clone()
    causal_report = graph_entity.report()
    assert torch.equal(causal_output, causal_expected)
    assert causal_report["cuda_graph"]["state"] == "captured"
    assert causal_report["cuda_graph"]["cache_size"] == 2

    graph_entity.causal = False
    eager_entity.causal = False
    with torch.inference_mode():
        restored_expected = eager_entity(x).clone()
        restored_output = graph_entity(x).clone()
    restored_report = graph_entity.report()
    assert torch.equal(restored_output, restored_expected)
    assert restored_report["cuda_graph"]["state"] == "replayed"
    assert restored_report["cuda_graph"]["cache_size"] == 2


@requires_cuda
@pytest.mark.parametrize(
    ("case", "error"),
    [
        ("dtype", "k_scale must have dtype torch.float32"),
        ("numel", "k_scale must contain exactly one element"),
        ("device", "k_scale must be on cuda:0"),
    ],
)
def test_fused_scale_tensor_contract_rejects_invalid_inputs(
    case: str,
    error: str,
) -> None:
    q = torch.randn(1, 1, 16, 16, device="cuda", dtype=torch.float16)
    k_int8 = torch.zeros(1, 1, 16, 16, device="cuda", dtype=torch.int8)
    v_int8 = torch.zeros_like(k_int8)
    k_scale = torch.tensor(0.01, device="cuda", dtype=torch.float32)
    v_scale = torch.tensor(0.02, device="cuda", dtype=torch.float32)
    if case == "dtype":
        k_scale = k_scale.half()
    elif case == "numel":
        k_scale = k_scale.repeat(2)
    elif case == "device":
        k_scale = k_scale.cpu()

    with pytest.raises(XQTBackendError, match=error):
        fused_kv_int8_attention_forward_tilelang(
            q,
            k_int8,
            v_int8,
            k_scale,
            v_scale,
        )


def _bench_cuda_event_mean_ms(
    fn: Callable[[], torch.Tensor],
    *,
    warmup: int,
    iters: int,
) -> float:
    """CUDA event 计时, 返回 iters 次调用的平均 latency (毫秒)."""

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end)) / iters


@requires_cuda_tilelang
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("seq", [128, 1024])
def test_benchmark_reference_vs_fused(
    seq: int,
    causal: bool,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """before/after benchmark: reference (SDPA + quant/dequant) vs fused."""

    reference_entity, x = _build_cuda_entity(seq=seq, causal=causal)
    fused_entity = copy.deepcopy(reference_entity)
    fused_entity.preferred_kernel = "tilelang"

    with torch.no_grad():
        reference_ms = _bench_cuda_event_mean_ms(
            lambda: reference_entity(x), warmup=20, iters=50
        )
        fused_ms = _bench_cuda_event_mean_ms(
            lambda: fused_entity(x), warmup=20, iters=50
        )

    with capsys.disabled():
        print(
            f"\n[benchmark] seq={seq} causal={causal}: "
            f"reference={reference_ms:.4f} ms, fused={fused_ms:.4f} ms, "
            f"fused/reference={fused_ms / reference_ms:.3f}x"
        )
    assert reference_ms > 0.0
    assert fused_ms > 0.0
    assert fused_entity.report()["cuda_fused_verified"] is True
