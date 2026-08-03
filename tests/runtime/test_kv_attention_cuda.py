"""CUDA tests for the KvScaleAttention TileLang KV-int8 fused attention path."""

from __future__ import annotations

import copy

import pytest
import torch

from xqt.operator_opt.kernels.tilelang._common import (
    tilelang_runtime_unavailability_reason,
    tilelang_runtime_usable,
)
from xqt.operator_opt.kernels.tilelang.kv_int8_attention import (
    KV_INT8_FUSED_KERNEL_NAME,
)
from xqt.contracts.runtime_quant import (
    RuntimeQuantContract,
    build_runtime_quant_contract,
)
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
        k_scale = float(entity.k_proj(x).abs().max().item()) / 127.0
        v_scale = float(entity.v_proj(x).abs().max().item()) / 127.0
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
    assert report["selected_kernel"] == KV_INT8_FUSED_KERNEL_NAME
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
    assert report["selected_kernel"] == KV_INT8_FUSED_KERNEL_NAME
    assert report["cuda_fused_verified"] is True


@requires_cuda
def test_fallback_to_reference_on_non_fp16_cuda() -> None:
    entity, x = _build_cuda_entity(seq=128, causal=False, dtype=torch.float32)
    entity.preferred_kernel = "tilelang"
    with torch.no_grad():
        output = entity(x)
    assert output.shape == x.shape
    report = entity.report()
    assert report["selected_kernel"] == "torch_sdpa_kv_scale_reference"
    assert report["cuda_fused_verified"] is False
    assert report["fallback_reason"] == "dtype_not_fp16"


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


def _bench_cuda_event_mean_ms(fn, *, warmup: int, iters: int) -> float:
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
