from __future__ import annotations

from collections.abc import Callable

import pytest
import torch
from torch import nn

from xqt.model.hunyuan_ocr_tilelang import (
    HunyuanOcrTileLangCudaGraphRunner,
    HunyuanOcrTileLangDecodeBlock,
    HunyuanOcrTileLangDecodeSpec,
    benchmark_hunyuan_ocr_tilelang_decode_graph,
)
from xqt.operator_opt.kernels.tilelang.hunyuan_block import (
    gqa_decode_attention_reference,
    residual_add_reference,
    residual_rmsnorm_reference,
    rmsnorm_reference,
    swiglu_reference,
)
from xqt.runtime.modules import Int8MmaLinear, SVDQuantInt8MmaLinear

ProjectionFactory = Callable[..., Int8MmaLinear | SVDQuantInt8MmaLinear]


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="HunyuanOCR TileLang decode requires CUDA",
)


class _RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, *, dtype: torch.dtype) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, device="cuda", dtype=dtype))
        self.variance_epsilon = 1e-5


def _quantized_linear(
    input_features: int,
    output_features: int,
    *,
    dtype: torch.dtype,
) -> SVDQuantInt8MmaLinear:
    linear = nn.Linear(
        input_features,
        output_features,
        bias=False,
        device="cuda",
        dtype=dtype,
    )
    return SVDQuantInt8MmaLinear.from_linear(
        linear,
        rank=16,
        group_size=64,
        quant_dtype="int4",
        engine="tilelang",
    ).eval()


def _w8a8_linear(
    input_features: int,
    output_features: int,
    *,
    dtype: torch.dtype,
) -> Int8MmaLinear:
    linear = nn.Linear(
        input_features,
        output_features,
        bias=False,
        device="cuda",
        dtype=dtype,
    )
    activation_scale = torch.tensor(1.0 / 127.0, device="cuda", dtype=torch.float32)
    return Int8MmaLinear.from_linear(
        linear,
        engine="tilelang",
        activation_scale_mode="static",
        activation_scale=activation_scale,
    ).eval()


def _hunyuan_layer(
    *,
    dtype: torch.dtype,
    hidden_size: int,
    intermediate_size: int,
    query_heads: int,
    key_value_heads: int,
    head_dim: int,
    projection_factory: ProjectionFactory | None = None,
) -> nn.Module:
    factory = _quantized_linear if projection_factory is None else projection_factory
    attention_dim = query_heads * head_dim
    key_value_dim = key_value_heads * head_dim

    class _Attention(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.q_proj = factory(hidden_size, attention_dim, dtype=dtype)
            self.k_proj = factory(hidden_size, key_value_dim, dtype=dtype)
            self.v_proj = factory(hidden_size, key_value_dim, dtype=dtype)
            self.o_proj = factory(attention_dim, hidden_size, dtype=dtype)
            self.query_layernorm = _RMSNorm(head_dim, dtype=dtype)
            self.key_layernorm = _RMSNorm(head_dim, dtype=dtype)

    class _MLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gate_proj = factory(hidden_size, intermediate_size, dtype=dtype)
            self.up_proj = factory(hidden_size, intermediate_size, dtype=dtype)
            self.down_proj = factory(intermediate_size, hidden_size, dtype=dtype)

    class _Layer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input_layernorm = _RMSNorm(hidden_size, dtype=dtype)
            self.post_attention_layernorm = _RMSNorm(hidden_size, dtype=dtype)
            self.self_attn = _Attention()
            self.mlp = _MLP()

    return _Layer()


def _identity_rope(
    query: torch.Tensor,
    key: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return query, key


def _reference_forward(
    block: HunyuanOcrTileLangDecodeBlock,
    hidden_states: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
) -> torch.Tensor:
    normed = rmsnorm_reference(
        hidden_states,
        block.input_norm_weight,
        eps=block.input_norm_eps,
    )
    query = block._reshape_q(block.q_proj(normed))
    key = block._reshape_kv(block.k_proj(normed))
    value = block._reshape_kv(block.v_proj(normed))
    query, key = _identity_rope(query, key)
    query = rmsnorm_reference(query, block.query_norm_weight, eps=block.query_norm_eps)
    key = rmsnorm_reference(key, block.key_norm_weight, eps=block.key_norm_eps)
    key_cache[:, :, -1, :].copy_(key.squeeze(2))
    value_cache[:, :, -1, :].copy_(value.squeeze(2))
    attention = gqa_decode_attention_reference(query.squeeze(2), key_cache, value_cache)
    attention_output = block.o_proj(block._flatten_attention(attention))
    residual, normed = residual_rmsnorm_reference(
        attention_output,
        hidden_states,
        block.post_attention_norm_weight,
        eps=block.post_attention_norm_eps,
    )
    activated = swiglu_reference(block.gate_proj(normed), block.up_proj(normed))
    return residual_add_reference(block.down_proj(activated), residual)


def _build_block(
    dtype: torch.dtype,
    *,
    hidden_size: int = 128,
    intermediate_size: int = 128,
    query_heads: int = 2,
    key_value_heads: int = 1,
    head_dim: int = 64,
    kv_cache_length: int = 64,
    projection_factory: ProjectionFactory | None = None,
) -> HunyuanOcrTileLangDecodeBlock:
    layer = _hunyuan_layer(
        dtype=dtype,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        query_heads=query_heads,
        key_value_heads=key_value_heads,
        head_dim=head_dim,
        projection_factory=projection_factory,
    )
    spec = HunyuanOcrTileLangDecodeSpec(
        batch_size=1,
        hidden_size=hidden_size,
        query_heads=query_heads,
        key_value_heads=key_value_heads,
        head_dim=head_dim,
        kv_cache_length=kv_cache_length,
        intermediate_size=intermediate_size,
    )
    return HunyuanOcrTileLangDecodeBlock(
        layer,
        spec,
        position_transform=_identity_rope,
    ).eval()


def _projection_execution_metadata(
    block: HunyuanOcrTileLangDecodeBlock,
) -> list[dict[str, object]]:
    modules = [
        block.q_proj,
        block.k_proj,
        block.v_proj,
        block.o_proj,
        block.gate_proj,
        block.up_proj,
        block.down_proj,
    ]
    return [module.execution_metadata() for module in modules]


def test_hunyuan_tilelang_decode_block_matches_composed_reference() -> None:
    torch.manual_seed(5)
    block = _build_block(torch.float16)
    hidden_states = torch.randn(1, 1, 128, device="cuda", dtype=torch.float16)
    key_cache = torch.randn(1, 1, 64, 64, device="cuda", dtype=torch.float16)
    value_cache = torch.randn_like(key_cache)
    reference_key_cache = key_cache.clone()
    reference_value_cache = value_cache.clone()

    actual = block(hidden_states, key_cache, value_cache)
    expected = _reference_forward(
        block,
        hidden_states,
        reference_key_cache,
        reference_value_cache,
    )

    torch.testing.assert_close(actual.float(), expected.float(), rtol=5e-2, atol=5e-2)
    torch.testing.assert_close(
        key_cache.float(), reference_key_cache.float(), rtol=1e-1, atol=1e-2
    )
    torch.testing.assert_close(
        value_cache.float(), reference_value_cache.float(), rtol=1e-1, atol=1e-2
    )


def test_hunyuan_tilelang_decode_block_accepts_w8a8_projections() -> None:
    torch.manual_seed(7)
    block = _build_block(torch.float16, projection_factory=_w8a8_linear)
    hidden_states = torch.randn(1, 1, 128, device="cuda", dtype=torch.float16)
    key_cache = torch.randn(1, 1, 64, 64, device="cuda", dtype=torch.float16)
    value_cache = torch.randn_like(key_cache)
    reference_key_cache = key_cache.clone()
    reference_value_cache = value_cache.clone()

    actual = block(hidden_states, key_cache, value_cache)
    expected = _reference_forward(
        block,
        hidden_states,
        reference_key_cache,
        reference_value_cache,
    )

    torch.testing.assert_close(actual.float(), expected.float(), rtol=5e-2, atol=5e-2)
    assert block.q_proj.execution_metadata()["engine"] == "bf16_fallback"
    assert block.o_proj.execution_metadata()["engine"] == "bf16_fallback"


def test_hunyuan_tilelang_decode_spec_allows_independent_attention_dim() -> None:
    spec = HunyuanOcrTileLangDecodeSpec(
        batch_size=1,
        hidden_size=1024,
        query_heads=16,
        key_value_heads=8,
        head_dim=128,
        kv_cache_length=128,
        intermediate_size=3584,
    )
    spec.validate()
    assert spec.attention_dim == 2048
    assert spec.key_value_dim == 1024
    assert spec.attention_dim != spec.hidden_size
    assert spec.int8_block_m == 16
    assert spec.gqa_query_tile_rows == 1
    assert spec.decode_min_int8_rows == 16


def test_hunyuan_tilelang_decode_block_applies_decode_int8_schedule() -> None:
    block = _build_block(torch.float16, projection_factory=_w8a8_linear)
    for module in (
        block.q_proj,
        block.k_proj,
        block.v_proj,
        block.o_proj,
        block.gate_proj,
        block.up_proj,
        block.down_proj,
    ):
        assert module.block_m == block.spec.int8_block_m
        assert module.min_int8_rows == block.spec.decode_min_int8_rows


def test_hunyuan_tilelang_decode_uses_m1_int8_gemv_path() -> None:
    torch.manual_seed(13)
    block = _build_block(torch.float16, projection_factory=_w8a8_linear)
    for module in (
        block.q_proj,
        block.k_proj,
        block.v_proj,
        block.o_proj,
        block.gate_proj,
        block.up_proj,
        block.down_proj,
    ):
        module.min_int8_rows = 0
    sample = torch.randn(1, 1, 128, device="cuda", dtype=torch.float16)
    _ = block.q_proj(sample)
    metadata = block.q_proj.execution_metadata()
    assert metadata.get("padded_rows") == 1
    assert metadata.get("activation_quant_engine") in {
        "ptx_sm89_m1_dp4a_gemv",
        "torch_m1_int8_products",
        "tilelang_m1_static_gemv",
    }
    assert metadata["runtime_precision"]["execution_kind"] == "native_w8a8_int8_mma"


def test_hunyuan_tilelang_decode_block_real_qo_dims_match_reference() -> None:
    torch.manual_seed(11)
    block = _build_block(
        torch.float16,
        hidden_size=128,
        intermediate_size=128,
        query_heads=4,
        key_value_heads=2,
        head_dim=64,
        kv_cache_length=64,
        projection_factory=_w8a8_linear,
    )
    assert block.spec.attention_dim == 256
    assert block.spec.attention_dim != block.spec.hidden_size
    assert block.q_proj.output_features == 256
    assert block.o_proj.input_features == 256
    assert block.o_proj.output_features == 128

    hidden_states = torch.randn(1, 1, 128, device="cuda", dtype=torch.float16)
    key_cache = torch.randn(1, 2, 64, 64, device="cuda", dtype=torch.float16)
    value_cache = torch.randn_like(key_cache)
    reference_key_cache = key_cache.clone()
    reference_value_cache = value_cache.clone()

    actual = block(hidden_states, key_cache, value_cache)
    expected = _reference_forward(
        block,
        hidden_states,
        reference_key_cache,
        reference_value_cache,
    )

    torch.testing.assert_close(actual.float(), expected.float(), rtol=5e-2, atol=5e-2)
    assert actual.shape == (1, 1, 128)


def test_hunyuan_tilelang_decode_block_cuda_graph_replays() -> None:
    torch.manual_seed(6)
    block = _build_block(torch.float16)
    key_cache = torch.randn(1, 1, 64, 64, device="cuda", dtype=torch.float16)
    value_cache = torch.randn_like(key_cache)
    hidden_states = torch.randn(1, 1, 128, device="cuda", dtype=torch.float16)
    runner = HunyuanOcrTileLangCudaGraphRunner(
        block,
        key_cache=key_cache,
        value_cache=value_cache,
    )

    runner.capture(hidden_states)
    actual = runner.replay(hidden_states)

    assert actual.shape == (1, 1, 128)
    assert torch.isfinite(actual).all()
    report = benchmark_hunyuan_ocr_tilelang_decode_graph(
        runner,
        hidden_states,
        warmup=1,
        iterations=2,
    )
    assert report.mean_ms > 0.0


def test_hunyuan_tilelang_decode_real_qo_cuda_graph_replays() -> None:
    torch.manual_seed(12)
    block = _build_block(
        torch.float16,
        hidden_size=128,
        intermediate_size=128,
        query_heads=4,
        key_value_heads=2,
        head_dim=64,
        kv_cache_length=64,
        projection_factory=_w8a8_linear,
    )
    key_cache = torch.randn(1, 2, 64, 64, device="cuda", dtype=torch.float16)
    value_cache = torch.randn_like(key_cache)
    hidden_states = torch.randn(1, 1, 128, device="cuda", dtype=torch.float16)
    runner = HunyuanOcrTileLangCudaGraphRunner(
        block,
        key_cache=key_cache,
        value_cache=value_cache,
    )

    runner.capture(hidden_states)
    eager = block(
        hidden_states,
        key_cache.clone(),
        value_cache.clone(),
    )
    graph_out = runner.replay(hidden_states)

    assert graph_out.shape == (1, 1, 128)
    assert torch.isfinite(graph_out).all()
    torch.testing.assert_close(graph_out.float(), eager.float(), rtol=5e-2, atol=5e-2)
    report = benchmark_hunyuan_ocr_tilelang_decode_graph(
        runner,
        hidden_states,
        warmup=2,
        iterations=5,
    )
    assert report.mean_ms > 0.0


def test_hunyuan_tilelang_decode_graph_benchmark_is_model_exported() -> None:
    from xqt.model import benchmark_hunyuan_ocr_tilelang_decode_graph as exported

    assert exported is benchmark_hunyuan_ocr_tilelang_decode_graph
