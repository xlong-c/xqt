"""CUDA-graph decode session tests; the runtime path requires CUDA."""

from __future__ import annotations

import pytest
import torch
from torch import nn
from transformers import LlamaConfig, LlamaForCausalLM

from xqt.core.errors import XQTBackendError
from xqt.runtime import CudaGraphDecodeSession


requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for the CUDA-graph decode session",
)


def _tiny_llama() -> nn.Module:
    torch.manual_seed(7)
    config = LlamaConfig(
        vocab_size=256,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=512,
    )
    model = LlamaForCausalLM(config)
    return model.eval().to(device="cuda", dtype=torch.bfloat16)


@requires_cuda
def test_graph_decode_matches_hf_greedy() -> None:
    model = _tiny_llama()
    input_ids = torch.randint(0, 200, (1, 12), device="cuda")
    with torch.inference_mode():
        reference = model.generate(
            input_ids=input_ids,
            max_new_tokens=16,
            do_sample=False,
            pad_token_id=0,
        )
    session = CudaGraphDecodeSession(model, max_cache_len=128, fuse_norms=False)
    result = session.generate(input_ids, max_new_tokens=16)

    assert result.hit_eos is False
    assert result.decode_steps == 15
    assert result.token_ids == reference[0, input_ids.shape[1] :].tolist()


@requires_cuda
def test_graph_decode_serves_a_second_prompt() -> None:
    model = _tiny_llama()
    session = CudaGraphDecodeSession(model, max_cache_len=128, fuse_norms=False)
    first_ids = torch.randint(0, 200, (1, 20), device="cuda")
    second_ids = torch.randint(0, 200, (1, 8), device="cuda")

    first = session.generate(first_ids, max_new_tokens=8)
    second = session.generate(second_ids, max_new_tokens=8)

    with torch.inference_mode():
        reference = model.generate(
            input_ids=second_ids,
            max_new_tokens=8,
            do_sample=False,
            pad_token_id=0,
        )
    assert first.generated_tokens == 8
    assert second.token_ids == reference[0, second_ids.shape[1] :].tolist()


@requires_cuda
def test_graph_decode_stops_before_prefill_limit() -> None:
    model = _tiny_llama()
    session = CudaGraphDecodeSession(model, max_cache_len=32, fuse_norms=False)
    input_ids = torch.randint(0, 200, (1, 8), device="cuda")

    with pytest.raises(ValueError, match="exceeds max_cache_len"):
        session.generate(input_ids, max_new_tokens=32)


@requires_cuda
def test_graph_decode_rejects_non_llama_model() -> None:
    model = nn.Sequential(nn.Linear(8, 8)).to("cuda")

    with pytest.raises(XQTBackendError, match="Llama-family"):
        CudaGraphDecodeSession(model, max_cache_len=64)


@requires_cuda
def test_fused_rms_norm_stays_close_to_the_composite_path() -> None:
    from transformers.models.llama.modeling_llama import LlamaRMSNorm

    from xqt.runtime.graph_decode import _rms_norm

    torch.manual_seed(11)
    norm = LlamaRMSNorm(256, eps=1e-5).to(device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        norm.weight.copy_(torch.rand_like(norm.weight) + 0.5)
    inputs = torch.randn(1, 1, 256, dtype=torch.bfloat16, device="cuda") * 3.0

    fused = _rms_norm(norm, inputs, fused=True)
    composite = _rms_norm(norm, inputs, fused=False)
    torch.testing.assert_close(fused, composite, rtol=2e-2, atol=2e-2)


@requires_cuda
def test_graph_decode_runs_with_fused_norms_enabled() -> None:
    model = _tiny_llama()
    input_ids = torch.randint(0, 200, (1, 12), device="cuda")
    session = CudaGraphDecodeSession(model, max_cache_len=128, fuse_norms=True)

    result = session.generate(input_ids, max_new_tokens=8)

    assert result.generated_tokens == 8
    assert result.captured_graphs == 1


def _quantized_mlp_llama() -> nn.Module:
    """Tiny Llama whose MLP projections expose the AWQ INT8 prefill view."""

    from xqt.contracts.weight_only import AWQGPTQWeightOnlyLinear
    from xqt.model.minicpm5 import _MiniCPM5W4A16HybridLinear

    model = _tiny_llama()
    for layer in model.model.layers:
        for name in ("gate_proj", "up_proj", "down_proj"):
            linear = getattr(layer.mlp, name)
            storage = AWQGPTQWeightOnlyLinear.from_linear(
                linear, bits=4, group_size=64, method="rtn"
            )
            setattr(
                layer.mlp,
                name,
                _MiniCPM5W4A16HybridLinear(storage, decode=None),
            )
    return model


@requires_cuda
def test_graph_decode_runs_with_int8_prefill() -> None:
    model = _quantized_mlp_llama()
    input_ids = torch.randint(0, 200, (1, 12), device="cuda")
    reference = CudaGraphDecodeSession(model, max_cache_len=128, fuse_norms=False)
    session = CudaGraphDecodeSession(
        model,
        max_cache_len=128,
        fuse_norms=False,
        int8_prefill=True,
        min_int8_prefill_rows=1,
    )

    reference_first = reference.prefill(input_ids)
    first = session.prefill(input_ids)

    assert isinstance(first, int)
    assert 0 <= first < model.config.vocab_size
    assert isinstance(reference_first, int)
    # The INT8 branch actually ran: the lazy weight view was materialized.
    gate = model.model.layers[0].mlp.gate_proj
    assert gate._int8_weight is not None
    assert gate._int8_scale is not None
    assert session.length == int(input_ids.shape[1])


@requires_cuda
def test_graph_decode_int8_prefill_falls_back_below_row_threshold() -> None:
    model = _quantized_mlp_llama()
    input_ids = torch.randint(0, 200, (1, 12), device="cuda")
    session = CudaGraphDecodeSession(
        model,
        max_cache_len=128,
        fuse_norms=False,
        int8_prefill=True,
        min_int8_prefill_rows=64,
    )

    first = session.prefill(input_ids)

    assert isinstance(first, int)
    assert model.model.layers[0].mlp.gate_proj._int8_weight is None


@requires_cuda
def test_graph_decode_rejects_bad_int8_prefill_rows() -> None:
    model = _tiny_llama()

    with pytest.raises(ValueError, match="min_int8_prefill_rows"):
        CudaGraphDecodeSession(model, max_cache_len=64, min_int8_prefill_rows=0)


@requires_cuda
def test_graph_decode_runs_with_int8_activations() -> None:
    model = _tiny_llama()
    input_ids = torch.randint(0, 200, (1, 12), device="cuda")
    session = CudaGraphDecodeSession(model, max_cache_len=128, int8_activations=True)

    result = session.generate(input_ids, max_new_tokens=8)

    assert result.generated_tokens == 8


@requires_cuda
def test_graph_decode_rejects_unknown_attention_impl() -> None:
    model = _tiny_llama()

    with pytest.raises(ValueError, match="attention_impl"):
        CudaGraphDecodeSession(model, max_cache_len=64, attention_impl="fa3")


@requires_cuda
def test_graph_decode_rejects_unknown_prefill_attention_impl() -> None:
    model = _tiny_llama()

    with pytest.raises(ValueError, match="prefill_attention_impl"):
        CudaGraphDecodeSession(model, max_cache_len=64, prefill_attention_impl="flash")


@requires_cuda
def test_graph_decode_prefill_triton_matches_sdpa_default() -> None:
    model = _tiny_llama()
    input_ids = torch.randint(0, 200, (1, 32), device="cuda")
    reference = CudaGraphDecodeSession(model, max_cache_len=128, fuse_norms=False)
    session = CudaGraphDecodeSession(
        model,
        max_cache_len=128,
        fuse_norms=False,
        prefill_attention_impl="triton",
    )

    reference_first = reference.prefill(input_ids)
    first = session.prefill(input_ids)

    assert first == reference_first
    assert session.length == int(input_ids.shape[1])


@requires_cuda
def test_graph_decode_rejects_tilelang_prefill_for_gqa() -> None:
    # _tiny_llama is GQA (4 q heads / 2 kv heads); TileLang has no GQA path.
    model = _tiny_llama()

    with pytest.raises(XQTBackendError, match="GQA"):
        CudaGraphDecodeSession(
            model, max_cache_len=64, prefill_attention_impl="tilelang"
        )


@requires_cuda
def test_graph_decode_tensor_core_attention_matches_simt() -> None:
    model = _tiny_llama()
    input_ids = torch.randint(0, 200, (1, 12), device="cuda")
    reference = CudaGraphDecodeSession(model, max_cache_len=128, fuse_norms=False)
    expected = reference.generate(input_ids, max_new_tokens=8).token_ids

    session = CudaGraphDecodeSession(
        model, max_cache_len=128, fuse_norms=False, attention_impl="tc"
    )
    result = session.generate(input_ids, max_new_tokens=8)

    assert result.token_ids == expected


@requires_cuda
def test_decode_batch_matches_single_steps() -> None:
    model = _tiny_llama()
    input_ids = torch.randint(0, 200, (1, 12), device="cuda")
    session = CudaGraphDecodeSession(model, max_cache_len=128, fuse_norms=False)
    single = CudaGraphDecodeSession(model, max_cache_len=128, fuse_norms=False)

    session.prefill(input_ids)
    single.prefill(input_ids)
    batched = session.decode_batch(5).tolist()
    one_by_one = [single.decode_step() for _ in range(5)]

    assert batched == one_by_one
    assert session.decode_steps == 5
    assert single.decode_steps == 5
    with pytest.raises(ValueError, match="steps must be positive"):
        session.decode_batch(0)
    with pytest.raises(ValueError, match="readback_chunk"):
        CudaGraphDecodeSession(model, max_cache_len=64, readback_chunk=0)


@requires_cuda
def test_capture_after_prefill_keeps_prompt_state() -> None:
    model = _tiny_llama()
    input_ids = torch.randint(0, 200, (1, 12), device="cuda")
    reference = CudaGraphDecodeSession(model, max_cache_len=128, fuse_norms=False)
    expected = reference.generate(input_ids, max_new_tokens=8).token_ids

    session = CudaGraphDecodeSession(model, max_cache_len=128, fuse_norms=False)
    with torch.inference_mode():
        first = session.prefill(input_ids)
        session.capture()
        tokens = [first] + [session.decode_step() for _ in range(7)]

    assert tokens == expected


@requires_cuda
def test_generate_truncates_at_eos_inside_a_readback_chunk() -> None:
    model = _tiny_llama()
    input_ids = torch.randint(0, 200, (1, 12), device="cuda")
    probe = CudaGraphDecodeSession(model, max_cache_len=128, fuse_norms=False)
    probe.prefill(input_ids)
    eos = probe.decode_step()
    del probe

    session = CudaGraphDecodeSession(
        model, max_cache_len=128, fuse_norms=False, readback_chunk=8
    )
    result = session.generate(input_ids, max_new_tokens=32, eos_token_ids=(eos,))

    assert result.hit_eos is True
    assert result.token_ids[-1] == eos
    assert result.token_ids.count(eos) == 1
    assert result.generated_tokens < 32
    # every replay is counted, including the ones that overran EOS
    assert result.decode_steps >= result.generated_tokens - 1


@requires_cuda
def test_graph_decode_with_int8_kv_cache() -> None:
    model = _tiny_llama()
    input_ids = torch.randint(0, 200, (1, 12), device="cuda")

    session = CudaGraphDecodeSession(
        model, max_cache_len=128, fuse_norms=False, kv_quant="int8"
    )
    result = session.generate(input_ids, max_new_tokens=16)

    assert result.hit_eos is False
    assert result.decode_steps == 15
    assert len(result.token_ids) == 16
    assert session.k_cache[0].dtype == torch.int8
    assert session.v_cache[0].dtype == torch.int8
    assert session.k_scale is not None
    assert session.k_scale[0].shape == (1, 2, 128)


@requires_cuda
def test_graph_decode_int8_kv_batch_matches_single_steps() -> None:
    model = _tiny_llama()
    input_ids = torch.randint(0, 200, (1, 12), device="cuda")
    session = CudaGraphDecodeSession(
        model, max_cache_len=128, fuse_norms=False, kv_quant="int8"
    )
    single = CudaGraphDecodeSession(
        model, max_cache_len=128, fuse_norms=False, kv_quant="int8"
    )

    session.prefill(input_ids)
    single.prefill(input_ids)
    batched = session.decode_batch(6).tolist()
    one_by_one = [single.decode_step() for _ in range(6)]

    assert batched == one_by_one
    assert session.decode_steps == 6
    assert single.decode_steps == 6


@requires_cuda
def test_graph_decode_int8_kv_serves_a_second_prompt() -> None:
    model = _tiny_llama()
    session = CudaGraphDecodeSession(
        model, max_cache_len=128, fuse_norms=False, kv_quant="int8"
    )
    first_ids = torch.randint(0, 200, (1, 16), device="cuda")
    second_ids = torch.randint(0, 200, (1, 10), device="cuda")

    first = session.generate(first_ids, max_new_tokens=8)
    second = session.generate(second_ids, max_new_tokens=8)

    assert first.generated_tokens == 8
    assert second.generated_tokens == 8
    assert session.length == 10 + 7


@requires_cuda
def test_graph_decode_with_int4_kv_cache() -> None:
    model = _tiny_llama()
    input_ids = torch.randint(0, 200, (1, 12), device="cuda")

    session = CudaGraphDecodeSession(
        model, max_cache_len=128, fuse_norms=False, kv_quant="int4"
    )
    result = session.generate(input_ids, max_new_tokens=16)

    assert result.hit_eos is False
    assert result.decode_steps == 15
    assert len(result.token_ids) == 16
    # In INT4, head_dim 64 (256/4) is packed into 32 bytes (or 128 packed into 64 bytes)
    head_dim = 256 // 4
    assert session.k_cache[0].shape == (1, 2, 128, head_dim // 2)
    assert session.v_cache[0].shape == (1, 2, 128, head_dim // 2)
    assert session.k_cache[0].dtype == torch.int8
    assert session.v_cache[0].dtype == torch.int8
    assert session.k_scale is not None
    assert session.k_scale[0].shape == (1, 2, 128)


@requires_cuda
def test_graph_decode_int4_kv_batch_matches_single_steps() -> None:
    model = _tiny_llama()
    input_ids = torch.randint(0, 200, (1, 12), device="cuda")
    session = CudaGraphDecodeSession(
        model, max_cache_len=128, fuse_norms=False, kv_quant="int4"
    )
    single = CudaGraphDecodeSession(
        model, max_cache_len=128, fuse_norms=False, kv_quant="int4"
    )

    session.prefill(input_ids)
    single.prefill(input_ids)
    batched = session.decode_batch(6).tolist()
    one_by_one = [single.decode_step() for _ in range(6)]

    assert batched == one_by_one
    assert session.decode_steps == 6
    assert single.decode_steps == 6


@requires_cuda
def test_graph_decode_int4_kv_serves_a_second_prompt() -> None:
    model = _tiny_llama()
    session = CudaGraphDecodeSession(
        model, max_cache_len=128, fuse_norms=False, kv_quant="int4"
    )
    first_ids = torch.randint(0, 200, (1, 16), device="cuda")
    second_ids = torch.randint(0, 200, (1, 10), device="cuda")

    first = session.generate(first_ids, max_new_tokens=8)
    second = session.generate(second_ids, max_new_tokens=8)

    assert first.generated_tokens == 8
    assert second.generated_tokens == 8
    assert session.length == 10 + 7
