from __future__ import annotations

import re

import pytest
import torch
from torch import nn

from xqt.model import resolve_model_profile
from xqt.model.minicpm5 import (
    materialize_minicpm5_int8_runtime,
    minicpm5_edge_protected_quantization_policy,
    minicpm5_mlp_only_quantization_policy,
)


def test_minicpm5_profile_declares_standard_llm_loader() -> None:
    profile = resolve_model_profile("hf.minicpm5-2b")

    assert profile.family == "llm"
    assert profile.loader_target == "xqt.model.minicpm5.load_minicpm5"
    assert profile.metadata["architecture"] == "LlamaForCausalLM"
    assert profile.metadata["text_only"] is True


def test_minicpm5_mlp_policy_keeps_attention_and_lm_head() -> None:
    policy = minicpm5_mlp_only_quantization_policy()

    assert any(re.search(pattern, "model.layers.3.self_attn.q_proj") for pattern in policy.exclude_name_patterns)
    assert any(re.search(pattern, "model.lm_head") for pattern in policy.exclude_name_patterns)
    assert not any(re.search(pattern, "model.layers.3.mlp.gate_proj") for pattern in policy.exclude_name_patterns[1:])


def test_minicpm5_edge_policy_protects_expected_layers() -> None:
    policy = minicpm5_edge_protected_quantization_policy(edge_layers=2)
    patterns = policy.exclude_name_patterns

    assert any(re.search(pattern, "model.layers.0.mlp.gate_proj") for pattern in patterns)
    assert any(re.search(pattern, "model.layers.41.mlp.gate_proj") for pattern in patterns)
    assert not any(re.search(pattern, "model.layers.3.mlp.gate_proj") for pattern in patterns)


def test_minicpm5_edge_policy_rejects_invalid_layer_count() -> None:
    with pytest.raises(ValueError, match="edge_layers"):
        minicpm5_edge_protected_quantization_policy(edge_layers=22)


def test_minicpm5_materializes_int8_runtime_view() -> None:
    from xqt.compression.quant.quantizers.int8_mma import Int8MmaLinear as StorageLinear
    from xqt.runtime.modules.int8_mma_linear import Int8MmaLinear as RuntimeLinear

    model = nn.Sequential(nn.Linear(16, 32, bias=False)).eval()
    storage = StorageLinear.from_linear(model[0], engine="auto")
    model[0] = storage

    materialized = materialize_minicpm5_int8_runtime(model)

    assert isinstance(materialized[0], RuntimeLinear)
    assert materialized[0].engine == "auto"


def test_minicpm5_w4_lm_head_helper_serves_decode_rows() -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")

    from transformers import LlamaConfig, LlamaForCausalLM

    from xqt.model.minicpm5 import quantize_minicpm5_lm_head_w4

    torch.manual_seed(3)
    config = LlamaConfig(
        vocab_size=256,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
    )
    model = LlamaForCausalLM(config).eval().to(device="cuda", dtype=torch.bfloat16)
    head = quantize_minicpm5_lm_head_w4(model)
    inputs = torch.randn(1, 256, dtype=torch.bfloat16, device="cuda")

    with torch.no_grad():
        quantized = head(inputs)
        reference = model.lm_head(inputs)

    assert quantized.shape == reference.shape
    assert quantized.dtype == reference.dtype
    assert float((quantized.float() - reference.float()).abs().mean()) < 1.0


def test_minicpm5_w4_lm_head_helper_rejects_missing_linear_head() -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")

    from xqt.model.minicpm5 import quantize_minicpm5_lm_head_w4

    with pytest.raises(TypeError, match="lm_head"):
        quantize_minicpm5_lm_head_w4(nn.Sequential(nn.Linear(8, 8)))
