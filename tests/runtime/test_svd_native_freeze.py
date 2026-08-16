from __future__ import annotations

import pytest
import torch
from diffusers.models.transformers.transformer_flux import FluxAttention
from torch import nn

from tests.xqt.svd_test_helpers import make_legacy_svd_linear
from xqt.runtime import pack_diffusers_flux_rotary_emb
from xqt.runtime.modules import (
    SVDQuantFluxAttention,
    SVDQuantGeluMLP,
    SVDQuantLinear,
)


def _require_native_w4a4() -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 9):
        pytest.skip("native sm_89 W4A4 backend unavailable")
    try:
        from xqt.operator_opt.kernels.cute.svdq_w4a4_sm89 import (
            native_w4a4_available,
            native_w4a4_smalln_available,
        )
    except Exception:
        pytest.skip("native sm_89 W4A4 backend unavailable")
    if not native_w4a4_available(build=False):
        pytest.skip("native sm_89 W4A4 backend unavailable")
    if not native_w4a4_smalln_available(build=False):
        pytest.skip("native sm_89 small-N W4A4 backend unavailable")


def _linear(
    input_features: int,
    output_features: int,
    *,
    seed: int,
) -> SVDQuantLinear:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    source = nn.Linear(
        input_features,
        output_features,
        bias=True,
        device="cuda",
        dtype=torch.float16,
    ).eval()
    with torch.no_grad():
        source.weight.copy_(
            torch.randn(
                source.weight.shape,
                generator=generator,
                device="cuda",
                dtype=torch.float16,
            )
            * 0.02
        )
        source.bias.copy_(
            torch.randn(
                source.bias.shape,
                generator=generator,
                device="cuda",
                dtype=torch.float16,
            )
            * 0.02
        )
    module = make_legacy_svd_linear(
        source,
        rank=16,
        group_size=64,
        quant_dtype="int4",
    ).half().cuda().eval()
    assert module.enable_fusion() is True
    return module


def _assert_native_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)


def test_freeze_native_inference_preserves_outputs_and_releases_canonical_state() -> None:
    _require_native_w4a4()
    module = _linear(128, 128, seed=101)
    inputs = torch.randn(512, 128, device="cuda", dtype=torch.float16)
    canonical_state = module.state_dict()
    canonical_bytes = module._canonical_storage_bytes()

    with torch.inference_mode():
        expected = module(inputs).clone()
    released_bytes = module.freeze_native_inference(
        device="cuda",
        dtype=torch.float16,
        layouts=("main",),
    )
    with torch.inference_mode():
        actual = module(inputs).clone()

    _assert_native_close(actual, expected)
    assert released_bytes == canonical_bytes
    assert released_bytes > 0
    assert module.native_only is True
    assert list(module.parameters()) == []
    assert list(module.buffers()) == []
    assert isinstance(module.down_proj, nn.Identity)
    assert isinstance(module.up_proj, nn.Identity)
    assert module.packed_residual is None
    assert module.residual_scale is None
    assert module.bias is None
    metadata = module.execution_metadata()
    assert metadata["native_only"] is True
    assert metadata["native_only_layouts"] == ["main"]
    assert metadata["native_only_device"] == "cuda:0"
    assert metadata["native_only_dtype"] == "torch.float16"
    assert metadata["native_only_released_bytes"] == released_bytes

    with pytest.raises(RuntimeError, match="cannot be serialized"):
        module.state_dict()
    with pytest.raises(RuntimeError, match="cannot load state"):
        module.load_state_dict(canonical_state)
    with pytest.raises(RuntimeError, match="cannot be moved or cast"):
        module.to(dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="inference-only"):
        module.train()
    with pytest.raises(RuntimeError, match="cannot change fused_norm"):
        module.set_fused_norm(torch.ones(128, device="cuda", dtype=torch.float16))
    with pytest.raises(RuntimeError, match="cannot clear fused_norm"):
        module.clear_fused_norm()
    with pytest.raises(RuntimeError, match="cannot dequantize"):
        module.dequantize_residual()
    with pytest.raises(RuntimeError, match="already frozen"):
        module.freeze_native_inference(
            device="cuda",
            dtype=torch.float16,
            layouts=("main",),
        )


def test_freeze_native_inference_keeps_layout_selection_strict() -> None:
    _require_native_w4a4()
    module = _linear(128, 512, seed=102)
    small_inputs = torch.randn(64, 128, device="cuda", dtype=torch.float16)
    main_inputs = torch.randn(512, 128, device="cuda", dtype=torch.float16)

    with torch.inference_mode():
        expected_small = module(small_inputs).clone()
        expected_main = module(main_inputs).clone()
    module.freeze_native_inference(
        device="cuda",
        dtype=torch.float16,
        layouts=("main", "smalln"),
    )
    with torch.inference_mode():
        actual_small = module(small_inputs).clone()
        actual_main = module(main_inputs).clone()

    _assert_native_close(actual_small, expected_small)
    _assert_native_close(actual_main, expected_main)
    assert module.execution_metadata()["native_only_layouts"] == ["main", "smalln"]

    main_only = _linear(128, 512, seed=103)
    main_only.freeze_native_inference(
        device="cuda",
        dtype=torch.float16,
        layouts=("main",),
    )
    with torch.inference_mode(), pytest.raises(
        RuntimeError,
        match="layout 'smalln' was not frozen",
    ):
        main_only(small_inputs)

    small_only = _linear(128, 512, seed=104)
    small_only.freeze_native_inference(
        device="cuda",
        dtype=torch.float16,
        layouts=("smalln",),
    )
    with torch.inference_mode(), pytest.raises(
        RuntimeError,
        match="layout 'main' was not frozen",
    ):
        small_only(main_inputs)


def test_freeze_native_inference_rejects_mismatch_autograd_and_mutation() -> None:
    _require_native_w4a4()
    module = _linear(128, 128, seed=105)
    inputs = torch.randn(512, 128, device="cuda", dtype=torch.float16)
    module.freeze_native_inference(
        device="cuda",
        dtype=torch.float16,
        layouts=("main",),
    )

    with torch.inference_mode(), pytest.raises(RuntimeError, match="input dtype"):
        module(inputs.bfloat16())
    with torch.inference_mode(), pytest.raises(RuntimeError, match="input is not CUDA"):
        module(torch.randn(512, 128, dtype=torch.float16))
    with pytest.raises(RuntimeError, match="autograd"):
        module(inputs)

    with torch.inference_mode():
        module(inputs)
        module._native_only_packed["main"].qweight.view(-1)[0].add_(1)
    with torch.inference_mode(), pytest.raises(RuntimeError, match="packed state was mutated"):
        module(inputs)


def test_frozen_linears_feed_native_gelu_mlp_without_fallback() -> None:
    _require_native_w4a4()
    module = SVDQuantGeluMLP(
        _linear(128, 256, seed=106),
        _linear(256, 128, seed=107),
        approximate="tanh",
    ).eval()
    inputs = torch.randn(1, 64, 128, device="cuda", dtype=torch.float16)

    with torch.inference_mode():
        expected = module(inputs).clone()
    module.fc1.freeze_native_inference(
        device="cuda",
        dtype=torch.float16,
        layouts=("main",),
    )
    module.fc2.freeze_native_inference(
        device="cuda",
        dtype=torch.float16,
        layouts=("main",),
    )
    with torch.inference_mode():
        actual = module(inputs).clone()

    _assert_native_close(actual, expected)
    metadata = module.execution_metadata()
    assert metadata["native_only"] is True
    assert metadata["fused_gelu_mlp_used"] is True
    assert metadata["fallback_reason"] is None

    module.approximate = "none"
    with torch.inference_mode(), pytest.raises(
        RuntimeError,
        match="cannot fall back",
    ):
        module(inputs)


def test_frozen_linears_feed_native_flux_attention() -> None:
    _require_native_w4a4()
    attention = FluxAttention(
        query_dim=128,
        heads=1,
        dim_head=128,
        out_dim=128,
        pre_only=True,
        eps=1e-6,
    ).to(device="cuda", dtype=torch.float16).eval()
    to_qkv = _linear(128, 384, seed=108)
    output_projection = _linear(128, 128, seed=109)
    module = SVDQuantFluxAttention(
        attention,
        to_qkv,
        output_projection=output_projection,
        attention_processor="nunchaku-fp16",
    ).eval()
    hidden = torch.randn(1, 64, 128, device="cuda", dtype=torch.float16)
    cos = torch.randn(64, 128, device="cuda", dtype=torch.float32)
    sin = torch.randn_like(cos)
    rotary = pack_diffusers_flux_rotary_emb(
        (cos, sin),
        hidden_rows=64,
    ).hidden

    with torch.inference_mode():
        expected = module(hidden, image_rotary_emb=rotary).clone()
    to_qkv.freeze_native_inference(
        device="cuda",
        dtype=torch.float16,
        layouts=("main",),
    )
    output_projection.freeze_native_inference(
        device="cuda",
        dtype=torch.float16,
        layouts=("main",),
    )
    with torch.inference_mode():
        actual = module(hidden, image_rotary_emb=rotary).clone()

    _assert_native_close(actual, expected)
    metadata = module.execution_metadata()
    assert metadata["native_qkv_used"] is True
    assert metadata["fallback_reason"] is None
    assert to_qkv.execution_metadata()["native_only"] is True
    assert output_projection.execution_metadata()["native_only"] is True
