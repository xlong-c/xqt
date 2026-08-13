from __future__ import annotations

import pytest
import torch
from diffusers.models.transformers.transformer_flux import FluxAttention
from torch import nn

from xqt.runtime import (
    SVDQuantFluxAttention,
    pack_diffusers_flux_rotary_emb,
)
from xqt.runtime.modules import SVDQuantLinear


def _make_linear(
    input_features: int,
    output_features: int,
    *,
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
) -> SVDQuantLinear:
    base = nn.Linear(
        input_features,
        output_features,
        bias=True,
        dtype=dtype,
        device=device,
    )
    return SVDQuantLinear.from_linear(
        base,
        rank=8,
        group_size=64,
        quant_dtype="int4",
    ).to(device=device, dtype=dtype)


def _make_joint_attention(
    *,
    head_dim: int = 128,
) -> tuple[FluxAttention, SVDQuantLinear, SVDQuantLinear]:
    attention = FluxAttention(
        query_dim=128,
        heads=1,
        dim_head=head_dim,
        added_kv_proj_dim=128,
    ).eval()
    to_qkv = _make_linear(128, 3 * head_dim)
    add_qkv_proj = _make_linear(128, 3 * head_dim)
    return attention, to_qkv, add_qkv_proj


def test_constructor_rejects_non_diffusers_flux_attention() -> None:
    with pytest.raises(TypeError, match="Diffusers FluxAttention"):
        SVDQuantFluxAttention(
            nn.MultiheadAttention(128, 1, batch_first=True),
            _make_linear(128, 384),
        )


def test_constructor_rejects_non_native_head_dimension() -> None:
    attention, _, _ = _make_joint_attention(head_dim=64)
    with pytest.raises(ValueError, match="head_dim=128"):
        SVDQuantFluxAttention(
            attention,
            _make_linear(128, 192),
            add_qkv_proj=_make_linear(128, 192),
        )


def test_constructor_requires_joint_projection_for_joint_attention() -> None:
    attention, to_qkv, _ = _make_joint_attention()
    with pytest.raises(ValueError, match="add_qkv_proj must be provided"):
        SVDQuantFluxAttention(attention, to_qkv)


def test_constructor_rejects_three_independent_projection_contract() -> None:
    attention = FluxAttention(
        query_dim=128,
        heads=1,
        dim_head=128,
        added_kv_proj_dim=None,
    ).eval()
    with pytest.raises(ValueError, match="output_features"):
        SVDQuantFluxAttention(attention, _make_linear(128, 128))


def test_constructor_requires_pre_only_for_explicit_output_projection() -> None:
    attention, to_qkv, add_qkv_proj = _make_joint_attention()
    with pytest.raises(ValueError, match="only valid for pre_only"):
        SVDQuantFluxAttention(
            attention,
            to_qkv,
            add_qkv_proj=add_qkv_proj,
            output_projection=_make_linear(128, 128),
        )


def test_rotary_helper_preserves_context_then_hidden_order() -> None:
    context_rows = 256
    hidden_rows = 64
    cos = torch.randn(context_rows + hidden_rows, 128, dtype=torch.float32)
    sin = torch.randn_like(cos)

    packed = pack_diffusers_flux_rotary_emb(
        (cos, sin),
        hidden_rows=hidden_rows,
        context_rows=context_rows,
    )

    assert packed.context is not None
    assert packed.context.shape == (1, 256, 128)
    assert packed.hidden.shape == (1, 256, 128)
    assert packed.context.dtype == torch.float32
    assert packed.hidden.dtype == torch.float32


def test_rotary_helper_rejects_wrong_dtype_and_length() -> None:
    cos = torch.randn(320, 128, dtype=torch.float16)
    sin = torch.randn_like(cos)
    with pytest.raises(ValueError, match="float32"):
        pack_diffusers_flux_rotary_emb(
            (cos, sin),
            hidden_rows=64,
            context_rows=256,
        )

    cos32 = cos.float()
    sin32 = sin.float()
    with pytest.raises(ValueError, match="sequence length"):
        pack_diffusers_flux_rotary_emb(
            (cos32[:-1], sin32[:-1]),
            hidden_rows=64,
            context_rows=256,
        )


def test_native_path_rejects_training_before_cuda_dispatch() -> None:
    attention, to_qkv, add_qkv_proj = _make_joint_attention()
    module = SVDQuantFluxAttention(
        attention,
        to_qkv,
        add_qkv_proj=add_qkv_proj,
    )
    module.train()
    hidden = torch.randn(1, 2, 128)
    context = torch.randn(1, 2, 128)
    packed = pack_diffusers_flux_rotary_emb(
        (
            torch.randn(4, 128),
            torch.randn(4, 128),
        ),
        hidden_rows=2,
        context_rows=2,
    )

    with pytest.raises(RuntimeError, match="requires eval mode"):
        module(hidden, context, image_rotary_emb=packed)
    assert module.execution_metadata()["native_qkv_used"] is False


def test_native_path_rejects_autograd_before_cuda_dispatch() -> None:
    attention, to_qkv, add_qkv_proj = _make_joint_attention()
    module = SVDQuantFluxAttention(
        attention,
        to_qkv,
        add_qkv_proj=add_qkv_proj,
    ).eval()
    hidden = torch.randn(1, 2, 128, requires_grad=True)
    context = torch.randn(1, 2, 128)
    packed = pack_diffusers_flux_rotary_emb(
        (
            torch.randn(4, 128),
            torch.randn(4, 128),
        ),
        hidden_rows=2,
        context_rows=2,
    )

    with pytest.raises(RuntimeError, match="requires no_grad"):
        module(hidden, context, image_rotary_emb=packed)
    assert module.execution_metadata()["native_qkv_used"] is False


def _require_native_w4a4() -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 9):
        pytest.skip("native sm_89 W4A4 backend unavailable")
    try:
        from xqt.operator_opt.kernels.cute.svdq_w4a4_sm89 import (
            native_w4a4_available,
        )
    except Exception:
        pytest.skip("native sm_89 W4A4 backend unavailable")
    if not native_w4a4_available(build=False):
        pytest.skip("native sm_89 W4A4 backend unavailable")


def test_cuda_joint_forward_reports_native_qkv_and_shapes() -> None:
    _require_native_w4a4()
    dtype = torch.float16
    device = "cuda"
    attention = FluxAttention(
        query_dim=128,
        heads=1,
        dim_head=128,
        added_kv_proj_dim=128,
    ).to(device=device, dtype=dtype).eval()
    to_qkv = _make_linear(128, 384, dtype=dtype, device=device)
    add_qkv_proj = _make_linear(128, 384, dtype=dtype, device=device)
    module = SVDQuantFluxAttention(
        attention,
        to_qkv,
        add_qkv_proj=add_qkv_proj,
    ).to(device=device, dtype=dtype).eval()
    hidden = torch.randn(1, 2, 128, device=device, dtype=dtype)
    context = torch.randn(1, 2, 128, device=device, dtype=dtype)
    cos = torch.randn(4, 128, device=device, dtype=torch.float32)
    sin = torch.randn_like(cos)
    rotary = pack_diffusers_flux_rotary_emb(
        (cos, sin),
        hidden_rows=2,
        context_rows=2,
    )

    with torch.inference_mode():
        hidden_output, context_output = module(
            hidden,
            context,
            image_rotary_emb=rotary,
        )

    assert hidden_output.shape == hidden.shape
    assert context_output.shape == context.shape
    metadata = module.execution_metadata()
    assert metadata["native_qkv_used"] is True
    assert metadata["implementation"] == "native_svdq_flux_attention_sdpa"
    assert metadata["fallback_reason"] is None


def test_cuda_joint_nunchaku_fp16_forward_reports_shapes() -> None:
    _require_native_w4a4()
    dtype = torch.float16
    device = "cuda"
    attention = FluxAttention(
        query_dim=128,
        heads=1,
        dim_head=128,
        added_kv_proj_dim=128,
    ).to(device=device, dtype=dtype).eval()
    module = SVDQuantFluxAttention(
        attention,
        _make_linear(128, 384, dtype=dtype, device=device),
        add_qkv_proj=_make_linear(128, 384, dtype=dtype, device=device),
        attention_processor="nunchaku-fp16",
    ).to(device=device, dtype=dtype).eval()
    hidden = torch.randn(1, 2, 128, device=device, dtype=dtype)
    context = torch.randn(1, 2, 128, device=device, dtype=dtype)
    cos = torch.randn(4, 128, device=device, dtype=torch.float32)
    sin = torch.randn_like(cos)
    rotary = pack_diffusers_flux_rotary_emb(
        (cos, sin),
        hidden_rows=2,
        context_rows=2,
    )

    with torch.inference_mode():
        hidden_output, context_output = module(
            hidden,
            context,
            image_rotary_emb=rotary,
        )

    assert hidden_output.shape == hidden.shape
    assert context_output.shape == context.shape
    metadata = module.execution_metadata()
    assert metadata["native_qkv_used"] is True
    assert metadata["implementation"] == "native_svdq_flux_attention_nunchaku_fp16"
    assert metadata["attention_processor"] == "nunchaku-fp16"
    assert metadata["fallback_reason"] is None


def test_cuda_single_nunchaku_fp16_forward_reports_shapes() -> None:
    _require_native_w4a4()
    dtype = torch.float16
    device = "cuda"
    attention = FluxAttention(
        query_dim=128,
        heads=1,
        dim_head=128,
        out_dim=128,
        pre_only=True,
        eps=1e-6,
    ).to(device=device, dtype=dtype).eval()
    module = SVDQuantFluxAttention(
        attention,
        _make_linear(128, 384, dtype=dtype, device=device),
        output_projection=_make_linear(128, 128, dtype=dtype, device=device),
        attention_processor="nunchaku-fp16",
    ).to(device=device, dtype=dtype).eval()
    hidden = torch.randn(1, 64, 128, device=device, dtype=dtype)
    cos = torch.randn(64, 128, device=device, dtype=torch.float32)
    sin = torch.randn_like(cos)
    rotary = pack_diffusers_flux_rotary_emb(
        (cos, sin),
        hidden_rows=64,
    )

    with torch.inference_mode():
        output = module(hidden, image_rotary_emb=rotary.hidden)

    assert output.shape == hidden.shape
    metadata = module.execution_metadata()
    assert metadata["native_qkv_used"] is True
    assert metadata["implementation"] == "native_svdq_flux_attention_nunchaku_fp16"
    assert metadata["attention_processor"] == "nunchaku-fp16"
    assert metadata["joint_attention"] is False
    assert metadata["fallback_reason"] is None
