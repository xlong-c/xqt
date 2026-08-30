from __future__ import annotations

from types import MethodType

import pytest
import torch
from diffusers.models.transformers.transformer_flux import FluxAttention
from torch import nn

from tests.xqt.svd_test_helpers import make_legacy_svd_linear
from xqt.kernels.ops.gemm import PackedWeight, PackedWeightMetadata
from xqt.runtime import pack_diffusers_flux_rotary_emb
from xqt.runtime.modules import (
    AWQW4A16Linear,
    SVDQuantAdaLayerNormZeroSingle,
    SVDQuantFluxAttention,
    SVDQuantFluxSingleTransformerBlock,
    SVDQuantGeluMLP,
    SVDQuantLinear,
)


def _awq_weight(
    *,
    dim: int,
    fields: int,
    device: str,
) -> PackedWeight:
    output_features = fields * dim
    generator = torch.Generator(device=device)
    generator.manual_seed(907)
    codes = torch.randint(
        0,
        16,
        (output_features, dim),
        generator=generator,
        device=device,
        dtype=torch.uint8,
    )
    qweight = (codes[:, 0::2] | (codes[:, 1::2] << 4)).contiguous()
    scales = (
        torch.rand(
            output_features,
            dim // 64,
            generator=generator,
            device=device,
        )
        * 0.002
        + 0.001
    )
    zero_points = torch.full_like(scales, 8.0)
    return PackedWeight(
        qweight=qweight,
        scales=scales,
        zero_points=zero_points,
        metadata=PackedWeightMetadata(
            logical_shape=(output_features, dim),
            storage_layout="xqt_int4_nk_v1",
            pack_version="xqt-w4a16-awq-v1",
            weight_dtype="int4",
            padded_k=dim,
            group_size=64,
            packed_bits=4,
            nibble_order="low_high",
            nibble_signed=False,
        ),
    )


def _awq_linear_stub(*, dim: int, fields: int) -> AWQW4A16Linear:
    """Build a type-correct AWQ test double without invoking CUDA prepack."""

    linear = AWQW4A16Linear.__new__(AWQW4A16Linear)
    nn.Module.__init__(linear)
    linear.input_features = dim
    linear.output_features = fields * dim
    linear.group_size = 64
    linear._last_execution = {
        "implementation": "test_double",
        "native": False,
    }
    return linear


def _svd_linear(
    input_features: int,
    output_features: int,
    *,
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
) -> SVDQuantLinear:
    linear = nn.Linear(
        input_features,
        output_features,
        bias=True,
        dtype=dtype,
        device=device,
    )
    return make_legacy_svd_linear(
        linear,
        rank=8,
        group_size=64,
        quant_dtype="int4",
    ).to(device=device, dtype=dtype)


def _single_block(
    *,
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
    attention_processor: str = "flashattn2",
) -> SVDQuantFluxSingleTransformerBlock:
    dim = 128
    hidden_features = 512
    attention_base = FluxAttention(
        query_dim=dim,
        heads=1,
        dim_head=128,
        out_dim=dim,
        pre_only=True,
        eps=1e-6,
    ).to(device=device, dtype=dtype).eval()
    attention = SVDQuantFluxAttention(
        attention_base,
        _svd_linear(dim, 3 * dim, dtype=dtype, device=device),
        output_projection=_svd_linear(dim, dim, dtype=dtype, device=device),
        attention_processor=attention_processor,
    ).eval()
    feed_forward = SVDQuantGeluMLP(
        _svd_linear(dim, hidden_features, dtype=dtype, device=device),
        _svd_linear(hidden_features, dim, dtype=dtype, device=device),
        approximate="tanh",
    ).eval()
    bias = torch.randn(3 * dim, device=device, dtype=dtype) * 0.02
    bias.view(dim, 3)[:, 1].add_(1.0)
    modulation_linear = (
        AWQW4A16Linear(
            _awq_weight(dim=dim, fields=3, device=device),
            bias=bias,
        )
        if device == "cuda"
        else _awq_linear_stub(dim=dim, fields=3)
    )
    norm = SVDQuantAdaLayerNormZeroSingle(
        nn.LayerNorm(
            dim,
            elementwise_affine=False,
            eps=1e-6,
            device=device,
            dtype=dtype,
        ),
        modulation_linear,
        scale_shift=0.0,
    ).eval()
    return SVDQuantFluxSingleTransformerBlock(
        norm=norm,
        attention=attention,
        feed_forward=feed_forward,
    ).eval()


def test_adaln_zero_single_uses_interleaved_layout_and_scale_shift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dim = 64
    linear = _awq_linear_stub(dim=dim, fields=3)
    shift = torch.full((1, dim), 2.0)
    scale = torch.full((1, dim), 3.0)
    gate = torch.full((1, dim), 4.0)
    modulation = torch.stack((shift, scale, gate), dim=-1).reshape(1, 3 * dim)

    def _forward(_: AWQW4A16Linear, inputs: torch.Tensor) -> torch.Tensor:
        return modulation.to(device=inputs.device, dtype=inputs.dtype)

    monkeypatch.setattr(AWQW4A16Linear, "forward", _forward)
    module = SVDQuantAdaLayerNormZeroSingle(
        nn.Identity(),
        linear,
        silu=nn.Identity(),
        scale_shift=1.0,
    )
    inputs = torch.randn(1, 2, dim)
    normalized, actual_gate = module(inputs, emb=torch.randn(1, dim))

    torch.testing.assert_close(normalized, inputs * 4.0 + 2.0)
    torch.testing.assert_close(actual_gate, gate)


def test_single_block_combines_attention_and_mlp_before_gate() -> None:
    block = _single_block()

    def _norm_forward(
        _: SVDQuantAdaLayerNormZeroSingle,
        inputs: torch.Tensor,
        *,
        emb: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del emb
        return inputs, torch.full(
            (inputs.shape[0], inputs.shape[-1]),
            2.0,
            device=inputs.device,
            dtype=inputs.dtype,
        )

    def _attention_forward(
        _: SVDQuantFluxAttention,
        hidden_states: torch.Tensor,
        **kwargs: object,
    ) -> torch.Tensor:
        del kwargs
        return torch.full_like(hidden_states, 3.0)

    def _ff_forward(
        _: SVDQuantGeluMLP,
        hidden_states: torch.Tensor,
        *args: object,
        **kwargs: object,
    ) -> torch.Tensor:
        del args, kwargs
        return torch.full_like(hidden_states, 5.0)

    block.norm.forward = MethodType(_norm_forward, block.norm)
    block.attn.forward = MethodType(_attention_forward, block.attn)
    block.ff.forward = MethodType(_ff_forward, block.ff)
    inputs = torch.randn(1, 2, 128)
    with torch.inference_mode():
        output = block(
            inputs,
            torch.randn(1, 128),
            torch.randn(1, 256, 128),
        )

    torch.testing.assert_close(output, inputs + 16.0)
    assert block.execution_metadata()["native_block_used"] is True


def test_single_block_rejects_training_and_autograd() -> None:
    block = _single_block()
    inputs = torch.randn(1, 2, 128)
    temb = torch.randn(1, 128)
    rotary = torch.randn(1, 256, 128)

    block.train()
    with torch.inference_mode(), pytest.raises(RuntimeError, match="requires eval"):
        block(inputs, temb, rotary)

    block.eval()
    with pytest.raises(RuntimeError, match="requires no_grad"):
        block(inputs, temb, rotary)


def _require_native_w4a4() -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 9):
        pytest.skip("native sm_89 W4A4 backend unavailable")
    from xqt.kernels.ops._impl.cute.svdq_w4a4_sm89 import (
        native_w4a4_available,
    )

    if not native_w4a4_available(build=False):
        pytest.skip("native sm_89 W4A4 backend unavailable")


@pytest.mark.parametrize("attention_processor", ["flashattn2", "nunchaku-fp16"])
def test_cuda_single_block_reports_all_native_paths(
    attention_processor: str,
) -> None:
    _require_native_w4a4()
    block = _single_block(
        dtype=torch.float16,
        device="cuda",
        attention_processor=attention_processor,
    )
    inputs = torch.randn(1, 64, 128, device="cuda", dtype=torch.float16)
    temb = torch.randn(1, 128, device="cuda", dtype=torch.float16)
    cos = torch.randn(64, 128, device="cuda", dtype=torch.float32)
    sin = torch.randn_like(cos)
    rotary = pack_diffusers_flux_rotary_emb(
        (cos, sin),
        hidden_rows=64,
    ).hidden

    with torch.inference_mode():
        output = block(inputs, temb, rotary)

    assert output.shape == inputs.shape
    metadata = block.execution_metadata()
    assert metadata["native_block_used"] is True
    assert metadata["norm"]["linear"]["native"] is True
    assert metadata["attention"]["native_qkv_used"] is True
    assert metadata["attention"]["attention_processor"] == attention_processor
    assert metadata["feed_forward"]["fused_gelu_mlp_used"] is True
