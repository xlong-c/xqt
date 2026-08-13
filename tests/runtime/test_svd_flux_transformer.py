from __future__ import annotations

from types import MethodType
from typing import Any

import pytest
import torch
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from torch import nn

from xqt.runtime import materialize_svd_flux_transformer
from xqt.runtime.modules import (
    SVDQuantFluxRotaryEmb,
    SVDQuantFluxSingleTransformerBlock,
    SVDQuantFluxTransformer2DModel,
    SVDQuantFluxTransformerBlock,
)


class _TimeTextEmbed(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[torch.Tensor, ...]] = []

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        self.calls.append(tuple(input_.clone() for input_ in inputs))
        return torch.zeros(
            inputs[0].shape[0],
            4,
            device=inputs[0].device,
            dtype=inputs[0].dtype,
        )


class _PositionEmbed(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[torch.Tensor] = []

    def forward(self, ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.calls.append(ids.clone())
        rows = int(ids.shape[0])
        cos = torch.arange(rows * 128, dtype=torch.float32).reshape(rows, 128)
        sin = cos + 0.5
        return cos, sin


class _OutputNorm(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[torch.Tensor, torch.Tensor]] = []

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
    ) -> torch.Tensor:
        self.calls.append((hidden_states.clone(), temb.clone()))
        return hidden_states + 5.0


class _Source(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.x_embedder = nn.Identity()
        self.time_text_embed = _TimeTextEmbed()
        self.context_embedder = nn.Identity()
        self.pos_embed = _PositionEmbed()
        self.norm_out = _OutputNorm()
        self.proj_out = nn.Identity()


def _double_block(
    *,
    context_delta: float,
    hidden_delta: float,
) -> SVDQuantFluxTransformerBlock:
    block = SVDQuantFluxTransformerBlock.__new__(SVDQuantFluxTransformerBlock)
    nn.Module.__init__(block)
    block.calls = []

    def _forward(
        self: SVDQuantFluxTransformerBlock,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: SVDQuantFluxRotaryEmb,
        joint_attention_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert joint_attention_kwargs is None
        self.calls.append(
            (
                hidden_states.clone(),
                encoder_hidden_states.clone(),
                temb.clone(),
                image_rotary_emb,
            )
        )
        return encoder_hidden_states + context_delta, hidden_states + hidden_delta

    def _metadata(self: SVDQuantFluxTransformerBlock) -> dict[str, Any]:
        return {
            "scope": "flux_double_stream_transformer_block",
            "native_block_used": bool(self.calls),
        }

    block.forward = MethodType(_forward, block)
    block.execution_metadata = MethodType(_metadata, block)
    return block


def _single_block(*, delta: float) -> SVDQuantFluxSingleTransformerBlock:
    block = SVDQuantFluxSingleTransformerBlock.__new__(
        SVDQuantFluxSingleTransformerBlock
    )
    nn.Module.__init__(block)
    block.calls = []

    def _forward(
        self: SVDQuantFluxSingleTransformerBlock,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: torch.Tensor,
        joint_attention_kwargs: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        assert joint_attention_kwargs is None
        self.calls.append(
            (hidden_states.clone(), temb.clone(), image_rotary_emb.clone())
        )
        return hidden_states + delta

    def _metadata(self: SVDQuantFluxSingleTransformerBlock) -> dict[str, Any]:
        return {
            "scope": "flux_single_stream_transformer_block",
            "native_block_used": bool(self.calls),
        }

    block.forward = MethodType(_forward, block)
    block.execution_metadata = MethodType(_metadata, block)
    return block


def _model(
    *,
    double_blocks: tuple[SVDQuantFluxTransformerBlock, ...] | None = None,
    single_blocks: tuple[SVDQuantFluxSingleTransformerBlock, ...] | None = None,
) -> tuple[SVDQuantFluxTransformer2DModel, _Source]:
    source = _Source()
    model = materialize_svd_flux_transformer(
        source,
        transformer_blocks=(
            _double_block(context_delta=10.0, hidden_delta=100.0),
        )
        if double_blocks is None
        else double_blocks,
        single_transformer_blocks=(_single_block(delta=1000.0),)
        if single_blocks is None
        else single_blocks,
    ).eval()
    return model, source


def _inputs() -> dict[str, torch.Tensor]:
    return {
        "hidden_states": torch.full((1, 2, 4), 2.0),
        "encoder_hidden_states": torch.full((1, 3, 4), 3.0),
        "pooled_projections": torch.full((1, 6), 4.0),
        "timestep": torch.tensor([0.25]),
        "img_ids": torch.arange(6, dtype=torch.float32).reshape(2, 3),
        "txt_ids": torch.arange(9, dtype=torch.float32).reshape(3, 3),
    }


def test_constructor_validates_source_and_block_types() -> None:
    with pytest.raises(TypeError, match="source must provide"):
        SVDQuantFluxTransformer2DModel(
            nn.Identity(),
            transformer_blocks=(_double_block(context_delta=1.0, hidden_delta=1.0),),
            single_transformer_blocks=(_single_block(delta=1.0),),
        )

    source = _Source()
    with pytest.raises(ValueError, match="requires double and single blocks"):
        SVDQuantFluxTransformer2DModel(
            source,
            transformer_blocks=(),
            single_transformer_blocks=(_single_block(delta=1.0),),
        )
    with pytest.raises(TypeError, match="transformer_blocks"):
        SVDQuantFluxTransformer2DModel(
            source,
            transformer_blocks=(nn.Identity(),),  # type: ignore[arg-type]
            single_transformer_blocks=(_single_block(delta=1.0),),
        )
    with pytest.raises(TypeError, match="single_transformer_blocks"):
        SVDQuantFluxTransformer2DModel(
            source,
            transformer_blocks=(_double_block(context_delta=1.0, hidden_delta=1.0),),
            single_transformer_blocks=(nn.Identity(),),  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("return_dict", [True, False])
def test_forward_preserves_flux_token_order_and_returns_only_image_rows(
    monkeypatch: pytest.MonkeyPatch,
    return_dict: bool,
) -> None:
    double = _double_block(context_delta=10.0, hidden_delta=100.0)
    single_first = _single_block(delta=1000.0)
    single_second = _single_block(delta=2000.0)
    model, source = _model(
        double_blocks=(double,),
        single_blocks=(single_first, single_second),
    )
    pack_calls: list[tuple[int, int | None]] = []

    def _pack(
        rotary_emb: tuple[torch.Tensor, torch.Tensor],
        *,
        hidden_rows: int,
        context_rows: int | None = None,
    ) -> SVDQuantFluxRotaryEmb:
        del rotary_emb
        pack_calls.append((hidden_rows, context_rows))
        hidden = torch.full((1, hidden_rows, 128), float(hidden_rows))
        context = (
            None
            if context_rows is None
            else torch.full((1, context_rows, 128), float(context_rows))
        )
        return SVDQuantFluxRotaryEmb(hidden=hidden, context=context)

    monkeypatch.setattr(
        "xqt.runtime.modules.svd_flux_transformer.pack_diffusers_flux_rotary_emb",
        _pack,
    )
    inputs = _inputs()
    with torch.inference_mode():
        output = model(**inputs, return_dict=return_dict)

    sample = output.sample if isinstance(output, Transformer2DModelOutput) else output[0]
    expected = inputs["hidden_states"] + 100.0 + 1000.0 + 2000.0 + 5.0
    torch.testing.assert_close(sample, expected)

    assert pack_calls == [(2, 3), (5, None)]
    assert len(source.pos_embed.calls) == 1
    torch.testing.assert_close(
        source.pos_embed.calls[0],
        torch.cat((inputs["txt_ids"], inputs["img_ids"]), dim=0),
    )
    assert len(double.calls) == 1
    assert len(single_first.calls) == 1
    assert len(single_second.calls) == 1
    expected_single_input = torch.cat(
        (
            inputs["encoder_hidden_states"] + 10.0,
            inputs["hidden_states"] + 100.0,
        ),
        dim=1,
    )
    torch.testing.assert_close(single_first.calls[0][0], expected_single_input)
    torch.testing.assert_close(
        single_second.calls[0][0],
        expected_single_input + 1000.0,
    )
    assert len(source.time_text_embed.calls) == 1
    torch.testing.assert_close(
        source.time_text_embed.calls[0][0],
        inputs["timestep"] * 1000,
    )

    metadata = model.execution_metadata()
    assert metadata["implementation"] == "native_svdq_flux_transformer"
    assert metadata["native_transformer_used"] is True
    assert metadata["all_blocks_native"] is True
    assert metadata["double_block_count"] == 1
    assert metadata["single_block_count"] == 2


def test_guidance_uses_three_argument_conditioning_path() -> None:
    model, source = _model()
    inputs = _inputs()
    guidance = torch.tensor([0.75])

    with torch.inference_mode():
        model(**inputs, guidance=guidance)

    call = source.time_text_embed.calls[0]
    assert len(call) == 3
    torch.testing.assert_close(call[0], inputs["timestep"] * 1000)
    torch.testing.assert_close(call[1], guidance * 1000)
    torch.testing.assert_close(call[2], inputs["pooled_projections"])


def test_forward_rejects_training_autograd_and_unsupported_inputs() -> None:
    model, _ = _model()
    inputs = _inputs()

    model.train()
    with torch.inference_mode(), pytest.raises(RuntimeError, match="requires eval"):
        model(**inputs)

    model.eval()
    with pytest.raises(RuntimeError, match="requires no_grad"):
        model(**inputs)

    unsupported = (
        {"joint_attention_kwargs": {"scale": 1.0}},
        {"controlnet_block_samples": []},
        {"controlnet_single_block_samples": []},
        {"controlnet_blocks_repeat": True},
    )
    for kwargs in unsupported:
        with torch.inference_mode(), pytest.raises(NotImplementedError):
            model(**inputs, **kwargs)

    batched = dict(inputs)
    batched["hidden_states"] = inputs["hidden_states"].expand(2, -1, -1)
    batched["encoder_hidden_states"] = inputs["encoder_hidden_states"].expand(
        2, -1, -1
    )
    with torch.inference_mode(), pytest.raises(ValueError, match="batch=1"):
        model(**batched)


def test_forward_validates_token_ids_and_rotary_contract() -> None:
    model, source = _model()
    inputs = _inputs()
    bad_ids = dict(inputs)
    bad_ids["img_ids"] = torch.zeros(3, 3)
    with torch.inference_mode(), pytest.raises(ValueError, match="must match"):
        model(**bad_ids)

    def _invalid_rotary(_: torch.Tensor) -> torch.Tensor:
        return torch.zeros(1)

    source.pos_embed.forward = _invalid_rotary  # type: ignore[method-assign]
    with torch.inference_mode(), pytest.raises(TypeError, match="cos, sin"):
        model(**inputs)
