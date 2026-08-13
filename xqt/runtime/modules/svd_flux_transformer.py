"""Inference-only classic FLUX transformer over native XQT SVDQuant blocks."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from torch import nn

from xqt.runtime.svd_flux_attention import pack_diffusers_flux_rotary_emb

from .svd_flux_block import (
    SVDQuantFluxSingleTransformerBlock,
    SVDQuantFluxTransformerBlock,
)


class SVDQuantFluxTransformer2DModel(nn.Module):
    """Execute one classic FLUX transformer with materialized SVDQuant blocks.

    The wrapper reuses the source model's dense input, conditioning, RoPE and
    output modules. Double-stream and single-stream blocks are supplied
    explicitly so model materialization cannot silently guess checkpoint or
    quantization layouts. The path is forward-only and currently assumes the
    batch-1 contract enforced by :class:`SVDQuantFluxAttention`.
    """

    def __init__(
        self,
        source: nn.Module,
        *,
        transformer_blocks: Iterable[SVDQuantFluxTransformerBlock],
        single_transformer_blocks: Iterable[SVDQuantFluxSingleTransformerBlock],
    ) -> None:
        super().__init__()
        required_modules = (
            "x_embedder",
            "time_text_embed",
            "context_embedder",
            "pos_embed",
            "norm_out",
            "proj_out",
        )
        missing = [
            name
            for name in required_modules
            if not isinstance(getattr(source, name, None), nn.Module)
        ]
        if missing:
            raise TypeError(
                "source must provide FLUX transformer modules: "
                + ", ".join(missing)
            )

        double_blocks = tuple(transformer_blocks)
        single_blocks = tuple(single_transformer_blocks)
        if not double_blocks or not single_blocks:
            raise ValueError(
                "classic FLUX materialization requires double and single blocks"
            )
        if any(
            not isinstance(block, SVDQuantFluxTransformerBlock)
            for block in double_blocks
        ):
            raise TypeError(
                "transformer_blocks must contain SVDQuantFluxTransformerBlock"
            )
        if any(
            not isinstance(block, SVDQuantFluxSingleTransformerBlock)
            for block in single_blocks
        ):
            raise TypeError(
                "single_transformer_blocks must contain "
                "SVDQuantFluxSingleTransformerBlock"
            )

        self.x_embedder = source.x_embedder
        self.time_text_embed = source.time_text_embed
        self.context_embedder = source.context_embedder
        self.pos_embed = source.pos_embed
        self.norm_out = source.norm_out
        self.proj_out = source.proj_out
        self.transformer_blocks = nn.ModuleList(double_blocks)
        self.single_transformer_blocks = nn.ModuleList(single_blocks)
        self._last_native_used = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        pooled_projections: torch.Tensor,
        timestep: torch.Tensor,
        img_ids: torch.Tensor,
        txt_ids: torch.Tensor,
        guidance: torch.Tensor | None = None,
        joint_attention_kwargs: dict[str, Any] | None = None,
        controlnet_block_samples: list[torch.Tensor] | None = None,
        controlnet_single_block_samples: list[torch.Tensor] | None = None,
        return_dict: bool = True,
        controlnet_blocks_repeat: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor] | Transformer2DModelOutput:
        if self.training:
            raise RuntimeError("SVDQuant FLUX transformer requires eval mode")
        if torch.is_grad_enabled():
            raise RuntimeError(
                "SVDQuant FLUX transformer requires no_grad or inference_mode"
            )
        if joint_attention_kwargs:
            raise NotImplementedError("joint_attention_kwargs are not supported")
        if controlnet_block_samples is not None:
            raise NotImplementedError("ControlNet double-block residuals are unsupported")
        if controlnet_single_block_samples is not None:
            raise NotImplementedError("ControlNet single-block residuals are unsupported")
        if controlnet_blocks_repeat:
            raise NotImplementedError("ControlNet block repetition is unsupported")
        if hidden_states.ndim != 3 or encoder_hidden_states.ndim != 3:
            raise ValueError("FLUX hidden and encoder states must be three-dimensional")
        if int(hidden_states.shape[0]) != 1 or int(encoder_hidden_states.shape[0]) != 1:
            raise ValueError("native SVDQuant FLUX transformer currently requires batch=1")

        hidden_states = self.x_embedder(hidden_states)
        timestep = timestep.to(hidden_states.dtype) * 1000
        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000
        temb = (
            self.time_text_embed(timestep, pooled_projections)
            if guidance is None
            else self.time_text_embed(timestep, guidance, pooled_projections)
        )
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        if txt_ids.ndim == 3:
            txt_ids = txt_ids[0]
        if img_ids.ndim == 3:
            img_ids = img_ids[0]
        if txt_ids.ndim != 2 or img_ids.ndim != 2:
            raise ValueError("FLUX txt_ids and img_ids must be two-dimensional")

        text_rows = int(encoder_hidden_states.shape[1])
        hidden_rows = int(hidden_states.shape[1])
        if int(txt_ids.shape[0]) != text_rows or int(img_ids.shape[0]) != hidden_rows:
            raise ValueError("FLUX token IDs must match encoder and hidden row counts")
        ids = torch.cat((txt_ids, img_ids), dim=0)
        raw_rotary = self.pos_embed(ids)
        if not isinstance(raw_rotary, tuple) or len(raw_rotary) != 2:
            raise TypeError("source FLUX pos_embed must return a (cos, sin) tuple")
        joint_rotary = pack_diffusers_flux_rotary_emb(
            raw_rotary,
            hidden_rows=hidden_rows,
            context_rows=text_rows,
        )
        single_rotary = pack_diffusers_flux_rotary_emb(
            raw_rotary,
            hidden_rows=hidden_rows + text_rows,
        ).hidden

        for block in self.transformer_blocks:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=joint_rotary,
            )

        hidden_states = torch.cat((encoder_hidden_states, hidden_states), dim=1)
        for block in self.single_transformer_blocks:
            hidden_states = block(
                hidden_states=hidden_states,
                temb=temb,
                image_rotary_emb=single_rotary,
            )

        hidden_states = hidden_states[:, text_rows:]
        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)
        self._last_native_used = True
        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)

    def execution_metadata(self) -> dict[str, Any]:
        """Describe the last full-transformer execution and every block scope."""

        double_metadata = [
            block.execution_metadata() for block in self.transformer_blocks
        ]
        single_metadata = [
            block.execution_metadata() for block in self.single_transformer_blocks
        ]
        all_blocks_native = all(
            bool(item["native_block_used"])
            for item in (*double_metadata, *single_metadata)
        )
        return {
            "implementation": (
                "native_svdq_flux_transformer"
                if self._last_native_used and all_blocks_native
                else "not_run_svdq_flux_transformer"
            ),
            "scope": "classic_flux_transformer",
            "inference_only": True,
            "native_transformer_used": self._last_native_used,
            "all_blocks_native": all_blocks_native,
            "double_block_count": len(double_metadata),
            "single_block_count": len(single_metadata),
            "double_blocks": double_metadata,
            "single_blocks": single_metadata,
        }


__all__ = ["SVDQuantFluxTransformer2DModel"]
