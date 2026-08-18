"""Explicit materialization for an inference-only SVDQuant FLUX transformer."""

from __future__ import annotations

from collections.abc import Iterable

from torch import nn

from xqt.runtime.modules.svd_flux_block import (
    SVDQuantFluxSingleTransformerBlock,
    SVDQuantFluxTransformerBlock,
)
from xqt.runtime.modules.svd_flux_transformer import (
    SVDQuantFluxTransformer2DModel,
)


def materialize_svd_flux_transformer(
    source: nn.Module,
    *,
    transformer_blocks: Iterable[SVDQuantFluxTransformerBlock],
    single_transformer_blocks: Iterable[SVDQuantFluxSingleTransformerBlock],
) -> SVDQuantFluxTransformer2DModel:
    """Reuse source FLUX boundary modules with explicit native block lists."""

    return SVDQuantFluxTransformer2DModel(
        source,
        transformer_blocks=transformer_blocks,
        single_transformer_blocks=single_transformer_blocks,
    )


__all__ = ["materialize_svd_flux_transformer"]
