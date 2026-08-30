from __future__ import annotations

from typing import Any

import torch
from torch import nn

from xqt.contracts import CompositeAddLinear
from xqt.compression.quant.quantizers.svd import quantize_with_svd
from xqt.runtime.modules import SVDQuantInt8MmaLinear, SVDQuantLinear


def make_svd_artifact(
    module: nn.Linear,
    *,
    rank: int,
    group_size: int = 128,
    quant_dtype: str = "int4",
) -> CompositeAddLinear:
    """Create the canonical SVD storage artifact for runtime tests."""

    result = quantize_with_svd(
        nn.Sequential(module),
        strategy=f"w4a16_{quant_dtype}",
        compute="dequant_fp16",
        policy={"materialize_compute": False},
        rank=rank,
        group_size=group_size,
        quant_dtype=quant_dtype,
        inplace=False,
        collect_analysis=False,
    )
    artifact = result.model[0]
    if not isinstance(artifact, CompositeAddLinear):
        raise TypeError("SVD quantizer did not return a CompositeAddLinear artifact")
    return artifact


def make_legacy_svd_linear(
    module: nn.Linear,
    *,
    rank: int,
    group_size: int = 128,
    quant_dtype: str = "int4",
) -> SVDQuantLinear:
    """Materialize the legacy W4A4 shell from a canonical artifact."""

    return SVDQuantLinear.from_composite(
        make_svd_artifact(
            module,
            rank=rank,
            group_size=group_size,
            quant_dtype=quant_dtype,
        )
    )


def make_legacy_svd_int8(
    module: nn.Linear,
    *,
    rank: int,
    group_size: int = 128,
    quant_dtype: str = "int4",
    engine: str = "auto",
    fallback_engine: str = "torch_int_mm",
    activation_scale_mode: str = "dynamic",
    activation_scale: torch.Tensor | float | None = None,
    cache_int8_compute_view: bool = True,
    **kwargs: Any,
) -> SVDQuantInt8MmaLinear:
    """Materialize the legacy W8A8 shell from a canonical artifact."""

    return SVDQuantInt8MmaLinear.from_composite(
        make_svd_artifact(
            module,
            rank=rank,
            group_size=group_size,
            quant_dtype=quant_dtype,
        ),
        engine=engine,
        fallback_engine=fallback_engine,
        activation_scale_mode=activation_scale_mode,
        activation_scale=activation_scale,
        cache_int8_compute_view=cache_int8_compute_view,
        **kwargs,
    )


__all__ = [
    "make_legacy_svd_int8",
    "make_legacy_svd_linear",
    "make_svd_artifact",
]
