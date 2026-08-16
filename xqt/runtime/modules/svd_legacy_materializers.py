"""Compute materializers for legacy SVDQuant runtime shells."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from xqt.runtime.composite_branch import register_materializer
from xqt.runtime.modules.svd_fp8_legacy import SVDQuantFp8Linear
from xqt.runtime.modules.svd_w4a4_legacy import SVDQuantLinear
from xqt.runtime.modules.svd_w8a8_legacy import SVDQuantInt8MmaLinear


@register_materializer(
    "svd_low_rank_plus_residual",
    mode="collapse",
    compute_precision="w8a8",
)
def _build_svd_collapsed_int8(shell: SVDQuantLinear, **options: Any) -> nn.Module:
    """Merge low-rank + residual into one per-channel INT8 GEMM."""

    from xqt.runtime.modules.int8_mma_linear import Int8MmaLinear
    from xqt.runtime.modules.w4_storage_int8_mma_linear import (
        _channel_int8_from_float_weight,
    )

    meta: Mapping[str, Any] = options["meta"]
    eps = float(meta.get("eps", 1e-6))
    dense_weight = shell.full_weight_dequant()
    qweight_t, channel_scale = _channel_int8_from_float_weight(dense_weight, eps=eps)
    return Int8MmaLinear(
        qweight_t,
        channel_scale,
        bias=None if shell.bias is None else shell.bias.detach(),
        input_features=shell.input_features,
        output_features=shell.output_features,
        engine=str(options["engine"]),
        fallback_engine=str(options["fallback"]),
        block_m=int(meta.get("block_m", 64)),
        block_n=int(meta.get("block_n", 64)),
        block_k=int(meta.get("block_k", 64)),
        threads=int(meta.get("threads", 128)),
        num_stages=int(meta.get("num_stages", 2)),
        output_dtype=shell.down_proj.weight.dtype,
        activation_scale_mode=str(options["act_mode"]),
        activation_scale=options.get("act_scale"),
        activation_quant_block_size=int(meta.get("activation_quant_block_size", 256)),
        eps=eps,
        min_fp8_rows=int(options["min_rows"]),
    )


@register_materializer(
    "svd_low_rank_plus_residual",
    mode="collapse",
    compute_precision="fp8",
)
def _build_svd_collapsed_fp8(shell: SVDQuantLinear, **options: Any) -> nn.Module:
    """Fold low-rank + residual into one tensorwise FP8 Linear."""

    from xqt.runtime.modules.fp8_mma_linear import Fp8MmaLinear

    meta: Mapping[str, Any] = options["meta"]
    source_dtype = shell.down_proj.weight.dtype
    output_dtype = (
        source_dtype
        if source_dtype in {torch.float16, torch.bfloat16}
        else torch.float16
    )
    dense_weight = shell.full_weight_dequant()
    return Fp8MmaLinear.from_dense_weight(
        dense_weight,
        bias=None if shell.bias is None else shell.bias.detach(),
        input_features=shell.input_features,
        output_features=shell.output_features,
        output_dtype=output_dtype,
        activation_scale_mode=str(options["act_mode"]),
        activation_scale=options.get("act_scale"),
        min_fp8_rows=int(options["min_rows"]),
        eps=float(meta.get("eps", 1e-8)),
    )


@register_materializer(
    "svd_low_rank_plus_residual",
    mode="split",
    compute_precision="fp8",
)
def _build_svd_split_fp8(shell: SVDQuantLinear, **options: Any) -> nn.Module:
    """Keep source-precision low-rank branch plus FP8 residual."""

    meta: Mapping[str, Any] = options["meta"]
    source_dtype = shell.down_proj.weight.dtype
    output_dtype = (
        source_dtype
        if source_dtype in {torch.float16, torch.bfloat16}
        else torch.float16
    )
    return SVDQuantFp8Linear(
        down_weight=shell.down_proj.weight.detach(),
        up_weight=shell.up_proj.weight.detach(),
        residual_weight=shell.dequantize_residual(),
        bias=None if shell.bias is None else shell.bias.detach(),
        input_features=shell.input_features,
        output_features=shell.output_features,
        output_dtype=output_dtype,
        activation_scale_mode=str(options["act_mode"]),
        activation_scale=options.get("act_scale"),
        min_int8_rows=int(options["min_rows"]),
        eps=float(meta.get("eps", 1e-8)),
    )


@register_materializer(
    "svd_low_rank_plus_residual",
    mode="split",
    compute_precision="w8a8",
)
def _build_svd_split_int8(shell: SVDQuantLinear, **options: Any) -> nn.Module:
    """Keep source-precision low-rank branch plus W4-storage INT8 MMA."""

    meta: Mapping[str, Any] = options["meta"]
    residual_dtype = str(
        meta.get("quant_dtype", getattr(shell, "quant_dtype", "fp4"))
    ).lower()
    return SVDQuantInt8MmaLinear(
        down_weight=shell.down_proj.weight.detach(),
        up_weight=shell.up_proj.weight.detach(),
        packed_residual=shell.packed_residual.detach(),
        residual_scale=shell.residual_scale.detach(),
        bias=None if shell.bias is None else shell.bias.detach(),
        input_features=shell.input_features,
        output_features=shell.output_features,
        group_size=shell.group_size,
        padded_input_features=shell.padded_input_features,
        output_dtype=shell.down_proj.weight.dtype,
        quant_dtype=residual_dtype,
        engine=str(options["engine"]),
        fallback_engine=str(options["fallback"]),
        block_m=int(meta.get("block_m", 64)),
        block_n=int(meta.get("block_n", 64)),
        block_k=int(meta.get("block_k", 64)),
        threads=int(meta.get("threads", 128)),
        num_stages=int(meta.get("num_stages", 2)),
        activation_scale_mode=str(options["act_mode"]),
        activation_scale=options.get("act_scale"),
        activation_quant_block_size=int(meta.get("activation_quant_block_size", 256)),
        eps=float(meta.get("eps", 1e-6)),
        cache_int8_compute_view=bool(meta.get("cache_int8_compute_view", True)),
        min_int8_rows=int(options["min_rows"]),
    )


__all__ = [
    "_build_svd_collapsed_fp8",
    "_build_svd_collapsed_int8",
    "_build_svd_split_fp8",
    "_build_svd_split_int8",
]
