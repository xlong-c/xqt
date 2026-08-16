"""Runtime materializers for additive composite artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from xqt.contracts import (
    ModuleComputeSpec,
    normalize_compute_contract,
    normalize_compute_precision,
)
from xqt.contracts.composite import CompositeAddLinear, CompositeAddModule


def materialize_composite_w4a4(
    module: CompositeAddLinear,
    *,
    native_fusion: bool = True,
    layout: str = "main",
) -> nn.Module:
    """Materialize one canonical artifact as a W4A4 runtime executor."""

    if not isinstance(module, CompositeAddLinear):
        raise TypeError("module must be a CompositeAddLinear artifact")
    from xqt.runtime.modules.composite_add_w4a4 import CompositeAddW4A4Linear

    return CompositeAddW4A4Linear.from_composite(
        module,
        native_fusion=native_fusion,
        layout=layout,
    )


def materialize_composite_compute(
    module: CompositeAddLinear,
    spec: ModuleComputeSpec | Mapping[str, Any],
) -> nn.Module:
    """Bind a runtime compute view to one canonical composite artifact."""

    if not isinstance(module, CompositeAddLinear):
        raise TypeError("module must be a CompositeAddLinear artifact")
    compute_spec = (
        spec
        if isinstance(spec, ModuleComputeSpec)
        else ModuleComputeSpec.from_mapping(spec)
    )
    if normalize_compute_contract(compute_spec.compute_contract) != "composite_add":
        return module
    if compute_spec.precision is None:
        return module

    precision = normalize_compute_precision(compute_spec.precision)
    if precision == "w4a4" and compute_spec.preferred_mode == "fused":
        layout = str(compute_spec.metadata.get("w4a4_layout", "main"))
        return materialize_composite_w4a4(module, layout=layout)

    if precision == "fp8":
        from xqt.runtime.modules.composite_add_fp8 import CompositeAddFp8Linear
        from xqt.runtime.modules.fp8_mma_linear import Fp8MmaLinear

        metadata = dict(compute_spec.metadata)
        execution = compute_spec.execution
        output_dtype = (
            module.down_proj.weight.dtype
            if module.down_proj.weight.dtype in {torch.float16, torch.bfloat16}
            else torch.float16
        )
        activation_scale_mode = str(
            execution.activation_scale_mode
            if execution.activation_scale_mode != "dynamic"
            else metadata.get(
                "activation_scale_mode",
                module._pending_activation_scale_mode,
            )
        )
        activation_scale = metadata.get(
            "activation_scale",
            module._pending_activation_scale,
        )
        min_fp8_rows = int(metadata.get("min_fp8_rows", 0))
        eps = float(metadata.get("eps", 1e-8))
        if (compute_spec.preferred_mode or "split") == "collapse":
            return Fp8MmaLinear.from_dense_weight(
                module.full_weight_dequant(),
                bias=None if module.bias is None else module.bias.detach(),
                input_features=module.input_features,
                output_features=module.output_features,
                output_dtype=output_dtype,
                activation_scale_mode=activation_scale_mode,
                activation_scale=activation_scale,
                min_fp8_rows=min_fp8_rows,
                eps=eps,
            )
        return CompositeAddFp8Linear.from_composite(
            module,
            output_dtype=output_dtype,
            activation_scale_mode=activation_scale_mode,
            activation_scale=activation_scale,
            min_fp8_rows=min_fp8_rows,
            eps=eps,
        )

    if precision != "w8a8":
        return module
    if (compute_spec.preferred_mode or "split") != "split":
        return module

    from xqt.runtime.modules.svd_w8a8_legacy import SVDQuantInt8MmaLinear

    metadata = dict(compute_spec.metadata)
    execution = compute_spec.execution
    activation_scale_mode = str(
        execution.activation_scale_mode
        if execution.activation_scale_mode != "dynamic"
        else metadata.get(
            "activation_scale_mode",
            module._pending_activation_scale_mode,
        )
    )
    activation_scale = metadata.get(
        "activation_scale",
        module._pending_activation_scale,
    )
    preferred_engines = list(compute_spec.preferred_engines)
    return SVDQuantInt8MmaLinear.from_composite(
        module,
        output_dtype=module.down_proj.weight.dtype,
        engine=preferred_engines[0] if preferred_engines else "auto",
        fallback_engine=str(metadata.get("fallback_engine", "torch_int_mm")),
        block_m=int(metadata.get("block_m", 64)),
        block_n=int(metadata.get("block_n", 64)),
        block_k=int(metadata.get("block_k", 64)),
        threads=int(metadata.get("threads", 128)),
        num_stages=int(metadata.get("num_stages", 2)),
        activation_scale_mode=activation_scale_mode,
        activation_scale=activation_scale,
        activation_quant_block_size=int(
            metadata.get("activation_quant_block_size", 256)
        ),
        eps=float(metadata.get("eps", 1e-6)),
        cache_int8_compute_view=bool(
            metadata.get("cache_int8_compute_view", True)
        ),
        min_int8_rows=int(execution.min_int8_rows),
    )


__all__ = [
    "CompositeAddLinear",
    "CompositeAddModule",
    "materialize_composite_compute",
    "materialize_composite_w4a4",
]
