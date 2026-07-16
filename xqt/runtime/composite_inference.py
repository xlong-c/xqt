"""Infer-side helpers to accelerate materialized SVD composite modules.

These consume an already-quantized model plus its ``compute_config`` and decide
*how* to realize the composite modules (mode A/B/C from the design):

- ``mode="collapse"``    - fold low-rank + residual into one INT8 GEMM (A)
- ``activation_scale_mode="static"`` + calibration - fused activation quant (B)
- ``min_int8_rows>0``    - small-M bf16 fallback for decode (C)

No quantizer, no gradient, no dataset construction. Static-scale calibration is
a one-shot per-tensor activation-range collection, matching the existing
``Int8MmaLinear`` contract; it is not training.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

import torch
from torch import nn

from xqt.contracts.compute import (
    ComputeConfig,
    ModuleExecutionSpec,
    compute_config_from_mapping,
    normalize_composite_mode,
)
from xqt.runtime.composite_branch import SupportsStaticActivationCalibration
from xqt.runtime.composite_materialize import materialize_composite_compute


def fuse_composite_modules(model: nn.Module, *, mode: str = "reduce-overhead") -> int:
    """Enable torch.compile fusion on every fusible composite module.

    Only recovers the low-rank branch overhead in compute-bound (large-M)
    regimes; it is a no-op on modules without ``enable_fusion`` and degrades to
    eager if compilation is unavailable. Returns the count of fused modules.
    """

    fused = 0
    for module in model.modules():
        enable = getattr(module, "enable_fusion", None)
        if callable(enable) and enable(mode=mode):
            fused += 1
    return fused


def override_composite_execution(
    compute_config: ComputeConfig | Mapping[str, Any] | None,
    *,
    mode: str | None = None,
    activation_scale_mode: str | None = None,
    min_int8_rows: int | None = None,
    compute_precision: str | None = None,
) -> ComputeConfig | None:
    """Return a compute_config with per-module execution knobs overridden.

    Only the provided arguments are changed; ``None`` leaves the existing value.
    """

    config = (
        compute_config
        if isinstance(compute_config, ComputeConfig)
        else compute_config_from_mapping(compute_config)
    )
    if config is None:
        return None
    resolved_mode = normalize_composite_mode(mode) if mode is not None else None
    for module in config.modules:
        if resolved_mode is not None:
            module.preferred_mode = resolved_mode
        current = module.execution or ModuleExecutionSpec()
        module.execution = ModuleExecutionSpec(
            activation_scale_mode=(
                current.activation_scale_mode
                if activation_scale_mode is None
                else activation_scale_mode
            ),
            min_int8_rows=(
                current.min_int8_rows if min_int8_rows is None else int(min_int8_rows)
            ),
            compute_precision=(
                current.compute_precision
                if compute_precision is None
                else compute_precision
            ),
        )
    return config


def materialize_svd_for_inference(
    model: nn.Module,
    compute_config: ComputeConfig | Mapping[str, Any] | None,
    *,
    mode: str | None = None,
    activation_scale_mode: str | None = None,
    min_int8_rows: int | None = None,
    compute_precision: str | None = None,
    calibration_inputs: Iterable[Any] | None = None,
    calibration_sample_limit: int | None = None,
    fuse: bool = False,
    fuse_mode: str = "reduce-overhead",
    inplace: bool = True,
) -> nn.Module:
    """Override execution knobs, materialize composite modules, then calibrate.

    Mirrors the design's A+B+C entry point: pick ``mode`` (collapse/split),
    ``compute_precision`` (``w8a8``/``fp8``), ``activation_scale_mode`` (static
    enables fused kernels), and ``min_int8_rows`` (decode fallback). When
    ``activation_scale_mode`` is ``"static"`` and ``calibration_inputs`` are
    given, static scales are collected in one pass. When ``fuse`` is set, the
    low-rank branch of split modules is folded via torch.compile (recovers
    overhead at large M; safe no-op otherwise).
    """

    config = override_composite_execution(
        compute_config,
        mode=mode,
        activation_scale_mode=activation_scale_mode,
        min_int8_rows=min_int8_rows,
        compute_precision=compute_precision,
    )
    materialized = materialize_composite_compute(model, config, inplace=inplace)
    if activation_scale_mode == "static" and calibration_inputs is not None:
        calibrate_static_activation_scales(
            materialized,
            calibration_inputs,
            sample_limit=calibration_sample_limit,
        )
    if fuse:
        fuse_composite_modules(materialized, mode=fuse_mode)
    return materialized


@torch.no_grad()
def calibrate_static_activation_scales(
    model: nn.Module,
    calibration_inputs: Iterable[Any],
    *,
    sample_limit: int | None = None,
) -> dict[str, torch.Tensor]:
    """Collect and set per-tensor static activation scales on INT8/FP8 modules.

    Registers forward hooks on every ``Int8MmaLinear`` / ``Fp8MmaLinear``, runs
    the representative inputs, and sets each layer's static scale from the
    observed activation ``amax`` (INT8: ``/127``; FP8: ``/448``). Returns the
    scale per module name. This is a one-shot Infer-side calibration, not QAT.
    """

    targets = {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, SupportsStaticActivationCalibration)
    }
    if not targets:
        return {}
    maxima: dict[str, torch.Tensor] = {}
    handles: list[Any] = []

    def make_hook(name: str) -> Any:
        def hook(module: nn.Module, inputs: tuple[Any, ...], _: Any) -> None:
            if not inputs or not isinstance(inputs[0], torch.Tensor):
                return
            activation = inputs[0].detach()
            if activation.shape[-1] != module.input_features:
                return
            current = activation.reshape(-1).to(torch.float32).abs().amax()
            previous = maxima.get(name)
            maxima[name] = current if previous is None else torch.maximum(previous, current)

        return hook

    # Collection pass must not require the scale it is about to produce: run the
    # targets in dynamic mode, then set the static scale (which flips to static).
    saved_modes = {name: module.activation_scale_mode for name, module in targets.items()}
    for name, module in targets.items():
        module.activation_scale_mode = "dynamic"
        handles.append(module.register_forward_hook(make_hook(name)))
    was_training = model.training
    model.eval()
    try:
        for index, batch in enumerate(calibration_inputs):
            if sample_limit is not None and index >= int(sample_limit):
                break
            if isinstance(batch, Mapping):
                model(**batch)
            elif isinstance(batch, (tuple, list)):
                model(*batch)
            else:
                model(batch)
    finally:
        for handle in handles:
            handle.remove()
        for name, module in targets.items():
            module.activation_scale_mode = saved_modes[name]
        model.train(was_training)

    scales: dict[str, torch.Tensor] = {}
    for name, max_abs in maxima.items():
        module = targets[name]
        # Each quant format declares its own quant_max (127 for INT8, 448 for FP8).
        quant_max = getattr(module, "quant_max", 127.0)
        scale = (max_abs / quant_max).clamp_min(float(module.eps))
        module.set_static_activation_scale(scale)
        scales[name] = module.static_activation_scale.detach().clone()
    return scales


__all__ = [
    "calibrate_static_activation_scales",
    "fuse_composite_modules",
    "materialize_svd_for_inference",
    "override_composite_execution",
]
