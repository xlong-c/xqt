"""Target collection and materialization for Wan 2.1 VAE Conv3d and RMSNorm."""

from __future__ import annotations

import copy
from typing import Any, Sequence

import torch
from torch import nn

from xqt.core.errors import XQTBackendError
from xqt.kernels.wrappers import OperatorOptimizationTargetPlan, materialize_operator_candidate_models

from .types import _resolve_vae


def _maybe_clone_container(model_or_pipeline: Any, *, inplace: bool) -> Any:
    return model_or_pipeline if inplace else copy.copy(model_or_pipeline)


def collect_wan21_vae_conv3d_targets(
    model_or_pipeline: Any,
    *,
    target_arch: str | None = None,
    include_names: Sequence[str] | None = None,
    exclude_names: Sequence[str] | None = None,
    min_speedup: float = 0.0,
) -> list[OperatorOptimizationTargetPlan]:
    """Collect TileLang Conv3d 1x1x1 targets from a Wan 2.1 VAE."""

    vae, _ = _resolve_vae(model_or_pipeline)
    include_set = None if include_names is None else set(include_names)
    exclude_set = None if exclude_names is None else set(exclude_names)
    targets: list[OperatorOptimizationTargetPlan] = []
    for name, child in vae.named_modules():
        if not name or not isinstance(child, nn.Conv3d):
            continue
        if include_set is not None and name not in include_set:
            continue
        if exclude_set is not None and name in exclude_set:
            continue
        if (
            tuple(int(value) for value in child.kernel_size) != (1, 1, 1)
            or tuple(int(value) for value in child.stride) != (1, 1, 1)
            or tuple(int(value) for value in child.padding) != (0, 0, 0)
            or tuple(int(value) for value in child.dilation) != (1, 1, 1)
            or int(child.groups) != 1
        ):
            continue
        targets.append(
            OperatorOptimizationTargetPlan(
                name=f"{name}_tilelang_conv3d_1x1x1",
                engine="tilelang",
                target_path=name,
                patterns=["conv3d_1x1x1"],
                fallback="eager",
                min_speedup=float(min_speedup),
                validate={"atol": 1e-2, "rtol": 1e-2},
                tilelang={
                    "target": "cuda",
                    "target_arch": target_arch,
                    "conv_fastpath": "tilelang",
                },
            )
        )
    return targets


def materialize_wan21_vae_conv3d_fastpath(
    model_or_pipeline: Any,
    *,
    target_arch: str | None = None,
    include_names: Sequence[str] | None = None,
    exclude_names: Sequence[str] | None = None,
    min_speedup: float = 0.0,
    inplace: bool = False,
) -> tuple[Any, int]:
    """Replace eligible Wan 2.1 VAE Conv3d 1x1x1 modules with TileLang wrappers."""

    container = model_or_pipeline if inplace else copy.deepcopy(model_or_pipeline)
    vae, component_name = _resolve_vae(container)
    targets = collect_wan21_vae_conv3d_targets(
        vae,
        target_arch=target_arch,
        include_names=include_names,
        exclude_names=exclude_names,
        min_speedup=min_speedup,
    )
    if not targets:
        return container, 0
    optimized_vae = materialize_operator_candidate_models(
        vae,
        targets,
        inplace=True,
    )
    if component_name is not None:
        setattr(container, component_name, optimized_vae)
    else:
        container = optimized_vae
    return container, len(targets)


def collect_wan21_vae_rmsnorm_targets(
    model_or_pipeline: Any,
    *,
    include_names: Sequence[str] | None = None,
    exclude_names: Sequence[str] | None = None,
    min_speedup: float = 0.0,
    eps: float = 1e-6,
    block_size: int = 1024,
    num_warps: int = 4,
    num_stages: int = 4,
) -> list[OperatorOptimizationTargetPlan]:
    """Collect Triton RMSNorm targets from a Wan 2.1 VAE."""

    vae, _ = _resolve_vae(model_or_pipeline)
    include_set = None if include_names is None else set(include_names)
    exclude_set = None if exclude_names is None else set(exclude_names)
    targets: list[OperatorOptimizationTargetPlan] = []
    for name, child in vae.named_modules():
        if not name:
            continue
        if include_set is not None and name not in include_set:
            continue
        if exclude_set is not None and name in exclude_set:
            continue
        if not (hasattr(child, "gamma") and hasattr(child, "scale")):
            continue
        targets.append(
            OperatorOptimizationTargetPlan(
                name=f"{name}_triton_rmsnorm",
                engine="triton",
                target_path=name,
                patterns=["rmsnorm"],
                fallback="eager",
                min_speedup=float(min_speedup),
                validate={"atol": 1e-2, "rtol": 1e-2},
                options={
                    "eps": float(eps),
                    "block_size": int(block_size),
                    "num_warps": int(num_warps),
                    "num_stages": int(num_stages),
                },
            )
        )
    return targets


def materialize_wan21_vae_rmsnorm_fastpath(
    model_or_pipeline: Any,
    *,
    include_names: Sequence[str] | None = None,
    exclude_names: Sequence[str] | None = None,
    min_speedup: float = 0.0,
    eps: float = 1e-6,
    block_size: int = 1024,
    num_warps: int = 4,
    num_stages: int = 4,
    inplace: bool = False,
) -> tuple[Any, int]:
    """Replace eligible Wan 2.1 VAE RMSNorm modules with Triton wrappers."""

    container = model_or_pipeline if inplace else copy.deepcopy(model_or_pipeline)
    vae, component_name = _resolve_vae(container)
    targets = collect_wan21_vae_rmsnorm_targets(
        vae,
        include_names=include_names,
        exclude_names=exclude_names,
        min_speedup=min_speedup,
        eps=eps,
        block_size=block_size,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    if not targets:
        return container, 0
    optimized_vae = materialize_operator_candidate_models(
        vae,
        targets,
        inplace=True,
    )
    if component_name is not None:
        setattr(container, component_name, optimized_vae)
    else:
        container = optimized_vae
    return container, len(targets)
