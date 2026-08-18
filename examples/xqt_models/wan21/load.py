"""Wan 2.1 VAE model loading and high-level inference entry points."""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn

from xqt.core.errors import XQTBackendError

from .types import (
    WAN21_VAE_REPO_ID,
    WAN21_VAE_SUBFOLDER,
)
from .optimize import optimize_wan21_vae


def load_wan21_vae(
    *,
    repo_id: str = WAN21_VAE_REPO_ID,
    subfolder: str = WAN21_VAE_SUBFOLDER,
    dtype: torch.dtype = torch.float16,
    device: str | torch.device | None = None,
    local_files_only: bool = False,
    **kwargs: Any,
) -> nn.Module:
    """Lazy-load a Wan 2.1 AutoencoderKLWan from Diffusers."""

    try:
        from diffusers import AutoencoderKLWan
    except ImportError as exc:
        raise XQTBackendError(
            "diffusers is required to load Wan 2.1 VAE. Install a version exposing AutoencoderKLWan."
        ) from exc
    vae = AutoencoderKLWan.from_pretrained(
        repo_id,
        subfolder=subfolder,
        torch_dtype=dtype,
        local_files_only=local_files_only,
        **kwargs,
    )
    vae.eval()
    if device is not None:
        vae.to(device=device)
    return vae


def load_wan21_pipeline_with_vae(
    *,
    pipeline_cls: type[Any] | None = None,
    repo_id: str = WAN21_VAE_REPO_ID,
    vae_repo_id: str = WAN21_VAE_REPO_ID,
    vae_subfolder: str = WAN21_VAE_SUBFOLDER,
    dtype: torch.dtype = torch.float16,
    device: str | torch.device | None = None,
    local_files_only: bool = False,
    **kwargs: Any,
) -> Any:
    """Load a Diffusers pipeline and replace its VAE with an explicit Wan 2.1 VAE."""

    if pipeline_cls is None:
        raise XQTBackendError(
            "load_wan21_pipeline_with_vae requires pipeline_cls=... to avoid baking a specific Wan pipeline variant into XQT"
        )
    vae = load_wan21_vae(
        repo_id=vae_repo_id,
        subfolder=vae_subfolder,
        dtype=dtype,
        device=device,
        local_files_only=local_files_only,
    )
    pipeline = pipeline_cls.from_pretrained(
        repo_id,
        vae=vae,
        torch_dtype=dtype,
        local_files_only=local_files_only,
        **kwargs,
    )
    if device is not None and hasattr(pipeline, "to"):
        pipeline.to(device)
    return pipeline


def run_wan21_vae_inference(
    model_or_pipeline: Any,
    *,
    tensor: torch.Tensor,
    run_mode: str = "decode",
    optimization_kind: str = "compile",
    compile_engine: str = "inductor",
    compile_mode: str | None = None,
    compile_fullgraph: bool = False,
    compile_dynamic: bool = False,
    compile_options: Mapping[str, Any] | None = None,
    enable_tiling: bool = False,
    enable_slicing: bool = False,
    tile_sample_min_height: int | None = None,
    tile_sample_min_width: int | None = None,
    tile_sample_stride_height: int | None = None,
    tile_sample_stride_width: int | None = None,
    materialize_conv3d_fastpath: bool = False,
    conv3d_target_arch: str | None = None,
    conv3d_include_names: list[str] | None = None,
    conv3d_exclude_names: list[str] | None = None,
    conv3d_min_speedup: float = 0.0,
    materialize_rmsnorm_fastpath: bool = False,
    rmsnorm_include_names: list[str] | None = None,
    rmsnorm_exclude_names: list[str] | None = None,
    rmsnorm_min_speedup: float = 0.0,
    rmsnorm_eps: float = 1e-6,
    rmsnorm_block_size: int = 1024,
    rmsnorm_num_warps: int = 4,
    rmsnorm_num_stages: int = 4,
    warmup_iterations: int = 0,
    inplace: bool = False,
) -> dict[str, Any]:
    """Optimize one Wan 2.1 VAE path and execute it once."""

    result, summary = optimize_wan21_vae(
        model_or_pipeline,
        run_mode=run_mode,
        optimization_kind=optimization_kind,
        compile_engine=compile_engine,
        compile_mode=compile_mode,
        compile_fullgraph=compile_fullgraph,
        compile_dynamic=compile_dynamic,
        compile_options=compile_options,
        tensor=tensor,
        enable_tiling=enable_tiling,
        enable_slicing=enable_slicing,
        tile_sample_min_height=tile_sample_min_height,
        tile_sample_min_width=tile_sample_min_width,
        tile_sample_stride_height=tile_sample_stride_height,
        tile_sample_stride_width=tile_sample_stride_width,
        materialize_conv3d_fastpath=materialize_conv3d_fastpath,
        conv3d_target_arch=conv3d_target_arch,
        conv3d_include_names=conv3d_include_names,
        conv3d_exclude_names=conv3d_exclude_names,
        conv3d_min_speedup=conv3d_min_speedup,
        materialize_rmsnorm_fastpath=materialize_rmsnorm_fastpath,
        rmsnorm_include_names=rmsnorm_include_names,
        rmsnorm_exclude_names=rmsnorm_exclude_names,
        rmsnorm_min_speedup=rmsnorm_min_speedup,
        rmsnorm_eps=rmsnorm_eps,
        rmsnorm_block_size=rmsnorm_block_size,
        rmsnorm_num_warps=rmsnorm_num_warps,
        rmsnorm_num_stages=rmsnorm_num_stages,
        warmup_iterations=warmup_iterations,
        inplace=inplace,
    )
    with torch.inference_mode():
        output = result.model(tensor)
    return {
        "summary": summary.to_dict(),
        "result": result.to_dict(),
        "output": output,
    }
