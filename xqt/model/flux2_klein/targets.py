"""Target collection, plan helpers, and engine materialization for FLUX.2 klein NVFP4."""

from __future__ import annotations

import copy
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch import nn

from xqt.core.errors import XQTBackendError
from xqt.kernels.wrappers import OperatorOptimizationTargetPlan, materialize_operator_candidate_models
from xqt.compression.quant import infer_nvfp4_tensor_layout

from .types import (
    FLUX2_KLEIN_NVFP4_ENGINES,
    Flux2KleinNVFP4EngineResult,
    Flux2KleinNVFP4TargetSummary,
    _cuda_arch,
    _resolve_module,
    _resolve_flux2_klein_nvfp4_engine,
    normalize_flux2_klein_nvfp4_engine,
)


def _matches_filters(
    name: str,
    *,
    include_names: Sequence[str] | None,
    exclude_names: Sequence[str] | None,
) -> bool:
    if include_names is not None and name not in set(include_names):
        return False
    if exclude_names is not None and name in set(exclude_names):
        return False
    return True


def _engine_patterns(engine: str) -> list[str]:
    if engine == "tilelang":
        return ["dequant_gemm_epilogue"]
    if engine == "cutile":
        return ["nvfp4_packed_dequant_gemm_epilogue"]
    return ["gemm_epilogue"]


def _target_plan_for_engine(
    *,
    name: str,
    engine: str,
    target_arch: str | None,
    min_speedup: float,
) -> OperatorOptimizationTargetPlan:
    patterns = _engine_patterns(engine)
    common: dict[str, Any] = {
        "name": f"{name}_{engine}",
        "engine": engine,
        "target_path": name,
        "patterns": patterns,
        "fallback": "eager",
        "min_speedup": float(min_speedup),
        "validate": {"atol": 1e-2, "rtol": 1e-2},
    }
    if engine == "tilelang":
        common["tilelang"] = {
            "target": "cuda",
            "target_arch": target_arch,
            "linear_runtime": "auto",
            "linear_fastpath": "auto",
        }
    elif engine == "cutile":
        common["cutile"] = {
            "target": "cuda",
            "target_arch": target_arch,
            "threads": 128,
        }
    else:
        common["cute_dsl"] = {
            "target_arch": target_arch,
            "tile_shape": [128, 128, 64],
            "cluster_shape": None,
        }
    return OperatorOptimizationTargetPlan(**common)


def collect_flux2_klein_nvfp4_targets(
    model_or_pipeline: Any,
    *,
    engine: str | None = None,
    target_arch: str | None = None,
    max_targets: int | None = None,
    include_names: Sequence[str] | None = None,
    exclude_names: Sequence[str] | None = None,
    min_speedup: float = 0.0,
) -> tuple[list[OperatorOptimizationTargetPlan], list[Flux2KleinNVFP4TargetSummary]]:
    """Collect bridgeable NVFP4 Linear targets for one XQT engine.

    Target paths are relative to the resolved module. For Diffusers pipelines this
    helper scans `pipeline.transformer`, and `materialize_flux2_klein_nvfp4_engine`
    writes the optimized transformer back to the pipeline.
    """

    resolved_engine = _resolve_flux2_klein_nvfp4_engine(
        engine=engine,
        context="collect_flux2_klein_nvfp4_targets",
    )
    module, _ = _resolve_module(model_or_pipeline)
    resolved_arch = target_arch or _cuda_arch()
    targets: list[OperatorOptimizationTargetPlan] = []
    summaries: list[Flux2KleinNVFP4TargetSummary] = []
    for name, child in module.named_modules():
        if not name:
            continue
        if not _matches_filters(name, include_names=include_names, exclude_names=exclude_names):
            continue
        layout = infer_nvfp4_tensor_layout(child)
        if layout is None:
            continue
        target = _target_plan_for_engine(
            name=name,
            engine=resolved_engine,
            target_arch=resolved_arch,
            min_speedup=min_speedup,
        )
        targets.append(target)
        summaries.append(
            Flux2KleinNVFP4TargetSummary(
                name=name,
                engine=resolved_engine,
                module_type=type(child).__name__,
                input_features=layout.input_features,
                output_features=layout.output_features,
                group_size=layout.group_size,
                patterns=list(target.patterns),
            )
        )
        if max_targets is not None and len(targets) >= int(max_targets):
            break
    return targets, summaries


def collect_flux2_klein_nvfp4_engine_targets(
    model_or_pipeline: Any,
    *,
    engines: Iterable[str] | None = None,
    target_arch: str | None = None,
    max_targets: int | None = None,
    include_names: Sequence[str] | None = None,
    exclude_names: Sequence[str] | None = None,
    min_speedup: float = 0.0,
) -> dict[str, tuple[list[OperatorOptimizationTargetPlan], list[Flux2KleinNVFP4TargetSummary]]]:
    """Collect target plans for requested FLUX.2 klein NVFP4 engines."""

    requested_engines = FLUX2_KLEIN_NVFP4_ENGINES
    if engines is not None:
        requested_engines = tuple(engines)

    return {
        normalize_flux2_klein_nvfp4_engine(engine): collect_flux2_klein_nvfp4_targets(
            model_or_pipeline,
            engine=engine,
            target_arch=target_arch,
            max_targets=max_targets,
            include_names=include_names,
            exclude_names=exclude_names,
            min_speedup=min_speedup,
        )
        for engine in requested_engines
    }


def materialize_flux2_klein_nvfp4_engine(
    model_or_pipeline: Any,
    *,
    engine: str | None = None,
    target_arch: str | None = None,
    max_targets: int | None = None,
    include_names: Sequence[str] | None = None,
    exclude_names: Sequence[str] | None = None,
    inplace: bool = False,
    min_speedup: float = 0.0,
) -> Flux2KleinNVFP4EngineResult:
    """Materialize a callable XQT engine inference path for FLUX.2 klein NVFP4."""

    resolved_engine = _resolve_flux2_klein_nvfp4_engine(
        engine=engine,
        context="materialize_flux2_klein_nvfp4_engine",
    )
    module, pipeline_component = _resolve_module(model_or_pipeline)
    targets, summaries = collect_flux2_klein_nvfp4_targets(
        module,
        engine=resolved_engine,
        target_arch=target_arch,
        max_targets=max_targets,
        include_names=include_names,
        exclude_names=exclude_names,
        min_speedup=min_speedup,
    )
    if not targets:
        raise XQTBackendError("no bridgeable FLUX.2 klein NVFP4 Linear targets found")
    optimized_module = materialize_operator_candidate_models(
        module,
        targets,
        inplace=inplace,
    )
    if pipeline_component is None:
        return Flux2KleinNVFP4EngineResult(
            model=optimized_module,
            engine=resolved_engine,
            targets=targets,
            target_summaries=summaries,
        )
    pipeline = model_or_pipeline if inplace else copy.copy(model_or_pipeline)
    setattr(pipeline, pipeline_component, optimized_module)
    return Flux2KleinNVFP4EngineResult(
        model=pipeline,
        engine=resolved_engine,
        targets=targets,
        target_summaries=summaries,
    )
