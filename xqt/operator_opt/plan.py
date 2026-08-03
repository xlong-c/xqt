"""Operator optimization plan assembly helpers."""

from __future__ import annotations

from xqt.core.schema import (
    OperatorOptimizationConfig,
    OperatorOptimizationTargetConfig,
)

from .execution_support import ordered_unique
from .types import (
    OperatorOptimizationExecutionPlan,
    OperatorOptimizationTargetPlan,
)


_CANDIDATE_KINDS = frozenset({"single_kernel", "block_kernel"})


def _normalized_candidate_kind(value: str | None) -> str:
    candidate_kind = str(value or "single_kernel").strip().lower()
    if candidate_kind not in _CANDIDATE_KINDS:
        allowed = ", ".join(sorted(_CANDIDATE_KINDS))
        raise ValueError(f"operator target candidate_kind must be one of: {allowed}")
    return candidate_kind


def _fallback_target_name(name: str) -> str:
    base_name = str(name or "operator_target").strip() or "operator_target"
    return f"{base_name}.manual_block_kernel"


def _build_target_plan(
    *,
    name: str,
    engine: str,
    target_path: str | None,
    candidate_kind: str,
    benchmark_target_path: str | None,
    block_kernel: str | None,
    block_kernel_engine: str | None,
    fallback_for: str | None,
    source_target: OperatorOptimizationTargetConfig,
) -> OperatorOptimizationTargetPlan:
    return OperatorOptimizationTargetPlan(
        name=name,
        engine=engine,
        target_path=target_path,
        candidate_kind=candidate_kind,
        benchmark_target_path=benchmark_target_path,
        block_kernel=block_kernel,
        block_kernel_engine=block_kernel_engine,
        fallback_for=fallback_for,
        mode=source_target.mode,
        fullgraph=source_target.fullgraph,
        dynamic=source_target.dynamic,
        options=dict(source_target.options),
        patterns=list(source_target.patterns),
        fallback=source_target.fallback,
        fallback_policy=source_target.fallback_policy,
        min_speedup=source_target.min_speedup,
        validate={
            "atol": float(source_target.validate.atol),
            "rtol": float(source_target.validate.rtol),
        },
        tilelang={
            "target": source_target.tilelang.target,
            "target_arch": source_target.tilelang.target_arch,
            "threads": source_target.tilelang.threads,
            "num_stages": source_target.tilelang.num_stages,
            "cache_dir": source_target.tilelang.cache_dir,
            "pass_configs": dict(source_target.tilelang.pass_configs),
            "linear_runtime": source_target.tilelang.linear_runtime,
            "linear_fastpath": source_target.tilelang.linear_fastpath,
            "attention_fastpath": source_target.tilelang.attention_fastpath,
            "conv_fastpath": source_target.tilelang.conv_fastpath,
            "norm_fastpath": source_target.tilelang.norm_fastpath,
        },
        cutile={
            "target": source_target.cutile.target,
            "target_arch": source_target.cutile.target_arch,
            "threads": source_target.cutile.threads,
            "cache_dir": source_target.cutile.cache_dir,
            "pass_configs": dict(source_target.cutile.pass_configs),
        },
        cutlass={
            "target_arch": source_target.cutlass.target_arch,
            "cache_dir": source_target.cutlass.cache_dir,
            "tile_shape": list(source_target.cutlass.tile_shape),
            "cluster_shape": (
                list(source_target.cutlass.cluster_shape)
                if source_target.cutlass.cluster_shape is not None
                else None
            ),
            "pass_configs": dict(source_target.cutlass.pass_configs),
        },
        cute_dsl={
            "target_arch": source_target.cute_dsl.target_arch,
            "cache_dir": source_target.cute_dsl.cache_dir,
            "tile_shape": list(source_target.cute_dsl.tile_shape),
            "cluster_shape": (
                list(source_target.cute_dsl.cluster_shape)
                if source_target.cute_dsl.cluster_shape is not None
                else None
            ),
            "pass_configs": dict(source_target.cute_dsl.pass_configs),
        },
    )


def build_operator_optimization_plan(
    operator_config: OperatorOptimizationConfig,
) -> OperatorOptimizationExecutionPlan:
    """Build a normalized operator optimization execution plan from config."""

    if not operator_config.enabled:
        return OperatorOptimizationExecutionPlan(
            targets=[],
            default_engine=operator_config.default_engine,
            stage=operator_config.stage,
        )

    targets: list[OperatorOptimizationTargetPlan] = []
    for target in operator_config.targets:
        candidate_kind = _normalized_candidate_kind(target.candidate_kind)
        engine = target.engine or operator_config.default_engine
        benchmark_target_path = (
            target.benchmark_target
            if target.benchmark_target is not None
            else target.target
        )
        if candidate_kind == "single_kernel" and target.block_kernel is not None:
            raise ValueError(
                "operator target block_kernel is only valid for "
                "candidate_kind=block_kernel"
            )
        if candidate_kind == "single_kernel" and target.block_kernel_engine is not None:
            raise ValueError(
                "operator target block_kernel_engine is only valid for "
                "candidate_kind=block_kernel"
            )
        if candidate_kind == "block_kernel" and benchmark_target_path != target.target:
            raise ValueError(
                "block_kernel candidates must benchmark the block they replace: "
                "benchmark_target must equal target"
            )
        if (
            candidate_kind == "block_kernel"
            and engine != "torch_compile"
            and target.block_kernel is None
        ):
            raise ValueError(
                "manual block_kernel candidates require a named block_kernel builder"
            )
        if target.block_kernel_engine is not None and not (
            candidate_kind == "block_kernel"
            and engine == "torch_compile"
            and target.block_kernel is not None
        ):
            raise ValueError(
                "block_kernel_engine is only valid for torch_compile block_kernel "
                "targets with a manual block_kernel fallback"
            )
        if (
            candidate_kind == "block_kernel"
            and engine == "torch_compile"
            and target.block_kernel is not None
            and target.block_kernel_engine is None
        ):
            raise ValueError(
                "torch_compile block_kernel fallback requires block_kernel_engine"
            )
        targets.append(
            _build_target_plan(
                name=target.name,
                engine=engine,
                target_path=target.target,
                candidate_kind=candidate_kind,
                benchmark_target_path=benchmark_target_path,
                block_kernel=(
                    None
                    if candidate_kind == "block_kernel"
                    and engine == "torch_compile"
                    and target.block_kernel_engine is not None
                    else target.block_kernel
                ),
                block_kernel_engine=target.block_kernel_engine,
                fallback_for=None,
                source_target=target,
            )
        )
        if (
            candidate_kind == "block_kernel"
            and engine == "torch_compile"
            and target.block_kernel is not None
            and target.block_kernel_engine is not None
        ):
            targets.append(
                _build_target_plan(
                    name=_fallback_target_name(target.name),
                    engine=target.block_kernel_engine,
                    target_path=target.target,
                    candidate_kind=candidate_kind,
                    benchmark_target_path=benchmark_target_path,
                    block_kernel=target.block_kernel,
                    block_kernel_engine=None,
                    fallback_for=target.name,
                    source_target=target,
                )
            )
    return OperatorOptimizationExecutionPlan(
        targets=targets,
        default_engine=operator_config.default_engine,
        stage=operator_config.stage,
        metadata={"target_names": ordered_unique(target.name for target in targets)},
    )
