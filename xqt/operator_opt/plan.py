"""Operator optimization plan assembly helpers."""

from __future__ import annotations

from xqt.core.schema import OperatorOptimizationConfig

from .execution_support import ordered_unique
from .types import (
    OperatorOptimizationExecutionPlan,
    OperatorOptimizationTargetPlan,
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
        targets.append(
            OperatorOptimizationTargetPlan(
                name=target.name,
                engine=target.engine or operator_config.default_engine,
                target_path=target.target,
                mode=target.mode,
                fullgraph=target.fullgraph,
                dynamic=target.dynamic,
                options=dict(target.options),
                patterns=list(target.patterns),
                fallback=target.fallback,
                fallback_policy=target.fallback_policy,
                min_speedup=target.min_speedup,
                validate={
                    "atol": float(target.validate.atol),
                    "rtol": float(target.validate.rtol),
                },
                tilelang={
                    "target": target.tilelang.target,
                    "target_arch": target.tilelang.target_arch,
                    "threads": target.tilelang.threads,
                    "num_stages": target.tilelang.num_stages,
                    "cache_dir": target.tilelang.cache_dir,
                    "pass_configs": dict(target.tilelang.pass_configs),
                    "linear_runtime": target.tilelang.linear_runtime,
                    "linear_fastpath": target.tilelang.linear_fastpath,
                    "attention_fastpath": target.tilelang.attention_fastpath,
                    "conv_fastpath": target.tilelang.conv_fastpath,
                    "norm_fastpath": target.tilelang.norm_fastpath,
                },
                cutile={
                    "target": target.cutile.target,
                    "target_arch": target.cutile.target_arch,
                    "threads": target.cutile.threads,
                    "cache_dir": target.cutile.cache_dir,
                    "pass_configs": dict(target.cutile.pass_configs),
                },
                cutlass={
                    "target_arch": target.cutlass.target_arch,
                    "cache_dir": target.cutlass.cache_dir,
                    "tile_shape": list(target.cutlass.tile_shape),
                    "cluster_shape": (
                        list(target.cutlass.cluster_shape)
                        if target.cutlass.cluster_shape is not None
                        else None
                    ),
                    "pass_configs": dict(target.cutlass.pass_configs),
                },
                cute_dsl={
                    "target_arch": target.cute_dsl.target_arch,
                    "cache_dir": target.cute_dsl.cache_dir,
                    "tile_shape": list(target.cute_dsl.tile_shape),
                    "cluster_shape": (
                        list(target.cute_dsl.cluster_shape)
                        if target.cute_dsl.cluster_shape is not None
                        else None
                    ),
                    "pass_configs": dict(target.cute_dsl.pass_configs),
                },
            )
        )
    return OperatorOptimizationExecutionPlan(
        targets=targets,
        default_engine=operator_config.default_engine,
        stage=operator_config.stage,
        metadata={"target_names": ordered_unique(target.name for target in targets)},
    )
