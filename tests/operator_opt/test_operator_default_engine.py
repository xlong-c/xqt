from __future__ import annotations

from xqt.core.schema import (
    OperatorOptimizationConfig,
    OperatorOptimizationTargetConfig,
)
from xqt.kernels.wrappers.capability import describe_operator_engine_capability
from xqt.kernels.wrappers.plan import build_operator_optimization_plan
from xqt.workflows.stage_specs import OperatorStageSpec


def test_torch_compile_is_default_operator_engine_across_fact_sources() -> None:
    config = OperatorOptimizationConfig(enabled=True)
    assert config.default_engine == "torch_compile"
    assert OperatorStageSpec().default_engine == "torch_compile"

    plan = build_operator_optimization_plan(
        OperatorOptimizationConfig(
            enabled=True,
            targets=[OperatorOptimizationTargetConfig(name="fc", target="fc")],
        )
    )
    assert plan.default_engine == "torch_compile"
    assert plan.targets[0].engine == "torch_compile"


def test_torch_compile_capability_is_executable() -> None:
    capability = describe_operator_engine_capability("torch_compile")

    assert capability.status == "available"
    assert capability.maturity == "executable"
    assert capability.available
    projection = capability.to_optimization_capability()
    assert projection.maturity == "executable"
    assert projection.available


def test_explicit_target_engine_overrides_default() -> None:
    plan = build_operator_optimization_plan(
        OperatorOptimizationConfig(
            enabled=True,
            default_engine="torch_compile",
            targets=[
                OperatorOptimizationTargetConfig(
                    name="fc",
                    target="fc",
                    engine="triton",
                )
            ],
        )
    )

    assert plan.targets[0].engine == "triton"
