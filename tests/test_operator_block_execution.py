"""Block-scoped runtime candidate materialization and acceptance tests."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

from xqt.core.errors import XQTBackendError
from xqt.core.schema import (
    BenchmarkConfig,
    OperatorOptimizationConfig,
    OperatorOptimizationTargetConfig,
)
from xqt.core.types import XQTContext
from xqt.operator_opt import (
    OperatorOptimizationExecutionPlan,
    OperatorOptimizationTargetPlan,
    register_block_kernel_builder,
)
from xqt.operator_opt.capability import OperatorOptimizationEngineCapability
from xqt.operator_opt.execute import execute_operator_optimization_plan
from xqt.operator_opt.materialize import materialize_operator_candidate_model
from xqt.operator_opt.plan import build_operator_optimization_plan


class _Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.inner = nn.Identity()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.inner(inputs)


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block = _Block()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


class _FusedBlock(nn.Module):
    def __init__(self, source: _Block) -> None:
        super().__init__()
        self.inner = copy.deepcopy(source.inner)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.inner(inputs)


def _plan(
    *,
    candidate_kind: str,
    target_path: str,
    benchmark_target_path: str,
    block_kernel: str | None = None,
) -> OperatorOptimizationTargetPlan:
    return OperatorOptimizationTargetPlan(
        name="block_candidate",
        engine="tilelang",
        target_path=target_path,
        candidate_kind=candidate_kind,
        benchmark_target_path=benchmark_target_path,
        block_kernel=block_kernel,
        min_speedup=1.01,
    )


def test_plan_keeps_single_kernel_replacement_and_block_benchmark_scope() -> None:
    plan = build_operator_optimization_plan(
        OperatorOptimizationConfig(
            enabled=True,
            targets=[
                OperatorOptimizationTargetConfig(
                    name="inner_kernel",
                    target="block.inner",
                    benchmark_target="block",
                    candidate_kind="single_kernel",
                    engine="tilelang",
                )
            ],
        )
    )

    target = plan.targets[0]
    assert target.candidate_kind == "single_kernel"
    assert target.target_path == "block.inner"
    assert target.benchmark_target_path == "block"


def test_plan_rejects_block_kernel_with_separate_benchmark_target() -> None:
    config = OperatorOptimizationConfig(
        enabled=True,
        targets=[
            OperatorOptimizationTargetConfig(
                name="invalid_block",
                target="block",
                benchmark_target="",
                candidate_kind="block_kernel",
                block_kernel="test_block_execution_fused_block",
                engine="tilelang",
            )
        ],
    )

    with pytest.raises(ValueError, match="must benchmark the block they replace"):
        build_operator_optimization_plan(config)


def test_plan_expands_auto_block_with_manual_block_kernel_fallback() -> None:
    plan = build_operator_optimization_plan(
        OperatorOptimizationConfig(
            enabled=True,
            targets=[
                OperatorOptimizationTargetConfig(
                    name="auto_block",
                    target="block",
                    candidate_kind="block_kernel",
                    block_kernel="test_block_execution_auto_fallback",
                    block_kernel_engine="tilelang",
                    engine="torch_compile",
                )
            ],
        )
    )

    assert len(plan.targets) == 2
    automatic, manual = plan.targets
    assert automatic.name == "auto_block"
    assert automatic.engine == "torch_compile"
    assert automatic.candidate_kind == "block_kernel"
    assert automatic.block_kernel is None
    assert automatic.block_kernel_engine == "tilelang"
    assert automatic.fallback_for is None
    assert manual.name == "auto_block.manual_block_kernel"
    assert manual.engine == "tilelang"
    assert manual.candidate_kind == "block_kernel"
    assert manual.block_kernel == "test_block_execution_auto_fallback"
    assert manual.block_kernel_engine is None
    assert manual.fallback_for == "auto_block"


def test_block_kernel_materialization_uses_registered_whole_block_builder() -> None:
    builder_name = "test_block_execution_fused_block"

    @register_block_kernel_builder(builder_name)
    def build_fused_block(
        block: nn.Module,
        target: OperatorOptimizationTargetPlan,
    ) -> nn.Module:
        assert isinstance(block, _Block)
        assert target.candidate_kind == "block_kernel"
        return _FusedBlock(block)

    source = _Model().eval()
    target = _plan(
        candidate_kind="block_kernel",
        target_path="block",
        benchmark_target_path="block",
        block_kernel=builder_name,
    )

    candidate, compile_time_ms = materialize_operator_candidate_model(source, target)

    assert compile_time_ms is None
    assert isinstance(candidate.block, _FusedBlock)
    assert isinstance(source.block, _Block)
    assert candidate.block._xqt_block_kernel_metadata["builder"] == builder_name


def test_block_kernel_materialization_rejects_missing_builder() -> None:
    target = _plan(
        candidate_kind="block_kernel",
        target_path="block",
        benchmark_target_path="block",
        block_kernel="missing_test_block_builder",
    )

    with pytest.raises(XQTBackendError, match="is not registered"):
        materialize_operator_candidate_model(_Model().eval(), target)


def test_single_kernel_acceptance_is_measured_at_parent_block(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import xqt.operator_opt.execute as execute_module

    model = _Model().eval()
    target = _plan(
        candidate_kind="single_kernel",
        target_path="block.inner",
        benchmark_target_path="block",
    )
    target.engine = "torch_compile"
    context = XQTContext(
        model=model,
        example_inputs=torch.randn(2, 4),
        device="cpu",
        artifact_dir=str(tmp_path),
        project_name="block_execution",
        benchmark_config=BenchmarkConfig(
            warmup=0,
            iterations=1,
            sync_cuda=False,
            measure_memory=False,
        ),
    )
    capability = OperatorOptimizationEngineCapability(
        engine="torch_compile",
        status="available",
        maturity="executable",
        runtime="pytorch",
        exportable=True,
        available=True,
    )
    seen_modules: list[nn.Module] = []
    latency_calls = 0
    original_call_module_no_grad = execute_module.call_module_no_grad

    def record_call_module_no_grad(module: nn.Module, inputs: Any) -> Any:
        seen_modules.append(module)
        return original_call_module_no_grad(module, inputs)

    def fake_materialize(
        source: nn.Module,
        plan_target: OperatorOptimizationTargetPlan,
    ) -> tuple[nn.Module, float | None]:
        candidate = copy.deepcopy(source)
        candidate.block.inner = nn.Identity()
        assert plan_target.target_path == "block.inner"
        return candidate, None

    def fake_benchmark(
        fn: Any,
        **_: Any,
    ) -> tuple[dict[str, float], str]:
        nonlocal latency_calls
        fn()
        latency_calls += 1
        latency = 2.0 if latency_calls == 1 else 1.0
        return {
            "mean_ms": latency,
            "p50_ms": latency,
            "p90_ms": latency,
            "p99_ms": latency,
        }, "single_call"

    monkeypatch.setattr(
        execute_module,
        "describe_operator_engine_capability",
        lambda _: capability,
    )
    monkeypatch.setattr(
        execute_module,
        "torch_compile_explain_report",
        lambda *_: {
            "status": "ok",
            "error": None,
            "graph_count": 1,
            "graph_break_count": 0,
            "break_reasons": [],
            "op_count": 1,
            "compile_times": None,
        },
    )
    monkeypatch.setattr(
        execute_module,
        "materialize_operator_candidate_model",
        fake_materialize,
    )
    monkeypatch.setattr(
        execute_module,
        "_benchmark_callable_for_execution",
        fake_benchmark,
    )
    monkeypatch.setattr(
        execute_module,
        "call_module_no_grad",
        record_call_module_no_grad,
    )

    result = execute_operator_optimization_plan(
        context,
        plan=OperatorOptimizationExecutionPlan(targets=[target]),
    )

    report = result.reports[0]
    assert report.applied
    assert report.candidate_kind == "single_kernel"
    assert report.benchmark_target_path == "block"
    assert report.metadata["benchmark_scope"] == "block"
    assert report.metadata["replacement_target_path"] == "block.inner"
    assert report.speedup == 2.0
    assert seen_modules
    assert all(isinstance(module, _Block) for module in seen_modules)


def test_manual_block_kernel_fallback_runs_when_auto_block_materialization_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import xqt.operator_opt.execute as execute_module
    import xqt.operator_opt.materialize as materialize_module

    builder_name = "test_block_execution_auto_fallback_builder"

    @register_block_kernel_builder(builder_name)
    def build_fused_block(
        block: nn.Module,
        target: OperatorOptimizationTargetPlan,
    ) -> nn.Module:
        assert isinstance(block, _Block)
        assert target.engine == "tilelang"
        assert target.fallback_for == "auto_block"
        return _FusedBlock(block)

    plan = build_operator_optimization_plan(
        OperatorOptimizationConfig(
            enabled=True,
            targets=[
                OperatorOptimizationTargetConfig(
                    name="auto_block",
                    target="block",
                    candidate_kind="block_kernel",
                    block_kernel=builder_name,
                    block_kernel_engine="tilelang",
                    engine="torch_compile",
                )
            ],
        )
    )
    context = XQTContext(
        model=_Model().eval(),
        example_inputs=torch.randn(2, 4),
        device="cpu",
        artifact_dir=str(tmp_path),
        project_name="block_execution",
        benchmark_config=BenchmarkConfig(
            warmup=0,
            iterations=1,
            sync_cuda=False,
            measure_memory=False,
        ),
    )

    def fake_capability(engine: str) -> OperatorOptimizationEngineCapability:
        return OperatorOptimizationEngineCapability(
            engine=engine,
            status="available",
            maturity="executable",
            runtime="pytorch",
            exportable=True,
            available=True,
        )

    def fail_compile(
        module: nn.Module,
        target: OperatorOptimizationTargetPlan,
    ) -> tuple[nn.Module, float | None]:
        del module, target
        raise RuntimeError("auto graph failed")

    benchmark_calls = 0

    def fake_benchmark(
        fn: Any,
        **_: Any,
    ) -> tuple[dict[str, float], str]:
        nonlocal benchmark_calls
        fn()
        benchmark_calls += 1
        latency = 1.0 if benchmark_calls == 3 else 2.0
        return {
            "mean_ms": latency,
            "p50_ms": latency,
            "p90_ms": latency,
            "p99_ms": latency,
        }, "single_call"

    monkeypatch.setattr(
        execute_module,
        "describe_operator_engine_capability",
        fake_capability,
    )
    monkeypatch.setattr(
        execute_module,
        "torch_compile_explain_report",
        lambda *_: {
            "status": "ok",
            "error": None,
            "graph_count": 0,
            "graph_break_count": 1,
            "break_reasons": ["auto graph failed"],
            "op_count": 0,
            "compile_times": None,
        },
    )
    monkeypatch.setattr(materialize_module, "compile_with_torch", fail_compile)
    monkeypatch.setattr(
        execute_module,
        "_benchmark_callable_for_execution",
        fake_benchmark,
    )

    result = execute_operator_optimization_plan(context, plan=plan)

    assert len(result.reports) == 2
    automatic, manual = result.reports
    assert not automatic.applied
    assert "auto graph failed" in str(automatic.skip_reason)
    assert manual.applied
    assert manual.engine == "tilelang"
    assert manual.metadata["candidate_layer"] == "manual_block_kernel"
    assert manual.metadata["fallback_for"] == "auto_block"
    assert manual.metadata["optimization_basis"] == "block"
    assert manual.speedup == 2.0
    assert isinstance(result.model.block, _FusedBlock)
