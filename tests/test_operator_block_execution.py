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
from xqt.kernels.wrappers import (
    OperatorOptimizationExecutionPlan,
    OperatorOptimizationTargetPlan,
    register_block_kernel_builder,
)
from xqt.kernels.wrappers.capability import OperatorOptimizationEngineCapability
from xqt.kernels.wrappers.execute import execute_operator_optimization_plan
from xqt.kernels.wrappers.materialize import materialize_operator_candidate_model
from xqt.kernels.wrappers.plan import build_operator_optimization_plan


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


class _Offset(nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + 1.0


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
    import xqt.kernels.wrappers.execute as execute_module

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


def test_operator_target_below_min_speedup_is_not_applied(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import xqt.kernels.wrappers.execute as execute_module
    from xqt.kernels.wrappers.reporting import summarize_operator_optimization_reports

    model = _Model().eval()
    target = _plan(
        candidate_kind="single_kernel",
        target_path="block.inner",
        benchmark_target_path="block",
    )
    target.engine = "torch_compile"
    target.min_speedup = 1.5
    context = XQTContext(
        model=model,
        example_inputs=torch.randn(2, 4),
        device="cpu",
        artifact_dir=str(tmp_path),
        project_name="block_execution_min_speedup",
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

    def fake_materialize(
        source: nn.Module,
        plan_target: OperatorOptimizationTargetPlan,
    ) -> tuple[nn.Module, float | None]:
        candidate = copy.deepcopy(source)
        candidate.block.inner = nn.Identity()
        assert plan_target.target_path == "block.inner"
        return candidate, None

    benchmark_calls = 0

    def fake_benchmark(
        fn: Any,
        **_: Any,
    ) -> tuple[dict[str, float], str]:
        nonlocal benchmark_calls
        fn()
        benchmark_calls += 1
        latency = 1.0 if benchmark_calls == 1 else 1.1
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

    result = execute_operator_optimization_plan(
        context,
        plan=OperatorOptimizationExecutionPlan(targets=[target]),
    )

    report = result.reports[0]
    summary = summarize_operator_optimization_reports(result.reports)
    assert report.applied is False
    assert report.speedup is not None and report.speedup < target.min_speedup
    assert "min_speedup" in str(report.skip_reason)
    assert summary["applied_count"] == 0
    assert summary["min_speedup_rejections"] == ["block_candidate"]
    assert summary["acceptance"][0]["decision"] == "rejected_min_speedup"
    assert summary["acceptance"][0]["meets_speedup"] is False


def test_operator_target_failed_numeric_validation_is_not_applied(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import xqt.kernels.wrappers.execute as execute_module
    from xqt.kernels.wrappers.reporting import summarize_operator_optimization_reports

    model = _Model().eval()
    inputs = torch.randn(2, 4)
    target = _plan(
        candidate_kind="single_kernel",
        target_path="block.inner",
        benchmark_target_path="block",
    )
    target.engine = "torch_compile"
    target.min_speedup = 1.0
    context = XQTContext(
        model=model,
        example_inputs=inputs,
        device="cpu",
        artifact_dir=str(tmp_path),
        project_name="block_execution_numeric_failure",
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

    def fake_materialize(
        source: nn.Module,
        plan_target: OperatorOptimizationTargetPlan,
    ) -> tuple[nn.Module, float | None]:
        candidate = copy.deepcopy(source)
        candidate.block.inner = _Offset()
        assert plan_target.target_path == "block.inner"
        return candidate, None

    benchmark_calls = 0

    def fake_benchmark(
        fn: Any,
        **_: Any,
    ) -> tuple[dict[str, float], str]:
        nonlocal benchmark_calls
        fn()
        benchmark_calls += 1
        latency = 2.0 if benchmark_calls == 1 else 1.0
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

    result = execute_operator_optimization_plan(
        context,
        plan=OperatorOptimizationExecutionPlan(targets=[target]),
    )

    report = result.reports[0]
    summary = summarize_operator_optimization_reports(result.reports)
    validation = report.metadata["numeric_validation"]
    assert report.applied is False
    assert report.skip_reason == "numeric validation failed"
    assert report.numeric_diff is not None
    assert report.numeric_diff["allclose"] is False
    assert validation["status"] == "failed"
    assert validation["allclose"] is False
    assert validation["thresholds"] == {"atol": 1e-5, "rtol": 1e-5}
    assert summary["numeric_rejections"] == ["block_candidate"]
    assert summary["numeric_validation_failures"] == ["block_candidate"]
    assert summary["acceptance"][0]["decision"] == "rejected_numeric"
    torch.testing.assert_close(result.model(inputs), inputs)


def test_torch_compile_graph_report_preserves_target_compile_options(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import xqt.kernels.wrappers.execute as execute_module

    model = _Model().eval()
    target = _plan(
        candidate_kind="single_kernel",
        target_path="block.inner",
        benchmark_target_path="block",
    )
    target.engine = "torch_compile"
    target.mode = "reduce-overhead"
    target.dynamic = True
    target.fullgraph = True
    context = XQTContext(
        model=model,
        example_inputs=torch.randn(2, 4),
        device="cpu",
        artifact_dir=str(tmp_path),
        project_name="block_execution_graph_report",
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

    def fake_materialize(
        source: nn.Module,
        plan_target: OperatorOptimizationTargetPlan,
    ) -> tuple[nn.Module, float | None]:
        del plan_target
        return copy.deepcopy(source), 1.0

    def fake_benchmark(fn: Any, **_: Any) -> tuple[dict[str, float], str]:
        fn()
        return {
            "mean_ms": 1.0,
            "p50_ms": 1.0,
            "p90_ms": 1.0,
            "p99_ms": 1.0,
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
            "graph_count": 2,
            "graph_break_count": 1,
            "break_reasons": ["dynamic shape guard"],
            "break_details": [
                {
                    "index": 0,
                    "reason": "dynamic shape guard",
                    "graph_break": True,
                    "user_stack": "forward:1",
                }
            ],
            "op_count": 4,
            "compile_times": "compile_inner 1.0",
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

    result = execute_operator_optimization_plan(
        context,
        plan=OperatorOptimizationExecutionPlan(targets=[target]),
    )

    graph_report = result.reports[0].metadata["graph_break_report"]
    assert graph_report["mode"] == "reduce-overhead"
    assert graph_report["dynamic"] is True
    assert graph_report["fullgraph"] is True
    assert graph_report["graph_count"] == 2
    assert graph_report["graph_break_count"] == 1
    assert graph_report["break_details"][0]["reason"] == "dynamic shape guard"
    fallback_detail = result.reports[0].metadata["fallback_detail"]
    assert fallback_detail["explain"] == graph_report


def test_cuda_only_planned_engine_reports_skip_without_cuda(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import xqt.kernels.wrappers.execute as execute_module

    target = _plan(
        candidate_kind="single_kernel",
        target_path="block.inner",
        benchmark_target_path="block",
    )
    target.engine = "cutlass"
    target.fallback = "eager"
    context = XQTContext(
        model=_Model().eval(),
        example_inputs=torch.randn(2, 4),
        device="cpu",
        artifact_dir=str(tmp_path),
        project_name="cutlass_no_cuda_skip",
        benchmark_config=BenchmarkConfig(
            warmup=0,
            iterations=1,
            sync_cuda=False,
            measure_memory=False,
        ),
    )

    def fake_benchmark(fn: Any, **_: Any) -> tuple[dict[str, float], str]:
        fn()
        return {
            "mean_ms": 1.0,
            "p50_ms": 1.0,
            "p90_ms": 1.0,
            "p99_ms": 1.0,
        }, "single_call"

    monkeypatch.setattr(execute_module.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        execute_module,
        "_benchmark_callable_for_execution",
        fake_benchmark,
    )

    result = execute_operator_optimization_plan(
        context,
        plan=OperatorOptimizationExecutionPlan(targets=[target]),
    )

    report = result.reports[0]
    skip_report = report.metadata["planned_skip_report"]
    assert report.applied is False
    assert report.engine == "cutlass"
    assert "requires CUDA-capable hardware" in str(report.skip_reason)
    assert skip_report["engine"] == "cutlass"
    assert skip_report["status"] == "planned"
    assert skip_report["execution_status"] == "planned"
    assert skip_report["requires_cuda"] is True
    assert skip_report["cuda_available"] is False
    assert skip_report["missing_requirements"] == ["cuda"]


def test_non_pytorch_quant_runtime_guard_skips_torch_compile(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import xqt.kernels.wrappers.execute as execute_module

    target = _plan(
        candidate_kind="single_kernel",
        target_path="block.inner",
        benchmark_target_path="block",
    )
    target.engine = "torch_compile"
    context = XQTContext(
        model=_Model().eval(),
        example_inputs=torch.randn(2, 4),
        device="cpu",
        artifact_dir=str(tmp_path),
        project_name="onnxruntime_quant_guard",
        benchmark_config=BenchmarkConfig(
            warmup=0,
            iterations=1,
            sync_cuda=False,
            measure_memory=False,
        ),
        metrics={
            "quant": {
                "backend": "onnxruntime_qdq",
                "components": [{"runtime": "onnxruntime"}],
            }
        },
    )
    capability = OperatorOptimizationEngineCapability(
        engine="torch_compile",
        status="available",
        maturity="executable",
        runtime="pytorch",
        exportable=True,
        available=True,
    )

    def fake_benchmark(fn: Any, **_: Any) -> tuple[dict[str, float], str]:
        fn()
        return {
            "mean_ms": 1.0,
            "p50_ms": 1.0,
            "p90_ms": 1.0,
            "p99_ms": 1.0,
        }, "single_call"

    monkeypatch.setattr(
        execute_module,
        "describe_operator_engine_capability",
        lambda _: capability,
    )
    monkeypatch.setattr(
        execute_module,
        "_benchmark_callable_for_execution",
        fake_benchmark,
    )

    result = execute_operator_optimization_plan(
        context,
        plan=OperatorOptimizationExecutionPlan(targets=[target]),
    )

    report = result.reports[0]
    guard = report.metadata["quant_runtime_guard"]
    assert report.applied is False
    assert "onnxruntime_qdq" in str(report.skip_reason)
    assert guard["guard_applies"] is True
    assert guard["backend"] == "onnxruntime_qdq"
    assert guard["runtime"] == "onnxruntime"
    assert guard["runtime_source"] == "components.0.runtime"
    assert guard["target_engine"] == "torch_compile"


def test_manual_block_kernel_fallback_runs_when_auto_block_materialization_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import xqt.kernels.wrappers.execute as execute_module
    import xqt.kernels.wrappers.materialize as materialize_module

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
