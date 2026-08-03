from dataclasses import fields
import inspect
from pathlib import Path
import tomllib
from types import SimpleNamespace
from typing import Any

from omegaconf import OmegaConf
import pytest
import torch

import xqt
import xqt.core as core_module
import xqt.core.config as core_config_module
import xqt.core.schema as core_schema_module
import xqt.pipeline as pipeline_module
import xqt.pipeline.preflight as preflight_module
import xqt.pipeline.passes as passes_module
import xqt.pipeline.pass_helpers.quant_stage as quant_stage_module
import xqt.pipeline.runner as runner_module
from xqt.core.errors import XQTConfigError
from xqt.core.schema import (
    BenchmarkConfig,
    ExportTargetConfig,
    OperatorOptimizationConfig,
    OperatorOptimizationTargetConfig,
    OutputDiffConfig,
    PruneConfig,
    QuantConfig,
)
from xqt.core.types import XQTContext
from xqt.pipeline.passes import (
    BenchmarkPass,
    LoadModelPass,
    OperatorOptimizationPass,
    run_analyze_stage,
    run_benchmark_stage,
    run_export_stage,
    run_operator_stage,
    run_prune_stage,
    run_quant_stage,
)
from xqt.pipeline.preflight import preflight_optimization_config
from xqt.pipeline.runner import create_context
from xqt.run_workflow import DEFAULT_CONFIG
from xqt.workflows import (
    OptimizationConfig,
    OptimizationStageConfig,
    OptimizationStageResult,
    XQTOptimizationSession,
    load_optimization_config,
    optimize_model,
)
from xqt.workflows.stage_specs import (
    AnalyzeStageSpec,
    BenchmarkStageSpec,
    DeployStageSpec,
    ExportStageSpec,
    OperatorStageSpec,
    PruneStageSpec,
    QuantStageSpec,
    ensure_stage_spec,
)
import xqt.workflows.optimization as optimization_module


def _legacy_context(
    *,
    model: Any = None,
    example_inputs: Any = None,
    calibration_inputs: Any = None,
    device: str = "cpu",
    artifact_dir: str = "artifacts/xqt/tests/runtime_context",
    project_name: str = "runtime_context",
    task_type: str = "classification",
) -> XQTContext:
    return XQTContext(
        model=model,
        example_inputs=example_inputs,
        calibration_inputs=calibration_inputs,
        device=device,
        artifact_dir=artifact_dir,
        project_name=project_name,
        task_type=task_type,
    )


def test_top_level_xqt_api_matches_framework_contract() -> None:
    expected = {
        "XQTOptimizationSession",
        "OptimizedModelResult",
        "OptimizationConfig",
        "OptimizationStageConfig",
        "OptimizationStageResult",
        "StageAcceptanceConfig",
        "load_optimization_config",
        "optimize_model",
        "ArtifactManifest",
        "ArtifactRecord",
        "MetricRecord",
        "XQTReadinessReport",
        "XQTReadinessScenario",
        "assess_xqt_readiness",
        "load_checkpoint_into_model",
        "xdl_checkpoint_to_xqt_context",
        "xdl_setup_to_xqt_context",
    }
    removed = {
        "load_xqt_config",
        "run_xqt_recipe",
        "preflight_xqt_config",
        "XQTConfig",
        "XQTRegistry",
        "PASS_REGISTRY",
        "RECIPE_REGISTRY",
        "EXPORTER_REGISTRY",
        "register_pass",
        "register_recipe",
        "register_exporter",
    }

    assert set(xqt.__all__) == expected
    assert removed.isdisjoint(set(xqt.__all__))
    for name in removed:
        assert not hasattr(xqt, name)


def test_core_init_does_not_reexport_legacy_recipe_config_api() -> None:
    removed = {
        "XQTConfig",
        "load_xqt_config",
    }

    assert removed.isdisjoint(set(core_module.__all__))
    for name in removed:
        assert not hasattr(core_module, name)


def test_core_legacy_modules_do_not_star_export_recipe_config_api() -> None:
    assert "load_xqt_config" not in core_config_module.__all__
    assert "xqt_config_to_dict" not in core_config_module.__all__
    assert "XQTConfig" not in core_schema_module.__all__
    assert not hasattr(core_config_module, "load_xqt_config")
    assert not hasattr(core_config_module, "xqt_config_to_dict")
    assert not hasattr(core_schema_module, "XQTConfig")


def test_default_workflow_config_is_stage_workflow() -> None:
    config = load_optimization_config(DEFAULT_CONFIG)

    assert DEFAULT_CONFIG.name == "smoke_workflow.yaml"
    assert config.project["name"] == "smoke_workflow"
    assert [stage.kind for stage in config.stages] == ["prune"]
    assert isinstance(config.stages[0].spec, PruneStageSpec)


def test_default_workflow_preflight_uses_stage_workflow_schema() -> None:
    report = preflight_optimization_config(DEFAULT_CONFIG)
    checks = {check.name: check for check in report.checks}

    assert "project.artifact_dir" in checks
    assert "stages.0.prune_l1.kind" in checks
    assert checks["stages.0.prune_l1.kind"].metadata["kind"] == "prune"


def test_legacy_pipeline_recipe_api_is_removed() -> None:
    removed_runner_api = {
        "DEFAULT_COMPRESSION_PASS_ORDER",
        "DEFAULT_PASS_ORDER",
        "build_pipeline_from_config",
        "create_manifest",
        "default_pass_names",
        "enabled_pass_names",
        "run_xqt_recipe",
    }

    assert removed_runner_api.isdisjoint(set(runner_module.__all__))
    assert removed_runner_api.isdisjoint(set(pipeline_module.__all__))
    for name in removed_runner_api:
        assert not hasattr(runner_module, name)
        assert not hasattr(pipeline_module, name)
    assert not hasattr(preflight_module, "preflight_xqt_config")
    assert "preflight_xqt_config" not in preflight_module.__all__


def test_create_context_accepts_optimization_config(tmp_path: Path) -> None:
    config = load_optimization_config(
        {
            "project": {
                "name": "create_context_workflow",
                "artifact_dir": str(tmp_path / "artifacts"),
            },
            "model": {"target": "torch.nn:Identity", "device": "cuda:3"},
        }
    )
    model = torch.nn.Identity().eval()

    context = create_context(
        config,
        model=model,
        example_inputs=torch.randn(1, 4),
    )

    assert context.model is model
    assert not hasattr(context, "config")
    assert context.reference_model is not None
    assert context.project_name == "create_context_workflow"
    assert context.device == "cuda:3"
    assert context.artifact_dir == str(tmp_path / "artifacts")
    assert context.manifest is not None
    assert (
        context.manifest.config_snapshot["project"]["name"] == "create_context_workflow"
    )


def test_create_context_accepts_workflow_mapping(tmp_path: Path) -> None:
    model = torch.nn.Identity().eval()

    context = create_context(
        {
            "project": {
                "name": "create_context_workflow_mapping",
                "artifact_dir": str(tmp_path / "artifacts"),
            },
            "model": {"target": "torch.nn:Identity"},
            "compression_axes": ["precision", "sparsity"],
            "stages": [
                {
                    "name": "quant_model",
                    "kind": "quant",
                    "params": {
                        "backend": "torchao",
                        "strategy": "dynamic_int8",
                    },
                }
            ],
        },
        model=model,
    )

    assert context.model is model
    assert not hasattr(context, "config")
    assert context.project_name == "create_context_workflow_mapping"
    assert context.compression_axes == ["precision", "sparsity"]


def test_create_context_accepts_workflow_path(tmp_path: Path) -> None:
    workflow_path = tmp_path / "workflow.yaml"
    workflow_path.write_text(
        "\n".join(
            [
                "project:",
                "  name: create_context_workflow_path",
                f"  artifact_dir: {tmp_path / 'artifacts'}",
                "model:",
                "  target: torch.nn:Identity",
                "compression_axes:",
                "  - precision",
                "stages:",
                "  - name: benchmark_stage",
                "    kind: benchmark",
                "    params:",
                "      warmup: 1",
                "      iterations: 2",
            ]
        ),
        encoding="utf-8",
    )

    context = create_context(workflow_path, model=torch.nn.Identity().eval())

    assert not hasattr(context, "config")
    assert context.project_name == "create_context_workflow_path"
    assert context.compression_axes == ["precision"]


def test_create_context_rejects_legacy_recipe_mapping() -> None:
    with pytest.raises(XQTConfigError) as exc_info:
        create_context(
            {
                "project": {
                    "name": "create_context_rejects_legacy_mapping",
                    "artifact_dir": "artifacts/xqt/tests/create_context_rejects_legacy_mapping",
                },
                "model": {"target": "torch.nn:Identity"},
                "compression": {
                    "quant": {
                        "enabled": True,
                        "backend": "torchao",
                        "strategy": "dynamic_int8",
                    }
                },
            }
        )

    assert "OptimizationConfig does not accept removed recipe top-level keys" in str(
        exc_info.value
    )


def test_xqt_context_has_no_legacy_config_view() -> None:
    context = XQTContext(
        device="cuda:7",
        artifact_dir="artifacts/xqt/tests/runtime_context",
        project_name="runtime_context",
        task_type="classification",
        compression_axes=["precision"],
        model_target="torch.nn:Identity",
        model_params={},
        quant_config=QuantConfig(),
        prune_config=PruneConfig(),
        export_targets=[ExportTargetConfig(format="onnx", output_path="model.onnx")],
    )

    assert not hasattr(context, "config")
    assert context.device == "cuda:7"
    assert context.compression_axes == ["precision"]
    assert context.export_targets[0].format == "onnx"


def test_load_model_pass_prefers_context_runtime_model_config() -> None:
    context = _legacy_context(
        artifact_dir="artifacts/xqt/tests/load_model_runtime_view",
        project_name="load_model_runtime_view",
    )
    context.model_target = "torch.nn.Linear"
    context.model_params = {"in_features": 3, "out_features": 2}

    output = LoadModelPass().run(context)

    assert output is context
    assert isinstance(output.model, torch.nn.Linear)
    assert output.model.in_features == 3
    assert output.model.out_features == 2


def test_run_quant_stage_sets_runtime_quant_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _legacy_context(
        model=torch.nn.Identity().eval(),
        artifact_dir="artifacts/xqt/tests/quant_runtime_view",
        project_name="quant_runtime_view",
    )
    context.quant_config = QuantConfig(
        enabled=True,
        backend="pytorch",
        strategy="dynamic_int8",
        policy={"dtype": "int8", "scheme": "dynamic"},
    )
    captured: dict[str, object] = {}

    def _fake_build_quantization_plan(config_value: QuantConfig) -> object:
        captured["backend"] = config_value.backend
        captured["strategy"] = config_value.strategy
        return object()

    class _FakeQuantExecution:
        def __init__(self, model: torch.nn.Module) -> None:
            self.model = model
            self.artifacts: dict[str, object] = {}
            self.reports: list[object] = []

    def _fake_execute_quantization_plan(
        context_value: XQTContext,
        plan: object,
        **kwargs: object,
    ) -> _FakeQuantExecution:
        del plan, kwargs
        return _FakeQuantExecution(context_value.require_model())

    monkeypatch.setattr(
        quant_stage_module,
        "build_quantization_plan",
        _fake_build_quantization_plan,
    )
    monkeypatch.setattr(
        quant_stage_module,
        "execute_quantization_plan",
        _fake_execute_quantization_plan,
    )
    monkeypatch.setattr(
        quant_stage_module,
        "summarize_quantization_reports",
        lambda reports: {"quantized_module_count": len(reports)},
    )
    monkeypatch.setattr(
        quant_stage_module,
        "_build_quant_layer_analysis_summary",
        lambda context, **kwargs: {"available": False, "reason": "test"},
    )

    output = run_quant_stage(
        context,
        QuantStageSpec(
            backend="pytorch",
            strategy="dynamic_int8",
        ),
    )

    assert output is context
    assert captured == {
        "backend": "pytorch",
        "strategy": "dynamic_int8",
    }
    assert output.quant_config is not None
    assert output.quant_config.backend == "pytorch"
    assert output.metrics["quant"]["layer_analysis"]["reason"] == "test"


def test_run_quant_stage_does_not_require_prepopulated_runtime_quant_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _legacy_context(
        model=torch.nn.Identity().eval(),
        artifact_dir="artifacts/xqt/tests/quant_runtime_required",
        project_name="quant_runtime_required",
    )
    context.quant_config = None

    monkeypatch.setattr(
        quant_stage_module, "build_quantization_plan", lambda resolved_quant: object()
    )
    monkeypatch.setattr(
        quant_stage_module,
        "execute_quantization_plan",
        lambda context_value, plan, **kwargs: SimpleNamespace(
            model=context_value.require_model(),
            artifacts={},
            reports=[],
        ),
    )
    monkeypatch.setattr(
        quant_stage_module, "summarize_quantization_reports", lambda reports: {}
    )
    monkeypatch.setattr(
        quant_stage_module,
        "_build_quant_layer_analysis_summary",
        lambda context, **kwargs: {"available": False, "reason": "test"},
    )

    output = run_quant_stage(
        context,
        QuantStageSpec(
            backend="pytorch",
            strategy="dynamic_int8",
        ),
    )

    assert output.quant_config is not None
    assert output.quant_config.backend == "pytorch"


def test_run_prune_stage_updates_runtime_prune_and_task_config() -> None:
    context = _legacy_context(
        model=torch.nn.Linear(4, 2).eval(),
        artifact_dir="artifacts/xqt/tests/prune_runtime_view",
        project_name="prune_runtime_view",
    )
    context.task_type = "detection"
    output = run_prune_stage(
        context,
        PruneStageSpec(
            method="global_l1_unstructured",
            target_sparsity=0.25,
        ),
    )

    assert output is context
    assert output.prune_config is not None
    assert output.prune_config.target_sparsity == pytest.approx(0.25)
    assert output.metrics["prune"]["target_sparsity"] == pytest.approx(0.25)
    assert output.metrics["prune"]["task_type"] == "detection"
    assert output.metrics["prune"]["baseline_kind"] == "unstructured_sparsity_report"


def test_run_prune_stage_records_sparse_runtime_capability() -> None:
    context = _legacy_context(
        model=torch.nn.Linear(8, 4, bias=False).eval(),
        artifact_dir="artifacts/xqt/tests/prune_sparse_runtime_capability",
        project_name="prune_sparse_runtime_capability",
    )

    output = run_prune_stage(
        context,
        PruneStageSpec(
            method="nm_structured",
            selection={"pattern": [2, 4]},
        ),
    )

    prune_report = output.metrics["prune"]
    assert prune_report["method"] == "nm_structured"
    assert prune_report["pattern_n"] == 2
    assert prune_report["pattern_m"] == 4
    assert prune_report["runtime_capability"]["method"] == "nm_structured"
    assert prune_report["runtime_capability"]["pattern_present"] is True
    assert prune_report["runtime_capability"]["speedup_verified"] is False


def test_optimize_model_updates_runtime_context_for_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_optimization_config(
        {
            "project": {
                "name": "stage_runtime_context",
                "artifact_dir": str(tmp_path / "artifacts"),
            },
            "model": {"target": "torch.nn:Identity"},
            "device": "cuda:5",
            "benchmark": {"warmup": 5, "iterations": 13},
            "stages": [
                {
                    "name": "bench_stage",
                    "kind": "benchmark",
                    "params": {
                        "warmup": 1,
                        "iterations": 2,
                    },
                }
            ],
        }
    )
    captured: dict[str, object] = {}

    def _fake_run_benchmark_stage(
        context: XQTContext,
        spec: BenchmarkStageSpec,
        *,
        base_benchmark_config: BenchmarkConfig | None = None,
    ) -> XQTContext:
        captured["device"] = context.device
        captured["artifact_dir"] = context.artifact_dir
        captured["warmup"] = spec.warmup
        captured["iterations"] = spec.iterations
        captured["base_iterations"] = (
            base_benchmark_config.iterations
            if base_benchmark_config is not None
            else None
        )
        context.metrics["benchmark"] = {
            "latency": {"p50_ms": 1.0},
        }
        return context

    monkeypatch.setattr(
        "xqt.workflows.optimization.run_benchmark_stage",
        _fake_run_benchmark_stage,
    )

    result = optimize_model(
        config,
        model=torch.nn.Identity().eval(),
        example_inputs=torch.randn(1, 4),
    )

    expected_dir = tmp_path / "artifacts" / "bench_stage"
    assert captured == {
        "device": "cuda:5",
        "artifact_dir": str(expected_dir),
        "warmup": 1,
        "iterations": 2,
        "base_iterations": 13,
    }
    assert result.context.device == "cuda:5"
    assert result.context.artifact_dir == str(expected_dir)
    assert not hasattr(result.context, "config")
    assert result.artifacts["workflow_result"] == expected_dir / "workflow_result.json"


def test_benchmark_pass_prefers_context_runtime_benchmark_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _legacy_context(
        model=torch.nn.Identity().eval(),
        example_inputs=torch.randn(1, 4),
        artifact_dir="artifacts/xqt/tests/benchmark_runtime_view",
        project_name="benchmark_runtime_view",
    )
    context.benchmark_config = BenchmarkConfig(
        warmup=3,
        iterations=7,
        sync_cuda=False,
        measure_memory=False,
    )
    captured: dict[str, object] = {}

    class _FakeBenchmarkReport:
        p50_ms = 1.0

        def to_dict(self) -> dict[str, object]:
            return {"p50_ms": 1.0}

    def _fake_benchmark_callable(
        fn: object,
        *,
        warmup: int,
        iterations: int,
        sync_cuda: bool,
        device: str,
    ) -> _FakeBenchmarkReport:
        del fn
        captured["warmup"] = warmup
        captured["iterations"] = iterations
        captured["sync_cuda"] = sync_cuda
        captured["device"] = device
        return _FakeBenchmarkReport()

    monkeypatch.setattr(passes_module, "benchmark_callable", _fake_benchmark_callable)
    monkeypatch.setattr(
        passes_module,
        "benchmark_memory",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("unexpected memory benchmark")
        ),
    )

    output = BenchmarkPass().run(context)

    assert output is context
    assert captured == {
        "warmup": 3,
        "iterations": 7,
        "sync_cuda": False,
        "device": "cpu",
    }
    assert output.metrics["benchmark"]["latency"]["p50_ms"] == 1.0


def test_operator_pass_prefers_context_runtime_configs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _legacy_context(
        model=torch.nn.Identity().eval(),
        example_inputs=torch.randn(1, 4),
        artifact_dir="artifacts/xqt/tests/operator_runtime_view",
        project_name="operator_runtime_view",
    )
    context.operator_config = OperatorOptimizationConfig(
        enabled=True,
        default_engine="tilelang",
        targets=[
            OperatorOptimizationTargetConfig(
                name="identity_target",
                target="",
                engine="tilelang",
            )
        ],
    )
    context.benchmark_config = BenchmarkConfig(
        warmup=2,
        iterations=4,
        sync_cuda=False,
        measure_memory=False,
    )
    captured: dict[str, object] = {}

    def _fake_build_operator_plan(
        config_value: OperatorOptimizationConfig,
    ) -> object:
        captured["default_engine"] = config_value.default_engine
        captured["target_count"] = len(config_value.targets)
        return object()

    class _FakeExecution:
        def __init__(self, model: torch.nn.Module) -> None:
            self.model = model
            self.artifacts: dict[str, object] = {}
            self.reports: list[object] = []

    def _fake_execute_operator_plan(
        context_value: XQTContext,
        plan: object,
        *,
        benchmark_config: BenchmarkConfig | None = None,
        device: str | None = None,
    ) -> _FakeExecution:
        del plan
        assert benchmark_config is not None
        captured["benchmark_warmup"] = benchmark_config.warmup
        captured["benchmark_iterations"] = benchmark_config.iterations
        captured["device"] = device
        return _FakeExecution(context_value.require_model())

    monkeypatch.setattr(
        passes_module,
        "build_operator_optimization_plan",
        _fake_build_operator_plan,
    )
    monkeypatch.setattr(
        passes_module,
        "execute_operator_optimization_plan",
        _fake_execute_operator_plan,
    )
    monkeypatch.setattr(
        passes_module,
        "summarize_operator_optimization_reports",
        lambda reports, candidate_reports=None: {
            "report_count": len(reports),
            "candidate_reports": candidate_reports,
        },
    )

    output = OperatorOptimizationPass().run(context)

    assert output is context
    assert captured == {
        "default_engine": "tilelang",
        "target_count": 1,
        "benchmark_warmup": 2,
        "benchmark_iterations": 4,
        "device": "cpu",
    }
    assert output.metrics["operator_optimization"]["report_count"] == 0


def test_operator_pass_summary_preserves_fallback_policy_and_reason() -> None:
    context = _legacy_context(
        model=torch.nn.Identity().eval(),
        example_inputs=torch.randn(1, 4),
        artifact_dir="artifacts/xqt/tests/operator_fallback_summary",
        project_name="operator_fallback_summary",
    )
    context.operator_config = OperatorOptimizationConfig(
        enabled=True,
        default_engine="tilelang",
        targets=[
            OperatorOptimizationTargetConfig(
                name="identity_target",
                target="",
                engine="tilelang",
                fallback="eager",
                fallback_policy="strict",
            )
        ],
    )
    context.benchmark_config = BenchmarkConfig(
        warmup=1,
        iterations=1,
        sync_cuda=False,
        measure_memory=False,
    )

    class _FakeExecution:
        def __init__(self, model: torch.nn.Module) -> None:
            from xqt.operator_opt.types import OperatorOptimizationReport

            self.model = model
            self.artifacts: dict[str, object] = {}
            self.reports = [
                OperatorOptimizationReport(
                    target_name="identity_target",
                    module_path="",
                    engine="tilelang",
                    runtime="pytorch",
                    applied=False,
                    fallback="eager",
                    fallback_policy="strict",
                    skip_reason="strict policy rejected fallback: requires CUDA tensors",
                    latency_before={"mean_ms": 1.0, "p50_ms": 1.0},
                    latency_after={"mean_ms": 1.0, "p50_ms": 1.0},
                    speedup=1.0,
                    numeric_diff={"allclose": True, "max_abs": 0.0, "mean_abs": 0.0},
                    metadata={
                        "execution_state": "fallback",
                        "fallback_policy": "strict",
                        "fallback_reason": "strict policy rejected fallback: requires CUDA tensors",
                    },
                )
            ]

    def _fake_execute_operator_plan(
        context_value: XQTContext,
        plan: object,
        *,
        benchmark_config: BenchmarkConfig | None = None,
        device: str | None = None,
    ) -> _FakeExecution:
        del plan, benchmark_config, device
        return _FakeExecution(context_value.require_model())

    from xqt.operator_opt.plan import build_operator_optimization_plan

    plan = build_operator_optimization_plan(context.operator_config)

    from unittest.mock import patch

    with patch(
        "xqt.pipeline.passes.execute_operator_optimization_plan",
        _fake_execute_operator_plan,
    ):
        output = OperatorOptimizationPass().run(context)

    summary = output.metrics["operator_optimization"]
    assert summary["fallback_policies"] == ["strict"]
    assert summary["fallback_reasons"] == [
        "strict policy rejected fallback: requires CUDA tensors"
    ]


def test_operator_pass_manifest_records_acceptance_decision() -> None:
    context = _legacy_context(
        model=torch.nn.Identity().eval(),
        example_inputs=torch.randn(1, 4),
        artifact_dir="artifacts/xqt/tests/operator_acceptance_manifest",
        project_name="operator_acceptance_manifest",
    )
    context.manifest = xqt.ArtifactManifest(project_name="operator_acceptance_manifest")
    context.operator_config = OperatorOptimizationConfig(
        enabled=True,
        default_engine="torch_compile",
        targets=[
            OperatorOptimizationTargetConfig(
                name="identity_compile",
                target="",
                engine="torch_compile",
                fallback="eager",
                min_speedup=1.5,
            )
        ],
    )
    context.benchmark_config = BenchmarkConfig(
        warmup=1,
        iterations=1,
        sync_cuda=False,
        measure_memory=False,
    )

    class _FakeExecution:
        def __init__(self, model: torch.nn.Module) -> None:
            from xqt.operator_opt.types import OperatorOptimizationReport

            self.model = model
            self.artifacts: dict[str, object] = {}
            self.reports = [
                OperatorOptimizationReport(
                    target_name="identity_compile",
                    module_path="",
                    engine="torch_compile",
                    runtime="pytorch",
                    applied=False,
                    fallback="eager",
                    fallback_policy="prefer_fallback",
                    skip_reason="speedup 1.0000 did not reach min_speedup 1.5000",
                    latency_before={"mean_ms": 1.0, "p50_ms": 1.0},
                    latency_after={"mean_ms": 1.0, "p50_ms": 1.0},
                    speedup=1.0,
                    numeric_diff={"allclose": True, "max_abs": 0.0, "mean_abs": 0.0},
                    metadata={
                        "execution_state": "fallback",
                        "fallback_policy": "prefer_fallback",
                        "fallback_reason": "speedup 1.0000 did not reach min_speedup 1.5000",
                        "mode": "reduce-overhead",
                        "dynamic": True,
                        "fullgraph": False,
                        "min_speedup": 1.5,
                        "effective_min_speedup": 1.5,
                        "graph_break_report": {
                            "backend": "torch_compile",
                            "status": "ok",
                            "error": None,
                            "graph_count": 1,
                            "graph_break_count": 2,
                            "break_reasons": ["dynamic shape"],
                            "break_details": [
                                {
                                    "index": 0,
                                    "reason": "dynamic shape",
                                    "graph_break": True,
                                    "user_stack": "forward:1",
                                }
                            ],
                            "op_count": 3,
                            "compile_times": "compile_inner 1.0",
                            "mode": "reduce-overhead",
                            "dynamic": True,
                            "fullgraph": False,
                        },
                        "planned_skip_report": {
                            "target_name": "identity_compile",
                            "engine": "torch_compile",
                            "status": "available",
                            "execution_status": "available",
                            "reason": "speedup 1.0000 did not reach min_speedup 1.5000",
                            "requires_cuda": False,
                            "cuda_available": False,
                            "missing_requirements": [],
                        },
                        "quant_runtime_guard": {
                            "guard_applies": False,
                            "reason": None,
                            "backend": "torchao",
                            "runtime": "pytorch",
                            "runtime_source": "quant.runtime",
                            "component_count": 1,
                            "target_name": "identity_compile",
                            "target_engine": "torch_compile",
                            "allowed_runtimes": ["pytorch", "torch"],
                        },
                        "numeric_validation": {
                            "target_name": "identity_compile",
                            "target_path": "",
                            "benchmark_target_path": "",
                            "status": "passed",
                            "reason": None,
                            "allclose": True,
                            "thresholds": {"atol": 1e-5, "rtol": 1e-5},
                            "max_abs": 0.0,
                            "mean_abs": 0.0,
                            "cosine_similarity": 1.0,
                        },
                        "fallback_detail": {
                            "graph_break_count": 2,
                            "graph_breaks": ["dynamic shape"],
                        },
                    },
                )
            ]

    def _fake_execute_operator_plan(
        context_value: XQTContext,
        plan: object,
        *,
        benchmark_config: BenchmarkConfig | None = None,
        device: str | None = None,
    ) -> _FakeExecution:
        del plan, benchmark_config, device
        return _FakeExecution(context_value.require_model())

    from unittest.mock import patch

    with patch(
        "xqt.pipeline.passes.execute_operator_optimization_plan",
        _fake_execute_operator_plan,
    ):
        output = OperatorOptimizationPass().run(context)

    assert output.manifest is not None
    metric = next(
        item
        for item in output.manifest.metrics
        if item.name == "operator_optimization.identity_compile.applied"
    )
    assert metric.value is False
    assert metric.metadata["decision"] == "rejected_min_speedup"
    assert metric.metadata["meets_speedup"] is False
    assert metric.metadata["meets_numeric"] is True
    assert metric.metadata["graph_break_count"] == 2
    assert metric.metadata["graph_breaks"] == ["dynamic shape"]
    assert metric.metadata["mode"] == "reduce-overhead"
    assert metric.metadata["dynamic"] is True
    assert metric.metadata["fullgraph"] is False
    assert metric.metadata["planned_skip_report"]["engine"] == "torch_compile"
    assert metric.metadata["planned_skip_report"]["execution_status"] == "available"
    assert metric.metadata["quant_runtime_guard"]["guard_applies"] is False
    assert metric.metadata["quant_runtime_guard"]["runtime"] == "pytorch"
    assert metric.metadata["numeric_validation"]["status"] == "passed"
    assert metric.metadata["numeric_validation"]["allclose"] is True
    assert (
        metric.metadata["graph_break_report"]["break_details"][0]["reason"]
        == "dynamic shape"
    )
    assert metric.metadata["acceptance"]["effective_min_speedup"] == 1.5


def test_preflight_rejects_invalid_operator_fallback_policy() -> None:
    workflow = {
        "project": {
            "name": "invalid_operator_fallback_policy",
            "artifact_dir": "artifacts/xqt/tests/invalid_operator_fallback_policy",
        },
        "model": {"target": "torch.nn:Identity"},
        "stages": [
            {
                "name": "operator_stage",
                "kind": "operator",
                "params": {
                    "default_engine": "tilelang",
                    "targets": [
                        {
                            "name": "identity_target",
                            "engine": "tilelang",
                            "fallback_policy": "invalid_policy",
                        }
                    ],
                },
            }
        ],
    }

    report = preflight_optimization_config(workflow)
    checks = {check.name: check for check in report.checks}
    fallback_policy_check = checks["stages.0.operator_stage.targets.0.fallback_policy"]

    assert fallback_policy_check.passed is False
    assert fallback_policy_check.level == "error"
    assert "strict" in fallback_policy_check.message
    assert "prefer_fallback" in fallback_policy_check.message


def test_typed_stage_helpers_project_stage_specs_to_runtime_configs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_benchmark = BenchmarkConfig(
        warmup=9,
        iterations=11,
        sync_cuda=False,
        measure_memory=False,
    )
    context = _legacy_context(
        model=torch.nn.Identity().eval(),
        example_inputs=torch.randn(1, 4),
        artifact_dir="artifacts/xqt/tests/typed_stage_helper_projection",
        project_name="typed_stage_helper_projection",
    )
    captured: dict[str, object] = {}

    def _fake_operator_run(
        self: object,
        context_value: XQTContext,
        *,
        operator_config: OperatorOptimizationConfig | None = None,
        benchmark_config: BenchmarkConfig | None = None,
    ) -> XQTContext:
        del self
        assert operator_config is not None
        assert benchmark_config is not None
        captured["operator_engine"] = operator_config.default_engine
        captured["operator_enabled"] = operator_config.enabled
        captured["operator_benchmark_warmup"] = benchmark_config.warmup
        captured["operator_benchmark_iterations"] = benchmark_config.iterations
        return context_value

    def _fake_analyze_run(
        self: object,
        context_value: XQTContext,
        *,
        analysis_config: object | None = None,
        output_diff: object | None = None,
    ) -> XQTContext:
        del self, output_diff
        assert analysis_config is not None
        captured["analysis_enabled"] = analysis_config.enabled
        captured["analysis_top_k"] = analysis_config.top_k
        captured["analysis_statistics"] = analysis_config.include_statistics
        return context_value

    def _fake_benchmark_run(
        self: object,
        context_value: XQTContext,
        *,
        benchmark_config: BenchmarkConfig | None = None,
    ) -> XQTContext:
        del self
        assert benchmark_config is not None
        captured["benchmark_warmup"] = benchmark_config.warmup
        captured["benchmark_iterations"] = benchmark_config.iterations
        return context_value

    def _fake_export_run(
        self: object,
        context_value: XQTContext,
        **kwargs: object,
    ) -> XQTContext:
        del self
        targets = kwargs["targets"]
        output_diff = kwargs["output_diff"]
        captured["export_target_count"] = len(targets)
        captured["export_stage_kind"] = kwargs["stage_kind"]
        captured["export_atol"] = output_diff.atol
        captured["export_runtime_handle"] = kwargs["runtime_handle_request"]
        return context_value

    monkeypatch.setattr(
        passes_module.OperatorOptimizationPass,
        "run",
        _fake_operator_run,
    )
    monkeypatch.setattr(passes_module.AnalyzePass, "run", _fake_analyze_run)
    monkeypatch.setattr(passes_module.BenchmarkPass, "run", _fake_benchmark_run)
    monkeypatch.setattr(
        "xqt.pipeline.export_pass.ExportPass.run",
        _fake_export_run,
    )

    run_operator_stage(
        context,
        OperatorStageSpec(
            default_engine="tilelang",
            benchmark=BenchmarkStageSpec(warmup=2),
        ),
        base_benchmark_config=base_benchmark,
    )
    run_analyze_stage(context, AnalyzeStageSpec(top_k=3, include_statistics=True))
    run_benchmark_stage(
        context,
        BenchmarkStageSpec(iterations=4),
        base_benchmark_config=base_benchmark,
    )
    run_export_stage(
        context,
        ExportStageSpec(
            targets=[ExportTargetConfig(format="onnx", output_path="model.onnx")],
            validate=OutputDiffConfig(atol=2e-4),
        ),
    )

    assert captured == {
        "operator_engine": "tilelang",
        "operator_enabled": True,
        "operator_benchmark_warmup": 2,
        "operator_benchmark_iterations": 11,
        "analysis_enabled": True,
        "analysis_top_k": 3,
        "analysis_statistics": True,
        "benchmark_warmup": 9,
        "benchmark_iterations": 4,
        "export_target_count": 1,
        "export_stage_kind": "export",
        "export_atol": pytest.approx(2e-4),
        "export_runtime_handle": None,
    }
    assert context.operator_config is not None
    assert context.operator_config.default_engine == "tilelang"
    assert context.analysis_config is not None
    assert context.analysis_config.top_k == 3
    assert context.benchmark_config is not None
    assert context.benchmark_config.iterations == 4
    assert context.output_diff_config is not None
    assert context.output_diff_config.atol == pytest.approx(2e-4)


def test_xdl_setup_adapter_accepts_optimization_config(tmp_path: Path) -> None:
    config = load_optimization_config(
        {
            "project": {
                "name": "xdl_setup_adapter_workflow",
                "artifact_dir": str(tmp_path / "artifacts"),
            },
            "model": {"target": "torch.nn:Identity"},
        }
    )
    setup = SimpleNamespace(model=torch.nn.Identity().eval(), device="cpu")

    context = xqt.xdl_setup_to_xqt_context(setup, config)

    assert context.model is setup.model
    assert not hasattr(context, "config")
    assert context.project_name == "xdl_setup_adapter_workflow"
    assert context.metrics["xdl_setup"]["device"] == "cpu"


def test_xdl_setup_adapter_rejects_legacy_recipe_mapping() -> None:
    setup = SimpleNamespace(model=torch.nn.Identity().eval(), device="cpu")

    with pytest.raises(XQTConfigError) as exc_info:
        xqt.xdl_setup_to_xqt_context(
            setup,
            {
                "project": {
                    "name": "legacy_xdl_setup_adapter_mapping",
                    "artifact_dir": "artifacts/xqt/tests/legacy_xdl_setup_adapter_mapping",
                },
                "model": {"target": "torch.nn:Identity"},
                "compression": {
                    "quant": {
                        "enabled": True,
                        "backend": "torchao",
                        "strategy": "dynamic_int8",
                    }
                },
            },
        )

    assert "OptimizationConfig does not accept removed recipe top-level keys" in str(
        exc_info.value
    )


def test_xdl_checkpoint_adapter_accepts_optimization_config(tmp_path: Path) -> None:
    config = load_optimization_config(
        {
            "project": {
                "name": "xdl_checkpoint_adapter_workflow",
                "artifact_dir": str(tmp_path / "artifacts"),
            },
            "model": {
                "target": "torch.nn:Linear",
                "params": {"in_features": 4, "out_features": 4},
            },
        }
    )
    source_model = torch.nn.Linear(4, 4).eval()
    checkpoint_path = tmp_path / "model.pt"
    torch.save(source_model.state_dict(), checkpoint_path)
    target_model = torch.nn.Linear(4, 4).eval()

    context = xqt.xdl_checkpoint_to_xqt_context(
        target_model,
        checkpoint_path,
        config,
    )

    assert context.model is target_model
    assert not hasattr(context, "config")
    assert context.project_name == "xdl_checkpoint_adapter_workflow"
    assert context.manifest is not None
    assert context.manifest.source_checkpoint == str(checkpoint_path)


def test_optimization_workflow_config_accepts_benchmark_defaults() -> None:
    config = load_optimization_config(
        {
            "project": {
                "name": "benchmark_defaults",
                "artifact_dir": "artifacts/xqt/tests/benchmark_defaults",
            },
            "benchmark": {
                "warmup": 3,
                "iterations": 7,
                "sync_cuda": False,
                "measure_memory": False,
            },
        }
    )

    assert config.benchmark.warmup == 3
    assert config.benchmark.iterations == 7
    assert config.benchmark.sync_cuda is False
    assert config.benchmark.measure_memory is False


def test_optimization_workflow_config_accepts_operator_stage_benchmark_override() -> (
    None
):
    config = load_optimization_config(
        {
            "project": {
                "name": "operator_benchmark_override",
                "artifact_dir": "artifacts/xqt/tests/operator_benchmark_override",
            },
            "stages": [
                {
                    "name": "tilelang_operator",
                    "kind": "operator",
                    "params": {
                        "benchmark": {
                            "warmup": 5,
                            "iterations": 9,
                            "sync_cuda": False,
                            "measure_memory": False,
                        },
                        "targets": [],
                    },
                }
            ],
        }
    )

    stage = config.stages[0]
    assert isinstance(stage.spec, OperatorStageSpec)
    benchmark = stage.spec.benchmark
    assert benchmark is not None
    assert benchmark.warmup == 5
    assert benchmark.iterations == 9
    assert benchmark.sync_cuda is False
    assert benchmark.measure_memory is False


def test_optimization_workflow_config_types_export_validate_and_deploy_runtime_handle() -> (
    None
):
    config = load_optimization_config(
        {
            "project": {
                "name": "export_validate_and_deploy_runtime_handle",
                "artifact_dir": "artifacts/xqt/tests/export_validate_and_deploy_runtime_handle",
            },
            "stages": [
                {
                    "name": "export_model",
                    "kind": "export",
                    "params": {
                        "targets": [
                            {
                                "format": "onnx",
                                "output_path": "artifacts/model.onnx",
                                "onnx": {
                                    "dynamo": False,
                                    "runtime_diff": False,
                                    "optimization": {
                                        "enabled": True,
                                        "level": "basic",
                                    },
                                },
                            }
                        ],
                        "validate": {"atol": 1e-4, "rtol": 1e-3},
                    },
                },
                {
                    "name": "deploy_model",
                    "kind": "deploy",
                    "params": {
                        "targets": [
                            {
                                "format": "tensorrt",
                                "output_path": "artifacts/model.engine",
                                "tensorrt": {"dry_run": True},
                            }
                        ],
                        "runtime_handle": {
                            "runtime": "tensorrt",
                            "handle_kind": "engine",
                            "materialize": False,
                            "tensorrt": {
                                "device": "cuda:1",
                                "plugin_libraries": ["plugins/custom.so"],
                            },
                        },
                    },
                },
            ],
        }
    )

    export_stage = config.stages[0]
    assert export_stage.kind == "export"
    assert export_stage.spec.validate is not None
    assert export_stage.spec.validate.atol == pytest.approx(1e-4)
    assert export_stage.spec.validate.rtol == pytest.approx(1e-3)
    export_target = export_stage.spec.targets[0]
    assert export_target.onnx.dynamo is False
    assert export_target.onnx.runtime_diff is False
    assert export_target.onnx.optimization.enabled is True
    assert export_target.onnx.optimization.level == "basic"

    deploy_stage = config.stages[1]
    assert isinstance(deploy_stage.spec, DeployStageSpec)
    assert deploy_stage.spec.targets[0].tensorrt.dry_run is True
    assert deploy_stage.spec.runtime_handle is not None
    assert deploy_stage.spec.runtime_handle.runtime == "tensorrt"
    assert deploy_stage.spec.runtime_handle.handle_kind == "engine"
    assert deploy_stage.spec.runtime_handle.materialize is False
    assert deploy_stage.spec.runtime_handle.tensorrt.device == "cuda:1"
    assert deploy_stage.spec.runtime_handle.tensorrt.plugin_libraries == [
        "plugins/custom.so"
    ]


def test_optimization_workflow_config_rejects_legacy_onnx_target_params() -> None:
    with pytest.raises(XQTConfigError, match="legacy ONNX keys"):
        load_optimization_config(
            {
                "stages": [
                    {
                        "name": "export_model",
                        "kind": "export",
                        "params": {
                            "targets": [
                                {
                                    "format": "onnx",
                                    "output_path": "artifacts/model.onnx",
                                    "params": {"dynamo": False},
                                }
                            ]
                        },
                    }
                ]
            }
        )


def test_optimization_workflow_config_rejects_legacy_tensorrt_target_params() -> None:
    with pytest.raises(XQTConfigError, match="legacy TensorRT keys"):
        load_optimization_config(
            {
                "stages": [
                    {
                        "name": "export_model",
                        "kind": "export",
                        "params": {
                            "targets": [
                                {
                                    "format": "tensorrt",
                                    "output_path": "artifacts/model.engine",
                                    "params": {"dry_run": True},
                                }
                            ]
                        },
                    }
                ]
            }
        )


def test_optimization_workflow_config_rejects_legacy_openvino_target_params() -> None:
    with pytest.raises(XQTConfigError, match="legacy OpenVINO keys"):
        load_optimization_config(
            {
                "stages": [
                    {
                        "name": "export_model",
                        "kind": "export",
                        "params": {
                            "targets": [
                                {
                                    "format": "openvino",
                                    "output_path": "artifacts/model.xml",
                                    "params": {"dry_run": True},
                                }
                            ]
                        },
                    }
                ]
            }
        )


def test_optimization_workflow_config_rejects_legacy_torch_export_target_params() -> (
    None
):
    with pytest.raises(XQTConfigError, match="legacy TorchExport keys"):
        load_optimization_config(
            {
                "stages": [
                    {
                        "name": "export_model",
                        "kind": "export",
                        "params": {
                            "targets": [
                                {
                                    "format": "torch_export",
                                    "output_path": "artifacts/model.pt2",
                                    "params": {"strict": True},
                                }
                            ]
                        },
                    }
                ]
            }
        )


def test_optimization_workflow_config_rejects_legacy_torchscript_target_params() -> (
    None
):
    with pytest.raises(XQTConfigError, match="legacy TorchScript keys"):
        load_optimization_config(
            {
                "stages": [
                    {
                        "name": "export_model",
                        "kind": "export",
                        "params": {
                            "targets": [
                                {
                                    "format": "torchscript",
                                    "output_path": "artifacts/model.pt",
                                    "params": {"method": "script"},
                                }
                            ]
                        },
                    }
                ]
            }
        )


def test_optimization_workflow_config_rejects_legacy_executorch_target_params() -> None:
    with pytest.raises(XQTConfigError, match="legacy ExecuTorch keys"):
        load_optimization_config(
            {
                "stages": [
                    {
                        "name": "export_model",
                        "kind": "export",
                        "params": {
                            "targets": [
                                {
                                    "format": "executorch",
                                    "output_path": "artifacts/model.pte",
                                    "params": {"dry_run": True},
                                }
                            ]
                        },
                    }
                ]
            }
        )


def test_optimization_workflow_config_rejects_legacy_ncnn_target_params() -> None:
    with pytest.raises(XQTConfigError, match="legacy ncnn keys"):
        load_optimization_config(
            {
                "stages": [
                    {
                        "name": "export_model",
                        "kind": "export",
                        "params": {
                            "targets": [
                                {
                                    "format": "ncnn",
                                    "output_path": "artifacts/model.param",
                                    "params": {"onnx_path": "artifacts/model.onnx"},
                                }
                            ]
                        },
                    }
                ]
            }
        )


def test_optimization_workflow_config_rejects_legacy_mnn_target_params() -> None:
    with pytest.raises(XQTConfigError, match="legacy MNN keys"):
        load_optimization_config(
            {
                "stages": [
                    {
                        "name": "export_model",
                        "kind": "export",
                        "params": {
                            "targets": [
                                {
                                    "format": "mnn",
                                    "output_path": "artifacts/model.mnn",
                                    "params": {"onnx_path": "artifacts/model.onnx"},
                                }
                            ]
                        },
                    }
                ]
            }
        )


def test_optimization_workflow_config_rejects_invalid_ncnn_converter() -> None:
    with pytest.raises(
        XQTConfigError,
        match="converter must be onnx2ncnn or pnnx",
    ):
        load_optimization_config(
            {
                "stages": [
                    {
                        "name": "export_model",
                        "kind": "export",
                        "params": {
                            "targets": [
                                {
                                    "format": "ncnn",
                                    "output_path": "artifacts/model.param",
                                    "ncnn": {"converter": "invalid"},
                                }
                            ]
                        },
                    }
                ]
            }
        )


def test_optimization_workflow_config_rejects_invalid_torchscript_method() -> None:
    with pytest.raises(XQTConfigError, match="method must be trace or script"):
        load_optimization_config(
            {
                "stages": [
                    {
                        "name": "export_model",
                        "kind": "export",
                        "params": {
                            "targets": [
                                {
                                    "format": "torchscript",
                                    "output_path": "artifacts/model.pt",
                                    "torchscript": {"method": "compile"},
                                }
                            ]
                        },
                    }
                ]
            }
        )


def test_optimization_workflow_config_rejects_legacy_runtime_handle_params() -> None:
    with pytest.raises(XQTConfigError, match="runtime_handle.params is removed"):
        load_optimization_config(
            {
                "stages": [
                    {
                        "name": "deploy_model",
                        "kind": "deploy",
                        "params": {
                            "targets": [],
                            "runtime_handle": {
                                "runtime": "onnxruntime",
                                "materialize": True,
                                "params": {"providers": ["CPUExecutionProvider"]},
                            },
                        },
                    }
                ]
            }
        )


def test_preflight_uses_typed_onnx_target_config() -> None:
    report = preflight_optimization_config(
        {
            "stages": [
                {
                    "name": "export_model",
                    "kind": "export",
                    "params": {
                        "targets": [
                            {
                                "format": "onnx",
                                "output_path": "artifacts/model.onnx",
                                "onnx": {
                                    "runtime_diff": False,
                                    "optimization": {"enabled": True},
                                },
                            }
                        ]
                    },
                }
            ]
        }
    )
    check_names = [check.name for check in report.checks]

    assert "dependency.onnx" in check_names
    assert check_names.count("dependency.onnxruntime") == 1


def test_preflight_uses_typed_openvino_target_config() -> None:
    report = preflight_optimization_config(
        {
            "stages": [
                {
                    "name": "export_model",
                    "kind": "export",
                    "params": {
                        "targets": [
                            {
                                "format": "openvino",
                                "output_path": "artifacts/model.xml",
                                "openvino": {"dry_run": True},
                            }
                        ]
                    },
                }
            ]
        }
    )
    checks = {check.name: check for check in report.checks}

    assert checks["dependency.openvino"].passed is True
    assert checks["dependency.openvino"].metadata["dry_run"] is True


def test_preflight_uses_typed_mobile_target_configs() -> None:
    report = preflight_optimization_config(
        {
            "stages": [
                {
                    "name": "export_model",
                    "kind": "export",
                    "params": {
                        "targets": [
                            {
                                "format": "executorch",
                                "output_path": "artifacts/model.pte",
                                "executorch": {"dry_run": True},
                            },
                            {
                                "format": "ncnn",
                                "output_path": "artifacts/model.param",
                                "ncnn": {
                                    "converter": "pnnx",
                                    "pnnx_path": "missing-pnnx",
                                    "dry_run": True,
                                },
                            },
                            {
                                "format": "mnn",
                                "output_path": "artifacts/model.mnn",
                                "mnn": {
                                    "converter_path": "missing-mnnconvert",
                                    "dry_run": True,
                                },
                            },
                        ]
                    },
                }
            ]
        }
    )
    checks = {check.name: check for check in report.checks}

    assert checks["dependency.executorch"].passed is True
    assert checks["dependency.executorch"].metadata["dry_run"] is True
    assert checks["stages.0.export_model.targets.1.ncnn.pnnx"].passed is True
    assert (
        checks["stages.0.export_model.targets.1.ncnn.pnnx"].metadata["dry_run"] is True
    )
    assert checks["stages.0.export_model.targets.2.mnn.MNNConvert"].passed is True
    assert (
        checks["stages.0.export_model.targets.2.mnn.MNNConvert"].metadata["dry_run"]
        is True
    )


def test_preflight_checks_materialized_typed_runtime_handle() -> None:
    report = preflight_optimization_config(
        {
            "stages": [
                {
                    "name": "deploy_onnxruntime",
                    "kind": "deploy",
                    "params": {
                        "targets": [
                            {
                                "format": "onnx",
                                "output_path": "artifacts/model.onnx",
                                "onnx": {"runtime_diff": False},
                            }
                        ],
                        "runtime_handle": {
                            "runtime": "onnxruntime",
                            "handle_kind": "inference_session",
                            "materialize": True,
                            "onnxruntime": {
                                "providers": ["CPUExecutionProvider"],
                            },
                        },
                    },
                }
            ]
        }
    )
    checks = {check.name: check for check in report.checks}

    runtime_check = checks["stages.0.deploy_onnxruntime.runtime_handle.runtime"]
    assert runtime_check.passed is True
    assert runtime_check.metadata["runtime"] == "onnxruntime"
    target_check = checks["stages.0.deploy_onnxruntime.runtime_handle.targets"]
    assert target_check.passed is True
    assert target_check.metadata["target_count"] == 1
    assert "dependency.onnxruntime" in checks


def test_optimization_workflow_config_rejects_materialized_runtime_handle_with_multiple_onnx_targets() -> None:
    with pytest.raises(
        XQTConfigError,
        match="requires exactly one ONNX target",
    ):
        load_optimization_config(
            {
                "stages": [
                    {
                        "name": "deploy_onnxruntime",
                        "kind": "deploy",
                        "params": {
                            "targets": [
                                {
                                    "format": "onnx",
                                    "output_path": "artifacts/model_a.onnx",
                                    "onnx": {"runtime_diff": False},
                                },
                                {
                                    "format": "onnx",
                                    "output_path": "artifacts/model_b.onnx",
                                    "onnx": {"runtime_diff": False},
                                },
                            ],
                            "runtime_handle": {
                                "runtime": "onnxruntime",
                                "handle_kind": "inference_session",
                                "materialize": True,
                            },
                        },
                    }
                ]
            }
        )


def test_optimization_workflow_config_rejects_empty_export_targets() -> None:
    with pytest.raises(
        XQTConfigError,
        match="must declare at least one export target",
    ):
        load_optimization_config(
            {
                "stages": [
                    {
                        "name": "empty_export",
                        "kind": "export",
                        "params": {"targets": []},
                    }
                ]
            }
        )


def test_optimization_workflow_config_rejects_old_top_level_schema() -> None:
    with pytest.raises(XQTConfigError) as exc_info:
        load_optimization_config(
            {
                "project": {
                    "name": "old_schema",
                    "artifact_dir": "artifacts/xqt/tests/old_schema",
                },
                "compression": {
                    "quant": {
                        "enabled": True,
                        "backend": "torchao",
                        "strategy": "dynamic_int8",
                    }
                },
            }
        )
    assert "compression -> split into stages[*].params for quant / prune stages" in str(
        exc_info.value
    )


def test_load_optimization_config_rebuilds_existing_stage_spec() -> None:
    config = load_optimization_config(
        {
            "project": {
                "name": "rebuild_stage_spec",
                "artifact_dir": "artifacts/xqt/tests/rebuild_stage_spec",
            },
            "stages": [
                {
                    "name": "quant_model",
                    "kind": "quant",
                    "params": {
                        "backend": "torchao",
                        "strategy": "dynamic_int8",
                    },
                }
            ],
        }
    )

    stage = config.stages[0]
    assert isinstance(stage.spec, QuantStageSpec)
    assert stage.spec.strategy == "dynamic_int8"

    stage.params["strategy"] = "fp8_dynamic"
    reloaded = load_optimization_config(config)

    assert reloaded is config
    assert isinstance(stage.spec, QuantStageSpec)
    assert stage.spec.strategy == "fp8_dynamic"


def test_run_stage_rebuilds_existing_stage_spec(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage = OptimizationStageConfig(
        name="direct_quant",
        kind="quant",
        params={
            "backend": "torchao",
            "strategy": "dynamic_int8",
        },
    )
    ensure_stage_spec(stage)
    assert isinstance(stage.spec, QuantStageSpec)
    assert stage.spec.strategy == "dynamic_int8"

    stage.params["strategy"] = "fp8_dynamic"
    captured: dict[str, str | None] = {}

    def fake_run_stage(
        state: object,
        incoming_stage: OptimizationStageConfig,
    ) -> OptimizationStageResult:
        assert isinstance(incoming_stage.spec, QuantStageSpec)
        captured["strategy"] = incoming_stage.spec.strategy
        return OptimizationStageResult(
            name=incoming_stage.name,
            kind=incoming_stage.kind,
            accepted=True,
        )

    monkeypatch.setattr(optimization_module, "_run_optimization_stage", fake_run_stage)
    result = XQTOptimizationSession().run_stage(stage)

    assert result.accepted is True
    assert captured["strategy"] == "fp8_dynamic"


def test_packaged_recipes_use_single_workflow_schema() -> None:
    recipe_paths = sorted(Path("xqt/recipes").glob("**/*.yaml"))
    allowed_top_level = {field.name for field in fields(OptimizationConfig)}
    removed_top_level = {
        "analysis",
        "compression",
        "config_version",
        "export",
        "operator_optimization",
        "validation",
    }

    assert recipe_paths
    for path in recipe_paths:
        raw = OmegaConf.to_container(OmegaConf.load(path), resolve=False)
        assert isinstance(raw, dict)
        assert removed_top_level.isdisjoint(raw), path
        assert set(raw).issubset(allowed_top_level), path

        config = load_optimization_config(path)
        assert config.stages, path


def test_stage_entrypoints_accept_typed_stage_specs() -> None:
    expected = {
        run_quant_stage: "QuantStageSpec",
        run_prune_stage: "PruneStageSpec",
        run_operator_stage: "OperatorStageSpec",
        run_export_stage: "ExportStageSpec | DeployStageSpec",
        run_analyze_stage: "AnalyzeStageSpec",
        run_benchmark_stage: "BenchmarkStageSpec",
    }

    for function, expected_spec in expected.items():
        signature = inspect.signature(function)
        spec_annotation = str(signature.parameters["spec"].annotation)
        assert spec_annotation == expected_spec


def test_recipe_tree_does_not_keep_placeholder_recipe_directories() -> None:
    recipe_root = Path("xqt/recipes")
    yaml_parent_dirs = {path.parent for path in recipe_root.glob("**/*.yaml")}
    placeholder_dirs = []

    for directory in sorted(recipe_root.glob("**")):
        if not directory.is_dir() or directory == recipe_root:
            continue
        if directory.name == "__pycache__":
            continue
        files = [path for path in directory.iterdir() if path.is_file()]
        if files and all(path.name == "AGENTS.md" for path in files):
            if directory not in yaml_parent_dirs:
                placeholder_dirs.append(directory.as_posix())

    assert placeholder_dirs == []


def test_default_workflow_runs_without_external_inputs() -> None:
    result = optimize_model(DEFAULT_CONFIG, write_outputs=False)

    assert [stage.name for stage in result.stages] == ["prune_l1"]
    assert result.stages[0].accepted is True
    assert result.best_stage == "prune_l1"


def test_pyproject_exposes_only_workflow_console_script() -> None:
    with open("pyproject.toml", "rb") as handle:
        pyproject = tomllib.load(handle)

    scripts = pyproject["project"]["scripts"]
    assert scripts["xqt-run-workflow"] == "xqt.run_workflow:main"
    assert "xqt-run-recipe" not in scripts
    assert "xqt-preflight" not in scripts

    package_data = pyproject["tool"]["setuptools"]["package-data"]["xqt"]
    assert "recipes/*/*.yaml" in package_data
    assert "recipes/*/*/*.yaml" in package_data
