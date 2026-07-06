from __future__ import annotations

from unittest.mock import patch

import torch

from xqt.core.config import load_xqt_config
from xqt.operator_opt.backends.cute_dsl import (
    build_cute_dsl_artifact_metadata,
    list_cute_dsl_kernel_specs,
)
from xqt.operator_opt.capability import describe_operator_engine_capability
from xqt.operator_opt.executor import (
    build_operator_optimization_plan,
    execute_operator_optimization_plan,
)
from xqt.pipeline.preflight import preflight_xqt_config
from xqt.pipeline.runner import create_context


def _cute_dsl_operator_config() -> dict[str, object]:
    return {
        "config_version": 1,
        "project": {
            "name": "cute_dsl_operator",
            "artifact_dir": "artifacts/xqt/tests/cute_dsl_operator",
        },
        "model": {
            "target": "xqt.operator_opt.toy_models.build_toy_dequant_gemm_block",
            "params": {
                "input_dim": 16,
                "output_dim": 8,
                "activation": "silu",
            },
            "device": "cpu",
        },
        "operator_optimization": {
            "enabled": True,
            "default_engine": "cute_dsl",
            "targets": [
                {
                    "name": "model",
                    "engine": "cute_dsl",
                    "patterns": ["gemm_epilogue"],
                    "cute_dsl": {
                        "target_arch": "sm_90",
                        "cache_dir": "artifacts/xqt/tests/cute_dsl_operator/cache",
                        "tile_shape": [128, 128, 64],
                        "cluster_shape": [1, 1, 1],
                        "pass_configs": {
                            "CUTE_DSL_ENABLE_EPILOGUE_FUSION": True,
                        },
                    },
                }
            ],
        },
        "benchmark": {
            "warmup": 1,
            "iterations": 1,
            "sync_cuda": False,
        },
    }


def test_cute_dsl_registry_reports_gemm_patterns() -> None:
    specs = list_cute_dsl_kernel_specs()

    assert set(specs) == {"gemm_epilogue", "grouped_gemm"}
    assert specs["gemm_epilogue"]["metadata"]["production_status"] == "reference_guarded"
    assert specs["grouped_gemm"]["metadata"]["production_status"] == "metadata_only"


def test_cute_dsl_artifact_metadata_is_stable() -> None:
    metadata = build_cute_dsl_artifact_metadata("gemm_epilogue")

    assert metadata["engine"] == "cute_dsl"
    assert metadata["pattern"] == "gemm_epilogue"
    assert metadata["compile_status"] == "metadata_only"
    assert metadata["exportable"] is False


def test_cute_dsl_capability_reports_missing_runtime() -> None:
    with patch("xqt.operator_opt.capability._package_available", return_value=False):
        capability = describe_operator_engine_capability("cute_dsl")

    assert capability.status == "planned"
    assert capability.available is False
    assert capability.requires_cuda is True
    assert any("cutlass.cute" in note for note in capability.notes)


def test_preflight_records_cute_dsl_config_and_missing_dependency() -> None:
    with (
        patch("xqt.operator_opt.capability._package_available", return_value=False),
        patch("xqt.pipeline.preflight._package_available", return_value=False),
    ):
        report = preflight_xqt_config(_cute_dsl_operator_config())

    checks = {check.name: check for check in report.checks}
    capability_check = checks["operator_optimization.targets.0.capability"]
    runtime_check = checks["operator_optimization.targets.0.cute_dsl.runtime"]
    config_check = checks["operator_optimization.targets.0.cute_dsl.config"]

    assert capability_check.passed is True
    assert runtime_check.passed is False
    assert runtime_check.level == "warning"
    assert "cutlass.cute" in runtime_check.message
    assert config_check.passed is True
    assert config_check.metadata["target_arch"] == "sm_90"
    assert config_check.metadata["cluster_shape"] == [1, 1, 1]


def test_cute_dsl_operator_executor_returns_reference_guarded_report() -> None:
    config = load_xqt_config(_cute_dsl_operator_config())
    context = create_context(
        config,
        model=None,
        example_inputs=torch.randn(2, 16),
    )
    from xqt.pipeline.passes import LoadModelPass

    LoadModelPass().run(context)
    plan = build_operator_optimization_plan(config.operator_optimization)

    execution = execute_operator_optimization_plan(context, plan)

    assert len(execution.reports) == 1
    report = execution.reports[0]
    assert report.engine == "cute_dsl"
    assert report.applied is False
    assert report.metadata["execution_state"] == "fallback"
    assert report.metadata["execution_mode"] == "reference_fallback"
    assert report.metadata["kernel_kind"] == "reference_fallback"
    assert "cute_dsl_artifacts" in report.metadata
    assert "gemm_epilogue" in report.metadata["cute_dsl_artifacts"]
    assert report.metadata["cute_dsl_artifacts"]["gemm_epilogue"]["compile"]["target_arch"] == "sm_90"
    assert report.metadata["cute_dsl_artifacts"]["gemm_epilogue"]["compile"]["cluster_shape"] == [1, 1, 1]
    assert report.skip_reason is not None
