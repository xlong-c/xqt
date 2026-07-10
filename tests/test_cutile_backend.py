from __future__ import annotations

from unittest.mock import patch

import torch

from xqt.operator_opt.backends.cutile import (
    build_cutile_artifact_metadata,
    list_cutile_kernel_specs,
    run_cutile_kernel,
)
from xqt.operator_opt.backends.tilelang import list_tilelang_kernel_specs
from xqt.operator_opt.capability import describe_operator_engine_capability
from xqt.operator_opt.execute import execute_operator_optimization_plan
from xqt.operator_opt.plan import build_operator_optimization_plan
from xqt.pipeline.preflight import preflight_optimization_config
from tests.xqt.runtime_helpers import operator_config_from_dict, operator_runtime_context


def _cutile_operator_config() -> dict[str, object]:
    return {
        "config_version": 1,
        "project": {
            "name": "cutile_operator",
            "artifact_dir": "artifacts/xqt/tests/cutile_operator",
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
            "default_engine": "cutile",
            "targets": [
                {
                    "name": "model",
                    "engine": "cutile",
                    "patterns": ["linear", "norm", "dequant_gemm_epilogue"],
                    "cutile": {
                        "target_arch": "sm_89",
                        "cache_dir": "artifacts/xqt/tests/cutile_operator/cache",
                        "pass_configs": {
                            "CUTILE_ENABLE_FAST_MATH": True,
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


def _cutile_operator_workflow_config() -> dict[str, object]:
    legacy_config = _cutile_operator_config()
    operator_params = dict(legacy_config["operator_optimization"])
    operator_params.pop("enabled")
    return {
        "project": legacy_config["project"],
        "model": legacy_config["model"],
        "benchmark": legacy_config["benchmark"],
        "stages": [
            {
                "name": "cutile_operator",
                "kind": "operator",
                "params": operator_params,
            }
        ],
    }


def test_cutile_registry_covers_tilelang_operator_patterns() -> None:
    cutile_specs = list_cutile_kernel_specs()
    tilelang_specs = list_tilelang_kernel_specs()

    tilelang_only = {"conv3d_1x1x1", "int8_mma", "linear_marlin"}
    assert (set(tilelang_specs) - tilelang_only).issubset(cutile_specs)
    assert "bias_silu" in cutile_specs
    assert (
        cutile_specs["linear"]["metadata"]["production_status"] == "reference_guarded"
    )
    assert cutile_specs["norm"]["metadata"]["normalized_last_dim_only"] is True
    assert (
        cutile_specs["fp4_packed_dequant_gemm_epilogue"]["metadata"]["weight_encoding"]
        == "packed_signed_int4"
    )


def test_cutile_artifact_metadata_is_stable() -> None:
    metadata = build_cutile_artifact_metadata("dequant_gemm_epilogue")

    assert metadata["engine"] == "cutile"
    assert metadata["pattern"] == "dequant_gemm_epilogue"
    assert metadata["compile_status"] == "metadata_only"
    assert metadata["exportable"] is False
    assert metadata["kernel"]["production_status"] == "reference_guarded"


def test_cutile_capability_uses_cuda_tile_runtime_probe() -> None:
    with patch("xqt.operator_opt.backends.cutile.cutile_available", return_value=True):
        capability = describe_operator_engine_capability("cutile")

    assert capability.status == "planned"
    assert capability.maturity == "reference_guarded"
    assert capability.available is True
    assert capability.requires_cuda is True
    assert any("linear, norm" in note for note in capability.notes)


def test_preflight_records_cutile_config_and_missing_dependency() -> None:
    with (
        patch("xqt.operator_opt.backends.cutile.cutile_available", return_value=False),
        patch("xqt.pipeline.preflight._cutile_available", return_value=False),
    ):
        report = preflight_optimization_config(_cutile_operator_workflow_config())

    checks = {check.name: check for check in report.checks}
    dependency_check = checks["dependency.cuda.tile"]
    capability_check = checks["stages.0.cutile_operator.targets.0.capability"]
    runtime_check = checks["stages.0.cutile_operator.targets.0.cutile.runtime"]
    config_check = checks["stages.0.cutile_operator.targets.0.cutile.config"]

    assert dependency_check.passed is False
    assert dependency_check.metadata["legacy_package"] == "cutile"
    assert capability_check.passed is True
    assert runtime_check.passed is False
    assert runtime_check.level == "warning"
    assert "cuda.tile" in runtime_check.message
    assert config_check.passed is True
    assert config_check.metadata["target_arch"] == "sm_89"


def test_cutile_operator_executor_reports_selected_artifacts() -> None:
    config_dict = _cutile_operator_config()
    context = operator_runtime_context(
        config_dict,
        model=None,
        example_inputs=torch.randn(2, 16),
    )
    from xqt.pipeline.passes import LoadModelPass

    LoadModelPass().run(context)
    plan = build_operator_optimization_plan(operator_config_from_dict(config_dict))

    execution = execute_operator_optimization_plan(context, plan)

    assert len(execution.reports) == 1
    report = execution.reports[0]
    assert report.engine == "cutile"
    assert report.applied is False
    assert report.metadata["execution_state"] == "fallback"
    assert report.metadata["execution_mode"] == "reference_fallback"
    assert report.metadata["kernel_kind"] == "reference_fallback"
    assert "cutile_artifacts" in report.metadata
    assert set(report.metadata["cutile_artifacts"]) == {
        "linear",
        "norm",
        "dequant_gemm_epilogue",
    }
    assert (
        report.metadata["cutile_artifacts"]["linear"]["compile"]["target_arch"]
        == "sm_89"
    )
    assert report.skip_reason is not None


def test_cutile_run_kernel_eager_fallback_filters_compile_kwargs() -> None:
    x = torch.randn(2, 4)
    weight = torch.randn(3, 4)
    bias = torch.randn(3)

    out = run_cutile_kernel(
        "linear",
        x,
        weight,
        bias,
        fallback="eager",
        block_m=64,
        block_n=64,
        block_k=64,
        target_arch="sm_89",
    )

    torch.testing.assert_close(out, torch.nn.functional.linear(x, weight, bias))
