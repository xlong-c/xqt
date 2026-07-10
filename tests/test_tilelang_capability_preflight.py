from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from xqt.core.errors import XQTBackendError
from xqt.operator_opt.backends.tilelang import run_tilelang_kernel
from xqt.operator_opt.capability import describe_operator_engine_capability
from xqt.operator_opt.kernels.tilelang._common import (
    tilelang_runtime_unavailability_reason,
    tilelang_runtime_usable,
)
from xqt.pipeline.preflight import preflight_optimization_config


requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for TileLang runtime guard test",
)


def _tilelang_operator_config() -> dict:
    return {
        "config_version": 1,
        "project": {
            "name": "tilelang_capability_preflight",
            "artifact_dir": "artifacts/xqt/tests/tilelang_capability_preflight",
        },
        "model": {
            "target": "xqt.operator_opt.toy_models.build_toy_attention_classifier",
            "params": {
                "hidden_dim": 16,
                "num_heads": 4,
                "num_classes": 4,
            },
            "device": "cpu",
        },
        "operator_optimization": {
            "enabled": True,
            "default_engine": "tilelang",
            "targets": [
                {
                    "name": "model",
                    "engine": "tilelang",
                    "patterns": ["attention"],
                }
            ],
        },
    }


def test_tilelang_capability_reports_reference_fallback_when_package_missing() -> None:
    with patch("xqt.operator_opt.capability._package_available", return_value=False):
        capability = describe_operator_engine_capability("tilelang")

    assert capability.status == "available"
    assert capability.maturity == "executable"
    assert capability.available is True
    assert any("reference fallback" in note for note in capability.notes)
    assert any(
        "attention, conv, linear, norm, and dequant_gemm_epilogue" in limitation
        for limitation in capability.limitations
    )


def test_tilelang_runtime_rejects_known_incompatible_packed_tensor_abi() -> None:
    with patch(
        "xqt.operator_opt.kernels.tilelang._common._tilelang_runtime_version",
        return_value="0.1.11",
    ):
        assert tilelang_runtime_usable() is False
        assert "incompatible packed-tensor ABI" in (
            tilelang_runtime_unavailability_reason() or ""
        )


def test_tilelang_capability_keeps_static_support_when_runtime_is_incompatible() -> None:
    with (
        patch(
            "xqt.operator_opt.capability._package_available", return_value=True
        ),
        patch(
            "xqt.operator_opt.kernels.tilelang._common.tilelang_runtime_usable",
            return_value=False,
        ),
        patch(
            "xqt.operator_opt.kernels.tilelang._common.tilelang_runtime_unavailability_reason",
            return_value="TileLang test ABI incompatibility",
        ),
    ):
        capability = describe_operator_engine_capability("tilelang")

    assert capability.status == "available"
    assert capability.maturity == "executable"
    assert capability.available is True
    assert "TileLang test ABI incompatibility" in capability.notes


@requires_cuda
def test_tilelang_runtime_guard_rejects_cuda_kernel_invocation() -> None:
    x = torch.empty(1, 1, 1, 1, device="cuda", dtype=torch.float16)
    weight = torch.empty(1, 1, 1, 1, device="cuda", dtype=torch.float16)

    with (
        patch(
            "xqt.operator_opt.kernels.tilelang._common.tilelang_runtime_usable",
            return_value=False,
        ),
        patch(
            "xqt.operator_opt.kernels.tilelang._common.tilelang_runtime_unavailability_reason",
            return_value="TileLang test ABI incompatibility",
        ),
        pytest.raises(XQTBackendError, match="TileLang test ABI incompatibility"),
    ):
        run_tilelang_kernel("conv", x, weight)


def test_workflow_preflight_warns_but_does_not_fail_when_tilelang_package_missing() -> None:
    workflow = {
        "project": {
            "name": "tilelang_workflow_preflight",
            "artifact_dir": "artifacts/xqt/tests/tilelang_workflow_preflight",
        },
        "model": _tilelang_operator_config()["model"],
        "stages": [
            {
                "name": "tilelang_operator",
                "kind": "operator",
                "params": {
                    "default_engine": "tilelang",
                    "targets": [
                        {
                            "name": "model",
                            "engine": "tilelang",
                            "patterns": ["attention"],
                        }
                    ],
                },
            }
        ],
    }

    with patch("xqt.pipeline.preflight._package_available", return_value=False):
        report = preflight_optimization_config(workflow)

    checks = {check.name: check for check in report.checks}
    capability_check = checks["stages.0.tilelang_operator.targets.0.capability"]
    runtime_check = checks["stages.0.tilelang_operator.targets.0.tilelang.runtime"]

    assert capability_check.passed is True
    assert runtime_check.passed is True
    assert runtime_check.level == "warning"
    assert "reference fallback" in runtime_check.message
