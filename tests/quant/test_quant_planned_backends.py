from __future__ import annotations

import torch

from xqt.core.config import load_xqt_config
from xqt.pipeline.runner import create_context
from xqt.quant.execution import execute_quantization_plan
from xqt.quant.plan import build_quantization_plan


def _base_awq_config() -> dict:
    return {
        "config_version": 1,
        "project": {
            "name": "quant_planned_backends",
            "artifact_dir": "artifacts/xqt/tests/quant_planned_backends",
        },
        "model": {
            "target": "torch.nn:Linear",
            "params": {"in_features": 4, "out_features": 4},
            "device": "cpu",
        },
        "compression": {
            "quant": {
                "enabled": True,
                "backend": "tilelang",
                "method": "awq",
                "strategy": "fp4_weight_only",
                "policy": {
                    "bits": 4,
                    "dtype": "fp4",
                    "group_size": 128,
                },
            }
        },
    }


def test_planned_awq_tilelang_quantization_returns_structured_report() -> None:
    config = load_xqt_config(_base_awq_config())
    model = torch.nn.Linear(4, 4)
    context = create_context(config, model=model)
    plan = build_quantization_plan(config.compression.quant)

    execution = execute_quantization_plan(context, plan)

    assert len(execution.reports) == 1
    report = execution.reports[0]
    assert report.backend == "tilelang"
    assert report.method == "awq"
    assert report.strategy == "fp4_weight_only"
    assert report.metadata["execution_state"] == "planned"
    assert report.metadata["executed"] is False
    assert "planned" in report.artifacts
    assert "quant_plan" in execution.artifacts
