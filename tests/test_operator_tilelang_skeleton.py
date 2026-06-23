from __future__ import annotations

import torch

from xqt.core.config import load_xqt_config
from xqt.pipeline.runner import create_context
from xqt.operator_opt.executor import (
    build_operator_optimization_plan,
    execute_operator_optimization_plan,
)


def _base_tilelang_operator_config() -> dict:
    return {
        "config_version": 1,
        "project": {
            "name": "tilelang_operator_skeleton",
            "artifact_dir": "artifacts/xqt/tests/tilelang_operator_skeleton",
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
            "default_backend": "tilelang",
            "targets": [
                {
                    "name": "model",
                    "backend": "tilelang",
                    "patterns": ["attention"],
                }
            ],
        },
        "benchmark": {
            "warmup": 1,
            "iterations": 1,
            "sync_cuda": False,
        },
    }


def test_tilelang_operator_skeleton_returns_planned_report() -> None:
    config = load_xqt_config(_base_tilelang_operator_config())
    model = torch.nn.Sequential()
    del model
    context = create_context(
        config,
        model=None,
        example_inputs=torch.randn(1, 4, 16),
    )
    from xqt.pipeline.passes import LoadModelPass

    LoadModelPass().run(context)
    plan = build_operator_optimization_plan(config.operator_optimization)

    execution = execute_operator_optimization_plan(context, plan)

    assert len(execution.reports) == 1
    report = execution.reports[0]
    assert report.backend == "tilelang"
    assert report.metadata["execution_state"] in {"executed", "fallback"}
    assert report.metadata["execution_mode"] == "reference_fallback"
    assert "requires CUDA tensors" in str(report.metadata["execution_reason"])
    assert "tilelang_artifacts" in report.metadata
