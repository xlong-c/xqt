from __future__ import annotations

from typing import Any

import torch
from omegaconf import OmegaConf

from xqt.core.schema import BenchmarkConfig, OperatorOptimizationConfig
from xqt.core.types import XQTContext
from xqt.operator_opt.execute import execute_operator_optimization_plan
from xqt.operator_opt.plan import build_operator_optimization_plan


def _runtime_context(
    config_dict: dict[str, Any],
    *,
    model: Any = None,
    example_inputs: Any = None,
) -> XQTContext:
    model_config = config_dict["model"]
    project = config_dict["project"]
    return XQTContext(
        model=model,
        example_inputs=example_inputs,
        device=str(model_config["device"]),
        artifact_dir=str(project["artifact_dir"]),
        project_name=str(project["name"]),
        task_type="classification",
        model_target=str(model_config["target"]),
        model_params=dict(model_config["params"]),
        operator_config=_operator_config(config_dict),
        benchmark_config=BenchmarkConfig(**config_dict["benchmark"]),
    )


def _operator_config(config_dict: dict[str, Any]) -> OperatorOptimizationConfig:
    merged = OmegaConf.merge(
        OmegaConf.structured(OperatorOptimizationConfig),
        config_dict["operator_optimization"],
    )
    return OmegaConf.to_object(merged)  # type: ignore[return-value]


def _base_tilelang_operator_config() -> dict:
    return {
        "config_version": 1,
        "project": {
            "name": "tilelang_operator_skeleton",
            "artifact_dir": "artifacts/xqt/tests/tilelang_operator_skeleton",
        },
        "model": {
            "target": "xqt.model.toy_models.build_toy_attention_classifier",
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
        "benchmark": {
            "warmup": 1,
            "iterations": 1,
            "sync_cuda": False,
        },
    }


def test_tilelang_operator_skeleton_returns_planned_report() -> None:
    config_dict = _base_tilelang_operator_config()
    operator_config = _operator_config(config_dict)
    model = torch.nn.Sequential()
    del model
    context = _runtime_context(
        config_dict,
        model=None,
        example_inputs=torch.randn(1, 4, 16),
    )
    from xqt.pipeline.passes import LoadModelPass

    LoadModelPass().run(context)
    plan = build_operator_optimization_plan(operator_config)

    execution = execute_operator_optimization_plan(context, plan)

    assert len(execution.reports) == 1
    report = execution.reports[0]
    assert report.engine == "tilelang"
    assert report.fallback_policy == "prefer_fallback"
    assert report.metadata["execution_state"] in {"executed", "fallback"}
    assert report.metadata["fallback_policy"] == "prefer_fallback"
    assert report.metadata["fallback_reason"] is not None
    assert report.metadata["execution_mode"] == "reference_fallback"
    assert "requires CUDA tensors" in str(report.metadata["execution_reason"])
    assert "tilelang_artifacts" in report.metadata


def test_tilelang_operator_skeleton_strict_policy_records_strict_fallback_reason() -> None:
    config_dict = _base_tilelang_operator_config()
    config_dict["operator_optimization"]["targets"][0]["fallback_policy"] = "strict"
    context = _runtime_context(
        config_dict,
        model=None,
        example_inputs=torch.randn(1, 4, 16),
    )
    from xqt.pipeline.passes import LoadModelPass

    LoadModelPass().run(context)
    plan = build_operator_optimization_plan(_operator_config(config_dict))

    execution = execute_operator_optimization_plan(context, plan)

    report = execution.reports[0]
    assert report.fallback_policy == "strict"
    assert report.skip_reason is not None
    assert report.skip_reason.startswith("strict policy rejected fallback:")
    assert report.metadata["fallback_policy"] == "strict"
    assert report.metadata["fallback_reason"] == report.skip_reason
