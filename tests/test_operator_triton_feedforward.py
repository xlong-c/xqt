from __future__ import annotations

import importlib.util

import pytest
import torch

import xqt.kernels.nn as xqt_nn
from xqt.kernels.wrappers.execute import execute_operator_optimization_plan
from xqt.kernels.wrappers.plan import build_operator_optimization_plan
from tests.xqt.runtime_helpers import operator_config_from_dict, operator_runtime_context


requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for Triton FeedForward operator CUDA test",
)

requires_triton = pytest.mark.skipif(
    importlib.util.find_spec("triton") is None,
    reason="triton package is required for Triton FeedForward operator CUDA test",
)


def _triton_feedforward_operator_config(
    device: str = "cpu",
    *,
    output_dtype: str = "fp32",
    validation: dict[str, float] | None = None,
) -> dict:
    target = {
        "name": "feedforward_triton",
        "engine": "triton",
        "patterns": ["feedforward"],
        "fallback": "eager",
        "min_speedup": 0.0,
        "options": {
            "activation_dtype": "fp16",
            "weight_dtype": "fp16",
            "mma_dtype": "fp16",
            "accum_dtype": "fp32",
            "output_dtype": output_dtype,
            "projection_policies": {
                "proj_out": {"output": output_dtype},
            },
        },
    }
    if validation is not None:
        target["validate"] = dict(validation)
    return {
        "config_version": 1,
        "project": {
            "name": f"triton_feedforward_{device}",
            "artifact_dir": f"artifacts/xqt/tests/triton_feedforward_{device}",
        },
        "model": {
            "target": "xqt.kernels.nn.FeedForward",
            "params": {},
            "device": device,
        },
        "operator_optimization": {
            "enabled": True,
            "default_engine": "triton",
            "targets": [target],
        },
        "benchmark": {
            "warmup": 1,
            "iterations": 2,
            "sync_cuda": device == "cuda",
        },
    }


def test_triton_feedforward_operator_stage_reports_cpu_fallback_metadata() -> None:
    config_dict = _triton_feedforward_operator_config()
    module = xqt_nn.FeedForward(8, inner_dim=12, activation="gelu").eval()
    context = operator_runtime_context(
        config_dict,
        model=module,
        example_inputs=torch.randn(4, 8),
    )
    plan = build_operator_optimization_plan(operator_config_from_dict(config_dict))

    execution = execute_operator_optimization_plan(context, plan)

    target = execution.reports[0].to_dict()
    assert target["applied"] is False
    assert target["metadata"]["execution_state"] == "fallback"
    assert target["metadata"]["execution_mode"] == "reference_fallback"
    assert target["metadata"]["kernel_kind"] == "reference_fallback"
    assert target["metadata"]["operator_family"] == "feedforward"
    assert target["metadata"]["kernel_pattern"] == "feedforward"
    assert target["metadata"]["selected_fastpath"] == "eager_reference_fallback"
    assert target["metadata"]["settings"]["preferred_patterns"] == ["feedforward"]
    assert target["metadata"]["runtime_config"]["engine"] == "triton"
    assert target["metadata"]["runtime_config"]["projections"]["proj_out"][
        "output"
    ] == "fp32"
    assert "require CUDA tensors" in str(target["skip_reason"])


@requires_cuda
@requires_triton
def test_triton_feedforward_operator_stage_uses_cuda_runtime_composition() -> None:
    torch.manual_seed(0)
    config_dict = _triton_feedforward_operator_config(
        "cuda",
        output_dtype="fp16",
        validation={"atol": 1e-3, "rtol": 1e-3},
    )
    module = xqt_nn.FeedForward(64, inner_dim=128, activation="gelu").eval()
    module = module.to(device="cuda", dtype=torch.float16)
    context = operator_runtime_context(
        config_dict,
        model=module,
        example_inputs=torch.randn(8, 64, device="cuda", dtype=torch.float16),
    )
    plan = build_operator_optimization_plan(operator_config_from_dict(config_dict))

    execution = execute_operator_optimization_plan(context, plan)

    target = execution.reports[0].to_dict()
    assert target["applied"] is True
    assert target["numeric_diff"]["allclose"] is True
    assert target["metadata"]["execution_state"] == "executed"
    assert target["metadata"]["execution_mode"] == "triton_runtime_configured"
    assert target["metadata"]["kernel_kind"] == "triton_composed_runtime"
    assert target["metadata"]["operator_family"] == "feedforward"
    assert target["metadata"]["runtime_config"]["engine"] == "triton"
    assert target["metadata"]["runtime_config"]["fallback"] is None
    assert target["latency_before"]["iterations"] == 2
    assert target["latency_after"]["iterations"] == 2
