from __future__ import annotations

import importlib.util

import pytest
import torch

from xqt.operator_opt.execute import execute_operator_optimization_plan
from xqt.operator_opt.materialize import materialize_operator_candidate_model
from xqt.operator_opt.plan import build_operator_optimization_plan
from xqt.operator_opt.kernels.tilelang.conv import (
    conv3d_1x1x1_reference,
    conv3d_1x1x1_tilelang,
)
from xqt.operator_opt.kernels.tilelang._common import tilelang_runtime_usable
from xqt.operator_opt.types import OperatorOptimizationTargetPlan
from xqt.pipeline.passes import LoadModelPass
from tests.xqt.runtime_helpers import operator_config_from_dict, operator_runtime_context


requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for TileLang conv3d operator CUDA test",
)

requires_tilelang = pytest.mark.skipif(
    not tilelang_runtime_usable(),
    reason="a runtime-compatible TileLang adapter is required for TileLang conv3d operator CUDA test",
)


def _tilelang_conv3d_operator_config(device: str) -> dict:
    return {
        "config_version": 1,
        "project": {
            "name": f"tilelang_conv3d_{device}",
            "artifact_dir": f"artifacts/xqt/tests/tilelang_conv3d_{device}",
        },
        "model": {
            "target": "xqt.model.toy_models.build_toy_conv3d_block",
            "params": {
                "in_channels": 8,
                "hidden_channels": 64,
                "out_channels": 32,
            },
            "device": device,
        },
        "operator_optimization": {
            "enabled": True,
            "default_engine": "tilelang",
            "targets": [
                {
                    "name": "conv3d_tilelang",
                    "target": "conv",
                    "engine": "tilelang",
                    "patterns": ["conv3d_1x1x1"],
                    "min_speedup": 1.000001,
                    "tilelang": {
                        "target_arch": "sm_89",
                        "conv_fastpath": "tilelang",
                    },
                }
            ],
        },
        "benchmark": {
            "warmup": 1,
            "iterations": 2,
            "sync_cuda": device == "cuda",
        },
    }


@requires_cuda
@requires_tilelang
def test_tilelang_half_conv3d_cuda_kernel_matches_reference() -> None:
    torch.manual_seed(0)
    x = torch.randn(1, 64, 2, 8, 8, device="cuda", dtype=torch.float16)
    weight = torch.randn(64, 64, 1, 1, 1, device="cuda", dtype=torch.float16)
    bias = torch.randn(64, device="cuda", dtype=torch.float16)

    output = conv3d_1x1x1_tilelang(
        x,
        weight,
        bias,
        stride=(1, 1, 1),
        padding=(0, 0, 0),
        block_m=64,
        block_n=64,
        block_k=64,
        threads=128,
        num_stages=2,
    )
    reference = conv3d_1x1x1_reference(
        x,
        weight,
        bias,
        stride=(1, 1, 1),
        padding=(0, 0, 0),
    )

    assert output.shape == reference.shape
    assert output.dtype == torch.float16
    assert torch.allclose(output.float(), reference.float(), atol=1e-2, rtol=1e-2)


def test_tilelang_conv3d_operator_stage_uses_reference_fallback_on_cpu() -> None:
    config_dict = _tilelang_conv3d_operator_config("cpu")
    context = operator_runtime_context(
        config_dict,
        model=None,
        example_inputs=torch.randn(1, 8, 2, 8, 8, dtype=torch.float32),
    )
    LoadModelPass().run(context)
    plan = build_operator_optimization_plan(operator_config_from_dict(config_dict))
    execution = execute_operator_optimization_plan(context, plan)

    target = execution.reports[0].to_dict()
    assert target["metadata"]["execution_mode"] == "reference_fallback"
    assert target["metadata"]["kernel_kind"] == "reference_fallback"
    assert target["metadata"]["operator_family"] == "conv"
    assert target["metadata"]["kernel_pattern"] == "conv3d_1x1x1"
    assert target["metadata"]["selected_fastpath"] == "eager_reference_fallback"
    assert target["metadata"]["settings"]["preferred_patterns"] == ["conv3d_1x1x1"]


def test_materialize_tilelang_conv3d_candidate_replaces_named_conv() -> None:
    import xqt.model.toy_models as toy_models

    module = toy_models.build_toy_conv3d_block()
    target = OperatorOptimizationTargetPlan(
        name="conv3d_tilelang",
        engine="tilelang",
        target_path="conv",
        patterns=["conv3d_1x1x1"],
        fallback="eager",
        min_speedup=0.0,
        tilelang={"target_arch": "sm_89", "conv_fastpath": "tilelang"},
    )

    candidate, compile_time_ms = materialize_operator_candidate_model(module, target)

    assert compile_time_ms is None
    wrapped = candidate.get_submodule("conv")
    assert type(wrapped).__name__ == "_TileLangConv3dWrapper"


@requires_cuda
@requires_tilelang
def test_tilelang_conv3d_operator_stage_uses_cuda_kernel_entry() -> None:
    config_dict = _tilelang_conv3d_operator_config("cuda")
    context = operator_runtime_context(
        config_dict,
        model=None,
        example_inputs=torch.randn(1, 8, 2, 8, 8, device="cuda", dtype=torch.float16),
    )
    LoadModelPass().run(context)
    context.model = context.require_model().to(device="cuda", dtype=torch.float16)
    plan = build_operator_optimization_plan(operator_config_from_dict(config_dict))
    execution = execute_operator_optimization_plan(context, plan)

    target = execution.reports[0].to_dict()
    assert target["metadata"]["execution_mode"] == "cuda_tilelang_entry"
    assert target["metadata"]["kernel_kind"] == "minimal_cuda_jit"
    assert target["metadata"]["operator_family"] == "conv"
    assert target["metadata"]["kernel_pattern"] == "conv3d_1x1x1"
    assert target["metadata"]["selected_fastpath"] == "tilelang_half_conv3d_1x1x1_gemm"
    assert target["metadata"]["settings"]["preferred_patterns"] == ["conv3d_1x1x1"]
