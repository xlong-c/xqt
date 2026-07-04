from __future__ import annotations

import importlib.util

import pytest
import torch

from xqt.core.config import load_xqt_config
from xqt.operator_opt.executor import (
    build_operator_optimization_plan,
    execute_operator_optimization_plan,
)
from xqt.pipeline.passes import LoadModelPass
from xqt.pipeline.runner import create_context


requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for TileLang conv operator CUDA test",
)

requires_tilelang = pytest.mark.skipif(
    importlib.util.find_spec("tilelang") is None,
    reason="tilelang package is required for TileLang conv operator CUDA test",
)


def _tilelang_conv_operator_config(
    device: str,
    *,
    conv_fastpath: str = "auto",
    min_speedup: float = 1.000001,
    in_channels: int = 3,
    hidden_channels: int = 8,
    out_channels: int = 4,
) -> dict:
    return {
        "config_version": 1,
        "project": {
            "name": f"tilelang_conv_{device}",
            "artifact_dir": f"artifacts/xqt/tests/tilelang_conv_{device}",
        },
        "model": {
            "target": "xqt.operator_opt.toy_models.build_toy_conv_block",
            "params": {
                "in_channels": in_channels,
                "hidden_channels": hidden_channels,
                "out_channels": out_channels,
            },
            "device": device,
        },
        "operator_optimization": {
            "enabled": True,
            "default_backend": "tilelang",
            "targets": [
                {
                    "name": "conv_tilelang",
                    "target": "conv",
                    "backend": "tilelang",
                    "patterns": ["conv"],
                    "min_speedup": min_speedup,
                    "tilelang": {
                        "target_arch": "sm_89",
                        "conv_fastpath": conv_fastpath,
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


def test_tilelang_conv_operator_stage_uses_reference_fallback_on_cpu() -> None:
    config = load_xqt_config(_tilelang_conv_operator_config("cpu"))
    context = create_context(
        config,
        model=None,
        example_inputs=torch.randn(2, 3, 32, 32, dtype=torch.float32),
    )
    LoadModelPass().run(context)
    plan = build_operator_optimization_plan(config.operator_optimization)
    execution = execute_operator_optimization_plan(context, plan)

    target = execution.reports[0].to_dict()
    assert target["metadata"]["execution_mode"] == "reference_fallback"
    assert target["metadata"]["kernel_kind"] == "reference_fallback"
    assert target["metadata"]["operator_family"] == "conv"
    assert target["metadata"]["selected_fastpath"] == "eager_reference_fallback"
    assert target["metadata"]["settings"]["preferred_patterns"] == ["conv"]


@requires_cuda
@requires_tilelang
def test_tilelang_conv_operator_stage_uses_native_cuda_fastpath_on_ada() -> None:
    config = load_xqt_config(_tilelang_conv_operator_config("cuda"))
    context = create_context(
        config,
        model=None,
        example_inputs=torch.randn(2, 3, 32, 32, device="cuda", dtype=torch.float16),
    )
    LoadModelPass().run(context)
    context.model = context.require_model().to(device="cuda", dtype=torch.float16)
    plan = build_operator_optimization_plan(config.operator_optimization)
    execution = execute_operator_optimization_plan(context, plan)

    target = execution.reports[0].to_dict()
    assert target["metadata"]["execution_mode"] == "cuda_native_fastpath"
    assert target["metadata"]["kernel_kind"] == "native_runtime_fastpath"
    assert target["metadata"]["operator_family"] == "conv"
    assert target["metadata"]["selected_fastpath"] == "native_cudnn_conv2d"
    assert target["metadata"]["settings"]["conv_fastpath"] == "auto"
    assert target["metadata"]["settings"]["preferred_patterns"] == ["conv"]


@requires_cuda
@requires_tilelang
def test_tilelang_conv_operator_stage_uses_cuda_kernel_entry() -> None:
    config = load_xqt_config(
        _tilelang_conv_operator_config(
            "cuda",
            conv_fastpath="tilelang",
            min_speedup=1.000001,
            in_channels=64,
            hidden_channels=64,
            out_channels=32,
        )
    )
    context = create_context(
        config,
        model=None,
        example_inputs=torch.randn(1, 64, 8, 8, device="cuda", dtype=torch.float16),
    )
    LoadModelPass().run(context)
    context.model = context.require_model().to(device="cuda", dtype=torch.float16)
    plan = build_operator_optimization_plan(config.operator_optimization)
    execution = execute_operator_optimization_plan(context, plan)

    target = execution.reports[0].to_dict()
    assert target["metadata"]["execution_mode"] == "cuda_tilelang_entry"
    assert target["metadata"]["kernel_kind"] == "minimal_cuda_jit"
    assert target["metadata"]["operator_family"] == "conv"
    assert target["metadata"]["selected_fastpath"] == "tilelang_half_conv2d_im2col_gemm"
    assert target["metadata"]["settings"]["conv_fastpath"] == "tilelang"
    assert target["metadata"]["settings"]["preferred_patterns"] == ["conv"]
