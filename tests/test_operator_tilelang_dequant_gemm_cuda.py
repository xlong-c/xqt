from __future__ import annotations

import importlib.util

import pytest
import torch

from xqt.core.config import load_xqt_config
from xqt.operator_opt.executor import (
    build_operator_optimization_plan,
    execute_operator_optimization_plan,
)
from xqt.pipeline.runner import create_context


requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for TileLang dequant GEMM operator CUDA test",
)

requires_tilelang = pytest.mark.skipif(
    importlib.util.find_spec("tilelang") is None,
    reason="tilelang package is required for TileLang dequant GEMM operator CUDA test",
)


def _tilelang_dequant_gemm_cuda_operator_config() -> dict:
    return {
        "config_version": 1,
        "project": {
            "name": "tilelang_dequant_gemm_operator_cuda",
            "artifact_dir": "artifacts/xqt/tests/tilelang_dequant_gemm_operator_cuda",
        },
        "model": {
            "target": "xqt.operator_opt.toy_models.build_toy_dequant_gemm_block",
            "params": {
                "input_dim": 32,
                "output_dim": 64,
                "activation": "silu",
            },
            "device": "cuda",
        },
        "operator_optimization": {
            "enabled": True,
            "default_backend": "tilelang",
            "targets": [
                {
                    "name": "model",
                    "backend": "tilelang",
                    "patterns": ["dequant_gemm_epilogue"],
                    "min_speedup": 1.01,
                }
            ],
        },
        "benchmark": {
            "warmup": 1,
            "iterations": 1,
            "sync_cuda": True,
        },
    }


@requires_cuda
@requires_tilelang
def test_tilelang_dequant_gemm_operator_executor_uses_cuda_kernel_entry() -> None:
    config = load_xqt_config(_tilelang_dequant_gemm_cuda_operator_config())
    context = create_context(
        config,
        model=None,
        example_inputs=torch.randn(64, 32, device="cuda", dtype=torch.float16),
    )
    from xqt.pipeline.passes import LoadModelPass

    LoadModelPass().run(context)
    context.model = context.require_model().to(device="cuda", dtype=torch.float16)
    plan = build_operator_optimization_plan(config.operator_optimization)

    execution = execute_operator_optimization_plan(context, plan)

    assert len(execution.reports) == 1
    report = execution.reports[0]
    assert report.backend == "tilelang"
    assert report.metadata["execution_mode"] == "cuda_tilelang_entry"
    assert report.metadata["kernel_kind"] == "minimal_cuda_jit"
    assert report.metadata["kernel_constraints"]["dtype"] == "float16"
    assert report.metadata["kernel_constraints"]["supported_patterns"] == [
        "dequant_gemm_epilogue"
    ]
