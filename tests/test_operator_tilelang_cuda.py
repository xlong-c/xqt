from __future__ import annotations

import importlib.util

import pytest
import torch

from xqt import nn as xqt_nn
from xqt.operator_opt.execute import execute_operator_optimization_plan
from xqt.operator_opt.kernels.tilelang._common import tilelang_runtime_usable
from xqt.operator_opt.tilelang_wrappers import (
    _TileLangAttentionWrapper,
    _TileLangXqtAttentionWrapper,
)
from xqt.operator_opt.plan import build_operator_optimization_plan
from tests.xqt.runtime_helpers import operator_config_from_dict, operator_runtime_context


requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for TileLang operator CUDA test",
)

requires_tilelang = pytest.mark.skipif(
    not tilelang_runtime_usable(),
    reason="a runtime-compatible TileLang adapter is required for TileLang operator CUDA test",
)


def _tilelang_cuda_operator_config(
    *,
    attention_fastpath: str = "tilelang",
    min_speedup: float = 1.01,
) -> dict:
    return {
        "config_version": 1,
        "project": {
            "name": "tilelang_operator_cuda",
            "artifact_dir": "artifacts/xqt/tests/tilelang_operator_cuda",
        },
        "model": {
            "target": "xqt.operator_opt.toy_models.build_toy_attention_classifier",
            "params": {
                "hidden_dim": 32,
                "num_heads": 4,
                "num_classes": 4,
            },
            "device": "cuda",
        },
        "operator_optimization": {
            "enabled": True,
            "default_engine": "tilelang",
            "targets": [
                {
                    "name": "model",
                    "engine": "tilelang",
                    "patterns": ["attention"],
                    "min_speedup": min_speedup,
                    "tilelang": {
                        "target_arch": "sm_89",
                        "attention_fastpath": attention_fastpath,
                    },
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
def test_tilelang_operator_executor_uses_cuda_kernel_entry() -> None:
    config_dict = _tilelang_cuda_operator_config(attention_fastpath="tilelang")
    context = operator_runtime_context(
        config_dict,
        model=None,
        example_inputs=torch.randn(1, 64, 32, device="cuda", dtype=torch.float16),
    )
    from xqt.pipeline.passes import LoadModelPass

    LoadModelPass().run(context)
    context.model = context.require_model().to(device="cuda", dtype=torch.float16)
    plan = build_operator_optimization_plan(operator_config_from_dict(config_dict))

    execution = execute_operator_optimization_plan(context, plan)

    assert len(execution.reports) == 1
    report = execution.reports[0]
    assert report.engine == "tilelang"
    assert report.metadata["execution_mode"] == "cuda_tilelang_entry"
    assert report.metadata["kernel_kind"] == "minimal_cuda_jit"
    assert report.metadata["operator_family"] == "attention"
    assert report.metadata["selected_fastpath"] == "tilelang_attention_kernel"
    assert report.metadata["settings"]["attention_fastpath"] == "tilelang"
    assert report.metadata["settings"]["preferred_patterns"] == ["attention"]
    assert report.metadata["kernel_constraints"]["dtype"] == "float16"


@requires_cuda
@requires_tilelang
def test_tilelang_operator_executor_uses_native_attention_fastpath_on_ada() -> None:
    config_dict = _tilelang_cuda_operator_config(
        attention_fastpath="auto",
        min_speedup=1.000001,
    )
    context = operator_runtime_context(
        config_dict,
        model=None,
        example_inputs=torch.randn(1, 64, 32, device="cuda", dtype=torch.float16),
    )
    from xqt.pipeline.passes import LoadModelPass

    LoadModelPass().run(context)
    context.model = context.require_model().to(device="cuda", dtype=torch.float16)
    plan = build_operator_optimization_plan(operator_config_from_dict(config_dict))

    execution = execute_operator_optimization_plan(context, plan)

    assert len(execution.reports) == 1
    report = execution.reports[0]
    assert report.engine == "tilelang"
    assert report.metadata["execution_mode"] == "cuda_native_fastpath"
    assert report.metadata["kernel_kind"] == "native_runtime_fastpath"
    assert report.metadata["operator_family"] == "attention"
    assert report.metadata["selected_fastpath"] == "native_sdpa"
    assert report.metadata["settings"]["attention_fastpath"] == "auto"
    assert report.metadata["settings"]["preferred_patterns"] == ["attention"]
    assert report.metadata["kernel_constraints"]["dtype"] == "float16"


@requires_cuda
@requires_tilelang
def test_tilelang_operator_executor_uses_cuda_graph_attention_fastpath() -> None:
    config_dict = _tilelang_cuda_operator_config(
        attention_fastpath="graph",
        min_speedup=1.000001,
    )
    context = operator_runtime_context(
        config_dict,
        model=None,
        example_inputs=torch.randn(1, 64, 32, device="cuda", dtype=torch.float16),
    )
    from xqt.pipeline.passes import LoadModelPass

    LoadModelPass().run(context)
    context.model = context.require_model().to(device="cuda", dtype=torch.float16)
    plan = build_operator_optimization_plan(operator_config_from_dict(config_dict))

    execution = execute_operator_optimization_plan(context, plan)

    assert len(execution.reports) == 1
    report = execution.reports[0]
    assert report.engine == "tilelang"
    assert report.metadata["execution_mode"] == "cuda_graph_tilelang_entry"
    assert report.metadata["kernel_kind"] == "cuda_graph_replay"
    assert report.metadata["operator_family"] == "attention"
    assert report.metadata["selected_fastpath"] == "tilelang_attention_cuda_graph"
    assert report.metadata["settings"]["attention_fastpath"] == "graph"
    assert report.metadata["settings"]["preferred_patterns"] == ["attention"]
    assert report.metadata["benchmark_strategy"] == "paired_steady_state_batched_mean"
    assert report.metadata["speedup_metric"] == "mean_ms"
    assert "paired_ratio_p50" in report.metadata["speedup_statistics"]
    assert report.metadata["cuda_graph"]["state"] == "replayed"
    assert report.metadata["cuda_graph"]["cache_size"] >= 1


@requires_cuda
@requires_tilelang
def test_tilelang_attention_wrapper_replays_cuda_graph_on_second_call() -> None:
    attention = torch.nn.MultiheadAttention(
        32,
        4,
        batch_first=True,
        dropout=0.0,
        device="cuda",
        dtype=torch.float16,
    )
    wrapper = _TileLangAttentionWrapper(
        attention,
        fallback="eager",
        settings={
            "target_arch": "sm_89",
            "attention_fastpath": "graph",
            "preferred_patterns": ["attention"],
        },
    ).to(device="cuda", dtype=torch.float16)
    tilelang_wrapper = _TileLangAttentionWrapper(
        attention,
        fallback="eager",
        settings={
            "target_arch": "sm_89",
            "attention_fastpath": "tilelang",
            "preferred_patterns": ["attention"],
        },
    ).to(device="cuda", dtype=torch.float16)
    x = torch.randn(1, 64, 32, device="cuda", dtype=torch.float16)

    first_output, _ = wrapper(x, x, x, need_weights=False)
    first_output = first_output.clone()
    first_metadata = wrapper.execution_metadata()
    second_input = x + 1
    second_output, _ = wrapper(second_input, second_input, second_input, need_weights=False)
    second_metadata = wrapper.execution_metadata()
    ref_first, _ = tilelang_wrapper(x, x, x, need_weights=False)
    ref_second, _ = tilelang_wrapper(second_input, second_input, second_input, need_weights=False)

    assert torch.allclose(first_output.float(), ref_first.float(), atol=1e-2, rtol=1e-2)
    assert torch.allclose(second_output.float(), ref_second.float(), atol=1e-2, rtol=1e-2)
    assert first_metadata["cuda_graph"]["state"] == "captured"
    assert second_metadata["cuda_graph"]["state"] == "replayed"


@requires_cuda
@requires_tilelang
def test_tilelang_xqt_attention_wrapper_replays_cuda_graph_on_second_call() -> None:
    attention = xqt_nn.Attention(32, heads=4, engine="torch").to(
        device="cuda",
        dtype=torch.float16,
    )
    wrapper = _TileLangXqtAttentionWrapper(
        attention,
        fallback="eager",
        settings={
            "target_arch": "sm_89",
            "attention_fastpath": "graph",
            "preferred_patterns": ["attention"],
        },
    ).to(device="cuda", dtype=torch.float16)
    tilelang_wrapper = _TileLangXqtAttentionWrapper(
        attention,
        fallback="eager",
        settings={
            "target_arch": "sm_89",
            "attention_fastpath": "tilelang",
            "preferred_patterns": ["attention"],
        },
    ).to(device="cuda", dtype=torch.float16)
    x = torch.randn(1, 64, 32, device="cuda", dtype=torch.float16)

    first_output = wrapper(x).clone()
    first_metadata = wrapper.execution_metadata()
    second_input = x + 1
    second_output = wrapper(second_input)
    second_metadata = wrapper.execution_metadata()
    ref_first = tilelang_wrapper(x)
    ref_second = tilelang_wrapper(second_input)

    assert torch.allclose(first_output.float(), ref_first.float(), atol=1e-2, rtol=1e-2)
    assert torch.allclose(second_output.float(), ref_second.float(), atol=1e-2, rtol=1e-2)
    assert first_metadata["cuda_graph"]["state"] == "captured"
    assert second_metadata["cuda_graph"]["state"] == "replayed"
