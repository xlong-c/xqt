from __future__ import annotations

import pytest
import torch

from xqt import nn as xqt_nn
from xqt.operator_opt.execute import execute_operator_optimization_plan
from xqt.operator_opt.kernels.tilelang._common import tilelang_runtime_usable
from xqt.operator_opt.kernels.tilelang.attention import (
    fused_attention_forward_reference,
)
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
                "hidden_dim": 64,
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
@pytest.mark.parametrize(
    "dtype",
    [torch.float16, torch.bfloat16],
    ids=["fp16", "bf16"],
)
def test_tilelang_operator_executor_uses_cuda_kernel_entry(
    dtype: torch.dtype,
) -> None:
    config_dict = _tilelang_cuda_operator_config(attention_fastpath="tilelang")
    context = operator_runtime_context(
        config_dict,
        model=None,
        example_inputs=torch.randn(1, 64, 64, device="cuda", dtype=dtype),
    )
    from xqt.pipeline.passes import LoadModelPass

    LoadModelPass().run(context)
    context.model = context.require_model().to(device="cuda", dtype=dtype)
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
    assert report.metadata["kernel_constraints"]["dtype"] == str(dtype).removeprefix(
        "torch."
    )
    assert report.metadata["kernel_constraints"]["supported_dtypes"] == [
        "float16",
        "bfloat16",
    ]


@requires_cuda
@requires_tilelang
@pytest.mark.parametrize(
    "dtype",
    [torch.float16, torch.bfloat16],
    ids=["fp16", "bf16"],
)
def test_tilelang_operator_executor_uses_native_attention_fastpath_on_ada(
    dtype: torch.dtype,
) -> None:
    config_dict = _tilelang_cuda_operator_config(
        attention_fastpath="auto",
        min_speedup=1.000001,
    )
    context = operator_runtime_context(
        config_dict,
        model=None,
        example_inputs=torch.randn(1, 64, 64, device="cuda", dtype=dtype),
    )
    from xqt.pipeline.passes import LoadModelPass

    LoadModelPass().run(context)
    context.model = context.require_model().to(device="cuda", dtype=dtype)
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
    assert report.metadata["kernel_constraints"]["dtype"] == str(dtype).removeprefix(
        "torch."
    )


@requires_cuda
@requires_tilelang
@pytest.mark.parametrize(
    "dtype",
    [torch.float16, torch.bfloat16],
    ids=["fp16", "bf16"],
)
def test_tilelang_operator_executor_uses_cuda_graph_attention_fastpath(
    dtype: torch.dtype,
) -> None:
    config_dict = _tilelang_cuda_operator_config(
        attention_fastpath="graph",
        min_speedup=1.000001,
    )
    context = operator_runtime_context(
        config_dict,
        model=None,
        example_inputs=torch.randn(1, 64, 64, device="cuda", dtype=dtype),
    )
    from xqt.pipeline.passes import LoadModelPass

    LoadModelPass().run(context)
    context.model = context.require_model().to(device="cuda", dtype=dtype)
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
    assert report.metadata["kernel_constraints"]["dtype"] == str(dtype).removeprefix(
        "torch."
    )
    assert report.metadata["kernel_constraints"]["bfloat16_head_dim_multiple"] == 16


@requires_cuda
@requires_tilelang
@pytest.mark.parametrize(
    "dtype",
    [torch.float16, torch.bfloat16],
    ids=["fp16", "bf16"],
)
def test_tilelang_attention_wrapper_replays_cuda_graph_on_second_call(
    dtype: torch.dtype,
) -> None:
    attention = torch.nn.MultiheadAttention(
        64,
        4,
        batch_first=True,
        dropout=0.0,
        device="cuda",
        dtype=dtype,
    )
    wrapper = _TileLangAttentionWrapper(
        attention,
        fallback="eager",
        settings={
            "target_arch": "sm_89",
            "attention_fastpath": "graph",
            "preferred_patterns": ["attention"],
        },
    ).to(device="cuda", dtype=dtype)
    tilelang_wrapper = _TileLangAttentionWrapper(
        attention,
        fallback="eager",
        settings={
            "target_arch": "sm_89",
            "attention_fastpath": "tilelang",
            "preferred_patterns": ["attention"],
        },
    ).to(device="cuda", dtype=dtype)
    x = torch.randn(1, 64, 64, device="cuda", dtype=dtype)

    first_output, _ = wrapper(x, x, x, need_weights=False)
    first_output = first_output.clone()
    first_metadata = wrapper.execution_metadata()
    second_input = x + 1
    second_output, _ = wrapper(second_input, second_input, second_input, need_weights=False)
    second_metadata = wrapper.execution_metadata()
    ref_first, _ = tilelang_wrapper(x, x, x, need_weights=False)
    ref_second, _ = tilelang_wrapper(second_input, second_input, second_input, need_weights=False)

    tolerance = 2e-2 if dtype == torch.bfloat16 else 1e-2
    assert torch.allclose(
        first_output.float(),
        ref_first.float(),
        atol=tolerance,
        rtol=tolerance,
    )
    assert torch.allclose(
        second_output.float(),
        ref_second.float(),
        atol=tolerance,
        rtol=tolerance,
    )
    assert first_output.dtype == dtype
    assert second_output.dtype == dtype
    assert first_metadata["cuda_graph"]["state"] == "captured"
    assert second_metadata["cuda_graph"]["state"] == "replayed"
    assert first_metadata["kernel_constraints"]["dtype"] == str(dtype).removeprefix(
        "torch."
    )
    assert second_metadata["kernel_constraints"]["supported_dtypes"] == [
        "float16",
        "bfloat16",
    ]


@requires_cuda
@requires_tilelang
@pytest.mark.parametrize(
    "dtype",
    [torch.float16, torch.bfloat16],
    ids=["fp16", "bf16"],
)
def test_tilelang_attention_wrapper_native_path_preserves_lower_right_causal_cross_attention(
    dtype: torch.dtype,
) -> None:
    attention = torch.nn.MultiheadAttention(
        64,
        4,
        batch_first=True,
        dropout=0.0,
        device="cuda",
        dtype=dtype,
    ).eval()
    tilelang_wrapper = _TileLangAttentionWrapper(
        attention,
        fallback="error",
        settings={
            "target_arch": "sm_89",
            "attention_fastpath": "tilelang",
        },
    ).to(device="cuda", dtype=dtype)
    native_wrapper = _TileLangAttentionWrapper(
        attention,
        fallback="error",
        settings={
            "target_arch": "sm_89",
            "attention_fastpath": "native",
        },
    ).to(device="cuda", dtype=dtype)
    query = torch.randn(1, 1, 64, device="cuda", dtype=dtype)
    key = torch.randn(1, 64, 64, device="cuda", dtype=dtype)
    value = torch.randn_like(key)

    q, k, v = tilelang_wrapper._project_qkv_for_tilelang(query, key, value)
    reference = tilelang_wrapper._finalize_attention_output(
        fused_attention_forward_reference(
            q,
            k,
            v,
            causal=True,
            dropout_p=0.0,
        )
    )
    tilelang_output, _ = tilelang_wrapper(
        query,
        key,
        value,
        need_weights=False,
        is_causal=True,
    )
    native_output, _ = native_wrapper(
        query,
        key,
        value,
        need_weights=False,
        is_causal=True,
    )

    tolerance = 2e-2 if dtype == torch.bfloat16 else 1e-2
    assert torch.allclose(
        tilelang_output.float(),
        reference.float(),
        atol=tolerance,
        rtol=tolerance,
    )
    assert torch.allclose(
        native_output.float(),
        reference.float(),
        atol=tolerance,
        rtol=tolerance,
    )


@requires_cuda
@requires_tilelang
@pytest.mark.parametrize(
    "dtype",
    [torch.float16, torch.bfloat16],
    ids=["fp16", "bf16"],
)
def test_tilelang_xqt_attention_wrapper_replays_cuda_graph_on_second_call(
    dtype: torch.dtype,
) -> None:
    attention = xqt_nn.Attention(64, heads=4, engine="torch").to(
        device="cuda",
        dtype=dtype,
    )
    wrapper = _TileLangXqtAttentionWrapper(
        attention,
        fallback="eager",
        settings={
            "target_arch": "sm_89",
            "attention_fastpath": "graph",
            "preferred_patterns": ["attention"],
        },
    ).to(device="cuda", dtype=dtype)
    tilelang_wrapper = _TileLangXqtAttentionWrapper(
        attention,
        fallback="eager",
        settings={
            "target_arch": "sm_89",
            "attention_fastpath": "tilelang",
            "preferred_patterns": ["attention"],
        },
    ).to(device="cuda", dtype=dtype)
    x = torch.randn(1, 64, 64, device="cuda", dtype=dtype)

    first_output = wrapper(x).clone()
    first_metadata = wrapper.execution_metadata()
    second_input = x + 1
    second_output = wrapper(second_input)
    second_metadata = wrapper.execution_metadata()
    ref_first = tilelang_wrapper(x)
    ref_second = tilelang_wrapper(second_input)

    tolerance = 2e-2 if dtype == torch.bfloat16 else 1e-2
    assert torch.allclose(
        first_output.float(),
        ref_first.float(),
        atol=tolerance,
        rtol=tolerance,
    )
    assert torch.allclose(
        second_output.float(),
        ref_second.float(),
        atol=tolerance,
        rtol=tolerance,
    )
    assert first_output.dtype == dtype
    assert second_output.dtype == dtype
    assert first_metadata["cuda_graph"]["state"] == "captured"
    assert second_metadata["cuda_graph"]["state"] == "replayed"
    assert first_metadata["kernel_constraints"]["dtype"] == str(dtype).removeprefix(
        "torch."
    )
    assert second_metadata["kernel_constraints"]["supported_dtypes"] == [
        "float16",
        "bfloat16",
    ]
