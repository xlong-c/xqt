from __future__ import annotations

import copy
import importlib.util
from typing import Any

from omegaconf import OmegaConf
import pytest
import torch
from torch import nn

from xqt.core.schema import BenchmarkConfig, OperatorOptimizationConfig
from xqt.core.types import XQTContext
from xqt.operator_opt.execute import execute_operator_optimization_plan
from xqt.operator_opt.kernels.tilelang._common import tilelang_runtime_usable
from xqt.operator_opt.tilelang_wrappers import _TileLangNormWrapper
from xqt.operator_opt.plan import build_operator_optimization_plan
from xqt.pipeline.passes import LoadModelPass


requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for TileLang norm operator CUDA test",
)

requires_tilelang = pytest.mark.skipif(
    not tilelang_runtime_usable(),
    reason="a runtime-compatible TileLang adapter is required for TileLang norm operator CUDA test",
)


def _operator_config(config_dict: dict[str, Any]) -> OperatorOptimizationConfig:
    merged = OmegaConf.merge(
        OmegaConf.structured(OperatorOptimizationConfig),
        config_dict["operator_optimization"],
    )
    return OmegaConf.to_object(merged)  # type: ignore[return-value]


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
        reference_model=copy.deepcopy(model) if model is not None else None,
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


def _tilelang_norm_operator_config(
    device: str,
    *,
    norm_fastpath: str = "tilelang",
    min_speedup: float = 1.01,
) -> dict:
    return {
        "config_version": 1,
        "project": {
            "name": f"tilelang_norm_{device}",
            "artifact_dir": f"artifacts/xqt/tests/tilelang_norm_{device}",
        },
        "model": {
            "target": "xqt.operator_opt.toy_models.build_toy_norm_block",
            "params": {
                "hidden_dim": 64,
            },
            "device": device,
        },
        "operator_optimization": {
            "enabled": True,
            "default_engine": "tilelang",
            "targets": [
                {
                    "name": "norm_tilelang",
                    "target": "norm",
                    "engine": "tilelang",
                    "patterns": ["norm"],
                    "min_speedup": min_speedup,
                    "tilelang": {
                        "target_arch": "sm_89",
                        "norm_fastpath": norm_fastpath,
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


def test_tilelang_norm_operator_stage_uses_reference_fallback_on_cpu() -> None:
    config_dict = _tilelang_norm_operator_config("cpu")
    context = _runtime_context(
        config_dict,
        model=None,
        example_inputs=torch.randn(8, 64, dtype=torch.float32),
    )
    LoadModelPass().run(context)
    plan = build_operator_optimization_plan(_operator_config(config_dict))
    execution = execute_operator_optimization_plan(context, plan)

    target = execution.reports[0].to_dict()
    assert target["metadata"]["execution_mode"] == "reference_fallback"
    assert target["metadata"]["kernel_kind"] == "reference_fallback"
    assert target["metadata"]["operator_family"] == "norm"
    assert target["metadata"]["selected_fastpath"] == "eager_reference_fallback"
    assert target["metadata"]["settings"]["preferred_patterns"] == ["norm"]


@requires_cuda
@requires_tilelang
def test_tilelang_norm_operator_stage_uses_cuda_kernel_entry() -> None:
    config_dict = _tilelang_norm_operator_config(
        "cuda",
        norm_fastpath="tilelang",
    )
    context = _runtime_context(
        config_dict,
        model=None,
        example_inputs=torch.randn(8, 64, device="cuda", dtype=torch.float16),
    )
    LoadModelPass().run(context)
    context.model = context.require_model().to(device="cuda", dtype=torch.float16)
    plan = build_operator_optimization_plan(_operator_config(config_dict))
    execution = execute_operator_optimization_plan(context, plan)

    target = execution.reports[0].to_dict()
    assert target["metadata"]["execution_mode"] == "cuda_tilelang_entry"
    assert target["metadata"]["kernel_kind"] == "minimal_cuda_jit"
    assert target["metadata"]["operator_family"] == "norm"
    assert target["metadata"]["selected_fastpath"] == "tilelang_half_layer_norm"
    assert target["metadata"]["settings"]["norm_fastpath"] == "tilelang"
    assert target["metadata"]["settings"]["preferred_patterns"] == ["norm"]


@requires_cuda
@requires_tilelang
def test_tilelang_norm_operator_stage_uses_native_cuda_fastpath_on_ada() -> None:
    config_dict = _tilelang_norm_operator_config(
        "cuda",
        norm_fastpath="auto",
        min_speedup=1.000001,
    )
    context = _runtime_context(
        config_dict,
        model=None,
        example_inputs=torch.randn(8, 64, device="cuda", dtype=torch.float16),
    )
    LoadModelPass().run(context)
    context.model = context.require_model().to(device="cuda", dtype=torch.float16)
    plan = build_operator_optimization_plan(_operator_config(config_dict))
    execution = execute_operator_optimization_plan(context, plan)

    target = execution.reports[0].to_dict()
    assert target["metadata"]["execution_mode"] == "cuda_native_fastpath"
    assert target["metadata"]["kernel_kind"] == "native_runtime_fastpath"
    assert target["metadata"]["operator_family"] == "norm"
    assert target["metadata"]["selected_fastpath"] == "native_torch_layer_norm"
    assert target["metadata"]["settings"]["norm_fastpath"] == "auto"
    assert target["metadata"]["settings"]["preferred_patterns"] == ["norm"]


@requires_cuda
@requires_tilelang
def test_tilelang_norm_operator_stage_uses_cuda_graph_fastpath() -> None:
    config_dict = _tilelang_norm_operator_config(
        "cuda",
        norm_fastpath="graph",
        min_speedup=1.000001,
    )
    context = _runtime_context(
        config_dict,
        model=None,
        example_inputs=torch.randn(8, 64, device="cuda", dtype=torch.float16),
    )
    LoadModelPass().run(context)
    context.model = context.require_model().to(device="cuda", dtype=torch.float16)
    plan = build_operator_optimization_plan(_operator_config(config_dict))
    execution = execute_operator_optimization_plan(context, plan)

    target = execution.reports[0].to_dict()
    assert target["metadata"]["execution_mode"] == "cuda_graph_tilelang_entry"
    assert target["metadata"]["kernel_kind"] == "cuda_graph_replay"
    assert target["metadata"]["operator_family"] == "norm"
    assert target["metadata"]["selected_fastpath"] == "tilelang_half_layer_norm_cuda_graph"
    assert target["metadata"]["settings"]["norm_fastpath"] == "graph"
    assert target["metadata"]["settings"]["preferred_patterns"] == ["norm"]
    assert target["metadata"]["cuda_graph"]["state"] == "replayed"
    assert target["metadata"]["cuda_graph"]["cache_size"] >= 1


@requires_cuda
@requires_tilelang
def test_tilelang_norm_wrapper_replays_cuda_graph_on_second_call() -> None:
    module = nn.LayerNorm(64).to(device="cuda", dtype=torch.float16)
    wrapper = _TileLangNormWrapper(
        module,
        fallback="eager",
        settings={
            "target_arch": "sm_89",
            "norm_fastpath": "graph",
            "preferred_patterns": ["norm"],
        },
    ).to(device="cuda", dtype=torch.float16)
    x = torch.randn(8, 64, device="cuda", dtype=torch.float16)

    first = wrapper(x).clone()
    first_metadata = wrapper.execution_metadata()
    second = wrapper(x + 1)
    second_metadata = wrapper.execution_metadata()

    assert torch.allclose(first, module(x), atol=1e-2, rtol=1e-2)
    assert torch.allclose(second, module(x + 1), atol=1e-2, rtol=1e-2)
    assert first_metadata["cuda_graph"]["state"] == "captured"
    assert second_metadata["cuda_graph"]["state"] == "replayed"
