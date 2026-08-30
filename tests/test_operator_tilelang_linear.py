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
from xqt.kernels.wrappers import OperatorOptimizationTargetPlan
from xqt.kernels.wrappers.execute import execute_operator_optimization_plan
from xqt.kernels.ops._impl.tilelang._common import tilelang_runtime_usable
from xqt.kernels.wrappers.tilelang_wrappers import build_tilelang_candidate_model
from xqt.kernels.wrappers.plan import build_operator_optimization_plan
from xqt.kernels.wrappers.linear import _TileLangLinearWrapper
from xqt.pipeline.passes import LoadModelPass


requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for TileLang linear operator CUDA test",
)

requires_tilelang = pytest.mark.skipif(
    not tilelang_runtime_usable(),
    reason="a runtime-compatible TileLang adapter is required for TileLang linear operator CUDA test",
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


def _tilelang_linear_operator_config(
    device: str,
    *,
    linear_runtime: str = "tilelang",
    min_speedup: float = 1.01,
    patterns: list[str] | None = None,
) -> dict:
    return {
        "config_version": 1,
        "project": {
            "name": f"tilelang_linear_{device}",
            "artifact_dir": f"artifacts/xqt/tests/tilelang_linear_{device}",
        },
        "model": {
            "target": "xqt.kernels.nn.fixtures.toy_models.build_toy_linear_block",
            "params": {
                "input_dim": 64,
                "hidden_dim": 64,
                "output_dim": 32,
            },
            "device": device,
        },
        "operator_optimization": {
            "enabled": True,
            "default_engine": "tilelang",
            "targets": [
                {
                    "name": "linear_tilelang",
                    "target": "linear",
                    "engine": "tilelang",
                    "patterns": patterns or ["linear"],
                    "min_speedup": min_speedup,
                    "tilelang": {
                        "target_arch": "sm_89",
                        "linear_runtime": linear_runtime,
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


def test_tilelang_linear_operator_stage_uses_reference_fallback_on_cpu() -> None:
    config_dict = _tilelang_linear_operator_config("cpu")
    context = _runtime_context(
        config_dict,
        model=None,
        example_inputs=torch.randn(64, 64, dtype=torch.float32),
    )
    LoadModelPass().run(context)
    plan = build_operator_optimization_plan(_operator_config(config_dict))
    execution = execute_operator_optimization_plan(context, plan)

    target = execution.reports[0].to_dict()
    assert target["metadata"]["execution_mode"] == "reference_fallback"
    assert target["metadata"]["kernel_kind"] == "reference_fallback"
    assert target["metadata"]["operator_family"] == "linear"
    assert target["metadata"]["selected_fastpath"] == "eager_reference_fallback"
    assert (
        target["metadata"]["candidate_materialization"]
        == "root_candidate_deepcopy_block_benchmark"
    )
    assert target["metadata"]["settings"]["preferred_patterns"] == ["linear"]


def test_tilelang_linear_marlin_operator_pattern_uses_reference_fallback_on_cpu() -> None:
    config_dict = _tilelang_linear_operator_config("cpu", patterns=["linear_marlin"])
    context = _runtime_context(
        config_dict,
        model=None,
        example_inputs=torch.randn(64, 64, dtype=torch.float32),
    )
    LoadModelPass().run(context)
    plan = build_operator_optimization_plan(_operator_config(config_dict))
    execution = execute_operator_optimization_plan(context, plan)

    target = execution.reports[0].to_dict()
    assert target["metadata"]["execution_mode"] == "reference_fallback"
    assert target["metadata"]["kernel_kind"] == "reference_fallback"
    assert target["metadata"]["operator_family"] == "linear"
    assert target["metadata"]["settings"]["preferred_patterns"] == ["linear_marlin"]


@requires_cuda
@requires_tilelang
def test_tilelang_linear_operator_stage_uses_cuda_kernel_entry() -> None:
    config_dict = _tilelang_linear_operator_config(
        "cuda",
        linear_runtime="tilelang",
    )
    context = _runtime_context(
        config_dict,
        model=None,
        example_inputs=torch.randn(64, 64, device="cuda", dtype=torch.float16),
    )
    LoadModelPass().run(context)
    context.model = context.require_model().to(device="cuda", dtype=torch.float16)
    plan = build_operator_optimization_plan(_operator_config(config_dict))
    execution = execute_operator_optimization_plan(context, plan)

    target = execution.reports[0].to_dict()
    assert target["metadata"]["execution_mode"] == "cuda_tilelang_entry"
    assert target["metadata"]["kernel_kind"] == "minimal_cuda_jit"
    assert target["metadata"]["operator_family"] == "linear"
    assert target["metadata"]["selected_fastpath"] == "tilelang_half_linear_kernel"
    assert target["metadata"]["settings"]["linear_runtime"] == "tilelang"
    assert target["metadata"]["settings"]["preferred_patterns"] == ["linear"]


@requires_cuda
@requires_tilelang
def test_tilelang_bf16_linear_operator_stage_uses_direct_cuda_kernel() -> None:
    config_dict = _tilelang_linear_operator_config(
        "cuda",
        linear_runtime="tilelang",
    )
    context = _runtime_context(
        config_dict,
        model=None,
        example_inputs=torch.randn(64, 64, device="cuda", dtype=torch.bfloat16),
    )
    LoadModelPass().run(context)
    context.model = context.require_model().to(device="cuda", dtype=torch.bfloat16)
    plan = build_operator_optimization_plan(_operator_config(config_dict))
    execution = execute_operator_optimization_plan(context, plan)

    target = execution.reports[0].to_dict()
    constraints = target["metadata"]["kernel_constraints"]
    assert target["metadata"]["execution_mode"] == "cuda_tilelang_entry"
    assert target["metadata"]["kernel_kind"] == "minimal_cuda_jit"
    assert target["metadata"]["selected_fastpath"] == "tilelang_half_linear_kernel"
    assert constraints["dtype"] == "bfloat16"
    assert constraints["supported_dtypes"] == ["float16", "bfloat16"]
    assert constraints["bfloat16_block_k_multiple"] == 16


@requires_cuda
@requires_tilelang
def test_tilelang_bf16_decode_wrapper_reports_promoted_schedule() -> None:
    torch.manual_seed(2)
    linear = nn.Linear(64, 96, bias=True, device="cuda", dtype=torch.bfloat16)
    wrapper = _TileLangLinearWrapper(
        linear,
        fallback="error",
        settings={
            "target_arch": "sm_89",
            "linear_runtime": "tilelang",
            "preferred_patterns": ["linear"],
        },
    )
    x = torch.randn(1, 64, device="cuda", dtype=torch.bfloat16)

    output = wrapper(x)
    reference = linear(x)
    metadata = wrapper.execution_metadata()

    assert torch.allclose(output.float(), reference.float(), atol=1e-2, rtol=1e-2)
    assert metadata["kernel_schedule"] == {
        "block_m": 16,
        "block_n": 64,
        "block_k": 32,
        "threads": 128,
        "num_stages": 2,
        "target_arch": "sm_89",
        "preset": "sm89_bf16_decode_m_le_4",
    }


@requires_cuda
@requires_tilelang
def test_tilelang_linear_operator_stage_uses_native_cuda_fastpath_on_ada() -> None:
    config_dict = _tilelang_linear_operator_config(
        "cuda",
        linear_runtime="auto",
        min_speedup=1.000001,
    )
    context = _runtime_context(
        config_dict,
        model=None,
        example_inputs=torch.randn(64, 64, device="cuda", dtype=torch.float16),
    )
    LoadModelPass().run(context)
    context.model = context.require_model().to(device="cuda", dtype=torch.float16)
    plan = build_operator_optimization_plan(_operator_config(config_dict))
    execution = execute_operator_optimization_plan(context, plan)

    target = execution.reports[0].to_dict()
    assert target["metadata"]["execution_mode"] == "cuda_native_fastpath"
    assert target["metadata"]["kernel_kind"] == "native_runtime_fastpath"
    assert target["metadata"]["operator_family"] == "linear"
    assert target["metadata"]["selected_fastpath"] == "native_torch_linear"
    assert target["metadata"]["settings"]["linear_runtime"] == "auto"
    assert target["metadata"]["settings"]["preferred_patterns"] == ["linear"]


def test_tilelang_dequant_materialization_prefers_fp16_for_float32_only_buffers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeNVFP4Linear(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.in_features = 8
            self.out_features = 4
            self.dense_weight = object()
            self.register_buffer("weight", torch.ones(4, 4, dtype=torch.uint8))
            self.register_buffer("weight_scale", torch.ones(4, 2, 1, dtype=torch.float32))
            self.register_buffer("weight_scale_2", torch.ones(1, dtype=torch.float32))
            self.register_buffer("bias", torch.zeros(4, dtype=torch.float32))

        def tilelang_dense_linear_args(
            self,
            *,
            dtype: torch.dtype,
            device: torch.device,
        ) -> tuple[torch.Tensor, torch.Tensor | None, None]:
            weight = torch.ones(
                self.out_features,
                self.in_features,
                dtype=dtype,
                device=device,
            )
            bias = torch.zeros(self.out_features, dtype=dtype, device=device)
            return weight, bias, None

    target = OperatorOptimizationTargetPlan(
        name="tilelang_nvfp4",
        engine="tilelang",
        target_path="",
        patterns=["dequant_gemm_epilogue"],
        fallback="eager",
        tilelang={
            "target_arch": "sm_89",
            "linear_runtime": "auto",
            "linear_fastpath": "auto",
        },
    )

    monkeypatch.setattr(
        "xqt.kernels.wrappers.dequant_gemm._module_has_cuda_state",
        lambda module: True,
    )
    candidate = build_tilelang_candidate_model(_FakeNVFP4Linear(), target)

    assert type(candidate).__name__ == "_TileLangEagerDenseLinearModule"
    assert candidate.linear.weight.dtype == torch.float16
    assert candidate.linear.bias is not None
    assert candidate.linear.bias.dtype == torch.float16
