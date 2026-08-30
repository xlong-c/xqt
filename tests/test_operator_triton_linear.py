from __future__ import annotations

import importlib.util
from typing import Any

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from xqt.kernels.wrappers import triton_wrappers
from xqt.core.errors import XQTBackendError
from xqt.kernels.wrappers.execute import execute_operator_optimization_plan
from xqt.kernels.wrappers.materialize import materialize_operator_candidate_model
from xqt.kernels.wrappers.plan import build_operator_optimization_plan
from xqt.kernels.wrappers.triton_wrappers import (
    _TritonLinearWrapper,
    build_triton_candidate_model,
)
from xqt.kernels.wrappers.types import OperatorOptimizationTargetPlan
from tests.xqt.runtime_helpers import operator_config_from_dict, operator_runtime_context


requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for Triton linear operator CUDA tests",
)

requires_triton = pytest.mark.skipif(
    importlib.util.find_spec("triton") is None,
    reason="triton package is required for Triton linear operator CUDA tests",
)


class _LinearBlock(nn.Module):
    def __init__(self, in_features: int = 64, out_features: int = 96) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def _target(
    *,
    target_path: str | None = "linear",
    patterns: list[str] | None = None,
    weight_layout: str = "transpose_stride",
    precision: str = "auto",
    linear_fastpath: str = "eager",
    cuda_graph_warmup: int = 2,
) -> OperatorOptimizationTargetPlan:
    return OperatorOptimizationTargetPlan(
        name="linear_triton",
        engine="triton",
        target_path=target_path,
        patterns=patterns or ["linear"],
        fallback="eager",
        min_speedup=0.0,
        options={
            "precision": precision,
            "target_arch": "sm_89",
            "weight_layout": weight_layout,
            "linear_fastpath": linear_fastpath,
            "cuda_graph_warmup": cuda_graph_warmup,
        },
    )


def _operator_config(
    device: str,
    *,
    weight_layout: str = "transpose_stride",
    linear_fastpath: str = "eager",
) -> dict[str, Any]:
    return {
        "config_version": 1,
        "project": {
            "name": f"triton_linear_{device}",
            "artifact_dir": f"artifacts/xqt/tests/triton_linear_{device}",
        },
        "model": {
            "target": "tests.xqt.test_operator_triton_linear:_LinearBlock",
            "params": {"in_features": 64, "out_features": 96},
            "device": device,
        },
        "operator_optimization": {
            "enabled": True,
            "default_engine": "triton",
            "targets": [
                {
                    "name": "linear_triton",
                    "target": "linear",
                    "engine": "triton",
                    "patterns": ["linear"],
                    "min_speedup": 0.0,
                    "validate": {"atol": 0.5, "rtol": 0.03},
                    "options": {
                        "precision": "bf16",
                        "target_arch": "sm_89",
                        "weight_layout": weight_layout,
                        "linear_fastpath": linear_fastpath,
                        "cuda_graph_warmup": 2,
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


def test_materialize_triton_linear_candidate_replaces_named_linear() -> None:
    model = _LinearBlock().eval()

    candidate, compile_time_ms = materialize_operator_candidate_model(
        model,
        _target(),
    )

    assert compile_time_ms is None
    assert isinstance(candidate.linear, _TritonLinearWrapper)
    assert candidate.linear.linear is not model.linear


def test_triton_linear_nested_builder_binds_to_copied_member() -> None:
    model = _LinearBlock().eval()
    original_weight = model.linear.weight.detach().clone()

    candidate = build_triton_candidate_model(
        model,
        _target(target_path=None),
    )
    assert isinstance(candidate.linear, _TritonLinearWrapper)
    with torch.no_grad():
        model.linear.weight.add_(1.0)

    assert torch.equal(candidate.linear.linear.weight, original_weight)


def test_triton_linear_prepack_is_nonpersistent_and_refreshes_after_update() -> None:
    linear = nn.Linear(8, 12, bias=True, dtype=torch.bfloat16)
    wrapper = _TritonLinearWrapper(
        linear,
        fallback="eager",
        settings={
            "precision": "bf16",
            "preferred_patterns": ["linear"],
            "weight_layout": "prepacked_kn",
        },
    )
    prepared_before, transpose_before = wrapper._runtime_weight()
    prepared_before = prepared_before.clone()

    with torch.no_grad():
        wrapper.linear.weight.add_(1.0)
    prepared_after, transpose_after = wrapper._runtime_weight()
    metadata = wrapper.execution_metadata()

    assert transpose_before is False
    assert transpose_after is False
    assert not torch.equal(prepared_before, prepared_after)
    assert torch.equal(prepared_after, wrapper.linear.weight.detach().t())
    assert "_prepared_weight_kn" not in wrapper.state_dict()
    assert metadata["kernel_constraints"]["prepack_refreshes"] == 2


def test_triton_linear_cpu_uses_reference_fallback() -> None:
    torch.manual_seed(3)
    linear = nn.Linear(8, 12, bias=True, dtype=torch.bfloat16)
    wrapper = _TritonLinearWrapper(
        linear,
        fallback="eager",
        settings={
            "precision": "bf16",
            "preferred_patterns": ["linear"],
            "weight_layout": "transpose_stride",
        },
    )
    x = torch.randn(2, 8, dtype=torch.bfloat16)

    output = wrapper(x)
    metadata = wrapper.execution_metadata()

    assert torch.equal(output, linear(x))
    assert metadata["execution_mode"] == "reference_fallback"
    assert metadata["selected_fastpath"] == "eager_reference_fallback"
    assert metadata["kernel_constraints"]["prepared_weight_bytes"] == 0
    assert metadata["cuda_graph"] == {
        "state": "disabled",
        "reason": "CUDA Graph requires CUDA tensor inputs",
        "cache_size": 0,
        "output_storage": None,
    }


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        ({"linear_fastpath": "auto"}, "linear_fastpath"),
        ({"cuda_graph_warmup": -1}, "cuda_graph_warmup"),
    ],
)
def test_triton_linear_rejects_invalid_graph_settings(
    settings: dict[str, Any],
    message: str,
) -> None:
    linear = nn.Linear(8, 12, dtype=torch.bfloat16)

    with pytest.raises(XQTBackendError, match=message):
        _TritonLinearWrapper(
            linear,
            fallback="eager",
            settings={
                "precision": "bf16",
                "preferred_patterns": ["linear"],
                "weight_layout": "transpose_stride",
                **settings,
            },
        )


def test_triton_linear_fp16_records_resolved_sm89_schedule() -> None:
    linear = nn.Linear(
        4096,
        4096,
        bias=False,
        device="meta",
        dtype=torch.float16,
    )
    wrapper = _TritonLinearWrapper(
        linear,
        fallback="error",
        settings={
            "precision": "fp16",
            "preferred_patterns": ["gemm_fp16"],
            "target_arch": "sm_89",
            "weight_layout": "transpose_stride",
        },
    )

    wrapper._record_schedule("gemm_fp16", 1, target_arch="sm_89")

    assert wrapper.execution_metadata()["kernel_schedule"] == {
        "block_m": 16,
        "block_n": 64,
        "block_k": 64,
        "group_m": 4,
        "num_warps": 4,
        "num_stages": 3,
        "target_arch": "sm_89",
        "preset": "sm89_fp16_decode_m1",
    }


@pytest.mark.parametrize(
    ("weight_layout", "expected_transpose_b"),
    [("transpose_stride", True), ("prepacked_kn", False)],
)
def test_triton_linear_fp16_resolver_receives_weight_layout(
    monkeypatch: pytest.MonkeyPatch,
    weight_layout: str,
    expected_transpose_b: bool,
) -> None:
    captured: dict[str, object] = {}
    original_resolver = triton_wrappers.resolve_triton_fp16_gemm_schedule

    def capture_resolver(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return original_resolver(**kwargs)

    monkeypatch.setattr(
        triton_wrappers,
        "resolve_triton_fp16_gemm_schedule",
        capture_resolver,
    )
    linear = nn.Linear(
        4096,
        4096,
        bias=False,
        device="meta",
        dtype=torch.float16,
    )
    wrapper = _TritonLinearWrapper(
        linear,
        fallback="error",
        settings={
            "precision": "fp16",
            "preferred_patterns": ["gemm_fp16"],
            "target_arch": "sm_89",
            "weight_layout": weight_layout,
        },
    )

    wrapper._record_schedule("gemm_fp16", 1, target_arch="sm_89")

    assert captured["transpose_b"] is expected_transpose_b


@requires_cuda
@requires_triton
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("weight_layout", ["transpose_stride", "prepacked_kn"])
def test_triton_linear_cuda_matches_torch_reference(
    dtype: torch.dtype,
    weight_layout: str,
) -> None:
    torch.manual_seed(4)
    linear = nn.Linear(64, 96, bias=True, device="cuda", dtype=dtype).eval()
    wrapper = _TritonLinearWrapper(
        linear,
        fallback="error",
        settings={
            "precision": "auto",
            "preferred_patterns": ["linear"],
            "target_arch": "sm_89",
            "weight_layout": weight_layout,
        },
    )
    x = torch.randn(2, 3, 64, device="cuda", dtype=dtype)

    output = wrapper(x)
    reference = F.linear(x, linear.weight, linear.bias)
    metadata = wrapper.execution_metadata()

    tolerance = 0.5 if dtype == torch.bfloat16 else 0.25
    assert output.shape == reference.shape
    assert torch.allclose(
        output.float(),
        reference.float(),
        atol=tolerance,
        rtol=0.03,
    )
    assert metadata["execution_mode"] == "cuda_triton_entry"
    assert metadata["kernel_pattern"] == (
        "gemm_bf16" if dtype == torch.bfloat16 else "gemm_fp16"
    )
    assert metadata["kernel_constraints"]["weight_layout"] == weight_layout
    prepared_bytes = metadata["kernel_constraints"]["prepared_weight_bytes"]
    if weight_layout == "prepacked_kn":
        assert prepared_bytes == linear.weight.numel() * linear.weight.element_size()
    else:
        assert prepared_bytes == 0


@requires_cuda
@requires_triton
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("weight_layout", ["transpose_stride", "prepacked_kn"])
def test_triton_linear_cuda_graph_captures_and_replays(
    dtype: torch.dtype,
    weight_layout: str,
) -> None:
    torch.manual_seed(5)
    linear = nn.Linear(64, 96, bias=True, device="cuda", dtype=dtype).eval()
    wrapper = _TritonLinearWrapper(
        linear,
        fallback="error",
        settings={
            "precision": "auto",
            "preferred_patterns": ["linear"],
            "target_arch": "sm_89",
            "weight_layout": weight_layout,
            "linear_fastpath": "graph",
            "cuda_graph_warmup": 1,
        },
    )
    x = torch.randn(2, 3, 64, device="cuda", dtype=dtype)
    x_next = (x + 0.25).contiguous()

    with torch.inference_mode():
        first_output = wrapper(x)
        first_snapshot = first_output.clone()
        first_metadata = wrapper.execution_metadata()
        second_output = wrapper(x_next)
        second_snapshot = second_output.clone()
        second_metadata = wrapper.execution_metadata()
    reference_first = F.linear(x, linear.weight, linear.bias)
    reference_second = F.linear(x_next, linear.weight, linear.bias)

    tolerance = 0.5 if dtype == torch.bfloat16 else 0.25
    assert torch.allclose(
        first_snapshot.float(),
        reference_first.float(),
        atol=tolerance,
        rtol=0.03,
    )
    assert torch.allclose(
        second_snapshot.float(),
        reference_second.float(),
        atol=tolerance,
        rtol=0.03,
    )
    assert first_output.data_ptr() == second_output.data_ptr()
    assert first_metadata["execution_mode"] == "cuda_graph_triton_entry"
    assert first_metadata["kernel_kind"] == "cuda_graph_replay"
    assert first_metadata["cuda_graph"] == {
        "state": "captured",
        "reason": None,
        "cache_size": 1,
        "output_storage": "graph_owned",
    }
    assert second_metadata["cuda_graph"] == {
        "state": "replayed",
        "reason": None,
        "cache_size": 1,
        "output_storage": "graph_owned",
    }
    assert second_metadata["selected_fastpath"] == (
        f"triton_{'bf16' if dtype == torch.bfloat16 else 'fp16'}_linear_"
        f"{weight_layout}_cuda_graph"
    )


@requires_cuda
@requires_triton
@pytest.mark.parametrize("weight_layout", ["transpose_stride", "prepacked_kn"])
def test_triton_linear_cuda_graph_supports_inference_tensor_parameters(
    weight_layout: str,
) -> None:
    with torch.inference_mode():
        linear = nn.Linear(
            64,
            96,
            bias=True,
            device="cuda",
            dtype=torch.bfloat16,
        ).eval()
        wrapper = _TritonLinearWrapper(
            linear,
            fallback="error",
            settings={
                "precision": "bf16",
                "preferred_patterns": ["linear"],
                "target_arch": "sm_89",
                "weight_layout": weight_layout,
                "linear_fastpath": "graph",
            },
        )
        x = torch.randn(2, 64, device="cuda", dtype=torch.bfloat16)

        wrapper(x)
        output = wrapper(x).clone()
        reference = F.linear(x, linear.weight, linear.bias)

    assert torch.allclose(output.float(), reference.float(), atol=0.5, rtol=0.03)
    assert wrapper.execution_metadata()["cuda_graph"]["state"] == "replayed"


@requires_cuda
@requires_triton
def test_triton_linear_cuda_graph_cache_key_tracks_input_contract() -> None:
    linear = nn.Linear(
        64,
        96,
        bias=True,
        device="cuda",
        dtype=torch.bfloat16,
    ).eval()
    wrapper = _TritonLinearWrapper(
        linear,
        fallback="error",
        settings={
            "precision": "bf16",
            "preferred_patterns": ["linear"],
            "target_arch": "sm_89",
            "weight_layout": "transpose_stride",
            "linear_fastpath": "graph",
        },
    )
    x = torch.randn(2, 64, device="cuda", dtype=torch.bfloat16)
    strided_storage = torch.empty(2, 128, device="cuda", dtype=torch.bfloat16)
    different_stride = strided_storage[:, ::2]
    runtime_weight, transpose_b = wrapper._runtime_weight()
    wrapper._record_schedule("gemm_bf16", 2, target_arch="sm_89")

    def cache_key(tensor: torch.Tensor) -> tuple[Any, ...]:
        return wrapper._linear_graph_cache_key(
            tensor,
            pattern="gemm_bf16",
            target_arch="sm_89",
            runtime_weight=runtime_weight,
            transpose_b=transpose_b,
        )

    base_key = cache_key(x)
    assert cache_key(x.clone()) == base_key
    assert cache_key(x[:1]) != base_key
    assert cache_key(different_stride) != base_key
    assert cache_key(x.float()) != base_key


@requires_cuda
@requires_triton
@pytest.mark.parametrize("parameter_name", ["weight", "bias"])
@pytest.mark.parametrize("weight_layout", ["transpose_stride", "prepacked_kn"])
def test_triton_linear_cuda_graph_recaptures_after_parameter_update(
    parameter_name: str,
    weight_layout: str,
) -> None:
    torch.manual_seed(6)
    linear = nn.Linear(
        64,
        96,
        bias=True,
        device="cuda",
        dtype=torch.bfloat16,
    ).eval()
    wrapper = _TritonLinearWrapper(
        linear,
        fallback="error",
        settings={
            "precision": "bf16",
            "preferred_patterns": ["linear"],
            "target_arch": "sm_89",
            "weight_layout": weight_layout,
            "linear_fastpath": "graph",
            "cuda_graph_warmup": 1,
        },
    )
    x = torch.randn(2, 64, device="cuda", dtype=torch.bfloat16)

    with torch.inference_mode():
        wrapper(x)
        parameter = getattr(wrapper.linear, parameter_name)
        assert isinstance(parameter, torch.Tensor)
        parameter.add_(0.125)
        output = wrapper(x).clone()
    metadata = wrapper.execution_metadata()
    reference = F.linear(x, wrapper.linear.weight, wrapper.linear.bias)

    assert torch.allclose(output.float(), reference.float(), atol=0.5, rtol=0.03)
    assert metadata["cuda_graph"]["state"] == "captured"
    assert metadata["cuda_graph"]["cache_size"] == 1
    if weight_layout == "prepacked_kn" and parameter_name == "weight":
        assert metadata["kernel_constraints"]["prepack_refreshes"] == 2


@requires_cuda
@requires_triton
def test_triton_linear_cuda_graph_cache_is_cleared_by_module_apply() -> None:
    linear = nn.Linear(
        64,
        96,
        bias=True,
        device="cuda",
        dtype=torch.bfloat16,
    ).eval()
    wrapper = _TritonLinearWrapper(
        linear,
        fallback="error",
        settings={
            "precision": "auto",
            "preferred_patterns": ["linear"],
            "target_arch": "sm_89",
            "weight_layout": "transpose_stride",
            "linear_fastpath": "graph",
        },
    )
    x = torch.randn(2, 64, device="cuda", dtype=torch.bfloat16)

    with torch.inference_mode():
        wrapper(x)
    assert wrapper.execution_metadata()["cuda_graph"]["cache_size"] == 1

    wrapper.to(dtype=torch.float16)
    metadata = wrapper.execution_metadata()

    assert metadata["cuda_graph"] == {
        "state": "disabled",
        "reason": "module device or dtype changed",
        "cache_size": 0,
        "output_storage": None,
    }


@requires_cuda
@requires_triton
def test_triton_linear_cuda_graph_capture_error_falls_back_to_eager_triton(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    linear = nn.Linear(
        64,
        96,
        bias=True,
        device="cuda",
        dtype=torch.bfloat16,
    ).eval()
    wrapper = _TritonLinearWrapper(
        linear,
        fallback="error",
        settings={
            "precision": "bf16",
            "preferred_patterns": ["linear"],
            "target_arch": "sm_89",
            "weight_layout": "transpose_stride",
            "linear_fastpath": "graph",
        },
    )
    x = torch.randn(2, 64, device="cuda", dtype=torch.bfloat16)

    def fail_capture(*_: Any, **__: Any) -> dict[str, Any]:
        raise RuntimeError("synthetic capture failure")

    monkeypatch.setattr(
        "xqt.kernels.wrappers.triton_wrappers.capture_cuda_graph_with_static_state",
        fail_capture,
    )

    output = wrapper(x)
    metadata = wrapper.execution_metadata()
    reference = F.linear(x, linear.weight, linear.bias)

    assert torch.allclose(output.float(), reference.float(), atol=0.5, rtol=0.03)
    assert metadata["execution_mode"] == "cuda_triton_entry"
    assert metadata["selected_fastpath"] == (
        "triton_bf16_linear_transpose_stride"
    )
    assert metadata["cuda_graph"]["state"] == "fallback_eager"
    assert "synthetic capture failure" in metadata["cuda_graph"]["reason"]
    assert metadata["cuda_graph"]["cache_size"] == 0


@requires_cuda
@requires_triton
def test_triton_linear_explicit_bf16_pattern_rejects_fp16_runtime() -> None:
    linear = nn.Linear(64, 96, device="cuda", dtype=torch.float16).eval()
    with pytest.raises(Exception, match="configured precision 'bf16'"):
        _TritonLinearWrapper(
            linear,
            fallback="error",
            settings={
                "preferred_patterns": ["gemm_bf16"],
                "weight_layout": "transpose_stride",
            },
        )


def test_triton_linear_operator_stage_reports_reference_fallback_on_cpu() -> None:
    config = _operator_config("cpu")
    model = _LinearBlock().eval().to(dtype=torch.bfloat16)
    context = operator_runtime_context(
        config,
        model=model,
        example_inputs=torch.randn(4, 64, dtype=torch.bfloat16),
    )
    plan = build_operator_optimization_plan(operator_config_from_dict(config))

    execution = execute_operator_optimization_plan(context, plan)
    report = execution.reports[0].to_dict()

    assert report["metadata"]["execution_mode"] == "reference_fallback"
    assert report["metadata"]["operator_family"] == "linear"
    assert report["metadata"]["kernel_constraints"]["weight_layout"] == (
        "transpose_stride"
    )


@requires_cuda
@requires_triton
@pytest.mark.parametrize("weight_layout", ["transpose_stride", "prepacked_kn"])
def test_triton_bf16_linear_operator_stage_uses_cuda_entry(
    weight_layout: str,
) -> None:
    config = _operator_config("cuda", weight_layout=weight_layout)
    model = _LinearBlock().eval().to(device="cuda", dtype=torch.bfloat16)
    context = operator_runtime_context(
        config,
        model=model,
        example_inputs=torch.randn(4, 64, device="cuda", dtype=torch.bfloat16),
    )
    plan = build_operator_optimization_plan(operator_config_from_dict(config))

    execution = execute_operator_optimization_plan(context, plan)
    report = execution.reports[0].to_dict()

    assert report["metadata"]["execution_mode"] == "cuda_triton_entry"
    assert report["metadata"]["kernel_kind"] == "minimal_cuda_jit"
    assert report["metadata"]["operator_family"] == "linear"
    assert report["metadata"]["kernel_pattern"] == "gemm_bf16"
    assert report["metadata"]["kernel_constraints"]["weight_layout"] == weight_layout


@requires_cuda
@requires_triton
@pytest.mark.parametrize("weight_layout", ["transpose_stride", "prepacked_kn"])
def test_triton_bf16_linear_operator_stage_uses_cuda_graph_entry(
    weight_layout: str,
) -> None:
    config = _operator_config(
        "cuda",
        weight_layout=weight_layout,
        linear_fastpath="graph",
    )
    model = _LinearBlock().eval().to(device="cuda", dtype=torch.bfloat16)
    context = operator_runtime_context(
        config,
        model=model,
        example_inputs=torch.randn(4, 64, device="cuda", dtype=torch.bfloat16),
    )
    plan = build_operator_optimization_plan(operator_config_from_dict(config))

    execution = execute_operator_optimization_plan(context, plan)
    report = execution.reports[0].to_dict()

    assert report["metadata"]["execution_mode"] == "cuda_graph_triton_entry"
    assert report["metadata"]["kernel_kind"] == "cuda_graph_replay"
    assert report["metadata"]["operator_family"] == "linear"
    assert report["metadata"]["kernel_pattern"] == "gemm_bf16"
    assert report["metadata"]["selected_fastpath"] == (
        f"triton_bf16_linear_{weight_layout}_cuda_graph"
    )
    assert report["metadata"]["linear_fastpath"] == "graph"
    assert report["metadata"]["cuda_graph"]["state"] == "replayed"
    assert report["metadata"]["cuda_graph"]["cache_size"] >= 1
