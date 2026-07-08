from __future__ import annotations

import importlib.util

import pytest
import torch
from torch import nn

from xqt.core.config import load_xqt_config
from xqt.operator_opt.executor import (
    _TritonRMSNormWrapper,
    build_operator_optimization_plan,
    execute_operator_optimization_plan,
    materialize_operator_candidate_model,
)
from xqt.operator_opt.kernels.triton.pointwise import (
    fused_channel_first_l2norm_reference,
    fused_rmsnorm_reference,
    fused_rmsnorm_triton,
)
from xqt.operator_opt.types import OperatorOptimizationTargetPlan
from xqt.pipeline.runner import create_context


requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for Triton RMSNorm operator CUDA test",
)

requires_triton = pytest.mark.skipif(
    importlib.util.find_spec("triton") is None,
    reason="triton package is required for Triton RMSNorm operator CUDA test",
)


class _FakeWanRMSNorm(nn.Module):
    def __init__(self, dim: int, *, eps: float = 1e-6) -> None:
        super().__init__()
        self.scale = float(dim) ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.gamma.to(device=x.device, dtype=x.dtype)
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * weight * self.scale


class _FakeWanChannelFirstRMSNorm(nn.Module):
    def __init__(self, dim: int, *, eps: float = 1e-12, images: bool = False) -> None:
        super().__init__()
        self.channel_first = True
        self.scale = float(dim) ** 0.5
        shape = (dim, 1, 1) if images else (dim, 1, 1, 1)
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = 0.0
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = torch.nn.functional.normalize(x.float(), dim=1, eps=self.eps).to(x.dtype)
        weight = self.gamma.to(device=x.device, dtype=x.dtype)
        return normalized * weight * self.scale


class _FakeWanRMSNormBlock(nn.Module):
    def __init__(self, hidden_dim: int = 64, *, eps: float = 1e-6) -> None:
        super().__init__()
        self.norm = _FakeWanRMSNorm(hidden_dim, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)


def _triton_rmsnorm_operator_config(device: str, *, min_speedup: float = 1.000001) -> dict:
    return {
        "config_version": 1,
        "project": {
            "name": f"triton_rmsnorm_{device}",
            "artifact_dir": f"artifacts/xqt/tests/triton_rmsnorm_{device}",
        },
        "model": {
            "target": "tests.xqt.test_operator_triton_rmsnorm:_FakeWanRMSNormBlock",
            "params": {
                "hidden_dim": 64,
            },
            "device": device,
        },
        "operator_optimization": {
            "enabled": True,
            "default_engine": "triton",
            "targets": [
                {
                    "name": "rmsnorm_triton",
                    "target": "norm",
                    "engine": "triton",
                    "patterns": ["rmsnorm"],
                    "min_speedup": min_speedup,
                    "options": {
                        "eps": 1e-6,
                        "block_size": 1024,
                        "num_warps": 4,
                        "num_stages": 4,
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


def test_triton_rmsnorm_reference_matches_fake_module() -> None:
    torch.manual_seed(0)
    module = _FakeWanRMSNorm(64).eval()
    x = torch.randn(4, 64, dtype=torch.float32)
    weight = module.gamma * module.scale

    reference = fused_rmsnorm_reference(x, weight, eps=module.eps)
    output = module(x)

    assert reference.shape == output.shape
    assert torch.allclose(reference, output, atol=1e-6, rtol=1e-6)


def test_triton_rmsnorm_reference_supports_bfloat16() -> None:
    torch.manual_seed(0)
    module = _FakeWanRMSNorm(64).eval()
    x = torch.randn(4, 64, dtype=torch.bfloat16)
    weight = (module.gamma * module.scale).to(dtype=torch.bfloat16)

    reference = fused_rmsnorm_reference(x, weight, eps=module.eps)

    assert reference.shape == x.shape
    assert reference.dtype == torch.bfloat16


def test_channel_first_l2norm_reference_matches_fake_vae_module() -> None:
    torch.manual_seed(0)
    module = _FakeWanChannelFirstRMSNorm(64).eval()
    x = torch.randn(2, 64, 3, 8, 8, dtype=torch.float32)
    weight = module.gamma.to(dtype=x.dtype).reshape(-1) * module.scale

    reference = fused_channel_first_l2norm_reference(x, weight, eps=module.eps)
    output = module(x)

    assert reference.shape == output.shape
    assert torch.allclose(reference, output, atol=1e-6, rtol=1e-6)


def test_triton_rmsnorm_operator_stage_uses_reference_fallback_on_cpu() -> None:
    config = load_xqt_config(_triton_rmsnorm_operator_config("cpu"))
    context = create_context(
        config,
        model=_FakeWanRMSNormBlock().eval(),
        example_inputs=torch.randn(8, 64, dtype=torch.float32),
    )
    plan = build_operator_optimization_plan(config.operator_optimization)
    execution = execute_operator_optimization_plan(context, plan)

    target = execution.reports[0].to_dict()
    assert target["metadata"]["execution_mode"] == "reference_fallback"
    assert target["metadata"]["kernel_kind"] == "reference_fallback"
    assert target["metadata"]["operator_family"] == "norm"
    assert target["metadata"]["kernel_pattern"] == "rmsnorm"
    assert target["metadata"]["selected_fastpath"] == "eager_reference_fallback"
    assert target["metadata"]["settings"]["preferred_patterns"] == ["rmsnorm"]


def test_materialize_triton_rmsnorm_candidate_replaces_named_norm() -> None:
    module = _FakeWanRMSNormBlock().eval()
    target = OperatorOptimizationTargetPlan(
        name="rmsnorm_triton",
        engine="triton",
        target_path="norm",
        patterns=["rmsnorm"],
        fallback="eager",
        min_speedup=0.0,
        options={
            "eps": 1e-6,
            "block_size": 1024,
            "num_warps": 4,
            "num_stages": 4,
        },
    )

    candidate, compile_time_ms = materialize_operator_candidate_model(module, target)

    assert compile_time_ms is None
    wrapped = candidate.get_submodule("norm")
    assert type(wrapped).__name__ == "_TritonRMSNormWrapper"


def test_triton_rmsnorm_operator_stage_reports_bfloat16_reference_metadata_on_cpu() -> None:
    config = load_xqt_config(_triton_rmsnorm_operator_config("cpu"))
    context = create_context(
        config,
        model=_FakeWanRMSNormBlock().eval().to(dtype=torch.bfloat16),
        example_inputs=torch.randn(8, 64, dtype=torch.bfloat16),
    )
    plan = build_operator_optimization_plan(config.operator_optimization)
    execution = execute_operator_optimization_plan(context, plan)

    target = execution.reports[0].to_dict()
    assert target["metadata"]["execution_mode"] == "reference_fallback"
    assert target["metadata"]["kernel_constraints"]["dtype"] == "bfloat16"
    assert target["metadata"]["kernel_constraints"]["supported_dtypes"] == [
        "float16",
        "bfloat16",
    ]


def test_triton_rmsnorm_cpu_fallback_metadata_reports_channel_first_when_needed() -> None:
    module = _FakeWanChannelFirstRMSNorm(64).eval()
    wrapped = _TritonRMSNormWrapper(  # type: ignore[name-defined]
        module,
        fallback="eager",
        settings={"eps": 1e-6, "block_size": 1024, "sites_per_program": 8},
    )
    x = torch.randn(2, 64, 3, 8, 8, dtype=torch.float32)

    _ = wrapped(x)
    metadata = wrapped.execution_metadata()

    assert metadata["kernel_pattern"] == "rmsnorm_channel_first"
    assert metadata["kernel_constraints"]["supports_channel_first"] is True
    assert metadata["kernel_constraints"]["channel_layout"] == "channel_first"


@requires_cuda
@requires_triton
def test_triton_half_rmsnorm_cuda_kernel_matches_reference() -> None:
    torch.manual_seed(0)
    x = torch.randn(8, 64, device="cuda", dtype=torch.float16)
    weight = torch.randn(64, device="cuda", dtype=torch.float16)

    output = fused_rmsnorm_triton(
        x,
        weight,
        eps=1e-6,
        block_size=1024,
        num_warps=4,
        num_stages=4,
    )
    reference = fused_rmsnorm_reference(x, weight, eps=1e-6)

    assert output.shape == reference.shape
    assert output.dtype == torch.float16
    assert torch.allclose(output.float(), reference.float(), atol=1e-2, rtol=1e-2)


@requires_cuda
@requires_triton
def test_triton_bfloat16_rmsnorm_cuda_kernel_matches_reference() -> None:
    torch.manual_seed(0)
    x = torch.randn(8, 64, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(64, device="cuda", dtype=torch.bfloat16)

    output = fused_rmsnorm_triton(
        x,
        weight,
        eps=1e-6,
        block_size=1024,
        num_warps=4,
        num_stages=4,
    )
    reference = fused_rmsnorm_reference(x, weight, eps=1e-6)

    assert output.shape == reference.shape
    assert output.dtype == torch.bfloat16
    assert torch.allclose(output.float(), reference.float(), atol=2e-2, rtol=2e-2)


@requires_cuda
@requires_triton
def test_triton_channel_first_half_cuda_kernel_matches_reference() -> None:
    torch.manual_seed(0)
    module = _FakeWanChannelFirstRMSNorm(64).eval().to(device="cuda", dtype=torch.float16)
    wrapped = _TritonRMSNormWrapper(  # type: ignore[name-defined]
        module,
        fallback="eager",
        settings={
            "eps": 1e-6,
            "block_size": 1024,
            "sites_per_program": 8,
            "num_warps": 4,
            "num_stages": 4,
        },
    ).cuda()
    x = torch.randn(2, 64, 3, 8, 8, device="cuda", dtype=torch.float16)

    output = wrapped(x)
    reference = module(x)
    metadata = wrapped.execution_metadata()

    assert output.shape == reference.shape
    assert torch.allclose(output.float(), reference.float(), atol=1e-2, rtol=1e-2)
    assert metadata["kernel_pattern"] == "rmsnorm_channel_first"
    assert metadata["selected_fastpath"] == "triton_half_channel_first_norm"


@requires_cuda
@requires_triton
def test_triton_rmsnorm_operator_stage_uses_cuda_kernel_entry() -> None:
    config = load_xqt_config(_triton_rmsnorm_operator_config("cuda"))
    context = create_context(
        config,
        model=_FakeWanRMSNormBlock().eval().to(device="cuda", dtype=torch.float16),
        example_inputs=torch.randn(8, 64, device="cuda", dtype=torch.float16),
    )
    plan = build_operator_optimization_plan(config.operator_optimization)
    execution = execute_operator_optimization_plan(context, plan)

    target = execution.reports[0].to_dict()
    assert target["metadata"]["execution_mode"] == "cuda_triton_entry"
    assert target["metadata"]["kernel_kind"] == "minimal_cuda_jit"
    assert target["metadata"]["operator_family"] == "norm"
    assert target["metadata"]["kernel_pattern"] == "rmsnorm"
    assert target["metadata"]["selected_fastpath"] == "triton_half_rmsnorm"
    assert target["metadata"]["settings"]["preferred_patterns"] == ["rmsnorm"]
    assert target["metadata"]["kernel_constraints"]["dtype"] == "float16"


@requires_cuda
@requires_triton
def test_triton_rmsnorm_operator_stage_uses_bfloat16_cuda_kernel_entry() -> None:
    config = load_xqt_config(_triton_rmsnorm_operator_config("cuda"))
    context = create_context(
        config,
        model=_FakeWanRMSNormBlock().eval().to(device="cuda", dtype=torch.bfloat16),
        example_inputs=torch.randn(8, 64, device="cuda", dtype=torch.bfloat16),
    )
    plan = build_operator_optimization_plan(config.operator_optimization)
    execution = execute_operator_optimization_plan(context, plan)

    target = execution.reports[0].to_dict()
    assert target["metadata"]["execution_mode"] == "cuda_triton_entry"
    assert target["metadata"]["kernel_kind"] == "minimal_cuda_jit"
    assert target["metadata"]["operator_family"] == "norm"
    assert target["metadata"]["kernel_pattern"] == "rmsnorm"
    assert target["metadata"]["selected_fastpath"] == "triton_bf16_rmsnorm"
    assert target["metadata"]["kernel_constraints"]["dtype"] == "bfloat16"
    assert target["metadata"]["kernel_constraints"]["supported_dtypes"] == [
        "float16",
        "bfloat16",
    ]
