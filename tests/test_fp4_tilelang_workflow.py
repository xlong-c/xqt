from __future__ import annotations

import importlib.util

import pytest
import torch

from xqt.workflows import XQTOptimizationSession


requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for FP4 TileLang CUDA workflow test",
)

requires_tilelang = pytest.mark.skipif(
    importlib.util.find_spec("tilelang") is None,
    reason="tilelang package is required for FP4 TileLang CUDA workflow test",
)


class _TinyMLP(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = torch.nn.Linear(64, 64)
        self.norm = torch.nn.LayerNorm(64)
        self.fc2 = torch.nn.Linear(64, 64)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.fc1(inputs)
        hidden = self.norm(hidden)
        return self.fc2(hidden)


def test_fp4_quant_stage_can_feed_tilelang_dequant_gemm_operator_stage() -> None:
    torch.manual_seed(0)
    model = _TinyMLP().eval()
    session = XQTOptimizationSession(
        project={
            "name": "fp4_tilelang_workflow",
            "artifact_dir": "artifacts/xqt/tests/fp4_tilelang_workflow",
        },
        model=model,
        example_inputs=torch.randn(64, 64, dtype=torch.float32),
    )

    quant_stage = session.quant(
        name="fp4_quant",
        backend="pytorch",
        method="awq",
        strategy="fp4_weight_only",
        policy={
            "dtype": "fp4",
            "scheme": "weight_only",
            "include_module_names": ["fc1"],
            "group_size": 64,
        },
    )
    operator_stage = session.operator(
        name="tilelang_fp4_fc1",
        from_stage="fp4_quant",
        targets=[
            {
                "name": "fc1_tilelang",
                "target": "fc1",
                "backend": "tilelang",
                "patterns": ["dequant_gemm_epilogue"],
                "min_speedup": 1.01,
                "tilelang": {
                    "target_arch": "sm_80",
                },
            }
        ],
    )

    assert quant_stage.accepted is True
    assert operator_stage.kind == "operator"

    quant_metrics = operator_stage.metrics
    assert quant_metrics["target_count"] == 1
    target = quant_metrics["targets"][0]
    assert target["backend"] == "tilelang"
    assert target["module_path"] == "fc1"
    assert target["metadata"]["execution_state"] in {"executed", "fallback"}
    assert target["metadata"]["execution_mode"] == "reference_fallback"
    assert target["metadata"]["kernel_kind"] == "reference_fallback"
    assert target["metadata"]["kernel_constraints"]["supported_patterns"] == [
        "dequant_gemm_epilogue"
    ]
    assert (
        target["metadata"]["kernel_constraints"]["supports_reference_fp4_linear_bridge"]
        is True
    )
    assert target["metadata"]["kernel_pattern"] == "fp4_packed_dequant_gemm_epilogue"
    assert target["metadata"]["weight_source"] == "reference_fp4_linear_packed_bridge"
    assert target["metadata"]["weight_representation"] == "packed_signed_int4_plus_group_scale"
    assert target["metadata"]["consumes_packed_weight"] is True
    assert target["metadata"]["unpack_stage"] == "eager_reference_fallback"
    assert target["metadata"]["settings"]["target_arch"] == "sm_80"
    assert "requires CUDA tensors" in str(target["metadata"]["execution_reason"])


@requires_cuda
@requires_tilelang
def test_fp4_tilelang_operator_stage_uses_packed_cuda_entry() -> None:
    torch.manual_seed(2)
    model = _TinyMLP().eval()
    session = XQTOptimizationSession(
        project={
            "name": "fp4_tilelang_cuda_workflow",
            "artifact_dir": "artifacts/xqt/tests/fp4_tilelang_cuda_workflow",
        },
        model=model,
        device="cuda",
        example_inputs=torch.randn(64, 64, dtype=torch.float16, device="cuda"),
    )

    session.quant(
        name="fp4_quant",
        backend="pytorch",
        method="awq",
        strategy="fp4_weight_only",
        policy={
            "dtype": "fp4",
            "scheme": "weight_only",
            "include_module_names": ["fc1"],
            "group_size": 64,
        },
    )
    operator_stage = session.operator(
        name="tilelang_fp4_fc1",
        from_stage="fp4_quant",
        targets=[
            {
                "name": "fc1_tilelang",
                "target": "fc1",
                "backend": "tilelang",
                "patterns": ["dequant_gemm_epilogue"],
                "min_speedup": 1.01,
            }
        ],
    )

    target = operator_stage.metrics["targets"][0]
    assert target["metadata"]["execution_mode"] == "cuda_tilelang_entry"
    assert target["metadata"]["kernel_kind"] == "minimal_cuda_jit"
    assert target["metadata"]["kernel_pattern"] == "fp4_packed_dequant_gemm_epilogue"
    assert target["metadata"]["weight_source"] == "reference_fp4_linear_packed_bridge"
    assert target["metadata"]["consumes_packed_weight"] is True
    assert target["metadata"]["unpack_stage"] == "tilelang_fused_gemm_kernel"
    assert target["metadata"]["fusion_status"] == "single_tilelang_kernel_for_unpack_dequant_gemm_epilogue"
    assert target["metadata"]["epilogue_stage"] == "tilelang_fused_bias_activation"
