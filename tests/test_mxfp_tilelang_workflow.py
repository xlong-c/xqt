from __future__ import annotations

import torch
from torch import nn

from xqt.operator_opt import (
    OperatorOptimizationTargetPlan,
    materialize_operator_candidate_model,
)
from xqt.quant import MXFPWeightOnlyLinear
from xqt.workflows import XQTOptimizationSession


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


def test_mxfp_quant_stage_can_feed_tilelang_dequant_gemm_operator_stage() -> None:
    torch.manual_seed(0)
    model = _TinyMLP().eval()
    session = XQTOptimizationSession(
        project={
            "name": "mxfp_tilelang_workflow",
            "artifact_dir": "artifacts/xqt/tests/mxfp_tilelang_workflow",
        },
        model=model,
        example_inputs=torch.randn(64, 64, dtype=torch.float32),
    )

    quant_stage = session.quant(
        name="mxfp_quant",
        backend="pytorch",
        method="awq",
        strategy="mxfp_weight_only",
        policy={
            "dtype": "mxfp",
            "scheme": "weight_only",
            "precision": 8,
            "block_size": 32,
            "include_module_names": ["fc1"],
        },
    )
    operator_stage = session.operator(
        name="tilelang_mxfp_fc1",
        from_stage="mxfp_quant",
        targets=[
            {
                "name": "fc1_tilelang",
                "target": "fc1",
                "engine": "tilelang",
                "patterns": ["dequant_gemm_epilogue"],
                "min_speedup": 1.01,
                "tilelang": {
                    "target_arch": "sm_89",
                },
            }
        ],
    )

    assert quant_stage.accepted is True
    assert operator_stage.kind == "operator"

    target = operator_stage.metrics["targets"][0]
    assert target["engine"] == "tilelang"
    assert target["module_path"] == "fc1"
    assert target["metadata"]["execution_mode"] == "reference_fallback"
    assert target["metadata"]["kernel_pattern"] == "dense_linear_epilogue"
    assert (
        target["metadata"]["weight_source"]
        == "mxfp_weight_only_linear_dense_cache_bridge"
    )
    assert (
        target["metadata"]["weight_representation"]
        == "dense_dequantized_mxfp8_weight_cache"
    )
    assert target["metadata"]["consumes_packed_weight"] is False


def test_mxfp4_quant_stage_can_feed_explicit_tilelang_packed_dequant_operator_stage() -> None:
    torch.manual_seed(0)
    model = _TinyMLP().eval()
    session = XQTOptimizationSession(
        project={
            "name": "mxfp4_tilelang_packed_workflow",
            "artifact_dir": "artifacts/xqt/tests/mxfp4_tilelang_packed_workflow",
        },
        model=model,
        example_inputs=torch.randn(64, 64, dtype=torch.float32),
    )

    quant_stage = session.quant(
        name="mxfp4_quant",
        backend="pytorch",
        method="awq",
        strategy="mxfp_weight_only",
        policy={
            "dtype": "mxfp",
            "scheme": "weight_only",
            "precision": 4,
            "block_size": 32,
            "include_module_names": ["fc1"],
        },
    )
    operator_stage = session.operator(
        name="tilelang_mxfp4_fc1_packed",
        from_stage="mxfp4_quant",
        targets=[
            {
                "name": "fc1_tilelang",
                "target": "fc1",
                "engine": "tilelang",
                "patterns": ["mxfp4_packed_dequant_gemm_epilogue"],
                "min_speedup": 1.01,
                "tilelang": {
                    "target_arch": "sm_80",
                    "linear_runtime": "tilelang",
                    "linear_fastpath": "packed",
                },
            }
        ],
    )

    assert quant_stage.accepted is True
    target = operator_stage.metrics["targets"][0]
    assert target["engine"] == "tilelang"
    assert target["module_path"] == "fc1"
    assert target["metadata"]["execution_mode"] == "reference_fallback"
    assert target["metadata"]["kernel_pattern"] == "mxfp4_packed_dequant_gemm_epilogue"
    assert (
        target["metadata"]["weight_source"]
        == "mxfp_weight_only_linear_packed_bridge"
    )
    assert (
        target["metadata"]["weight_representation"]
        == "packed_mxfp4_plus_block_scale"
    )
    assert target["metadata"]["consumes_packed_weight"] is True
    assert target["metadata"]["unpack_stage"] == "eager_reference_fallback"


def test_materialize_tilelang_candidate_accepts_mxfp4_weight_only_linear_packed_path() -> None:
    torch.manual_seed(0)
    module = MXFPWeightOnlyLinear.from_linear(
        nn.Linear(64, 64),
        mx_precision=4,
        block_size=32,
    ).eval()

    candidate, compile_time_ms = materialize_operator_candidate_model(
        module,
        OperatorOptimizationTargetPlan(
            name="mxfp4_tilelang_packed",
            engine="tilelang",
            patterns=["mxfp4_packed_dequant_gemm_epilogue"],
            min_speedup=1.000001,
            tilelang={
                "target": "cuda",
                "target_arch": "sm_80",
                "linear_runtime": "tilelang",
                "linear_fastpath": "packed",
            },
        ),
    )

    inputs = torch.randn(8, 64, dtype=torch.float32)
    expected = module(inputs)
    actual = candidate(inputs)
    metadata = candidate.execution_metadata()

    assert compile_time_ms is None
    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)
    assert metadata["kernel_pattern"] == "mxfp4_packed_dequant_gemm_epilogue"
    assert metadata["weight_source"] == "mxfp_weight_only_linear_packed_bridge"
    assert metadata["weight_representation"] == "packed_mxfp4_plus_block_scale"
    assert metadata["consumes_packed_weight"] is True
