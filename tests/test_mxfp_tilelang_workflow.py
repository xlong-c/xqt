from __future__ import annotations

import torch

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
