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
    assert "requires CUDA tensors" in str(target["metadata"]["execution_reason"])
