from __future__ import annotations

import torch

from xqt import XQTOptimizationSession
from xqt.quant import bridge_module_to_nvfp4_linear


class _FakeCompressedNVFP4Linear(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.in_features = 64
        self.out_features = 64
        self.register_buffer("qweight", torch.full((64, 32), 0x21, dtype=torch.uint8))
        self.register_buffer("weight_scale", torch.ones((64, 1, 1), dtype=torch.float32))
        self.register_buffer("weight_global_scale", torch.tensor([1.0], dtype=torch.float32))
        self.register_buffer("bias", torch.zeros(64, dtype=torch.float32))
        self.bridge = bridge_module_to_nvfp4_linear(self)
        assert self.bridge is not None

    def tilelang_packed_nvfp4_dequant_gemm_args(
        self,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, None, int, int, torch.Tensor | None]:
        return self.bridge.tilelang_packed_nvfp4_dequant_gemm_args(dtype=dtype, device=device)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        assert self.bridge is not None
        return self.bridge(inputs)


class _TinyNVFP4MLP(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = _FakeCompressedNVFP4Linear()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.fc1(inputs)


def test_nvfp4_tilelang_operator_stage_uses_nvfp4_metadata() -> None:
    torch.manual_seed(0)
    model = _TinyNVFP4MLP().eval()
    session = XQTOptimizationSession(
        project={
            "name": "nvfp4_tilelang_workflow",
            "artifact_dir": "artifacts/xqt/tests/nvfp4_tilelang_workflow",
        },
        model=model,
        example_inputs=torch.randn(64, 64, dtype=torch.float32),
    )

    operator_stage = session.operator(
        name="tilelang_nvfp4_fc1",
        targets=[
            {
                "name": "fc1_tilelang",
                "target": "fc1",
                "backend": "tilelang",
                "patterns": ["nvfp4_packed_dequant_gemm_epilogue"],
                "min_speedup": 1.01,
                "tilelang": {
                    "target_arch": "sm_89",
                },
            }
        ],
    )

    target = operator_stage.metrics["targets"][0]
    assert target["backend"] == "tilelang"
    assert target["module_path"] == "fc1"
    assert target["metadata"]["kernel_pattern"] == "nvfp4_packed_dequant_gemm_epilogue"
    assert target["metadata"]["weight_source"] == "compressed_tensors_nvfp4_packed_bridge"
    assert target["metadata"]["weight_representation"] == "packed_nvfp4_e2m1_plus_group_scale"
    assert target["metadata"]["consumes_packed_weight"] is True
    assert target["metadata"]["kernel_constraints"]["supports_packed_nvfp4_bridge"] is True
    assert target["metadata"]["execution_mode"] == "reference_fallback"
