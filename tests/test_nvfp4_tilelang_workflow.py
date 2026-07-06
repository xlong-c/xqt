from __future__ import annotations

import torch

from xqt import XQTOptimizationSession
from xqt.operator_opt import (
    OperatorOptimizationTargetPlan,
    materialize_operator_candidate_model,
    materialize_operator_candidate_models,
)
from xqt.operator_opt.executor import _effective_min_speedup, _native_runtime_near_equal
from xqt.quant import bridge_module_to_nvfp4_linear


class _FakeCompressedNVFP4Linear(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.in_features = 64
        self.out_features = 64
        self.register_buffer("qweight", torch.full((64, 32), 0x21, dtype=torch.uint8))
        self.register_buffer(
            "weight_scale",
            torch.ones((64, 4), dtype=torch.float32).to(torch.float8_e4m3fn),
        )
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


class _ExternalCompressedNVFP4Linear(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.in_features = 64
        self.out_features = 64
        self.register_parameter(
            "weight_packed",
            torch.nn.Parameter(torch.full((64, 32), 0x21, dtype=torch.uint8), requires_grad=False),
        )
        self.register_parameter(
            "weight_scale",
            torch.nn.Parameter(
                torch.ones((64, 4), dtype=torch.float32).to(torch.float8_e4m3fn),
                requires_grad=False,
            ),
        )
        self.register_parameter(
            "weight_global_scale",
            torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float32), requires_grad=False),
        )
        self.register_parameter(
            "bias",
            torch.nn.Parameter(torch.zeros(64, dtype=torch.float32), requires_grad=False),
        )


class _TinyNVFP4MLP(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = _FakeCompressedNVFP4Linear()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.fc1(inputs)


class _TinyExternalNVFP4Stack(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj1 = _ExternalCompressedNVFP4Linear()
        self.proj2 = _ExternalCompressedNVFP4Linear()


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
                "engine": "tilelang",
                "patterns": ["nvfp4_packed_dequant_gemm_epilogue"],
                "min_speedup": 1.01,
                "tilelang": {
                    "target_arch": "sm_89",
                },
            }
        ],
    )

    target = operator_stage.metrics["targets"][0]
    assert target["engine"] == "tilelang"
    assert target["module_path"] == "fc1"
    assert target["metadata"]["kernel_pattern"] == "nvfp4_packed_dequant_gemm_epilogue"
    assert target["metadata"]["weight_source"] == "compressed_tensors_nvfp4_packed_bridge"
    assert target["metadata"]["weight_representation"] == "packed_nvfp4_e2m1_plus_group_scale"
    assert target["metadata"]["consumes_packed_weight"] is True
    assert target["metadata"]["kernel_constraints"]["supports_packed_nvfp4_bridge"] is True
    assert target["metadata"]["execution_mode"] == "reference_fallback"


def test_nvfp4_tilelang_operator_stage_can_force_packed_metadata() -> None:
    torch.manual_seed(0)
    model = _TinyNVFP4MLP().eval()
    session = XQTOptimizationSession(
        project={
            "name": "nvfp4_tilelang_workflow_packed",
            "artifact_dir": "artifacts/xqt/tests/nvfp4_tilelang_workflow_packed",
        },
        model=model,
        example_inputs=torch.randn(64, 64, dtype=torch.float32),
    )

    operator_stage = session.operator(
        name="tilelang_nvfp4_fc1_packed",
        targets=[
            {
                "name": "fc1_tilelang",
                "target": "fc1",
                "engine": "tilelang",
                "patterns": ["nvfp4_packed_dequant_gemm_epilogue"],
                "min_speedup": 1.01,
                "tilelang": {
                    "target_arch": "sm_89",
                    "linear_fastpath": "packed",
                    "linear_runtime": "tilelang",
                },
            }
        ],
    )

    target = operator_stage.metrics["targets"][0]
    assert target["metadata"]["kernel_pattern"] == "nvfp4_packed_dequant_gemm_epilogue"
    assert target["metadata"]["weight_source"] == "compressed_tensors_nvfp4_packed_bridge"
    assert target["metadata"]["weight_representation"] == "packed_nvfp4_e2m1_plus_group_scale"
    assert target["metadata"]["consumes_packed_weight"] is True


def test_materialize_tilelang_candidate_accepts_external_nvfp4_linear_layout() -> None:
    torch.manual_seed(0)
    module = _ExternalCompressedNVFP4Linear().eval()
    bridge = bridge_module_to_nvfp4_linear(module)
    assert bridge is not None

    candidate, compile_time_ms = materialize_operator_candidate_model(
        module,
        OperatorOptimizationTargetPlan(
            name="external_nvfp4_tilelang",
            engine="tilelang",
            patterns=["dequant_gemm_epilogue"],
            min_speedup=1.000001,
            tilelang={
                "target": "cuda",
                "target_arch": "sm_89",
                "linear_runtime": "auto",
                "linear_fastpath": "auto",
            },
        ),
    )

    inputs = torch.randn(8, 64, dtype=torch.float32)
    expected = bridge(inputs)
    actual = candidate(inputs)
    metadata = candidate.execution_metadata()

    assert compile_time_ms is None
    assert torch.allclose(actual, expected)
    assert metadata["kernel_pattern"] == "dense_linear_epilogue"
    assert metadata["weight_source"] == "auto_inferred_nvfp4_dense_cache_bridge"
    assert metadata["weight_representation"] == "dense_dequantized_weight_cache"
    assert metadata["consumes_packed_weight"] is False
    assert metadata["unpack_stage"] == "one_time_eager_dequant_cache"


def test_materialize_tilelang_candidate_accepts_external_nvfp4_linear_layout_for_packed_path() -> None:
    torch.manual_seed(0)
    module = _ExternalCompressedNVFP4Linear().eval()

    candidate, compile_time_ms = materialize_operator_candidate_model(
        module,
        OperatorOptimizationTargetPlan(
            name="external_nvfp4_tilelang_packed",
            engine="tilelang",
            patterns=["nvfp4_packed_dequant_gemm_epilogue"],
            min_speedup=1.000001,
            tilelang={
                "target": "cuda",
                "target_arch": "sm_89",
                "linear_runtime": "tilelang",
                "linear_fastpath": "packed",
            },
        ),
    )

    inputs = torch.randn(8, 64, dtype=torch.float32)
    actual = candidate(inputs)
    metadata = candidate.execution_metadata()

    assert compile_time_ms is None
    assert actual.shape == (8, 64)
    assert metadata["kernel_pattern"] == "nvfp4_packed_dequant_gemm_epilogue"
    assert metadata["weight_source"] == "auto_inferred_nvfp4_packed_bridge"
    assert metadata["weight_representation"] == "packed_nvfp4_e2m1_plus_group_scale"
    assert metadata["consumes_packed_weight"] is True


def test_materialize_tilelang_candidate_models_replaces_multiple_external_nvfp4_layers() -> None:
    torch.manual_seed(0)
    model = _TinyExternalNVFP4Stack().eval()
    bridge1 = bridge_module_to_nvfp4_linear(model.proj1)
    bridge2 = bridge_module_to_nvfp4_linear(model.proj2)
    assert bridge1 is not None
    assert bridge2 is not None
    targets = [
        OperatorOptimizationTargetPlan(
            name="proj1_tilelang",
            engine="tilelang",
            target_path="proj1",
            patterns=["dequant_gemm_epilogue"],
            min_speedup=1.000001,
            tilelang={
                "target": "cuda",
                "target_arch": "sm_89",
                "linear_runtime": "auto",
                "linear_fastpath": "auto",
            },
        ),
        OperatorOptimizationTargetPlan(
            name="proj2_tilelang",
            engine="tilelang",
            target_path="proj2",
            patterns=["dequant_gemm_epilogue"],
            min_speedup=1.000001,
            tilelang={
                "target": "cuda",
                "target_arch": "sm_89",
                "linear_runtime": "auto",
                "linear_fastpath": "auto",
            },
        ),
    ]

    candidate = materialize_operator_candidate_models(model, targets, inplace=False)
    inputs = torch.randn(8, 64, dtype=torch.float32)
    expected = bridge2(bridge1(inputs))
    actual = candidate.proj2(candidate.proj1(inputs))

    assert candidate is not model
    assert type(candidate.proj1).__name__ == "_TileLangDequantGemmWrapper"
    assert type(candidate.proj2).__name__ == "_TileLangDequantGemmWrapper"
    assert torch.allclose(actual, expected)
    assert candidate.proj1.execution_metadata()["weight_source"] == "auto_inferred_nvfp4_dense_cache_bridge"
    assert candidate.proj2.execution_metadata()["weight_source"] == "auto_inferred_nvfp4_dense_cache_bridge"


def test_native_runtime_fastpath_allows_near_equal_speedup_threshold() -> None:
    target = type(
        "_Plan",
        (),
        {
            "min_speedup": 1.000001,
        },
    )()

    native_threshold = _effective_min_speedup(
        target,
        execution_detail={"kernel_kind": "native_runtime_fastpath"},
    )
    packed_threshold = _effective_min_speedup(
        target,
        execution_detail={"kernel_kind": "minimal_cuda_jit"},
    )

    assert native_threshold == 0.99
    assert packed_threshold == 1.000001


def test_native_runtime_fastpath_accepts_micro_latency_near_equal_gap() -> None:
    assert _native_runtime_near_equal(
        {"kernel_kind": "native_runtime_fastpath"},
        latency_before={"p50_ms": 0.06684450090688188},
        latency_after={"p50_ms": 0.06443049824156333},
    )
    assert not _native_runtime_near_equal(
        {"kernel_kind": "native_runtime_fastpath"},
        latency_before={"p50_ms": 0.06684450090688188},
        latency_after={"p50_ms": 0.05743049824156333},
    )
    assert _native_runtime_near_equal(
        {"kernel_kind": "native_runtime_fastpath"},
        latency_before={"p50_ms": 0.1480864993936848},
        latency_after={"p50_ms": 0.15003000044089276},
    )
