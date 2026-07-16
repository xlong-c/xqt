from __future__ import annotations

import importlib.util

import pytest
import torch
from torch import nn

import xqt.operator_opt.kernels.triton.gemm as triton_gemm_module
import xqt.operator_opt.kernels.triton.mxfp_gemm as triton_mxfp_module
from xqt import XQTOptimizationSession
from xqt.operator_opt import (
    OperatorOptimizationTargetPlan,
    materialize_operator_candidate_model,
)
from xqt.operator_opt.kernels.triton.gemm import (
    dequantize_int4_weight_triton,
    dequantize_nvfp4_weight_triton,
    gemm_int4_dequant_reference,
    gemm_int4_dequant_triton,
    gemm_nvfp4_packed_activation_reference,
    gemm_nvfp4_packed_activation_triton,
    gemm_nvfp4_packed_dequant_reference,
    gemm_nvfp4_packed_dequant_triton,
)
from xqt.operator_opt.kernels.triton.fp4_quant import (
    scaled_mxfp4_quant_reference,
    scaled_nvfp4_quant_reference,
)
from xqt.operator_opt.kernels.triton.mxfp_gemm import (
    dequantize_mxfp_weight_triton,
    gemm_mxfp_packed_activation_reference,
    gemm_mxfp_packed_activation_triton,
    gemm_mxfp_reference,
    gemm_mxfp_triton,
    unpack_mxfp,
)
from xqt.quant import FP4WeightOnlyLinear, MXFPWeightOnlyLinear, bridge_module_to_nvfp4_linear


requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for Triton low-bit dequant CUDA tests",
)

requires_triton = pytest.mark.skipif(
    importlib.util.find_spec("triton") is None,
    reason="triton package is required for Triton low-bit dequant CUDA tests",
)


class _ExternalCompressedNVFP4Linear(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.in_features = 64
        self.out_features = 64
        self.register_parameter(
            "weight_packed",
            nn.Parameter(torch.full((64, 32), 0x21, dtype=torch.uint8), requires_grad=False),
        )
        self.register_parameter(
            "weight_scale",
            nn.Parameter(
                torch.ones((64, 4), dtype=torch.float32).to(torch.float8_e4m3fn),
                requires_grad=False,
            ),
        )
        self.register_parameter(
            "weight_global_scale",
            nn.Parameter(torch.tensor([1.0], dtype=torch.float32), requires_grad=False),
        )
        self.register_parameter(
            "bias",
            nn.Parameter(torch.zeros(64, dtype=torch.float32), requires_grad=False),
        )
        self.bridge = bridge_module_to_nvfp4_linear(self)
        assert self.bridge is not None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        assert self.bridge is not None
        return self.bridge(inputs)


class _TinyNVFP4MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = _ExternalCompressedNVFP4Linear()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.fc1(inputs)


class _TinyFP4MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = FP4WeightOnlyLinear.from_linear(nn.Linear(64, 64), group_size=16)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.fc1(inputs)


class _TinyMXFPMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = MXFPWeightOnlyLinear.from_linear(nn.Linear(64, 64), mx_precision=4, block_size=32)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.fc1(inputs)


class _TinyDenseMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(64, 64)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.fc1(inputs)


def test_materialize_triton_candidate_accepts_external_nvfp4_linear_layout() -> None:
    torch.manual_seed(0)
    module = _ExternalCompressedNVFP4Linear().eval()
    bridge = bridge_module_to_nvfp4_linear(module)
    assert bridge is not None

    candidate, compile_time_ms = materialize_operator_candidate_model(
        module,
        OperatorOptimizationTargetPlan(
            name="external_nvfp4_triton",
            engine="triton",
            patterns=["gemm_nvfp4_packed_dequant"],
            min_speedup=1.000001,
        ),
    )

    inputs = torch.randn(8, 64, dtype=torch.float32)
    expected = bridge(inputs)
    actual = candidate(inputs)
    metadata = candidate.execution_metadata()

    assert compile_time_ms is None
    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)
    assert metadata["kernel_pattern"] == "gemm_nvfp4_packed_dequant"
    assert metadata["weight_source"] == "auto_inferred_nvfp4_packed_bridge"
    assert metadata["weight_representation"] == "packed_nvfp4_e2m1_plus_group_scale"
    assert metadata["consumes_packed_weight"] is True
    assert metadata["execution_mode"] == "reference_fallback"


def test_materialize_triton_candidate_accepts_fp4_weight_only_linear() -> None:
    torch.manual_seed(0)
    module = FP4WeightOnlyLinear.from_linear(nn.Linear(64, 64), group_size=16).eval()

    candidate, compile_time_ms = materialize_operator_candidate_model(
        module,
        OperatorOptimizationTargetPlan(
            name="fp4_triton",
            engine="triton",
            patterns=["gemm_int4_dequant"],
            min_speedup=1.000001,
        ),
    )

    inputs = torch.randn(8, 64, dtype=torch.float32)
    expected = module(inputs)
    actual = candidate(inputs)
    metadata = candidate.execution_metadata()

    assert compile_time_ms is None
    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)
    assert metadata["kernel_pattern"] == "gemm_int4_dequant"
    assert metadata["weight_source"] == "fp4_weight_only_linear_packed_bridge"
    assert metadata["weight_representation"] == "packed_signed_int4_plus_group_scale"
    assert metadata["consumes_packed_weight"] is True
    assert metadata["execution_mode"] == "reference_fallback"


def test_materialize_triton_candidate_accepts_mxfp_weight_only_linear() -> None:
    torch.manual_seed(0)
    module = MXFPWeightOnlyLinear.from_linear(
        nn.Linear(64, 64),
        mx_precision=4,
        block_size=32,
    ).eval()

    candidate, compile_time_ms = materialize_operator_candidate_model(
        module,
        OperatorOptimizationTargetPlan(
            name="mxfp4_triton",
            engine="triton",
            patterns=["gemm_mxfp4"],
            min_speedup=1.000001,
        ),
    )

    inputs = torch.randn(8, 64, dtype=torch.float32)
    expected = module(inputs)
    actual = candidate(inputs)
    metadata = candidate.execution_metadata()

    assert compile_time_ms is None
    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)
    assert metadata["kernel_pattern"] == "gemm_mxfp4"
    assert metadata["weight_source"] == "mxfp_weight_only_linear_packed_bridge"
    assert metadata["weight_representation"] == "packed_mxfp4_plus_block_scale"
    assert metadata["consumes_packed_weight"] is True
    assert metadata["execution_mode"] == "reference_fallback"


def test_triton_nvfp4_operator_stage_reports_packed_metadata_on_cpu() -> None:
    torch.manual_seed(0)
    model = _TinyNVFP4MLP().eval()
    session = XQTOptimizationSession(
        project={
            "name": "nvfp4_triton_workflow",
            "artifact_dir": "artifacts/xqt/tests/nvfp4_triton_workflow",
        },
        model=model,
        example_inputs=torch.randn(64, 64, dtype=torch.float32),
    )

    operator_stage = session.operator(
        name="triton_nvfp4_fc1",
        targets=[
            {
                "name": "fc1_triton",
                "target": "fc1",
                "engine": "triton",
                "patterns": ["gemm_nvfp4_packed_dequant"],
                "min_speedup": 1.000001,
            }
        ],
    )

    target = operator_stage.metrics["targets"][0]
    assert target["engine"] == "triton"
    assert target["metadata"]["execution_mode"] == "reference_fallback"
    assert target["metadata"]["kernel_pattern"] == "gemm_nvfp4_packed_dequant"
    assert target["metadata"]["weight_source"] == "auto_inferred_nvfp4_packed_bridge"


def test_triton_fp4_operator_stage_reports_packed_metadata_on_cpu() -> None:
    torch.manual_seed(0)
    model = _TinyFP4MLP().eval()
    session = XQTOptimizationSession(
        project={
            "name": "fp4_triton_workflow",
            "artifact_dir": "artifacts/xqt/tests/fp4_triton_workflow",
        },
        model=model,
        example_inputs=torch.randn(64, 64, dtype=torch.float32),
    )

    operator_stage = session.operator(
        name="triton_fp4_fc1",
        targets=[
            {
                "name": "fc1_triton",
                "target": "fc1",
                "engine": "triton",
                "patterns": ["gemm_int4_dequant"],
                "min_speedup": 1.000001,
            }
        ],
    )

    target = operator_stage.metrics["targets"][0]
    assert target["engine"] == "triton"
    assert target["metadata"]["execution_mode"] == "reference_fallback"
    assert target["metadata"]["kernel_pattern"] == "gemm_int4_dequant"
    assert target["metadata"]["weight_source"] == "fp4_weight_only_linear_packed_bridge"


def test_triton_mxfp_operator_stage_reports_packed_metadata_on_cpu() -> None:
    torch.manual_seed(0)
    model = _TinyDenseMLP().eval()
    session = XQTOptimizationSession(
        project={
            "name": "mxfp4_triton_workflow",
            "artifact_dir": "artifacts/xqt/tests/mxfp4_triton_workflow",
        },
        model=model,
        example_inputs=torch.randn(64, 64, dtype=torch.float32),
    )

    quant_stage = session.quant(
        name="mxfp_quant",
        backend="pytorch",
        method="awq",
        strategy="w4a16_mxfp4",
        compute="dequant_fp16",
        policy={
            "dtype": "mxfp",
            "scheme": "weight_only",
            "precision": 4,
            "block_size": 32,
            "include_module_names": ["fc1"],
        },
    )
    assert quant_stage.accepted is True

    operator_stage = session.operator(
        name="triton_mxfp_fc1",
        from_stage="mxfp_quant",
        targets=[
            {
                "name": "fc1_triton",
                "target": "fc1",
                "engine": "triton",
                "patterns": ["gemm_mxfp4"],
                "min_speedup": 1.000001,
            }
        ],
    )

    target = operator_stage.metrics["targets"][0]
    assert target["engine"] == "triton"
    assert target["metadata"]["execution_mode"] == "reference_fallback"
    assert target["metadata"]["kernel_pattern"] == "gemm_mxfp4"
    assert target["metadata"]["weight_source"] == "mxfp_weight_only_linear_packed_bridge"


@requires_cuda
@requires_triton
def test_triton_int4_weight_decode_matches_reference_cuda() -> None:
    torch.manual_seed(0)
    module = FP4WeightOnlyLinear.from_linear(nn.Linear(64, 64), group_size=16).to("cuda").eval()
    decoded = dequantize_int4_weight_triton(
        module.packed_weight,
        module.weight_scale,
        group_size=module.group_size,
        cols=module.input_features,
        output_dtype=torch.float16,
    )
    expected = module.dequantize_weight().to(device="cuda", dtype=torch.float16)
    assert torch.allclose(decoded, expected, atol=1e-3, rtol=1e-3)


@requires_cuda
@requires_triton
def test_triton_mxfp_weight_decode_matches_reference_cuda() -> None:
    torch.manual_seed(0)
    module = MXFPWeightOnlyLinear.from_linear(
        nn.Linear(64, 64),
        mx_precision=4,
        block_size=32,
    ).to("cuda").eval()
    decoded = dequantize_mxfp_weight_triton(
        module.packed_weight,
        module.weight_scale,
        precision=module.mx_precision,
        block_size=module.block_size,
        cols=module.input_features,
        output_dtype=torch.float16,
    )
    expected = (
        unpack_mxfp(
            module.packed_weight,
            module.weight_scale,
            module.mx_precision,
            module.block_size,
            module.input_features,
        )
        .reshape(module.output_features, module.input_features)
        .to(device="cuda", dtype=torch.float16)
    )
    assert torch.allclose(decoded, expected, atol=1e-3, rtol=1e-3)


@requires_cuda
@requires_triton
def test_triton_nvfp4_weight_decode_matches_reference_cuda() -> None:
    torch.manual_seed(0)
    module = _ExternalCompressedNVFP4Linear().to("cuda").eval()
    bridge = bridge_module_to_nvfp4_linear(module)
    assert bridge is not None
    decoded = dequantize_nvfp4_weight_triton(
        module.weight_packed.detach(),
        module.weight_scale.detach(),
        cols=module.in_features,
        group_size=16,
        weight_global_scale=module.weight_global_scale.detach(),
        output_dtype=torch.float16,
    )
    expected = bridge.dequantize_weight().to(device="cuda", dtype=torch.float16)
    assert torch.allclose(decoded, expected, atol=1e-3, rtol=1e-3)


@requires_cuda
@requires_triton
def test_triton_int4_gemm_matches_reference_cuda() -> None:
    torch.manual_seed(0)
    module = FP4WeightOnlyLinear.from_linear(nn.Linear(64, 64), group_size=16).to("cuda").eval()
    x = torch.randn(8, 64, device="cuda", dtype=torch.float16)
    actual = gemm_int4_dequant_triton(
        x,
        module.packed_weight,
        module.weight_scale,
        None,
        None if module.bias is None else module.bias.to(device="cuda", dtype=torch.float16),
        group_size=module.group_size,
        activation=None,
    )
    expected = gemm_int4_dequant_reference(
        x,
        module.packed_weight,
        module.weight_scale,
        None,
        None if module.bias is None else module.bias.to(device="cuda", dtype=torch.float16),
        group_size=module.group_size,
        activation=None,
    )
    assert torch.allclose(actual, expected, atol=2e-2, rtol=2e-2)


@requires_cuda
@requires_triton
def test_triton_mxfp_gemm_matches_reference_cuda() -> None:
    torch.manual_seed(0)
    module = MXFPWeightOnlyLinear.from_linear(
        nn.Linear(64, 64),
        mx_precision=4,
        block_size=32,
    ).to("cuda").eval()
    x = torch.randn(8, 64, device="cuda", dtype=torch.float16)
    actual = gemm_mxfp_triton(
        x,
        module.packed_weight,
        module.weight_scale,
        None if module.bias is None else module.bias.to(device="cuda", dtype=torch.float16),
        mx_precision=module.mx_precision,
        block_size=module.block_size,
        activation=None,
        transpose_b=True,
    )
    expected = gemm_mxfp_reference(
        x,
        module.packed_weight,
        module.weight_scale,
        None if module.bias is None else module.bias.to(device="cuda", dtype=torch.float16),
        mx_precision=module.mx_precision,
        block_size=module.block_size,
        activation=None,
        transpose_b=True,
    )
    assert torch.allclose(actual, expected, atol=2e-2, rtol=2e-2)


@requires_cuda
@requires_triton
def test_triton_nvfp4_gemm_matches_reference_cuda() -> None:
    torch.manual_seed(0)
    module = _ExternalCompressedNVFP4Linear().to("cuda").eval()
    x = torch.randn(8, 64, device="cuda", dtype=torch.float16)
    actual = gemm_nvfp4_packed_dequant_triton(
        x,
        module.weight_packed.detach(),
        module.weight_scale.detach().to(torch.float16),
        module.bias.detach().to(torch.float16),
        input_features=module.in_features,
        group_size=16,
        weight_global_scale=module.weight_global_scale.detach(),
        activation=None,
    )
    expected = gemm_nvfp4_packed_dequant_reference(
        x,
        module.weight_packed.detach(),
        module.weight_scale.detach().to(torch.float16),
        module.bias.detach().to(torch.float16),
        input_features=module.in_features,
        group_size=16,
        weight_global_scale=module.weight_global_scale.detach(),
        activation=None,
    )
    assert torch.allclose(actual, expected, atol=2e-2, rtol=2e-2)


@requires_cuda
@requires_triton
def test_triton_nvfp4_packed_activation_gemm_matches_reference_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(0)
    module = _ExternalCompressedNVFP4Linear().to("cuda").eval()
    x = torch.randn(8, 64, device="cuda", dtype=torch.float16)
    activation_global_scale = (
        2688.0 / x.detach().to(torch.float32).abs().amax().clamp_min(1e-8)
    ).reshape(1).to(device="cuda", dtype=torch.float32)
    packed_activation, activation_scale = scaled_nvfp4_quant_reference(
        x,
        activation_global_scale,
        group_size=16,
    )
    expected = gemm_nvfp4_packed_activation_reference(
        packed_activation,
        activation_scale,
        module.weight_packed.detach(),
        module.weight_scale.detach().to(torch.float16),
        module.bias.detach().to(torch.float16),
        input_features=module.in_features,
        group_size=16,
        activation_global_scale=activation_global_scale,
        weight_global_scale=module.weight_global_scale.detach(),
        activation=None,
        output_dtype=torch.float16,
    )

    def _forbidden_dequant(*args: object, **kwargs: object) -> torch.Tensor:
        raise AssertionError("common dequantize_nvfp4_codes should not be used in Triton packed-activation fastpath")

    monkeypatch.setattr(triton_gemm_module, "dequantize_nvfp4_codes", _forbidden_dequant)
    actual = gemm_nvfp4_packed_activation_triton(
        packed_activation,
        activation_scale,
        module.weight_packed.detach(),
        module.weight_scale.detach().to(torch.float16),
        module.bias.detach().to(torch.float16),
        input_features=module.in_features,
        group_size=16,
        activation_global_scale=activation_global_scale,
        weight_global_scale=module.weight_global_scale.detach(),
        activation=None,
        output_dtype=torch.float16,
    )

    assert torch.allclose(actual, expected, atol=2e-2, rtol=2e-2)


@requires_cuda
@requires_triton
def test_triton_mxfp4_packed_activation_gemm_matches_reference_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(0)
    module = MXFPWeightOnlyLinear.from_linear(
        nn.Linear(64, 64),
        mx_precision=4,
        block_size=32,
    ).to("cuda").eval()
    x = torch.randn(8, 64, device="cuda", dtype=torch.float16)
    packed_activation, activation_scale = scaled_mxfp4_quant_reference(
        x,
        group_size=32,
    )
    expected = gemm_mxfp_packed_activation_reference(
        packed_activation,
        activation_scale,
        module.packed_weight,
        module.weight_scale,
        None if module.bias is None else module.bias.to(device="cuda", dtype=torch.float16),
        input_features=module.input_features,
        block_size=module.block_size,
        mx_precision=module.mx_precision,
        activation=None,
        output_dtype=torch.float16,
    )

    def _forbidden_dequant(*args: object, **kwargs: object) -> torch.Tensor:
        raise AssertionError("common dequantize_nvfp4_codes should not be used in Triton packed-activation fastpath")

    monkeypatch.setattr(triton_mxfp_module, "dequantize_nvfp4_codes", _forbidden_dequant)
    actual = gemm_mxfp_packed_activation_triton(
        packed_activation,
        activation_scale,
        module.packed_weight,
        module.weight_scale,
        None if module.bias is None else module.bias.to(device="cuda", dtype=torch.float16),
        input_features=module.input_features,
        block_size=module.block_size,
        mx_precision=module.mx_precision,
        activation=None,
        output_dtype=torch.float16,
    )

    assert torch.allclose(actual, expected, atol=2e-2, rtol=2e-2)
