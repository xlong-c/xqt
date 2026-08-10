import torch
import pytest

from xqt.quant import (
    ConvRotInt8Linear,
    ConvRotNormInt8Linear,
    build_regular_hadamard_matrix,
    decode_comfy_quant_marker,
    encode_int8_tensorwise_marker,
    normalize_int8_tensorwise_marker,
    quantize_with_convrot_int8,
)
from xqt.workflows import XQTOptimizationSession


class _TinyLinearModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = torch.nn.Linear(16, 32, bias=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.fc(inputs)


def test_comfy_quant_marker_roundtrip_stock_shape() -> None:
    marker = encode_int8_tensorwise_marker(convrot=True, convrot_groupsize=256)
    payload = decode_comfy_quant_marker(marker)

    assert payload["format"] == "int8_tensorwise"
    assert payload["convrot"] is True
    assert payload["convrot_groupsize"] == 256


def test_normalize_legacy_int8_fast_marker() -> None:
    legacy = {"convrot": True, "convrot_groupsize": 64, "per_row": True}
    normalized = normalize_int8_tensorwise_marker(legacy)

    assert normalized["format"] == "int8_tensorwise"
    assert normalized["convrot"] is True
    assert normalized["convrot_groupsize"] == 64
    assert "per_row" not in normalized


def test_quantize_with_convrot_int8_replaces_linear_on_cpu() -> None:
    model = _TinyLinearModel().eval()
    result = quantize_with_convrot_int8(
        model,
        policy={
            "dtype": "int8",
            "scheme": "convrot_w8a8",
            "include_module_names": ["fc"],
            "rot_size": 4,
        },
        inplace=False,
        engine="torch_int_mm",
        fallback_engine="torch_int_mm",
    )
    output = result.model(torch.randn(4, 16))

    assert isinstance(result.model.fc, ConvRotInt8Linear)
    assert result.quantized_modules == ["fc"]
    assert result.strategy == "w8a8_int8"
    assert result.method == "convrot"
    assert output.shape == (4, 32)
    assert result.metadata["algorithm_metadata"]["rotation_kind"] == "regular_hadamard"
    marker = decode_comfy_quant_marker(result.model.fc.comfy_quant)
    assert marker["format"] == "int8_tensorwise"
    assert marker["convrot"] is True
    assert marker["convrot_groupsize"] == 4


@pytest.mark.parametrize("norm_kind", ["rmsnorm", "layernorm"])
def test_convrot_norm_fusion_is_cpu_equivalent_and_explicit(
    norm_kind: str,
) -> None:
    torch.manual_seed(7)
    norm = (
        torch.nn.RMSNorm(16, eps=1e-5)
        if norm_kind == "rmsnorm"
        else torch.nn.LayerNorm(16, eps=1e-5)
    )
    linear = torch.nn.Linear(16, 32, bias=True).eval()
    model = torch.nn.Sequential(norm, linear).eval()
    result = quantize_with_convrot_int8(
        model,
        policy={
            "dtype": "int8",
            "scheme": "convrot_w8a8",
            "include_module_names": ["1"],
            "rot_size": 4,
            "activation_scale_mode": "static",
            "activation_scales": {"1": 0.05},
            "fuse_norm": True,
        },
        inplace=False,
        engine="torch_int_mm",
        fallback_engine="torch_int_mm",
    )
    inputs = torch.randn(3, 16)
    expected_linear = ConvRotInt8Linear.from_linear(
        linear,
        rot_size=4,
        engine="torch_int_mm",
        fallback_engine="torch_int_mm",
        activation_scale_mode="static",
        activation_scale=0.05,
    )
    expected = expected_linear(norm(inputs))
    actual = result.model(inputs)

    assert isinstance(result.model[0], ConvRotNormInt8Linear)
    assert isinstance(result.model[1], torch.nn.Identity)
    assert result.metadata["norm_fused_module_count"] == 1
    assert result.metadata["norm_fused_modules"] == {"1": "0"}
    assert result.model[0].execution_metadata()["norm_fused"] is False
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_convrot_int8_hadamard_matches_w4a4_base() -> None:
    matrix = build_regular_hadamard_matrix(4)
    expected = torch.tensor(
        [
            [1.0, 1.0, 1.0, -1.0],
            [1.0, 1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0, 1.0],
        ]
    )
    assert torch.equal(matrix, expected)


def test_convrot_int8_preserves_source_output_dtype() -> None:
    source = torch.nn.Linear(16, 32, dtype=torch.float16).eval()
    module = ConvRotInt8Linear.from_linear(
        source,
        rot_size=4,
        engine="torch_int_mm",
    )

    output = module(torch.randn(2, 16, dtype=torch.float16))

    assert output.dtype == torch.float16


def test_convrot_int8_pads_to_mma_alignment_and_preserves_sequence_shape() -> None:
    source = torch.nn.Linear(18, 11, bias=True).eval()
    module = ConvRotInt8Linear.from_linear(
        source,
        rot_size=16,
        engine="torch_int_mm",
        fallback_engine="torch_int_mm",
    ).eval()
    inputs = torch.randn(2, 3, 18)

    output = module(inputs)
    metadata = module.execution_metadata()

    assert output.shape == (2, 3, 11)
    assert metadata["logical_input_features"] == 18
    assert metadata["padded_input_features"] == 64


def test_convrot_int8_preserves_pre_rotated_input_marker() -> None:
    source = torch.nn.Linear(16, 11, bias=False).eval()
    source.input_already_rotated = True
    module = ConvRotInt8Linear.from_linear(
        source,
        rot_size=16,
        engine="torch_int_mm",
        fallback_engine="torch_int_mm",
    ).eval()
    inputs = torch.randn(2, 16)

    rotated = module._rotate_inputs(inputs)

    assert module.input_already_rotated is True
    assert torch.equal(rotated[..., :16], inputs)
    assert torch.count_nonzero(rotated[..., 16:]) == 0
    assert module.execution_metadata()["input_already_rotated"] is True


def test_convrot_int8_rejects_non_power_of_four_rotation_size() -> None:
    with pytest.raises(ValueError, match="power of four"):
        ConvRotInt8Linear.from_linear(
            torch.nn.Linear(18, 11),
            rot_size=8,
            engine="torch_int_mm",
        )


def test_convrot_int8_validates_rotation_without_candidates() -> None:
    model = torch.nn.Sequential(torch.nn.Linear(16, 32, bias=False)).eval()

    with pytest.raises(ValueError, match="power of four"):
        quantize_with_convrot_int8(
            model,
            policy={
                "dtype": "int8",
                "scheme": "convrot_w8a8",
                "include_module_names": ["missing"],
                "rot_size": 8,
            },
            inplace=False,
            engine="torch_int_mm",
        )


def test_quantize_with_convrot_int8_reports_internal_feature_padding() -> None:
    model = torch.nn.Sequential(torch.nn.Linear(18, 11, bias=False)).eval()
    result = quantize_with_convrot_int8(
        model,
        policy={
            "dtype": "int8",
            "scheme": "convrot_w8a8",
            "include_module_names": ["0"],
            "rot_size": 16,
        },
        inplace=False,
        engine="torch_int_mm",
        fallback_engine="torch_int_mm",
    )

    assert result.metadata["module_feature_shapes"]["0"] == {
        "logical_input_features": 18,
        "padded_input_features": 64,
        "rotation_size": 16,
    }


def test_convrot_int8_calibrates_static_activation_scale() -> None:
    model = _TinyLinearModel().eval()
    calibration_inputs = [torch.randn(4, 16), torch.randn(3, 16)]

    result = quantize_with_convrot_int8(
        model,
        policy={
            "dtype": "int8",
            "scheme": "convrot_w8a8",
            "include_module_names": ["fc"],
            "rot_size": 4,
            "activation_scale_mode": "static",
        },
        calibration_inputs=calibration_inputs,
        inplace=False,
        engine="torch_int_mm",
    )
    output = result.model(torch.randn(2, 16))

    assert output.shape == (2, 32)
    assert result.model.fc.int8_compute.activation_scale_mode == "static"
    assert result.model.fc.int8_compute.static_activation_scale.item() > 0.0
    assert result.metadata["static_scale_module_count"] == 1
    assert result.metadata["calibrated_static_scale_module_count"] == 1
    assert result.metadata["dynamic_fallback_module_count"] == 0


def test_convrot_int8_static_mode_falls_back_without_scale() -> None:
    model = _TinyLinearModel().eval()

    result = quantize_with_convrot_int8(
        model,
        policy={
            "dtype": "int8",
            "scheme": "convrot_w8a8",
            "include_module_names": ["fc"],
            "rot_size": 4,
            "activation_scale_mode": "static",
        },
        inplace=False,
        engine="torch_int_mm",
    )
    output = result.model(torch.randn(2, 16))

    assert output.shape == (2, 32)
    assert result.model.fc.int8_compute.activation_scale_mode == "dynamic"
    assert result.metadata["static_scale_module_count"] == 0
    assert result.metadata["dynamic_fallback_module_count"] == 1


def test_convrot_int8_defaults_to_static_with_calibration_inputs() -> None:
    model = _TinyLinearModel().eval()

    result = quantize_with_convrot_int8(
        model,
        policy={
            "dtype": "int8",
            "scheme": "convrot_w8a8",
            "include_module_names": ["fc"],
            "rot_size": 4,
        },
        calibration_inputs=[torch.randn(4, 16)],
        inplace=False,
        engine="torch_int_mm",
    )

    assert result.model.fc.int8_compute.activation_scale_mode == "static"
    assert result.metadata["activation_scale_mode"] == "static"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_convrot_int8_auto_static_uses_cuda_sm89_gemm_after_fused_rotation() -> None:
    from xqt.operator_opt.kernels.cute.int8mma_binding import int8mma_available

    if torch.cuda.get_device_capability() != (8, 9) or not int8mma_available():
        pytest.skip("cuda_sm89 CUTLASS extension unavailable")

    source = torch.nn.Linear(256, 256, bias=True, dtype=torch.float16, device="cuda").eval()
    module = ConvRotInt8Linear.from_linear(
        source,
        rot_size=64,
        engine="auto",
        activation_scale_mode="static",
        activation_scale=0.02,
    ).eval()

    output = module(torch.randn(64, 256, dtype=torch.float16, device="cuda"))
    torch.cuda.synchronize()
    metadata = module.execution_metadata()

    assert output.shape == (64, 256)
    assert metadata["engine"] == "cuda_sm89"
    assert metadata["activation_quant_engine"] == "tilelang_hadamard_static"
    assert metadata["fused_static_status"] == "tilelang_rotation_quant_then_cuda_gemm"
    assert metadata["rotation_fused"] is True
    assert metadata["prepacked_b"] is True


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("rot_size", [4, 16])
def test_convrot_norm_fusion_uses_fused_triton_input_path(rot_size: int) -> None:
    model = torch.nn.Sequential(
        torch.nn.RMSNorm(64, eps=1e-5, device="cuda", dtype=torch.float16),
        torch.nn.Linear(64, 128, device="cuda", dtype=torch.float16),
    ).eval()
    result = quantize_with_convrot_int8(
        model,
        policy={
            "dtype": "int8",
            "scheme": "convrot_w8a8",
            "include_module_names": ["1"],
            "rot_size": rot_size,
            "activation_scale_mode": "static",
            "activation_scales": {"1": 0.05},
            "fuse_norm": True,
        },
        inplace=False,
        engine="triton",
        fallback_engine="torch_int_mm",
    )
    output = result.model(torch.randn(32, 64, device="cuda", dtype=torch.float16))
    torch.cuda.synchronize()
    metadata = result.model[0].execution_metadata()

    assert output.shape == (32, 128)
    assert metadata["norm_fused"] is True
    assert metadata["activation_quant_engine"] == "triton_norm_hadamard_static"
    assert metadata["fused_static_status"] == "norm_hadamard_quant_then_gemm"
    assert metadata["fused_norm_fallback_reason"] is None


def test_session_quant_convrot_w8a8_replaces_linear(tmp_path) -> None:
    session = XQTOptimizationSession(
        project={
            "name": "session_convrot_w8a8",
            "artifact_dir": str(tmp_path / "artifacts"),
        },
        model=_TinyLinearModel().eval(),
        example_inputs=torch.randn(4, 16),
        calibration_inputs=[torch.randn(4, 16)],
    )

    stage = session.quant(
        name="convrot_w8a8",
        backend="pytorch",
        method="convrot",
        strategy="w8a8_int8",
        compute="w8a8_int8_mma",
        policy={
            "dtype": "int8",
            "scheme": "convrot_w8a8",
            "include_module_names": ["fc"],
            "rot_size": 4,
            "engine": "torch_int_mm",
        },
    )
    output = session.model(torch.randn(4, 16))

    assert stage.accepted is True
    assert isinstance(session.model.fc, ConvRotInt8Linear)
    assert output.shape == (4, 32)
    assert stage.metrics["strategy"] == "w8a8_int8"
    assert stage.metrics["algorithm_executable"] is True
    assert stage.metrics["metadata"]["execution_state"] == "convrot_int8"
