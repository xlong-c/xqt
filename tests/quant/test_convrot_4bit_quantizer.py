import pytest
import torch

from xqt.quant import (
    ConvRotMixedPrecisionLinear,
    build_regular_hadamard_matrix,
    quantize_with_convrot_4bit,
)
from xqt.runtime import apply_execution_policy
from xqt.workflows import XQTOptimizationSession


class _TinyLinearModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = torch.nn.Linear(16, 32, bias=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.fc(inputs)


def test_build_regular_hadamard_matrix_order_4_matches_convrot_base() -> None:
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


def test_build_regular_hadamard_matrix_order_16_is_orthogonal_up_to_scale() -> None:
    matrix = build_regular_hadamard_matrix(16)
    gram = matrix @ matrix.t()

    assert matrix.shape == (16, 16)
    assert torch.allclose(gram, 16.0 * torch.eye(16), atol=1e-5, rtol=1e-5)


def test_build_regular_hadamard_matrix_rejects_non_power_of_four() -> None:
    with pytest.raises(ValueError, match="power of four"):
        build_regular_hadamard_matrix(8)


def test_convrot_rotation_helper_rejects_non_power_of_four() -> None:
    from xqt.quant.quantizers.convrot_4bit import _apply_groupwise_rotation

    with pytest.raises(ValueError, match="power of four"):
        _apply_groupwise_rotation(torch.randn(2, 8), rot_size=8)


def test_convrot_4bit_pads_logical_features_and_preserves_batch_sequence_shape() -> None:
    source = torch.nn.Linear(18, 11, bias=True).eval()
    module = ConvRotMixedPrecisionLinear.from_linear(
        source,
        rot_size=16,
        group_size=16,
        compute_precision="bf16",
    ).eval()
    inputs = torch.randn(2, 3, 18)

    output = module(inputs)
    expected = source(inputs)

    assert module.input_features == 18
    assert module.padded_input_features == 32
    assert output.shape == (2, 3, 11)
    assert torch.allclose(output, expected, atol=2e-5, rtol=2e-5)


def test_convrot_4bit_keeps_requested_block_when_features_are_smaller() -> None:
    module = ConvRotMixedPrecisionLinear.from_linear(
        torch.nn.Linear(18, 11, bias=False).eval(),
        rot_size=256,
        group_size=128,
        compute_precision="bf16",
    )

    assert module.rot_size == 256
    assert module.padded_input_features == 256


def test_convrot_4bit_dynamic_activation_scale_is_per_token() -> None:
    source = torch.nn.Linear(18, 11, bias=False).eval()
    module = ConvRotMixedPrecisionLinear.from_linear(
        source,
        rot_size=16,
        group_size=16,
        activation_scale_mode="dynamic",
    ).eval()
    inputs = torch.randn(2, 3, 18)

    rotated = module._rotate_inputs(inputs)
    scale = module._activation_scale(rotated)

    assert scale.shape == (2, 3, 1)
    assert module(inputs).shape == (2, 3, 11)


def test_convrot_4bit_rejects_input_trailing_dimension_mismatch() -> None:
    module = ConvRotMixedPrecisionLinear.from_linear(
        torch.nn.Linear(18, 11, bias=False).eval(),
        rot_size=16,
        group_size=16,
    )

    with pytest.raises(ValueError, match="trailing dimension"):
        module(torch.randn(2, 17))


def test_convrot_4bit_accepts_tensor_channel_indices() -> None:
    module = ConvRotMixedPrecisionLinear.from_linear(
        torch.nn.Linear(16, 11, bias=False).eval(),
        rot_size=16,
        group_size=16,
        high_precision_channels=torch.tensor([1, 3]),
    )

    spec = module.channel_hybrid_spec()
    assert spec is not None
    assert spec.high_precision_channels == (1, 3)
    assert module.high_precision_channel_mask.tolist()[1:4] == [True, False, True]


def test_quantize_with_convrot_4bit_replaces_linear_on_cpu() -> None:
    model = _TinyLinearModel().eval()
    calibration_inputs = [torch.randn(4, 16)]

    result = quantize_with_convrot_4bit(
        model,
        policy={
            "dtype": "int4",
            "scheme": "convrot_w4a4",
            "include_module_names": ["fc"],
            "group_size": 8,
            "rot_size": 4,
            "mixed_precision_ratio": 1.0,
        },
        calibration_inputs=calibration_inputs,
        inplace=False,
    )
    output = result.model(torch.randn(4, 16))

    assert isinstance(result.model.fc, ConvRotMixedPrecisionLinear)
    assert result.quantized_modules == ["fc"]
    assert output.shape == (4, 32)
    assert result.metadata["algorithm_metadata"]["rotation_kind"] == "regular_hadamard"
    assert result.metadata["execution_policies"][0]["policy_kind"] == "mixed_precision"
    assert result.metadata["execution_policies"][0]["precision_overrides"][0]["module"] == "fc"


def test_quantize_with_convrot_4bit_reports_internal_feature_padding() -> None:
    model = torch.nn.Sequential(torch.nn.Linear(18, 11, bias=False)).eval()
    result = quantize_with_convrot_4bit(
        model,
        policy={
            "dtype": "int4",
            "scheme": "convrot_w4a4",
            "include_module_names": ["0"],
            "group_size": 16,
            "rot_size": 16,
        },
        inplace=False,
    )

    module = result.model[0]
    assert isinstance(module, ConvRotMixedPrecisionLinear)
    assert result.metadata["module_feature_shapes"]["0"] == {
        "logical_input_features": 18,
        "padded_input_features": 32,
        "rotation_size": 16,
        "group_size": 16,
    }


def test_materialize_convrot_execution_policy_switches_runtime_precision() -> None:
    model = _TinyLinearModel().eval()
    result = quantize_with_convrot_4bit(
        model,
        policy={
            "dtype": "int4",
            "scheme": "convrot_w4a4",
            "include_module_names": ["fc"],
            "group_size": 8,
            "rot_size": 4,
        },
        inplace=False,
    )

    candidate = apply_execution_policy(
        result.model,
        precision_overrides=[{"module": "fc", "precision": "bf16"}],
        default_precision="w4a4",
        inplace=False,
    )

    assert isinstance(candidate.fc, ConvRotMixedPrecisionLinear)
    assert candidate.fc.compute_precision == "bf16"
    assert result.model.fc.compute_precision == "w4a4"


def test_convrot_channel_hybrid_selects_input_outliers() -> None:
    model = _TinyLinearModel().eval()
    with torch.no_grad():
        model.fc.weight.zero_()
        model.fc.weight[:, 0] = 10.0
        model.fc.weight[:, 1] = 8.0
        model.fc.weight[:, 2:] = 0.1
    result = quantize_with_convrot_4bit(
        model,
        policy={
            "dtype": "int4",
            "scheme": "convrot_w4a4",
            "include_module_names": ["fc"],
            "group_size": 8,
            "rot_size": 4,
            "channel_hybrid_ratio": 0.125,
            "channel_hybrid_axis": "input",
        },
        calibration_inputs=[torch.randn(4, 16)],
        inplace=False,
    )
    module = result.model.fc
    assert module.channel_hybrid_enabled is True
    channels = module.channel_hybrid_spec()
    assert channels is not None
    assert len(channels.high_precision_channels) == 2
    output = module(torch.randn(2, 16))
    assert output.shape == (2, 32)
    policy = result.metadata["execution_policies"][0]
    assert policy["policy_kind"] == "channel_mixed_precision"
    assert policy["channel_overrides"][0]["module"] == "fc"


def test_convrot_channel_calibration_accepts_one_dimensional_linear_input() -> None:
    model = torch.nn.Sequential(torch.nn.Linear(16, 32, bias=False)).eval()
    result = quantize_with_convrot_4bit(
        model,
        policy={
            "dtype": "int4",
            "scheme": "convrot_w4a4",
            "include_module_names": ["0"],
            "group_size": 8,
            "rot_size": 4,
            "channel_hybrid_ratio": 0.125,
        },
        calibration_inputs=[torch.randn(16)],
        inplace=False,
    )

    module = result.model[0]
    spec = module.channel_hybrid_spec()
    assert spec is not None
    assert len(spec.high_precision_channels) == 2


def test_convrot_4bit_validates_rotation_without_candidates() -> None:
    model = torch.nn.Sequential(torch.nn.Linear(16, 32, bias=False)).eval()

    with pytest.raises(ValueError, match="power of four"):
        quantize_with_convrot_4bit(
            model,
            policy={
                "dtype": "int4",
                "scheme": "convrot_w4a4",
                "include_module_names": ["missing"],
                "rot_size": 8,
            },
            inplace=False,
        )


def test_convrot_w4a16_and_set_compute_precision_paths() -> None:
    model = _TinyLinearModel().eval()
    result = quantize_with_convrot_4bit(
        model,
        policy={
            "dtype": "int4",
            "scheme": "convrot_w4a4",
            "include_module_names": ["fc"],
            "group_size": 8,
            "rot_size": 4,
            "activation_scale_mode": "dynamic",
            "default_compute_precision": "w4a16",
        },
        inplace=False,
    )
    module = result.model.fc
    inputs = torch.randn(3, 16)

    assert isinstance(module, ConvRotMixedPrecisionLinear)
    assert module.compute_precision == "w4a16"
    assert module(inputs).shape == (3, 32)

    module.set_compute_precision("w4a4")
    assert module.compute_precision == "w4a4"
    assert module(inputs).shape == (3, 32)

    module.set_compute_precision("bf16")
    assert module(inputs).shape == (3, 32)
    assert "w4a16" in result.metadata["supported_compute_precisions"]


def test_session_quant_convrot_w4a4_replaces_linear(tmp_path) -> None:
    session = XQTOptimizationSession(
        project={
            "name": "session_convrot_w4a4",
            "artifact_dir": str(tmp_path / "artifacts"),
        },
        model=_TinyLinearModel().eval(),
        example_inputs=torch.randn(4, 16),
        calibration_inputs=[torch.randn(4, 16)],
    )

    stage = session.quant(
        name="convrot",
        backend="pytorch",
        method="convrot",
        strategy="w4a4_int4",
        compute="dequant_fp16",
        policy={
            "dtype": "int4",
            "scheme": "convrot_w4a4",
            "include_module_names": ["fc"],
            "group_size": 8,
            "rot_size": 4,
            "mixed_precision_ratio": 1.0,
        },
    )
    output = session.model(torch.randn(4, 16))

    assert stage.accepted is True
    assert isinstance(session.model.fc, ConvRotMixedPrecisionLinear)
    assert output.shape == (4, 32)
    assert stage.metrics["strategy"] == "w4a4_int4"
    assert stage.metrics["algorithm_executable"] is True
    assert stage.metrics["metadata"]["execution_state"] == "convrot_4bit"
    assert stage.metrics["metadata"]["recommended_high_precision_modules"] == ["fc"]
