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
        strategy="convrot_w4a4",
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
    assert stage.metrics["strategy"] == "convrot_w4a4"
    assert stage.metrics["algorithm_executable"] is True
    assert stage.metrics["metadata"]["execution_state"] == "convrot_4bit"
    assert stage.metrics["metadata"]["recommended_high_precision_modules"] == ["fc"]
