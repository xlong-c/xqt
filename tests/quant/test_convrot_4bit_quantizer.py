import pytest
import torch

from xqt.compression.quant import (
    ConvRotMixedPrecisionLinear,
    build_regular_hadamard_matrix,
    quantize_with_convrot_4bit,
)
from xqt.runtime import apply_execution_policy
from xqt.runtime import ConvRotW4A4ExecutionView
from xqt.workflows import XQTOptimizationSession


class _TinyLinearModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = torch.nn.Linear(16, 32, bias=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.fc(inputs)


def _runtime_convrot_w4a4(
    storage: ConvRotMixedPrecisionLinear,
) -> ConvRotW4A4ExecutionView:
    return ConvRotW4A4ExecutionView.from_storage(storage).eval()


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
    from xqt.compression.quant.quantizers.convrot_4bit import _apply_groupwise_rotation

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


def test_convrot_w4a4_runtime_backend_rejects_unknown_value() -> None:
    with pytest.raises(ValueError, match="w4a4_runtime_backend must be one of"):
        ConvRotMixedPrecisionLinear.from_linear(
            torch.nn.Linear(16, 32, bias=False),
            rot_size=4,
            group_size=4,
            w4a4_runtime_backend="unknown",
        )


def test_convrot_w4a4_auto_preserves_grouped_artifact_semantics() -> None:
    grouped = ConvRotMixedPrecisionLinear.from_linear(
        torch.nn.Linear(16, 32, bias=False),
        rot_size=4,
        group_size=4,
        w4a4_runtime_backend="auto",
    )
    rowwise = ConvRotMixedPrecisionLinear.from_linear(
        torch.nn.Linear(16, 32, bias=False),
        rot_size=4,
        group_size=16,
        w4a4_runtime_backend="auto",
    )

    assert grouped._rowwise_w4a4_requested() is False
    assert rowwise._rowwise_w4a4_requested() is True


def test_convrot_w4a4_explicit_rowwise_overrides_grouped_artifact_policy() -> None:
    module = ConvRotMixedPrecisionLinear.from_linear(
        torch.nn.Linear(16, 32, bias=False),
        rot_size=4,
        group_size=4,
        w4a4_runtime_backend="rowwise",
    )

    assert module._rowwise_w4a4_requested() is True
    assert module.w4a4_runtime_backend == "rowwise"


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


def _rowwise_convrot_w4a4_cuda_available() -> bool:
    if not torch.cuda.is_available():
        return False
    if torch.cuda.get_device_capability() != (8, 9):
        return False
    try:
        from xqt.kernels.ops._impl.cute.convrot_w4a4_rowwise_sm89 import (
            native_rowwise_convrot_w4a4_available,
        )
    except Exception:
        return False
    return native_rowwise_convrot_w4a4_available(build=False)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_convrot_rowwise_w4a4_cuda_matches_packed_reference(
    dtype: torch.dtype,
) -> None:
    if not _rowwise_convrot_w4a4_cuda_available():
        pytest.skip("rowwise sm_89 ConvRot W4A4 backend unavailable")

    from xqt.kernels.ops._impl.cute.convrot_w4a4_rowwise_sm89 import (
        allocate_convrot_w4a4_rowwise_workspace,
        convrot_w4a4_rowwise_linear,
        pack_convrot_w4a4_rowwise_weight,
    )
    from xqt.compression.quant.quantizers.convrot_4bit import _apply_groupwise_rotation
    from xqt.compression.quant.quantizers.fp4_weight_only import _unpack_int4

    torch.manual_seed(101)
    rows = 7
    features = 1024
    storage = torch.randn(rows, features * 2, device="cuda", dtype=dtype)
    inputs = storage[:, ::2]
    rotated_weight = torch.randn(
        features,
        features,
        device="cuda",
        dtype=dtype,
    )
    bias = torch.randn(features, device="cuda", dtype=dtype)
    packed = pack_convrot_w4a4_rowwise_weight(rotated_weight, bias)
    workspace = allocate_convrot_w4a4_rowwise_workspace(rows, packed)

    output = convrot_w4a4_rowwise_linear(inputs, packed, workspace=workspace)
    torch.cuda.synchronize()

    activation_codes = _unpack_int4(
        workspace.quantized_activation.view(torch.uint8),
        features,
    )
    weight_codes = _unpack_int4(packed.qweight.view(torch.uint8), features)
    accumulator = activation_codes @ weight_codes.t()
    expected = (
        accumulator
        * workspace.activation_scales[:, None]
        * packed.weight_scales[None, :]
        + packed.bias
    ).to(dtype)
    rotated = _apply_groupwise_rotation(inputs, rot_size=256)
    reference_scales = torch.clamp(
        rotated.float().abs().amax(dim=1) / 7.0,
        min=1.0e-10,
    )
    reference_codes = torch.clamp(
        torch.round(rotated.float() / reference_scales[:, None]),
        min=-7,
        max=7,
    )

    assert not inputs.is_contiguous()
    torch.testing.assert_close(output, expected, rtol=0.0, atol=1.0e-2)
    assert int((activation_codes - reference_codes).abs().max().item()) <= 1
    torch.testing.assert_close(
        workspace.activation_scales,
        reference_scales,
        rtol=5.0e-3,
        atol=5.0e-3,
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_convrot_rowwise_w4a4_cuda_runtime_cache_and_mutation(
    dtype: torch.dtype,
) -> None:
    if not _rowwise_convrot_w4a4_cuda_available():
        pytest.skip("rowwise sm_89 ConvRot W4A4 backend unavailable")

    torch.manual_seed(103)
    source = torch.nn.Linear(
        1024,
        1024,
        bias=True,
        device="cuda",
        dtype=dtype,
    ).eval()
    module = _runtime_convrot_w4a4(ConvRotMixedPrecisionLinear.from_linear(
        source,
        rot_size=256,
        group_size=128,
        compute_precision="w4a4",
        activation_scale_mode="dynamic",
        w4a4_runtime_backend="rowwise",
    ))
    storage = torch.randn(2, 3, 2048, device="cuda", dtype=dtype)
    inputs = storage[..., ::2]

    first = module(inputs)
    second = module(inputs)
    first_runner_cache = module._rowwise_w4a4_runner_cache
    assert first_runner_cache is not None
    first_runner = first_runner_cache[-1]

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        stream_output = module(inputs)
    stream.synchronize()
    torch.cuda.synchronize()

    with torch.no_grad():
        module.weight_scale.mul_(1.25)
    after_scale = module(inputs)
    scale_runner_cache = module._rowwise_w4a4_runner_cache
    assert scale_runner_cache is not None

    before_bias = module(inputs)
    with torch.no_grad():
        assert module.bias is not None
        module.bias.add_(0.25)
    after_bias = module(inputs)
    bias_runner_cache = module._rowwise_w4a4_runner_cache
    assert bias_runner_cache is not None
    torch.cuda.synchronize()

    metadata = module.execution_metadata()
    assert not inputs.is_contiguous()
    assert first.shape == (2, 3, 1024)
    assert torch.equal(first, second)
    assert torch.equal(first, stream_output)
    assert first_runner.workspace_count() == 2
    assert scale_runner_cache[-1] is not first_runner
    assert bias_runner_cache[-1] is not scale_runner_cache[-1]
    assert not torch.equal(first, after_scale)
    torch.testing.assert_close(
        (after_bias - before_bias).float(),
        torch.full_like(after_bias.float(), 0.25),
        rtol=0.0,
        atol=2.0e-2,
    )
    assert metadata["implementation"] == "native_convrot_w4a4_rowwise_dynamic_runner"
    assert metadata["rowwise_w4a4_used"] is True
    assert metadata["runtime_weight_layout"] == "row_major_signed_int4_rowwise"
    assert metadata["fused_epilogue"] == "activation_scale_weight_scale_bias"
    assert metadata["norm_fused"] is False

    module.to(dtype=torch.bfloat16 if dtype == torch.float16 else torch.float16)
    assert module._rowwise_w4a4_packed_cache is None
    assert module._rowwise_w4a4_runner_cache is None


def test_convrot_rowwise_w4a4_gate_rejects_static_activation_scale() -> None:
    if not _rowwise_convrot_w4a4_cuda_available():
        pytest.skip("rowwise sm_89 ConvRot W4A4 backend unavailable")

    module = _runtime_convrot_w4a4(ConvRotMixedPrecisionLinear.from_linear(
        torch.nn.Linear(1024, 1024, bias=False, device="cuda", dtype=torch.float16),
        rot_size=256,
        group_size=128,
        compute_precision="w4a4",
        activation_scale_mode="static",
        w4a4_runtime_backend="rowwise",
    ))
    allowed, reason = module._rowwise_w4a4_gate(
        torch.randn(2, 1024, device="cuda", dtype=torch.float16)
    )

    assert allowed is False
    assert "dynamic activation scales" in reason
