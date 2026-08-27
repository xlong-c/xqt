import torch

from xqt.quant import quantize_with_convrot_4bit
from xqt.runtime import (
    ConvRotW4A4ExecutionView,
    HybridInferenceEngine,
    apply_execution_policy,
    normalize_compute_precision,
)


class _TinyLinearModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = torch.nn.Linear(16, 32, bias=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.fc(inputs)


def _quantize_tiny() -> object:
    model = _TinyLinearModel().eval()
    return quantize_with_convrot_4bit(
        model,
        policy={
            "dtype": "int4",
            "scheme": "convrot_w4a4",
            "include_module_names": ["fc"],
            "group_size": 8,
            "rot_size": 4,
            "mixed_precision_ratio": 1.0,
        },
        calibration_inputs=[torch.randn(4, 16)],
        inplace=False,
    )


def test_normalize_compute_precision_aliases() -> None:
    assert normalize_compute_precision("W4A4") == "w4a4"
    assert normalize_compute_precision("w4") == "w4a16"
    assert normalize_compute_precision("int8") == "w8a8"
    assert normalize_compute_precision("fp8") == "fp8"
    assert normalize_compute_precision("bfloat16") == "bf16"


def test_hybrid_engine_from_quantized_model_applies_policy_without_requant() -> None:
    result = _quantize_tiny()
    engine = HybridInferenceEngine.from_quantized_model(
        result,
        default_precision="w4a4",
        apply_policy_on_init=True,
    )
    output = engine(torch.randn(4, 16))
    precision_map = engine.precision_map()

    assert output.shape == (4, 32)
    assert isinstance(engine.model.fc, ConvRotW4A4ExecutionView)
    assert precision_map["fc"] == "w8a8"
    assert engine.policy is not None
    assert engine.policy.policy_kind == "mixed_precision"


def test_hybrid_engine_switches_precision_inplace() -> None:
    result = _quantize_tiny()
    engine = HybridInferenceEngine.from_quantized_model(
        result,
        default_precision="w4a4",
        apply_policy_on_init=False,
    )
    engine.set_module_precision("fc", "w4a16")
    run = engine.run(torch.randn(2, 16))

    assert engine.precision_map()["fc"] == "w4a16"
    assert run.output.shape == (2, 32)
    assert run.precision_map["fc"] == "w4a16"


def test_hybrid_engine_can_keep_reference_storage_explicitly() -> None:
    result = _quantize_tiny()
    engine = HybridInferenceEngine.from_quantized_model(
        result,
        materialize_execution_views=False,
        apply_policy_on_init=False,
    )

    assert not isinstance(engine.model.fc, ConvRotW4A4ExecutionView)
    assert engine.model.fc._xqt_runtime_execution_enabled is False


def test_apply_execution_policy_does_not_mutate_source_when_not_inplace() -> None:
    result = _quantize_tiny()
    source = result.model
    source.fc.set_compute_precision("w4a4")
    candidate = apply_execution_policy(
        source,
        precision_overrides=[{"module": "fc", "precision": "bf16"}],
        default_precision="w4a4",
        inplace=False,
    )

    assert candidate.fc.compute_precision == "bf16"
    assert source.fc.compute_precision == "w4a4"


def test_channel_hybrid_quant_and_engine_dual_path() -> None:
    model = _TinyLinearModel().eval()
    result = quantize_with_convrot_4bit(
        model,
        policy={
            "dtype": "int4",
            "scheme": "convrot_w4a4",
            "include_module_names": ["fc"],
            "group_size": 8,
            "rot_size": 4,
            "channel_hybrid_ratio": 0.25,
            "channel_hybrid_axis": "input",
            "mixed_precision_ratio": 0.0,
        },
        calibration_inputs=[torch.randn(8, 16)],
        inplace=False,
    )
    module = result.model.fc
    assert module.channel_hybrid_enabled is True
    assert int(module.high_precision_channel_mask.sum().item()) == 4

    engine = HybridInferenceEngine.from_quantized_model(
        result,
        default_precision="w4a4",
        apply_policy_on_init=True,
    )
    run = engine.run(torch.randn(3, 16))
    assert run.output.shape == (3, 32)
    assert "fc" in run.channel_hybrid_map
    assert run.channel_hybrid_map["fc"]["axis"] == "input"
    assert len(run.channel_hybrid_map["fc"]["high_precision_channels"]) == 4

    engine.set_module_channel_hybrid(
        "fc",
        high_precision_channels=[0, 1],
        axis="input",
    )
    assert engine.channel_hybrid_map()["fc"]["high_precision_channels"] == [0, 1]
    engine.clear_module_channel_hybrid("fc")
    assert engine.channel_hybrid_map() == {}
