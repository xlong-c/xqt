import torch
from torch import nn

from xqt.compression.quant import quantize_with_convrot_4bit, quantize_with_convrot_int8
from xqt.runtime import (
    ConvRotInt8ExecutionView,
    ConvRotW4A4ExecutionView,
    apply_execution_policy,
    materialize_convrot_execution_views,
)


class _TinyLinearModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(16, 32, bias=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.fc(inputs)


def test_materialize_convrot_int8_execution_view_keeps_storage_model_unchanged() -> None:
    result = quantize_with_convrot_int8(
        _TinyLinearModel().eval(),
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
    inputs = torch.randn(3, 16)
    storage = result.model.fc
    reference = storage(inputs)

    runtime_model = materialize_convrot_execution_views(result.model, inplace=False)
    runtime = runtime_model.fc

    assert isinstance(runtime, ConvRotInt8ExecutionView)
    assert runtime.execution_metadata()["artifact_view"] == "runtime_execution"
    assert storage.execution_metadata()["artifact_view"] == "contracts_reference"
    assert result.model.fc is storage
    torch.testing.assert_close(runtime(inputs), reference, rtol=0.0, atol=0.0)


def test_convrot_storage_constructors_default_to_reference_execution() -> None:
    int8_result = quantize_with_convrot_int8(
        _TinyLinearModel().eval(),
        policy={
            "dtype": "int8",
            "scheme": "convrot_w8a8",
            "include_module_names": ["fc"],
            "rot_size": 4,
        },
        inplace=False,
        engine="auto",
    )
    w4a4_result = quantize_with_convrot_4bit(
        _TinyLinearModel().eval(),
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

    assert int8_result.model.fc._xqt_runtime_execution_enabled is False
    assert w4a4_result.model.fc._xqt_runtime_execution_enabled is False
    assert int8_result.model.fc.execution_metadata()["artifact_view"] == (
        "contracts_reference"
    )
    assert w4a4_result.model.fc.execution_metadata()["artifact_view"] == (
        "contracts_reference"
    )


def test_materialize_convrot_w4a4_execution_view_exposes_storage_policy() -> None:
    result = quantize_with_convrot_4bit(
        _TinyLinearModel().eval(),
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
    runtime_model = materialize_convrot_execution_views(result.model, inplace=False)
    runtime = runtime_model.fc

    assert isinstance(runtime, ConvRotW4A4ExecutionView)
    assert runtime.compute_precision == "w4a4"
    runtime_model = apply_execution_policy(
        runtime_model,
        precision_overrides=[{"module": "fc", "precision": "bf16"}],
        default_precision="w4a4",
        inplace=True,
    )
    runtime = runtime_model.fc
    assert isinstance(runtime, ConvRotW4A4ExecutionView)
    assert runtime.storage.compute_precision == "bf16"
    assert runtime.execution_metadata()["artifact_view"] == "runtime_execution"


def test_materialize_convrot_execution_views_can_enable_in_place() -> None:
    result = quantize_with_convrot_4bit(
        _TinyLinearModel().eval(),
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

    materialize_convrot_execution_views(result.model, inplace=True)

    assert isinstance(result.model.fc, ConvRotW4A4ExecutionView)
    assert result.model.fc.storage._xqt_runtime_execution_enabled is True


def test_materialize_convrot_execution_views_is_idempotent() -> None:
    result = quantize_with_convrot_int8(
        _TinyLinearModel().eval(),
        policy={
            "dtype": "int8",
            "scheme": "convrot_w8a8",
            "include_module_names": ["fc"],
            "rot_size": 4,
        },
        inplace=False,
        engine="torch_int_mm",
    )

    materialize_convrot_execution_views(result.model, inplace=True)
    first = result.model.fc
    materialize_convrot_execution_views(result.model, inplace=True)

    assert result.model.fc is first
    assert isinstance(result.model.fc, ConvRotInt8ExecutionView)
    assert not isinstance(result.model.fc.storage, ConvRotInt8ExecutionView)
