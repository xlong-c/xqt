import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from xqt.core.errors import XQTBackendError
from xqt.quant import (
    IterableCalibrationDataReader,
    QuantizationPolicy,
    analyze_layer_sensitivity,
    calibrate_activation_statistics,
    list_quantizable_modules,
    quantize_onnx_qdq_static,
    should_quantize_module,
    suggest_high_precision_modules,
)
from xqt.quant.torchao_backend import quantize_with_torchao
from xqt.data import SyntheticClassificationSpec, build_synthetic_classification_loader
from xqt.data.torchvision import (
    TorchvisionImageClassificationSpec,
    build_torchvision_image_classification_loader,
)


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(3, 4),
            nn.ReLU(),
            nn.Linear(4, 4),
        )
        self.norm = nn.LayerNorm(4)
        self.head = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.norm(x)
        return self.head(x)


def test_quantization_policy_selects_expected_modules() -> None:
    model = TinyModel()
    policy = QuantizationPolicy(min_parameters=1)

    candidates = list_quantizable_modules(model, policy)
    selected = {candidate.name for candidate in candidates if candidate.quantize}

    assert "features.0" in selected
    assert "features.2" in selected
    assert "head" not in selected
    assert "norm" not in selected
    assert should_quantize_module("manual", nn.Linear(2, 2), policy) is True


def test_quantization_policy_include_name_overrides_default_filters() -> None:
    policy = QuantizationPolicy(include_module_names=("head",))

    assert should_quantize_module("head", nn.Linear(4, 2), policy) is True


def test_calibrate_activation_statistics_collects_ranges() -> None:
    model = TinyModel()
    batches = [torch.ones(2, 3), torch.full((2, 3), 2.0)]

    stats = calibrate_activation_statistics(
        model,
        batches,
        module_names=["features.0", "head"],
    )

    assert [stat.name for stat in stats] == ["features.0", "head"]
    assert all(stat.samples > 0 for stat in stats)
    assert all(stat.maximum >= stat.minimum for stat in stats)


def test_synthetic_classification_loader_supports_image_shape() -> None:
    loader = build_synthetic_classification_loader(
        SyntheticClassificationSpec(
            sample_limit=2,
            batch_size=1,
            input_shape=[3, 16, 16],
            num_classes=4,
        )
    )

    inputs, targets = next(iter(loader))

    assert inputs.shape == (1, 3, 16, 16)
    assert targets.shape == (1,)


def test_torchvision_image_classification_loader_uses_local_cifar100() -> None:
    loader = build_torchvision_image_classification_loader(
        TorchvisionImageClassificationSpec(
            dataset_name="CIFAR100",
            root="data",
            train=False,
            download=False,
            sample_limit=1,
            batch_size=1,
            transform_params={"image_size": 32},
        )
    )

    inputs, targets = next(iter(loader))

    assert inputs.shape == (1, 3, 32, 32)
    assert targets.shape == (1,)


def test_calibrate_activation_statistics_rejects_missing_module() -> None:
    with pytest.raises(KeyError, match="Modules not found"):
        calibrate_activation_statistics(TinyModel(), [torch.ones(1, 3)], module_names=["x"])


def test_analyze_layer_sensitivity_and_suggestions() -> None:
    reference = TinyModel()
    candidate = TinyModel()
    candidate.load_state_dict(reference.state_dict())
    with torch.no_grad():
        candidate.features[0].bias.add_(0.5)
        candidate.features[2].bias.add_(0.1)

    records = analyze_layer_sensitivity(
        reference,
        candidate,
        torch.ones(2, 3),
        module_names=["features.0", "features.2"],
    )

    assert {record.name for record in records} == {"features.0", "features.2"}
    assert records[0].diff.max_abs >= records[1].diff.max_abs
    assert suggest_high_precision_modules(records, top_k=1) == [records[0].name]
    threshold = (records[0].diff.max_abs + records[1].diff.max_abs) / 2.0
    assert suggest_high_precision_modules(records, max_abs_threshold=threshold) == [
        records[0].name
    ]


def test_analyze_layer_sensitivity_rejects_shape_mismatch() -> None:
    reference = TinyModel()
    candidate = nn.Sequential(nn.Linear(3, 5))

    with pytest.raises(KeyError):
        analyze_layer_sensitivity(
            reference,
            candidate,
            torch.ones(1, 3),
            module_names=["features.0"],
        )


def test_quantize_with_torchao_quantizes_selected_modules() -> None:
    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2))

    result = quantize_with_torchao(
        model,
        policy={
            "dtype": "dynamic_int8",
            "include_module_types": ["Linear"],
            "exclude_name_patterns": ["2"],
        },
    )

    assert result.model is model
    assert result.backend == "torchao"
    assert result.strategy == "dynamic_int8"
    assert result.quantized_modules == ["0"]


def test_quantize_with_torchao_rejects_unknown_strategy() -> None:
    with pytest.raises(XQTBackendError, match="Unsupported torchao"):
        quantize_with_torchao(nn.Linear(2, 2), strategy="unknown")


def test_iterable_calibration_data_reader_handles_tensor_tuple_and_mapping() -> None:
    tensor_reader = IterableCalibrationDataReader([torch.ones(1, 3)], input_names=["input"])
    assert tensor_reader.samples == 1
    assert tensor_reader.get_next()["input"].shape == (1, 3)
    assert tensor_reader.get_next() is None
    tensor_reader.rewind()
    assert tensor_reader.get_next()["input"].shape == (1, 3)

    tuple_reader = IterableCalibrationDataReader(
        [(torch.ones(1, 3), torch.tensor([1]))],
        input_names=["input"],
    )
    assert tuple_reader.get_next()["input"].shape == (1, 3)

    mapping_reader = IterableCalibrationDataReader(
        [{"input_ids": torch.ones(1, 3), "labels": torch.tensor([1])}],
        input_names=["input_ids"],
    )
    assert set(mapping_reader.get_next()) == {"input_ids"}


def test_quantize_onnx_qdq_static_calls_onnxruntime_quantizer(monkeypatch, tmp_path) -> None:
    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_bytes(b"onnx")
    output_path = tmp_path / "model_qdq.onnx"
    calls = {}

    class FakeQuantFormat:
        QDQ = "QDQ"

    class FakeQuantType:
        QInt8 = "QInt8"
        QUInt8 = "QUInt8"

    def fake_quantize_static(*args, **kwargs):
        calls["args"] = args
        calls["kwargs"] = kwargs
        output_path.write_bytes(b"qdq")

    original_import = __import__

    def fake_import(
        name: str,
        globals=None,
        locals=None,
        fromlist=(),
        level: int = 0,
    ):
        if name == "onnxruntime.quantization":
            class FakeQuantizationModule:
                QuantFormat = FakeQuantFormat
                QuantType = FakeQuantType
                quantize_static = staticmethod(fake_quantize_static)

            return FakeQuantizationModule
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr("builtins.__import__", fake_import)
    loader = DataLoader(TensorDataset(torch.ones(2, 3), torch.zeros(2, dtype=torch.long)))

    result = quantize_onnx_qdq_static(
        onnx_path,
        output_path,
        loader,
        input_names=["input"],
        sample_limit=1,
        op_types_to_quantize=["Conv", "MatMul"],
    )

    assert result.path == output_path
    assert result.checksum
    assert result.calibration_samples == 1
    assert calls["args"][:2] == (str(onnx_path), str(output_path))
    assert calls["kwargs"]["quant_format"] == "QDQ"
    assert calls["kwargs"]["activation_type"] == "QUInt8"
    assert calls["kwargs"]["weight_type"] == "QInt8"
    assert calls["kwargs"]["op_types_to_quantize"] == ["Conv", "MatMul"]


def test_quantize_onnx_qdq_static_rejects_missing_onnx(tmp_path) -> None:
    with pytest.raises(XQTBackendError, match="ONNX file not found"):
        quantize_onnx_qdq_static(
            tmp_path / "missing.onnx",
            tmp_path / "out.onnx",
            [torch.ones(1, 3)],
        )
