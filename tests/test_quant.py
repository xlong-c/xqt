from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from xqt.core.config import load_xqt_config
from xqt.core.errors import XQTBackendError
from xqt.quant import (
    IterableCalibrationDataReader,
    QuantizationPolicy,
    analyze_activation_drift,
    analyze_layer_errors,
    analyze_layer_sensitivity,
    build_fake_qdq_surrogate,
    calibrate_activation_statistics,
    describe_quant_backend_capability,
    list_quant_backend_capabilities,
    list_quantizable_modules,
    quantize_onnx_qdq_static,
    recommend_high_precision_modules,
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


class TinyChainModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(2, 2, bias=False),
            nn.Linear(2, 2, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


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


def test_quant_backend_capability_matrix_describes_supported_and_planned_paths() -> None:
    torchao_fp8 = describe_quant_backend_capability(
        "torchao",
        strategy="fp8_dynamic",
    )
    qdq = describe_quant_backend_capability("onnxruntime_qdq")
    available = list_quant_backend_capabilities(include_planned=False)
    full_matrix = list_quant_backend_capabilities()

    assert torchao_fp8.runtime == "pytorch"
    assert torchao_fp8.requires_cuda is True
    assert torchao_fp8.primary_module_types == ("Linear",)
    assert "LayerNorm" in torchao_fp8.default_high_precision
    assert qdq.requires_calibration is True
    assert qdq.requires_exportable_graph is True
    assert "Conv2d" in qdq.primary_module_types
    assert "gptq" not in available
    assert full_matrix["gptq"]["status"] == "planned"


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


def test_quant_analysis_defaults_to_quantizable_modules_only() -> None:
    reference = TinyModel()
    candidate = TinyModel()
    candidate.load_state_dict(reference.state_dict())

    stats = calibrate_activation_statistics(reference, [torch.ones(2, 3)])
    sensitivity_records = analyze_layer_sensitivity(reference, candidate, torch.ones(2, 3))
    drift_records = analyze_activation_drift(reference, candidate, [torch.ones(2, 3)])

    assert [stat.name for stat in stats] == ["features.0", "features.2"]
    assert {record.name for record in sensitivity_records} == {"features.0", "features.2"}
    assert {record.name for record in drift_records} == {"features.0", "features.2"}


def test_analyze_activation_drift_reports_stat_deltas() -> None:
    reference = TinyModel()
    candidate = TinyModel()
    candidate.load_state_dict(reference.state_dict())
    with torch.no_grad():
        candidate.features[0].bias.add_(0.5)

    drift = analyze_activation_drift(
        reference,
        candidate,
        [torch.ones(2, 3), torch.full((2, 3), 2.0)],
        module_names=["features.0"],
    )

    assert len(drift) == 1
    assert drift[0].name == "features.0"
    assert drift[0].mean_delta != 0.0
    assert drift[0].range_ratio is not None
    assert drift[0].to_dict()["reference"]["name"] == "features.0"
    assert "saturation_ratio" in drift[0].to_dict()["reference"]
    assert "clipping_ratio_delta" in drift[0].to_dict()


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


def test_analyze_layer_sensitivity_isolated_mode_measures_final_output_effect() -> None:
    reference = TinyChainModel()
    candidate = TinyChainModel()
    with torch.no_grad():
        reference.features[0].weight.copy_(torch.eye(2))
        reference.features[1].weight.copy_(torch.eye(2))
        candidate.load_state_dict(reference.state_dict())
        candidate.features[0].weight.mul_(2.0)
        candidate.features[1].weight.mul_(3.0)

    records = analyze_layer_sensitivity(
        reference,
        candidate,
        torch.tensor([[1.0, 2.0]]),
        module_names=["features.0", "features.1"],
    )

    by_name = {record.name: record for record in records}
    assert by_name["features.1"].diff.mean_abs > by_name["features.0"].diff.mean_abs
    assert by_name["features.1"].diff.mean_abs == pytest.approx(3.0)
    assert by_name["features.0"].diff.mean_abs == pytest.approx(1.5)


def test_analyze_layer_errors_collects_summaries_and_weight_diff() -> None:
    reference = TinyModel()
    candidate = TinyModel()
    candidate.load_state_dict(reference.state_dict())
    with torch.no_grad():
        candidate.features[0].weight.add_(0.2)
        candidate.features[0].bias.add_(0.5)
        candidate.features[2].bias.add_(0.1)

    records = analyze_layer_errors(
        reference,
        candidate,
        torch.ones(2, 3),
        module_names=["features.0", "features.2"],
    )

    assert {record.name for record in records} == {"features.0", "features.2"}
    assert records[0].diff.mean_abs >= records[1].diff.mean_abs
    by_name = {record.name: record for record in records}
    assert by_name["features.0"].reference_summary["shape"] == [2, 4]
    assert by_name["features.0"].candidate_summary["shape"] == [2, 4]
    assert by_name["features.0"].weight_diff is not None
    assert by_name["features.0"].weight_diff.mean_abs > 0.0
    assert "high_error" in by_name["features.0"].tags
    assert by_name["features.0"].recommendation == "consider_higher_precision"
    assert by_name["features.0"].to_dict()["weight_diff"] is not None


def test_analyze_layer_errors_respects_sample_budget() -> None:
    reference = TinyModel()
    candidate = TinyModel()
    candidate.load_state_dict(reference.state_dict())
    with torch.no_grad():
        candidate.features[0].bias[0] += 2.0

    full_records = analyze_layer_errors(
        reference,
        candidate,
        torch.ones(2, 3),
        module_names=["features.0"],
        sample_budget=None,
    )
    sampled_records = analyze_layer_errors(
        reference,
        candidate,
        torch.ones(2, 3),
        module_names=["features.0"],
        sample_budget=1,
        sample_seed=0,
    )

    assert full_records[0].reference_summary["shape"] == [2, 4]
    assert sampled_records[0].diff.reference_summary is not None
    assert sampled_records[0].diff.reference_summary.shape == (1,)
    assert sampled_records[0].diff.mean_abs != full_records[0].diff.mean_abs


def test_recommend_high_precision_modules_uses_analysis_records() -> None:
    reference = TinyModel()
    candidate = TinyModel()
    candidate.load_state_dict(reference.state_dict())
    with torch.no_grad():
        candidate.features[0].weight.add_(0.2)
        candidate.features[0].bias.add_(0.5)
        candidate.features[2].bias.add_(0.1)

    records = analyze_layer_errors(
        reference,
        candidate,
        torch.ones(2, 3),
        module_names=["features.0", "features.2"],
    )

    recommended = recommend_high_precision_modules(records, top_k=1)
    sorted_by_mean_abs = sorted(records, key=lambda record: record.diff.mean_abs, reverse=True)
    assert recommended == [sorted_by_mean_abs[0].name]
    threshold = (records[0].diff.mean_abs + records[1].diff.mean_abs) / 2.0
    threshold_expected = [
        record.name for record in sorted_by_mean_abs if record.diff.mean_abs >= threshold
    ]
    assert recommend_high_precision_modules(records, mean_abs_threshold=threshold) == threshold_expected
    assert recommend_high_precision_modules(records, require_weight_shift=True) == [
        record.name
        for record in sorted_by_mean_abs
        if record.weight_diff is not None and record.weight_diff.mean_abs > 0.0
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


def test_build_fake_qdq_surrogate_introduces_weight_and_activation_drift() -> None:
    class TinyConvModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 1, kernel_size=2, bias=False),
                nn.ReLU(),
            )
            with torch.no_grad():
                self.features[0].weight.copy_(
                    torch.tensor([[[[0.1234, 0.0510], [0.2000, -0.1700]]]])
                )

        def forward(self, image: torch.Tensor) -> torch.Tensor:
            return self.features(image)

    model = TinyConvModel()
    config = load_xqt_config(
        {
            "project": {"artifact_dir": "artifacts/xqt/test_fake_qdq_surrogate"},
            "model": {"device": "cpu"},
            "compression": {
                "quant": {
                    "enabled": True,
                    "backend": "onnxruntime_qdq",
                    "calibration_split": "calibration",
                    "policy": {
                        "sample_limit": 1,
                        "activation_type": "QUInt8",
                        "weight_type": "QInt8",
                        "op_types_to_quantize": ["Conv"],
                        "include_module_types": ["Conv2d"],
                        "exclude_name_patterns": [],
                    },
                }
            },
        }
    )
    calibration_loader = [
        {"image": torch.full((1, 1, 2, 2), 0.75), "labels": torch.tensor([0])}
    ]
    context = SimpleNamespace(
        config=config,
        model=model,
        data={"calibration": calibration_loader},
    )

    surrogate = build_fake_qdq_surrogate(context)

    assert surrogate is not None
    assert surrogate.source_split == "calibration"
    assert surrogate.sample_count == 1
    assert "features.0" in surrogate.quantized_modules
    reference_weight = model.features[0].weight.detach().clone()
    candidate_weight = surrogate.model.features[0].weight.detach().clone()
    assert not torch.allclose(reference_weight, candidate_weight)

    records = analyze_layer_errors(
        model,
        surrogate.model,
        torch.full((1, 1, 2, 2), 0.75),
        module_names=["features.0"],
        sample_budget=None,
    )
    assert records
    assert records[0].diff.mean_abs > 0.0
    assert records[0].weight_diff is not None
    assert records[0].weight_diff.mean_abs > 0.0


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
    assert result.metadata["calibration_summary"]["batch_count"] == 1
    assert result.metadata["calibration_summary"]["input_names"] == ["input"]


def test_iterable_calibration_data_reader_supports_tuple_multi_input_names() -> None:
    reader = IterableCalibrationDataReader(
        [
            (
                torch.ones(1, 3),
                torch.zeros(1, 3),
                torch.tensor([1]),
            )
        ],
        input_names=["left", "right"],
    )

    record = reader.get_next()
    assert record is not None
    assert set(record.keys()) == {"left", "right"}
    assert record["left"].shape == (1, 3)
    assert record["right"].shape == (1, 3)


def test_iterable_calibration_data_reader_supports_unlabeled_tuple_multi_input_names() -> None:
    reader = IterableCalibrationDataReader(
        [
            (
                torch.ones(1, 3),
                torch.zeros(1, 3),
            )
        ],
        input_names=["left", "right"],
    )

    record = reader.get_next()
    assert record is not None
    assert set(record.keys()) == {"left", "right"}
    assert record["left"].shape == (1, 3)
    assert record["right"].shape == (1, 3)


def test_iterable_calibration_data_reader_supports_mapping_multi_input_names() -> None:
    reader = IterableCalibrationDataReader(
        [
            {
                "input_ids": torch.ones(1, 3),
                "attention_mask": torch.zeros(1, 3),
                "labels": torch.tensor([1]),
            }
        ],
        input_names=["input_ids", "attention_mask"],
    )

    record = reader.get_next()
    assert record is not None
    assert set(record.keys()) == {"input_ids", "attention_mask"}


def test_quantize_onnx_qdq_static_rejects_missing_onnx(tmp_path) -> None:
    with pytest.raises(XQTBackendError, match="ONNX file not found"):
        quantize_onnx_qdq_static(
            tmp_path / "missing.onnx",
            tmp_path / "out.onnx",
            [torch.ones(1, 3)],
        )
