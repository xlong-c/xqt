import pytest
import torch
from torch import nn

from xqt.eval.accuracy import evaluate_pytorch_model, topk_accuracy
from xqt.eval.compare import compare_tensors, summarize_tensor
from xqt.eval.detection import (
    compare_decoded_detections,
    decode_detection_output,
    evaluate_detection_model,
    evaluate_detection_runtime_model,
    evaluate_onnx_detection_model,
    evaluate_detection_predictions,
)
from xqt.export import convert_onnx_to_fp16, export_onnx
from xqt.eval.report import (
    build_pareto_points,
    flatten_metrics,
    records_to_dataframe,
    records_to_rows,
    write_csv_report,
    write_json_report,
    write_markdown_report,
)
from xqt.core.schema import DetectionMetricConfig, DetectionPostprocessConfig
from xqt.quant.sensitivity import LayerAnalysisRecord


class FixedLogitModel(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class PairLogitModel(nn.Module):
    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return left + right


class FixedDetectionModel(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        anchor = x.mean() * 0.0
        return anchor + torch.tensor(
            [
                [
                    [10.0],
                    [10.0],
                    [30.0],
                    [30.0],
                    [0.95],
                    [0.05],
                ]
            ],
            dtype=torch.float32,
        )


def test_compare_tensors_reports_common_diff_metrics() -> None:
    reference = torch.tensor([1.0, 2.0, 3.0])
    candidate = torch.tensor([1.0, 2.001, 2.999])

    diff = compare_tensors(reference, candidate, atol=0.01, rtol=0.01)

    assert diff.allclose is True
    assert diff.max_abs == pytest.approx(0.001, abs=1e-6)
    assert diff.mean_abs > 0.0
    assert diff.mean_squared > 0.0
    assert diff.relative_error is not None and diff.relative_error > 0.0
    assert diff.cosine_similarity == pytest.approx(1.0, abs=1e-5)
    assert diff.correlation == pytest.approx(1.0, abs=1e-5)
    assert diff.argmax_mismatch_rate == 0.0
    assert diff.valid is True
    assert diff.message == "ok"
    assert diff.reference_summary is not None
    assert diff.candidate_summary is not None
    assert diff.to_dict()["allclose"] is True
    assert diff.to_dict()["reference_summary"] is not None


def test_compare_tensors_handles_empty_and_zero_vectors() -> None:
    empty = compare_tensors(torch.empty(0), torch.empty(0))
    zeros = compare_tensors(torch.zeros(3), torch.zeros(3))

    assert empty.max_abs == 0.0
    assert empty.cosine_similarity is None
    assert empty.correlation is None
    assert empty.relative_error is None
    assert zeros.cosine_similarity is None
    assert zeros.correlation is None
    assert zeros.relative_error is None
    assert zeros.argmax_mismatch_rate == 0.0


def test_compare_tensors_supports_structured_details_and_argmax_mismatch() -> None:
    reference = torch.tensor([[0.1, 0.9], [0.8, 0.2]])
    candidate = torch.tensor([[0.6, 0.4], [0.8, 0.2]])

    diff = compare_tensors(
        reference,
        candidate,
        structured={"per_token": 0, "per_channel": 1},
    )

    assert diff.argmax_mismatch_rate == pytest.approx(0.5)
    assert diff.details is not None
    assert diff.details["per_token"]["axis"] == 0
    assert diff.details["per_token"]["size"] == 2
    assert len(diff.details["per_channel"]["mean_abs"]) == 2
    assert diff.to_dict()["details"] is not None


def test_compare_tensors_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="Tensor shapes differ"):
        compare_tensors(torch.zeros(1, 2), torch.zeros(2, 1))


def test_summarize_tensor_reports_statistics_and_special_values() -> None:
    tensor = torch.tensor([0.0, 1.0, float("nan"), float("inf"), -1.0])

    summary = summarize_tensor(tensor)

    assert summary.shape == (5,)
    assert summary.dtype == "torch.float32"
    assert summary.numel == 5
    assert summary.mean == pytest.approx(0.0)
    assert summary.std == pytest.approx((2.0 / 3.0) ** 0.5)
    assert summary.minimum == -1.0
    assert summary.maximum == 1.0
    assert summary.zero_ratio == pytest.approx(1.0 / 3.0)
    assert summary.nan_count == 1
    assert summary.inf_count == 1
    assert summary.quantiles["p01"] <= summary.quantiles["p50"] <= summary.quantiles["p99"]
    assert summary.to_dict()["shape"] == [5]


def test_flatten_metrics_uses_dotted_keys() -> None:
    metrics = {
        "baseline": {"top1": 0.8, "loss": 1.2},
        "latency": {"p50_ms": 3.4},
        "passed": True,
    }

    assert flatten_metrics(metrics) == {
        "baseline.top1": 0.8,
        "baseline.loss": 1.2,
        "latency.p50_ms": 3.4,
        "passed": True,
    }


def test_records_to_rows_and_dataframe_flatten_analysis_records() -> None:
    reference = torch.tensor([1.0, 2.0, 3.0])
    candidate = torch.tensor([1.0, 2.5, 2.5])
    diff = compare_tensors(reference, candidate)
    weight_diff = compare_tensors(torch.tensor([1.0, 2.0]), torch.tensor([1.1, 1.9]))
    record = LayerAnalysisRecord(
        name="features.0",
        module_type="Linear",
        diff=diff,
        parameter_count=16,
        reference_summary=diff.reference_summary.to_dict() if diff.reference_summary else {},
        candidate_summary=diff.candidate_summary.to_dict() if diff.candidate_summary else {},
        weight_diff=weight_diff,
        recommendation="consider_higher_precision",
        tags=("high_error",),
    )

    rows = records_to_rows([record])
    frame = records_to_dataframe([record])

    assert rows[0]["name"] == "features.0"
    assert rows[0]["diff.max_abs"] == diff.max_abs
    assert rows[0]["weight_diff.mean_abs"] == weight_diff.mean_abs
    assert rows[0]["reference_summary.shape"] == [3]
    assert rows[0]["tags"] == ["high_error"]
    assert frame.loc[0, "name"] == "features.0"
    assert float(frame.loc[0, "diff.max_abs"]) == pytest.approx(diff.max_abs)


def test_build_pareto_points_extracts_nested_metrics() -> None:
    runs = [
        {
            "config_id": "recipe_a",
            "benchmark": {
                "latency": {"p50_ms": 3.2},
                "memory": {"delta_bytes": 1024},
            },
            "analysis": {
                "records": [{"diff": {"mean_abs": 0.02}}],
            },
            "baseline": {
                "metrics": {"top1": 0.81},
            },
        }
    ]

    points = build_pareto_points(runs)

    assert points == [
        {
            "config_id": "recipe_a",
            "latency_ms": 3.2,
            "memory_bytes": 1024,
            "error": 0.02,
            "metric_delta": 0.81,
        }
    ]


def test_topk_accuracy_handles_multiclass_and_binary_logits() -> None:
    logits = torch.tensor([[0.1, 0.9], [0.8, 0.2]])
    targets = torch.tensor([1, 0])
    binary_logits = torch.tensor([1.0, -1.0, 2.0])
    binary_targets = torch.tensor([1, 0, 1])

    assert topk_accuracy(logits, targets) == 1.0
    assert topk_accuracy(logits, torch.tensor([0, 1]), k=2) == 1.0
    assert topk_accuracy(binary_logits, binary_targets) == 1.0

    with pytest.raises(ValueError, match="k must be positive"):
        topk_accuracy(logits, targets, k=0)


def test_evaluate_pytorch_model_computes_default_top1_and_restores_mode() -> None:
    model = FixedLogitModel()
    model.train()
    dataloader = [
        (torch.tensor([[0.1, 0.9], [0.8, 0.2]]), torch.tensor([1, 0])),
        (torch.tensor([[0.6, 0.4]]), torch.tensor([1])),
    ]

    report = evaluate_pytorch_model(model, dataloader)

    assert report.samples == 3
    assert report.metrics["top1"] == pytest.approx(2 / 3)
    assert report.to_dict()["samples"] == 3
    assert model.training is True


def test_evaluate_pytorch_model_supports_mapping_batches_and_custom_metrics() -> None:
    model = FixedLogitModel()
    dataloader = [
        {
            "x": torch.tensor([[0.1, 0.9], [0.8, 0.2]]),
            "y": torch.tensor([1, 1]),
        }
    ]

    report = evaluate_pytorch_model(
        model,
        dataloader,
        metrics={"top2": lambda output, target: topk_accuracy(output, target, k=2)},
    )

    assert report.samples == 2
    assert report.metrics == {"top2": 1.0}


def test_evaluate_pytorch_model_supports_unlabeled_multi_input_batches() -> None:
    report = evaluate_pytorch_model(
        PairLogitModel(),
        [
            (
                torch.tensor([[0.1, 0.9], [0.8, 0.2]]),
                torch.tensor([[0.0, 0.0], [0.0, 0.0]]),
            )
        ],
    )

    assert report.samples == 2
    assert report.metrics["top1"] == 0.0


def test_evaluate_pytorch_model_counts_samples_without_targets() -> None:
    report = evaluate_pytorch_model(FixedLogitModel(), [torch.zeros(4, 2)])

    assert report.samples == 4
    assert report.metrics["top1"] == 0.0


def test_evaluate_pytorch_model_rejects_unsupported_output() -> None:
    class BadModel(nn.Module):
        def forward(self, x: torch.Tensor) -> str:
            return "bad"

    with pytest.raises(TypeError, match="model output must be"):
        evaluate_pytorch_model(BadModel(), [(torch.zeros(1, 2), torch.zeros(1))])


def test_decode_detection_output_supports_end2end_shape() -> None:
    output = torch.tensor([[[10.0, 10.0, 20.0, 20.0, 0.9, 1.0]]], dtype=torch.float32)
    predictions = decode_detection_output(
        output,
        DetectionPostprocessConfig(format="end2end", max_detections=10),
    )

    assert len(predictions) == 1
    assert predictions[0].boxes.shape == (1, 4)
    assert predictions[0].scores.tolist() == [pytest.approx(0.9)]
    assert predictions[0].labels.tolist() == [1]


def test_decode_detection_output_supports_yolo_raw_shape() -> None:
    output = torch.tensor(
        [
            [
                [20.0],  # cx
                [20.0],  # cy
                [20.0],  # w
                [20.0],  # h
                [0.1],   # cls0
                [0.9],   # cls1
            ]
        ],
        dtype=torch.float32,
    )
    predictions = decode_detection_output(
        output,
        DetectionPostprocessConfig(
            format="yolo_raw",
            box_format="cxcywh",
            score_threshold=0.2,
            score_activation="sigmoid",
            max_detections=10,
        ),
    )

    assert len(predictions) == 1
    assert predictions[0].boxes.shape == (1, 4)
    assert predictions[0].labels.tolist() == [1]


def test_evaluate_detection_predictions_reports_map() -> None:
    predictions = decode_detection_output(
        torch.tensor([[[10.0, 10.0, 20.0, 20.0, 0.9, 1.0]]], dtype=torch.float32),
        DetectionPostprocessConfig(format="end2end", max_detections=10),
    )
    targets = [
        {
            "boxes": torch.tensor([[10.0, 10.0, 20.0, 20.0]], dtype=torch.float32),
            "labels": torch.tensor([1], dtype=torch.long),
        }
    ]
    metrics = evaluate_detection_predictions(
        predictions,
        targets,
        DetectionMetricConfig(iou_thresholds=[0.5, 0.75], max_detections=10),
    )

    assert metrics["map50_95"] == pytest.approx(1.0)
    assert metrics["map50"] == pytest.approx(1.0)
    assert metrics["map75"] == pytest.approx(1.0)


def test_compare_decoded_detections_reports_diff_summary() -> None:
    reference = decode_detection_output(
        torch.tensor([[[10.0, 10.0, 20.0, 20.0, 0.9, 1.0]]], dtype=torch.float32),
        DetectionPostprocessConfig(format="end2end", max_detections=10),
    )
    candidate = decode_detection_output(
        torch.tensor([[[11.0, 10.0, 20.0, 21.0, 0.8, 1.0]]], dtype=torch.float32),
        DetectionPostprocessConfig(format="end2end", max_detections=10),
    )
    diff = compare_decoded_detections(reference, candidate)

    assert diff.box_mae > 0.0
    assert diff.score_mae > 0.0
    assert diff.label_match_rate == pytest.approx(1.0)


def test_evaluate_detection_model_uses_detection_targets() -> None:
    dataloader = [
        {
            "image": torch.randn(1, 3, 32, 32),
            "boxes": [torch.tensor([[10.0, 10.0, 30.0, 30.0]], dtype=torch.float32)],
            "labels": [torch.tensor([0], dtype=torch.long)],
            "orig_size": [torch.tensor([32, 32], dtype=torch.long)],
        }
    ]

    report = evaluate_detection_model(
        FixedDetectionModel(),
        dataloader,
        postprocess=DetectionPostprocessConfig(format="yolo_raw", box_format="xyxy", max_detections=10),
        metric_config=DetectionMetricConfig(iou_thresholds=[0.5], max_detections=10),
    )

    assert report.samples == 1
    assert report.metrics["map50"] == pytest.approx(1.0)


def test_evaluate_onnx_detection_model_reports_runtime_metrics(tmp_path) -> None:
    dataloader = [
        {
            "image": torch.randn(1, 3, 32, 32),
            "boxes": [torch.tensor([[10.0, 10.0, 30.0, 30.0]], dtype=torch.float32)],
            "labels": [torch.tensor([0], dtype=torch.long)],
            "orig_size": [torch.tensor([32, 32], dtype=torch.long)],
            "image_path": ["demo.jpg"],
            "class_names": [["a"]],
        }
    ]
    model = FixedDetectionModel().eval()
    onnx_result = export_onnx(
        model,
        torch.randn(1, 3, 32, 32),
        tmp_path / "fixed_detection.onnx",
        dynamo=False,
        input_names=["input"],
        output_names=["predictions"],
    )

    report = evaluate_onnx_detection_model(
        str(onnx_result.path),
        dataloader,
        postprocess=DetectionPostprocessConfig(format="yolo_raw", box_format="xyxy", max_detections=10),
        metric_config=DetectionMetricConfig(iou_thresholds=[0.5], max_detections=10),
        input_names=["input"],
        reference_model=model,
        benchmark_warmup=0,
        benchmark_iterations=1,
    )

    assert report.runtime == "onnxruntime"
    assert report.samples == 1
    assert report.metrics["map50"] == pytest.approx(1.0)
    assert report.raw_output_diff is not None
    assert report.raw_output_diff.allclose is True
    assert report.decoded_diff is not None
    assert report.decoded_diff.label_match_rate == pytest.approx(1.0)
    assert report.latency is not None
    assert report.latency["iterations"] == 1


def test_evaluate_onnx_detection_model_casts_fp16_inputs(tmp_path) -> None:
    dataloader = [
        {
            "image": torch.randn(1, 3, 32, 32),
            "boxes": [torch.tensor([[10.0, 10.0, 30.0, 30.0]], dtype=torch.float32)],
            "labels": [torch.tensor([0], dtype=torch.long)],
            "orig_size": [torch.tensor([32, 32], dtype=torch.long)],
        }
    ]
    model = FixedDetectionModel().eval()
    fp32 = export_onnx(
        model,
        torch.randn(1, 3, 32, 32),
        tmp_path / "fixed_detection_fp32.onnx",
        dynamo=False,
        input_names=["input"],
        output_names=["predictions"],
    )
    fp16 = convert_onnx_to_fp16(
        fp32.path,
        tmp_path / "fixed_detection_fp16.onnx",
        keep_io_types=False,
    )

    report = evaluate_onnx_detection_model(
        str(fp16.path),
        dataloader,
        postprocess=DetectionPostprocessConfig(
            format="yolo_raw",
            box_format="xyxy",
            max_detections=10,
        ),
        metric_config=DetectionMetricConfig(iou_thresholds=[0.5], max_detections=10),
        input_names=["input"],
        reference_model=model,
        benchmark_warmup=0,
        benchmark_iterations=1,
    )

    assert report.runtime == "onnxruntime"
    assert report.metrics["map50"] == pytest.approx(1.0)
    assert report.raw_output_diff is not None
    assert report.raw_output_diff.candidate_summary is not None
    assert report.raw_output_diff.candidate_summary.dtype == "torch.float16"
    assert report.latency is not None
    assert report.latency["iterations"] == 1


def test_evaluate_detection_runtime_model_reports_runtime_metrics() -> None:
    dataloader = [
        {
            "image": torch.randn(1, 3, 32, 32),
            "boxes": [torch.tensor([[10.0, 10.0, 30.0, 30.0]], dtype=torch.float32)],
            "labels": [torch.tensor([0], dtype=torch.long)],
            "orig_size": [torch.tensor([32, 32], dtype=torch.long)],
        }
    ]
    model = FixedDetectionModel().eval()

    report = evaluate_detection_runtime_model(
        model,
        dataloader,
        postprocess=DetectionPostprocessConfig(format="yolo_raw", box_format="xyxy", max_detections=10),
        metric_config=DetectionMetricConfig(iou_thresholds=[0.5], max_detections=10),
        reference_model=model,
        benchmark_warmup=0,
        benchmark_iterations=1,
    )

    assert report.runtime == "pytorch"
    assert report.samples == 1
    assert report.metrics["map50"] == pytest.approx(1.0)
    assert report.raw_output_diff is not None
    assert report.raw_output_diff.allclose is True
    assert report.decoded_diff is not None
    assert report.decoded_diff.label_match_rate == pytest.approx(1.0)
    assert report.latency is not None
    assert report.latency["iterations"] == 1


def test_report_writers_create_json_csv_and_markdown(tmp_path) -> None:
    json_path = write_json_report(
        {"metrics": {"top1": 1.0}},
        tmp_path / "reports" / "metrics.json",
    )
    csv_path = write_csv_report(
        [
            {"name": "top1", "value": 1.0},
            {"name": "latency", "value": 2.5, "unit": "ms"},
        ],
        tmp_path / "reports" / "metrics.csv",
    )
    markdown_path = write_markdown_report(
        "XQT Report",
        {"Baseline": {"top1": 1.0}},
        tmp_path / "reports" / "metrics.md",
    )

    assert '"top1": 1.0' in json_path.read_text(encoding="utf-8")
    assert "name,value,unit" in csv_path.read_text(encoding="utf-8")
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "# XQT Report" in markdown
    assert "| top1 | 1.0 |" in markdown
