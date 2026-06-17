import pytest
import torch
from torch import nn

from xqt.eval.accuracy import evaluate_pytorch_model, topk_accuracy
from xqt.eval.compare import compare_tensors
from xqt.eval.report import (
    flatten_metrics,
    write_csv_report,
    write_json_report,
    write_markdown_report,
)


class FixedLogitModel(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def test_compare_tensors_reports_common_diff_metrics() -> None:
    reference = torch.tensor([1.0, 2.0, 3.0])
    candidate = torch.tensor([1.0, 2.001, 2.999])

    diff = compare_tensors(reference, candidate, atol=0.01, rtol=0.01)

    assert diff.allclose is True
    assert diff.max_abs == pytest.approx(0.001, abs=1e-6)
    assert diff.mean_abs > 0.0
    assert diff.mean_squared > 0.0
    assert diff.cosine_similarity == pytest.approx(1.0, abs=1e-5)
    assert diff.to_dict()["allclose"] is True


def test_compare_tensors_handles_empty_and_zero_vectors() -> None:
    empty = compare_tensors(torch.empty(0), torch.empty(0))
    zeros = compare_tensors(torch.zeros(3), torch.zeros(3))

    assert empty.max_abs == 0.0
    assert empty.cosine_similarity is None
    assert zeros.cosine_similarity is None


def test_compare_tensors_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="Tensor shapes differ"):
        compare_tensors(torch.zeros(1, 2), torch.zeros(2, 1))


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
