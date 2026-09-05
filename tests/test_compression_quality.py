"""Tests for end-to-end model compression quality assessment."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from xqt.compression.quant.quality import (
    ModelCompressionQualityReport,
    evaluate_model_compression_quality,
)
from xqt.compression.quant.quantizers.awq_gptq_weight_only import quantize_with_awq_weight_only


class _ToyClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(32, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.relu(self.fc1(x)))


def test_evaluate_model_compression_quality_numerical() -> None:
    torch.manual_seed(42)
    float_model = _ToyClassifier().eval()

    quantized = quantize_with_awq_weight_only(
        float_model,
        policy={"include_module_types": ["Linear"], "bits": 4, "group_size": 16},
        strategy="w4a16_int4",
        inplace=False,
    )
    quant_model = quantized.model.eval()

    dataloader = [torch.randn(4, 32) for _ in range(5)]

    report = evaluate_model_compression_quality(
        float_model,
        quant_model,
        dataloader,
        sample_limit=5,
        measure_latency=True,
        latency_warmup=2,
        latency_repeat=5,
    )

    assert isinstance(report, ModelCompressionQualityReport)
    assert report.sample_count == 20
    assert 0.90 <= report.mean_cosine_similarity <= 1.0001
    assert report.max_abs_error >= 0.0
    assert report.mean_abs_error >= 0.0
    assert report.mean_squared_error >= 0.0
    assert report.root_mean_squared_error >= 0.0
    assert report.relative_l2_error >= 0.0

    # Top-1 agreement on 10-class output
    assert report.top_1_agreement_rate is not None
    assert 0.0 <= report.top_1_agreement_rate <= 1.0

    # Size reduction
    assert report.float_size_bytes > 0
    assert report.quant_size_bytes > 0
    assert report.compression_ratio > 1.0

    # Latency
    assert report.median_latency_ms_float is not None
    assert report.median_latency_ms_quant is not None
    assert report.speedup_ratio is not None

    # Gate verification
    assert report.passes_gate(min_cosine=0.85, max_mae=5.0)

    # Dictionary and markdown conversion
    data = report.to_dict()
    assert "mean_cosine_similarity" in data
    assert "compression_ratio" in data

    md = report.to_markdown()
    assert "Model Compression Quality Assessment Report" in md
    assert "Storage Size" in md
    assert "Mean Cosine Similarity" in md
