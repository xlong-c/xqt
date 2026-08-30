"""W3: offline prefill/decode phase latency."""

from __future__ import annotations

import torch
from torch import nn

from xqt.kernels.wrappers.bench import benchmark_prefill_decode, merge_phase_into_metrics


def test_benchmark_prefill_decode_offline_estimate() -> None:
    model = nn.Linear(32, 16).eval()
    prefill_x = torch.randn(64, 32)
    decode_x = torch.randn(1, 32)

    def prefill() -> torch.Tensor:
        return model(prefill_x)

    def decode() -> torch.Tensor:
        return model(decode_x)

    report = benchmark_prefill_decode(
        prefill,
        decode,
        warmup=0,
        iterations=2,
        sync_cuda=False,
    )
    assert report.offline_estimate is True
    assert report.prefill.mean_ms >= 0.0
    assert report.decode.mean_ms >= 0.0
    payload = report.to_dict()
    assert payload["offline_estimate"] is True
    assert "prefill_mean_ms" in payload
    merged = merge_phase_into_metrics({}, report)
    assert merged["ttft_ms"] == report.prefill.mean_ms
    assert merged["tpot_ms"] == report.decode.mean_ms
