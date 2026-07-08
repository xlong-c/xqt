from __future__ import annotations

import torch
from torch import nn

from xqt.operator_opt.patterns import (
    scan_export_candidates,
    scan_fx_candidates,
    summarize_candidate_report,
)


class _FakeWanRMSNorm(nn.Module):
    def __init__(self, dim: int = 8, *, eps: float = 1e-6) -> None:
        super().__init__()
        self.scale = float(dim) ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.gamma.to(device=x.device, dtype=x.dtype)
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * weight * self.scale


def test_scan_fx_candidates_finds_standalone_rmsnorm() -> None:
    module = _FakeWanRMSNorm().eval()
    x = torch.randn(2, 8)

    candidates = scan_fx_candidates(module, x)

    patterns = [candidate.pattern for candidate in candidates]
    assert "rmsnorm" in patterns
    rmsnorm = next(candidate for candidate in candidates if candidate.pattern == "rmsnorm")
    assert rmsnorm.source == "fx"
    assert rmsnorm.recommended_backend == "triton"
    assert rmsnorm.estimated_kernel_count >= 7


def test_scan_export_candidates_finds_standalone_rmsnorm() -> None:
    if not hasattr(torch, "export"):
        raise RuntimeError("torch.export is not available in the current PyTorch build")
    module = _FakeWanRMSNorm().eval()
    x = torch.randn(2, 8)

    candidates = scan_export_candidates(module, x)

    patterns = [candidate.pattern for candidate in candidates]
    assert "rmsnorm" in patterns
    rmsnorm = next(candidate for candidate in candidates if candidate.pattern == "rmsnorm")
    assert rmsnorm.source == "torch_export"
    assert rmsnorm.recommended_backend == "triton"
    assert rmsnorm.estimated_kernel_count >= 7


def test_summarize_candidate_report_includes_rmsnorm_pattern() -> None:
    module = _FakeWanRMSNorm().eval()
    x = torch.randn(2, 8)

    summary = summarize_candidate_report(scan_fx_candidates(module, x))

    assert summary["candidate_count"] >= 1
    assert "rmsnorm" in summary["patterns"]
    assert "triton" in summary["recommended_backends"]
