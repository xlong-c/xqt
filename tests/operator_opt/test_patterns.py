from __future__ import annotations

import torch
from torch import nn

from xqt.operator_opt.patterns import (
    operator_pattern_coverage_report,
    scan_candidate_report,
    scan_export_candidates,
    scan_fx_candidates,
    scan_operator_candidate_reports,
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


class _LinearGeluBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(8, 8)
        self.bias = nn.Parameter(torch.zeros(8))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.gelu(self.proj(x) + self.bias)


class _FakeRoPE(nn.Module):
    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        even = x[..., ::2]
        odd = x[..., 1::2]
        rotated_even = even * cos - odd * sin
        rotated_odd = even * sin + odd * cos
        return torch.stack((rotated_even, rotated_odd), dim=-1).flatten(-2)


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


def test_scan_fx_candidates_finds_linear_gemm_and_activation_epilogue() -> None:
    module = _LinearGeluBlock().eval()
    x = torch.randn(2, 8)

    candidates = scan_fx_candidates(module, x)
    patterns = [candidate.pattern for candidate in candidates]

    assert "linear_gemm" in patterns
    assert "bias_gelu" in patterns
    linear = next(candidate for candidate in candidates if candidate.pattern == "linear_gemm")
    assert linear.recommended_backend == "torch_compile"


def test_scan_fx_candidates_finds_rope_pattern() -> None:
    module = _FakeRoPE().eval()
    x = torch.randn(2, 4, 8)
    cos = torch.randn(2, 4, 4)
    sin = torch.randn(2, 4, 4)

    candidates = scan_fx_candidates(module, (x, cos, sin))

    rope = next(candidate for candidate in candidates if candidate.pattern == "rope")
    assert rope.source == "fx"
    assert rope.recommended_backend == "triton"
    assert rope.estimated_kernel_count >= 8


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


def test_scan_export_candidates_finds_linear_gemm_and_rope() -> None:
    if not hasattr(torch, "export"):
        raise RuntimeError("torch.export is not available in the current PyTorch build")
    linear_module = _LinearGeluBlock().eval()
    rope_module = _FakeRoPE().eval()
    x = torch.randn(2, 8)
    rope_x = torch.randn(2, 4, 8)
    cos = torch.randn(2, 4, 4)
    sin = torch.randn(2, 4, 4)

    linear_patterns = [
        candidate.pattern for candidate in scan_export_candidates(linear_module, x)
    ]
    rope_patterns = [
        candidate.pattern
        for candidate in scan_export_candidates(rope_module, (rope_x, cos, sin))
    ]

    assert "linear_gemm" in linear_patterns
    assert "bias_gelu" in linear_patterns
    assert "rope" in rope_patterns


def test_summarize_candidate_report_includes_rmsnorm_pattern() -> None:
    module = _FakeWanRMSNorm().eval()
    x = torch.randn(2, 8)

    summary = summarize_candidate_report(scan_fx_candidates(module, x))

    assert summary["candidate_count"] >= 1
    assert "rmsnorm" in summary["patterns"]
    assert summary["pattern_counts"]["rmsnorm"] >= 1
    assert summary["source_counts"]["fx"] >= 1
    assert "triton" in summary["recommended_backends"]
    assert summary["shape_signatures"][0]["shape"]
    assert "torch.float32" in summary["dtypes"]


def test_operator_pattern_coverage_report_tracks_high_frequency_groups() -> None:
    module = _LinearGeluBlock().eval()
    x = torch.randn(2, 8)

    coverage = operator_pattern_coverage_report(scan_fx_candidates(module, x))

    assert "linear_gemm" in coverage["covered_groups"]
    assert "activation_epilogue" in coverage["covered_groups"]
    assert "attention" in coverage["missing_groups"]
    assert coverage["groups"]["linear_gemm"]["candidate_count"] >= 1


def test_scan_operator_candidate_reports_merges_fx_and_export_sources() -> None:
    if not hasattr(torch, "export"):
        raise RuntimeError("torch.export is not available in the current PyTorch build")
    module = _FakeWanRMSNorm().eval()
    x = torch.randn(2, 8)

    report = scan_operator_candidate_reports(module, x)

    assert report["status"] == "ok"
    assert report["error"] is None
    assert report["source_reports"]["fx"]["status"] == "ok"
    assert report["source_reports"]["torch_export"]["status"] == "ok"
    assert report["fx"]["candidate_count"] >= 1
    assert report["torch_export"]["candidate_count"] >= 1
    assert report["source_counts"]["fx"] >= 1
    assert report["source_counts"]["torch_export"] >= 1
    assert "rmsnorm" in report["pattern_counts"]
    assert report["coverage"]["groups"]["norm"]["covered"] is True


def test_scan_candidate_report_preserves_scanner_failure() -> None:
    module = _FakeWanRMSNorm().eval()
    x = torch.randn(2, 8)

    def failing_scanner(model: nn.Module, example_input: object) -> list[object]:
        del model, example_input
        raise RuntimeError("scanner unavailable")

    report = scan_candidate_report(failing_scanner, module, x)

    assert report["status"] == "error"
    assert report["error"] == "scanner unavailable"
    assert report["candidate_count"] == 0
    assert report["shape_signatures"] == []
