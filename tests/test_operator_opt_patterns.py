import torch
from torch import nn

from xqt.operator_opt.patterns import (
    scan_export_candidates,
    scan_fx_candidates,
    summarize_candidate_report,
)


class SwiGLUToy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin1 = nn.Linear(8, 16)
        self.lin2 = nn.Linear(8, 16)
        self.out = nn.Linear(16, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.lin1(x)
        b = self.lin2(x)
        return self.out(torch.nn.functional.silu(a) * b)


class AttentionToy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(8, 2, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.attn(x, x, x, need_weights=False)
        return output


class DequantGemmToy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("qweight", torch.randint(-4, 4, (8, 8), dtype=torch.int8))
        self.register_buffer("scale", torch.tensor(0.125))
        self.bias = nn.Parameter(torch.randn(8))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.qweight.to(torch.float32) * self.scale
        return torch.nn.functional.linear(x, weight, self.bias)


class QDQEpilogueToy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("qweight", torch.randint(-4, 4, (8, 8), dtype=torch.int8))
        self.register_buffer("scale", torch.tensor(0.125))
        self.register_buffer("out_scale", torch.tensor(0.25))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.qweight.to(torch.float32) * self.scale
        output = torch.nn.functional.linear(x, weight)
        q = torch.clamp(torch.round(output / self.out_scale), -128, 127)
        dq = q * self.out_scale
        return torch.nn.functional.gelu(dq + 1.0)


class DequantGemmEpilogueToy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("qweight", torch.randint(-4, 4, (8, 8), dtype=torch.int8))
        self.register_buffer("scale", torch.tensor(0.125))
        self.bias = nn.Parameter(torch.randn(8))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.qweight.to(torch.float32) * self.scale
        output = torch.nn.functional.linear(x, weight)
        return torch.nn.functional.gelu(output + self.bias)


class RMSNormResidualToy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(8))

    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        merged = x + residual
        variance = merged.pow(2).mean(dim=-1, keepdim=True)
        return merged * torch.rsqrt(variance + 1e-6) * self.weight


class RoPEToy(nn.Module):
    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        even = x[..., 0::2]
        odd = x[..., 1::2]
        rotated_even = even * cos - odd * sin
        rotated_odd = even * sin + odd * cos
        return torch.stack((rotated_even, rotated_odd), dim=-1).flatten(-2)


def test_scan_fx_candidates_detects_swiglu_pattern() -> None:
    model = SwiGLUToy().eval()
    candidates = scan_fx_candidates(model, torch.randn(2, 8))

    assert any(candidate.pattern == "swiglu" for candidate in candidates)
    swiglu = next(candidate for candidate in candidates if candidate.pattern == "swiglu")
    assert swiglu.source == "fx"
    assert swiglu.recommended_backend == "triton"
    assert swiglu.estimated_kernel_count == 2


def test_scan_fx_candidates_detects_rmsnorm_residual_and_rope_patterns() -> None:
    rms_model = RMSNormResidualToy().eval()
    rms_candidates = scan_fx_candidates(rms_model, (torch.randn(2, 8), torch.randn(2, 8)))

    assert any(candidate.pattern == "rmsnorm_residual" for candidate in rms_candidates)
    rmsnorm = next(
        candidate for candidate in rms_candidates if candidate.pattern == "rmsnorm_residual"
    )
    assert rmsnorm.source == "fx"
    assert rmsnorm.recommended_backend == "triton"
    assert rmsnorm.estimated_kernel_count >= 7

    rope_model = RoPEToy().eval()
    rope_candidates = scan_fx_candidates(
        rope_model,
        (torch.randn(2, 8), torch.randn(2, 4), torch.randn(2, 4)),
    )

    assert any(candidate.pattern == "rope" for candidate in rope_candidates)
    rope = next(candidate for candidate in rope_candidates if candidate.pattern == "rope")
    assert rope.source == "fx"
    assert rope.recommended_backend == "triton"
    assert rope.estimated_kernel_count >= 10


def test_scan_export_candidates_detects_attention_pattern() -> None:
    model = AttentionToy().eval()
    candidates = scan_export_candidates(model, torch.randn(2, 4, 8))

    assert any(candidate.pattern == "attention" for candidate in candidates)
    attention = next(candidate for candidate in candidates if candidate.pattern == "attention")
    assert attention.source == "torch_export"
    assert attention.recommended_backend == "tilelang"
    assert attention.estimated_kernel_count == 1


def test_scan_export_candidates_detects_rmsnorm_residual_and_rope_patterns() -> None:
    rms_model = RMSNormResidualToy().eval()
    rms_candidates = scan_export_candidates(
        rms_model,
        (torch.randn(2, 8), torch.randn(2, 8)),
    )

    assert any(candidate.pattern == "rmsnorm_residual" for candidate in rms_candidates)
    rmsnorm = next(
        candidate for candidate in rms_candidates if candidate.pattern == "rmsnorm_residual"
    )
    assert rmsnorm.source == "torch_export"
    assert rmsnorm.recommended_backend == "triton"
    assert rmsnorm.estimated_kernel_count >= 7

    rope_model = RoPEToy().eval()
    rope_candidates = scan_export_candidates(
        rope_model,
        (torch.randn(2, 8), torch.randn(2, 4), torch.randn(2, 4)),
    )

    assert any(candidate.pattern == "rope" for candidate in rope_candidates)
    rope = next(candidate for candidate in rope_candidates if candidate.pattern == "rope")
    assert rope.source == "torch_export"
    assert rope.recommended_backend == "triton"
    assert rope.estimated_kernel_count >= 10


def test_scan_export_candidates_detects_dequant_gemm_and_qdq_epilogue() -> None:
    dequant_candidates = scan_export_candidates(DequantGemmToy().eval(), torch.randn(2, 8))
    assert any(candidate.pattern == "dequant_gemm" for candidate in dequant_candidates)
    dequant = next(
        candidate for candidate in dequant_candidates if candidate.pattern == "dequant_gemm"
    )
    assert dequant.source == "torch_export"
    assert dequant.recommended_backend == "tilelang"

    qdq_candidates = scan_export_candidates(QDQEpilogueToy().eval(), torch.randn(2, 8))
    assert any(candidate.pattern == "dequant_gemm" for candidate in qdq_candidates)
    assert any(candidate.pattern == "qdq_epilogue" for candidate in qdq_candidates)
    qdq = next(candidate for candidate in qdq_candidates if candidate.pattern == "qdq_epilogue")
    assert qdq.source == "torch_export"
    assert qdq.recommended_backend == "deployment_backend"


def test_scan_export_candidates_detects_dequant_gemm_epilogue() -> None:
    candidates = scan_export_candidates(DequantGemmEpilogueToy().eval(), torch.randn(2, 8))

    assert any(candidate.pattern == "dequant_gemm" for candidate in candidates)
    assert any(candidate.pattern == "dequant_gemm_epilogue" for candidate in candidates)
    epilogue = next(
        candidate for candidate in candidates if candidate.pattern == "dequant_gemm_epilogue"
    )
    assert epilogue.source == "torch_export"
    assert epilogue.recommended_backend == "tilelang"
    assert epilogue.estimated_kernel_count >= 4


def test_scan_export_candidates_detects_torchao_weight_only_epilogues() -> None:
    from xqt.quant.torchao_backend import quantize_with_torchao

    int8_model = nn.Sequential(nn.Linear(8, 8), nn.GELU(), nn.Linear(8, 8)).eval()
    quantize_with_torchao(
        int8_model,
        policy={"include_module_types": ["Linear"]},
        strategy="weight_only_int8",
    )
    int8_candidates = scan_export_candidates(int8_model, torch.randn(2, 8))
    assert any(
        candidate.pattern == "weight_only_matmul_epilogue"
        for candidate in int8_candidates
    )

    fp8_model = nn.Sequential(nn.Linear(8, 8), nn.GELU(), nn.Linear(8, 8)).eval()
    quantize_with_torchao(
        fp8_model,
        policy={"include_module_types": ["Linear"]},
        strategy="fp8_weight_only",
    )
    fp8_candidates = scan_export_candidates(fp8_model, torch.randn(2, 8))
    assert any(
        candidate.pattern == "fp8_scale_cast_matmul_epilogue"
        for candidate in fp8_candidates
    )


def test_summarize_candidate_report_stays_advisory_only() -> None:
    model = SwiGLUToy().eval()
    candidates = scan_fx_candidates(model, torch.randn(2, 8))
    report = summarize_candidate_report(candidates)

    assert report["candidate_count"] >= 1
    assert "swiglu" in report["patterns"]
    assert report["recommended_backends"]
    assert "applied" not in report["candidates"][0]
