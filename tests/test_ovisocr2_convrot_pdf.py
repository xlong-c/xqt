"""Automated tests for OvisOCR2 rotation quantization on PDF document workloads."""

from __future__ import annotations

from pathlib import Path
import pytest
import torch

from xqt.model.ovisocr2 import (
    materialize_ovisocr2_convrot_int8_runtime,
    quantize_ovisocr2_convrot_int8,
)
from scripts.verify_ovisocr2_convrot_pdf_speedup import (
    OvisOcr2LayerProxy,
    smart_resize,
)


_REPO_ROOT = Path(__file__).resolve().parents[2]
_PDF_PATH = (
    _REPO_ROOT
    / "others"
    / "Quality-Based_rPPG_Compensation_With_Temporal_Difference_Transformer_for_Camera-Based_Driver_Monitoring(科研通-ablesci.com).pdf"
)


def test_ovisocr2_pdf_file_exists() -> None:
    assert _PDF_PATH.exists(), f"Target PDF not found at {_PDF_PATH}"
    assert _PDF_PATH.stat().st_size > 0


def test_ovisocr2_smart_resize_page_geometry() -> None:
    # Academic paper page at 150 DPI: 1275 x 1650
    h, w = 1650, 1275
    rh, rw = smart_resize(h, w, factor=32)
    assert rh % 32 == 0
    assert rw % 32 == 0
    assert rh == 1664
    assert rw == 1280
    visual_tokens = (rh // 32) * (rw // 32)
    assert visual_tokens == 2080
    prefill_tokens = visual_tokens + 80  # prompt tokens
    assert prefill_tokens == 2160


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_ovisocr2_convrot_prefill_numerical_fidelity() -> None:
    torch.manual_seed(20260906)
    layer = OvisOcr2LayerProxy().to(device="cuda", dtype=torch.bfloat16).eval()
    quantized = quantize_ovisocr2_convrot_int8(
        layer,
        policy={"min_parameters": 0},
        activation_scale_mode="dynamic",
        rot_size=256,
        engine="auto",
        min_int8_rows=256,
        inplace=False,
    )
    runtime = materialize_ovisocr2_convrot_int8_runtime(quantized.model, inplace=False).eval()

    # Test M=2160 (full page prefill token count)
    m = 2160
    x = torch.randn(m, layer.hidden_size, device="cuda", dtype=torch.bfloat16)

    with torch.inference_mode():
        ref_up = layer.llm.up_proj(x)
        cand_up = runtime.llm.up_proj(x)

        cos_sim = torch.nn.functional.cosine_similarity(
            ref_up.float().flatten(), cand_up.float().flatten(), dim=0
        ).item()
        assert cos_sim > 0.9999, f"Expected high cosine similarity, got {cos_sim}"

        exec_meta = runtime.llm.up_proj.execution_metadata()
        assert exec_meta["engine"] == "native_convrot_w8a8_sm89"
        assert exec_meta["true_int8_mma"] is True


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_ovisocr2_convrot_small_m_guard_taken() -> None:
    layer = OvisOcr2LayerProxy().to(device="cuda", dtype=torch.bfloat16).eval()
    quantized = quantize_ovisocr2_convrot_int8(
        layer,
        policy={"min_parameters": 0},
        rot_size=256,
        engine="auto",
        min_int8_rows=256,
        inplace=False,
    )
    runtime = materialize_ovisocr2_convrot_int8_runtime(quantized.model, inplace=False).eval()

    # Test M=1 (decode step)
    x = torch.randn(1, layer.hidden_size, device="cuda", dtype=torch.bfloat16)
    with torch.inference_mode():
        _ = runtime.llm.q_proj(x)
        exec_meta = runtime.llm.q_proj.execution_metadata()
        assert exec_meta["engine"] == "float_fallback"
        assert exec_meta["implementation"] == "bf16_dense_small_m_fallback"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_ovisocr2_convrot_shared_activation_group_fidelity() -> None:
    from xqt.model.ovisocr2 import make_shared_convrot_w8a8_group

    layer = OvisOcr2LayerProxy().to(device="cuda", dtype=torch.bfloat16).eval()
    quantized = quantize_ovisocr2_convrot_int8(
        layer,
        policy={"min_parameters": 0},
        rot_size=256,
        engine="auto",
        min_int8_rows=256,
        inplace=False,
    )
    runtime = materialize_ovisocr2_convrot_int8_runtime(quantized.model, inplace=False).eval()

    shared_qkv = make_shared_convrot_w8a8_group(
        [runtime.llm.q_proj, runtime.llm.k_proj, runtime.llm.v_proj],
        rot_size=256,
    )

    m = 2160
    x = torch.randn(m, layer.hidden_size, device="cuda", dtype=torch.bfloat16)

    with torch.inference_mode():
        q_ind = runtime.llm.q_proj(x)
        k_ind = runtime.llm.k_proj(x)
        v_ind = runtime.llm.v_proj(x)
        q_sh, k_sh, v_sh = shared_qkv(x)

    assert torch.equal(q_ind, q_sh), f"Q diff: {(q_ind - q_sh).abs().max()}"
    assert torch.equal(k_ind, k_sh), f"K diff: {(k_ind - k_sh).abs().max()}"
    assert torch.equal(v_ind, v_sh), f"V diff: {(v_ind - v_sh).abs().max()}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_ovisocr2_layer_proxy_shared_forward() -> None:
    layer = OvisOcr2LayerProxy().to(device="cuda", dtype=torch.bfloat16).eval()
    quantized = quantize_ovisocr2_convrot_int8(
        layer,
        policy={"min_parameters": 0},
        rot_size=256,
        engine="auto",
        min_int8_rows=256,
        inplace=False,
    )
    runtime = materialize_ovisocr2_convrot_int8_runtime(quantized.model, inplace=False).eval()
    runtime.setup_shared_groups(rot_size=256)

    m = 2160
    x = torch.randn(m, layer.hidden_size, device="cuda", dtype=torch.bfloat16)
    x_mlp = torch.randn(m, layer.intermediate_size, device="cuda", dtype=torch.bfloat16)

    with torch.inference_mode():
        q1, k1, v1, o1 = runtime.forward_attention_projections(x, use_shared=False)
        q2, k2, v2, o2 = runtime.forward_attention_projections(x, use_shared=True)

        assert torch.equal(q1, q2)
        assert torch.equal(k1, k2)
        assert torch.equal(v1, v2)
        assert torch.equal(o1, o2)

        g1, u1, d1 = runtime.forward_mlp_projections(x, x_mlp, use_shared=False)
        g2, u2, d2 = runtime.forward_mlp_projections(x, x_mlp, use_shared=True)

        assert torch.equal(g1, g2)
        assert torch.equal(u1, u2)
        assert torch.equal(d1, d2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_ovisocr2_convrot_fused_qkv_fidelity() -> None:
    from xqt.model.ovisocr2 import make_fused_convrot_w8a8_group

    layer = OvisOcr2LayerProxy().to(device="cuda", dtype=torch.bfloat16).eval()
    quantized = quantize_ovisocr2_convrot_int8(
        layer,
        policy={"min_parameters": 0},
        rot_size=256,
        engine="auto",
        min_int8_rows=256,
        inplace=False,
    )
    runtime = materialize_ovisocr2_convrot_int8_runtime(quantized.model, inplace=False).eval()

    fused_qkv = make_fused_convrot_w8a8_group(
        [runtime.llm.q_proj, runtime.llm.k_proj, runtime.llm.v_proj],
        rot_size=256,
    )

    m = 2160
    x = torch.randn(m, layer.hidden_size, device="cuda", dtype=torch.bfloat16)

    with torch.inference_mode():
        q_ind = runtime.llm.q_proj(x)
        k_ind = runtime.llm.k_proj(x)
        v_ind = runtime.llm.v_proj(x)
        q_fu, k_fu, v_fu = fused_qkv(x)

    assert torch.equal(q_ind, q_fu), f"Q diff: {(q_ind - q_fu).abs().max()}"
    assert torch.equal(k_ind, k_fu), f"K diff: {(k_ind - k_fu).abs().max()}"
    assert torch.equal(v_ind, v_fu), f"V diff: {(v_ind - v_fu).abs().max()}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_ovisocr2_convrot_fused_gate_up_fidelity() -> None:
    from xqt.model.ovisocr2 import make_fused_convrot_w8a8_group

    layer = OvisOcr2LayerProxy().to(device="cuda", dtype=torch.bfloat16).eval()
    quantized = quantize_ovisocr2_convrot_int8(
        layer,
        policy={"min_parameters": 0},
        rot_size=256,
        engine="auto",
        min_int8_rows=256,
        inplace=False,
    )
    runtime = materialize_ovisocr2_convrot_int8_runtime(quantized.model, inplace=False).eval()

    fused_gate_up = make_fused_convrot_w8a8_group(
        [runtime.llm.gate_proj, runtime.llm.up_proj],
        rot_size=256,
    )

    m = 2160
    x = torch.randn(m, layer.hidden_size, device="cuda", dtype=torch.bfloat16)

    with torch.inference_mode():
        g_ind = runtime.llm.gate_proj(x)
        u_ind = runtime.llm.up_proj(x)
        g_fu, u_fu = fused_gate_up(x)

    assert torch.equal(g_ind, g_fu), f"Gate diff: {(g_ind - g_fu).abs().max()}"
    assert torch.equal(u_ind, u_fu), f"Up diff: {(u_ind - u_fu).abs().max()}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fused_rmsnorm_hadamard_quant_parity() -> None:
    from xqt.kernels.ops._impl.cute.convrot_w8a8_sm89 import (
        allocate_convrot_w8a8_workspace,
        pack_convrot_w8a8_linear,
        quantize_rotated_activation_sm89,
    )

    m, k = 256, 1024
    torch.manual_seed(20260906)
    x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
    norm_weight = torch.randn(k, dtype=torch.bfloat16, device="cuda")

    # Reference: RMSNorm then separate quantize_rotated_activation
    normed = (x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6) * norm_weight.float()).to(torch.bfloat16)

    qw = torch.randint(-127, 127, (k, 1024), dtype=torch.int8, device="cuda")
    ws = torch.ones(1024, dtype=torch.float32, device="cuda")
    packed = pack_convrot_w8a8_linear(qw, ws, None, dtype=torch.bfloat16)

    ws_ref = allocate_convrot_w8a8_workspace(m, packed)
    quantize_rotated_activation_sm89(normed, ws_ref, rotated_input_features=k, rot_size=256)

    ws_fused = allocate_convrot_w8a8_workspace(m, packed)
    quantize_rotated_activation_sm89(x, ws_fused, rotated_input_features=k, rot_size=256, norm_weight=norm_weight, eps=1e-6)

    diff_scale = (ws_ref.activation_scales - ws_fused.activation_scales).abs().max().item()
    diff_act = (ws_ref.quantized_activation.float() - ws_fused.quantized_activation.float()).abs().max().item()

    assert diff_scale < 1e-3, f"Scale diff too large: {diff_scale}"
    assert diff_act <= 2.0, f"Quant act diff too large: {diff_act}"

    dequant_ref = ws_ref.quantized_activation.float() * ws_ref.activation_scales.unsqueeze(-1).float()
    dequant_fused = ws_fused.quantized_activation.float() * ws_fused.activation_scales.unsqueeze(-1).float()
    cos_sim = torch.nn.functional.cosine_similarity(dequant_ref.flatten(), dequant_fused.flatten(), dim=0).item()
    assert cos_sim > 0.9999, f"Cosine similarity too low: {cos_sim}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fused_swiglu_parity() -> None:
    from xqt.kernels.ops._impl.cute.convrot_w8a8_sm89 import fused_swiglu

    m, d = 32, 3584
    torch.manual_seed(20260906)
    gate_up = torch.randn(m, 2 * d, dtype=torch.bfloat16, device="cuda")
    out = fused_swiglu(gate_up)

    g, u = gate_up.chunk(2, dim=-1)
    ref = torch.nn.functional.silu(g.float()) * u.float()
    diff = (out.float() - ref).abs().max().item()

    assert diff < 0.05, f"SwiGLU diff too large: {diff}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_decode_m1_fused_group_parity() -> None:
    from xqt.model.ovisocr2 import make_fused_convrot_w8a8_group

    layer = OvisOcr2LayerProxy().to(device="cuda", dtype=torch.bfloat16).eval()
    quantized = quantize_ovisocr2_convrot_int8(
        layer,
        policy={"min_parameters": 0},
        rot_size=256,
        engine="auto",
        min_int8_rows=256,
        inplace=False,
    )
    runtime = materialize_ovisocr2_convrot_int8_runtime(quantized.model, inplace=False).eval()

    fused_gate_up = make_fused_convrot_w8a8_group(
        [runtime.llm.gate_proj, runtime.llm.up_proj],
        rot_size=256,
    )

    x_m1 = torch.randn(1, layer.hidden_size, device="cuda", dtype=torch.bfloat16)
    with torch.inference_mode():
        g_m1, u_m1 = fused_gate_up(x_m1)
        swiglu_out = fused_gate_up.forward_swiglu(x_m1)

    assert g_m1.shape == (1, layer.intermediate_size)
    assert u_m1.shape == (1, layer.intermediate_size)
    assert swiglu_out.shape == (1, layer.intermediate_size)

    ref_swiglu = torch.nn.functional.silu(g_m1.float()) * u_m1.float()
    diff = (swiglu_out.float() - ref_swiglu).abs().max().item()
    assert diff < 0.05, f"M=1 SwiGLU diff too large: {diff}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_graph_runner_fidelity() -> None:
    from xqt.runtime.cuda_graph import CUDAGraphBlockRunner

    class ToyBlock(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(1024, 1024, bias=False, dtype=torch.bfloat16, device="cuda")

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.nn.functional.silu(self.linear(x))

    toy = ToyBlock().eval()
    runner = CUDAGraphBlockRunner(toy, warmup_steps=2)

    x = torch.randn(16, 1024, dtype=torch.bfloat16, device="cuda")
    with torch.inference_mode():
        ref = toy(x)
        out1 = runner(x)
        out2 = runner(x)

    assert runner.last_execution_report["mode"] == "cuda_graph"
    assert torch.allclose(ref, out2, atol=1e-3, rtol=1e-3)



