from pathlib import Path

import pytest
import torch

from xqt.kernels.ops._impl.engines.tilelang import get_tilelang_kernel_spec
from xqt.kernels.ops._impl.tilelang.int8_mma import (
    build_tilelang_int8_mma_kernel,
    int8_linear_static_activation_tilelang,
    int8_linear_tilelang,
    int8_mma_reference,
    int8_mma_tilelang,
    static_activation_quantize_tilelang,
)


def test_tilelang_registry_exposes_true_int8_mma() -> None:
    spec = get_tilelang_kernel_spec("int8_mma")

    assert spec.metadata["quantization_nature"] == "true"
    assert spec.metadata["accumulation"] == "int32"
    assert "s8.s8" in spec.metadata["mma_instruction"]

    linear_spec = get_tilelang_kernel_spec("int8_linear")
    fused_spec = get_tilelang_kernel_spec("int8_linear_static_activation")
    assert linear_spec.metadata["quantization_nature"] == "true"
    assert fused_spec.metadata["quantization_nature"] == "true_with_fused_activation_quant"


def test_int8_mma_reference_matches_int32_matmul_cpu() -> None:
    torch.manual_seed(0)
    a = torch.randint(-8, 8, (8, 16), dtype=torch.int8)
    b = torch.randint(-8, 8, (16, 12), dtype=torch.int8)

    output = int8_mma_reference(a, b)
    reference = a.to(torch.int32) @ b.to(torch.int32)

    assert output.dtype == torch.int32
    assert torch.equal(output, reference)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for TileLang")
def test_tilelang_int8_mma_matches_torch_int_mm_cuda(tmp_path: Path) -> None:
    torch.manual_seed(1)
    a = torch.randint(-16, 16, (128, 128), device="cuda", dtype=torch.int8)
    b = torch.randint(-16, 16, (128, 128), device="cuda", dtype=torch.int8)

    output = int8_mma_tilelang(a, b)
    reference = torch._int_mm(a, b)
    torch.cuda.synchronize()

    assert output.dtype == torch.int32
    assert torch.equal(output, reference)

    kernel = build_tilelang_int8_mma_kernel(128, 128, 128, target_arch="sm_89")
    ptx_path = tmp_path / "int8_mma.ptx"
    kernel.export_ptx(str(ptx_path))
    ptx = ptx_path.read_text(errors="ignore")
    assert "mma.sync" in ptx
    assert "m16n8k32" in ptx
    assert "s32.s8.s8.s32" in ptx


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for TileLang")
def test_tilelang_static_activation_quantize_matches_torch_cuda() -> None:
    torch.manual_seed(2)
    inputs = torch.randn(9, 64, device="cuda", dtype=torch.bfloat16)
    scale = (inputs.float().abs().amax().clamp_min(1e-6) / 127.0).reshape(())

    output = static_activation_quantize_tilelang(inputs, scale)
    reference = torch.round(inputs.float() / scale).clamp(-127, 127).to(torch.int8)
    torch.cuda.synchronize()

    assert output.dtype == torch.int8
    assert torch.equal(output, reference)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for TileLang")
def test_tilelang_fused_static_activation_linear_matches_two_stage_cuda() -> None:
    torch.manual_seed(3)
    inputs = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
    max_abs = weight.float().abs().amax(dim=1, keepdim=True).clamp_min(1e-6)
    weight_scale = (max_abs / 127.0).reshape(-1)
    qweight_t = torch.round(weight.float() / weight_scale.reshape(-1, 1)).clamp(-127, 127).to(torch.int8).t().contiguous()
    activation_scale = (inputs.float().abs().amax().clamp_min(1e-6) / 127.0).reshape(())
    qactivation = static_activation_quantize_tilelang(inputs, activation_scale)

    fused = int8_linear_static_activation_tilelang(
        inputs,
        qweight_t,
        activation_scale,
        weight_scale,
        output_dtype=torch.bfloat16,
    )
    staged = int8_linear_tilelang(
        qactivation,
        qweight_t,
        activation_scale,
        weight_scale,
        output_dtype=torch.bfloat16,
    )
    torch.cuda.synchronize()

    assert torch.equal(fused, staged)
