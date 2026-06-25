"""Tests for multi-precision GEMM kernels."""

import pytest
import torch

from xqt.operator_opt.backends.gemm_precision import (
    describe_gemm_precision_capability,
    gemm_with_precision,
    list_available_precisions,
)
from xqt.operator_opt.kernels.triton.gemm import (
    gemm_bf16_triton,
    gemm_fp16_triton,
    gemm_reference,
)
from xqt.operator_opt.kernels.triton.mxfp_gemm import pack_mxfp, unpack_mxfp


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def small_matrices(device):
    """Small test matrices for quick validation."""
    M, N, K = 64, 64, 64
    a = torch.randn(M, K, device=device, dtype=torch.float16)
    b = torch.randn(N, K, device=device, dtype=torch.float16)
    bias = torch.randn(N, device=device, dtype=torch.float16)
    return a, b, bias


@pytest.fixture
def medium_matrices(device):
    """Medium test matrices for performance validation."""
    M, N, K = 256, 256, 256
    a = torch.randn(M, K, device=device, dtype=torch.float16)
    b = torch.randn(N, K, device=device, dtype=torch.float16)
    bias = torch.randn(N, device=device, dtype=torch.float16)
    return a, b, bias


class TestGEMMReference:
    """Test reference GEMM implementation."""

    def test_basic_matmul(self, small_matrices):
        a, b, bias = small_matrices
        output = gemm_reference(a, b, transpose_b=True)
        expected = torch.matmul(a, b.t())
        assert torch.allclose(output, expected, rtol=1e-3, atol=1e-3)

    def test_with_bias(self, small_matrices):
        a, b, bias = small_matrices
        output = gemm_reference(a, b, bias, transpose_b=True)
        expected = torch.matmul(a, b.t()) + bias
        assert torch.allclose(output, expected, rtol=1e-3, atol=1e-3)

    def test_with_relu(self, small_matrices):
        a, b, bias = small_matrices
        output = gemm_reference(a, b, bias, activation="relu", transpose_b=True)
        expected = torch.relu(torch.matmul(a, b.t()) + bias)
        assert torch.allclose(output, expected, rtol=1e-3, atol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestTritonGEMMKernels:
    """Test Triton GEMM kernels."""

    def test_fp16_basic(self, small_matrices):
        a, b, bias = small_matrices
        output = gemm_fp16_triton(a, b, bias, transpose_b=True)
        expected = torch.matmul(a, b.t()) + bias
        assert torch.allclose(output, expected, rtol=1e-2, atol=1e-2)

    def test_fp16_activations(self, small_matrices):
        a, b, _ = small_matrices

        # ReLU
        out_relu = gemm_fp16_triton(a, b, activation="relu", transpose_b=True)
        expected_relu = torch.relu(torch.matmul(a, b.t()))
        assert torch.allclose(out_relu, expected_relu, rtol=1e-2, atol=1e-2)

        # GELU
        out_gelu = gemm_fp16_triton(a, b, activation="gelu", transpose_b=True)
        expected_gelu = torch.nn.functional.gelu(torch.matmul(a, b.t()))
        assert torch.allclose(out_gelu, expected_gelu, rtol=1e-2, atol=1e-2)

    def test_bf16_basic(self, small_matrices):
        a, b, bias = small_matrices
        a_bf16 = a.to(torch.bfloat16)
        b_bf16 = b.to(torch.bfloat16)
        bias_bf16 = bias.to(torch.bfloat16)

        output = gemm_bf16_triton(a_bf16, b_bf16, bias_bf16, transpose_b=True)
        expected = torch.matmul(a_bf16, b_bf16.t()) + bias_bf16
        assert torch.allclose(output.float(), expected.float(), rtol=5e-2, atol=5e-2)


class TestMXFPPacking:
    """Test MXFP packing/unpacking utilities."""

    def test_mxfp8_roundtrip(self, device):
        if device.type == "cpu":
            pytest.skip("MXFP requires CUDA")

        tensor = torch.randn(128, 128, device=device)
        packed, scales = pack_mxfp(tensor, precision=8, block_size=32)
        unpacked = unpack_mxfp(packed, scales, precision=8, block_size=32, original_numel=tensor.numel())
        unpacked = unpacked.reshape(tensor.shape)

        # Check approximate equality (quantization introduces error)
        # MXFP8 has 7 mantissa bits, expect quantization step of ~1/127
        # Typical relative error: 1/127 ≈ 0.8%, use 5% margin for safety
        assert torch.allclose(unpacked, tensor, rtol=5e-2, atol=5e-2)

    def test_mxfp4_roundtrip(self, device):
        if device.type == "cpu":
            pytest.skip("MXFP requires CUDA")

        tensor = torch.randn(128, 128, device=device)
        packed, scales = pack_mxfp(tensor, precision=4, block_size=32)
        unpacked = unpack_mxfp(packed, scales, precision=4, block_size=32, original_numel=tensor.numel())
        unpacked = unpacked.reshape(tensor.shape)

        # MXFP4 has 3 mantissa bits, quantization step ~1/7 ≈ 14%
        # Allow 25% margin for safety
        assert unpacked.shape == tensor.shape
        assert torch.allclose(unpacked, tensor, rtol=0.25, atol=0.25)


class TestUnifiedGEMMInterface:
    """Test unified gemm_with_precision interface."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_fp16_unified(self, small_matrices):
        a, b, bias = small_matrices
        output = gemm_with_precision(a, b, bias, precision="fp16", backend="triton", transpose_b=True)
        expected = torch.matmul(a, b.t()) + bias
        assert torch.allclose(output, expected, rtol=1e-2, atol=1e-2)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_bf16_unified(self, small_matrices):
        a, b, bias = small_matrices
        a_bf16 = a.to(torch.bfloat16)
        b_bf16 = b.to(torch.bfloat16)
        bias_bf16 = bias.to(torch.bfloat16)

        output = gemm_with_precision(
            a_bf16, b_bf16, bias_bf16,
            precision="bf16",
            backend="triton",
            transpose_b=True,
        )
        expected = torch.matmul(a_bf16, b_bf16.t()) + bias_bf16
        assert torch.allclose(output.float(), expected.float(), rtol=5e-2, atol=5e-2)

    def test_torch_fallback(self, small_matrices):
        a, b, bias = small_matrices
        output = gemm_with_precision(a, b, bias, precision="fp16", backend="torch", transpose_b=True)
        expected = torch.matmul(a, b.t()) + bias
        assert torch.allclose(output, expected, rtol=1e-3, atol=1e-3)


class TestCapabilityReporting:
    """Test precision capability reporting."""

    def test_list_available_precisions(self):
        precisions = list_available_precisions()
        assert isinstance(precisions, list)
        assert len(precisions) > 0
        assert "fp16" in precisions

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_fp16_capability_cuda(self):
        device = torch.device("cuda")
        cap = describe_gemm_precision_capability("fp16", device)
        assert cap["precision"] == "fp16"
        assert cap["available"] is True
        assert cap["backend"] == "triton"

    def test_fp16_capability_cpu(self):
        device = torch.device("cpu")
        cap = describe_gemm_precision_capability("fp16", device)
        assert cap["precision"] == "fp16"
        # CPU should have torch fallback
        assert cap["backend"] in {"torch", "none"}

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_fp8_capability(self):
        device = torch.device("cuda")
        cap = describe_gemm_precision_capability("fp8", device)
        assert cap["precision"] == "fp8"

        # Check hardware support
        major, minor = torch.cuda.get_device_capability(device)
        sm = major * 10 + minor
        if sm >= 89:
            assert cap["hardware_native"] is True
        else:
            assert cap["hardware_native"] is False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestPerformanceBaseline:
    """Baseline performance tests (not strict, for profiling)."""

    def test_fp16_performance(self, medium_matrices):
        a, b, bias = medium_matrices

        def run_gemm():
            return gemm_fp16_triton(a, b, bias, transpose_b=True)

        # Warmup
        for _ in range(3):
            run_gemm()

        # Run once for correctness
        result = run_gemm()

        # Sanity check
        expected = torch.matmul(a, b.t()) + bias
        assert torch.allclose(result, expected, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
