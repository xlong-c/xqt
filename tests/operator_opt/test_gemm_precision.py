"""Tests for multi-precision GEMM kernels."""

import pytest
import torch
import triton.language as tl

from xqt.operator_opt.backends.gemm_precision import (
    MatmulPrecisionSpec,
    describe_gemm_precision_capability,
    gemm_with_precision,
    list_available_precisions,
)
from xqt.contracts import PrecisionPolicy
from xqt.operator_opt.kernels.tilelang.gemm import (
    nvfp4_packed_dequant_gemm_epilogue_reference,
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
        unpacked = unpack_mxfp(
            packed, scales, precision=8, block_size=32, original_numel=tensor.numel()
        )
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
        unpacked = unpack_mxfp(
            packed, scales, precision=4, block_size=32, original_numel=tensor.numel()
        )
        unpacked = unpacked.reshape(tensor.shape)

        # MXFP4 has 3 mantissa bits, quantization step ~1/7 ≈ 14%
        # Validate against the per-block quantization step instead of global
        # allclose. Values near zero can exceed a fixed relative tolerance even
        # when packing and unpacking are correct.
        assert unpacked.shape == tensor.shape
        block_error = (unpacked.flatten() - tensor.flatten()).abs().reshape(-1, 32)
        assert torch.all(block_error <= (scales.reshape(-1, 1) * 0.5 + 1e-6))


class TestUnifiedGEMMInterface:
    """Test unified gemm_with_precision interface."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_fp16_unified(self, small_matrices):
        a, b, bias = small_matrices
        output = gemm_with_precision(
            a, b, bias, precision="fp16", engine="triton", transpose_b=True
        )
        expected = torch.matmul(a, b.t()) + bias
        assert torch.allclose(output, expected, rtol=1e-2, atol=1e-2)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_bf16_unified(self, small_matrices):
        a, b, bias = small_matrices
        a_bf16 = a.to(torch.bfloat16)
        b_bf16 = b.to(torch.bfloat16)
        bias_bf16 = bias.to(torch.bfloat16)

        output = gemm_with_precision(
            a_bf16,
            b_bf16,
            bias_bf16,
            precision="bf16",
            engine="triton",
            transpose_b=True,
        )
        expected = torch.matmul(a_bf16, b_bf16.t()) + bias_bf16
        assert torch.allclose(output.float(), expected.float(), rtol=5e-2, atol=5e-2)

    def test_torch_fallback(self, small_matrices):
        a, b, bias = small_matrices
        output = gemm_with_precision(
            a, b, bias, precision="fp16", engine="torch", transpose_b=True
        )
        expected = torch.matmul(a, b.t()) + bias
        assert torch.allclose(output, expected, rtol=1e-3, atol=1e-3)

    def test_torch_fallback_accepts_structured_matmul_precision_spec(
        self, small_matrices
    ):
        a, b, bias = small_matrices
        output = gemm_with_precision(
            a,
            b,
            bias,
            precision=MatmulPrecisionSpec(
                activation="fp16",
                weight="fp16",
                mma="fp16",
                accum="fp32",
                output="bf16",
            ),
            engine="torch",
            transpose_b=True,
        )

        assert output.dtype == torch.bfloat16
        expected = (torch.matmul(a, b.t()) + bias).to(torch.bfloat16)
        assert torch.allclose(output.float(), expected.float(), rtol=1e-3, atol=1e-3)

    def test_torch_fallback_accepts_mapping_precision_spec(self, small_matrices):
        a, b, bias = small_matrices
        output = gemm_with_precision(
            a,
            b,
            bias,
            precision={
                "activation": "fp16",
                "weight": "fp16",
                "mma": "fp16",
                "accum": "fp32",
                "output": "fp32",
            },
            engine="torch",
            transpose_b=True,
        )

        assert output.dtype == torch.float32
        expected = (torch.matmul(a, b.t()) + bias).to(torch.float32)
        assert torch.allclose(output, expected, rtol=1e-3, atol=1e-3)

    def test_torch_fallback_accepts_abco_precision_roles(self, small_matrices):
        a, b, bias = small_matrices
        output = gemm_with_precision(
            a,
            b,
            bias,
            precision={
                "A": "nvfp4",
                "B": "fp16",
                "C": "fp32",
                "MMA": "fp16",
                "ACCUM": "fp32",
                "O": "fp16",
            },
            engine="torch",
            transpose_b=True,
        )

        assert output.dtype == torch.float16
        expected = (
            torch.matmul(a.to(torch.float16), b.t().to(torch.float16))
            + bias.to(torch.float32)
        ).to(torch.float16)
        assert torch.allclose(output, expected, rtol=1e-3, atol=1e-3)

    def test_matmul_precision_spec_from_roles_records_low_bit_storage(self) -> None:
        assert MatmulPrecisionSpec is PrecisionPolicy
        spec = MatmulPrecisionSpec.from_roles(
            A="nvfp4",
            B="fp16",
            C="fp32",
            mma="fp16",
            accum="fp32",
            O="bf16",
        )

        assert spec.to_dict() == {
            "activation": "nvfp4",
            "weight": "fp16",
            "bias": "fp32",
            "mma": "fp16",
            "accum": "fp32",
            "output": "bf16",
        }

    def test_torch_fallback_rejects_low_bit_mma_without_kernel(self, small_matrices):
        a, b, bias = small_matrices

        with pytest.raises(Exception, match="mma precision nvfp4"):
            gemm_with_precision(
                a,
                b,
                bias,
                precision={
                    "A": "fp16",
                    "B": "fp16",
                    "MMA": "nvfp4",
                    "O": "fp16",
                },
                engine="torch",
                transpose_b=True,
            )

    def test_tilelang_nvfp4_reference_path(self) -> None:
        a = torch.randn(64, 64, dtype=torch.float32)
        packed = torch.randint(0, 256, (64, 32), dtype=torch.uint8)
        scale = torch.ones(64, 4, 1, dtype=torch.float32) * 0.125
        bias = torch.randn(64, dtype=torch.float32)
        global_scale = torch.tensor([2.0], dtype=torch.float32)

        output = gemm_with_precision(
            a,
            packed,
            bias,
            precision="nvfp4",
            engine="tilelang",
            transpose_b=True,
            b_scale=scale,
            group_size=16,
            input_features=64,
            weight_global_scale=global_scale,
        )

        expected = nvfp4_packed_dequant_gemm_epilogue_reference(
            a,
            packed,
            scale,
            bias,
            input_features=64,
            group_size=16,
            weight_global_scale=global_scale,
        )
        assert torch.allclose(output, expected, rtol=1e-3, atol=1e-3)

def test_gemm_fp16_triton_forwards_accum_and_output_dtype_to_kernel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeKernelLaunch:
        def __getitem__(self, grid):
            captured["grid"] = grid

            def runner(*args, **kwargs):
                captured["args"] = args
                captured["kwargs"] = kwargs

            return runner

    monkeypatch.setattr(
        "xqt.operator_opt.kernels.triton.gemm._require_cuda_tensors",
        lambda *args: None,
    )
    monkeypatch.setattr(
        "xqt.operator_opt.kernels.triton.gemm._gemm_kernel",
        FakeKernelLaunch(),
    )

    a = torch.randn(8, 16, dtype=torch.float16)
    b = torch.randn(12, 16, dtype=torch.float16)
    bias = torch.randn(12, dtype=torch.float32)

    output = gemm_fp16_triton(
        a,
        b,
        bias,
        transpose_b=True,
        accum_dtype=torch.float16,
        output_dtype=torch.float32,
    )

    assert output.dtype == torch.float32
    assert captured["kwargs"]["ACC_TYPE"] == tl.float16
    assert captured["kwargs"]["has_bias"] is True
    launched_bias = captured["args"][3]
    assert isinstance(launched_bias, torch.Tensor)
    assert launched_bias.dtype == torch.float32


def test_gemm_bf16_triton_preserves_bf16_inputs_and_forwards_precision_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_fp16_entry(a, b, bias, **kwargs):
        captured["a_dtype"] = a.dtype
        captured["b_dtype"] = b.dtype
        captured["bias_dtype"] = None if bias is None else bias.dtype
        captured["kwargs"] = kwargs
        return torch.empty((a.shape[0], b.shape[0]), dtype=kwargs["output_dtype"])

    monkeypatch.setattr(
        "xqt.operator_opt.kernels.triton.gemm._require_cuda_tensors",
        lambda *args: None,
    )
    monkeypatch.setattr(
        "xqt.operator_opt.kernels.triton.gemm.gemm_fp16_triton",
        fake_fp16_entry,
    )

    a = torch.randn(8, 16, dtype=torch.bfloat16)
    b = torch.randn(12, 16, dtype=torch.bfloat16)
    bias = torch.randn(12, dtype=torch.bfloat16)

    output = gemm_bf16_triton(
        a,
        b,
        bias,
        transpose_b=True,
        accum_dtype=torch.float32,
        output_dtype=torch.bfloat16,
    )

    assert output.dtype == torch.bfloat16
    assert captured["a_dtype"] == torch.bfloat16
    assert captured["b_dtype"] == torch.bfloat16
    assert captured["bias_dtype"] == torch.bfloat16
    assert captured["kwargs"]["accum_dtype"] == torch.float32
    assert captured["kwargs"]["output_dtype"] == torch.bfloat16


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
        assert cap["engine"] == "triton"

    def test_fp16_capability_cpu(self):
        device = torch.device("cpu")
        cap = describe_gemm_precision_capability("fp16", device)
        assert cap["precision"] == "fp16"
        # CPU should have torch fallback
        assert cap["engine"] in {"torch", "none"}

    def test_nvfp4_capability_cpu(self):
        device = torch.device("cpu")
        cap = describe_gemm_precision_capability("nvfp4", device)
        assert cap["precision"] == "nvfp4"
        assert cap["available"] is True
        assert cap["engine"] == "tilelang"

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
