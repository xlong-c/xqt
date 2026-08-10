"""Tests for multi-precision GEMM kernels."""

import pytest
import torch
import torch.nn.functional as F
import triton.language as tl

from xqt.operator_opt.backends.gemm_precision import (
    MatmulPrecisionSpec,
    conv1x1_as_gemm_with_precision,
    conv2d_as_gemm_with_precision,
    conv3x3_im2col_gemm_with_precision,
    attention_score_gemm_with_precision,
    attention_value_gemm_with_precision,
    batched_gemm_with_precision,
    describe_gemm_precision_capability,
    expert_gemm_with_precision,
    gemm_with_precision,
    lm_head_gemm_with_precision,
    grouped_gemm_with_precision,
    list_gemm_variant_dispatch_specs,
    projection_gemm_with_precision,
    gemm_variant_with_precision,
    list_available_precisions,
    router_gemm_with_precision,
)
from xqt.operator_opt.backends.triton import get_triton_kernel_spec
from xqt.operator_opt.backends.tilelang import get_tilelang_kernel_spec
from xqt.contracts import PrecisionPolicy
from xqt.operator_opt.kernels.tilelang.gemm import (
    mxfp4_packed_dequant_gemm_epilogue_reference,
    nvfp4_packed_dequant_gemm_epilogue_reference,
)
from xqt.operator_opt.kernels.triton.gemm import (
    TritonGemmSchedule,
    gemm_bf16_triton,
    gemm_fp16_triton,
    gemm_int8_triton,
    gemm_reference,
    resolve_triton_bf16_gemm_schedule,
    resolve_triton_fp16_gemm_schedule,
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

    @pytest.mark.parametrize("activation", ["gelu", "silu"])
    def test_bf16_bias_activation_epilogue_matches_fp32_reference(
        self,
        small_matrices: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        activation: str,
    ) -> None:
        a, b, bias = small_matrices
        a_bf16 = a.to(torch.bfloat16)
        b_bf16 = b.to(torch.bfloat16)
        bias_bf16 = bias.to(torch.bfloat16)

        output = gemm_bf16_triton(
            a_bf16,
            b_bf16,
            bias_bf16,
            activation=activation,
            transpose_b=True,
        )
        reference = a_bf16.float() @ b_bf16.float().t() + bias_bf16.float()
        reference = (
            F.gelu(reference) if activation == "gelu" else F.silu(reference)
        ).to(torch.bfloat16)
        error = (output.float() - reference.float()).abs()

        assert output.dtype == torch.bfloat16
        assert error.max().item() <= 0.5
        assert error.mean().item() <= 0.05
        assert torch.allclose(output.float(), reference.float(), atol=0.5, rtol=0.03)

    @pytest.mark.parametrize("activation", ["gelu", "silu"])
    def test_fp16_bias_activation_epilogue_regression(
        self,
        small_matrices: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        activation: str,
    ) -> None:
        a, b, bias = small_matrices

        output = gemm_fp16_triton(
            a,
            b,
            bias,
            activation=activation,
            transpose_b=True,
        )
        reference = a.float() @ b.float().t() + bias.float()
        reference = (
            F.gelu(reference) if activation == "gelu" else F.silu(reference)
        ).to(torch.float16)
        error = (output.float() - reference.float()).abs()

        assert output.dtype == torch.float16
        assert error.max().item() <= 0.25
        assert error.mean().item() <= 0.025
        assert torch.allclose(output.float(), reference.float(), atol=0.25, rtol=0.03)

    @pytest.mark.parametrize(
        ("m", "n", "k", "has_bias", "activation"),
        [
            (1, 4096, 4096, False, None),
            (4, 4096, 4096, True, None),
            (8, 11008, 4096, True, "silu"),
            (64, 1024, 1024, True, None),
            (256, 4096, 4096, True, "gelu"),
        ],
    )
    def test_bf16_sm89_schedule_shapes_match_fp32_reference(
        self,
        m: int,
        n: int,
        k: int,
        has_bias: bool,
        activation: str | None,
    ) -> None:
        if torch.cuda.get_device_capability() != (8, 9):
            pytest.skip("SM89 schedule regression requires sm_89")
        torch.manual_seed(20260809)
        a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
        b = torch.randn((k, n), device="cuda", dtype=torch.bfloat16)
        bias = (
            torch.randn((n,), device="cuda", dtype=torch.bfloat16)
            if has_bias
            else None
        )

        output = gemm_bf16_triton(
            a,
            b,
            bias,
            activation=activation,
            transpose_b=False,
        )
        reference = a.float() @ b.float()
        if bias is not None:
            reference = reference + bias.float()
        if activation == "gelu":
            reference = F.gelu(reference)
        elif activation == "silu":
            reference = F.silu(reference)
        reference = reference.to(torch.bfloat16)
        error = (output.float() - reference.float()).abs()

        assert output.dtype == torch.bfloat16
        assert error.mean().item() <= 0.001
        assert torch.allclose(output.float(), reference.float(), atol=0.5, rtol=0.03)

    def test_int8_true_w8a8_basic(self):
        torch.manual_seed(11)
        a = torch.randint(-16, 16, (64, 128), device="cuda", dtype=torch.int8)
        b = torch.randint(-16, 16, (128, 96), device="cuda", dtype=torch.int8)
        a_scale = torch.tensor([0.02], device="cuda", dtype=torch.float32)
        b_scale = torch.linspace(0.001, 0.003, 96, device="cuda")
        bias = torch.randn(96, device="cuda", dtype=torch.float32) * 0.01

        output = gemm_int8_triton(
            a,
            b,
            a_scale,
            b_scale,
            bias,
            transpose_b=False,
            block_m=64,
            block_n=128,
            block_k=32,
            output_dtype=torch.float16,
        )
        reference = torch._int_mm(a, b).to(torch.float32)
        reference = reference * (a_scale.reshape(1) * b_scale.reshape(1, -1)) + bias

        assert output.dtype == torch.float16
        assert torch.allclose(output.float(), reference, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize(
    (
        "m",
        "n",
        "k",
        "has_bias",
        "activation",
        "transpose_b",
        "expected",
        "preset",
    ),
    [
        (
            1,
            4096,
            4096,
            False,
            None,
            True,
            (16, 64, 64, 4, 4, 3),
            "sm89_fp16_decode_m1",
        ),
        (
            1,
            4096,
            4096,
            False,
            None,
            False,
            (16, 64, 64, 4, 4, 3),
            "sm89_fp16_decode_m1",
        ),
        (
            4,
            4096,
            4096,
            True,
            None,
            True,
            (16, 64, 64, 4, 4, 3),
            "sm89_fp16_decode_m4_bias",
        ),
        (
            4,
            4096,
            4096,
            True,
            None,
            False,
            (16, 64, 64, 4, 4, 3),
            "sm89_fp16_decode_m4_bias",
        ),
        (
            8,
            11008,
            4096,
            True,
            "silu",
            True,
            (32, 128, 32, 4, 4, 3),
            "sm89_fp16_decode_m8_silu",
        ),
        (
            8,
            11008,
            4096,
            True,
            "silu",
            False,
            (32, 128, 32, 4, 4, 3),
            "sm89_fp16_decode_m8_silu",
        ),
        (
            64,
            1024,
            1024,
            True,
            None,
            True,
            (16, 128, 32, 4, 4, 3),
            "sm89_fp16_small_prefill_bias",
        ),
        (
            64,
            1024,
            1024,
            True,
            None,
            False,
            (16, 128, 32, 4, 4, 3),
            "sm89_fp16_small_prefill_bias",
        ),
        (
            256,
            4096,
            4096,
            True,
            "gelu",
            False,
            (64, 64, 32, 8, 4, 3),
            "sm89_fp16_medium_prefill_gelu_kn",
        ),
    ],
)
def test_triton_fp16_sm89_schedule_presets_are_exact_signature_scoped(
    m: int,
    n: int,
    k: int,
    has_bias: bool,
    activation: str | None,
    transpose_b: bool,
    expected: tuple[int, int, int, int, int, int],
    preset: str,
) -> None:
    schedule = resolve_triton_fp16_gemm_schedule(
        m=m,
        n=n,
        k=k,
        has_bias=has_bias,
        activation=activation,
        target_arch="sm_89",
        transpose_b=transpose_b,
    )

    assert isinstance(schedule, TritonGemmSchedule)
    assert (
        schedule.block_m,
        schedule.block_n,
        schedule.block_k,
        schedule.group_m,
        schedule.num_warps,
        schedule.num_stages,
    ) == expected
    assert schedule.preset == preset


@pytest.mark.parametrize(
    ("target_arch", "m", "has_bias", "activation", "transpose_b"),
    [
        ("sm_90", 1, False, None, True),
        ("sm_89", 2, False, None, True),
        ("sm_89", 1, True, None, True),
        ("sm_89", 1, False, "silu", True),
        ("sm_89", 256, True, "gelu", True),
    ],
)
def test_triton_fp16_schedule_keeps_default_outside_exact_presets(
    target_arch: str,
    m: int,
    has_bias: bool,
    activation: str | None,
    transpose_b: bool,
) -> None:
    schedule = resolve_triton_fp16_gemm_schedule(
        m=m,
        n=4096,
        k=4096,
        has_bias=has_bias,
        activation=activation,
        target_arch=target_arch,
        transpose_b=transpose_b,
    )

    assert (
        schedule.block_m,
        schedule.block_n,
        schedule.block_k,
        schedule.group_m,
        schedule.num_warps,
        schedule.num_stages,
    ) == (128, 128, 32, 8, 4, 3)
    assert schedule.preset == "default"


def test_triton_fp16_schedule_preserves_per_field_explicit_overrides() -> None:
    schedule = resolve_triton_fp16_gemm_schedule(
        m=64,
        n=1024,
        k=1024,
        has_bias=True,
        activation=None,
        block_m=32,
        block_n=256,
        block_k=16,
        group_m=2,
        num_warps=8,
        num_stages=2,
        target_arch="sm_89",
        transpose_b=True,
    )

    assert schedule.to_dict() == {
        "block_m": 32,
        "block_n": 256,
        "block_k": 16,
        "group_m": 2,
        "num_warps": 8,
        "num_stages": 2,
        "target_arch": "sm_89",
        "preset": "sm89_fp16_small_prefill_bias",
    }


@pytest.mark.parametrize(
    (
        "m",
        "n",
        "k",
        "has_bias",
        "activation",
        "expected",
        "preset",
    ),
    [
        (
            1,
            4096,
            4096,
            False,
            None,
            (16, 64, 64, 4, 4, 3),
            "sm89_bf16_decode_m1",
        ),
        (
            4,
            4096,
            4096,
            True,
            None,
            (16, 64, 64, 4, 4, 3),
            "sm89_bf16_decode_m4_bias",
        ),
        (
            8,
            11008,
            4096,
            True,
            "silu",
            (32, 128, 32, 4, 4, 3),
            "sm89_bf16_decode_m8_silu",
        ),
        (
            64,
            1024,
            1024,
            True,
            None,
            (16, 64, 32, 4, 4, 3),
            "sm89_bf16_small_prefill_bias",
        ),
        (
            256,
            4096,
            4096,
            True,
            "gelu",
            (64, 64, 32, 8, 4, 3),
            "sm89_bf16_medium_prefill_gelu",
        ),
    ],
)
def test_triton_bf16_sm89_schedule_presets_are_exact_signature_scoped(
    m: int,
    n: int,
    k: int,
    has_bias: bool,
    activation: str | None,
    expected: tuple[int, int, int, int, int, int],
    preset: str,
) -> None:
    schedule = resolve_triton_bf16_gemm_schedule(
        m=m,
        n=n,
        k=k,
        has_bias=has_bias,
        activation=activation,
        target_arch="sm_89",
    )

    assert isinstance(schedule, TritonGemmSchedule)
    assert (
        schedule.block_m,
        schedule.block_n,
        schedule.block_k,
        schedule.group_m,
        schedule.num_warps,
        schedule.num_stages,
    ) == expected
    assert schedule.preset == preset


@pytest.mark.parametrize(
    ("target_arch", "m", "has_bias", "activation"),
    [
        ("sm_90", 1, False, None),
        ("sm_89", 2, False, None),
        ("sm_89", 1, True, None),
        ("sm_89", 1, False, "silu"),
    ],
)
def test_triton_bf16_schedule_keeps_default_outside_exact_presets(
    target_arch: str,
    m: int,
    has_bias: bool,
    activation: str | None,
) -> None:
    schedule = resolve_triton_bf16_gemm_schedule(
        m=m,
        n=4096,
        k=4096,
        has_bias=has_bias,
        activation=activation,
        target_arch=target_arch,
    )

    assert (
        schedule.block_m,
        schedule.block_n,
        schedule.block_k,
        schedule.group_m,
        schedule.num_warps,
        schedule.num_stages,
    ) == (128, 128, 32, 8, 4, 3)
    assert schedule.preset == "default"


def test_triton_bf16_schedule_preserves_per_field_explicit_overrides() -> None:
    schedule = resolve_triton_bf16_gemm_schedule(
        m=1,
        n=4096,
        k=4096,
        has_bias=False,
        activation=None,
        block_m=32,
        block_n=256,
        block_k=16,
        group_m=2,
        num_warps=8,
        num_stages=2,
        target_arch="sm_89",
    )

    assert schedule.to_dict() == {
        "block_m": 32,
        "block_n": 256,
        "block_k": 16,
        "group_m": 2,
        "num_warps": 8,
        "num_stages": 2,
        "target_arch": "sm_89",
        "preset": "sm89_bf16_decode_m1",
    }


def test_gemm_with_precision_forwards_triton_schedule_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_run_triton_kernel(
        pattern: str,
        *args: torch.Tensor,
        **kwargs: object,
    ) -> torch.Tensor:
        captured["pattern"] = pattern
        captured["kwargs"] = dict(kwargs)
        return torch.empty(
            (int(args[0].shape[0]), int(args[1].shape[0])),
            dtype=kwargs["output_dtype"],
        )

    monkeypatch.setattr(
        "xqt.operator_opt.backends.gemm_precision.run_triton_kernel",
        fake_run_triton_kernel,
    )
    a = torch.randn(4, 16, dtype=torch.bfloat16)
    b = torch.randn(12, 16, dtype=torch.bfloat16)

    output = gemm_with_precision(
        a,
        b,
        precision="bf16",
        engine="triton",
        target_arch="sm_89",
        block_m=16,
        block_n=64,
        block_k=64,
        group_m=4,
        num_warps=4,
        num_stages=3,
    )

    assert output.shape == (4, 12)
    assert captured["pattern"] == "gemm_bf16"
    forwarded = captured["kwargs"]
    assert isinstance(forwarded, dict)
    assert forwarded["target_arch"] == "sm_89"
    assert forwarded["block_m"] == 16
    assert forwarded["block_n"] == 64
    assert forwarded["block_k"] == 64
    assert forwarded["group_m"] == 4
    assert forwarded["num_warps"] == 4
    assert forwarded["num_stages"] == 3


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

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_bf16_tilelang_unified_uses_direct_dense_gemm(self, small_matrices):
        a, b, bias = small_matrices
        a_bf16 = a.to(torch.bfloat16)
        b_bf16 = b.to(torch.bfloat16)
        bias_bf16 = bias.to(torch.bfloat16)

        output = gemm_with_precision(
            a_bf16,
            b_bf16,
            bias_bf16,
            precision="bf16",
            engine="tilelang",
            transpose_b=True,
            target_arch="sm_89",
        )
        expected = torch.matmul(a_bf16, b_bf16.t()) + bias_bf16

        assert output.dtype == torch.bfloat16
        assert torch.allclose(output.float(), expected.float(), rtol=5e-2, atol=5e-2)

    def test_bf16_tilelang_dispatch_defaults_to_dense_linear_epilogue(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        captured: dict[str, object] = {}

        def fake_run_tilelang_kernel(
            pattern: str,
            *args: torch.Tensor,
            **kwargs: object,
        ) -> torch.Tensor:
            captured["pattern"] = pattern
            captured["kwargs"] = dict(kwargs)
            return torch.empty(
                (int(args[0].shape[0]), int(args[1].shape[0])),
                dtype=args[0].dtype,
            )

        monkeypatch.setattr(
            "xqt.operator_opt.backends.gemm_precision.run_tilelang_kernel",
            fake_run_tilelang_kernel,
        )
        a = torch.randn(1, 64, dtype=torch.bfloat16)
        b = torch.randn(96, 64, dtype=torch.bfloat16)
        bias = torch.randn(96, dtype=torch.bfloat16)

        output = gemm_with_precision(
            a,
            b,
            bias,
            precision="bf16",
            engine="tilelang",
            transpose_b=True,
            block_m=16,
            block_n=32,
            block_k=32,
        )

        assert output.shape == (1, 96)
        assert captured["pattern"] == "dense_linear_epilogue"

    def test_bf16_tilelang_dispatch_rejects_fp16_inputs(self, small_matrices) -> None:
        a, b, bias = small_matrices

        with pytest.raises(Exception, match="to match torch.bfloat16"):
            gemm_with_precision(
                a,
                b,
                bias,
                precision="bf16",
                engine="tilelang",
                transpose_b=True,
            )

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

    def test_tilelang_mxfp4_reference_path(self) -> None:
        a = torch.randn(64, 64, dtype=torch.float32)
        packed = torch.full((64, 32), 0x21, dtype=torch.uint8)
        scale = torch.ones(64, 2, 1, dtype=torch.float32) * 0.125
        bias = torch.randn(64, dtype=torch.float32)

        output = gemm_with_precision(
            a,
            packed,
            bias,
            precision="mxfp4",
            engine="tilelang",
            transpose_b=True,
            b_scale=scale,
            group_size=32,
            input_features=64,
        )

        expected = mxfp4_packed_dequant_gemm_epilogue_reference(
            a,
            packed,
            scale,
            bias,
            input_features=64,
            group_size=32,
        )
        assert torch.allclose(output, expected, rtol=1e-3, atol=1e-3)


def test_batched_gemm_shared_weight_uses_one_flattened_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[tuple[tuple[int, ...], tuple[int, ...], dict[str, object]]] = []

    def fake_gemm_with_precision(
        a: torch.Tensor,
        b: torch.Tensor,
        bias: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        del bias
        captured.append((tuple(a.shape), tuple(b.shape), dict(kwargs)))
        return torch.empty((a.shape[0], b.shape[0]), dtype=torch.float32)

    monkeypatch.setattr(
        "xqt.operator_opt.backends.gemm_precision.gemm_with_precision",
        fake_gemm_with_precision,
    )

    a = torch.randn(2, 3, 4)
    b = torch.randn(5, 4)

    output = batched_gemm_with_precision(
        a,
        b,
        precision="fp32",
        engine="tilelang",
        pattern="linear_marlin",
    )

    assert output.shape == (2, 3, 5)
    assert captured == [
        (
            (6, 4),
            (5, 4),
            {
                "precision": "fp32",
                "engine": "tilelang",
                "pattern": "linear_marlin",
                "activation": None,
                "transpose_b": True,
            },
        )
    ]


def test_batched_gemm_per_batch_weight_matches_torch_reference() -> None:
    torch.manual_seed(11)
    a = torch.randn(2, 3, 4)
    b = torch.randn(2, 5, 4)
    bias = torch.randn(2, 5)

    output = batched_gemm_with_precision(
        a,
        b,
        bias,
        precision="fp32",
        engine="torch",
    )
    expected = torch.stack(
        [torch.matmul(a[index], b[index].t()) + bias[index] for index in range(2)],
        dim=0,
    )

    assert torch.allclose(output, expected)


def test_grouped_gemm_with_precision_matches_torch_reference() -> None:
    torch.manual_seed(12)
    a_groups = (
        torch.randn(3, 4),
        torch.randn(5, 4),
    )
    b_groups = (
        torch.randn(6, 4),
        torch.randn(2, 4),
    )
    bias_groups = (
        torch.randn(6),
        torch.randn(2),
    )

    outputs = grouped_gemm_with_precision(
        a_groups,
        b_groups,
        bias_groups,
        precision="fp32",
        engine="torch",
    )

    assert isinstance(outputs, tuple)
    assert len(outputs) == 2
    for output, a, b, bias in zip(
        outputs, a_groups, b_groups, bias_groups, strict=True
    ):
        assert torch.allclose(output, torch.matmul(a, b.t()) + bias)


def test_expert_gemm_with_precision_aliases_grouped_gemm() -> None:
    torch.manual_seed(13)
    token_groups = (
        torch.randn(2, 4),
        torch.randn(1, 4),
    )
    expert_weights = (
        torch.randn(8, 4),
        torch.randn(8, 4),
    )

    outputs = expert_gemm_with_precision(
        token_groups,
        expert_weights,
        precision="fp32",
        engine="torch",
    )

    assert len(outputs) == 2
    assert torch.allclose(outputs[0], torch.matmul(token_groups[0], expert_weights[0].t()))
    assert torch.allclose(outputs[1], torch.matmul(token_groups[1], expert_weights[1].t()))


def test_projection_gemm_with_precision_matches_torch_reference() -> None:
    torch.manual_seed(17)
    x = torch.randn(2, 3, 4)
    weight = torch.randn(7, 4)
    bias = torch.randn(7)

    output = projection_gemm_with_precision(
        x,
        weight,
        bias,
        precision="fp32",
        engine="torch",
    )

    expected = torch.matmul(x, weight.t()) + bias
    assert output.shape == (2, 3, 7)
    assert torch.allclose(output, expected)


def test_gemm_variant_dispatch_table_lists_expected_entries() -> None:
    table = list_gemm_variant_dispatch_specs()
    assert "gemm_bias_gelu" in table
    assert "qkv_projection_gemm" in table
    assert "conv3x3_im2col_gemm" in table
    assert table["gemm_bias_gelu"]["op"] == "gemm"
    assert table["qkv_projection_gemm"]["op"] == "projection"
    assert table["conv3x3_im2col_gemm"]["op"] == "conv3x3"


def test_gemm_variant_with_precision_routes_dense_variants_to_torch_reference() -> None:
    torch.manual_seed(1711)
    a = torch.randn(4, 8)
    b = torch.randn(6, 8)
    bias = torch.randn(6)

    output = gemm_variant_with_precision(
        "gemm_bias_gelu",
        a,
        b,
        bias,
        precision="fp32",
        engine="torch",
    )

    expected = torch.nn.functional.gelu(torch.matmul(a, b.t()) + bias)
    assert torch.allclose(output, expected)


def test_gemm_variant_with_precision_routes_projection_variants_to_torch_reference() -> None:
    torch.manual_seed(1712)
    x = torch.randn(2, 3, 4)
    weight = torch.randn(7, 4)

    output = gemm_variant_with_precision(
        "qkv_projection_gemm",
        x,
        weight,
        precision="fp32",
        engine="torch",
    )

    expected = torch.matmul(x, weight.t())
    assert output.shape == expected.shape
    assert torch.allclose(output, expected)


def test_gemm_variant_with_precision_routes_conv_variants_to_torch_reference() -> None:
    torch.manual_seed(1713)
    x = torch.randn(2, 3, 5, 7)
    weight = torch.randn(4, 3, 3, 3)
    bias = torch.randn(4)

    output = gemm_variant_with_precision(
        "conv3x3_im2col_gemm",
        x,
        weight,
        bias,
        precision="fp32",
        engine="torch",
        padding=1,
    )
    expected = F.conv2d(x, weight, bias=bias, padding=1)

    assert output.shape == expected.shape
    assert torch.allclose(output, expected, rtol=1e-5, atol=1e-6)


def test_gemm_variant_with_precision_routes_int8_tilelang_pattern(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_run_tilelang_kernel(pattern: str, *args: object, **kwargs: object) -> torch.Tensor:
        captured["pattern"] = pattern
        captured["kwargs"] = kwargs
        a = args[0]
        weight = args[1]
        assert isinstance(a, torch.Tensor)
        assert isinstance(weight, torch.Tensor)
        return torch.empty((a.shape[0], weight.shape[0]), dtype=kwargs["output_dtype"])

    monkeypatch.setattr(
        "xqt.operator_opt.backends.gemm_precision.run_tilelang_kernel",
        fake_run_tilelang_kernel,
    )

    a = torch.randint(-8, 8, (8, 16), dtype=torch.int8)
    b = torch.randint(-8, 8, (12, 16), dtype=torch.int8)
    a_scale = torch.tensor([0.02], dtype=torch.float32)
    b_scale = torch.ones(12, dtype=torch.float32) * 0.01

    output = gemm_variant_with_precision(
        "int8_linear",
        a,
        b,
        precision="int8",
        engine="tilelang",
        a_scale=a_scale,
        b_scale=b_scale,
    )

    assert output.dtype == torch.float16
    assert captured["pattern"] == "int8_linear"


def test_gemm_variant_with_precision_rejects_non_gemm_names() -> None:
    with pytest.raises(Exception, match="Unsupported GEMM variant"):
        gemm_variant_with_precision("router_softmax_topk")


@pytest.mark.parametrize("engine", ["triton", "tilelang"])
def test_projection_gemm_with_precision_reuses_configured_engines(engine: str) -> None:
    torch.manual_seed(171)
    x = torch.randn(2, 3, 4, dtype=torch.float16)
    weight = torch.randn(7, 4, dtype=torch.float16)
    bias = torch.randn(7, dtype=torch.float16)

    output = projection_gemm_with_precision(
        x,
        weight,
        bias,
        precision="fp16",
        engine=engine,
    )

    expected = torch.matmul(x, weight.t()) + bias
    assert output.shape == (2, 3, 7)
    assert torch.allclose(output.float(), expected.float(), rtol=1e-3, atol=1e-3)


def test_lm_head_gemm_with_precision_matches_torch_reference() -> None:
    torch.manual_seed(18)
    x = torch.randn(2, 5, 4)
    weight = torch.randn(11, 4)

    output = lm_head_gemm_with_precision(
        x,
        weight,
        precision="fp32",
        engine="torch",
    )

    expected = torch.matmul(x, weight.t())
    assert output.shape == (2, 5, 11)
    assert torch.allclose(output, expected)


def test_conv1x1_as_gemm_with_precision_matches_torch_reference() -> None:
    torch.manual_seed(19)
    x = torch.randn(2, 3, 5, 7)
    weight = torch.randn(4, 3, 1, 1)
    bias = torch.randn(4)

    output = conv1x1_as_gemm_with_precision(
        x,
        weight,
        bias,
        precision="fp32",
        engine="torch",
    )
    expected = F.conv2d(x, weight, bias=bias)

    assert output.shape == expected.shape
    assert torch.allclose(output, expected, rtol=1e-5, atol=1e-6)


def test_conv3x3_im2col_gemm_with_precision_matches_torch_reference() -> None:
    torch.manual_seed(20)
    x = torch.randn(2, 3, 6, 6)
    weight = torch.randn(4, 3, 3, 3)
    bias = torch.randn(4)

    output = conv3x3_im2col_gemm_with_precision(
        x,
        weight,
        bias,
        precision="fp32",
        engine="torch",
        padding=1,
    )
    expected = F.conv2d(x, weight, bias=bias, padding=1)

    assert output.shape == expected.shape
    assert torch.allclose(output, expected)


def test_conv2d_as_gemm_with_precision_matches_torch_reference() -> None:
    torch.manual_seed(21)
    x = torch.randn(2, 3, 5, 7)
    weight = torch.randn(4, 3, 3, 3)
    bias = torch.randn(4)

    output = conv2d_as_gemm_with_precision(
        x,
        weight,
        bias,
        precision="fp32",
        engine="torch",
        padding=1,
        activation="relu",
    )
    expected = F.relu(F.conv2d(x, weight, bias=bias, padding=1))

    assert output.shape == expected.shape
    assert torch.allclose(output, expected, rtol=1e-5, atol=1e-6)


def test_gemm_with_precision_rejects_cross_engine_pattern() -> None:
    a = torch.randn(4, 8)
    b = torch.randn(6, 8)

    with pytest.raises(Exception, match="not compatible"):
        gemm_with_precision(
            a,
            b,
            precision="fp16",
            engine="triton",
            pattern="linear_marlin",
            transpose_b=True,
        )


def test_tilelang_int8_explicit_pattern_is_respected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_run_tilelang_kernel(pattern: str, *args: object, **kwargs: object) -> torch.Tensor:
        captured["pattern"] = pattern
        captured["args"] = args
        captured["kwargs"] = kwargs
        a = args[0]
        weight = args[1]
        assert isinstance(a, torch.Tensor)
        assert isinstance(weight, torch.Tensor)
        return torch.empty(
            (a.shape[0], weight.shape[1]),
            dtype=kwargs["output_dtype"],
        )

    monkeypatch.setattr(
        "xqt.operator_opt.backends.gemm_precision.run_tilelang_kernel",
        fake_run_tilelang_kernel,
    )

    a = torch.randint(-8, 8, (8, 16), dtype=torch.int8)
    b = torch.randint(-8, 8, (12, 16), dtype=torch.int8)
    a_scale = torch.tensor([0.02], dtype=torch.float32)
    b_scale = torch.ones(12, dtype=torch.float32) * 0.01

    output = gemm_with_precision(
        a,
        b,
        precision="int8",
        engine="tilelang",
        pattern="int8_linear",
        transpose_b=True,
        a_scale=a_scale,
        b_scale=b_scale,
    )

    assert output.dtype == torch.float16
    assert captured["pattern"] == "int8_linear"


def test_attention_score_gemm_with_precision_matches_torch_reference() -> None:
    torch.manual_seed(14)
    q = torch.randn(2, 3, 4, 5)
    k = torch.randn(2, 3, 6, 5)

    output = attention_score_gemm_with_precision(
        q,
        k,
        scale=0.5,
        precision="fp32",
        engine="torch",
    )
    expected = torch.matmul(q, k.transpose(-1, -2)) * 0.5

    assert output.shape == (2, 3, 4, 6)
    assert torch.allclose(output, expected)


def test_attention_value_gemm_with_precision_matches_torch_reference() -> None:
    torch.manual_seed(15)
    probabilities = torch.softmax(torch.randn(2, 3, 4, 6), dim=-1)
    value = torch.randn(2, 3, 6, 5)

    output = attention_value_gemm_with_precision(
        probabilities,
        value,
        precision="fp32",
        engine="torch",
    )
    expected = torch.matmul(probabilities, value)

    assert output.shape == (2, 3, 4, 5)
    assert torch.allclose(output, expected)


def test_router_gemm_with_precision_matches_torch_reference() -> None:
    torch.manual_seed(16)
    x = torch.randn(2, 3, 4)
    router_weight = torch.randn(7, 4)
    bias = torch.randn(7)

    output = router_gemm_with_precision(
        x,
        router_weight,
        bias,
        precision="fp32",
        engine="torch",
    )
    expected = torch.matmul(x, router_weight.t()) + bias

    assert output.shape == (2, 3, 7)
    assert torch.allclose(output, expected)


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
    assert captured["args"][1] is b
    assert captured["args"][9] == b.stride(1)
    assert captured["args"][10] == b.stride(0)
    launched_bias = captured["args"][3]
    assert isinstance(launched_bias, torch.Tensor)
    assert launched_bias.dtype == torch.float32


def test_gemm_fp16_triton_applies_resolved_sm89_schedule(
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
    a = torch.empty((1, 4096), dtype=torch.float16)
    b = torch.empty((4096, 4096), dtype=torch.float16, device="meta")

    output = gemm_fp16_triton(
        a,
        b,
        transpose_b=True,
        target_arch="sm_89",
    )

    assert output.shape == (1, 4096)
    kwargs = captured["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["BLOCK_M"] == 16
    assert kwargs["BLOCK_N"] == 64
    assert kwargs["BLOCK_K"] == 64
    assert kwargs["GROUP_M"] == 4
    assert kwargs["num_warps"] == 4
    assert kwargs["num_stages"] == 3


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


def test_gemm_bf16_triton_applies_resolved_sm89_schedule(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_fp16_entry(
        a: torch.Tensor,
        b: torch.Tensor,
        bias: torch.Tensor | None,
        **kwargs: object,
    ) -> torch.Tensor:
        captured["kwargs"] = dict(kwargs)
        return torch.empty((int(a.shape[0]), int(b.shape[0])), dtype=torch.bfloat16)

    monkeypatch.setattr(
        "xqt.operator_opt.kernels.triton.gemm._require_cuda_tensors",
        lambda *args: None,
    )
    monkeypatch.setattr(
        "xqt.operator_opt.kernels.triton.gemm.gemm_fp16_triton",
        fake_fp16_entry,
    )
    a = torch.empty((1, 4096), dtype=torch.bfloat16)
    b = torch.empty((4096, 4096), dtype=torch.bfloat16, device="meta")

    output = gemm_bf16_triton(
        a,
        b,
        transpose_b=True,
        target_arch="sm_89",
    )

    assert output.shape == (1, 4096)
    kwargs = captured["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["block_m"] == 16
    assert kwargs["block_n"] == 64
    assert kwargs["block_k"] == 64
    assert kwargs["group_m"] == 4
    assert kwargs["num_warps"] == 4
    assert kwargs["num_stages"] == 3


def test_gemm_bf16_triton_does_not_query_cuda_capability_for_cpu_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_fp16_entry(
        a: torch.Tensor,
        b: torch.Tensor,
        bias: torch.Tensor | None,
        **kwargs: object,
    ) -> torch.Tensor:
        captured["kwargs"] = dict(kwargs)
        return torch.empty((int(a.shape[0]), int(b.shape[0])), dtype=torch.bfloat16)

    def fail_capability_query(device: torch.device | None = None) -> tuple[int, int]:
        raise AssertionError(f"unexpected CUDA capability query for {device}")

    monkeypatch.setattr(
        "xqt.operator_opt.kernels.triton.gemm._require_cuda_tensors",
        lambda *args: None,
    )
    monkeypatch.setattr(
        "xqt.operator_opt.kernels.triton.gemm.gemm_fp16_triton",
        fake_fp16_entry,
    )
    monkeypatch.setattr(torch.cuda, "get_device_capability", fail_capability_query)
    a = torch.empty((8, 16), dtype=torch.bfloat16)
    b = torch.empty((12, 16), dtype=torch.bfloat16)

    output = gemm_bf16_triton(a, b, transpose_b=True)

    assert output.shape == (8, 12)
    kwargs = captured["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["block_m"] == 128
    assert kwargs["block_n"] == 128
    assert kwargs["block_k"] == 32


def test_triton_registry_exposes_mxfp_gemm_families() -> None:
    for pattern in ("gemm_mxfp8", "gemm_mxfp6", "gemm_mxfp4"):
        spec = get_triton_kernel_spec(pattern)
        assert spec.metadata["precision"] in {"mxfp8", "mxfp6", "mxfp4"}
        assert "mxfp" in spec.pattern


def test_tilelang_registry_exposes_int8_linear_families() -> None:
    linear = get_tilelang_kernel_spec("int8_linear")
    fused = get_tilelang_kernel_spec("int8_linear_static_activation")

    assert linear.metadata["fusion_status"] == "tilelang_dequant_output_epilogue"
    assert (
        fused.metadata["fusion_status"]
        == "tilelang_static_activation_quant_dequant_output_epilogue"
    )


def test_tilelang_registry_exposes_mxfp4_packed_gemm_family() -> None:
    spec = get_tilelang_kernel_spec("mxfp4_packed_dequant_gemm_epilogue")

    assert spec.metadata["weight_encoding"] == "packed_mxfp4_signed_int4"
    assert "mxfp4" in spec.pattern


@pytest.mark.parametrize(
    ("pattern", "expected_pattern"),
    [
        (None, "dense_linear_epilogue"),
        ("linear_marlin", "linear_marlin"),
    ],
)
def test_gemm_with_precision_tilelang_bf16_registry_selection(
    monkeypatch: pytest.MonkeyPatch,
    pattern: str | None,
    expected_pattern: str,
) -> None:
    captured: dict[str, object] = {}

    def fake_run_tilelang_kernel(pattern: str, *args: object, **kwargs: object) -> torch.Tensor:
        captured["pattern"] = pattern
        captured["kwargs"] = kwargs
        a = args[0]
        weight = args[1]
        assert isinstance(a, torch.Tensor)
        assert isinstance(weight, torch.Tensor)
        return torch.empty((a.shape[0], weight.shape[0]), dtype=torch.bfloat16)

    monkeypatch.setattr(
        "xqt.operator_opt.backends.gemm_precision.run_tilelang_kernel",
        fake_run_tilelang_kernel,
    )

    a = torch.randn(8, 16, dtype=torch.bfloat16)
    b = torch.randn(12, 16, dtype=torch.bfloat16)
    bias = torch.randn(12, dtype=torch.bfloat16)

    output = gemm_with_precision(
        a,
        b,
        bias,
        precision="bf16",
        engine="tilelang",
        pattern=pattern,
        transpose_b=True,
    )

    assert output.dtype == torch.bfloat16
    assert captured["pattern"] == expected_pattern
    if expected_pattern == "linear_marlin":
        assert captured["kwargs"]["precision"] == "bf16"


def test_gemm_with_precision_tilelang_int8_static_activation_uses_fused_family(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_run_tilelang_kernel(pattern: str, *args: object, **kwargs: object) -> torch.Tensor:
        captured["pattern"] = pattern
        captured["args"] = args
        captured["kwargs"] = kwargs
        a = args[0]
        weight = args[1]
        assert isinstance(a, torch.Tensor)
        assert isinstance(weight, torch.Tensor)
        return torch.empty((a.shape[0], weight.shape[1]), dtype=torch.float16)

    monkeypatch.setattr(
        "xqt.operator_opt.backends.gemm_precision.run_tilelang_kernel",
        fake_run_tilelang_kernel,
    )

    a = torch.randn(8, 16, dtype=torch.float16)
    b = torch.randint(-8, 8, (12, 16), dtype=torch.int8)
    a_scale = torch.tensor([0.02], dtype=torch.float32)
    b_scale = torch.ones(12, dtype=torch.float32) * 0.01

    output = gemm_with_precision(
        a,
        b,
        precision="int8",
        engine="tilelang",
        transpose_b=True,
        a_scale=a_scale,
        b_scale=b_scale,
    )

    assert output.dtype == torch.float16
    assert captured["pattern"] == "int8_linear_static_activation"


def test_gemm_with_precision_tilelang_int8_marlin_pattern_uses_linear_marlin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_run_tilelang_kernel(
        pattern: str, *args: object, **kwargs: object
    ) -> torch.Tensor:
        captured["pattern"] = pattern
        captured["args"] = args
        captured["kwargs"] = kwargs
        a = args[0]
        weight = args[1]
        assert isinstance(a, torch.Tensor)
        assert isinstance(weight, torch.Tensor)
        return torch.empty((a.shape[0], weight.shape[0]), dtype=torch.float16)

    monkeypatch.setattr(
        "xqt.operator_opt.backends.gemm_precision.run_tilelang_kernel",
        fake_run_tilelang_kernel,
    )

    a = torch.randn(8, 16, dtype=torch.float16)
    b = torch.randint(-8, 8, (12, 16), dtype=torch.int8)
    b_scale = torch.ones(12, 1, 1, dtype=torch.float16) * 0.01

    output = gemm_with_precision(
        a,
        b,
        precision={"activation": "fp16", "weight": "int8", "mma": "int8", "output": "fp16"},
        engine="tilelang",
        pattern="linear_marlin",
        transpose_b=True,
        b_scale=b_scale,
        group_size=16,
    )

    assert output.dtype == torch.float16
    assert captured["pattern"] == "linear_marlin"
    assert captured["kwargs"]["precision"] == "int8"


def test_gemm_with_precision_tilelang_int8_dequant_pattern_uses_dequant_family(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_run_tilelang_kernel(
        pattern: str, *args: object, **kwargs: object
    ) -> torch.Tensor:
        captured["pattern"] = pattern
        captured["args"] = args
        captured["kwargs"] = kwargs
        a = args[0]
        qweight = args[1]
        assert isinstance(a, torch.Tensor)
        assert isinstance(qweight, torch.Tensor)
        return torch.empty((a.shape[0], qweight.shape[0]), dtype=torch.float16)

    monkeypatch.setattr(
        "xqt.operator_opt.backends.gemm_precision.run_tilelang_kernel",
        fake_run_tilelang_kernel,
    )

    a = torch.randn(8, 16, dtype=torch.float16)
    b = torch.randint(-8, 8, (12, 16), dtype=torch.int8)
    b_scale = torch.ones(12, dtype=torch.float16) * 0.01

    output = gemm_with_precision(
        a,
        b,
        precision={"activation": "fp16", "weight": "int8", "mma": "int8", "output": "fp16"},
        engine="tilelang",
        pattern="dequant_gemm_epilogue",
        transpose_b=True,
        b_scale=b_scale,
        activation="relu",
    )

    assert output.dtype == torch.float16
    assert captured["pattern"] == "dequant_gemm_epilogue"


def test_gemm_with_precision_tilelang_int4_marlin_pattern_uses_linear_marlin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_run_tilelang_kernel(
        pattern: str, *args: object, **kwargs: object
    ) -> torch.Tensor:
        captured["pattern"] = pattern
        captured["args"] = args
        captured["kwargs"] = kwargs
        a = args[0]
        weight = args[1]
        assert isinstance(a, torch.Tensor)
        assert isinstance(weight, torch.Tensor)
        return torch.empty((a.shape[0], weight.shape[0]), dtype=torch.float16)

    monkeypatch.setattr(
        "xqt.operator_opt.backends.gemm_precision.run_tilelang_kernel",
        fake_run_tilelang_kernel,
    )

    a = torch.randn(8, 16, dtype=torch.float16)
    b = torch.randint(0, 255, (12, 8), dtype=torch.uint8)
    b_scale = torch.ones(12, 1, 1, dtype=torch.float16) * 0.05

    output = gemm_with_precision(
        a,
        b,
        precision={"activation": "fp16", "weight": "int4", "mma": "int4", "output": "fp16"},
        engine="tilelang",
        pattern="linear_marlin",
        transpose_b=True,
        b_scale=b_scale,
        input_features=16,
        group_size=16,
    )

    assert output.dtype == torch.float16
    assert captured["pattern"] == "linear_marlin"
    assert captured["kwargs"]["precision"] == "int4"


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
