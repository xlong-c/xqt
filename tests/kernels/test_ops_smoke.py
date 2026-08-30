"""Ops smoke - real BaseFusedOp pilots and registry targets."""

from __future__ import annotations

import pytest
import torch

from xqt.kernels import KernelBackend
from xqt.kernels.registry import registry
from xqt.kernels.selector import clear_cache, get_kernel
from xqt.kernels.spec import FormatSignature

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda required")


def test_registry_contains_expected_ops() -> None:
    import xqt.kernels.ops  # noqa: F401  # trigger group registration

    ops = set(registry.ops())
    assert "gemm.bmm_fp8" in ops
    assert "gemm.gemm_bf16" in ops
    assert "attention.fused_attention" in ops
    assert "quantization.svd_fused_dequant_gemm_low_rank" in ops
    assert "layernorm.rmsnorm" in ops
    assert "activation.silu_and_mul" in ops


def test_rmsnorm_fused_op_importable() -> None:
    from xqt.kernels.ops.layernorm import RMSNormOp, _RMSNORM

    assert RMSNormOp.op == "layernorm.rmsnorm"
    assert _RMSNORM is not None


def test_silu_fused_op_importable() -> None:
    from xqt.kernels.ops.activation import SiluAndMulOp, _SILU

    assert SiluAndMulOp.op == "activation.silu_and_mul"
    assert _SILU is not None


def test_rmsnorm_native_matches_torch() -> None:
    from xqt.kernels.ops.layernorm import _rmsnorm_torch

    x = torch.randn(4, 16)
    w = torch.randn(16)
    out = _rmsnorm_torch(x, w, eps=1e-5)
    ref = torch.nn.functional.rms_norm(x, (16,), w, eps=1e-5)
    assert torch.allclose(out, ref, atol=1e-5)


def test_silu_native_matches_torch() -> None:
    from xqt.kernels.ops.activation import _silu_and_mul_torch

    x = torch.randn(4, 16)
    out = _silu_and_mul_torch(x)
    gate, up = x[..., :8], x[..., 8:]
    assert torch.allclose(out, torch.nn.functional.silu(gate) * up)


def test_base_fused_op_dispatch_native_cpu() -> None:
    from xqt.kernels.ops.layernorm import _RMSNORM

    x = torch.randn(2, 8)
    w = torch.randn(8)
    out = _RMSNORM(x, w, backend=KernelBackend.TORCH)
    assert out.shape == (2, 8)


def test_get_kernel_resolves_real_gemm() -> None:
    from xqt.kernels.selector import get_kernel

    fn = get_kernel("gemm.gemm_fp16", KernelBackend.TORCH)
    assert callable(fn)
    clear_cache()


def test_cuda_gemm_triton_target_resolves() -> None:
    from xqt.kernels.spec import KernelSpec

    spec = KernelSpec(
        op="gemm.gemm_fp16",
        backend=KernelBackend.TRITON,
        target="xqt.kernels.ops._impl.triton.gemm:gemm_fp16_triton",
    )
    fn = spec.load()
    assert callable(fn)


def test_tilelang_public_targets_resolve() -> None:
    import xqt.kernels.ops  # noqa: F401  # trigger group registration

    attention = get_kernel("attention.fused_attention", KernelBackend.TILELANG)
    svd_fused = get_kernel(
        "quantization.svd_fused_dequant_gemm_low_rank",
        KernelBackend.TILELANG,
    )
    assert callable(attention)
    assert callable(svd_fused)
    clear_cache()
