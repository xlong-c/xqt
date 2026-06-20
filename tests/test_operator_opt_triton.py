import pytest
import torch
import torch.nn.functional as F

from xqt.core.errors import XQTBackendError
from xqt.operator_opt.backends.triton import (
    get_triton_kernel_spec,
    list_triton_kernel_specs,
    run_triton_kernel,
)
from xqt.operator_opt.kernels.triton import (
    fused_bias_gelu_reference,
    fused_bias_gelu_triton,
    fused_rope_reference,
    fused_rope_triton,
    fused_rmsnorm_residual_reference,
    fused_rmsnorm_residual_triton,
    fused_swiglu_reference,
    fused_swiglu_triton,
)


def test_triton_references_match_pytorch_ops() -> None:
    x = torch.randn(2, 4)
    bias = torch.randn(4)
    gate = torch.randn(2, 4)
    up = torch.randn(2, 4)
    residual = torch.randn(2, 4)
    weight = torch.randn(4)
    cos = torch.randn(2, 2)
    sin = torch.randn(2, 2)

    assert torch.allclose(fused_bias_gelu_reference(x, bias), F.gelu(x + bias))
    assert torch.allclose(fused_swiglu_reference(gate, up), F.silu(gate) * up)
    expected_rms = (x + residual) * torch.rsqrt((x + residual).pow(2).mean(dim=-1, keepdim=True) + 1e-6) * weight
    assert torch.allclose(
        fused_rmsnorm_residual_reference(x, residual, weight),
        expected_rms,
    )
    assert fused_rope_reference(x, cos, sin).shape == x.shape


def test_triton_registry_records_kernel_metadata() -> None:
    specs = list_triton_kernel_specs()

    assert set(specs) == {"bias_gelu", "rope", "rmsnorm_residual", "swiglu"}
    assert specs["swiglu"]["metadata"]["block_size"] > 0
    assert specs["swiglu"]["metadata"]["num_warps"] > 0
    assert specs["swiglu"]["metadata"]["num_stages"] > 0
    assert specs["swiglu"]["metadata"]["autotune_key"]
    assert get_triton_kernel_spec("swiglu").fallback == "eager"


def test_triton_backend_uses_eager_fallback_on_cpu() -> None:
    gate = torch.randn(2, 4)
    up = torch.randn(2, 4)

    output = run_triton_kernel("swiglu", gate, up, fallback="eager")

    assert torch.allclose(output, F.silu(gate) * up)


def test_triton_backend_requires_cuda_without_fallback() -> None:
    gate = torch.randn(2, 4)
    up = torch.randn(2, 4)

    with pytest.raises(XQTBackendError, match="requires CUDA tensors"):
        run_triton_kernel("swiglu", gate, up, fallback="raise")


def test_triton_kernel_entry_rejects_cpu_tensors() -> None:
    gate = torch.randn(2, 4)
    up = torch.randn(2, 4)
    x = torch.randn(2, 4)
    bias = torch.randn(4)
    residual = torch.randn(2, 4)
    weight = torch.randn(4)
    cos = torch.randn(2)
    sin = torch.randn(2)

    with pytest.raises(XQTBackendError, match="CUDA tensors"):
        fused_swiglu_triton(gate, up)
    with pytest.raises(XQTBackendError, match="CUDA tensors"):
        fused_bias_gelu_triton(x, bias)
    with pytest.raises(XQTBackendError, match="CUDA tensors"):
        fused_rmsnorm_residual_triton(x, residual, weight)
    with pytest.raises(XQTBackendError, match="CUDA tensors"):
        fused_rope_triton(x, cos, sin)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_triton_backend_cuda_smoke_matches_reference() -> None:
    gate = torch.randn(2, 4, device="cuda")
    up = torch.randn(2, 4, device="cuda")
    x = torch.randn(2, 4, device="cuda")
    bias = torch.randn(4, device="cuda")
    residual = torch.randn(2, 4, device="cuda")
    weight = torch.randn(4, device="cuda")
    cos = torch.randn(2, device="cuda")
    sin = torch.randn(2, device="cuda")

    swiglu = run_triton_kernel("swiglu", gate, up)
    bias_gelu = run_triton_kernel("bias_gelu", x, bias)
    rmsnorm = run_triton_kernel("rmsnorm_residual", x, residual, weight)
    rope = run_triton_kernel("rope", x, cos, sin)

    assert torch.allclose(swiglu, F.silu(gate) * up, atol=1e-4, rtol=1e-4)
    assert torch.allclose(bias_gelu, F.gelu(x + bias), atol=1e-4, rtol=1e-4)
    assert torch.allclose(
        rmsnorm,
        fused_rmsnorm_residual_reference(x, residual, weight),
        atol=1e-4,
        rtol=1e-4,
    )
    assert torch.allclose(
        rope,
        fused_rope_reference(x, cos, sin),
        atol=1e-4,
        rtol=1e-4,
    )
