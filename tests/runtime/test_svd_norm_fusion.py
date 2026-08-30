from __future__ import annotations

import pytest
import torch
from torch import nn

from tests.xqt.svd_test_helpers import make_legacy_svd_linear
from xqt.runtime.modules import SVDQuantLinear


def _require_native_w4a4() -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 9):
        pytest.skip("native sm_89 W4A4 backend unavailable")
    try:
        from xqt.kernels.ops._impl.cute.svdq_w4a4_sm89 import (
            native_w4a4_available,
        )
    except Exception:
        pytest.skip("native sm_89 W4A4 backend unavailable")
    if not native_w4a4_available(build=False):
        pytest.skip("native sm_89 W4A4 backend unavailable")


def _rel_rmse(a: torch.Tensor, b: torch.Tensor) -> float:
    diff = (a.float() - b.float()).pow(2).mean().sqrt()
    base = b.float().pow(2).mean().sqrt().clamp_min(1e-12)
    return float(diff / base)


def _make_pair(
    in_features: int = 256,
    out_features: int = 256,
    rank: int = 16,
    dtype: torch.dtype = torch.float16,
) -> tuple[SVDQuantLinear, SVDQuantLinear, torch.Tensor, float]:
    torch.manual_seed(7)
    eps = 1e-6
    base = nn.Linear(in_features, out_features, bias=True, dtype=dtype, device="cuda")
    with torch.no_grad():
        base.weight.mul_(0.05)
    module_ref = make_legacy_svd_linear(
        base,
        rank=rank,
        group_size=64,
    ).to(dtype)
    module_fused = make_legacy_svd_linear(
        base,
        rank=rank,
        group_size=64,
    ).to(dtype)
    norm_weight = torch.randn(in_features, dtype=dtype, device="cuda") * 0.5 + 1.0
    module_fused.set_fused_norm(norm_weight, eps=eps)
    return module_ref, module_fused, norm_weight, eps


def _rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    row_scale = torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + eps)
    return (x.float() * row_scale * weight.float()).to(x.dtype)


def test_fused_norm_reference_fallback_applies_norm_explicitly() -> None:
    _require_native_w4a4()
    module_ref, module_fused, norm_weight, eps = _make_pair()
    x = torch.randn(256, 256, dtype=torch.float16, device="cuda") * 2.0

    expected = module_ref(_rmsnorm(x, norm_weight, eps))
    actual = module_fused(x)

    assert _rel_rmse(actual, expected) < 1e-3
    assert module_fused.execution_metadata()["fused_norm_enabled"] is True


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fused_norm_native_path_matches_unfused_native_path(
    dtype: torch.dtype,
) -> None:
    _require_native_w4a4()
    module_ref, module_fused, norm_weight, eps = _make_pair(dtype=dtype)
    module_ref.enable_fusion()
    module_fused.enable_fusion()
    x = torch.randn(256, 256, dtype=dtype, device="cuda") * 2.0

    with torch.no_grad():
        expected = module_ref(_rmsnorm(x, norm_weight, eps))
        actual = module_fused(x)

    metadata = module_fused.execution_metadata()
    assert metadata["cuda_fused_backend"] == "native_w4a4_dynamic_norm"
    assert metadata["implementation"] == "native_svdq_w4a4_dynamic_norm_fused_lora"
    # The fused norm kernel and explicit RMSNorm can differ by one FP16 ULP.
    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fused_norm_native_smalln_path_matches_unfused_smalln_path(
    dtype: torch.dtype,
) -> None:
    _require_native_w4a4()
    module_ref, module_fused, norm_weight, eps = _make_pair(
        out_features=512,
        dtype=dtype,
    )
    module_ref.enable_fusion()
    module_fused.enable_fusion()
    x = torch.randn(256, 256, dtype=dtype, device="cuda") * 2.0

    with torch.no_grad():
        expected = module_ref(_rmsnorm(x, norm_weight, eps))
        actual = module_fused(x)

    reference_metadata = module_ref.execution_metadata()
    metadata = module_fused.execution_metadata()
    assert reference_metadata["cuda_fused_backend"] == "native_w4a4_dynamic_smalln"
    assert metadata["cuda_fused_backend"] == "native_w4a4_dynamic_smalln_norm"
    assert metadata["implementation"] == "native_svdq_w4a4_dynamic_norm_fused_lora_smalln"
    assert metadata["cuda_fused_fallback_reason"] is None
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_fused_norm_native_path_captures_cuda_graph() -> None:
    _require_native_w4a4()
    module_ref, module_fused, norm_weight, eps = _make_pair()
    module_ref.enable_fusion()
    module_fused.enable_fusion(cuda_graph=True)
    x = torch.randn(256, 256, dtype=torch.float16, device="cuda") * 2.0

    with torch.no_grad():
        expected = module_ref(_rmsnorm(x, norm_weight, eps))
        first = module_fused(x).clone()
        second = module_fused(x).clone()

    metadata = module_fused.execution_metadata()
    assert metadata["cuda_graph_used"] is True
    assert _rel_rmse(first, expected) < 2e-2
    assert _rel_rmse(second, first) < 1e-5


def test_fused_norm_native_path_falls_back_when_autograd_is_required() -> None:
    _require_native_w4a4()
    _, module_fused, _, _ = _make_pair(out_features=512)
    module_fused.enable_fusion()
    x = torch.randn(
        16,
        256,
        dtype=torch.float16,
        device="cuda",
        requires_grad=True,
    )

    with torch.no_grad():
        _ = module_fused(x.detach())
    assert (
        module_fused.execution_metadata()["cuda_fused_backend"]
        == "native_w4a4_dynamic_smalln_norm"
    )

    output = module_fused(x)
    output.float().square().mean().backward()

    metadata = module_fused.execution_metadata()
    assert output.requires_grad is True
    assert output.grad_fn is not None
    assert x.grad is not None
    assert module_fused.down_proj.weight.grad is not None
    assert module_fused.up_proj.weight.grad is not None
    assert metadata["cuda_fused_used"] is False
    assert metadata["cuda_fused_backend"] is None
    assert "forward-only when autograd is required" in str(
        metadata["cuda_fused_fallback_reason"]
    )


def test_clear_fused_norm_restores_post_norm_input_contract() -> None:
    _require_native_w4a4()
    module_ref, module_fused, norm_weight, eps = _make_pair()
    module_fused.clear_fused_norm()
    x = torch.randn(256, 256, dtype=torch.float16, device="cuda") * 2.0

    expected = module_ref(_rmsnorm(x, norm_weight, eps))
    actual = module_fused(_rmsnorm(x, norm_weight, eps))

    assert _rel_rmse(actual, expected) < 1e-3
    assert module_fused.execution_metadata()["fused_norm_enabled"] is False


def test_set_fused_norm_validates_weight_shape() -> None:
    module = make_legacy_svd_linear(
        nn.Linear(16, 8),
        rank=4,
        group_size=8,
    )

    with pytest.raises(ValueError, match="input_features"):
        module.set_fused_norm(torch.ones(8))
