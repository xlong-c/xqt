from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from tests.xqt.svd_test_helpers import make_legacy_svd_linear
from xqt.quant.quantizers.svd import quantize_with_svd
from xqt.runtime import (
    fuse_composite_modules,
    materialize_svd_for_inference,
    materialize_svd_gelu_mlps,
)
from xqt.runtime.modules import (
    CompositeAddW4A4Linear,
    SVDQuantGeluMLP,
    SVDQuantLinear,
)


def _require_native_w4a4() -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 9):
        pytest.skip("native sm_89 W4A4 backend unavailable")
    try:
        from xqt.operator_opt.kernels.cute.svdq_w4a4_sm89 import (
            native_w4a4_available,
        )
    except Exception:
        pytest.skip("native sm_89 W4A4 backend unavailable")
    if not native_w4a4_available(build=False):
        pytest.skip("native sm_89 W4A4 backend unavailable")


def _make_linear(
    input_features: int,
    output_features: int,
    *,
    rank: int,
    dtype: torch.dtype,
    device: str,
) -> SVDQuantLinear:
    base = nn.Linear(
        input_features,
        output_features,
        bias=True,
        dtype=dtype,
        device=device,
    )
    with torch.no_grad():
        base.weight.mul_(0.05)
        base.bias.mul_(0.05)
    return make_legacy_svd_linear(
        base,
        rank=rank,
        group_size=64,
        quant_dtype="int4",
    ).to(device=device, dtype=dtype)


def _make_mlp(
    *,
    dtype: torch.dtype,
    device: str,
    native_fusion: bool = True,
) -> SVDQuantGeluMLP:
    torch.manual_seed(71)
    fc1 = _make_linear(128, 256, rank=16, dtype=dtype, device=device)
    fc2 = _make_linear(256, 128, rank=16, dtype=dtype, device=device)
    return SVDQuantGeluMLP(fc1, fc2, native_fusion=native_fusion)


def test_constructor_rejects_hidden_feature_mismatch() -> None:
    fc1 = _make_linear(8, 16, rank=4, dtype=torch.float32, device="cpu")
    fc2 = _make_linear(12, 8, rank=4, dtype=torch.float32, device="cpu")

    with pytest.raises(ValueError, match="output_features"):
        SVDQuantGeluMLP(fc1, fc2)


def test_constructor_rejects_unknown_gelu_approximation() -> None:
    fc1 = _make_linear(8, 16, rank=4, dtype=torch.float32, device="cpu")
    fc2 = _make_linear(16, 8, rank=4, dtype=torch.float32, device="cpu")

    with pytest.raises(ValueError, match="approximate"):
        SVDQuantGeluMLP(fc1, fc2, approximate="fast")


def test_cpu_autograd_fallback_matches_sequential_and_backpropagates() -> None:
    module = _make_mlp(dtype=torch.float32, device="cpu")
    inputs = torch.randn(3, 5, 128, requires_grad=True)

    expected = module.fc2(F.gelu(module.fc1(inputs), approximate="tanh"))
    actual = module(inputs)

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    actual.square().mean().backward()
    assert inputs.grad is not None
    assert module.fc1.down_proj.weight.grad is not None
    assert module.fc2.up_proj.weight.grad is not None
    metadata = module.execution_metadata()
    assert metadata["fused_gelu_mlp_used"] is False
    assert metadata["implementation"] == "sequential_svdq_gelu_mlp"
    assert "CUDA" in str(metadata["fallback_reason"])
    assert metadata["gelu_approximate"] == "tanh"


def test_exact_gelu_is_preserved_by_sequential_fallback() -> None:
    base = _make_mlp(dtype=torch.float32, device="cpu")
    module = SVDQuantGeluMLP(
        base.fc1,
        base.fc2,
        approximate="none",
    )
    inputs = torch.randn(3, 128)

    expected = module.fc2(F.gelu(module.fc1(inputs), approximate="none"))
    actual = module(inputs)

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    assert module.execution_metadata()["gelu_approximate"] == "none"


def test_disabled_fusion_reports_explicit_fallback() -> None:
    module = _make_mlp(
        dtype=torch.float32,
        device="cpu",
        native_fusion=False,
    )
    inputs = torch.randn(4, 128)

    module(inputs)

    metadata = module.execution_metadata()
    assert metadata["native_fusion_enabled"] is False
    assert metadata["fused_gelu_mlp_used"] is False
    assert metadata["fallback_reason"] == "native fused GELU MLP is disabled"


def test_generic_fusion_entry_accepts_mode_for_gelu_mlp() -> None:
    module = _make_mlp(dtype=torch.float32, device="cpu", native_fusion=False)

    fused_count = fuse_composite_modules(module, mode="reduce-overhead")

    assert isinstance(fused_count, int)
    assert fused_count >= 0
    assert module.execution_metadata()["native_fusion_enabled"] is True


def _quantized_diffusers_feedforward(
    *,
    activation_fn: str,
    dropout: float = 0.0,
) -> nn.Module:
    attention = pytest.importorskip("diffusers.models.attention")
    feedforward = attention.FeedForward(
        8,
        inner_dim=16,
        activation_fn=activation_fn,
        dropout=dropout,
    ).eval()
    result = quantize_with_svd(
        feedforward,
        strategy="w4a16_int4",
        compute="dequant_fp16",
        rank=4,
        group_size=8,
        quant_dtype="int4",
        inplace=False,
    )
    return result.model


def test_materialize_diffusers_tanh_gelu_feedforward_matches_quantized_module() -> None:
    module = _quantized_diffusers_feedforward(activation_fn="gelu-approximate")
    inputs = torch.randn(2, 3, 8)

    expected = module(inputs)
    materialized = materialize_svd_gelu_mlps(module, inplace=False)
    actual = materialized(inputs, scale=1.0)

    assert isinstance(materialized, SVDQuantGeluMLP)
    assert materialized.approximate == "tanh"
    assert isinstance(materialized.fc1, CompositeAddW4A4Linear)
    assert isinstance(materialized.fc2, CompositeAddW4A4Linear)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_materialize_diffusers_nested_feedforward_replaces_parent_path() -> None:
    class Block(nn.Module):
        def __init__(self, ff: nn.Module) -> None:
            super().__init__()
            self.ff = ff

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return self.ff(inputs)

    block = Block(
        _quantized_diffusers_feedforward(activation_fn="gelu-approximate")
    ).eval()
    inputs = torch.randn(2, 3, 8)

    expected = block(inputs)
    materialized = materialize_svd_gelu_mlps(block, inplace=False)
    actual = materialized(inputs)

    assert isinstance(materialized.ff, SVDQuantGeluMLP)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_materialize_svd_for_inference_can_opt_in_gelu_mlp_rewrite() -> None:
    attention = pytest.importorskip("diffusers.models.attention")
    feedforward = attention.FeedForward(
        8,
        inner_dim=16,
        activation_fn="gelu-approximate",
    ).eval()
    result = quantize_with_svd(
        feedforward,
        strategy="w4a16_int4",
        compute="dequant_fp16",
        rank=4,
        group_size=8,
        quant_dtype="int4",
        inplace=False,
    )
    inputs = torch.randn(2, 3, 8)
    expected = result.model(inputs)

    materialized = materialize_svd_for_inference(
        result.model,
        result.compute_config,
        fuse_gelu_mlp=True,
        inplace=False,
    )
    actual = materialized(inputs)

    assert isinstance(materialized, SVDQuantGeluMLP)
    assert not isinstance(result.model, SVDQuantGeluMLP)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_materialize_diffusers_exact_gelu_keeps_original_graph() -> None:
    module = _quantized_diffusers_feedforward(activation_fn="gelu")

    materialized = materialize_svd_gelu_mlps(module, inplace=False)

    assert not isinstance(materialized, SVDQuantGeluMLP)
    assert getattr(materialized.net[0], "approximate") == "none"


def test_materialize_diffusers_nonzero_dropout_keeps_original_graph() -> None:
    module = _quantized_diffusers_feedforward(
        activation_fn="gelu-approximate",
        dropout=0.1,
    )

    materialized = materialize_svd_gelu_mlps(module, inplace=False)

    assert not isinstance(materialized, SVDQuantGeluMLP)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_native_fused_wrapper_matches_bound_runner(dtype: torch.dtype) -> None:
    _require_native_w4a4()
    from xqt.operator_opt.kernels.cute.svdq_w4a4_sm89 import (
        allocate_svdq_w4a4_gelu_mlp_workspace,
        bind_svdq_w4a4_gelu_mlp,
    )

    module = _make_mlp(dtype=dtype, device="cuda").eval()
    inputs = torch.randn(1, 64, 128, device="cuda", dtype=dtype)
    flat = inputs.reshape(-1, 128)
    packed_fc1 = module.fc1._native_w4a4_packed(flat, smalln=False)
    packed_fc2 = module.fc2._native_w4a4_packed(flat, smalln=False)
    workspace = allocate_svdq_w4a4_gelu_mlp_workspace(
        int(flat.shape[0]),
        packed_fc1,
        packed_fc2,
    )
    bound = bind_svdq_w4a4_gelu_mlp(
        packed_fc1,
        packed_fc2,
        workspace,
        rows=int(flat.shape[0]),
    )

    with torch.no_grad():
        expected = bound(flat).reshape(1, 64, 128).clone()
        actual = module(inputs).clone()

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    metadata = module.execution_metadata()
    assert metadata["implementation"] == "native_svdq_w4a4_gelu_mlp"
    assert metadata["fused_gelu_mlp_used"] is True
    assert metadata["fallback_reason"] is None


def test_native_fused_wrapper_supports_two_dimensional_inputs() -> None:
    _require_native_w4a4()
    module = _make_mlp(dtype=torch.float16, device="cuda").eval()
    inputs = torch.randn(64, 128, device="cuda", dtype=torch.float16)

    with torch.no_grad():
        output = module(inputs)

    assert output.shape == (64, 128)
    assert module.execution_metadata()["fused_gelu_mlp_used"] is True


def test_native_cache_rebuilds_after_parameter_mutation() -> None:
    _require_native_w4a4()
    module = _make_mlp(dtype=torch.float16, device="cuda").eval()
    inputs = torch.randn(64, 128, device="cuda", dtype=torch.float16)

    with torch.no_grad():
        before = module(inputs).clone()
        first_runner = next(iter(module._native_hot_cache.values()))[4]
        module.fc1.up_proj.weight.add_(0.25)
        after = module(inputs).clone()
        second_runner = next(iter(module._native_hot_cache.values()))[4]

    assert second_runner is not first_runner
    assert not torch.equal(before, after)
    assert module.execution_metadata()["fused_gelu_mlp_used"] is True


def test_cuda_autograd_uses_fallback_and_preserves_gradients() -> None:
    _require_native_w4a4()
    module = _make_mlp(dtype=torch.float16, device="cuda").train()
    inputs = torch.randn(
        64,
        128,
        device="cuda",
        dtype=torch.float16,
        requires_grad=True,
    )

    output = module(inputs)
    output.float().square().mean().backward()

    assert inputs.grad is not None
    metadata = module.execution_metadata()
    assert metadata["fused_gelu_mlp_used"] is False
    assert "autograd" in str(metadata["fallback_reason"])
