from __future__ import annotations

import torch
import pytest

from xqt.kernels.ops._impl.prepack import (
    INT8_SM89_B_NK,
    list_prepack_specs,
    prepack_weight,
    unpack_weight,
)


def test_list_prepack_specs_includes_placeholders_and_int8() -> None:
    rows = list_prepack_specs()
    keys = {r["layout"] for r in rows}
    assert INT8_SM89_B_NK in keys
    assert "fp4:sm_89:b_mma" in keys
    statuses = {r["layout"]: r["status"] for r in rows}
    assert statuses[INT8_SM89_B_NK] == "implemented"
    assert statuses["fp4:sm_89:b_mma"] == "placeholder"


def test_int8_b_nk_roundtrip() -> None:
    k, n = 128, 256
    w = torch.randint(-8, 8, (k, n), dtype=torch.int8)
    result = prepack_weight(w, INT8_SM89_B_NK)
    assert result.packed.shape == (n, k)
    assert result.math_shape == (k, n)
    back = unpack_weight(result.packed, INT8_SM89_B_NK, math_shape=result.math_shape)
    assert torch.equal(back, w)


def test_placeholder_raises() -> None:
    w = torch.zeros(16, 16, dtype=torch.int8)
    with pytest.raises(Exception):
        prepack_weight(w, "fp4:sm_89:b_mma")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda required")
def test_ptx_prepacked_matches_math() -> None:
    from xqt.kernels.ops._impl.cute.int8mma_binding import (
        int8_linear_ptx_sm89,
        int8mma_available,
        prepack_qweight_t_for_ptx_sm89,
    )

    if not int8mma_available():
        pytest.skip("int8mma so not built")
    major, minor = torch.cuda.get_device_capability()
    if (major, minor) != (8, 9):
        pytest.skip("sm_89 only")

    m, k, n = 64, 256, 256
    a = torch.randint(-8, 8, (m, k), device="cuda", dtype=torch.int8)
    b = torch.randint(-8, 8, (k, n), device="cuda", dtype=torch.int8)
    sw = torch.ones(n, device="cuda") * 0.01
    sa = 0.02
    packed = prepack_qweight_t_for_ptx_sm89(b)
    y0 = int8_linear_ptx_sm89(a, b, sa, sw, None, output_dtype=torch.float16)
    y1 = int8_linear_ptx_sm89(
        a, b, sa, sw, None, output_dtype=torch.float16, prepacked_b=packed
    )
    assert (y0.float() - y1.float()).abs().max().item() < 1e-3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda required")
def test_cutlass_sm89_fused_scale_bias_matches_int_mm() -> None:
    from xqt.kernels.ops._impl.cute.int8mma_binding import (
        int8_linear_cutlass_sm89,
        int8mma_available,
        prepack_qweight_t_for_ptx_sm89,
    )

    if not int8mma_available():
        pytest.skip("int8mma so not built")
    if torch.cuda.get_device_capability() != (8, 9):
        pytest.skip("sm_89 only")

    torch.manual_seed(3)
    m, k, n = 64, 256, 256
    a = torch.randint(-8, 8, (m, k), device="cuda", dtype=torch.int8)
    b = torch.randint(-8, 8, (k, n), device="cuda", dtype=torch.int8)
    activation_scale = torch.tensor(0.02, device="cuda", dtype=torch.float32)
    weight_scale = torch.rand(n, device="cuda", dtype=torch.float32) * 0.02 + 0.001
    bias = torch.randn(n, device="cuda", dtype=torch.float32)
    packed = prepack_qweight_t_for_ptx_sm89(b)
    scale_bias = torch.stack((activation_scale * weight_scale, bias), dim=1).contiguous()

    output = int8_linear_cutlass_sm89(
        a,
        b,
        activation_scale,
        weight_scale,
        bias,
        prepacked_b=packed,
        scale_bias=scale_bias,
    )
    expected = (
        torch._int_mm(a, b).float()
        * (activation_scale * weight_scale).view(1, -1)
        + bias.view(1, -1)
    ).half()

    assert output.dtype == torch.float16
    assert (output.float() - expected.float()).abs().max().item() < 2e-3
