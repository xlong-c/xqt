"""Tests for xqt.operator_opt.backends.gemm_selector.

Style mirrors test_prepack_int8_sm89.py: function-level tests, no fixture
indirection, explicit torch.cuda skips.
"""

from __future__ import annotations

import torch
import pytest

from xqt.contracts import PrecisionPolicy
from xqt.operator_opt.backends.gemm_precision import gemm_with_precision
from xqt.operator_opt.backends.gemm_selector import (
    GemmShape,
    select_gemm_engine,
)

_ALIGNED_SHAPE = GemmShape(m=128, n=128, k=128)
_UNALIGNED_SHAPE = GemmShape(m=100, n=100, k=100)
_LARGE_ALIGNED_SHAPE = GemmShape(m=512, n=4096, k=4096)

_NON_INT8_PRECISIONS = [
    "fp16",
    "bf16",
    "fp8",
    "int4",
    "fp4",
    "nvfp4",
    "mxfp8",
    "mxfp6",
    "mxfp4",
]


def _cuda_device_or_skip() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("cuda required")
    return torch.device("cuda")


def _legacy_select_engine(precision: str, device: torch.device) -> str:
    """Snapshot of the pre-refactor _select_engine logic. Only the non-int8
    precisions are expected to stay behavior-identical; int8 intentionally
    diverges in Phase 2 (tilelang is now dispatchable for pre-quantized
    aligned shapes).
    """
    if not device.type == "cuda":
        if precision in {"fp4", "nvfp4"}:
            return "tilelang"
        return "torch"
    if precision in {"fp16", "bf16", "int8", "fp8", "mxfp8", "mxfp6", "mxfp4"}:
        return "triton"
    if precision == "int4":
        return "triton"
    if precision in {"fp4", "nvfp4"}:
        return "tilelang"
    return "torch"


def test_dense_fp16_selects_torch_on_cpu() -> None:
    selection = select_gemm_engine(
        precision=PrecisionPolicy(mma="fp16"),
        shape=_ALIGNED_SHAPE,
        device=torch.device("cpu"),
    )
    assert selection.selected_engine == "torch"


def test_dense_fp16_selects_triton_on_cuda() -> None:
    device = _cuda_device_or_skip()
    selection = select_gemm_engine(
        precision=PrecisionPolicy(mma="fp16"), shape=_ALIGNED_SHAPE, device=device
    )
    assert selection.selected_engine == "triton"


def test_dense_bf16_selects_triton_on_cuda() -> None:
    device = _cuda_device_or_skip()
    selection = select_gemm_engine(
        precision=PrecisionPolicy(mma="bf16"), shape=_ALIGNED_SHAPE, device=device
    )
    assert selection.selected_engine == "triton"


def test_dense_fp16_tilelang_candidate_no_caveat_when_aligned() -> None:
    device = _cuda_device_or_skip()
    selection = select_gemm_engine(
        precision=PrecisionPolicy(mma="fp16"), shape=_ALIGNED_SHAPE, device=device
    )
    tilelang = next(c for c in selection.candidates if c.engine == "tilelang")
    assert tilelang.dispatchable_by_gemm_with_precision is True
    assert tilelang.caveats == ()


def test_dense_fp16_tilelang_candidate_caveat_when_unaligned() -> None:
    device = _cuda_device_or_skip()
    selection = select_gemm_engine(
        precision=PrecisionPolicy(mma="fp16"), shape=_UNALIGNED_SHAPE, device=device
    )
    tilelang = next(c for c in selection.candidates if c.engine == "tilelang")
    assert tilelang.caveats != ()


def test_int8_prequantized_aligned_selects_tilelang() -> None:
    # Phase 2: int8 with pre-quantized a/b (no activation_quant in fused_ops)
    # and shape aligned to 64 now dispatches to the real TileLang W8A8 kernel.
    device = _cuda_device_or_skip()
    selection = select_gemm_engine(
        precision=PrecisionPolicy(mma="int8"),
        shape=_LARGE_ALIGNED_SHAPE,
        device=device,
        fused_ops=frozenset(),
    )
    assert selection.selected_engine == "tilelang"


def test_int8_prequantized_unaligned_still_selects_triton() -> None:
    # Pre-quantized but shape not aligned: TileLang would raise inside
    # _validate_int8_mma_inputs, so selector must fall back to triton.
    device = _cuda_device_or_skip()
    selection = select_gemm_engine(
        precision=PrecisionPolicy(mma="int8"),
        shape=_UNALIGNED_SHAPE,
        device=device,
        fused_ops=frozenset(),
    )
    assert selection.selected_engine == "triton"


def test_int8_fused_activation_quant_still_selects_triton() -> None:
    # activation_quant in fused_ops means a.dtype != torch.int8 (caller hasn't
    # pre-quantized). _gemm_tilelang int8 branch requires torch.int8 a/b, so
    # the only dispatchable path through gemm_with_precision is still triton.
    device = _cuda_device_or_skip()
    selection = select_gemm_engine(
        precision=PrecisionPolicy(mma="int8"),
        shape=_ALIGNED_SHAPE,
        device=device,
        fused_ops=frozenset({"activation_quant"}),
    )
    assert selection.selected_engine == "triton"


def test_int4_selects_triton_on_cuda() -> None:
    device = _cuda_device_or_skip()
    selection = select_gemm_engine(
        precision=PrecisionPolicy(mma="int4"), shape=_ALIGNED_SHAPE, device=device
    )
    assert selection.selected_engine == "triton"


def test_fp4_selects_tilelang_on_cpu() -> None:
    selection = select_gemm_engine(
        precision=PrecisionPolicy(mma="fp4"),
        shape=_ALIGNED_SHAPE,
        device=torch.device("cpu"),
    )
    assert selection.selected_engine == "tilelang"


def test_fp4_selects_tilelang_on_cuda() -> None:
    device = _cuda_device_or_skip()
    selection = select_gemm_engine(
        precision=PrecisionPolicy(mma="fp4"), shape=_ALIGNED_SHAPE, device=device
    )
    assert selection.selected_engine == "tilelang"


def test_nvfp4_selects_tilelang_on_cuda() -> None:
    device = _cuda_device_or_skip()
    selection = select_gemm_engine(
        precision=PrecisionPolicy(mma="nvfp4"), shape=_ALIGNED_SHAPE, device=device
    )
    assert selection.selected_engine == "tilelang"


def test_ptx_sm89_candidate_present_but_not_dispatchable() -> None:
    device = _cuda_device_or_skip()
    selection = select_gemm_engine(
        precision=PrecisionPolicy(mma="int8"),
        shape=_LARGE_ALIGNED_SHAPE,
        device=device,
    )
    ptx_candidates = [c for c in selection.candidates if c.engine == "ptx_sm89"]
    assert len(ptx_candidates) == 1
    assert ptx_candidates[0].dispatchable_by_gemm_with_precision is False
    assert selection.selected_engine != "ptx_sm89"


def test_tilelang_int8_candidate_is_dispatchable_in_p2() -> None:
    # Phase 2: _gemm_tilelang now has a real int8 branch; the tilelang
    # candidate must be marked dispatchable (flipped from P1's False).
    device = _cuda_device_or_skip()
    selection = select_gemm_engine(
        precision=PrecisionPolicy(mma="int8"), shape=_ALIGNED_SHAPE, device=device
    )
    tilelang_candidates = [c for c in selection.candidates if c.engine == "tilelang"]
    assert len(tilelang_candidates) == 1
    assert tilelang_candidates[0].dispatchable_by_gemm_with_precision is True


def test_ptx_sm89_candidate_maturity_reflects_sm89_hardware() -> None:
    if not torch.cuda.is_available():
        pytest.skip("cuda required")
    major, minor = torch.cuda.get_device_capability()
    if (major, minor) != (8, 9):
        pytest.skip("sm_89 only")
    selection = select_gemm_engine(
        precision=PrecisionPolicy(mma="int8"),
        shape=_LARGE_ALIGNED_SHAPE,
        device=torch.device("cuda"),
    )
    ptx_candidate = next(c for c in selection.candidates if c.engine == "ptx_sm89")
    assert ptx_candidate.maturity == "executable"


def test_goal_rejects_unknown_value() -> None:
    with pytest.raises(ValueError):
        select_gemm_engine(
            precision=PrecisionPolicy(mma="fp16"),
            shape=_ALIGNED_SHAPE,
            device=torch.device("cpu"),
            goal="not_a_real_goal",
        )


@pytest.mark.parametrize("precision_name", _NON_INT8_PRECISIONS)
@pytest.mark.parametrize(
    "device",
    [torch.device("cpu"), torch.device("cuda")],
    ids=["cpu", "cuda"],
)
def test_selected_engine_matches_legacy_behavior_for_non_int8_precisions(
    precision_name: str, device: torch.device
) -> None:
    # Phase 2 intentionally changes int8 behavior; only non-int8 precisions
    # must remain legacy-identical.
    if device.type == "cuda" and not torch.cuda.is_available():
        pytest.skip("cuda required")
    expected = _legacy_select_engine(precision_name, device)
    selection = select_gemm_engine(
        precision=PrecisionPolicy(mma=precision_name),
        shape=_ALIGNED_SHAPE,
        device=device,
    )
    assert selection.selected_engine == expected


def test_gemm_with_precision_auto_matches_explicit_triton_for_fp16() -> None:
    device = _cuda_device_or_skip()
    torch.manual_seed(0)
    a = torch.randn(64, 64, device=device, dtype=torch.float16)
    b = torch.randn(64, 64, device=device, dtype=torch.float16)

    auto_output = gemm_with_precision(a, b, precision="fp16", engine="auto")
    explicit_output = gemm_with_precision(a, b, precision="fp16", engine="triton")

    assert torch.equal(auto_output, explicit_output)


def test_gemm_with_precision_auto_routes_int8_prequantized_to_tilelang() -> None:
    # End-to-end: gemm_with_precision(engine="auto") routes pre-quantized
    # int8 through the new _gemm_tilelang int8 branch.
    device = _cuda_device_or_skip()
    m, n, k = 64, 64, 64
    a = torch.randint(-8, 8, (m, k), device=device, dtype=torch.int8)
    b = torch.randint(-8, 8, (n, k), device=device, dtype=torch.int8)
    a_scale = torch.tensor([0.02], device=device)
    b_scale = torch.ones(n, device=device) * 0.01

    auto_out = gemm_with_precision(
        a, b, precision="int8", engine="auto",
        a_scale=a_scale, b_scale=b_scale, transpose_b=True,
    )
    explicit_out = gemm_with_precision(
        a, b, precision="int8", engine="tilelang",
        a_scale=a_scale, b_scale=b_scale, transpose_b=True,
    )
    assert auto_out.shape == (m, n)
    assert torch.equal(auto_out, explicit_out)
