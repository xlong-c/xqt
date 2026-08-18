from __future__ import annotations

from pathlib import Path

import pytest
import torch

from xqt.core.errors import XQTBackendError
from xqt.gemm import (
    EpilogueSpec,
    GemmProblem,
    GemmSpec,
    QuantSpec,
    calibrate_fp8_scale,
    default_registry,
    reference_gemm,
)
from xqt.gemm.backends.sm89.fp8_sm89 import (
    _scale_scalar,
    fp8_sm89_executor,
    install_sm89_fp8_executors,
    sm89_fp8_artifact_available,
)
from xqt.gemm.common.fp8 import quantize_fp8


_ARTIFACT = Path.home() / ".cache/xqt/gemm/sm89/fp8_cutlass_sm89.so"


def _native_spec(format_name: str, *, m: int, n: int, k: int, output: str) -> GemmSpec:
    quant = QuantSpec(
        weight_dtype=format_name,
        activation_dtype=format_name,
        output_dtype=output,
        weight_granularity="per_tensor",
        activation_granularity="per_tensor",
        weight_scale_source="weight_offline",
        activation_scale_source="activation_static",
        storage_layout="xqt_fp8_rowmajor_v1",
        pack_version="xqt-fp8-v1",
    )
    return GemmSpec(
        problem=GemmProblem(m=m, n=n, k=k, sm=89, device="cuda:0"),
        quant=quant,
        epilogue=EpilogueSpec(output_dtype=output),
    )


def test_missing_fp8_artifact_does_not_promote_registry() -> None:
    registry = default_registry()
    assert install_sm89_fp8_executors(
        registry, artifact="/tmp/xqt-missing-fp8-sm89.so"
    ) is False
    assert registry.get("sm89_fp8_e4m3_cutlass").maturity == "metadata_only"
    assert registry.get("sm89_fp8_e5m2_cutlass").maturity == "metadata_only"
    assert sm89_fp8_artifact_available("/tmp/xqt-missing-fp8-sm89.so") is False


def test_tensorwise_scale_scalar_caches_and_invalidates_by_tensor_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scale = torch.tensor(1.5)
    original_isfinite = torch.isfinite
    calls = 0

    def tracked_isfinite(value: torch.Tensor) -> torch.Tensor:
        nonlocal calls
        calls += 1
        return original_isfinite(value)

    monkeypatch.setattr(torch, "isfinite", tracked_isfinite)

    assert _scale_scalar(scale, name="scale") == 1.5
    assert _scale_scalar(scale, name="scale") == 1.5
    assert calls == 1

    scale.fill_(2.0)
    assert _scale_scalar(scale, name="scale") == 2.0
    assert calls == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("format_name", ["fp8_e4m3", "fp8_e5m2"])
@pytest.mark.parametrize("output", ["fp16", "bf16"])
@pytest.mark.parametrize("shape", [(1, 7, 33), (32, 40, 64), (256, 129, 65)])
def test_sm89_fp8_cutlass_matches_reference(
    format_name: str, output: str, shape: tuple[int, int, int]
) -> None:
    if not _ARTIFACT.is_file():
        pytest.skip("SM89 FP8 CUTLASS artifact is not built")
    m, n, k = shape
    torch.manual_seed(20260730 + m + n + k)
    activation = torch.randn(m, k, device="cuda", dtype=torch.float16)
    weight = torch.randn(n, k, device="cuda", dtype=torch.float16)
    activation_scale = calibrate_fp8_scale(
        activation,
        format_name=format_name,
        granularity="per_tensor",
        role="activation",
    )
    weight_scale = calibrate_fp8_scale(
        weight,
        format_name=format_name,
        granularity="per_tensor",
        role="weight",
    )
    quantized_activation = quantize_fp8(
        activation,
        format_name=format_name,
        granularity="per_tensor",
        role="activation",
        source="activation_static",
        scale=activation_scale,
    )
    quantized_weight = quantize_fp8(
        weight,
        format_name=format_name,
        granularity="per_tensor",
        role="weight",
        source="weight_offline",
        scale=weight_scale,
    )
    spec = _native_spec(format_name, m=m, n=n, k=k, output=output)
    actual = fp8_sm89_executor(
        quantized_activation.storage,
        quantized_weight.storage,
        spec=spec,
        weight_scales=weight_scale,
        activation_scales=activation_scale,
        artifact=_ARTIFACT,
    )
    expected = reference_gemm(
        quantized_activation.storage,
        quantized_weight.storage,
        spec=spec,
        weight_scales=weight_scale,
        activation_scales=activation_scale,
    )
    torch.testing.assert_close(actual, expected, atol=0.125, rtol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_sm89_fp8_cutlass_bias_epilogue_matches_reference() -> None:
    if not _ARTIFACT.is_file():
        pytest.skip("SM89 FP8 CUTLASS artifact is not built")
    m, n, k = 32, 24, 65
    fmt = "fp8_e4m3"
    torch.manual_seed(20260731)
    activation = torch.randn(m, k, device="cuda", dtype=torch.float16)
    weight = torch.randn(n, k, device="cuda", dtype=torch.float16)
    activation_scale = calibrate_fp8_scale(
        activation, format_name=fmt, granularity="per_tensor", role="activation"
    )
    weight_scale = calibrate_fp8_scale(
        weight, format_name=fmt, granularity="per_tensor", role="weight"
    )
    quantized_activation = quantize_fp8(
        activation,
        format_name=fmt,
        granularity="per_tensor",
        role="activation",
        source="activation_static",
        scale=activation_scale,
    )
    quantized_weight = quantize_fp8(
        weight,
        format_name=fmt,
        granularity="per_tensor",
        role="weight",
        source="weight_offline",
        scale=weight_scale,
    )
    bias = torch.randn(n, device="cuda", dtype=torch.float16)
    spec = GemmSpec(
        problem=GemmProblem(m=m, n=n, k=k, sm=89, device="cuda:0"),
        quant=_native_spec(fmt, m=m, n=n, k=k, output="fp16").quant,
        epilogue=EpilogueSpec(output_dtype="fp16", has_bias=True),
    )
    actual = fp8_sm89_executor(
        quantized_activation.storage,
        quantized_weight.storage,
        spec=spec,
        weight_scales=weight_scale,
        activation_scales=activation_scale,
        bias=bias,
        artifact=_ARTIFACT,
    )
    expected = reference_gemm(
        quantized_activation.storage,
        quantized_weight.storage,
        spec=spec,
        weight_scales=weight_scale,
        activation_scales=activation_scale,
        bias=bias,
    )
    torch.testing.assert_close(actual, expected, atol=0.125, rtol=2e-2)


def test_sm89_fp8_rejects_dynamic_input_in_native_executor() -> None:
    spec = _native_spec("fp8_e4m3", m=1, n=8, k=32, output="fp16")
    spec = GemmSpec(
        problem=spec.problem,
        quant=QuantSpec(
            **{
                **spec.quant.to_dict(),
                "activation_scale_source": "activation_dynamic",
            }
        ),
        epilogue=spec.epilogue,
    )
    with pytest.raises(XQTBackendError, match="dynamic FP8 activation"):
        fp8_sm89_executor(
            torch.zeros(1, 32, dtype=torch.uint8),
            torch.zeros(8, 32, dtype=torch.uint8),
            spec=spec,
            weight_scales=torch.ones(1, 1),
            activation_scales=torch.ones(1, 1),
            artifact="/tmp/xqt-missing-fp8-sm89.so",
        )


def _blockwise_spec(
    format_name: str,
    *,
    m: int,
    n: int,
    k: int,
    block_k: int,
    output: str,
    has_bias: bool = False,
) -> GemmSpec:
    quant = QuantSpec(
        weight_dtype=format_name,
        activation_dtype=format_name,
        output_dtype=output,
        weight_granularity="blockwise",
        activation_granularity="blockwise",
        group_size=block_k,
        weight_scale_source="weight_offline",
        activation_scale_source="activation_static",
        storage_layout="xqt_fp8_rowmajor_v1",
        pack_version="xqt-fp8-v1",
    )
    return GemmSpec(
        problem=GemmProblem(m=m, n=n, k=k, sm=89, device="cuda:0"),
        quant=quant,
        epilogue=EpilogueSpec(output_dtype=output, has_bias=has_bias),
    )


def _blockwise_case(
    format_name: str,
    *,
    m: int,
    n: int,
    k: int,
    block_k: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    activation = torch.randn(m, k, device="cuda", dtype=torch.float16)
    weight = torch.randn(n, k, device="cuda", dtype=torch.float16)
    activation_scale = calibrate_fp8_scale(
        activation,
        format_name=format_name,
        granularity="blockwise",
        role="activation",
        block_k=block_k,
    )
    weight_scale = calibrate_fp8_scale(
        weight,
        format_name=format_name,
        granularity="blockwise",
        role="weight",
        block_k=block_k,
    )
    encoded_activation = quantize_fp8(
        activation,
        format_name=format_name,
        granularity="blockwise",
        role="activation",
        source="activation_static",
        scale=activation_scale,
        block_k=block_k,
    )
    encoded_weight = quantize_fp8(
        weight,
        format_name=format_name,
        granularity="blockwise",
        role="weight",
        source="weight_offline",
        scale=weight_scale,
        block_k=block_k,
    )
    return (
        encoded_activation.storage,
        encoded_weight.storage,
        encoded_weight.scale,
        encoded_activation.scale,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("format_name", ["fp8_e4m3", "fp8_e5m2"])
@pytest.mark.parametrize("output", ["fp16", "bf16"])
@pytest.mark.parametrize("block_k", [32, 64, 128])
@pytest.mark.parametrize("shape", [(32, 40, 64), (17, 13, 100)])
def test_sm89_fp8_blockwise_matches_reference(
    format_name: str, output: str, block_k: int, shape: tuple[int, int, int]
) -> None:
    if not _ARTIFACT.is_file():
        pytest.skip("SM89 FP8 CUTLASS artifact is not built")
    m, n, k = shape
    a_bytes, w_bytes, weight_scale, activation_scale = _blockwise_case(
        format_name, m=m, n=n, k=k, block_k=block_k, seed=20260803 + m + n + k + block_k
    )
    spec = _blockwise_spec(format_name, m=m, n=n, k=k, block_k=block_k, output=output)
    actual = fp8_sm89_executor(
        a_bytes,
        w_bytes,
        spec=spec,
        weight_scales=weight_scale,
        activation_scales=activation_scale,
        artifact=_ARTIFACT,
    )
    expected = reference_gemm(
        a_bytes,
        w_bytes,
        spec=spec,
        weight_scales=weight_scale,
        activation_scales=activation_scale,
    )
    assert tuple(actual.shape) == (m, n)
    torch.testing.assert_close(actual, expected, atol=0.125, rtol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_sm89_fp8_blockwise_bias_epilogue_matches_reference() -> None:
    if not _ARTIFACT.is_file():
        pytest.skip("SM89 FP8 CUTLASS artifact is not built")
    m, n, k, block_k = 32, 24, 96, 64
    a_bytes, w_bytes, weight_scale, activation_scale = _blockwise_case(
        "fp8_e4m3", m=m, n=n, k=k, block_k=block_k, seed=20260806
    )
    bias = torch.randn(n, device="cuda", dtype=torch.float16)
    spec = _blockwise_spec(
        "fp8_e4m3", m=m, n=n, k=k, block_k=block_k, output="fp16", has_bias=True
    )
    actual = fp8_sm89_executor(
        a_bytes,
        w_bytes,
        spec=spec,
        weight_scales=weight_scale,
        activation_scales=activation_scale,
        bias=bias,
        artifact=_ARTIFACT,
    )
    expected = reference_gemm(
        a_bytes,
        w_bytes,
        spec=spec,
        weight_scales=weight_scale,
        activation_scales=activation_scale,
        bias=bias,
    )
    torch.testing.assert_close(actual, expected, atol=0.125, rtol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("split_k", [2, 4])
@pytest.mark.parametrize("block_k", [32, 64, 128])
def test_sm89_fp8_blockwise_splitk_matches_reference(split_k: int, block_k: int) -> None:
    if not _ARTIFACT.is_file():
        pytest.skip("SM89 FP8 CUTLASS artifact is not built")
    m, n, k = 64, 128, 512
    a_bytes, w_bytes, weight_scale, activation_scale = _blockwise_case(
        "fp8_e4m3", m=m, n=n, k=k, block_k=block_k, seed=20260807 + split_k + block_k
    )
    spec = _blockwise_spec("fp8_e4m3", m=m, n=n, k=k, block_k=block_k, output="fp16")
    actual = fp8_sm89_executor(
        a_bytes,
        w_bytes,
        spec=spec,
        weight_scales=weight_scale,
        activation_scales=activation_scale,
        artifact=_ARTIFACT,
        split_k=split_k,
    )
    expected = reference_gemm(
        a_bytes,
        w_bytes,
        spec=spec,
        weight_scales=weight_scale,
        activation_scales=activation_scale,
    )
    torch.testing.assert_close(actual, expected, atol=0.125, rtol=2e-2)


def test_fp8_blockwise_split_k_partition_is_block_aligned() -> None:
    from xqt.gemm import fp8_blockwise_split_k_partition

    k_per_split, split_count = fp8_blockwise_split_k_partition(1024, 64, 4)
    assert k_per_split % 64 == 0
    assert split_count * k_per_split >= 1024
    assert (split_count - 1) * k_per_split < 1024
    # A trailing partial execution block belongs to the last split.
    k_per_split, split_count = fp8_blockwise_split_k_partition(96, 64, 2)
    assert k_per_split == 64
    assert split_count == 2
    with pytest.raises(ValueError, match="split_k >= 2"):
        fp8_blockwise_split_k_partition(1024, 64, 1)
    with pytest.raises(ValueError, match="% 32 == 0"):
        fp8_blockwise_split_k_partition(100, 64, 2)


def test_sm89_fp8_rejects_splitk_outside_blockwise() -> None:
    spec = _native_spec("fp8_e4m3", m=16, n=8, k=32, output="fp16")
    with pytest.raises(XQTBackendError, match="not tensorwise"):
        fp8_sm89_executor(
            torch.zeros(16, 32, dtype=torch.uint8),
            torch.zeros(8, 32, dtype=torch.uint8),
            spec=spec,
            weight_scales=torch.ones(1, 1),
            activation_scales=torch.ones(1, 1),
            artifact="/tmp/xqt-missing-fp8-sm89.so",
            split_k=2,
        )
    blockwise_spec = _blockwise_spec("fp8_e4m3", m=16, n=8, k=64, block_k=32, output="fp16")
    with pytest.raises(XQTBackendError, match="split_k must be"):
        fp8_sm89_executor(
            torch.zeros(16, 64, dtype=torch.uint8),
            torch.zeros(8, 64, dtype=torch.uint8),
            spec=blockwise_spec,
            weight_scales=torch.ones(8, 2),
            activation_scales=torch.ones(16, 2),
            artifact="/tmp/xqt-missing-fp8-sm89.so",
            split_k=0,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("block_k", [32, 64, 128])
def test_sm89_fp8_blockwise_resource_query(block_k: int) -> None:
    if not _ARTIFACT.is_file():
        pytest.skip("SM89 FP8 CUTLASS artifact is not built")
    from xqt.gemm import query_sm89_fp8_blockwise_resources

    report = query_sm89_fp8_blockwise_resources(
        _ARTIFACT, format_name="fp8_e4m3", output_dtype="fp16", block_k=block_k
    )
    assert report.registers_per_thread > 0
    assert report.max_active_blocks_per_sm >= 1
    assert report.occupancy is not None and 0.0 < report.occupancy <= 1.0
    assert report.block_k == block_k


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_sm89_fp8_install_promotes_blockwise_dispatch_when_manifest_ready() -> None:
    from xqt.gemm import artifact_ready_for_execution, dispatch_gemm

    if not _ARTIFACT.is_file() or not artifact_ready_for_execution(
        _ARTIFACT, kernel_name="sm89_fp8_cutlass", target_arch="sm_89"
    ):
        pytest.skip("SM89 FP8 artifact is not promoted to executable")
    registry = default_registry()
    assert install_sm89_fp8_executors(registry, artifact=_ARTIFACT) is True
    entry = registry.get("sm89_fp8_e4m3_cutlass")
    assert entry.maturity == "executable"
    assert "w:blockwise/a:blockwise" in entry.capability.scale_modes
    m, n, k, block_k = 32, 40, 96, 64
    a_bytes, w_bytes, weight_scale, activation_scale = _blockwise_case(
        "fp8_e4m3", m=m, n=n, k=k, block_k=block_k, seed=20260808
    )
    spec = _blockwise_spec("fp8_e4m3", m=m, n=n, k=k, block_k=block_k, output="fp16")
    result = dispatch_gemm(
        a_bytes,
        w_bytes,
        spec=spec,
        weight_scales=weight_scale,
        activation_scales=activation_scale,
        registry=registry,
    )
    assert result.report.selected_kernel == "sm89_fp8_e4m3_cutlass"
    assert result.report.native is True
    assert result.report.scale_mode == "w:blockwise/a:blockwise"
    expected = reference_gemm(
        a_bytes,
        w_bytes,
        spec=spec,
        weight_scales=weight_scale,
        activation_scales=activation_scale,
    )
    torch.testing.assert_close(result.output, expected, atol=0.125, rtol=2e-2)
