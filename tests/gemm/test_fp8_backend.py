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
from xqt.gemm.backends.fp8_sm89 import (
    fp8_sm89_executor,
    install_sm89_fp8_executors,
    sm89_fp8_artifact_available,
)
from xqt.gemm.fp8 import quantize_fp8


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
