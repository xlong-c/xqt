from __future__ import annotations

import torch
import torch.nn as nn

from tests.xqt.svd_test_helpers import (
    make_legacy_svd_int8,
    make_legacy_svd_linear,
)
from xqt.compression.quant.quantizers.convrot_int8 import ConvRotInt8Linear
from xqt.compression.quant.quantizers.convrot_4bit import ConvRotMixedPrecisionLinear
from xqt.runtime.modules.svd_legacy import SVDQuantInt8MmaLinear, SVDQuantLinear
from xqt.runtime.modules.svd_composite import (
    SVDQuantInt8MmaLinear as CompatSVDQuantInt8MmaLinear,
    SVDQuantLinear as CompatSVDQuantLinear,
)


def test_svd_runtime_compat_facade_points_to_legacy_implementation() -> None:
    assert CompatSVDQuantLinear is SVDQuantLinear
    assert CompatSVDQuantInt8MmaLinear is SVDQuantInt8MmaLinear


def test_convrot_dequantized_weight_cache_reuses_and_invalidates() -> None:
    module = ConvRotMixedPrecisionLinear.from_linear(
        nn.Linear(16, 12, bias=False).eval(),
        rot_size=4,
        group_size=8,
        compute_precision="w4a16",
    ).eval()

    first = module.dequantized_weight(
        dtype=torch.float32,
        device=torch.device("cpu"),
        include_padding=True,
    )
    assert module.dequantized_weight(
        dtype=torch.float32,
        device=torch.device("cpu"),
        include_padding=True,
    ) is first

    with torch.no_grad():
        module.weight_scale[0, 0].mul_(2.0)
    updated = module.dequantized_weight(
        dtype=torch.float32,
        device=torch.device("cpu"),
        include_padding=True,
    )

    assert updated is not first
    assert not torch.equal(updated, first)


def test_convrot_int8_fallback_weight_cache_rebuilds_after_scale_mutation() -> None:
    module = ConvRotInt8Linear.from_linear(
        nn.Linear(16, 16, bias=False).eval(),
        rot_size=4,
        engine="torch_int_mm",
    ).eval()

    first = module._dense_unrotated_weight(torch.float32, torch.device("cpu"))
    assert module._dense_unrotated_weight(torch.float32, torch.device("cpu")) is first

    with torch.no_grad():
        module.int8_compute.weight_scale[0].mul_(2.0)
    updated = module._dense_unrotated_weight(torch.float32, torch.device("cpu"))

    assert updated is not first
    assert not torch.equal(updated, first)


def test_convrot_int8_dtype_conversion_updates_output_contract() -> None:
    module = ConvRotInt8Linear.from_linear(
        nn.Linear(256, 256, bias=False).eval(),
        rot_size=256,
        engine="auto",
    ).eval()

    module.half()
    assert module.int8_compute.output_dtype is torch.float16

    module.bfloat16()
    assert module.int8_compute.output_dtype is torch.bfloat16


def test_svd_reference_residual_cache_reuses_and_invalidates() -> None:
    module = make_legacy_svd_linear(
        nn.Linear(16, 12, bias=False).eval(),
        rank=4,
        group_size=8,
        quant_dtype="int4",
    ).eval()

    first = module.dequantize_residual()
    assert module.dequantize_residual() is first

    with torch.no_grad():
        module.residual_scale[0, 0].mul_(2.0)
    updated = module.dequantize_residual()

    assert updated is not first
    assert not torch.equal(updated, first)


def test_svd_group_size_is_normalized_for_narrow_linear() -> None:
    module = make_legacy_svd_linear(
        nn.Linear(16, 12, bias=False).eval(),
        rank=4,
        group_size=128,
        quant_dtype="int4",
    ).eval()

    assert module.group_size == 16
    assert module.dequantize_residual().shape == (12, 16)
    assert module(torch.randn(2, 16)).shape == (2, 12)


def test_svd_int8_compute_view_rebuilds_after_storage_mutation() -> None:
    module = make_legacy_svd_int8(
        nn.Linear(16, 12, bias=False).eval(),
        rank=4,
        group_size=8,
        quant_dtype="int4",
        engine="torch_int_mm",
        cache_int8_compute_view=True,
    ).eval()
    inputs = torch.randn(3, 16)

    module(inputs)
    first_compute = module.residual_int8._compute
    assert first_compute is not None

    with torch.no_grad():
        module.residual_int8.group_scale[0, 0].mul_(2.0)
    module(inputs)
    assert module.residual_int8._compute is not first_compute


def test_svd_int8_half_updates_cuda_output_dtype_contract() -> None:
    module = make_legacy_svd_int8(
        nn.Linear(16, 12, bias=False).eval(),
        rank=4,
        group_size=8,
        quant_dtype="int4",
        engine="torch_int_mm",
    ).eval()

    assert module.output_dtype is torch.float32
    module.half()

    assert module.output_dtype is torch.float16
    assert module.residual_int8.output_dtype is torch.float16


def test_svd_reference_fusion_metadata_is_explicit_on_cpu() -> None:
    module = make_legacy_svd_linear(
        nn.Linear(16, 12, bias=False).eval(),
        rank=4,
        group_size=8,
        quant_dtype="int4",
    ).eval()

    output = module(torch.randn(2, 16))
    metadata = module.execution_metadata()

    assert output.shape == (2, 12)
    assert metadata["cuda_fusion_enabled"] is False
    assert metadata["cuda_fused_used"] is False
