from __future__ import annotations

import torch
from torch import nn

from xqt.contracts import ComputeConfig
from xqt.compression.quant.quantizers.svd import quantize_with_svd
from xqt.runtime.composite_materialize import materialize_composite_compute
from xqt.runtime.modules import (
    CompositeAddLinear,
    CompositeAddW4A4Linear,
    RMSNormCompositeLinear,
)


def _build_reference_artifact() -> CompositeAddLinear:
    model = nn.Sequential(nn.Linear(32, 32)).eval()
    result = quantize_with_svd(
        model,
        strategy="w4a16_int4",
        compute="dequant_fp16",
        rank=4,
        group_size=16,
        quant_dtype="int4",
        inplace=False,
    )
    artifact = result.model[0]
    assert isinstance(artifact, CompositeAddLinear)
    return artifact


def _rms_norm(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    row_scale = torch.rsqrt(
        inputs.float().pow(2).mean(dim=-1, keepdim=True) + eps
    )
    return (inputs.float() * row_scale * weight.float()).to(inputs.dtype)


def test_rmsnorm_is_a_separate_reference_wrapper() -> None:
    artifact = _build_reference_artifact()
    norm_weight = torch.randn(32) * 0.2 + 1.0
    wrapper = RMSNormCompositeLinear(artifact, norm_weight)
    inputs = torch.randn(3, 32)

    expected = artifact(_rms_norm(inputs, norm_weight, 1e-6))
    torch.testing.assert_close(wrapper(inputs), expected)
    metadata = wrapper.execution_metadata()
    assert metadata["input_transform"] == "rmsnorm"
    assert metadata["composite"]["compute_contract"] == "composite_add"


def test_rmsnorm_wrapper_survives_generic_w4a4_materialization() -> None:
    artifact = _build_reference_artifact()
    wrapper = RMSNormCompositeLinear(artifact, torch.ones(32))
    materialized = wrapper.materialize_w4a4()

    assert isinstance(materialized, RMSNormCompositeLinear)
    assert isinstance(materialized.composite, CompositeAddW4A4Linear)
    inputs = torch.randn(2, 32)
    torch.testing.assert_close(materialized(inputs), wrapper(inputs))


def test_compute_config_materializes_inner_executor_only() -> None:
    artifact = _build_reference_artifact()
    wrapper = RMSNormCompositeLinear(artifact, torch.ones(32))
    config = ComputeConfig.from_mapping(
        {
            "modules": [
                {
                    "name": "0",
                    "compute_contract": "composite_add",
                    "precision": "w4a4",
                    "preferred_mode": "fused",
                }
            ]
        }
    )
    model = nn.Sequential(wrapper)

    materialized = materialize_composite_compute(model, config, inplace=False)

    assert isinstance(materialized[0], RMSNormCompositeLinear)
    assert isinstance(materialized[0].composite, CompositeAddW4A4Linear)
