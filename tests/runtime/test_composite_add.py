from __future__ import annotations

import torch
import torch.nn as nn

from xqt.compression.quant.quantizers.svd import quantize_with_svd
from xqt.runtime.composite_materialize import materialize_composite_compute
from xqt.contracts import CompositeAddLinear, CompositeAddModule, ComputeConfig
from xqt.runtime.modules import (
    CompositeAddFp8Linear,
    CompositeAddW4A4Linear,
    SVDQuantInt8MmaLinear,
    materialize_composite_w4a4,
)


def test_runtime_executors_are_not_storage_artifact_subclasses() -> None:
    assert not issubclass(CompositeAddW4A4Linear, CompositeAddLinear)
    assert not issubclass(SVDQuantInt8MmaLinear, CompositeAddLinear)
    assert issubclass(CompositeAddW4A4Linear, CompositeAddModule)
    assert issubclass(SVDQuantInt8MmaLinear, CompositeAddModule)


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


def test_w4a4_executor_falls_back_to_generic_reference_on_cpu() -> None:
    artifact = _build_reference_artifact()
    executor = materialize_composite_w4a4(artifact)
    assert isinstance(executor, CompositeAddW4A4Linear)

    inputs = torch.randn(3, 32)
    torch.testing.assert_close(executor(inputs), artifact(inputs))
    metadata = executor.execution_metadata()
    assert metadata["implementation"] == "composite_add_reference"
    assert metadata["native_w4a4_used"] is False
    assert "CUDA" in str(metadata["fallback_reason"])


def test_w4a4_executor_can_be_disabled_without_changing_reference_values() -> None:
    artifact = _build_reference_artifact()
    executor = materialize_composite_w4a4(artifact)
    executor.disable_fusion()

    inputs = torch.randn(2, 32)
    torch.testing.assert_close(executor(inputs), artifact(inputs))
    metadata = executor.execution_metadata()
    assert metadata["native_w4a4_used"] is False
    assert metadata["fallback_reason"] == "native W4A4 fusion is disabled"


def test_w4a4_smalln_executor_is_an_explicit_layout() -> None:
    artifact = _build_reference_artifact()
    executor = materialize_composite_w4a4(artifact, layout="smalln")
    assert isinstance(executor, CompositeAddW4A4Linear)

    inputs = torch.randn(2, 32)
    torch.testing.assert_close(executor(inputs), artifact(inputs))
    metadata = executor.execution_metadata()
    assert metadata["w4a4_layout"] == "smalln"
    assert metadata["native_w4a4_used"] is False
    assert "CUDA" in str(metadata["fallback_reason"])


def test_compute_config_can_materialize_generic_w4a4_executor() -> None:
    artifact = _build_reference_artifact()
    config = ComputeConfig.from_modules(
        module_names=["0"],
        compute_contract="composite_add",
        precision="w4a4",
        required_capabilities=["composite_add"],
        preferred_mode="fused",
        combine="add",
    )

    materialized = materialize_composite_compute(
        nn.Sequential(artifact),
        config,
        inplace=False,
    )
    assert isinstance(materialized[0], CompositeAddW4A4Linear)
    inputs = torch.randn(2, 32)
    torch.testing.assert_close(materialized(inputs), artifact(inputs))


def test_compute_config_selects_explicit_w4a4_layout() -> None:
    artifact = _build_reference_artifact()
    config = ComputeConfig.from_mapping(
        {
            "modules": [
                {
                    "name": "0",
                    "compute_contract": "composite_add",
                    "precision": "w4a4",
                    "required_capabilities": ["composite_add"],
                    "preferred_mode": "fused",
                    "combine": "add",
                    "metadata": {"w4a4_layout": "smalln"},
                }
            ]
        }
    )

    materialized = materialize_composite_compute(
        nn.Sequential(artifact),
        config,
        inplace=False,
    )
    assert isinstance(materialized[0], CompositeAddW4A4Linear)
    assert materialized[0].layout == "smalln"


def test_compute_config_materializes_generic_w8a8_split_executor() -> None:
    artifact = _build_reference_artifact()
    config = ComputeConfig.from_mapping(
        {
            "modules": [
                {
                    "name": "0",
                    "compute_contract": "composite_add",
                    "precision": "w8a8",
                    "preferred_mode": "split",
                    "preferred_engines": ["torch_int_mm"],
                    "metadata": {"fallback_engine": "torch_int_mm"},
                }
            ]
        }
    )

    materialized = materialize_composite_compute(
        nn.Sequential(artifact),
        config,
        inplace=False,
    )
    assert isinstance(materialized[0], SVDQuantInt8MmaLinear)
    inputs = torch.randn(2, 32)
    output = materialized(inputs)
    reference = artifact(inputs)
    assert torch.isfinite(output).all()
    assert float((output - reference).abs().max().detach()) < 3e-2


def test_compute_config_materializes_generic_fp8_split_executor() -> None:
    artifact = _build_reference_artifact()
    config = ComputeConfig.from_mapping(
        {
            "modules": [
                {
                    "name": "0",
                    "compute_contract": "composite_add",
                    "precision": "fp8",
                    "preferred_mode": "split",
                    "metadata": {
                        "activation_scale_mode": "dynamic",
                        "min_fp8_rows": 0,
                    },
                }
            ]
        }
    )

    materialized = materialize_composite_compute(
        nn.Sequential(artifact),
        config,
        inplace=False,
    )
    assert isinstance(materialized[0], CompositeAddFp8Linear)
    output = materialized(torch.randn(2, 32))
    assert output.dtype == torch.float16
    assert torch.isfinite(output).all()
