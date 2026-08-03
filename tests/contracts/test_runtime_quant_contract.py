"""T1: RuntimeQuantContract round-trip and required-field validation."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from xqt.contracts import QuantizedModel, RuntimeQuantContract
from xqt.core.errors import XQTConfigError
from xqt.quant.types import QuantScheme


def _w4a16_scheme() -> QuantScheme:
    return QuantScheme(
        weight_dtype="int4",
        weight_granularity="groupwise",
        group_size=128,
        activation_dtype=None,
        activation_mode="none",
        sym=True,
    )


def test_runtime_quant_contract_round_trip_dict() -> None:
    """Given a full contract, to_dict/from_dict preserves fields."""

    contract = RuntimeQuantContract(
        quant_spec=_w4a16_scheme(),
        storage_layout="xqt_awq_gptq_int4_v1",
        repack_version=None,
        required_kernels=("dequant_gemm_int4",),
        global_shape=(4096, 4096),
        local_shape=(4096, 1024),
        shard_axis=1,
        prefill_supported=True,
        decode_supported=True,
        kv_cache_dtype="fp8_e4m3",
    )
    payload = contract.to_dict()
    restored = RuntimeQuantContract.from_dict(payload)
    assert restored == contract
    assert payload["quant_spec"]["weight_dtype"] == "int4"
    assert payload["required_kernels"] == ["dequant_gemm_int4"]
    assert "hf_quantization_config" not in payload


def test_runtime_quant_contract_rejects_missing_storage_layout() -> None:
    """Given incomplete mapping, from_dict raises typed config error."""

    with pytest.raises(XQTConfigError, match="storage_layout"):
        RuntimeQuantContract.from_dict(
            {
                "quant_spec": _w4a16_scheme().to_dict(),
                "required_kernels": [],
                "global_shape": [64, 32],
                "local_shape": [64, 32],
                "prefill_supported": True,
                "decode_supported": True,
            }
        )


def test_runtime_quant_contract_rejects_invalid_quant_spec() -> None:
    with pytest.raises((XQTConfigError, ValueError)):
        RuntimeQuantContract.from_dict(
            {
                "quant_spec": {"weight_dtype": "not_a_dtype"},
                "storage_layout": "x",
                "required_kernels": [],
                "global_shape": [],
                "local_shape": [],
                "prefill_supported": True,
                "decode_supported": True,
            }
        )


def test_quantized_model_attaches_and_resolves_runtime_quant_contract() -> None:
    """Given QuantizedModel, contract serializes into metadata and resolves back."""

    model = nn.Linear(8, 4)
    contract = RuntimeQuantContract(
        quant_spec=_w4a16_scheme(),
        storage_layout="xqt_awq_gptq_int4_v1",
        repack_version="1",
        required_kernels=(),
        global_shape=(4, 8),
        local_shape=(4, 8),
        shard_axis=None,
        prefill_supported=True,
        decode_supported=True,
        kv_cache_dtype=None,
    )
    quantized = QuantizedModel(
        model=model,
        backend="pytorch",
        method="gptq",
        strategy="w4a16_int4",
    ).with_runtime_quant_contract(contract)

    resolved = quantized.resolve_runtime_quant_contract()
    assert resolved is not None
    assert resolved == contract
    summary = quantized.to_dict()
    assert "runtime_quant_contract" in summary["metadata"]
    assert summary["metadata"]["runtime_quant_contract"]["storage_layout"] == (
        "xqt_awq_gptq_int4_v1"
    )


def test_quantized_model_resolve_contract_none_when_absent() -> None:
    quantized = QuantizedModel(model=torch.nn.Linear(2, 2), backend="pytorch")
    assert quantized.resolve_runtime_quant_contract() is None


def test_quantize_awq_and_int8_mma_attach_runtime_contract() -> None:
    """U8: quant main paths attach RuntimeQuantContract by default."""

    from xqt.quant.quantizers.awq_gptq_weight_only import quantize_with_awq_weight_only
    from xqt.quant.quantizers.int8_mma import quantize_with_int8_mma

    model = nn.Sequential(nn.Linear(16, 8), nn.Linear(8, 4)).eval()
    awq = quantize_with_awq_weight_only(
        model,
        policy={"include_module_types": ["Linear"], "group_size": 8, "bits": 4},
        strategy="w4a16_int4",
        inplace=False,
    )
    awq_contract = awq.resolve_runtime_quant_contract()
    assert awq_contract is not None
    assert awq_contract.storage_layout == "xqt_awq_gptq_int4_v1"
    assert awq_contract.quant_spec.weight_dtype == "int4"

    int8 = quantize_with_int8_mma(
        nn.Linear(16, 8).eval(),
        policy={"include_module_types": ["Linear"]},
        engine="torch_int_mm",
        inplace=False,
    )
    int8_contract = int8.resolve_runtime_quant_contract()
    assert int8_contract is not None
    assert int8_contract.storage_layout == "xqt_int8_mma_v1"
    assert "w8a8_int8_mma" in int8_contract.required_kernels
