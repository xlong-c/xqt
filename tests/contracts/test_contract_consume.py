"""W5: RuntimeQuantContract consumer smoke."""

from __future__ import annotations

from torch import nn

from xqt.contracts import consume_runtime_quant_contract
from xqt.quant.quantizers.int8_mma import quantize_with_int8_mma


def test_consume_contract_from_int8_result() -> None:
    model = nn.Linear(16, 8).eval()
    result = quantize_with_int8_mma(
        model,
        policy={"include_module_types": ["Linear"]},
        engine="torch_int_mm",
        inplace=False,
    )
    report = consume_runtime_quant_contract(result)
    assert report.ok is True
    assert report.contract is not None
    assert report.contract.storage_layout
    assert report.contract.required_kernels


def test_consume_missing_contract() -> None:
    report = consume_runtime_quant_contract({})
    assert report.ok is False
    assert "runtime_quant_contract_missing" in report.errors
