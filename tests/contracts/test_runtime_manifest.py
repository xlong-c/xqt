"""U2: RuntimeManifest aggregates contract + layout + kernels."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from xqt.contracts import (
    LayoutKernelReport,
    QuantizedModel,
    RuntimeManifest,
    RuntimeQuantContract,
    build_runtime_manifest,
)
from xqt.quant.types import QuantScheme
from xqt.contracts.quant_pair import write_quant_pair


def _scheme() -> QuantScheme:
    return QuantScheme(
        weight_dtype="int4",
        weight_granularity="groupwise",
        group_size=32,
        activation_dtype=None,
        activation_mode="none",
        sym=True,
    )


def test_runtime_manifest_round_trip_dict() -> None:
    contract = RuntimeQuantContract(
        quant_spec=_scheme(),
        storage_layout="xqt_awq_gptq_int4_v1",
        required_kernels=("dequant_fp16",),
        global_shape=(16, 64),
        local_shape=(16, 64),
        prefill_supported=True,
        decode_supported=True,
    )
    layout = LayoutKernelReport(
        bits=4,
        group_size=32,
        storage_layout="xqt_awq_gptq_int4_v1",
        selected_kernel="dequant_fp16_reference",
        g_idx_applied=True,
        desc_act=False,
    )
    manifest = RuntimeManifest(
        contract=contract,
        layout_reports=(layout,),
        selected_kernels=("dequant_fp16_reference",),
        fallback_kernels=(),
        prefill_supported=True,
        decode_supported=True,
    )
    restored = RuntimeManifest.from_dict(manifest.to_dict())
    assert restored.contract == contract
    assert restored.layout_reports[0].selected_kernel == "dequant_fp16_reference"
    assert restored.selected_kernels == ("dequant_fp16_reference",)


def test_build_runtime_manifest_from_quantized_model() -> None:
    contract = RuntimeQuantContract(
        quant_spec=_scheme(),
        storage_layout="xqt_awq_gptq_int4_v1",
        required_kernels=("dequant_fp16",),
        global_shape=(4, 8),
        local_shape=(4, 8),
        prefill_supported=True,
        decode_supported=True,
    )
    layout = LayoutKernelReport(
        bits=4,
        group_size=32,
        storage_layout="xqt_awq_gptq_int4_v1",
        selected_kernel="dequant_fp16_reference",
    )
    quantized = QuantizedModel(
        model=nn.Linear(8, 4),
        backend="pytorch",
        method="gptq",
        metadata={"layout_kernel": layout.to_dict()},
    ).with_runtime_quant_contract(contract)
    manifest = build_runtime_manifest(quantized)
    assert manifest.contract == contract
    assert "dequant_fp16_reference" in manifest.selected_kernels
    assert manifest.prefill_supported is True
    assert "runtime_quant_contract_absent" not in manifest.notes


def test_build_runtime_manifest_from_quant_pair(tmp_path: Path) -> None:
    contract = RuntimeQuantContract(
        quant_spec=_scheme(),
        storage_layout="xqt_awq_gptq_int4_v1",
        required_kernels=("dequant_fp16", "tilelang"),
        global_shape=(4, 8),
        local_shape=(4, 8),
        prefill_supported=True,
        decode_supported=False,
    )
    model = nn.Linear(8, 4)
    pair_dir = write_quant_pair(
        model,
        tmp_path / "manifest_pair",
        runtime_quant_contract=contract,
        metadata={
            "layout_kernel": LayoutKernelReport(
                bits=4,
                storage_layout="xqt_awq_gptq_int4_v1",
                selected_kernel="tilelang",
            ).to_dict()
        },
    )
    manifest = build_runtime_manifest(pair_dir)
    assert manifest.contract is not None
    assert manifest.contract.decode_supported is False
    assert "tilelang" in manifest.selected_kernels
    payload = manifest.to_dict()
    assert payload["contract"]["storage_layout"] == "xqt_awq_gptq_int4_v1"


def test_build_runtime_manifest_honest_absent_contract() -> None:
    quantized = QuantizedModel(model=torch.nn.Linear(2, 2), backend="pytorch")
    manifest = build_runtime_manifest(quantized)
    assert manifest.contract is None
    assert "runtime_quant_contract_absent" in manifest.notes
