from __future__ import annotations

import torch.nn as nn

from xqt.contracts import (
    ExportBundlePayload,
    PrunedModelPayload,
    QuantizedModelPayload,
    RuntimePlanPayload,
)


def test_quantized_model_payload_accepts_module_contract() -> None:
    contract = {"operator_kind": "linear", "policy": {"activation": "fp16"}}
    payload = QuantizedModelPayload.from_stage_metrics(
        stage_name="quant",
        source_model_stage="baseline",
        model=nn.Linear(4, 2),
        metrics={"backend": "torchao", "method": "fp4", "strategy": "fp4_weight_only"},
        params={},
        artifacts={},
        module_contract=contract,
    )
    assert payload.to_dict().get("module_contract") == contract


def test_quantized_model_payload_auto_extracts_module_contract_from_model() -> None:
    model = nn.Linear(4, 2)
    contract = {"operator_kind": "linear"}
    setattr(model, "_xqt_module_contract", contract)
    payload = QuantizedModelPayload.from_stage_metrics(
        stage_name="quant",
        source_model_stage="baseline",
        model=model,
        metrics={"backend": "torchao"},
        params={},
        artifacts={},
    )
    assert payload.to_dict().get("module_contract") == contract


def test_pruned_model_payload_accepts_module_contract() -> None:
    contract = {"operator_kind": "conv2d"}
    payload = PrunedModelPayload.from_stage_metrics(
        stage_name="prune",
        source_model_stage="baseline",
        model=nn.Linear(4, 2),
        metrics={"method": "l1", "sparsity": 0.5, "execution_state": "applied"},
        params={},
        artifacts={},
        module_contract=contract,
    )
    assert payload.to_dict().get("module_contract") == contract


def test_runtime_plan_payload_accepts_module_contract() -> None:
    contract = {"operator_kind": "attention"}
    payload = RuntimePlanPayload.from_stage_metrics(
        stage_name="operator",
        source_model_stage="baseline",
        metrics={"targets": [{"engine": "tilelang"}]},
        artifacts={},
        module_contract=contract,
    )
    assert payload.to_dict().get("module_contract") == contract


def test_export_bundle_payload_accepts_module_contract() -> None:
    contract = {"operator_kind": "linear"}
    payload = ExportBundlePayload.from_stage_metrics(
        stage_name="export",
        source_model_stage="baseline",
        metrics={"targets": [{"format": "onnx"}]},
        artifacts={},
        module_contract=contract,
    )
    assert payload.to_dict().get("module_contract") == contract
