from __future__ import annotations

import json
from pathlib import Path

import torch

from xqt.contracts import ExecutionPolicyPayload, QuantizedModel, QuantizedModelPayload
from xqt.quant.backends.torchao import TorchAOQuantizationResult
from xqt.quant.quantizers.awq_gptq_weight_only import (
    AWQGPTQWeightOnlyQuantizationResult,
)
from xqt.quant.quantizers.base import QuantizerResult
from xqt.quant.quantizers.fp4_weight_only import FP4QuantizationResult
from xqt.quant.quantizers.int8_mma import Int8MmaQuantizationResult
from xqt.quant.quantizers.mxfp_weight_only import MXFPQuantizationResult
from xqt.quant.quantizers.svd import SVDQuantResult
from xqt.quant.quantizers.w4_storage_int8_mma import (
    W4StorageInt8MmaQuantizationResult,
)


def test_model_side_quantization_results_share_quantized_model_contract() -> None:
    result_types = (
        FP4QuantizationResult,
        MXFPQuantizationResult,
        AWQGPTQWeightOnlyQuantizationResult,
        Int8MmaQuantizationResult,
        W4StorageInt8MmaQuantizationResult,
        SVDQuantResult,
        TorchAOQuantizationResult,
    )

    assert QuantizerResult is QuantizedModel
    assert all(issubclass(result_type, QuantizedModel) for result_type in result_types)


def test_quantized_model_contract_serializes_algorithm_metadata_and_stage_provenance() -> None:
    model = torch.nn.Linear(2, 2)
    result = QuantizedModel(
        model=model,
        backend="pytorch",
        method="awq",
        strategy="weight_only_int4",
        quantized_modules=["proj"],
        metadata={"artifact_path": Path("artifacts/quantized.pt")},
    )
    payload = QuantizedModelPayload(
        stage_name="quant",
        source_model_stage="baseline",
        model=model,
        backend=result.backend,
        method=result.method,
        strategy=result.strategy,
        quantized_modules=result.quantized_modules,
        metadata=result.metadata,
        artifacts={"checkpoint": "artifacts/quantized.pt"},
    )

    assert isinstance(payload, QuantizedModel)
    assert json.loads(json.dumps(result.to_dict())) == {
        "model_type": "torch.nn.modules.linear.Linear",
        "backend": "pytorch",
        "method": "awq",
        "strategy": "weight_only_int4",
        "quantized_module_count": 1,
        "quantized_modules": ["proj"],
        "metadata": {"artifact_path": "artifacts/quantized.pt"},
    }
    assert payload.to_dict()["artifact_kind"] == "quantized_model"
    assert payload.to_dict()["stage_name"] == "quant"
    assert payload.to_dict()["metadata"] == result.to_dict()["metadata"]


def test_quantized_model_payload_serializes_algorithm_metadata_and_execution_policies() -> None:
    payload = QuantizedModelPayload.from_stage_metrics(
        stage_name="quant",
        source_model_stage="baseline",
        model=torch.nn.Linear(2, 2),
        metrics={
            "backend": "pytorch",
            "method": "convrot",
            "strategy": "convrot_w4a4",
            "quantized_modules": ["proj"],
            "algorithm_metadata": {
                "rotation_kind": "regular_hadamard",
                "weight_bits": 4,
                "activation_bits": 4,
            },
            "execution_policies": [
                {
                    "policy_kind": "mixed_precision",
                    "runtime": "pytorch",
                    "precision_overrides": [{"module": "proj", "precision": "w8a8"}],
                }
            ],
        },
    )

    serialized = payload.to_dict()
    assert serialized["algorithm_metadata"]["rotation_kind"] == "regular_hadamard"
    assert serialized["execution_policies"][0]["policy_kind"] == "mixed_precision"
    assert serialized["execution_policies"][0]["precision_overrides"][0]["precision"] == "w8a8"


def test_execution_policy_payload_serializes_mixed_precision_policy() -> None:
    payload = ExecutionPolicyPayload(
        stage_name="quant",
        source_model_stage="baseline",
        policy_kind="mixed_precision",
        runtime="pytorch",
        module_count=2,
        precision_overrides=[
            {"module": "proj", "precision": "w8a8"},
            {"module": "fc2", "precision": "bf16"},
        ],
        metadata={"runtime_strategy": "mixed_precision_linear"},
    )

    assert payload.to_dict() == {
        "artifact_kind": "execution_policy",
        "stage_name": "quant",
        "source_model_stage": "baseline",
        "artifacts": {},
        "policy_kind": "mixed_precision",
        "runtime": "pytorch",
        "module_count": 2,
        "precision_overrides": [
            {"module": "proj", "precision": "w8a8"},
            {"module": "fc2", "precision": "bf16"},
        ],
        "required_capabilities": [],
        "preferred_engines": [],
        "metadata": {"runtime_strategy": "mixed_precision_linear"},
    }


def test_quantized_model_infer_handoff_excludes_method_identity() -> None:
    model = torch.nn.Linear(2, 2)
    from xqt.contracts import ComputeConfig

    config = ComputeConfig.from_modules(
        module_names=["proj"],
        compute_contract="int8_mma",
        precision="w8a8",
        required_capabilities=["int8_mma"],
        preferred_engines=["tilelang"],
    )
    result = QuantizedModel(
        model=model,
        backend="pytorch",
        method="awq",
        strategy="weight_only_int4",
        quantized_modules=["proj"],
        compute_config=config,
    )
    handoff = result.infer_handoff()
    assert handoff["model"] is model
    assert handoff["compute_config"]["modules"][0]["compute_contract"] == "int8_mma"
    assert "required_engine" not in handoff["compute_config"]
    assert "method" not in handoff
    assert "backend" not in handoff
    serialized = result.to_dict()
    assert serialized["compute_config"]["modules"][0]["required_capabilities"] == [
        "int8_mma"
    ]


def test_compute_config_ignores_required_engine_primary_key() -> None:
    from xqt.contracts import ComputeConfig

    config = ComputeConfig.from_mapping(
        {
            "schema_version": "1.0",
            "required_engine": "tilelang",
            "modules": [
                {
                    "name": "fc",
                    "compute_contract": "int8_mma",
                    "required_capabilities": ["int8_mma"],
                    "required_engine": "ptx_sm89",
                }
            ],
        }
    )
    assert config is not None
    assert "required_engine" not in config.to_dict()
    assert "required_engine" not in config.modules[0].to_dict()
    assert "ignored_forbidden_engine_keys" in config.metadata
