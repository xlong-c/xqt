from __future__ import annotations

import json
from pathlib import Path

import torch

from xqt.contracts import QuantizedModel, QuantizedModelPayload
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
