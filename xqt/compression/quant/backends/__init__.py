"""External quantization backend adapters."""

from .onnx_qdq import (
    IterableCalibrationDataReader,
    ONNXQDQQuantizationResult,
    quantize_onnx_qdq_static,
)
from .torchao import TorchAOQuantizationResult, quantize_with_torchao

__all__ = [
    "IterableCalibrationDataReader",
    "ONNXQDQQuantizationResult",
    "TorchAOQuantizationResult",
    "quantize_onnx_qdq_static",
    "quantize_with_torchao",
]
