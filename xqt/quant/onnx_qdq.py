"""ONNX Runtime static QDQ quantization adapter."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np
import torch

from xqt.core.artifact import file_sha256
from xqt.core.errors import XQTBackendError
from xqt.data.input_utils import extract_model_inputs
from xqt.export.input_utils import build_onnx_feed


@dataclass
class ONNXQDQQuantizationResult:
    """ONNX QDQ quantization metadata."""

    path: Path
    source_path: Path
    checksum: str
    calibration_samples: int
    metadata: dict[str, Any] = field(default_factory=dict)


def _as_numpy_inputs(
    batch: Any,
    input_names: Sequence[str],
) -> dict[str, np.ndarray]:
    return build_onnx_feed(
        extract_model_inputs(batch, expected_input_count=len(input_names)),
        input_names=input_names,
    )


class IterableCalibrationDataReader:
    """Small ONNX Runtime CalibrationDataReader for PyTorch iterables."""

    def __init__(
        self,
        dataloader: Iterable[Any],
        *,
        input_names: Sequence[str] = ("input",),
        sample_limit: Optional[int] = None,
    ) -> None:
        self.input_names = tuple(input_names)
        self.sample_limit = sample_limit
        self._records = [
            _as_numpy_inputs(batch, self.input_names)
            for index, batch in enumerate(dataloader)
            if sample_limit is None or index < sample_limit
        ]
        self._index = 0

    @property
    def samples(self) -> int:
        """Return the number of calibration batches captured."""

        return len(self._records)

    @property
    def summary(self) -> dict[str, Any]:
        """Return a lightweight summary of captured calibration inputs."""

        input_names: list[str] = []
        shapes: dict[str, list[list[int]]] = {}
        dtypes: dict[str, list[str]] = {}
        if self._records:
            input_names = sorted(self._records[0].keys())
        for record in self._records:
            for name, value in record.items():
                shapes.setdefault(name, []).append(list(value.shape))
                dtypes.setdefault(name, []).append(str(value.dtype))
        return {
            "input_names": input_names,
            "batch_count": len(self._records),
            "shapes": {
                name: shape_list[: min(3, len(shape_list))]
                for name, shape_list in shapes.items()
            },
            "dtypes": {
                name: sorted(set(dtype_list))
                for name, dtype_list in dtypes.items()
            },
        }

    def get_next(self) -> Optional[dict[str, np.ndarray]]:
        """Return the next calibration sample for ONNX Runtime."""

        if self._index >= len(self._records):
            return None
        record = self._records[self._index]
        self._index += 1
        return record

    def rewind(self) -> None:
        """Reset iteration for tests or repeated calibration."""

        self._index = 0


def quantize_onnx_qdq_static(
    onnx_path: str | Path,
    output_path: str | Path,
    calibration_data: Iterable[Any],
    *,
    input_names: Sequence[str] = ("input",),
    sample_limit: Optional[int] = None,
    activation_type: str = "QUInt8",
    weight_type: str = "QInt8",
    per_channel: bool = False,
    reduce_range: bool = False,
    op_types_to_quantize: Optional[Sequence[str]] = None,
    extra_options: Optional[Mapping[str, Any]] = None,
) -> ONNXQDQQuantizationResult:
    """Run ONNX Runtime static QDQ quantization."""

    source = Path(onnx_path)
    if not source.is_file():
        raise XQTBackendError(f"ONNX file not found: {source}")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    try:
        from onnxruntime.quantization import (  # type: ignore[import-untyped]
            QuantFormat,
            QuantType,
            quantize_static,
        )
    except ImportError as exc:
        raise XQTBackendError(
            "onnxruntime.quantization is required for ONNX QDQ quantization"
        ) from exc

    quant_types = {
        "QInt8": QuantType.QInt8,
        "QUInt8": QuantType.QUInt8,
    }
    if activation_type not in quant_types:
        raise ValueError("activation_type must be QInt8 or QUInt8")
    if weight_type not in quant_types:
        raise ValueError("weight_type must be QInt8 or QUInt8")

    reader = IterableCalibrationDataReader(
        calibration_data,
        input_names=input_names,
        sample_limit=sample_limit,
    )
    if reader.samples == 0:
        raise ValueError("calibration_data must yield at least one batch")

    quantize_static(
        str(source),
        str(output),
        reader,
        quant_format=QuantFormat.QDQ,
        activation_type=quant_types[activation_type],
        weight_type=quant_types[weight_type],
        per_channel=per_channel,
        reduce_range=reduce_range,
        op_types_to_quantize=list(op_types_to_quantize or []),
        extra_options=dict(extra_options or {}),
    )
    return ONNXQDQQuantizationResult(
        path=output,
        source_path=source,
        checksum=file_sha256(output),
        calibration_samples=reader.samples,
        metadata={
            "input_names": list(input_names),
            "activation_type": activation_type,
            "weight_type": weight_type,
            "per_channel": per_channel,
            "reduce_range": reduce_range,
            "op_types_to_quantize": list(op_types_to_quantize or []),
            "calibration_summary": reader.summary,
        },
    )


__all__ = [
    "IterableCalibrationDataReader",
    "ONNXQDQQuantizationResult",
    "quantize_onnx_qdq_static",
]
