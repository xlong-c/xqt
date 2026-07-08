from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import pytest
import torch

from xqt.core.config import load_xqt_config
from xqt.pipeline.runner import create_context
from xqt.quant import execute_quantization_plan
from xqt.quant.backends.onnx_qdq import ONNXQDQQuantizationResult
from xqt.quant.plan import build_quantization_plan
import xqt.quant.execution.executor as executor_module


def _base_config(tmp_path: Path) -> dict[str, Any]:
    return {
        "config_version": 1,
        "project": {
            "name": "quant_onnx_qdq_report",
            "artifact_dir": str(tmp_path / "artifacts"),
        },
        "compression": {
            "quant": {
                "enabled": True,
                "backend": "onnxruntime_qdq",
                "method": "static_qdq_int8",
                "strategy": "static_qdq_int8",
                "policy": {
                    "onnx_path": str(tmp_path / "source.onnx"),
                    "input_names": ["input"],
                    "op_types_to_quantize": ["MatMul", "Gemm"],
                },
            }
        },
    }


def test_onnx_qdq_requires_calibration_inputs(tmp_path: Path) -> None:
    config = load_xqt_config(_base_config(tmp_path))
    context = create_context(config)
    plan = build_quantization_plan(config.compression.quant)

    with pytest.raises(ValueError, match="calibration_inputs are required"):
        execute_quantization_plan(context, plan)


def test_onnx_qdq_report_includes_calibration_and_quantized_ops(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_xqt_config(_base_config(tmp_path))
    calibration_inputs = [torch.randn(1, 4), torch.randn(1, 4)]
    context = create_context(config, calibration_inputs=calibration_inputs)
    plan = build_quantization_plan(config.compression.quant)

    def fake_quantize(
        onnx_path: str | Path,
        output_path: str | Path,
        calibration_data: Iterable[Any],
        **_: Any,
    ) -> ONNXQDQQuantizationResult:
        samples = list(calibration_data)
        return ONNXQDQQuantizationResult(
            path=Path(output_path),
            source_path=Path(onnx_path),
            checksum="fake-checksum",
            calibration_samples=len(samples),
            metadata={
                "calibration_summary": {
                    "sample_count": len(samples),
                    "batch_count": len(samples),
                    "input_signature": {
                        "input": {
                            "shapes": [[1, 4]],
                            "dtypes": ["float32"],
                        }
                    },
                    "calibrator_type": "FakeCalibrationDataReader",
                    "observer_type": "fake-observer",
                }
            },
        )

    monkeypatch.setattr(
        executor_module,
        "_onnx_qdq_graph_summary",
        lambda _: {
            "node_count": 5,
            "op_type_counts": {
                "QuantizeLinear": 2,
                "DequantizeLinear": 2,
                "MatMul": 1,
            },
            "qdq_node_count": 4,
            "quantize_linear_count": 2,
            "dequantize_linear_count": 2,
        },
    )

    execution = execute_quantization_plan(
        context,
        plan,
        quantize_onnx_qdq_static_fn=fake_quantize,
    )

    report = execution.reports[0]
    assert report.backend == "onnxruntime_qdq"
    assert report.strategy == "static_qdq_int8"
    assert report.calibration_samples == 2
    assert report.calibration_summary["sample_count"] == 2
    assert report.metadata["qdq_node_count"] == 4
    assert report.metadata["quantized_op_types"] == ["MatMul"]
    assert report.quantized_modules == ["onnx::MatMul"]
