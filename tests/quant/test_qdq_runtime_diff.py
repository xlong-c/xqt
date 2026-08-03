"""AA2: QDQ ONNX runtime diff probe (or honest skip reason)."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from xqt.export.onnx_exporter import export_onnx
from xqt.quant.backends.onnx_qdq import assess_qdq_runtime_diff


def test_assess_qdq_runtime_diff_missing_file() -> None:
    report = assess_qdq_runtime_diff("/tmp/xqt_no_such_qdq_model.onnx")
    assert report["session_created"] is False
    assert report["diff_ran"] is False
    assert str(report["runtime_diff_skip_reason"]).startswith("onnx_file_missing:")


def test_assess_qdq_runtime_diff_session_without_compare(tmp_path: Path) -> None:
    model = nn.Linear(4, 2).eval()
    onnx_path = tmp_path / "linear.onnx"
    export_onnx(
        model,
        torch.randn(1, 4),
        onnx_path,
        input_names=["input"],
        dynamo=False,
        validate=False,
    )
    report = assess_qdq_runtime_diff(onnx_path, run_diff=False)
    assert report["onnxruntime_available"] is True
    assert report["session_created"] is True
    assert report["diff_ran"] is False
    assert "runtime_diff_not_requested" in str(report["runtime_diff_skip_reason"])


def test_assess_qdq_runtime_diff_with_reference(tmp_path: Path) -> None:
    model = nn.Linear(4, 2).eval()
    example = torch.randn(2, 4)
    onnx_path = tmp_path / "linear_diff.onnx"
    export_onnx(
        model,
        example,
        onnx_path,
        input_names=["input"],
        dynamo=False,
        validate=False,
    )
    report = assess_qdq_runtime_diff(
        onnx_path,
        run_diff=True,
        example_input=example,
        reference_model=model,
        input_names=["input"],
    )
    assert report["session_created"] is True
    assert report["diff_ran"] is True
    assert report["runtime_diff_skip_reason"] is None
    assert isinstance(report["diff"], dict)
    assert "allclose" in report["diff"]
