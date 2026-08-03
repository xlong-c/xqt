from __future__ import annotations

from pathlib import Path

import pytest
import torch

from xqt.export import export_onnx, onnx_graph_diagnostics_report


def test_legacy_onnx_export_uses_dynamic_axes_from_shared_dynamic_shapes(
    tmp_path: Path,
) -> None:
    onnx = pytest.importorskip("onnx")
    output = tmp_path / "identity.onnx"

    result = export_onnx(
        torch.nn.Identity().eval(),
        torch.randn(8, 4),
        output,
        dynamo=False,
        dynamic_shapes={"input": {0: "batch"}},
    )

    exported = onnx.load(str(result.path))
    batch_dimension = exported.graph.input[0].type.tensor_type.shape.dim[0]
    assert result.metadata["dynamo"] is False
    assert result.metadata["dynamic_shapes"] == {"input": {0: "batch"}}
    diagnostics = result.metadata["onnx_graph_diagnostics"]
    assert diagnostics["status"] == "available"
    assert diagnostics["unsupported_op_count"] == 0
    assert {
        "value_name": "input",
        "axis": 0,
        "symbol": "batch",
    } in diagnostics["dynamic_axes"]
    assert batch_dimension.dim_param == "batch"


def test_onnx_graph_diagnostics_reports_custom_domain_ops(tmp_path: Path) -> None:
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper

    graph = helper.make_graph(
        [helper.make_node("CustomOp", ["input"], ["output"], domain="com.example")],
        "custom_graph",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, ["batch", 4])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, ["batch", 4])],
    )
    model = helper.make_model(graph)
    path = tmp_path / "custom.onnx"
    onnx.save(model, str(path))

    report = onnx_graph_diagnostics_report(path)

    assert report["status"] == "available"
    assert report["unsupported_op_count"] == 1
    assert report["unsupported_ops"][0]["op_type"] == "CustomOp"
    assert report["unsupported_ops"][0]["domain"] == "com.example"
    assert report["unsupported_ops"][0]["reasons"] == ["unsupported_domain"]
    assert {
        "value_name": "input",
        "axis": 0,
        "symbol": "batch",
    } in report["dynamic_axes"]
