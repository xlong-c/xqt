from __future__ import annotations

from pathlib import Path

import pytest
import torch

from xqt.export import export_onnx


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
    assert batch_dimension.dim_param == "batch"
