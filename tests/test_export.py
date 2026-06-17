import pytest
import torch
from torch import nn

from xqt.core.errors import XQTBackendError
from xqt.export import (
    compare_onnxruntime_outputs,
    export_onnx,
    export_torch_program,
    export_torchscript,
    validate_onnx,
)


def test_export_onnx_validates_and_compares_with_onnxruntime(tmp_path) -> None:
    model = nn.Linear(4, 2)
    example = torch.randn(3, 4)
    with torch.no_grad():
        reference = model(example)

    result = export_onnx(
        model,
        example,
        tmp_path / "linear.onnx",
        opset=18,
        dynamo=True,
    )
    diff = compare_onnxruntime_outputs(
        result.path,
        reference,
        example,
        atol=1e-5,
        rtol=1e-5,
    )

    assert result.path.is_file()
    assert result.checked is True
    assert result.checksum
    assert validate_onnx(result.path) is True
    assert diff.allclose is True


def test_compare_onnxruntime_outputs_rejects_multi_input(tmp_path) -> None:
    model = nn.Linear(4, 2)
    example = torch.randn(1, 4)
    result = export_onnx(model, example, tmp_path / "linear.onnx", dynamo=False)

    with pytest.raises(ValueError, match="one Tensor input"):
        compare_onnxruntime_outputs(
            result.path,
            model(example),
            (example, example),
        )


def test_export_torch_program_saves_and_validates_exported_program(tmp_path) -> None:
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    example = torch.randn(3, 4)

    result = export_torch_program(
        model,
        example,
        tmp_path / "linear.pt2",
        validate=True,
        compare_output=True,
    )
    loaded = torch.export.load(result.path)
    loaded_module = loaded.module()

    assert result.path.is_file()
    assert result.checked is True
    assert result.checksum
    assert result.output_diff is not None
    assert result.output_diff.allclose is True
    torch.testing.assert_close(model(example), loaded_module(example))


def test_export_torchscript_saves_and_compares_trace(tmp_path) -> None:
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    example = torch.randn(3, 4)

    result = export_torchscript(
        model,
        example,
        tmp_path / "linear.pt",
        method="trace",
    )
    loaded = torch.jit.load(str(result.path))

    assert result.path.is_file()
    assert result.checksum
    assert result.output_diff is not None
    assert result.output_diff.allclose is True
    torch.testing.assert_close(model(example), loaded(example))


def test_export_torchscript_rejects_unknown_method(tmp_path) -> None:
    model = nn.Linear(4, 2)
    example = torch.randn(1, 4)

    with pytest.raises(ValueError, match="trace or script"):
        export_torchscript(model, example, tmp_path / "linear.pt", method="compile")


def test_validate_onnx_wraps_missing_dependency(monkeypatch, tmp_path) -> None:
    path = tmp_path / "empty.onnx"
    path.write_bytes(b"not an onnx model")

    def fake_import(
        name: str,
        globals=None,
        locals=None,
        fromlist=(),
        level: int = 0,
    ):
        if name == "onnx":
            raise ImportError("missing")
        return original_import(name, globals, locals, fromlist, level)

    original_import = __import__
    monkeypatch.setattr("builtins.__import__", fake_import)

    with pytest.raises(XQTBackendError, match="onnx is required"):
        validate_onnx(path)
