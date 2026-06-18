import pytest
import torch
from torch import nn

from xqt.core.errors import XQTBackendError
from xqt.export import (
    apply_pre_export_fusion,
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

    with pytest.raises(ValueError, match="arity must match input_names"):
        compare_onnxruntime_outputs(
            result.path,
            model(example),
            (example, example),
        )


class PairAddModel(nn.Module):
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x + y


class MappingAddModel(nn.Module):
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return {"logits": input_ids + attention_mask}


def test_compare_onnxruntime_outputs_supports_tuple_inputs(tmp_path) -> None:
    model = PairAddModel().eval()
    example = (torch.randn(2, 4), torch.randn(2, 4))
    with torch.no_grad():
        reference = model(*example)

    result = export_onnx(
        model,
        example,
        tmp_path / "pair.onnx",
        dynamo=False,
        input_names=["left", "right"],
    )
    diff = compare_onnxruntime_outputs(
        result.path,
        reference,
        example,
        input_names=["left", "right"],
        atol=1e-5,
        rtol=1e-5,
    )

    assert diff.allclose is True


def test_compare_onnxruntime_outputs_supports_mapping_inputs(tmp_path) -> None:
    model = MappingAddModel().eval()
    example = {
        "input_ids": torch.randn(2, 4),
        "attention_mask": torch.randn(2, 4),
    }
    with torch.no_grad():
        reference = model(**example)["logits"]

    result = export_onnx(
        model,
        example,
        tmp_path / "mapping.onnx",
        dynamo=False,
        input_names=["input_ids", "attention_mask"],
    )
    diff = compare_onnxruntime_outputs(
        result.path,
        reference,
        example,
        input_names=["input_ids", "attention_mask"],
        atol=1e-5,
        rtol=1e-5,
    )

    assert diff.allclose is True


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


def test_export_torch_program_supports_mapping_inputs(tmp_path) -> None:
    model = MappingAddModel().eval()
    example = {
        "input_ids": torch.randn(2, 4),
        "attention_mask": torch.randn(2, 4),
    }

    result = export_torch_program(
        model,
        example,
        tmp_path / "mapping.pt2",
        validate=True,
        compare_output=True,
    )
    loaded = torch.export.load(result.path)
    loaded_module = loaded.module()

    assert result.output_diff is not None
    assert result.output_diff.allclose is True
    torch.testing.assert_close(
        model(**example)["logits"],
        loaded_module(**example)["logits"],
    )


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


def test_export_torchscript_supports_mapping_inputs(tmp_path) -> None:
    model = MappingAddModel().eval()
    example = {
        "input_ids": torch.randn(2, 4),
        "attention_mask": torch.randn(2, 4),
    }

    result = export_torchscript(
        model,
        example,
        tmp_path / "mapping.pt",
        method="trace",
    )
    loaded = torch.jit.load(str(result.path))

    assert result.output_diff is not None
    assert result.output_diff.allclose is True
    torch.testing.assert_close(
        model(**example)["logits"],
        loaded(**example)["logits"],
    )


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


class TinyConvBnRelu(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 4, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(4)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


def test_apply_pre_export_fusion_eager_fuses_conv_bn_relu_without_inplace() -> None:
    model = TinyConvBnRelu().eval()
    result = apply_pre_export_fusion(
        model,
        {
            "enabled": True,
            "mode": "eager",
            "modules_to_fuse": [["conv", "bn", "relu"]],
        },
    )

    assert result.applied is True
    assert result.mode == "eager"
    assert result.inplace is False
    assert result.fused_groups == [["conv", "bn", "relu"]]
    assert result.metadata["fused_groups"] == [["conv", "bn", "relu"]]
    assert type(result.model.conv).__name__ == "ConvReLU2d"
    assert isinstance(result.model.bn, nn.Identity)
    assert isinstance(result.model.relu, nn.Identity)
    assert isinstance(model.conv, nn.Conv2d)
    assert isinstance(model.bn, nn.BatchNorm2d)
    assert isinstance(model.relu, nn.ReLU)


def test_apply_pre_export_fusion_fx_returns_graph_module() -> None:
    model = TinyConvBnRelu().eval()
    result = apply_pre_export_fusion(
        model,
        {
            "enabled": True,
            "mode": "fx",
        },
    )

    assert result.applied is True
    assert result.mode == "fx"
    assert result.metadata["fused_groups"] == []
    assert result.model.__class__.__name__.startswith("GraphModule")


def test_export_onnx_records_pre_export_fusion_metadata(tmp_path) -> None:
    model = TinyConvBnRelu().eval()
    example = torch.randn(1, 3, 8, 8)

    result = export_onnx(
        model,
        example,
        tmp_path / "conv.onnx",
        dynamo=False,
        pre_export_fusion={
            "enabled": True,
            "mode": "eager",
            "modules_to_fuse": [["conv", "bn", "relu"]],
        },
    )

    fusion = result.metadata["pre_export_fusion"]
    assert fusion["enabled"] is True
    assert fusion["mode"] == "eager"
    assert fusion["fused_groups"] == [["conv", "bn", "relu"]]
