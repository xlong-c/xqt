import pytest
import torch
from torch import nn

from xqt.core.errors import XQTBackendError
from xqt.export import compare_openvino_outputs, export_openvino_ir


def test_export_openvino_ir_requires_openvino_dependency() -> None:
    with pytest.raises(XQTBackendError, match="openvino is required"):
        export_openvino_ir(nn.Linear(2, 2), "model.xml", example_input=torch.randn(1, 2))


def test_export_openvino_ir_requires_example_input_for_pytorch_model(monkeypatch) -> None:
    class FakeOpenVINO:
        @staticmethod
        def convert_model(*args, **kwargs):
            return object()

        @staticmethod
        def save_model(*args, **kwargs):
            return None

    monkeypatch.setattr("xqt.export.openvino._import_openvino", lambda: FakeOpenVINO)

    with pytest.raises(ValueError, match="example_input"):
        export_openvino_ir(nn.Linear(2, 2), "model.xml")


def test_compare_openvino_outputs_requires_dependency() -> None:
    with pytest.raises(XQTBackendError, match="openvino is required"):
        compare_openvino_outputs(
            "model.xml",
            torch.zeros(1, 2),
            torch.zeros(1, 2),
        )
