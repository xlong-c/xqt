from pathlib import Path

import numpy as np
import pytest
import torch

from xqt.contracts import InferenceContract
from xqt.core.errors import XQTConfigError
from xqt.runtime import (
    ImageClassificationAdapter,
    InferenceSession,
    TensorInferenceAdapter,
    create_inference_adapter,
    create_inference_session,
    inference_adapter_names,
    register_inference_adapter,
    write_model_package,
)
from xqt.workflows.session_targets import build_session_export_targets


def test_inference_contract_falls_back_to_legacy_io() -> None:
    contract = InferenceContract.from_dict(
        {},
        io={
            "inputs": [{"name": "input"}],
            "outputs": [{"name": "output"}],
        },
    )

    assert contract.adapter == "tensor"
    assert contract.input_names == ("input",)
    assert contract.output_names == ("output",)


def test_tensor_inference_session_maps_semantic_inputs() -> None:
    contract = InferenceContract.from_dict(
        {
            "family": "transformer",
            "adapter": "tensor",
            "inputs": [{"name": "pixel_values", "semantic": "image"}],
            "outputs": [{"name": "output", "semantic": "logits"}],
        }
    )
    captured: dict[str, object] = {}

    def runner(inputs: object) -> list[object]:
        captured["inputs"] = inputs
        return ["result"]

    session = InferenceSession(runner, contract, TensorInferenceAdapter())
    result = session.predict({"image": torch.ones(1, 3)})

    assert result == {"logits": "result"}
    inputs = captured["inputs"]
    assert isinstance(inputs, dict)
    assert torch.equal(inputs["pixel_values"], torch.ones(1, 3))


def test_image_classification_adapter_preprocesses_and_decodes() -> None:
    contract = InferenceContract.from_dict(
        {
            "family": "vision",
            "adapter": "vision.classification",
            "inputs": [{"name": "input", "semantic": "image"}],
            "outputs": [{"name": "output", "semantic": "logits"}],
            "config": {
                "resize": [2, 2],
                "mean": [0.0, 0.0, 0.0],
                "std": [1.0, 1.0, 1.0],
            },
        }
    )
    captured: dict[str, object] = {}

    def runner(inputs: object) -> list[object]:
        captured["inputs"] = inputs
        return [np.asarray([[1.0, 3.0]], dtype=np.float32)]

    session = InferenceSession(
        runner,
        contract,
        ImageClassificationAdapter(),
    )
    image = torch.zeros(4, 4, 3, dtype=torch.uint8)
    result = session.predict({"image": image})

    inputs = captured["inputs"]
    assert isinstance(inputs, dict)
    assert tuple(inputs["input"].shape) == (1, 3, 2, 2)
    assert np.asarray(result["logits"]).tolist() == [[1.0, 3.0]]
    assert np.asarray(result["class_ids"]).tolist() == [1]
    assert np.asarray(result["probabilities"]).shape == (1, 2)


def test_create_inference_session_uses_manifest_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"fake-onnx")
    package_dir = write_model_package(
        model_path=model_path,
        output_dir=tmp_path / "model.xqtpkg",
        model_format="onnx",
        runtime_name="onnxruntime",
        runtime_config={"providers": ["CPUExecutionProvider"]},
        io={
            "inputs": [{"name": "input"}],
            "outputs": [{"name": "output", "semantic": "logits"}],
        },
        inference={
            "family": "transformer",
            "adapter": "tensor",
        },
    )

    class FakeSession:
        def run(self, output_names: object, feeds: object) -> list[object]:
            del output_names, feeds
            return ["ok"]

    monkeypatch.setattr(
        "xqt.runtime.package.create_onnxruntime_session",
        lambda path, *, providers=None: FakeSession(),
    )

    session = create_inference_session(package_dir)
    assert session.contract.adapter == "tensor"
    assert session.predict(torch.ones(1, 4)) == {"logits": "ok"}


def test_model_family_adapter_can_be_registered_once() -> None:
    class FamilyAdapter(TensorInferenceAdapter):
        name = "test.family"

    register_inference_adapter(FamilyAdapter.name, FamilyAdapter, version="2")
    contract = InferenceContract.from_dict(
        {
            "family": "test_family",
            "adapter": FamilyAdapter.name,
            "adapter_version": "2",
        }
    )

    resolved = create_inference_adapter(contract)

    assert isinstance(resolved, FamilyAdapter)
    assert FamilyAdapter.name in inference_adapter_names()


def test_inference_adapter_version_must_match_contract() -> None:
    class VersionedAdapter(TensorInferenceAdapter):
        name = "test.versioned"

    register_inference_adapter(VersionedAdapter.name, VersionedAdapter, version="2")
    contract = InferenceContract.from_dict(
        {
            "family": "test_family",
            "adapter": VersionedAdapter.name,
        }
    )

    with pytest.raises(XQTConfigError, match="does not support contract version"):
        create_inference_adapter(contract)


def test_session_export_target_declares_inference_contract() -> None:
    targets = build_session_export_targets(
        "export",
        format="onnx",
        output_path="artifacts/model.onnx",
        targets=None,
        target_params=None,
        opset=17,
        inference={
            "family": "transformer",
            "adapter": "tensor",
            "inputs": [{"name": "input", "semantic": "tokens"}],
        },
    )

    assert targets == [
        {
            "format": "onnx",
            "output_path": "artifacts/model.onnx",
            "opset": 17,
            "inference": {
                "family": "transformer",
                "adapter": "tensor",
                "inputs": [{"name": "input", "semantic": "tokens"}],
            },
        }
    ]
