import json
from pathlib import Path

import pytest
import torch

from xqt.core.errors import XQTArtifactError, XQTConfigError
from xqt.contracts import InferenceContractConfig
from xqt.core.schema import ExportTargetConfig
from xqt.core.types import XQTContext
import xqt.pipeline.export_pass as export_pass_module
from xqt.pipeline.passes import run_export_stage
from xqt.pipeline.runner import create_context
from xqt.runtime import create_inference_runner, load_model_package, write_model_package
from xqt.workflows import load_optimization_config
from xqt.workflows.stage_specs import ExportStageSpec, build_stage_spec


def _runtime_context(tmp_path: Path) -> XQTContext:
    config = load_optimization_config(
        {
            "project": {
                "name": "runtime_model_package_test",
                "artifact_dir": str(tmp_path / "artifacts"),
            },
            "model": {"target": "torch.nn:Identity"},
        }
    )
    return create_context(
        config,
        model=torch.nn.Identity().eval(),
        example_inputs=torch.randn(1, 4),
    )


def test_write_and_load_model_package_roundtrip(tmp_path: Path) -> None:
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"fake-onnx")

    package_dir = write_model_package(
        model_path=model_path,
        output_dir=tmp_path / "model.xqtpkg",
        model_format="onnx",
        runtime_name="onnxruntime",
        runtime_config={"providers": ["CPUExecutionProvider"]},
        model_metadata={
            "opset": 17,
            "checked": True,
            "input_names": ["input"],
            "output_names": ["output"],
        },
        io={
            "inputs": [{"name": "input"}],
            "outputs": [{"name": "output"}],
        },
    )

    package = load_model_package(package_dir)

    assert package.package_dir == package_dir.resolve()
    assert package.model_path.read_bytes() == b"fake-onnx"
    assert package.runtime_config["runtime"] == "onnxruntime"
    assert package.runtime_config["providers"] == ["CPUExecutionProvider"]
    assert package.manifest.entrypoints["model"] == "model/model.onnx"
    assert package.manifest.model["input_names"] == ["input"]
    assert package.manifest.inference["adapter"] == "tensor"
    assert package.manifest.inference["inputs"] == [{"name": "input"}]


def test_load_model_package_rejects_entrypoint_escape(tmp_path: Path) -> None:
    package_dir = tmp_path / "bad.xqtpkg"
    (package_dir / "runtime").mkdir(parents=True, exist_ok=True)
    (package_dir / "runtime" / "config.json").write_text(
        '{"runtime": "onnxruntime"}',
        encoding="utf-8",
    )
    (package_dir / "manifest.json").write_text(
        "\n".join(
            [
                "{",
                '  "schema_version": "1.0",',
                '  "artifact_type": "xqt_model_package",',
                '  "package_version": "1.0",',
                '  "entrypoints": {',
                '    "model": "../escape.onnx",',
                '    "runtime_config": "runtime/config.json"',
                "  },",
                '  "model": {"format": "onnx"},',
                '  "runtime": {"preferred_backend": "onnxruntime"},',
                '  "io": {},',
                '  "quantization": {},',
                '  "metadata": {}',
                "}",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(XQTArtifactError, match="escapes the model package root"):
        load_model_package(package_dir)


def test_load_model_package_rejects_invalid_inference_contract(
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"fake-onnx")
    package_dir = write_model_package(
        model_path=model_path,
        output_dir=tmp_path / "invalid_inference.xqtpkg",
        model_format="onnx",
        runtime_name="onnxruntime",
        io={"inputs": [{"name": "input"}], "outputs": [{"name": "output"}]},
    )
    manifest_path = package_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["inference"]["adapter"] = ""
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(XQTArtifactError, match="manifest.inference is invalid"):
        load_model_package(package_dir)


def test_load_model_package_legacy_manifest_falls_back_to_io(
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"fake-onnx")
    package_dir = write_model_package(
        model_path=model_path,
        output_dir=tmp_path / "legacy.xqtpkg",
        model_format="onnx",
        runtime_name="onnxruntime",
        io={
            "inputs": [{"name": "input"}],
            "outputs": [{"name": "output"}],
        },
    )
    manifest_path = package_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("inference")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    package = load_model_package(package_dir)

    assert package.inference_contract.adapter == "tensor"
    assert package.inference_contract.input_names == ("input",)
    assert package.inference_contract.output_names == ("output",)


def test_create_inference_runner_uses_package_runtime_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"fake-onnx")
    package_dir = write_model_package(
        model_path=model_path,
        output_dir=tmp_path / "runner.xqtpkg",
        model_format="onnx",
        runtime_name="onnxruntime",
        runtime_config={"providers": ["CUDAExecutionProvider"]},
        model_metadata={"input_names": ["input"], "output_names": ["output"]},
        io={
            "inputs": [{"name": "input"}],
            "outputs": [{"name": "output"}],
        },
    )
    captured: dict[str, object] = {}

    class _FakeSession:
        def run(self, output_names, feeds):
            captured["output_names"] = output_names
            captured["feeds"] = feeds
            return ["ok"]

    def _fake_create_onnxruntime_session(path: Path, *, providers=None):
        captured["path"] = path
        captured["providers"] = list(providers or [])
        return _FakeSession()

    monkeypatch.setattr(
        "xqt.runtime.package.create_onnxruntime_session",
        _fake_create_onnxruntime_session,
    )

    runner = create_inference_runner(package_dir)
    outputs = runner(torch.randn(1, 4))

    assert outputs == ["ok"]
    assert captured["path"] == package_dir / "model" / "model.onnx"
    assert captured["providers"] == ["CUDAExecutionProvider"]
    feeds = captured["feeds"]
    assert isinstance(feeds, dict)
    assert set(feeds) == {"input"}


def test_run_export_stage_attaches_onnx_model_package(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _runtime_context(tmp_path)
    spec = ExportStageSpec(
        targets=[
            ExportTargetConfig(
                format="onnx",
                output_path=str(tmp_path / "artifacts" / "export" / "model.onnx"),
                opset=17,
                inference={
                    "family": "vision",
                    "adapter": "vision.classification",
                    "inputs": [{"name": "input", "semantic": "image"}],
                    "outputs": [{"name": "output", "semantic": "logits"}],
                    "config": {"resize": [224, 224]},
                },
            )
        ]
    )

    def _fake_handle_onnx(context_value, target, index, **kwargs):
        del context_value, kwargs
        output_path = Path(str(target.output_path))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"fake-onnx")
        item = {
            "path": str(output_path),
            "format": "onnx",
            "opset": 17,
            "checked": True,
            "checksum": "deadbeef",
            "input_names": ["input"],
            "output_names": ["output"],
            "dynamic_shapes": {},
            "output_diff": None,
            "pre_export_fusion": {},
            "pre_export_lowering": {},
            "onnx_optimization": None,
            "export_guard": {"guarded": False},
        }
        return item, {"format": "onnx", "path": str(output_path), "index": index}

    monkeypatch.setitem(
        export_pass_module._FORMAT_HANDLERS,
        "onnx",
        _fake_handle_onnx,
    )

    run_export_stage(context, spec, stage_kind="export")

    package_dir = Path(context.artifacts["model_package_0"])
    assert package_dir.is_dir()
    package = load_model_package(package_dir)
    assert package.runtime_config["providers"] == ["CPUExecutionProvider"]
    assert package.manifest.inference["family"] == "vision"
    assert package.manifest.inference["adapter"] == "vision.classification"
    assert package.manifest.inference["config"] == {"resize": [224, 224]}
    assert context.metrics["export"]["targets"][0]["model_package"] == str(package_dir)
    assert any(
        artifact.format == "xqt_model_package"
        for artifact in context.manifest.artifacts
    )


def test_export_stage_loads_and_validates_inference_contract() -> None:
    spec = build_stage_spec(
        "export",
        {
            "targets": [
                {
                    "format": "onnx",
                    "inference": {
                        "family": "transformer",
                        "adapter": "tensor",
                        "inputs": [{"name": "input", "semantic": "tokens"}],
                    },
                }
            ]
        },
    )

    assert isinstance(spec, ExportStageSpec)
    assert isinstance(spec.targets[0].inference, InferenceContractConfig)
    assert spec.targets[0].inference.family == "transformer"
    assert spec.targets[0].inference.inputs[0]["semantic"] == "tokens"

    with pytest.raises(XQTConfigError, match="inference is invalid"):
        build_stage_spec(
            "export",
            {
                "targets": [
                    {
                        "format": "onnx",
                        "inference": {
                            "adapter": "",
                        },
                    }
                ]
            },
        )


def test_write_and_load_model_package_with_compute_config(tmp_path: Path) -> None:
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"fake-onnx")
    package_dir = write_model_package(
        model_path=model_path,
        output_dir=tmp_path / "model_compute.xqtpkg",
        model_format="onnx",
        runtime_name="onnxruntime",
        runtime_config={"providers": ["CPUExecutionProvider"]},
        compute_config={
            "schema_version": "1.0",
            "default_precision": "w8a8",
            "modules": [
                {
                    "name": "proj",
                    "compute_contract": "int8_mma",
                    "required_capabilities": ["int8_mma"],
                    "preferred_engines": ["tilelang"],
                }
            ],
        },
    )
    package = load_model_package(package_dir)
    assert package.compute_config is not None
    assert package.compute_config["modules"][0]["compute_contract"] == "int8_mma"
    assert "required_engine" not in package.compute_config
    assert package.manifest.entrypoints["compute_config"] == "runtime/compute.json"
    assert package.manifest.runtime.get("has_compute_config") is True
