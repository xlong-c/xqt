from __future__ import annotations

from pathlib import Path

import torch

import xqt.pipeline.export_pass as export_pass_module
from xqt.core.artifact import ArtifactManifest
from xqt.core.schema import (
    ExportTargetConfig,
    OpenVINOExportConfig,
    OutputDiffConfig,
    TensorRTExportConfig,
)
from xqt.core.types import XQTContext
from xqt.export import (
    OpenVINOExportResult,
    TensorRTBuildResult,
    TensorRTPluginLibraryCheck,
    TensorRTPluginValidationResult,
    export_target_capability_report,
    openvino_runtime_layer_report,
    tensorrt_runtime_layer_report,
)
from xqt.pipeline.export_pass import ExportPass


def test_export_target_capability_report_covers_dynamic_quant_and_plugins() -> None:
    targets = [
        ExportTargetConfig(
            format="onnx",
            precision="int8",
            opset=17,
            dynamic_shapes={"input": {0: "batch"}},
        ),
        ExportTargetConfig(format="tensorrt"),
    ]
    targets[1].tensorrt.plugin_libraries = ["plugins/libcustom.so"]
    targets[1].tensorrt.validate_plugin_libraries_loadable = True

    report = export_target_capability_report(targets)

    onnx = report["targets"][0]
    tensorrt = report["targets"][1]
    assert report["target_count"] == 2
    assert report["unsupported_formats"] == []
    assert onnx["format"] == "onnx"
    assert onnx["precision_supported"] is True
    assert onnx["dynamic_shapes_requested"] is True
    assert onnx["dynamic_shapes_compatible"] is True
    assert onnx["quantization_supported"] is True
    assert tensorrt["plugin_support"] == "supported"
    assert tensorrt["plugin_libraries"] == ["plugins/libcustom.so"]
    assert tensorrt["plugin_loadability_requested"] is True


def test_export_pass_records_target_capability_and_artifact_lineage(
    tmp_path: Path,
    monkeypatch,
) -> None:
    model = torch.nn.Linear(4, 2).eval()
    context = XQTContext(
        model=model,
        example_inputs=torch.randn(1, 4),
        artifact_dir=str(tmp_path),
        project_name="export-reporting",
        manifest=ArtifactManifest(project_name="export-reporting"),
    )
    context.metrics["quant"] = {"backend": "torchao", "strategy": "w8a8_int8"}
    context.metrics["operator_optimization"] = {
        "target_count": 1,
        "targets": [{"name": "linear", "applied": True}],
    }
    target = ExportTargetConfig(format="onnx", opset=17)

    def _fake_handle_onnx(context_value, target_value, index, **kwargs):
        del context_value, kwargs
        output_path = Path(str(target_value.output_path or tmp_path / "model.onnx"))
        output_path.write_bytes(b"fake-onnx")
        item = {
            "path": str(output_path),
            "format": "onnx",
            "opset": target_value.opset,
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

    monkeypatch.setitem(export_pass_module._FORMAT_HANDLERS, "onnx", _fake_handle_onnx)

    ExportPass().run(
        context,
        targets=[target],
        output_diff=OutputDiffConfig(),
        stage_kind="export",
    )

    metrics = context.metrics["export"]
    capability = metrics["target_capabilities"]
    lineage = metrics["artifact_lineage"]
    assert capability["targets"][0]["format"] == "onnx"
    assert capability["targets"][0]["opset"] == 17
    assert lineage["artifacts"][0]["artifact_checksum"] == "deadbeef"
    assert lineage["artifacts"][0]["artifact_size_bytes"] == len(b"fake-onnx")
    assert lineage["artifacts"][0]["input_signature"] == {
        "input_names": ["input"],
        "dynamic_shapes": {},
    }
    assert lineage["upstream_stages"] == ["quant", "operator_optimization"]
    assert any(
        metric.name == "export.target_capabilities"
        for metric in context.manifest.metrics
    )
    assert any(
        metric.name == "export.artifact_lineage"
        for metric in context.manifest.metrics
    )


def test_openvino_runtime_layer_report_marks_dry_run_command_only() -> None:
    target = ExportTargetConfig(format="openvino")
    target.openvino.runtime_diff = True
    result = OpenVINOExportResult(
        xml_path=Path("model.xml"),
        bin_path=Path("model.bin"),
        checksum=None,
        dry_run=True,
        source_path=Path("model.onnx"),
        metadata={
            "input_shape": [1, 3, 224, 224],
            "command": ["openvino.convert_model", "model.onnx"],
        },
    )

    report = openvino_runtime_layer_report(
        target=target,
        export_result=result,
        output_diff=None,
        source="model.onnx",
    )

    layers = report["layers"]
    assert report["status"] == "command_only"
    assert report["blocking_failures"] == []
    assert layers["conversion"]["status"] == "command_only"
    assert layers["runtime_load"]["status"] == "skipped_dry_run"
    assert layers["output_diff"]["status"] == "skipped_dry_run"
    assert layers["runtime_benchmark"]["status"] == "not_configured"


def test_openvino_runtime_layer_report_records_runtime_diff_success(
    tmp_path: Path,
) -> None:
    xml_path = tmp_path / "model.xml"
    bin_path = tmp_path / "model.bin"
    xml_path.write_text("<xml />", encoding="utf-8")
    bin_path.write_bytes(b"weights")
    target = ExportTargetConfig(format="openvino")
    target.openvino.device = "CPU"
    output_diff = {
        "valid": True,
        "allclose": True,
        "max_abs": 0.0,
        "mean_abs": 0.0,
        "message": "ok",
    }
    result = OpenVINOExportResult(
        xml_path=xml_path,
        bin_path=bin_path,
        checksum="deadbeef",
        dry_run=False,
        source_path=Path("model.onnx"),
        metadata={"input_shape": [1, 4]},
    )

    report = openvino_runtime_layer_report(
        target=target,
        export_result=result,
        output_diff=output_diff,
        source="model.onnx",
    )

    layers = report["layers"]
    assert report["status"] == "runtime_diffed"
    assert layers["conversion"]["status"] == "converted"
    assert layers["runtime_load"]["status"] == "loaded"
    assert layers["output_diff"]["status"] == "passed"
    assert layers["output_diff"]["diff"] == output_diff
    assert layers["runtime_benchmark"]["status"] == "not_configured"


def test_export_pass_records_openvino_runtime_layers(
    tmp_path: Path,
    monkeypatch,
) -> None:
    onnx_path = tmp_path / "model.onnx"
    xml_path = tmp_path / "model.xml"
    onnx_path.write_bytes(b"onnx-model")

    def _export_openvino(
        source: object,
        output_path: object,
        *,
        example_input: object = None,
        input_shape: list[int] | None = None,
        dry_run: bool = False,
    ) -> OpenVINOExportResult:
        assert source == str(onnx_path)
        assert output_path == str(xml_path)
        assert example_input is None
        assert input_shape == [1, 4]
        assert dry_run is True
        return OpenVINOExportResult(
            xml_path=xml_path,
            bin_path=xml_path.with_suffix(".bin"),
            checksum=None,
            dry_run=True,
            source_path=onnx_path,
            metadata={
                "input_shape": input_shape,
                "command": ["openvino.convert_model", str(onnx_path)],
            },
        )

    monkeypatch.setattr(export_pass_module, "export_openvino_ir", _export_openvino)
    context = XQTContext(
        artifact_dir=str(tmp_path),
        export_targets=[
            ExportTargetConfig(
                format="openvino",
                output_path=str(xml_path),
                openvino=OpenVINOExportConfig(
                    onnx_path=str(onnx_path),
                    input_shape=[1, 4],
                    dry_run=True,
                    runtime_diff=False,
                    device="CPU",
                ),
            )
        ],
        output_diff_config=OutputDiffConfig(),
    )

    ExportPass().run(context)

    artifact = context.metrics["export"]["artifacts"][0]
    target = context.metrics["export"]["targets"][0]
    layers = artifact["openvino_runtime_layers"]["layers"]
    assert layers["conversion"]["status"] == "command_only"
    assert layers["runtime_load"]["status"] == "skipped_dry_run"
    assert layers["output_diff"]["requested"] is False
    assert target["openvino_runtime_layers"]["layers"]["conversion"][
        "status"
    ] == "command_only"


def test_tensorrt_runtime_layer_report_marks_dry_run_command_only() -> None:
    target = ExportTargetConfig(format="tensorrt")
    result = TensorRTBuildResult(
        engine_path=Path("model.engine"),
        command=["trtexec", "--dryRun"],
        dry_run=True,
        metadata={"backend": "trtexec"},
    )

    report = tensorrt_runtime_layer_report(
        target=target,
        build_result=result,
        plugin_validation=None,
        runtime_benchmark=None,
        source_onnx="model.onnx",
    )

    layers = report["layers"]
    assert report["status"] == "command_only"
    assert report["blocking_failures"] == []
    assert layers["dry_run"]["status"] == "command_only"
    assert layers["engine_build"]["status"] == "skipped_dry_run"
    assert layers["plugin_presence"]["status"] == "not_requested"
    assert layers["plugin_loadability"]["status"] == "not_requested"
    assert layers["runtime_benchmark"]["status"] == "not_requested"


def test_tensorrt_runtime_layer_report_splits_plugin_presence_and_loadability() -> None:
    target = ExportTargetConfig(format="tensorrt")
    result = TensorRTBuildResult(
        engine_path=Path("model.engine"),
        command=["trtexec", "--dryRun"],
        dry_run=True,
        metadata={"backend": "trtexec"},
    )
    validation = TensorRTPluginValidationResult(
        status="missing",
        loadability_requested=True,
        plugin_libraries=[
            TensorRTPluginLibraryCheck(
                path="plugins/missing.so",
                exists=False,
                load_requested=True,
                error="TensorRT plugin library not found",
            )
        ],
    )

    report = tensorrt_runtime_layer_report(
        target=target,
        build_result=result,
        plugin_validation=validation,
        runtime_benchmark=None,
        source_onnx="model.onnx",
    )

    layers = report["layers"]
    assert report["status"] == "attention_required"
    assert report["blocking_failures"] == [
        "plugin_presence",
        "plugin_loadability",
    ]
    assert layers["plugin_presence"]["status"] == "missing"
    assert layers["plugin_presence"]["missing_paths"] == ["plugins/missing.so"]
    assert layers["plugin_loadability"]["status"] == "missing"
    assert layers["plugin_loadability"]["failed_paths"] == ["plugins/missing.so"]


def test_export_pass_records_tensorrt_runtime_layers(
    tmp_path: Path,
    monkeypatch,
) -> None:
    onnx_path = tmp_path / "model.onnx"
    engine_path = tmp_path / "model.engine"
    plugin_path = tmp_path / "libcustom_plugin.so"
    onnx_path.write_bytes(b"onnx-model")
    plugin_path.write_bytes(b"")

    def _build_engine(
        onnx: object,
        engine: object,
        **kwargs: object,
    ) -> TensorRTBuildResult:
        assert onnx == str(onnx_path)
        assert engine == str(engine_path)
        assert kwargs["plugin_libraries"] == [str(plugin_path)]
        return TensorRTBuildResult(
            engine_path=engine_path,
            command=["trtexec", "--dryRun"],
            dry_run=True,
            metadata={
                "backend": "trtexec",
                "plugin_libraries": [str(plugin_path)],
                "serialize_plugin_libraries": True,
            },
        )

    monkeypatch.setattr(export_pass_module, "build_tensorrt_engine", _build_engine)
    context = XQTContext(
        artifact_dir=str(tmp_path),
        export_targets=[
            ExportTargetConfig(
                format="tensorrt",
                output_path=str(engine_path),
                tensorrt=TensorRTExportConfig(
                    onnx_path=str(onnx_path),
                    dry_run=True,
                    plugin_libraries=[str(plugin_path)],
                    validate_plugin_libraries_loadable=False,
                ),
            )
        ],
        output_diff_config=OutputDiffConfig(),
    )

    ExportPass().run(context)

    artifact = context.metrics["export"]["artifacts"][0]
    target = context.metrics["export"]["targets"][0]
    layers = artifact["tensorrt_runtime_layers"]["layers"]
    assert artifact["plugin_validation"]["status"] == "present"
    assert layers["plugin_presence"]["status"] == "present"
    assert layers["plugin_loadability"]["status"] == "not_requested"
    assert target["tensorrt_runtime_layers"]["layers"]["plugin_presence"][
        "status"
    ] == "present"
