from __future__ import annotations

from pathlib import Path

import pytest
from xqt.core.errors import XQTBackendError
from xqt.core.schema import (
    ExportTargetConfig,
    OpenVINOBenchmarkConfig,
    OpenVINOExportConfig,
    QNNExportConfig,
)
from xqt.export.capability import deployment_capability_matrix
from xqt.export.mobile import (
    build_qnn_onnx_converter_command,
    export_qnn_from_onnx,
    mobile_export_diagnosis,
)
from xqt.export.openvino import OpenVINOExportResult
from xqt.export.reporting import openvino_runtime_layer_report
from xqt.pipeline.preflight_checks.export import _check_export_targets
from xqt.pipeline.preflight_checks._base import PreflightReport
from xqt.workflows.stage_specs import _validate_export_targets
from xqt.workflows.session_targets import build_session_export_targets


def test_qnn_capability_is_adapter_not_runtime() -> None:
    capabilities = {
        item.format: item for item in deployment_capability_matrix()
    }
    qnn = capabilities["qnn"]

    assert qnn.status == "adapter"
    assert qnn.maturity == "reference_guarded"
    assert qnn.quantization
    assert not qnn.dynamic_shapes
    assert "official QNN SDK" in qnn.notes


def test_build_qnn_onnx_converter_command() -> None:
    command = build_qnn_onnx_converter_command(
        "/tmp/model.onnx",
        "/tmp/qnn_out",
        converter_path="qnn-onnx-converter",
        extra_args=["--batch", "1"],
    )

    assert command == [
        "qnn-onnx-converter",
        "--input_model",
        "/tmp/model.onnx",
        "--output_dir",
        "/tmp/qnn_out",
        "--batch",
        "1",
    ]


def test_export_qnn_from_onnx_dry_run(tmp_path: Path) -> None:
    onnx = tmp_path / "model.onnx"
    onnx.write_bytes(b"fake onnx")
    output_dir = tmp_path / "qnn_out"

    result = export_qnn_from_onnx(onnx, output_dir, dry_run=True)

    assert result.dry_run
    assert result.output_paths[0].is_dir()
    assert result.command[0] == "qnn-onnx-converter"
    assert result.checksums == {}


def test_export_qnn_from_onnx_missing_converter(tmp_path: Path) -> None:
    onnx = tmp_path / "model.onnx"
    onnx.write_bytes(b"fake onnx")

    with pytest.raises(XQTBackendError, match="executable not found"):
        export_qnn_from_onnx(
            onnx,
            tmp_path / "qnn_out",
            converter_path="definitely-not-a-qnn-converter",
        )


def test_mobile_export_diagnosis_reasons() -> None:
    dry_run = mobile_export_diagnosis(target_format="qnn", dry_run=True)
    source_missing = mobile_export_diagnosis(
        target_format="mnn",
        dry_run=False,
        source_missing=True,
    )
    converter_missing = mobile_export_diagnosis(
        target_format="ncnn",
        dry_run=False,
        converter_missing=True,
    )
    materialized = mobile_export_diagnosis(
        target_format="executorch",
        dry_run=False,
        materialized=True,
    )

    assert dry_run["reason"] == "dry_run_preflight_only"
    assert source_missing["reason"] == "source_artifact_missing"
    assert converter_missing["reason"] == "converter_executable_missing"
    assert materialized["status"] == "materialized"
    assert materialized["blocked_by"] is None


def test_qnn_target_config_loads_and_validates() -> None:
    target = ExportTargetConfig(
        format="qnn",
        output_path="artifacts/qnn",
        qnn=QNNExportConfig(converter_path="qnn-onnx-converter", dry_run=True),
    )
    _validate_export_targets([target], location="stages.0.params.targets")


def test_qnn_preflight_reports_missing_converter(tmp_path: Path) -> None:
    from xqt.core.schema import ExportTargetConfig

    target = ExportTargetConfig(
        format="qnn",
        output_path=str(tmp_path / "qnn"),
        qnn=QNNExportConfig(converter_path="definitely-not-a-qnn-converter"),
    )
    report = PreflightReport()
    _check_export_targets(report, [target], prefix="export.targets")

    entry = report.checks[0]
    assert entry.passed is False
    assert "qnn_onnx_converter" in entry.name


def test_session_export_accepts_qnn_typed_target(tmp_path: Path) -> None:
    targets = build_session_export_targets(
        "export",
        format="qnn",
        targets=None,
        target_params=None,
        opset=None,
        output_path=str(tmp_path / "qnn_out"),
        qnn={"converter_path": "qnn-onnx-converter", "dry_run": True},
    )

    assert len(targets) == 1
    target = targets[0]
    assert target["format"] == "qnn"
    assert target["qnn"]["dry_run"] is True


def test_openvino_benchmark_layer_becomes_configured() -> None:
    target = ExportTargetConfig(
        format="openvino",
        openvino=OpenVINOExportConfig(
            benchmark=OpenVINOBenchmarkConfig(enabled=True, warmup=3, iterations=5)
        ),
    )
    result = OpenVINOExportResult(
        xml_path=Path("/tmp/model.xml"),
        bin_path=None,
        checksum=None,
        dry_run=False,
        metadata={},
    )

    report = openvino_runtime_layer_report(target=target, export_result=result, output_diff=None)
    benchmark = report["layers"]["runtime_benchmark"]

    assert benchmark["requested"] is True
    assert benchmark["status"] in {
        "configured",
        "not_run_missing_dependency",
        "not_run_missing_ir",
    }
    assert benchmark["benchmark"]["warmup"] == 3
    assert benchmark["benchmark"]["iterations"] == 5


def test_openvino_benchmark_layer_stays_not_configured_by_default() -> None:
    target = ExportTargetConfig(format="openvino", openvino=OpenVINOExportConfig())
    result = OpenVINOExportResult(
        xml_path=Path("/tmp/model.xml"),
        bin_path=None,
        checksum=None,
        dry_run=True,
        metadata={},
    )

    report = openvino_runtime_layer_report(target=target, export_result=result, output_diff=None)
    benchmark = report["layers"]["runtime_benchmark"]

    assert benchmark["status"] == "not_configured"
    assert benchmark["requested"] is False


def test_openvino_benchmark_dry_run_reports_not_run() -> None:
    target = ExportTargetConfig(
        format="openvino",
        openvino=OpenVINOExportConfig(
            dry_run=True,
            benchmark=OpenVINOBenchmarkConfig(enabled=True),
        ),
    )
    result = OpenVINOExportResult(
        xml_path=Path("/tmp/model.xml"),
        bin_path=None,
        checksum=None,
        dry_run=True,
        metadata={},
    )

    report = openvino_runtime_layer_report(target=target, export_result=result, output_diff=None)
    benchmark = report["layers"]["runtime_benchmark"]

    assert benchmark["status"] == "not_run_dry_run"
