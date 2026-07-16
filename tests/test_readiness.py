from __future__ import annotations

import json
from pathlib import Path

import pytest
from xqt import XQTOptimizationSession, XQTReadinessReport, assess_xqt_readiness
from xqt.core.artifact import ArtifactManifest
from xqt.readiness import XQTReadinessScenario
from xqt.operator_opt.backends.tilelang_validation import TileLangFP4ValidationResult
from xqt.quant.capability import describe_quant_backend_capability


def _scenarios(report: XQTReadinessReport) -> dict[str, XQTReadinessScenario]:
    return {scenario.name: scenario for scenario in report.scenarios}


def test_fp4_weight_only_capability_is_pseudo_quantization() -> None:
    capability = describe_quant_backend_capability(
        "pytorch",
        method="awq",
        strategy="w4a16_fp4",
        compute="dequant_fp16",
        policy={"dtype": "fp4", "scheme": "weight_only"},
    )

    assert capability.nature.value == "pseudo"
    assert any("PSEUDO quantization" in note for note in capability.notes)


def test_assess_xqt_readiness_default_report_is_structured() -> None:
    report = assess_xqt_readiness()
    scenarios = _scenarios(report)
    payload = report.to_dict()

    assert report.overall_status == "partial"
    assert report.status_counts == {"partial": 2, "ready": 1}
    assert payload["status_counts"] == {"partial": 2, "ready": 1}
    assert payload["required_action_count"] == report.required_action_count
    assert set(scenarios) == {
        "fp4_tilelang_megakernel",
        "tensorrt_so_plugin",
        "prune_quant_error_analysis",
    }
    fp4 = scenarios["fp4_tilelang_megakernel"]
    assert fp4.status == "partial"
    assert fp4.checks["quantization"]["nature"] == "pseudo"
    assert fp4.checks["tilelang_probe"]["status"] == "not_requested"
    assert fp4.distance_to_ready["ready"] is False
    assert fp4.distance_to_ready["required_action_count"] == len(fp4.required_actions)
    assert any("run assess_xqt_readiness" in action for action in fp4.required_actions)
    trt = scenarios["tensorrt_so_plugin"]
    assert trt.status == "partial"
    assert trt.checks["plugin_status"] == "not_provided"
    assert any("provide target TensorRT plugin" in action for action in trt.required_actions)
    analysis = scenarios["prune_quant_error_analysis"]
    assert analysis.status == "ready"
    assert analysis.checks["layer_statistics"] == "supported"
    assert analysis.required_actions == []
    assert analysis.to_dict()["distance_to_ready"]["ready"] is True


def test_xqt_readiness_report_writes_json_and_markdown(tmp_path: Path) -> None:
    report = assess_xqt_readiness()

    paths = report.write_artifacts(tmp_path)

    assert set(paths) == {"json", "markdown"}
    assert paths["json"].is_file()
    assert paths["markdown"].is_file()
    payload = json.loads(paths["json"].read_text(encoding="utf-8"))
    markdown = paths["markdown"].read_text(encoding="utf-8")
    assert payload["overall_status"] == "partial"
    assert payload["required_action_count"] == report.required_action_count
    assert "XQT Readiness Report" in markdown
    assert "fp4_tilelang_megakernel" in markdown
    assert "Required actions:" in markdown


def test_xqt_readiness_report_can_attach_to_manifest(tmp_path: Path) -> None:
    report = assess_xqt_readiness()
    paths = report.write_artifacts(tmp_path)
    manifest = ArtifactManifest(project_name="readiness_test")

    returned = report.add_to_manifest(manifest, artifact_paths=paths)

    assert returned is manifest
    payload = manifest.to_dict()
    metric_names = {item["name"] for item in payload["metrics"]}
    artifact_formats = {item["format"] for item in payload["artifacts"]}
    assert "readiness.overall_status" in metric_names
    assert "readiness.fp4_tilelang_megakernel.status" in metric_names
    assert "readiness.tensorrt_so_plugin.status" in metric_names
    assert "readiness.prune_quant_error_analysis.status" in metric_names
    assert artifact_formats == {"json", "markdown"}
    assert all(item["checksum"] for item in payload["artifacts"])


def test_xqt_optimization_session_can_write_readiness_artifacts(tmp_path: Path) -> None:
    session = XQTOptimizationSession(
        project={
            "name": "readiness_session",
            "artifact_dir": str(tmp_path / "artifacts"),
        }
    )

    report = session.readiness(name="readiness_check")

    assert report.overall_status == "partial"
    assert "readiness_check" in session.context.metrics
    assert session.context.artifacts["readiness_check_json"].is_file()
    assert session.context.artifacts["readiness_check_markdown"].is_file()
    assert session.context.artifacts["manifest"].is_file()
    manifest_text = session.context.artifacts["manifest"].read_text(encoding="utf-8")
    assert "readiness.overall_status" in manifest_text


def test_assess_xqt_readiness_reports_missing_tensorrt_plugin_as_blocked(
    tmp_path: Path,
) -> None:
    missing_plugin = tmp_path / "missing_plugin.so"

    report = assess_xqt_readiness(tensorrt_plugin_libraries=[missing_plugin])
    trt = _scenarios(report)["tensorrt_so_plugin"]

    assert report.overall_status == "blocked"
    assert trt.status == "blocked"
    assert trt.checks["plugin_status"] == "missing"
    assert trt.checks["plugin_validation"]["status"] == "missing"
    assert trt.checks["plugin_libraries"][0]["exists"] is False
    assert any("missing TensorRT plugin" in action for action in trt.required_actions)


def test_assess_xqt_readiness_reports_present_tensorrt_plugin(tmp_path: Path) -> None:
    plugin_path = tmp_path / "libcustom_plugin.so"
    plugin_path.write_bytes(b"")

    report = assess_xqt_readiness(tensorrt_plugin_libraries=[plugin_path])
    trt = _scenarios(report)["tensorrt_so_plugin"]

    assert trt.status == "partial"
    assert trt.checks["plugin_status"] == "present"
    assert trt.checks["plugin_validation"]["status"] == "present"
    assert trt.checks["plugin_libraries"][0]["exists"] is True
    assert any("validate_tensorrt_plugin_loadability=True" in action for action in trt.required_actions)


def test_assess_xqt_readiness_can_include_tilelang_compile_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_result = TileLangFP4ValidationResult(
        status="ok",
        reason=None,
        device=None,
        dtype="torch.float16",
        shape={
            "m": 64,
            "in_features": 32,
            "out_features": 64,
            "group_size": 16,
            "block_m": 64,
            "block_n": 64,
        },
        activation="silu",
        has_bias=True,
        target_arch="sm_80",
        max_abs_error=None,
        mean_abs_error=None,
        rtol=1e-2,
        atol=1e-2,
        allclose=None,
        compile_only=True,
        compile_status="ok",
        latency_ms_tilelang=None,
        latency_ms_reference=None,
        speedup=None,
    )

    monkeypatch.setattr(
        "xqt.readiness.validate_tilelang_packed_fp4_fused_gemm",
        lambda **_: fake_result,
    )

    report = assess_xqt_readiness(run_tilelang_probe=True)
    fp4 = _scenarios(report)["fp4_tilelang_megakernel"]

    assert fp4.status == "partial"
    assert fp4.checks["tilelang_probe"]["status"] == "ok"
    assert fp4.checks["tilelang_probe"]["compile_only"] is True
    assert any("compile-only probe passed" in item for item in fp4.evidence)
    assert any("tilelang_compile_only=False" in action for action in fp4.required_actions)
