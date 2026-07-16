from __future__ import annotations

from xqt.core.reporting import build_stage_report, reporting_schema_payload
from xqt.export.capability import deployment_capability_matrix
from xqt.operator_opt.capability import describe_operator_engine_capability
from xqt.prune.capability import describe_prune_runtime_capability
from xqt.quant.capability import describe_quant_backend_capability
from xqt.readiness import assess_xqt_readiness
from xqt.run_workflow import DEFAULT_CONFIG
from xqt.workflows import optimize_model


def test_optimization_capability_projection_uses_shared_fields() -> None:
    quant = describe_quant_backend_capability(
        "onnxruntime_qdq",
        method="none",
        strategy="w8a8_int8",
        compute="qdq_static",
    ).to_dict()["optimization_capability"]
    operator = describe_operator_engine_capability("cute_dsl").to_dict()[
        "optimization_capability"
    ]
    prune = describe_prune_runtime_capability(
        method="nm_structured",
        device="cuda",
        pattern=(2, 4),
    ).to_dict()["optimization_capability"]
    export = next(
        capability
        for capability in deployment_capability_matrix()
        if capability.format == "onnx"
    ).to_dict()["optimization_capability"]

    for capability in (quant, operator, prune, export):
        assert {
            "kind",
            "name",
            "engine",
            "status",
            "maturity",
            "runtime",
            "artifact_kind",
            "requires_cuda",
            "requires_calibration",
            "requires_exportable_graph",
            "limitations",
        }.issubset(capability)

    assert quant["requires_calibration"] is True
    assert quant["requires_exportable_graph"] is True
    assert operator["status"] == "planned"
    assert quant["maturity"] == "executable"
    assert operator["maturity"] == "reference_guarded"
    assert prune["maturity"] == "reference_guarded"
    assert export["maturity"] == "executable"
    assert operator["status"] not in {"optimized", "applied"}
    assert prune["metadata"]["pattern"] == [2, 4]
    assert export["status"] == "available"


def test_readiness_report_includes_inference_capability_matrix() -> None:
    report = assess_xqt_readiness()
    payload = report.to_dict()
    matrix = payload["capability_matrix"]
    schemas = payload["reporting_schemas"]

    assert {"quantization", "pruning", "operator", "export", "runtime_features"} <= set(matrix)
    assert any(
        capability["engine"] == "onnxruntime_qdq"
        and capability["requires_calibration"] is True
        and capability["maturity"] == "executable"
        for capability in matrix["quantization"]
    )
    assert any(
        capability["name"] == "structured"
        and capability["metadata"]["speedup_verified"] is False
        and capability["maturity"] == "executable"
        for capability in matrix["pruning"]
    )
    assert any(
        capability["name"] == "openvino"
        and capability["maturity"] == "reference_guarded"
        for capability in matrix["export"]
    )
    assert any(
        capability["engine"] == "cute_dsl"
        and capability["maturity"] == "reference_guarded"
        for capability in matrix["operator"]
    )
    assert "ttft_ms" in schemas["benchmark"]["llm_workload"]
    assert "failed_tensor_count" in schemas["numeric_diff"]["numeric_diff"]
    assert "Capability matrix" in report.to_markdown()


def test_capability_matrix_exposes_all_shared_maturity_levels() -> None:
    report = assess_xqt_readiness()
    matrix = report.to_dict()["capability_matrix"]
    maturities = {
        capability["maturity"]
        for capabilities in matrix.values()
        for capability in capabilities
    }

    assert {"executable", "reference_guarded", "metadata_only", "planned"} <= maturities


def test_stage_report_attaches_workflow_stage_to_manifest() -> None:
    result = optimize_model(DEFAULT_CONFIG, write_outputs=False)

    stage_report = result.metrics["stage_reports"]["prune_l1"]
    manifest = result.context.manifest
    assert manifest is not None
    payload = manifest.to_dict()
    metric_names = {item["name"] for item in payload["metrics"]}

    assert stage_report["stage_name"] == "prune_l1"
    assert stage_report["stage_kind"] == "prune"
    assert stage_report["status"] == "accepted"
    assert "engine" in stage_report
    assert "target_module" in stage_report
    assert "p50_ms" in stage_report["benchmark"]
    assert "max_abs" in stage_report["numeric_diff"]
    assert set(stage_report["execution"]) == {
        "backend",
        "engine",
        "device",
        "shape",
        "warmup",
        "iterations",
        "fallback",
        "artifact_kinds",
    }
    assert "prune:prune_l1" in payload["passes"]
    assert "stage.prune_l1.status" in metric_names
    assert "stage.prune_l1.accepted" in metric_names


def test_reporting_schema_payload_reserves_runtime_and_llm_fields() -> None:
    schemas = reporting_schema_payload()

    assert "prefix_cache" in schemas["runtime_features"]["runtime_features"]
    assert "paged_kv" in schemas["runtime_features"]["runtime_features"]
    assert "tokens_per_s" in schemas["benchmark"]["llm_workload"]


def test_stage_report_extracts_engine_target_benchmark_and_diff() -> None:
    report = build_stage_report(
        stage_name="quant_encoder",
        stage_kind="quant",
        accepted=True,
        message="ok",
        metrics={
            "engine": "torchao",
            "components": [
                {
                    "target_path": "encoder.layers.0",
                    "output_diff": {
                        "allclose": True,
                        "max_abs": 0.01,
                        "mean_abs": 0.001,
                    },
                }
            ],
            "latency": {
                "p50_ms": 2.0,
                "p90_ms": 2.5,
            },
        },
        artifacts={"quant_model": "artifacts/quant.pt"},
    ).to_dict()

    assert report["engine"] == "torchao"
    assert report["target_module"] == "encoder.layers.0"
    assert report["benchmark"]["p50_ms"] == 2.0
    assert report["numeric_diff"]["allclose"] is True
    assert report["numeric_diff"]["max_abs"] == 0.01
    assert report["execution"] == {
        "backend": "torchao",
        "engine": "torchao",
        "device": None,
        "shape": None,
        "warmup": None,
        "iterations": None,
        "fallback": None,
        "artifact_kinds": ["quant_model"],
    }


def test_stage_report_serializes_tensorrt_runtime_session_without_live_handles() -> None:
    from pathlib import Path

    import tensorrt as trt

    from xqt.export.tensorrt import TensorRTRuntimeSession

    session = TensorRTRuntimeSession(
        engine_path=Path("artifacts/model.engine"),
        device="cuda:0",
        trt=trt,
        runtime=object(),
        engine=object(),
        context=object(),
        engine_inspector={"layer_count": 2},
    )
    report = build_stage_report(
        stage_name="deploy_tensorrt",
        stage_kind="deploy",
        accepted=True,
        message="ok",
        metrics={
            "runtime_handle": {
                "runtime": "tensorrt",
                "handle_kind": "runtime_session",
                "handle": session,
                "metadata": {
                    "runtime_validation": {
                        "status": "session_created",
                        "engine_deserialized": True,
                        "execution_context_created": True,
                    }
                },
            }
        },
        artifacts={"tensorrt_engine": "artifacts/model.engine"},
    ).to_dict()

    handle = report["metrics"]["runtime_handle"]["handle"]
    assert handle == {
        "engine_path": "artifacts/model.engine",
        "device": "cuda:0",
        "handle_materialized": True,
        "engine_inspector": {"layer_count": 2},
    }
    assert report["metrics"]["runtime_handle"]["metadata"]["runtime_validation"] == {
        "status": "session_created",
        "engine_deserialized": True,
        "execution_context_created": True,
    }
