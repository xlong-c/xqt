from __future__ import annotations

from xqt.core.reporting import build_stage_report, reporting_schema_payload
from xqt.export.capability import deployment_capability_matrix
from xqt.operator_opt.capability import describe_operator_backend_capability
from xqt.prune.capability import describe_prune_runtime_capability
from xqt.quant.capability import describe_quant_backend_capability
from xqt.readiness import assess_xqt_readiness
from xqt.run_workflow import DEFAULT_CONFIG
from xqt.workflows import optimize_model


def test_optimization_capability_projection_uses_shared_fields() -> None:
    quant = describe_quant_backend_capability(
        "onnxruntime_qdq",
        method="static_qdq_int8",
        strategy="static_qdq_int8",
    ).to_dict()["optimization_capability"]
    operator = describe_operator_backend_capability("cute_dsl").to_dict()[
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
            "backend",
            "status",
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
        capability["backend"] == "onnxruntime_qdq"
        and capability["requires_calibration"] is True
        for capability in matrix["quantization"]
    )
    assert any(
        capability["name"] == "structured"
        and capability["metadata"]["speedup_verified"] is False
        for capability in matrix["pruning"]
    )
    assert any(capability["name"] == "openvino" for capability in matrix["export"])
    assert "ttft_ms" in schemas["benchmark"]["llm_workload"]
    assert "failed_tensor_count" in schemas["numeric_diff"]["numeric_diff"]
    assert "Capability matrix" in report.to_markdown()


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
    assert "backend" in stage_report
    assert "target_module" in stage_report
    assert "p50_ms" in stage_report["benchmark"]
    assert "max_abs" in stage_report["numeric_diff"]
    assert "prune:prune_l1" in payload["passes"]
    assert "stage.prune_l1.status" in metric_names
    assert "stage.prune_l1.accepted" in metric_names


def test_reporting_schema_payload_reserves_runtime_and_llm_fields() -> None:
    schemas = reporting_schema_payload()

    assert "prefix_cache" in schemas["runtime_features"]["runtime_features"]
    assert "paged_kv" in schemas["runtime_features"]["runtime_features"]
    assert "tokens_per_s" in schemas["benchmark"]["llm_workload"]


def test_stage_report_extracts_backend_target_benchmark_and_diff() -> None:
    report = build_stage_report(
        stage_name="quant_encoder",
        stage_kind="quant",
        accepted=True,
        message="ok",
        metrics={
            "backend": "torchao",
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

    assert report["backend"] == "torchao"
    assert report["target_module"] == "encoder.layers.0"
    assert report["benchmark"]["p50_ms"] == 2.0
    assert report["numeric_diff"]["allclose"] is True
    assert report["numeric_diff"]["max_abs"] == 0.01
