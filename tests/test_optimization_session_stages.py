from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from xqt.workflows import XQTOptimizationSession
from xqt.workflows.stage import (
    ExportBundlePayload,
    QuantizedModelPayload,
    RuntimeHandlePayload,
    payload_capabilities_for_kind,
    payload_can_restore_model,
    payload_kind_for_stage,
    RuntimeArtifactPayload,
    RuntimePlanPayload,
    stage_kind_for_transform,
    transform_family_for_kind,
)


class _TinyLinear(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = torch.nn.Linear(16, 16)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.fc(inputs)


def test_session_initializes_formal_baseline_stage(tmp_path: Path) -> None:
    session = XQTOptimizationSession(
        project={
            "name": "session_stage_baseline",
            "artifact_dir": str(tmp_path / "artifacts"),
        },
        model=_TinyLinear().eval(),
        example_inputs=torch.randn(4, 16),
    )

    assert session.baseline_stage == "baseline"
    assert session.best_stage == "baseline"
    assert [stage.name for stage in session.session_stages] == ["baseline"]
    baseline = session.session_stages[0]
    assert baseline.stage_kind == "baseline"
    assert baseline.payload.payload_kind == "torch_module"
    assert baseline.created_by.kind == "session_init"


def test_quant_and_operator_create_stage_lineage(tmp_path: Path) -> None:
    session = XQTOptimizationSession(
        project={
            "name": "session_stage_lineage",
            "artifact_dir": str(tmp_path / "artifacts"),
        },
        model=_TinyLinear().eval(),
        example_inputs=torch.randn(4, 16),
    )

    quant_stage = session.quant(
        name="fp4_quant",
        backend="pytorch",
        method="awq",
        strategy="fp4_weight_only",
        policy={
            "dtype": "fp4",
            "scheme": "weight_only",
            "include_module_names": ["fc"],
            "group_size": 16,
        },
    )
    operator_stage = session.operator(
        name="compile_try",
        from_stage="fp4_quant",
        targets=[
            {
                "name": "fc_compile",
                "target": "fc",
                "engine": "torch_compile",
                "fallback": "eager",
                "min_speedup": 1.01,
            }
        ],
    )

    assert quant_stage.accepted is True
    assert operator_stage.accepted is True
    names = [stage.name for stage in session.session_stages]
    assert names == ["baseline", "fp4_quant", "compile_try"]

    quantized = session.session_stages[1]
    optimized = session.session_stages[2]
    assert quantized.stage_kind == "quantized"
    assert optimized.stage_kind == "optimized"
    assert quantized.payload.payload_kind == "quantized_model"
    assert isinstance(quantized.payload.value, QuantizedModelPayload)
    assert isinstance(quantized.payload.value.model, _TinyLinear)
    assert quantized.payload.value.backend == "pytorch"
    assert quantized.payload.value.method == "awq"
    assert quantized.payload.value.strategy == "fp4_weight_only"
    assert quantized.payload.metadata["quantized_model"]["artifact_kind"] == "quantized_model"
    assert optimized.payload.payload_kind == "runtime_plan"
    assert isinstance(optimized.payload.value, RuntimePlanPayload)
    assert isinstance(optimized.payload.value, RuntimeArtifactPayload)
    assert optimized.payload.value.artifact_kind == "runtime_plan"
    assert optimized.payload.value.engine == "torch_compile"
    assert optimized.payload.value.source_model_stage == "fp4_quant"
    assert optimized.payload.metadata["model_stage_name"] == "compile_try"
    assert optimized.payload.metadata["source_model_stage"] == "fp4_quant"
    assert optimized.payload.metadata["runtime_plan"]["engine"] == "torch_compile"
    assert optimized.payload.metadata["runtime_plan"]["artifact_kind"] == "runtime_plan"
    assert quantized.created_by.transform == "quant"
    assert optimized.created_by.transform == "operator"
    assert optimized.created_by.from_stage == "fp4_quant"
    assert optimized.parent_stage_ids == [quantized.stage_id]

    session.use("baseline")
    assert isinstance(session.model, _TinyLinear)
    session.use("compile_try")
    assert isinstance(session.model, _TinyLinear)


def test_quant_stage_without_explicit_from_stage_uses_baseline_as_parent(tmp_path: Path) -> None:
    session = XQTOptimizationSession(
        project={
            "name": "session_stage_default_parent",
            "artifact_dir": str(tmp_path / "artifacts"),
        },
        model=_TinyLinear().eval(),
        example_inputs=torch.randn(4, 16),
    )

    session.quant(
        name="fp4_quant",
        backend="pytorch",
        method="awq",
        strategy="fp4_weight_only",
        policy={
            "dtype": "fp4",
            "scheme": "weight_only",
            "include_module_names": ["fc"],
            "group_size": 16,
        },
    )

    baseline = session.session_stages[0]
    quantized = session.session_stages[1]
    assert quantized.parent_stage_ids == [baseline.stage_id]


def test_workflow_result_writes_session_stages(tmp_path: Path) -> None:
    session = XQTOptimizationSession(
        project={
            "name": "session_stage_result",
            "artifact_dir": str(tmp_path / "artifacts"),
        },
        model=_TinyLinear().eval(),
        example_inputs=torch.randn(4, 16),
    )

    session.quant(
        name="fp4_quant",
        backend="pytorch",
        method="awq",
        strategy="fp4_weight_only",
        policy={
            "dtype": "fp4",
            "scheme": "weight_only",
            "include_module_names": ["fc"],
            "group_size": 16,
        },
    )
    result = session.write_outputs()

    assert len(result.session_stages) == 2
    workflow_path = result.context.artifacts["workflow_result"]
    payload = json.loads(workflow_path.read_text(encoding="utf-8"))
    assert [item["name"] for item in payload["session_stages"]] == ["baseline", "fp4_quant"]
    assert payload["session_stages"][1]["stage_kind"] == "quantized"
    assert payload["session_stages"][1]["payload"]["payload_kind"] == "quantized_model"
    assert payload["session_stages"][1]["payload"]["value"]["artifact_kind"] == "quantized_model"


def test_runtime_plan_payload_serializes_as_plain_mapping(tmp_path: Path) -> None:
    session = XQTOptimizationSession(
        project={
            "name": "session_stage_runtime_plan_serialization",
            "artifact_dir": str(tmp_path / "artifacts"),
        },
        model=_TinyLinear().eval(),
        example_inputs=torch.randn(4, 16),
    )

    session.quant(
        name="fp4_quant",
        backend="pytorch",
        method="awq",
        strategy="fp4_weight_only",
        policy={
            "dtype": "fp4",
            "scheme": "weight_only",
            "include_module_names": ["fc"],
            "group_size": 16,
        },
    )
    session.operator(
        name="compile_try",
        from_stage="fp4_quant",
        targets=[
            {
                "name": "fc_compile",
                "target": "fc",
                "engine": "torch_compile",
                "fallback": "eager",
            }
        ],
    )

    result = session.write_outputs()
    workflow_path = result.context.artifacts["workflow_result"]
    payload = json.loads(workflow_path.read_text(encoding="utf-8"))
    stage_payload = payload["session_stages"][-1]["payload"]
    assert stage_payload["payload_kind"] == "runtime_plan"
    assert stage_payload["value"]["engine"] == "torch_compile"
    assert stage_payload["value"]["source_model_stage"] == "fp4_quant"


def test_export_bundle_payload_serializes_as_plain_mapping(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = XQTOptimizationSession(
        project={
            "name": "session_stage_export_bundle_serialization",
            "artifact_dir": str(tmp_path / "artifacts"),
        },
        model=_TinyLinear().eval(),
        example_inputs=torch.randn(4, 16),
    )

    def _fake_run_export(config, stage, context) -> None:
        output = tmp_path / "model.onnx"
        output.write_bytes(b"fake-onnx")
        context.artifacts["export_onnx"] = output
        context.metrics["export"] = {
            "target_count": 1,
            "targets": [{"format": "onnx", "output_path": str(output)}],
        }

    monkeypatch.setattr("xqt.workflows.optimization._run_export", _fake_run_export)

    session.export(
        name="onnx_export",
        format="onnx",
        output_path=tmp_path / "model.onnx",
        opset=17,
    )

    result = session.write_outputs()
    workflow_path = result.context.artifacts["workflow_result"]
    payload = json.loads(workflow_path.read_text(encoding="utf-8"))
    stage_payload = payload["session_stages"][-1]["payload"]
    assert stage_payload["payload_kind"] == "export_bundle"
    assert stage_payload["value"]["artifact_kind"] == "export_bundle"
    assert stage_payload["value"]["format"] == "onnx"
    assert stage_payload["value"]["source_model_stage"] == "baseline"


def test_export_stage_uses_export_bundle_payload_kind(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = XQTOptimizationSession(
        project={
            "name": "session_stage_export",
            "artifact_dir": str(tmp_path / "artifacts"),
        },
        model=_TinyLinear().eval(),
        example_inputs=torch.randn(4, 16),
    )

    def _fake_run_export(config, stage, context) -> None:
        output = tmp_path / "model.onnx"
        output.write_bytes(b"fake-onnx")
        context.artifacts["export_onnx"] = output
        context.metrics["export"] = {
            "target_count": 1,
            "targets": [{"format": "onnx", "output_path": str(output)}],
        }

    monkeypatch.setattr("xqt.workflows.optimization._run_export", _fake_run_export)

    export_stage = session.export(
        name="onnx_export",
        format="onnx",
        output_path=tmp_path / "model.onnx",
        opset=17,
    )

    assert export_stage.accepted is True
    exported = session.session_stages[-1]
    assert exported.name == "onnx_export"
    assert exported.stage_kind == "exported"
    assert exported.payload.payload_kind == "export_bundle"
    assert isinstance(exported.payload.value, ExportBundlePayload)
    assert isinstance(exported.payload.value, RuntimeArtifactPayload)
    assert exported.payload.value.artifact_kind == "export_bundle"
    assert exported.payload.value.format == "onnx"
    assert exported.payload.value.source_model_stage == "baseline"
    assert exported.payload.metadata["export_bundle"]["artifact_kind"] == "export_bundle"
    assert exported.created_by.transform_family == "export"


def test_stage_created_by_tracks_transform_metadata(tmp_path: Path) -> None:
    session = XQTOptimizationSession(
        project={
            "name": "session_stage_transform_metadata",
            "artifact_dir": str(tmp_path / "artifacts"),
        },
        model=_TinyLinear().eval(),
        example_inputs=torch.randn(4, 16),
    )

    session.quant(
        name="fp4_quant",
        backend="pytorch",
        method="awq",
        strategy="fp4_weight_only",
        policy={
            "dtype": "fp4",
            "scheme": "weight_only",
            "include_module_names": ["fc"],
            "group_size": 16,
        },
    )

    quantized = session.session_stages[-1]
    assert quantized.created_by.transform == "quant"
    assert quantized.created_by.transform_family == "model_quantizer"
    assert quantized.created_by.transform_name == "quant"
    assert quantized.created_by.params["backend"] == "pytorch"
    assert quantized.created_by.to_dict()["transform_family"] == "model_quantizer"


def test_stage_protocol_helpers_define_stable_contracts() -> None:
    assert stage_kind_for_transform("quant") == "quantized"
    assert stage_kind_for_transform("operator") == "optimized"
    assert stage_kind_for_transform("unknown_transform") == "custom"

    assert transform_family_for_kind("quant") == "model_quantizer"
    assert transform_family_for_kind("operator") == "operator_optimizer"
    assert transform_family_for_kind("unknown_transform") == "transform"

    assert payload_kind_for_stage("optimized", transform_kind="operator") == "runtime_plan"
    assert payload_kind_for_stage("exported", transform_kind="export") == "export_bundle"
    assert payload_kind_for_stage("quantized", transform_kind="quant") == "quantized_model"

    runtime_capabilities = payload_capabilities_for_kind("runtime_plan")
    assert runtime_capabilities["can_restore_model"] is False
    assert payload_can_restore_model("torch_module") is True
    assert payload_can_restore_model("quantized_model") is True
    assert payload_can_restore_model("runtime_plan") is False

    runtime_capabilities["can_export"] = False
    fresh_runtime_capabilities = payload_capabilities_for_kind("runtime_plan")
    assert fresh_runtime_capabilities["can_export"] is True


def test_stage_payload_dispatch_uses_specialized_and_default_builders(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = XQTOptimizationSession(
        project={
            "name": "session_stage_payload_dispatch",
            "artifact_dir": str(tmp_path / "artifacts"),
        },
        model=_TinyLinear().eval(),
        example_inputs=torch.randn(4, 16),
    )

    def _fake_run_export(config, stage, context) -> None:
        output = tmp_path / "model.onnx"
        output.write_bytes(b"fake-onnx")
        context.artifacts["export_onnx"] = output
        context.metrics["export"] = {
            "target_count": 1,
            "targets": [{"format": "onnx", "output_path": str(output)}],
        }

    monkeypatch.setattr("xqt.workflows.optimization._run_export", _fake_run_export)

    session.quant(
        name="fp4_quant",
        backend="pytorch",
        method="awq",
        strategy="fp4_weight_only",
        policy={
            "dtype": "fp4",
            "scheme": "weight_only",
            "include_module_names": ["fc"],
            "group_size": 16,
        },
    )
    session.operator(
        name="compile_try",
        from_stage="fp4_quant",
        targets=[
            {
                "name": "fc_compile",
                "target": "fc",
                "engine": "torch_compile",
                "fallback": "eager",
            }
        ],
    )
    session.export(
        name="onnx_export",
        format="onnx",
        output_path=tmp_path / "model.onnx",
        opset=17,
    )

    quantized = session.session_stages[1]
    optimized = session.session_stages[2]
    exported = session.session_stages[3]
    assert quantized.stage_kind == "quantized"
    assert optimized.stage_kind == "optimized"
    assert exported.stage_kind == "exported"
    assert isinstance(quantized.payload.value, QuantizedModelPayload)
    assert isinstance(quantized.payload.value.model, _TinyLinear)
    assert isinstance(optimized.payload.value, RuntimePlanPayload)
    assert isinstance(exported.payload.value, ExportBundlePayload)


def test_runtime_handle_payload_serializes_as_plain_mapping() -> None:
    payload = RuntimeHandlePayload(
        stage_name="runtime_session",
        source_model_stage="onnx_export",
        runtime="onnxruntime",
        handle_kind="inference_session",
        target_count=1,
        targets=[{"name": "onnx", "format": "onnx"}],
        handle=object(),
        artifacts={"model": "model.onnx"},
        metadata={"provider": "CPUExecutionProvider"},
    )

    serialized = payload.to_dict()
    assert serialized["artifact_kind"] == "runtime_handle"
    assert serialized["runtime"] == "onnxruntime"
    assert serialized["handle_kind"] == "inference_session"
    assert serialized["handle_materialized"] is True
    assert serialized["metadata"]["provider"] == "CPUExecutionProvider"


def test_session_stage_compare_helper_is_payload_capability_and_artifact_aware(
    tmp_path: Path,
) -> None:
    session = XQTOptimizationSession(
        project={
            "name": "session_stage_compare",
            "artifact_dir": str(tmp_path / "artifacts"),
        },
        model=_TinyLinear().eval(),
        example_inputs=torch.randn(4, 16),
    )

    session.quant(
        name="fp4_quant",
        backend="pytorch",
        method="awq",
        strategy="fp4_weight_only",
        policy={
            "dtype": "fp4",
            "scheme": "weight_only",
            "include_module_names": ["fc"],
            "group_size": 16,
        },
    )

    comparison = session.compare_to_baseline("fp4_quant")
    assert comparison.source_stage == "baseline"
    assert comparison.target_stage == "fp4_quant"
    assert comparison.source_payload_kind == "torch_module"
    assert comparison.target_payload_kind == "quantized_model"
    assert comparison.payload_kind_changed is True
    assert comparison.payload_capability_delta["can_quantize"]["source"] is True
    assert comparison.payload_capability_delta["can_quantize"]["target"] is False
    assert "summary" in comparison.metrics_added
    assert comparison.source_can_restore_model is True
    assert comparison.target_can_restore_model is True
    assert comparison.to_dict()["target_stage"] == "fp4_quant"
