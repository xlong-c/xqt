from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch

from xqt.contracts import RuntimeHandlePayload as ContractRuntimeHandlePayload
from xqt.contracts import RuntimePlanPayload as ContractRuntimePlanPayload
from xqt.contracts import PrunedModelPayload as ContractPrunedModelPayload
from xqt.contracts import QuantizedModelPayload as ContractQuantizedModelPayload
from xqt.workflows import XQTOptimizationSession
from xqt.workflows.stage import (
    ExportBundlePayload,
    PrunedModelPayload,
    QuantizedModelPayload,
    RuntimeHandlePayload,
    payload_capabilities_for_kind,
    payload_can_restore_model,
    payload_kind_for_stage,
    RuntimeArtifactPayload,
    RuntimePlanPayload,
    StageReportPayload,
    stage_kind_for_transform,
    transform_family_for_kind,
)
from xqt.workflows.stage_specs import DeployStageSpec


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
        strategy="w4a16_fp4",
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
    assert quantized.payload.value.strategy == "w4a16_fp4"
    assert (
        quantized.payload.metadata["quantized_model"]["artifact_kind"]
        == "quantized_model"
    )
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


def test_quantized_model_payload_is_contract_reexport() -> None:
    assert QuantizedModelPayload is ContractQuantizedModelPayload
    payload = ContractQuantizedModelPayload(
        stage_name="fp4_quant",
        source_model_stage="baseline",
        model=torch.nn.Linear(2, 2),
        backend="pytorch",
        method="awq",
        strategy="w4a16_fp4",
        calibration_summary={"samples": 4},
        components=[{"target": "weight"}],
        artifacts={"checkpoint": "quantized.pt"},
    )

    assert payload.to_dict() == {
        "artifact_kind": "quantized_model",
        "stage_name": "fp4_quant",
        "source_model_stage": "baseline",
        "model_type": "torch.nn.modules.linear.Linear",
        "backend": "pytorch",
        "method": "awq",
        "strategy": "w4a16_fp4",
        "quantized_module_count": 0,
        "quantized_modules": [],
        "metadata": {},
        "calibration_samples": None,
        "calibration_summary": {"samples": 4},
        "components": [{"target": "weight"}],
        "artifacts": {"checkpoint": "quantized.pt"},
        "capability": None,
        "algorithm_metadata": None,
        "execution_policies": [],
    }


def test_pruned_model_payload_is_contract_reexport() -> None:
    assert PrunedModelPayload is ContractPrunedModelPayload
    payload = ContractPrunedModelPayload(
        stage_name="l1_prune",
        source_model_stage="baseline",
        model=torch.nn.Linear(2, 2),
        method="global_l1_unstructured",
        granularity="weight",
        target_sparsity=0.5,
        sparsity=0.5,
        execution_state="applied",
        applied=True,
        report={"sparsity": 0.5},
        artifacts={"checkpoint": "pruned.pt"},
    )

    assert payload.to_dict() == {
        "artifact_kind": "pruned_model",
        "stage_name": "l1_prune",
        "source_model_stage": "baseline",
        "model_type": "torch.nn.modules.linear.Linear",
        "method": "global_l1_unstructured",
        "granularity": "weight",
        "target_sparsity": 0.5,
        "sparsity": 0.5,
        "execution_state": "applied",
        "applied": True,
        "report": {"sparsity": 0.5},
        "artifacts": {"checkpoint": "pruned.pt"},
        "capability": None,
    }


def test_quant_stage_without_explicit_from_stage_uses_baseline_as_parent(
    tmp_path: Path,
) -> None:
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
        strategy="w4a16_fp4",
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


def test_session_quant_routes_through_run_quant_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = XQTOptimizationSession(
        project={
            "name": "session_quant_stage_dispatch",
            "artifact_dir": str(tmp_path / "artifacts"),
        },
        model=_TinyLinear().eval(),
        example_inputs=torch.randn(4, 16),
    )

    captured: dict[str, object] = {}

    def _fake_run_quant_stage(context, spec) -> object:
        captured["backend"] = spec.backend
        captured["strategy"] = spec.strategy
        context.metrics["quant"] = {
            "quantized_module_count": 1,
            "backend": spec.backend,
            "strategy": spec.strategy,
        }
        return context

    monkeypatch.setattr(
        "xqt.workflows.optimization.run_quant_stage", _fake_run_quant_stage
    )

    result = session.quant(
        name="fp4_quant",
        backend="pytorch",
        method="awq",
        strategy="w4a16_fp4",
        policy={
            "dtype": "fp4",
            "scheme": "weight_only",
            "include_module_names": ["fc"],
            "group_size": 16,
        },
    )

    assert result.accepted is True
    assert captured == {
        "backend": "pytorch",
        "strategy": "w4a16_fp4",
    }
    assert session.session_stages[-1].name == "fp4_quant"


def test_session_prune_routes_through_run_prune_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = XQTOptimizationSession(
        project={
            "name": "session_prune_stage_dispatch",
            "artifact_dir": str(tmp_path / "artifacts"),
        },
        model=_TinyLinear().eval(),
        example_inputs=torch.randn(4, 16),
    )

    captured: dict[str, object] = {}

    def _fake_run_prune_stage(context, spec) -> object:
        captured["method"] = spec.method
        captured["target_sparsity"] = spec.target_sparsity
        context.metrics["prune"] = {
            "sparsity": spec.target_sparsity,
            "method": spec.method,
            "target_sparsity": spec.target_sparsity,
        }
        return context

    monkeypatch.setattr(
        "xqt.workflows.optimization.run_prune_stage", _fake_run_prune_stage
    )

    result = session.prune(
        name="l1_prune",
        method="global_l1_unstructured",
        target_sparsity=0.4,
    )

    assert result.accepted is True
    assert captured == {
        "method": "global_l1_unstructured",
        "target_sparsity": 0.4,
    }
    pruned = session.session_stages[-1]
    assert pruned.name == "l1_prune"
    assert pruned.stage_kind == "optimized"
    assert pruned.payload.payload_kind == "pruned_model"
    assert isinstance(pruned.payload.value, PrunedModelPayload)
    assert pruned.payload.value.method == "global_l1_unstructured"
    assert pruned.payload.value.target_sparsity == pytest.approx(0.4)
    assert pruned.payload.value.sparsity == pytest.approx(0.4)
    assert pruned.payload.value.execution_state == "applied"
    assert pruned.payload.value.applied is True
    assert pruned.payload.metadata["pruned_model"]["report"] == {
        "sparsity": 0.4,
        "method": "global_l1_unstructured",
        "target_sparsity": 0.4,
    }


def test_session_prune_materializes_pruned_model_payload(tmp_path: Path) -> None:
    session = XQTOptimizationSession(
        project={
            "name": "session_prune_payload",
            "artifact_dir": str(tmp_path / "artifacts"),
        },
        model=_TinyLinear().eval(),
        example_inputs=torch.randn(4, 16),
    )

    result = session.prune(
        name="l1_prune",
        method="global_l1_unstructured",
        target_sparsity=0.5,
    )

    assert result.accepted is True
    pruned = session.session_stages[-1]
    assert pruned.payload.payload_kind == "pruned_model"
    assert isinstance(pruned.payload.value, PrunedModelPayload)
    assert pruned.payload.value.applied is True
    assert pruned.payload.value.sparsity == pytest.approx(
        pruned.payload.value.report["sparsity"]
    )
    assert pruned.payload.value.report["method"] == "global_l1_unstructured"
    session.use("l1_prune")
    assert isinstance(session.model, _TinyLinear)


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
        strategy="w4a16_fp4",
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
    assert [item["name"] for item in payload["session_stages"]] == [
        "baseline",
        "fp4_quant",
    ]
    assert payload["session_stages"][1]["stage_kind"] == "quantized"
    assert payload["session_stages"][1]["payload"]["payload_kind"] == "quantized_model"
    assert (
        payload["session_stages"][1]["payload"]["value"]["artifact_kind"]
        == "quantized_model"
    )


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
        strategy="w4a16_fp4",
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
        assert stage.spec.targets[0].onnx.dynamo is False
        assert stage.spec.targets[0].onnx.runtime_diff is False
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
        onnx={"dynamo": False, "runtime_diff": False},
    )

    result = session.write_outputs()
    workflow_path = result.context.artifacts["workflow_result"]
    payload = json.loads(workflow_path.read_text(encoding="utf-8"))
    stage_payload = payload["session_stages"][-1]["payload"]
    assert stage_payload["payload_kind"] == "export_bundle"
    assert stage_payload["value"]["artifact_kind"] == "export_bundle"
    assert stage_payload["value"]["format"] == "onnx"
    assert stage_payload["value"]["source_model_stage"] == "baseline"
    assert (
        session.session_stages[-1].created_by.params["targets"][0]["onnx"]["dynamo"]
        is False
    )


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
        target = stage.spec.targets[0]
        assert target.tensorrt.onnx_path == "artifacts/model.onnx"
        assert target.tensorrt.dry_run is True
        output = tmp_path / "model.engine"
        output.write_bytes(b"fake-engine")
        context.artifacts["export_engine"] = output
        context.metrics["export"] = {
            "target_count": 1,
            "targets": [{"format": "tensorrt", "output_path": str(output)}],
        }

    monkeypatch.setattr("xqt.workflows.optimization._run_export", _fake_run_export)

    export_stage = session.export(
        name="tensorrt_export",
        format="tensorrt",
        output_path=tmp_path / "model.engine",
        tensorrt={"onnx_path": "artifacts/model.onnx", "dry_run": True},
    )

    assert export_stage.accepted is True
    exported = session.session_stages[-1]
    assert exported.name == "tensorrt_export"
    assert exported.stage_kind == "exported"
    assert exported.payload.payload_kind == "export_bundle"
    assert isinstance(exported.payload.value, ExportBundlePayload)
    assert isinstance(exported.payload.value, RuntimeArtifactPayload)
    assert exported.payload.value.artifact_kind == "export_bundle"
    assert exported.payload.value.format == "tensorrt"
    assert exported.payload.value.source_model_stage == "baseline"
    assert (
        exported.payload.metadata["export_bundle"]["artifact_kind"] == "export_bundle"
    )
    assert exported.created_by.transform_family == "export"


def test_export_stage_accepts_openvino_target_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = XQTOptimizationSession(
        project={
            "name": "session_stage_openvino_export",
            "artifact_dir": str(tmp_path / "artifacts"),
        },
        model=_TinyLinear().eval(),
        example_inputs=torch.randn(4, 16),
    )

    def _fake_run_export(config: Any, stage: Any, context: Any) -> None:
        target = stage.spec.targets[0]
        assert target.openvino.onnx_path == "artifacts/model.onnx"
        assert target.openvino.input_shape == [1, 16]
        assert target.openvino.dry_run is True
        assert target.openvino.runtime_diff is False
        assert target.openvino.device == "GPU"
        output = tmp_path / "model.xml"
        output.write_bytes(b"fake-openvino")
        context.artifacts["export_openvino"] = output
        context.metrics["export"] = {
            "target_count": 1,
            "targets": [{"format": "openvino", "output_path": str(output)}],
        }

    monkeypatch.setattr("xqt.workflows.optimization._run_export", _fake_run_export)

    export_stage = session.export(
        name="openvino_export",
        format="openvino",
        output_path=tmp_path / "model.xml",
        openvino={
            "onnx_path": "artifacts/model.onnx",
            "input_shape": [1, 16],
            "dry_run": True,
            "runtime_diff": False,
            "device": "GPU",
        },
    )

    assert export_stage.accepted is True


def test_deploy_stage_accepts_openvino_target_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = XQTOptimizationSession(
        project={
            "name": "session_stage_openvino_deploy",
            "artifact_dir": str(tmp_path / "artifacts"),
        },
        model=_TinyLinear().eval(),
        example_inputs=torch.randn(4, 16),
    )

    def _fake_run_export(config: Any, stage: Any, context: Any) -> None:
        assert isinstance(stage.spec, DeployStageSpec)
        target = stage.spec.targets[0]
        assert target.openvino.onnx_path == "artifacts/model.onnx"
        assert target.openvino.dry_run is True
        output = tmp_path / "model.xml"
        output.write_bytes(b"fake-openvino")
        context.artifacts["deploy_openvino"] = output
        context.metrics["export"] = {
            "target_count": 1,
            "targets": [{"format": "openvino", "output_path": str(output)}],
            "stage_kind": "deploy",
        }

    monkeypatch.setattr("xqt.workflows.optimization._run_export", _fake_run_export)

    deploy_stage = session.deploy(
        name="openvino_deploy",
        format="openvino",
        output_path=tmp_path / "model.xml",
        openvino={
            "onnx_path": "artifacts/model.onnx",
            "dry_run": True,
        },
    )

    assert deploy_stage.accepted is True


def test_export_stage_accepts_torch_export_target_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = XQTOptimizationSession(
        project={
            "name": "session_stage_torch_export",
            "artifact_dir": str(tmp_path / "artifacts"),
        },
        model=_TinyLinear().eval(),
        example_inputs=torch.randn(4, 16),
    )

    def _fake_run_export(config: Any, stage: Any, context: Any) -> None:
        target = stage.spec.targets[0]
        assert target.torch_export.strict is True
        assert target.torch_export.validate is False
        assert target.torch_export.runtime_diff is False
        output = tmp_path / "model.pt2"
        output.write_bytes(b"fake-torch-export")
        context.artifacts["export_torch_export"] = output
        context.metrics["export"] = {
            "target_count": 1,
            "targets": [{"format": "torch_export", "output_path": str(output)}],
        }

    monkeypatch.setattr("xqt.workflows.optimization._run_export", _fake_run_export)

    export_stage = session.export(
        name="torch_export",
        format="torch_export",
        output_path=tmp_path / "model.pt2",
        torch_export={
            "strict": True,
            "validate": False,
            "runtime_diff": False,
        },
    )

    assert export_stage.accepted is True


def test_deploy_stage_accepts_torchscript_target_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = XQTOptimizationSession(
        project={
            "name": "session_stage_torchscript_deploy",
            "artifact_dir": str(tmp_path / "artifacts"),
        },
        model=_TinyLinear().eval(),
        example_inputs=torch.randn(4, 16),
    )

    def _fake_run_export(config: Any, stage: Any, context: Any) -> None:
        assert isinstance(stage.spec, DeployStageSpec)
        target = stage.spec.targets[0]
        assert target.torchscript.method == "script"
        assert target.torchscript.check_trace is False
        assert target.torchscript.runtime_diff is False
        output = tmp_path / "model.pt"
        output.write_bytes(b"fake-torchscript")
        context.artifacts["deploy_torchscript"] = output
        context.metrics["export"] = {
            "target_count": 1,
            "targets": [{"format": "torchscript", "output_path": str(output)}],
            "stage_kind": "deploy",
        }

    monkeypatch.setattr("xqt.workflows.optimization._run_export", _fake_run_export)

    deploy_stage = session.deploy(
        name="torchscript_deploy",
        format="torchscript",
        output_path=tmp_path / "model.pt",
        torchscript={
            "method": "script",
            "check_trace": False,
            "runtime_diff": False,
        },
    )

    assert deploy_stage.accepted is True


@pytest.mark.parametrize("operation", ["export", "deploy"])
@pytest.mark.parametrize(
    ("format_name", "config_name", "config", "expected"),
    [
        ("executorch", "executorch", {"dry_run": True}, {"dry_run": True}),
        (
            "ncnn",
            "ncnn",
            {
                "source_path": "artifacts/model.pt",
                "converter": "pnnx",
                "pnnx_path": "custom-pnnx",
                "dry_run": True,
            },
            {
                "source_path": "artifacts/model.pt",
                "converter": "pnnx",
                "pnnx_path": "custom-pnnx",
                "dry_run": True,
            },
        ),
        (
            "mnn",
            "mnn",
            {
                "source_path": "artifacts/model.onnx",
                "converter_path": "custom-mnnconvert",
                "dry_run": True,
            },
            {
                "source_path": "artifacts/model.onnx",
                "converter_path": "custom-mnnconvert",
                "dry_run": True,
            },
        ),
    ],
)
def test_session_accepts_typed_mobile_target_config(
    operation: str,
    format_name: str,
    config_name: str,
    config: dict[str, Any],
    expected: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = XQTOptimizationSession(
        project={
            "name": f"session_stage_{operation}_{format_name}",
            "artifact_dir": str(tmp_path / "artifacts"),
        },
        model=_TinyLinear().eval(),
        example_inputs=torch.randn(4, 16),
    )
    suffix = {"executorch": ".pte", "ncnn": ".param", "mnn": ".mnn"}[format_name]
    output = tmp_path / f"model{suffix}"

    def _fake_run_export(config_: Any, stage: Any, context: Any) -> None:
        target = stage.spec.targets[0]
        typed_config = getattr(target, config_name)
        for key, value in expected.items():
            assert getattr(typed_config, key) == value
        output.write_bytes(b"fake-mobile-export")
        context.artifacts[f"{operation}_{format_name}"] = output
        context.metrics["export"] = {
            "target_count": 1,
            "targets": [{"format": format_name, "output_path": str(output)}],
            "stage_kind": operation,
        }

    monkeypatch.setattr("xqt.workflows.optimization._run_export", _fake_run_export)

    stage_method = getattr(session, operation)
    stage_result = stage_method(
        name=f"{format_name}_{operation}",
        format=format_name,
        output_path=output,
        **{config_name: config},
    )

    assert stage_result.accepted is True


def test_deploy_stage_accepts_runtime_handle_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = XQTOptimizationSession(
        project={
            "name": "session_stage_deploy",
            "artifact_dir": str(tmp_path / "artifacts"),
        },
        model=_TinyLinear().eval(),
        example_inputs=torch.randn(4, 16),
    )

    captured: dict[str, object] = {}

    def _fake_run_export(config, stage, context) -> None:
        assert isinstance(stage.spec, DeployStageSpec)
        assert stage.spec.runtime_handle is not None
        target = stage.spec.targets[0]
        assert target.tensorrt.onnx_path == "artifacts/model.onnx"
        assert target.tensorrt.backend == "python_api"
        assert target.tensorrt.dry_run is True
        captured["runtime"] = stage.spec.runtime_handle.runtime
        captured["handle_kind"] = stage.spec.runtime_handle.handle_kind
        captured["materialize"] = stage.spec.runtime_handle.materialize
        output = tmp_path / "model.engine"
        output.write_bytes(b"fake-engine")
        context.artifacts["deploy_engine"] = output
        context.metrics["export"] = {
            "target_count": 1,
            "targets": [{"format": "tensorrt", "output_path": str(output)}],
            "stage_kind": "deploy",
            "runtime_handle_request": {
                "runtime": "tensorrt",
                "handle_kind": "engine",
                "materialize": False,
            },
        }

    monkeypatch.setattr("xqt.workflows.optimization._run_export", _fake_run_export)

    deploy_stage = session.deploy(
        name="build_engine",
        format="tensorrt",
        output_path=tmp_path / "model.engine",
        tensorrt={
            "onnx_path": "artifacts/model.onnx",
            "backend": "python_api",
            "dry_run": True,
        },
        runtime_handle={
            "runtime": "tensorrt",
            "handle_kind": "engine",
            "materialize": False,
        },
    )

    assert deploy_stage.accepted is True
    assert captured == {
        "runtime": "tensorrt",
        "handle_kind": "engine",
        "materialize": False,
    }
    exported = session.session_stages[-1]
    assert exported.name == "build_engine"
    assert exported.stage_kind == "exported"
    assert exported.payload.payload_kind == "export_bundle"
    assert exported.created_by.transform == "deploy"
    assert exported.created_by.transform_family == "export"


def test_deploy_stage_materialized_runtime_handle_uses_runtime_handle_payload_kind(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = XQTOptimizationSession(
        project={
            "name": "session_stage_runtime_handle",
            "artifact_dir": str(tmp_path / "artifacts"),
        },
        model=_TinyLinear().eval(),
        example_inputs=torch.randn(4, 16),
    )

    def _fake_run_export(config, stage, context) -> None:
        output = tmp_path / "model.onnx"
        output.write_bytes(b"fake-onnx")
        handle = object()
        context.artifacts["deploy_onnx"] = output
        context.metrics["export"] = {
            "target_count": 1,
            "targets": [{"format": "onnx", "output_path": str(output)}],
            "runtime_handle": {
                "runtime": "onnxruntime",
                "handle_kind": "inference_session",
                "handle": handle,
                "target_count": 1,
                "targets": [{"format": "onnx", "output_path": str(output)}],
                "artifacts": {"onnx": str(output)},
                "metadata": {"runtime_validation": {"status": "session_created"}},
            },
        }

    monkeypatch.setattr("xqt.workflows.optimization._run_export", _fake_run_export)

    deploy_stage = session.deploy(
        name="materialize_onnxruntime",
        format="onnx",
        output_path=tmp_path / "model.onnx",
        runtime_handle={
            "runtime": "onnxruntime",
            "handle_kind": "inference_session",
            "materialize": True,
        },
    )

    assert deploy_stage.accepted is True
    exported = session.session_stages[-1]
    assert exported.stage_kind == "exported"
    assert exported.payload.payload_kind == "runtime_handle"
    assert isinstance(exported.payload.value, RuntimeHandlePayload)
    assert exported.payload.value.handle is not None
    assert exported.payload.capabilities == payload_capabilities_for_kind(
        "runtime_handle"
    )
    assert exported.payload.capabilities["can_evaluate"] is True

    result = session.write_outputs()
    payload = json.loads(
        result.context.artifacts["workflow_result"].read_text(encoding="utf-8")
    )
    stage_payload = payload["session_stages"][-1]["payload"]
    assert stage_payload["payload_kind"] == "runtime_handle"
    assert stage_payload["value"]["artifact_kind"] == "runtime_handle"
    assert stage_payload["value"]["handle_materialized"] is True
    assert payload["metrics"]["export"]["runtime_handle"]["handle"] is None

    manifest = result.context.manifest
    assert manifest is not None
    manifest_path = tmp_path / "manifest.json"
    manifest.write_json(manifest_path)
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    stage_report = next(
        metric["metadata"]
        for metric in manifest_payload["metrics"]
        if metric["name"] == "stage.materialize_onnxruntime.status"
    )
    assert stage_report["metrics"]["runtime_handle"]["handle"] is None


def test_session_export_rejects_empty_targets(tmp_path: Path) -> None:
    session = XQTOptimizationSession(
        project={
            "name": "session_empty_export_targets",
            "artifact_dir": str(tmp_path / "artifacts"),
        },
        model=_TinyLinear().eval(),
        example_inputs=torch.randn(4, 16),
    )

    with pytest.raises(ValueError, match="export requires at least one target"):
        session.export(name="empty_export", targets=[])


def test_session_deploy_rejects_empty_targets(tmp_path: Path) -> None:
    session = XQTOptimizationSession(
        project={
            "name": "session_empty_deploy_targets",
            "artifact_dir": str(tmp_path / "artifacts"),
        },
        model=_TinyLinear().eval(),
        example_inputs=torch.randn(4, 16),
    )

    with pytest.raises(ValueError, match="deploy requires at least one target"):
        session.deploy(name="empty_deploy", targets=[])


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
        strategy="w4a16_fp4",
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
    assert stage_kind_for_transform("benchmark") == "observed"
    assert stage_kind_for_transform("analyze") == "observed"
    assert stage_kind_for_transform("unknown_transform") == "custom"

    assert transform_family_for_kind("quant") == "model_quantizer"
    assert transform_family_for_kind("operator") == "operator_optimizer"
    assert transform_family_for_kind("benchmark") == "evaluation"
    assert transform_family_for_kind("unknown_transform") == "transform"

    assert (
        payload_kind_for_stage("optimized", transform_kind="operator") == "runtime_plan"
    )
    assert payload_kind_for_stage("optimized", transform_kind="prune") == "pruned_model"
    assert (
        payload_kind_for_stage("exported", transform_kind="export") == "export_bundle"
    )
    assert (
        payload_kind_for_stage("observed", transform_kind="benchmark")
        == "stage_report"
    )
    assert (
        payload_kind_for_stage("quantized", transform_kind="quant") == "quantized_model"
    )

    runtime_capabilities = payload_capabilities_for_kind("runtime_plan")
    assert runtime_capabilities["can_restore_model"] is False
    assert payload_can_restore_model("torch_module") is True
    assert payload_can_restore_model("pruned_model") is True
    assert payload_can_restore_model("quantized_model") is True
    assert payload_can_restore_model("runtime_plan") is False
    assert payload_can_restore_model("stage_report") is False

    runtime_capabilities["can_export"] = False
    fresh_runtime_capabilities = payload_capabilities_for_kind("runtime_plan")
    assert fresh_runtime_capabilities["can_export"] is True


def test_benchmark_stage_uses_observation_payload_and_keeps_best_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = XQTOptimizationSession(
        project={
            "name": "session_stage_benchmark_observation",
            "artifact_dir": str(tmp_path / "artifacts"),
        },
        model=_TinyLinear().eval(),
        example_inputs=torch.randn(4, 16),
    )

    session.quant(
        name="fp4_quant",
        backend="pytorch",
        method="awq",
        strategy="w4a16_fp4",
        policy={
            "dtype": "fp4",
            "scheme": "weight_only",
            "include_module_names": ["fc"],
            "group_size": 16,
        },
    )

    def _fake_run_benchmark_stage(context, spec, *, base_benchmark_config=None) -> object:
        del spec, base_benchmark_config
        context.metrics["benchmark"] = {
            "latency": {"p50_ms": 1.5},
            "p50_ms": 1.5,
        }
        return context

    monkeypatch.setattr(
        "xqt.workflows.optimization.run_benchmark_stage", _fake_run_benchmark_stage
    )

    result = session.benchmark(name="post_quant_benchmark", save_model=True)

    assert result.accepted is True
    assert session.best_stage == "fp4_quant"
    observed = session.session_stages[-1]
    assert observed.name == "post_quant_benchmark"
    assert observed.stage_kind == "observed"
    assert observed.payload.payload_kind == "stage_report"
    assert isinstance(observed.payload.value, StageReportPayload)
    assert observed.payload.value.report_kind == "benchmark"
    assert observed.payload.capabilities == payload_capabilities_for_kind("stage_report")

    with pytest.raises(ValueError, match="restorable model payload"):
        session.use("post_quant_benchmark")


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
        strategy="w4a16_fp4",
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
    assert RuntimeHandlePayload is ContractRuntimeHandlePayload
    assert RuntimePlanPayload is ContractRuntimePlanPayload
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
        strategy="w4a16_fp4",
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
