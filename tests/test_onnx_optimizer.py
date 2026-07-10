from pathlib import Path

import pytest
import torch

from xqt.core.artifact import ArtifactManifest, file_sha256
from xqt.core.schema import ExportTargetConfig, OutputDiffConfig
from xqt.core.types import XQTContext
from xqt.export.onnx_exporter import ONNXExportResult
from xqt.export.tensorrt import TensorRTBuildResult
from xqt.export.onnx_optimizer import (
    ONNXOptimizationResult,
    optimize_onnx,
    optimize_qdq_native,
)
from xqt.pipeline import export_pass
from xqt.pipeline.export_pass import ExportPass, resolve_onnx_optimization_config


def test_resolve_onnx_optimization_config_defaults_disabled() -> None:
    assert resolve_onnx_optimization_config({}) == {"enabled": False}


def test_resolve_onnx_optimization_config_accepts_boolean_true() -> None:
    assert resolve_onnx_optimization_config({"onnx_optimization": True}) == {
        "enabled": True
    }


def test_resolve_onnx_optimization_config_mapping_defaults_enabled() -> None:
    assert resolve_onnx_optimization_config({"onnx_optimization": {"level": "all"}}) == {
        "level": "all",
        "enabled": True,
    }


def test_resolve_onnx_optimization_config_rejects_invalid_value() -> None:
    with pytest.raises(ValueError, match="onnx_optimization"):
        resolve_onnx_optimization_config({"onnx_optimization": "extended"})


def test_export_pass_uses_context_runtime_export_targets() -> None:
    context = XQTContext(
        model=torch.nn.Identity().eval(),
        example_inputs=torch.randn(1, 4),
        export_targets=[],
    )

    output = ExportPass().run(context)

    assert output is context
    assert context.export_targets == []
    assert "export" not in context.metrics
    assert context.artifacts == {}


def test_deploy_onnxruntime_materializes_runtime_handle(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output = tmp_path / "model.onnx"
    output.write_bytes(b"onnx-model")

    class _FakeSession:
        def get_providers(self) -> list[str]:
            return ["CPUExecutionProvider"]

    monkeypatch.setattr(
        export_pass,
        "export_onnx",
        lambda *args, **kwargs: ONNXExportResult(
            path=output,
            opset=17,
            checksum=file_sha256(output),
            checked=True,
            metadata={"input_names": ["input"], "output_names": ["output"]},
        ),
    )
    monkeypatch.setattr(
        export_pass,
        "create_onnxruntime_session",
        lambda path, providers=None: _FakeSession(),
    )
    context = XQTContext(
        model=torch.nn.Identity().eval(),
        example_inputs=torch.randn(1, 4),
        artifact_dir=str(tmp_path),
        export_targets=[
            ExportTargetConfig(
                format="onnx",
                output_path=str(output),
                params={"runtime_diff": False},
            )
        ],
        output_diff_config=OutputDiffConfig(),
    )

    ExportPass().run(
        context,
        stage_kind="deploy",
        runtime_handle_request={
            "runtime": "onnxruntime",
            "handle_kind": "inference_session",
            "materialize": True,
            "params": {"providers": ["CPUExecutionProvider"]},
        },
    )

    handle = context.metrics["export"]["runtime_handle"]
    assert handle["runtime"] == "onnxruntime"
    assert handle["handle_kind"] == "inference_session"
    assert isinstance(handle["handle"], _FakeSession)
    assert handle["artifacts"] == {"onnx": str(output)}


def test_deploy_tensorrt_materializes_runtime_handle(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    onnx_path = tmp_path / "model.onnx"
    engine_path = tmp_path / "model.engine"
    onnx_path.write_bytes(b"onnx-model")
    engine_path.write_bytes(b"tensorrt-engine")

    class _FakeTensorRTSession:
        def __init__(self) -> None:
            self.engine_path = engine_path
            self.device = "cuda:1"
            self.engine_inspector = {"layer_count": 3}

    monkeypatch.setattr(
        export_pass,
        "build_tensorrt_engine",
        lambda *args, **kwargs: TensorRTBuildResult(
            engine_path=engine_path,
            command=["trtexec"],
            checksum=file_sha256(engine_path),
            dry_run=False,
            metadata={"plugin_libraries": ["plugins/custom.so"]},
        ),
    )
    monkeypatch.setattr(
        export_pass,
        "create_tensorrt_runtime_session",
        lambda path, *, device, plugin_libraries: _FakeTensorRTSession(),
    )
    context = XQTContext(
        artifact_dir=str(tmp_path),
        export_targets=[
            ExportTargetConfig(
                format="tensorrt",
                output_path=str(engine_path),
                params={"onnx_path": str(onnx_path)},
            )
        ],
        output_diff_config=OutputDiffConfig(),
    )

    ExportPass().run(
        context,
        stage_kind="deploy",
        runtime_handle_request={
            "runtime": "tensorrt",
            "handle_kind": "runtime_session",
            "materialize": True,
            "params": {"device": "cuda:1"},
        },
    )

    handle = context.metrics["export"]["runtime_handle"]
    assert handle["runtime"] == "tensorrt"
    assert handle["handle_kind"] == "runtime_session"
    assert isinstance(handle["handle"], _FakeTensorRTSession)
    assert handle["artifacts"] == {"tensorrt_engine": str(engine_path)}
    assert handle["metadata"]["runtime_validation"] == {
        "status": "session_created",
        "engine_deserialized": True,
        "execution_context_created": True,
    }


def test_optimize_onnx_writes_metadata_without_real_onnxruntime(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "model.onnx"
    source.write_bytes(b"source-onnx")

    def fake_optimize(
        source_path: Path,
        output_path: Path,
        *,
        level: str,
        providers: list[str],
    ) -> None:
        output_path.write_bytes(
            b"optimized:" + source_path.read_bytes() + level.encode() + b":".join(
                provider.encode() for provider in providers
            )
        )

    monkeypatch.setattr(
        "xqt.export.onnx_optimizer._optimize_with_onnxruntime",
        fake_optimize,
    )
    monkeypatch.setattr("xqt.export.onnx_optimizer.validate_onnx", lambda path: True)

    result = optimize_onnx(
        source,
        level="basic",
        providers=["CPUExecutionProvider"],
        native_qdq=False,
        metadata={"stage": "export"},
    )

    assert result.path == tmp_path / "model.optimized.onnx"
    assert result.source_path == source
    assert result.backend == "onnxruntime"
    assert result.level == "basic"
    assert result.checked is True
    assert result.checksum == file_sha256(result.path)
    assert result.metadata["providers"] == ["CPUExecutionProvider"]
    assert result.metadata["stage"] == "export"


def test_optimize_qdq_native_removes_redundant_quantized_round_trip(tmp_path: Path) -> None:
    onnx = pytest.importorskip("onnx")
    helper = onnx.helper
    tensor_proto = onnx.TensorProto
    numpy_helper = onnx.numpy_helper

    source = tmp_path / "qdq.onnx"
    output = tmp_path / "qdq.optimized.onnx"
    scale = numpy_helper.from_array(torch.tensor(0.1, dtype=torch.float32).numpy(), "scale")
    zero_point = numpy_helper.from_array(
        torch.tensor(0, dtype=torch.uint8).numpy(),
        "zero_point",
    )
    graph = helper.make_graph(
        [
            helper.make_node(
                "DequantizeLinear",
                ["input", "scale", "zero_point"],
                ["dequantized"],
                name="dequantize",
            ),
            helper.make_node(
                "QuantizeLinear",
                ["dequantized", "scale", "zero_point"],
                ["requantized"],
                name="quantize",
            ),
            helper.make_node("Identity", ["requantized"], ["output"], name="identity"),
        ],
        "redundant_quantized_round_trip",
        [helper.make_tensor_value_info("input", tensor_proto.UINT8, [1, 4])],
        [helper.make_tensor_value_info("output", tensor_proto.UINT8, [1, 4])],
        [scale, zero_point],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_operatorsetid("", 13)],
    )
    onnx.save(model, source)

    result = optimize_qdq_native(source, output)
    optimized = onnx.load(output)
    op_types = [node.op_type for node in optimized.graph.node]

    assert result.removed_quantize_nodes == 1
    assert result.removed_dequantize_nodes == 1
    assert result.removed_nodes == 2
    assert result.rewired_edges == 1
    assert op_types == ["Identity"]
    assert optimized.graph.node[0].input[0] == "input"


def test_optimize_qdq_native_keeps_fake_quant_boundary(tmp_path: Path) -> None:
    onnx = pytest.importorskip("onnx")
    helper = onnx.helper
    tensor_proto = onnx.TensorProto
    numpy_helper = onnx.numpy_helper

    source = tmp_path / "shared_qdq.onnx"
    output = tmp_path / "shared_qdq.optimized.onnx"
    scale = numpy_helper.from_array(torch.tensor(0.1, dtype=torch.float32).numpy(), "scale")
    zero_point = numpy_helper.from_array(
        torch.tensor(0, dtype=torch.uint8).numpy(),
        "zero_point",
    )
    graph = helper.make_graph(
        [
            helper.make_node(
                "QuantizeLinear",
                ["input", "scale", "zero_point"],
                ["quantized"],
                name="quantize",
            ),
            helper.make_node(
                "DequantizeLinear",
                ["quantized", "scale", "zero_point"],
                ["dequantized"],
                name="dequantize",
            ),
            helper.make_node("Relu", ["dequantized"], ["output"], name="relu"),
        ],
        "fake_quant_boundary",
        [helper.make_tensor_value_info("input", tensor_proto.FLOAT, [1, 4])],
        [helper.make_tensor_value_info("output", tensor_proto.FLOAT, [1, 4])],
        [scale, zero_point],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_operatorsetid("", 13)],
    )
    onnx.save(model, source)

    result = optimize_qdq_native(source, output)
    optimized = onnx.load(output)
    op_types = [node.op_type for node in optimized.graph.node]

    assert result.removed_nodes == 0
    assert op_types == ["QuantizeLinear", "DequantizeLinear", "Relu"]


def test_optimize_qdq_native_keeps_shared_dequantized_boundary(tmp_path: Path) -> None:
    onnx = pytest.importorskip("onnx")
    helper = onnx.helper
    tensor_proto = onnx.TensorProto
    numpy_helper = onnx.numpy_helper

    source = tmp_path / "shared_dequantized_qdq.onnx"
    output = tmp_path / "shared_dequantized_qdq.optimized.onnx"
    scale = numpy_helper.from_array(torch.tensor(0.1, dtype=torch.float32).numpy(), "scale")
    zero_point = numpy_helper.from_array(
        torch.tensor(0, dtype=torch.uint8).numpy(),
        "zero_point",
    )
    graph = helper.make_graph(
        [
            helper.make_node(
                "DequantizeLinear",
                ["input", "scale", "zero_point"],
                ["dequantized"],
                name="dequantize",
            ),
            helper.make_node(
                "QuantizeLinear",
                ["dequantized", "scale", "zero_point"],
                ["requantized"],
                name="quantize",
            ),
            helper.make_node("Relu", ["dequantized"], ["float_output"], name="relu"),
            helper.make_node("Identity", ["requantized"], ["uint8_output"], name="identity"),
        ],
        "shared_dequantized_qdq",
        [helper.make_tensor_value_info("input", tensor_proto.UINT8, [1, 4])],
        [
            helper.make_tensor_value_info("float_output", tensor_proto.FLOAT, [1, 4]),
            helper.make_tensor_value_info("uint8_output", tensor_proto.UINT8, [1, 4]),
        ],
        [scale, zero_point],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_operatorsetid("", 13)],
    )
    onnx.save(model, source)

    result = optimize_qdq_native(source, output)
    optimized = onnx.load(output)
    op_types = [node.op_type for node in optimized.graph.node]

    assert result.removed_nodes == 0
    assert op_types == ["DequantizeLinear", "QuantizeLinear", "Relu", "Identity"]


def test_export_pass_uses_optimized_onnx_as_last_artifact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model = torch.nn.Linear(2, 1)
    example_inputs = torch.randn(1, 2)
    source = tmp_path / "model.onnx"
    optimized = tmp_path / "model.optimized.onnx"

    def fake_export_onnx(*args: object, **kwargs: object) -> ONNXExportResult:
        source.write_bytes(b"source")
        return ONNXExportResult(
            path=source,
            opset=17,
            checksum=file_sha256(source),
            checked=True,
            metadata={
                "input_names": ["input"],
                "pre_export_fusion": {},
            },
        )

    def fake_optimize_onnx(*args: object, **kwargs: object) -> ONNXOptimizationResult:
        optimized.write_bytes(b"optimized")
        return ONNXOptimizationResult(
            path=optimized,
            source_path=source,
            backend="onnxruntime",
            level="extended",
            checksum=file_sha256(optimized),
            checked=True,
            metadata={
                "providers": ["CPUExecutionProvider"],
                "source_export_path": str(source),
            },
        )

    monkeypatch.setattr(export_pass, "export_onnx", fake_export_onnx)
    monkeypatch.setattr(export_pass, "optimize_onnx", fake_optimize_onnx)

    export_targets = [
        ExportTargetConfig(
            format="onnx",
            output_path=str(source),
            opset=17,
            params={
                "runtime_diff": False,
                "onnx_optimization": {
                    "enabled": True,
                    "level": "extended",
                },
            },
        )
    ]
    context = XQTContext(
        model=model,
        example_inputs=example_inputs,
        artifact_dir=str(tmp_path),
        export_targets=export_targets,
        output_diff_config=OutputDiffConfig(),
        manifest=ArtifactManifest(project_name="test"),
    )

    ExportPass().run(context)

    assert context.artifacts["export_0"] == optimized
    assert context.artifacts["last_onnx"] == optimized
    assert context.artifacts["export_0_source_onnx"] == source
    assert context.artifacts["last_onnx_source"] == source
    assert context.metrics["export"]["artifacts"][0]["path"] == str(optimized)
    assert context.metrics["export"]["artifacts"][0]["onnx_optimization"][
        "path"
    ] == str(optimized)
    assert context.metrics["export"]["target_count"] == 1
    assert context.metrics["export"]["targets"][0]["format"] == "onnx"
    assert context.metrics["export"]["targets"][0]["path"] == str(optimized)
    assert context.metrics["export"]["stage_kind"] == "export"
    assert context.manifest is not None
    assert context.manifest.artifacts[0].path == str(optimized)
    assert context.manifest.artifacts[0].metadata["onnx_optimization"]["source_path"] == str(
        source
    )


def test_export_pass_prefers_context_artifact_dir_for_default_outputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model = torch.nn.Identity().eval()
    example_inputs = torch.randn(1, 4)
    runtime_dir = tmp_path / "runtime"
    source = runtime_dir / "model_0.onnx"
    optimized = runtime_dir / "model_0.optimized.onnx"

    def fake_export_onnx(
        module: torch.nn.Module,
        example_input: object,
        output_path: str | Path,
        **_: object,
    ) -> ONNXExportResult:
        assert module is model
        assert example_input is example_inputs
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"source")
        return ONNXExportResult(
            path=output,
            opset=17,
            checksum=file_sha256(output),
            checked=True,
            metadata={
                "input_names": ["input"],
                "pre_export_fusion": {},
            },
        )

    def fake_optimize_onnx(*args: object, **kwargs: object) -> ONNXOptimizationResult:
        assert args[0] == source
        optimized.parent.mkdir(parents=True, exist_ok=True)
        optimized.write_bytes(b"optimized")
        return ONNXOptimizationResult(
            path=optimized,
            source_path=source,
            backend="onnxruntime",
            level="extended",
            checksum=file_sha256(optimized),
            checked=True,
            metadata={
                "providers": ["CPUExecutionProvider"],
                "source_export_path": str(source),
            },
        )

    monkeypatch.setattr(export_pass, "export_onnx", fake_export_onnx)
    monkeypatch.setattr(export_pass, "optimize_onnx", fake_optimize_onnx)

    export_targets = [
        ExportTargetConfig(
            format="onnx",
            output_path=None,
            opset=17,
            params={
                "runtime_diff": False,
                "onnx_optimization": {
                    "enabled": True,
                    "level": "extended",
                },
            },
        )
    ]
    context = XQTContext(
        model=model,
        example_inputs=example_inputs,
        artifact_dir=str(runtime_dir),
        export_targets=export_targets,
        output_diff_config=OutputDiffConfig(),
        manifest=ArtifactManifest(project_name="test"),
    )

    ExportPass().run(context)

    assert context.artifacts["export_0"] == optimized
    assert context.artifacts["last_onnx"] == optimized
    assert context.artifacts["export_0_source_onnx"] == source
    assert context.artifacts["last_onnx_source"] == source


def test_export_pass_prefers_context_output_diff_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model = torch.nn.Identity().eval()
    example_inputs = torch.randn(1, 4)
    source = tmp_path / "model.onnx"
    captured: dict[str, object] = {}

    def fake_export_onnx(*args: object, **kwargs: object) -> ONNXExportResult:
        source.write_bytes(b"source")
        return ONNXExportResult(
            path=source,
            opset=17,
            checksum=file_sha256(source),
            checked=True,
            metadata={
                "input_names": ["input"],
                "pre_export_fusion": {},
            },
        )

    class _FakeDiff:
        def to_dict(self) -> dict[str, object]:
            return {"max_abs": 0.0, "mean_abs": 0.0}

    def fake_compare_onnxruntime_outputs(
        path: Path,
        reference_output: object,
        example_input: object,
        *,
        input_names: object,
        atol: float,
        rtol: float,
    ) -> _FakeDiff:
        del path, reference_output, example_input, input_names
        captured["atol"] = atol
        captured["rtol"] = rtol
        return _FakeDiff()

    monkeypatch.setattr(export_pass, "export_onnx", fake_export_onnx)
    monkeypatch.setattr(
        export_pass,
        "compare_onnxruntime_outputs",
        fake_compare_onnxruntime_outputs,
    )

    export_targets = [
        ExportTargetConfig(
            format="onnx",
            output_path=str(source),
            opset=17,
            params={
                "runtime_diff": True,
            },
        )
    ]
    context = XQTContext(
        model=model,
        example_inputs=example_inputs,
        artifact_dir=str(tmp_path / "config_artifacts"),
        export_targets=export_targets,
        output_diff_config=OutputDiffConfig(atol=2e-3, rtol=3e-3),
        manifest=ArtifactManifest(project_name="test"),
    )

    ExportPass().run(context)

    assert captured == {
        "atol": pytest.approx(2e-3),
        "rtol": pytest.approx(3e-3),
    }
