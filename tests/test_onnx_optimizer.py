from pathlib import Path

import pytest
import torch

from xqt.core.artifact import ArtifactManifest, file_sha256
from xqt.core.schema import (
    ExecuTorchExportConfig,
    ExportTargetConfig,
    MNNExportConfig,
    NCNNExportConfig,
    ONNXExportConfig,
    ONNXOptimizationConfig,
    OpenVINOExportConfig,
    OutputDiffConfig,
    TensorRTExportConfig,
    TorchExportConfig,
    TorchScriptExportConfig,
)
from xqt.core.types import XQTContext
from xqt.export.onnx_exporter import ONNXExportResult
from xqt.export.openvino import OpenVINOExportResult
from xqt.export.mobile import CommandExportResult, ExecuTorchExportResult
from xqt.export.tensorrt import TensorRTBuildResult
from xqt.export.torch_exporter import TorchExportResult, TorchScriptExportResult
from xqt.export.onnx_optimizer import (
    ONNXOptimizationResult,
    optimize_onnx,
    optimize_qdq_native,
)
from xqt.pipeline import export_pass
from xqt.pipeline.export_pass import ExportPass
from xqt.workflows.stage_specs import (
    DeployRuntimeHandleSpec,
    ONNXRuntimeHandleConfig,
    TensorRTRuntimeHandleConfig,
)


def test_onnx_target_options_are_explicit_dataclasses() -> None:
    target = ExportTargetConfig(
        format="onnx",
        onnx=ONNXExportConfig(
            dynamo=False,
            runtime_diff=False,
            optimization=ONNXOptimizationConfig(enabled=True, level="all"),
        ),
    )

    assert target.params == {}
    assert target.onnx.dynamo is False
    assert target.onnx.runtime_diff is False
    assert target.onnx.optimization.enabled is True
    assert target.onnx.optimization.level == "all"


def test_openvino_target_options_are_explicit_dataclasses() -> None:
    target = ExportTargetConfig(
        format="openvino",
        openvino=OpenVINOExportConfig(
            onnx_path="artifacts/model.onnx",
            input_shape=[1, 3, 224, 224],
            dry_run=True,
            runtime_diff=False,
            device="GPU",
        ),
    )

    assert target.params == {}
    assert target.openvino.onnx_path == "artifacts/model.onnx"
    assert target.openvino.input_shape == [1, 3, 224, 224]
    assert target.openvino.dry_run is True
    assert target.openvino.runtime_diff is False
    assert target.openvino.device == "GPU"


def test_pytorch_export_target_options_are_explicit_dataclasses() -> None:
    torch_export = ExportTargetConfig(
        format="torch_export",
        torch_export=TorchExportConfig(
            strict=True,
            validate=False,
            runtime_diff=False,
        ),
    )
    torchscript = ExportTargetConfig(
        format="torchscript",
        torchscript=TorchScriptExportConfig(
            method="script",
            check_trace=False,
            runtime_diff=False,
        ),
    )

    assert torch_export.params == {}
    assert torch_export.torch_export.strict is True
    assert torch_export.torch_export.validate is False
    assert torch_export.torch_export.runtime_diff is False
    assert torchscript.params == {}
    assert torchscript.torchscript.method == "script"
    assert torchscript.torchscript.check_trace is False
    assert torchscript.torchscript.runtime_diff is False


def test_mobile_export_target_options_are_explicit_dataclasses() -> None:
    executorch = ExportTargetConfig(
        format="executorch",
        executorch=ExecuTorchExportConfig(dry_run=True),
    )
    ncnn = ExportTargetConfig(
        format="ncnn",
        ncnn=NCNNExportConfig(
            source_path="artifacts/model.pt",
            converter="pnnx",
            pnnx_path="custom-pnnx",
            bin_path="artifacts/model.bin",
            extra_args=["inputshape=[1,4]"],
            timeout=12.5,
            dry_run=True,
        ),
    )
    mnn = ExportTargetConfig(
        format="mnn",
        mnn=MNNExportConfig(
            source_path="artifacts/model.onnx",
            converter_path="custom-mnnconvert",
            framework="ONNX",
            extra_args=["--bizCode", "xqt"],
            timeout=8.0,
            dry_run=True,
        ),
    )

    assert executorch.params == {}
    assert executorch.executorch.dry_run is True
    assert ncnn.params == {}
    assert ncnn.ncnn.converter == "pnnx"
    assert ncnn.ncnn.source_path == "artifacts/model.pt"
    assert ncnn.ncnn.bin_path == "artifacts/model.bin"
    assert mnn.params == {}
    assert mnn.mnn.source_path == "artifacts/model.onnx"
    assert mnn.mnn.converter_path == "custom-mnnconvert"


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


def test_export_pass_empty_targets_overwrites_stale_export_metrics() -> None:
    context = XQTContext(
        model=torch.nn.Identity().eval(),
        example_inputs=torch.randn(1, 4),
        export_targets=[],
        output_diff_config=OutputDiffConfig(),
    )
    context.metrics["export"] = {"target_count": 99, "targets": ["stale"]}

    ExportPass().run(context, stage_kind="deploy")

    assert "export" not in context.metrics


def test_deploy_onnxruntime_materializes_runtime_handle(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output = tmp_path / "model.onnx"
    output.write_bytes(b"onnx-model")

    class _FakeSession:
        def get_providers(self) -> list[str]:
            return ["CPUExecutionProvider"]

    captured: dict[str, object] = {}

    def _create_session(path: object, providers: object = None) -> _FakeSession:
        captured["path"] = path
        captured["providers"] = providers
        return _FakeSession()

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
        _create_session,
    )
    context = XQTContext(
        model=torch.nn.Identity().eval(),
        example_inputs=torch.randn(1, 4),
        artifact_dir=str(tmp_path),
        export_targets=[
            ExportTargetConfig(
                format="onnx",
                output_path=str(output),
                onnx=ONNXExportConfig(runtime_diff=False),
            )
        ],
        output_diff_config=OutputDiffConfig(),
    )

    ExportPass().run(
        context,
        stage_kind="deploy",
        runtime_handle_request=DeployRuntimeHandleSpec(
            runtime="onnxruntime",
            handle_kind="inference_session",
            materialize=True,
            onnxruntime=ONNXRuntimeHandleConfig(
                providers=["CPUExecutionProvider"],
            ),
        ),
    )

    handle = context.metrics["export"]["runtime_handle"]
    assert handle["runtime"] == "onnxruntime"
    assert handle["handle_kind"] == "inference_session"
    assert isinstance(handle["handle"], _FakeSession)
    assert handle["artifacts"] == {"onnx": str(output)}
    assert captured["path"] == output
    assert captured["providers"] == ["CPUExecutionProvider"]


def test_deploy_onnxruntime_runtime_handle_rejects_multiple_onnx_targets(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.onnx"
    second = tmp_path / "second.onnx"
    first.write_bytes(b"onnx-model-a")
    second.write_bytes(b"onnx-model-b")

    monkeypatch.setattr(
        export_pass,
        "export_onnx",
        lambda model, example_input, output_path, **kwargs: ONNXExportResult(
            path=Path(output_path),
            opset=17,
            checksum=file_sha256(Path(output_path)),
            checked=True,
            metadata={"input_names": ["input"], "output_names": ["output"]},
        ),
    )

    def _unexpected_session(*args: object, **kwargs: object) -> None:
        raise AssertionError("create_onnxruntime_session should not be called")

    monkeypatch.setattr(
        export_pass,
        "create_onnxruntime_session",
        _unexpected_session,
    )
    context = XQTContext(
        model=torch.nn.Identity().eval(),
        example_inputs=torch.randn(1, 4),
        artifact_dir=str(tmp_path),
        export_targets=[
            ExportTargetConfig(
                format="onnx",
                output_path=str(first),
                onnx=ONNXExportConfig(runtime_diff=False),
            ),
            ExportTargetConfig(
                format="onnx",
                output_path=str(second),
                onnx=ONNXExportConfig(runtime_diff=False),
            ),
        ],
        output_diff_config=OutputDiffConfig(),
    )

    with pytest.raises(
        ValueError,
        match="requires exactly one ONNX export target",
    ):
        ExportPass().run(
            context,
            stage_kind="deploy",
            runtime_handle_request=DeployRuntimeHandleSpec(
                runtime="onnxruntime",
                handle_kind="inference_session",
                materialize=True,
                onnxruntime=ONNXRuntimeHandleConfig(
                    providers=["CPUExecutionProvider"],
                ),
            ),
        )


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
    captured: dict[str, object] = {}

    def _create_session(
        path: object,
        *,
        device: str,
        plugin_libraries: list[str] | None,
    ) -> _FakeTensorRTSession:
        captured["path"] = path
        captured["device"] = device
        captured["plugin_libraries"] = plugin_libraries
        return _FakeTensorRTSession()

    monkeypatch.setattr(
        export_pass,
        "create_tensorrt_runtime_session",
        _create_session,
    )
    context = XQTContext(
        artifact_dir=str(tmp_path),
        export_targets=[
            ExportTargetConfig(
                format="tensorrt",
                output_path=str(engine_path),
                tensorrt=TensorRTExportConfig(onnx_path=str(onnx_path)),
            )
        ],
        output_diff_config=OutputDiffConfig(),
    )

    ExportPass().run(
        context,
        stage_kind="deploy",
        runtime_handle_request=DeployRuntimeHandleSpec(
            runtime="tensorrt",
            handle_kind="runtime_session",
            materialize=True,
            tensorrt=TensorRTRuntimeHandleConfig(
                device="cuda:1",
                plugin_libraries=["plugins/custom.so"],
            ),
        ),
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
    assert captured["path"] == engine_path
    assert captured["device"] == "cuda:1"
    assert captured["plugin_libraries"] == ["plugins/custom.so"]


def test_deploy_tensorrt_runtime_handle_rejects_multiple_tensorrt_targets(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    onnx_path = tmp_path / "model.onnx"
    first_engine = tmp_path / "first.engine"
    second_engine = tmp_path / "second.engine"
    onnx_path.write_bytes(b"onnx-model")
    first_engine.write_bytes(b"tensorrt-engine-a")
    second_engine.write_bytes(b"tensorrt-engine-b")

    def _build_engine(
        onnx: object,
        output_path: object,
        **kwargs: object,
    ) -> TensorRTBuildResult:
        del onnx, kwargs
        path = Path(output_path)
        return TensorRTBuildResult(
            engine_path=path,
            command=["trtexec", str(path)],
            checksum=file_sha256(path),
            dry_run=False,
            metadata={},
        )

    def _unexpected_runtime_session(
        *args: object,
        **kwargs: object,
    ) -> None:
        raise AssertionError("create_tensorrt_runtime_session should not be called")

    monkeypatch.setattr(export_pass, "build_tensorrt_engine", _build_engine)
    monkeypatch.setattr(
        export_pass,
        "create_tensorrt_runtime_session",
        _unexpected_runtime_session,
    )
    context = XQTContext(
        artifact_dir=str(tmp_path),
        export_targets=[
            ExportTargetConfig(
                format="tensorrt",
                output_path=str(first_engine),
                tensorrt=TensorRTExportConfig(onnx_path=str(onnx_path)),
            ),
            ExportTargetConfig(
                format="tensorrt",
                output_path=str(second_engine),
                tensorrt=TensorRTExportConfig(onnx_path=str(onnx_path)),
            ),
        ],
        output_diff_config=OutputDiffConfig(),
    )

    with pytest.raises(
        ValueError,
        match="requires exactly one TensorRT export target",
    ):
        ExportPass().run(
            context,
            stage_kind="deploy",
            runtime_handle_request=DeployRuntimeHandleSpec(
                runtime="tensorrt",
                handle_kind="runtime_session",
                materialize=True,
            ),
        )


def test_export_pass_forwards_typed_tensorrt_target_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    onnx_path = tmp_path / "model.onnx"
    engine_path = tmp_path / "model.engine"
    onnx_path.write_bytes(b"onnx-model")
    captured: dict[str, object] = {}

    def _build_engine(
        onnx: object,
        engine: object,
        **kwargs: object,
    ) -> TensorRTBuildResult:
        captured["onnx_path"] = onnx
        captured["engine_path"] = engine
        captured.update(kwargs)
        return TensorRTBuildResult(
            engine_path=engine_path,
            command=["trtexec", "--dryRun"],
            dry_run=True,
        )

    monkeypatch.setattr(export_pass, "build_tensorrt_engine", _build_engine)
    context = XQTContext(
        artifact_dir=str(tmp_path),
        export_targets=[
            ExportTargetConfig(
                format="tensorrt",
                output_path=str(engine_path),
                precision="fp16",
                profiles={"input": {"min": [1, 4], "opt": [2, 4], "max": [4, 4]}},
                tensorrt=TensorRTExportConfig(
                    onnx_path=str(onnx_path),
                    backend="python_api",
                    trtexec_path="custom-trtexec",
                    extra_args=["--verbose"],
                    timeout=12.5,
                    dry_run=True,
                    performance_thresholds={"latency_ms": 1.5},
                    workspace_mib=512,
                    builder_optimization_level=3,
                    timing_cache_path="artifacts/timing.cache",
                    log_level="verbose",
                    plugin_libraries=["plugins/custom.so"],
                    serialize_plugin_libraries=False,
                ),
            )
        ],
        output_diff_config=OutputDiffConfig(),
    )

    ExportPass().run(context)

    assert captured == {
        "onnx_path": str(onnx_path),
        "engine_path": str(engine_path),
        "precision": "fp16",
        "profiles": {"input": {"min": [1, 4], "opt": [2, 4], "max": [4, 4]}},
        "trtexec_path": "custom-trtexec",
        "extra_args": ["--verbose"],
        "timeout": 12.5,
        "dry_run": True,
        "performance_thresholds": {"latency_ms": 1.5},
        "backend": "python_api",
        "workspace_mib": 512,
        "builder_optimization_level": 3,
        "timing_cache_path": "artifacts/timing.cache",
        "log_level": "verbose",
        "plugin_libraries": ["plugins/custom.so"],
        "serialize_plugin_libraries": False,
    }


def test_export_pass_forwards_typed_openvino_target_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    onnx_path = tmp_path / "model.onnx"
    xml_path = tmp_path / "model.xml"
    onnx_path.write_bytes(b"onnx-model")
    captured: dict[str, object] = {}

    def _export_openvino(
        source: object,
        output_path: object,
        *,
        example_input: object = None,
        input_shape: list[int] | None = None,
        dry_run: bool = False,
    ) -> OpenVINOExportResult:
        captured["source"] = source
        captured["output_path"] = output_path
        captured["example_input"] = example_input
        captured["input_shape"] = input_shape
        captured["dry_run"] = dry_run
        return OpenVINOExportResult(
            xml_path=xml_path,
            bin_path=xml_path.with_suffix(".bin"),
            checksum=None,
            dry_run=True,
            source_path=onnx_path,
            metadata={"input_shape": input_shape},
        )

    monkeypatch.setattr(export_pass, "export_openvino_ir", _export_openvino)
    context = XQTContext(
        artifact_dir=str(tmp_path),
        export_targets=[
            ExportTargetConfig(
                format="openvino",
                output_path=str(xml_path),
                openvino=OpenVINOExportConfig(
                    onnx_path=str(onnx_path),
                    input_shape=[1, 3, 224, 224],
                    dry_run=True,
                    runtime_diff=False,
                    device="GPU",
                ),
            )
        ],
        output_diff_config=OutputDiffConfig(),
    )

    ExportPass().run(context)

    assert captured == {
        "source": str(onnx_path),
        "output_path": str(xml_path),
        "example_input": None,
        "input_shape": [1, 3, 224, 224],
        "dry_run": True,
    }
    target_summary = context.metrics["export"]["targets"][0]
    assert target_summary["openvino"] == {
        "onnx_path": str(onnx_path),
        "input_shape": [1, 3, 224, 224],
        "dry_run": True,
        "runtime_diff": False,
        "device": "GPU",
    }


def test_export_pass_forwards_typed_pytorch_export_target_configs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    torch_export_path = tmp_path / "model.pt2"
    torchscript_path = tmp_path / "model.pt"
    torch_export_path.write_bytes(b"torch-export")
    torchscript_path.write_bytes(b"torchscript")
    captured: dict[str, object] = {}

    def _export_torch_program(
        model: torch.nn.Module,
        example_input: object,
        output_path: object,
        *,
        dynamic_shapes: object = None,
        strict: bool = False,
        validate: bool = True,
        compare_output: bool = True,
        atol: float = 1e-5,
        rtol: float = 1e-5,
    ) -> TorchExportResult:
        del model, example_input, output_path, atol, rtol
        captured["torch_export"] = {
            "dynamic_shapes": dynamic_shapes,
            "strict": strict,
            "validate": validate,
            "compare_output": compare_output,
        }
        return TorchExportResult(
            path=torch_export_path,
            checksum=file_sha256(torch_export_path),
            checked=False,
        )

    def _export_torchscript(
        model: torch.nn.Module,
        example_input: object,
        output_path: object,
        *,
        method: str = "trace",
        check_trace: bool = True,
        compare_output: bool = True,
        atol: float = 1e-5,
        rtol: float = 1e-5,
    ) -> TorchScriptExportResult:
        del model, example_input, output_path, atol, rtol
        captured["torchscript"] = {
            "method": method,
            "check_trace": check_trace,
            "compare_output": compare_output,
        }
        return TorchScriptExportResult(
            path=torchscript_path,
            checksum=file_sha256(torchscript_path),
        )

    monkeypatch.setattr(export_pass, "export_torch_program", _export_torch_program)
    monkeypatch.setattr(export_pass, "export_torchscript", _export_torchscript)
    context = XQTContext(
        model=torch.nn.Identity().eval(),
        example_inputs=torch.randn(1, 4),
        artifact_dir=str(tmp_path),
        export_targets=[
            ExportTargetConfig(
                format="torch_export",
                output_path=str(torch_export_path),
                dynamic_shapes={"input": {0: "batch"}},
                torch_export=TorchExportConfig(
                    strict=True,
                    validate=False,
                    runtime_diff=False,
                ),
            ),
            ExportTargetConfig(
                format="torchscript",
                output_path=str(torchscript_path),
                torchscript=TorchScriptExportConfig(
                    method="script",
                    check_trace=False,
                    runtime_diff=False,
                ),
            ),
        ],
        output_diff_config=OutputDiffConfig(),
    )

    ExportPass().run(context)

    assert captured == {
        "torch_export": {
            "dynamic_shapes": {"input": {0: "batch"}},
            "strict": True,
            "validate": False,
            "compare_output": False,
        },
        "torchscript": {
            "method": "script",
            "check_trace": False,
            "compare_output": False,
        },
    }
    summaries = context.metrics["export"]["targets"]
    assert summaries[0]["torch_export"] == {
        "strict": True,
        "validate": False,
        "runtime_diff": False,
    }
    assert summaries[1]["torchscript"] == {
        "method": "script",
        "check_trace": False,
        "runtime_diff": False,
    }


def test_export_pass_forwards_typed_mobile_target_configs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "model.onnx"
    executorch_path = tmp_path / "model.pte"
    pnnx_param_path = tmp_path / "pnnx.param"
    pnnx_bin_path = tmp_path / "pnnx.bin"
    onnx_param_path = tmp_path / "onnx.param"
    onnx_bin_path = tmp_path / "onnx.bin"
    mnn_path = tmp_path / "model.mnn"
    source_path.write_bytes(b"onnx-model")
    captured: dict[str, object] = {}

    def _export_executorch(
        model: torch.nn.Module,
        example_input: object,
        output_path: object,
        *,
        dry_run: bool = False,
        metadata: dict[str, object] | None = None,
    ) -> ExecuTorchExportResult:
        del model, example_input
        captured["executorch"] = {
            "output_path": output_path,
            "dry_run": dry_run,
            "metadata": metadata,
        }
        return ExecuTorchExportResult(
            pte_path=executorch_path,
            dry_run=True,
            metadata=dict(metadata or {}),
        )

    def _export_ncnn_pnnx(
        model_path: object,
        *,
        pnnx_path: str = "pnnx",
        param_path: object = None,
        bin_path: object = None,
        extra_args: list[str] | None = None,
        timeout: float | None = None,
        dry_run: bool = False,
    ) -> CommandExportResult:
        captured["pnnx"] = {
            "model_path": model_path,
            "pnnx_path": pnnx_path,
            "param_path": param_path,
            "bin_path": bin_path,
            "extra_args": extra_args,
            "timeout": timeout,
            "dry_run": dry_run,
        }
        return CommandExportResult(
            output_paths=[Path(param_path), Path(bin_path)],
            command=[pnnx_path, str(model_path)],
            dry_run=True,
        )

    def _export_ncnn_onnx(
        onnx_path: object,
        param_path: object,
        bin_path: object,
        *,
        onnx2ncnn_path: str = "onnx2ncnn",
        extra_args: list[str] | None = None,
        timeout: float | None = None,
        dry_run: bool = False,
    ) -> CommandExportResult:
        captured["onnx2ncnn"] = {
            "onnx_path": onnx_path,
            "onnx2ncnn_path": onnx2ncnn_path,
            "param_path": param_path,
            "bin_path": bin_path,
            "extra_args": extra_args,
            "timeout": timeout,
            "dry_run": dry_run,
        }
        return CommandExportResult(
            output_paths=[Path(param_path), Path(bin_path)],
            command=[onnx2ncnn_path, str(onnx_path)],
            dry_run=True,
        )

    def _export_mnn(
        onnx_path: object,
        output_path: object,
        *,
        converter_path: str = "MNNConvert",
        framework: str = "ONNX",
        extra_args: list[str] | None = None,
        timeout: float | None = None,
        dry_run: bool = False,
    ) -> CommandExportResult:
        captured["mnn"] = {
            "source_path": onnx_path,
            "output_path": output_path,
            "converter_path": converter_path,
            "framework": framework,
            "extra_args": extra_args,
            "timeout": timeout,
            "dry_run": dry_run,
        }
        return CommandExportResult(
            output_paths=[Path(output_path)],
            command=[converter_path, str(onnx_path)],
            dry_run=True,
        )

    monkeypatch.setattr(export_pass, "export_executorch_program", _export_executorch)
    monkeypatch.setattr(export_pass, "export_ncnn_with_pnnx", _export_ncnn_pnnx)
    monkeypatch.setattr(export_pass, "export_ncnn_from_onnx", _export_ncnn_onnx)
    monkeypatch.setattr(export_pass, "export_mnn_from_onnx", _export_mnn)
    context = XQTContext(
        model=torch.nn.Identity().eval(),
        example_inputs=torch.randn(1, 4),
        artifact_dir=str(tmp_path),
        export_targets=[
            ExportTargetConfig(
                format="executorch",
                output_path=str(executorch_path),
                executorch=ExecuTorchExportConfig(dry_run=True),
            ),
            ExportTargetConfig(
                format="ncnn",
                output_path=str(pnnx_param_path),
                ncnn=NCNNExportConfig(
                    source_path=str(source_path),
                    converter="pnnx",
                    pnnx_path="custom-pnnx",
                    bin_path=str(pnnx_bin_path),
                    extra_args=["inputshape=[1,4]"],
                    timeout=12.5,
                    dry_run=True,
                ),
            ),
            ExportTargetConfig(
                format="ncnn",
                output_path=str(onnx_param_path),
                ncnn=NCNNExportConfig(
                    source_path=str(source_path),
                    converter="onnx2ncnn",
                    onnx2ncnn_path="custom-onnx2ncnn",
                    bin_path=str(onnx_bin_path),
                    extra_args=["--fp16"],
                    timeout=7.0,
                    dry_run=True,
                ),
            ),
            ExportTargetConfig(
                format="mnn",
                output_path=str(mnn_path),
                mnn=MNNExportConfig(
                    source_path=str(source_path),
                    converter_path="custom-mnnconvert",
                    framework="ONNX",
                    extra_args=["--bizCode", "xqt"],
                    timeout=8.0,
                    dry_run=True,
                ),
            ),
        ],
        output_diff_config=OutputDiffConfig(),
    )

    ExportPass().run(context)

    assert captured == {
        "executorch": {
            "output_path": str(executorch_path),
            "dry_run": True,
            "metadata": {"precision": None},
        },
        "pnnx": {
            "model_path": str(source_path),
            "pnnx_path": "custom-pnnx",
            "param_path": str(pnnx_param_path),
            "bin_path": str(pnnx_bin_path),
            "extra_args": ["inputshape=[1,4]"],
            "timeout": 12.5,
            "dry_run": True,
        },
        "onnx2ncnn": {
            "onnx_path": str(source_path),
            "onnx2ncnn_path": "custom-onnx2ncnn",
            "param_path": str(onnx_param_path),
            "bin_path": str(onnx_bin_path),
            "extra_args": ["--fp16"],
            "timeout": 7.0,
            "dry_run": True,
        },
        "mnn": {
            "source_path": str(source_path),
            "output_path": str(mnn_path),
            "converter_path": "custom-mnnconvert",
            "framework": "ONNX",
            "extra_args": ["--bizCode", "xqt"],
            "timeout": 8.0,
            "dry_run": True,
        },
    }
    summaries = context.metrics["export"]["targets"]
    assert summaries[0]["executorch"] == {"dry_run": True}
    assert summaries[1]["ncnn"]["converter"] == "pnnx"
    assert summaries[2]["ncnn"]["converter"] == "onnx2ncnn"
    assert summaries[3]["mnn"]["converter_path"] == "custom-mnnconvert"


def test_export_pass_updates_last_torchscript_before_default_pnnx_source(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.pt"
    second = tmp_path / "second.pt"
    param_path = tmp_path / "model.param"
    bin_path = tmp_path / "model.bin"
    captured: dict[str, object] = {}

    def _export_torchscript(
        model: torch.nn.Module,
        example_input: object,
        output_path: object,
        **kwargs: object,
    ) -> TorchScriptExportResult:
        del model, example_input, kwargs
        path = Path(output_path)
        path.write_bytes(path.name.encode("utf-8"))
        return TorchScriptExportResult(
            path=path,
            checksum=file_sha256(path),
            output_diff=None,
            metadata={},
        )

    def _export_ncnn_pnnx(
        model_path: object,
        **kwargs: object,
    ) -> CommandExportResult:
        captured["model_path"] = model_path
        return CommandExportResult(
            output_paths=[param_path, bin_path],
            command=["pnnx", str(model_path)],
            dry_run=True,
        )

    monkeypatch.setattr(export_pass, "export_torchscript", _export_torchscript)
    monkeypatch.setattr(export_pass, "export_ncnn_with_pnnx", _export_ncnn_pnnx)
    context = XQTContext(
        model=torch.nn.Identity().eval(),
        example_inputs=torch.randn(1, 4),
        artifact_dir=str(tmp_path),
        export_targets=[
            ExportTargetConfig(
                format="torchscript",
                output_path=str(first),
                torchscript=TorchScriptExportConfig(runtime_diff=False),
            ),
            ExportTargetConfig(
                format="torchscript",
                output_path=str(second),
                torchscript=TorchScriptExportConfig(runtime_diff=False),
            ),
            ExportTargetConfig(
                format="ncnn",
                output_path=str(param_path),
                ncnn=NCNNExportConfig(
                    converter="pnnx",
                    bin_path=str(bin_path),
                    dry_run=True,
                ),
            ),
        ],
        output_diff_config=OutputDiffConfig(),
    )

    ExportPass().run(context)

    assert context.artifacts["last_torchscript"] == second
    assert captured["model_path"] == second


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
            b"optimized:"
            + source_path.read_bytes()
            + level.encode()
            + b":".join(provider.encode() for provider in providers)
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


def test_optimize_qdq_native_removes_redundant_quantized_round_trip(
    tmp_path: Path,
) -> None:
    onnx = pytest.importorskip("onnx")
    helper = onnx.helper
    tensor_proto = onnx.TensorProto
    numpy_helper = onnx.numpy_helper

    source = tmp_path / "qdq.onnx"
    output = tmp_path / "qdq.optimized.onnx"
    scale = numpy_helper.from_array(
        torch.tensor(0.1, dtype=torch.float32).numpy(), "scale"
    )
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
    scale = numpy_helper.from_array(
        torch.tensor(0.1, dtype=torch.float32).numpy(), "scale"
    )
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
    scale = numpy_helper.from_array(
        torch.tensor(0.1, dtype=torch.float32).numpy(), "scale"
    )
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
            helper.make_node(
                "Identity", ["requantized"], ["uint8_output"], name="identity"
            ),
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
    captured: dict[str, dict[str, object]] = {}

    def fake_export_onnx(*args: object, **kwargs: object) -> ONNXExportResult:
        captured["export"] = dict(kwargs)
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
        captured["optimization"] = dict(kwargs)
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
            onnx=ONNXExportConfig(
                input_names=["features"],
                output_names=["logits"],
                dynamo=False,
                runtime_diff=False,
                optimization=ONNXOptimizationConfig(enabled=True, level="extended"),
            ),
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
    assert captured["export"]["input_names"] == ["features"]
    assert captured["export"]["output_names"] == ["logits"]
    assert captured["export"]["dynamo"] is False
    assert captured["export"]["validate"] is True
    assert captured["optimization"]["level"] == "extended"
    assert captured["optimization"]["native_qdq"] is True
    assert context.manifest is not None
    assert context.manifest.artifacts[0].path == str(optimized)
    assert context.manifest.artifacts[0].metadata["onnx_optimization"][
        "source_path"
    ] == str(source)


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
            onnx=ONNXExportConfig(
                runtime_diff=False,
                optimization=ONNXOptimizationConfig(enabled=True, level="extended"),
            ),
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
            onnx=ONNXExportConfig(runtime_diff=True),
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
