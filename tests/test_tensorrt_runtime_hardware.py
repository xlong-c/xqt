from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest
import torch

from xqt.core.schema import (
    ExportTargetConfig,
    OutputDiffConfig,
    TensorRTExportConfig,
    TensorRTRuntimeBenchmarkConfig,
)
from xqt.core.types import XQTContext
from xqt.export.onnx_exporter import export_onnx
from xqt.export.tensorrt import execute_tensorrt_session
from xqt.pipeline.export_pass import ExportPass
from xqt.workflows.stage_specs import (
    DeployRuntimeHandleSpec,
    TensorRTRuntimeHandleConfig,
)


requires_tensorrt_hardware = pytest.mark.skipif(
    not (
        os.environ.get("XQT_RUN_TENSORRT_HARDWARE_TESTS") == "1"
        and torch.cuda.is_available()
        and importlib.util.find_spec("onnx") is not None
        and importlib.util.find_spec("tensorrt") is not None
    ),
    reason=(
        "XQT_RUN_TENSORRT_HARDWARE_TESTS=1, CUDA, ONNX, and TensorRT are required "
        "for the TensorRT hardware smoke test"
    ),
)


@requires_tensorrt_hardware
def test_export_pass_materializes_and_executes_tensorrt_runtime_handle(
    tmp_path: Path,
) -> None:
    """Verify the deploy producer with a real TensorRT engine and session."""

    pytest.importorskip("onnx")
    pytest.importorskip("tensorrt")
    torch.manual_seed(20260710)
    model = (
        torch.nn.Sequential(
            torch.nn.Linear(16, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 8),
        )
        .eval()
        .to("cuda")
    )
    inputs = torch.randn(8, 16, device="cuda")
    with torch.no_grad():
        reference = model(inputs)

    onnx_path = tmp_path / "linear.onnx"
    engine_path = tmp_path / "linear.engine"
    onnx_result = export_onnx(
        model,
        inputs,
        onnx_path,
        opset=17,
        input_names=["input"],
        output_names=["output"],
        dynamo=False,
        validate=True,
    )
    context = XQTContext(
        artifact_dir=str(tmp_path),
        device="cuda:0",
        export_targets=[
            ExportTargetConfig(
                format="tensorrt",
                output_path=str(engine_path),
                tensorrt=TensorRTExportConfig(
                    onnx_path=str(onnx_result.path),
                    backend="python_api",
                    dry_run=False,
                    workspace_mib=256,
                    builder_optimization_level=3,
                    runtime_benchmark=TensorRTRuntimeBenchmarkConfig(
                        enabled=True,
                        input_shapes={"input": [8, 16]},
                        warmup=2,
                        iterations=5,
                        device="cuda:0",
                    ),
                ),
            )
        ],
        output_diff_config=OutputDiffConfig(atol=1e-3, rtol=1e-3),
    )

    ExportPass().run(
        context,
        stage_kind="deploy",
        runtime_handle_request=DeployRuntimeHandleSpec(
            runtime="tensorrt",
            handle_kind="runtime_session",
            materialize=True,
            tensorrt=TensorRTRuntimeHandleConfig(device="cuda:0"),
        ),
    )

    exported = context.metrics["export"]["artifacts"][0]
    handle = context.metrics["export"]["runtime_handle"]
    validation = handle["metadata"]["runtime_validation"]
    assert onnx_result.checked is True
    assert engine_path.is_file()
    assert exported["dry_run"] is False
    assert exported["backend_execution"] == "executed"
    assert validation == {
        "status": "session_created",
        "engine_deserialized": True,
        "execution_context_created": True,
    }
    assert handle["metadata"]["engine_inspector"]["layer_count"] >= 1

    execution = execute_tensorrt_session(handle["handle"], inputs={"input": inputs})
    assert execution.input_shapes == {"input": [8, 16]}
    assert execution.output_shapes == {"output": [8, 8]}
    assert torch.allclose(
        reference,
        execution.output_tensors["output"],
        atol=1e-3,
        rtol=1e-3,
    )

    benchmark = exported["runtime_benchmark"]
    assert benchmark is not None
    assert benchmark["backend"] == "python_api"
    assert benchmark["latency"]["warmup"] == 2
    assert benchmark["latency"]["iterations"] == 5
    assert benchmark["latency"]["mean_ms"] > 0.0
