import pytest
import onnx
from onnx import TensorProto, helper, numpy_helper

from xqt.core.errors import XQTBackendError
from xqt.export import (
    benchmark_tensorrt_engine,
    build_tensorrt_engine,
    build_trtexec_command,
    execute_tensorrt_engine,
    evaluate_tensorrt_performance_thresholds,
    parse_trtexec_performance,
    summarize_tensorrt_engine_inspector,
)
from xqt.export.tensorrt import _materialize_tensorrt_compatible_onnx


TRTEXEC_OUTPUT = """
[06/17/2026-10:00:00] [I] === Performance summary ===
[06/17/2026-10:00:00] [I] Throughput: 1234.56 qps
[06/17/2026-10:00:00] [I] Latency: min = 0.80 ms, max = 1.20 ms, mean = 0.95 ms, median = 0.94 ms, percentile(90%) = 1.05 ms, percentile(95%) = 1.10 ms, percentile(99%) = 1.18 ms
[06/17/2026-10:00:00] [I] Enqueue Time: min = 12.0 us, max = 30.0 us, mean = 20.0 us, median = 19.0 us, percentile(99%) = 28.0 us
[06/17/2026-10:00:00] [I] H2D Latency: min = 0.01 ms, max = 0.03 ms, mean = 0.02 ms, median = 0.02 ms, percentile(99%) = 0.03 ms
[06/17/2026-10:00:00] [I] D2H Latency: min = 0.02 ms, max = 0.04 ms, mean = 0.03 ms, median = 0.03 ms, percentile(99%) = 0.04 ms
[06/17/2026-10:00:00] [I] GPU Compute Time: min = 0.70 ms, max = 1.00 ms, mean = 0.82 ms, median = 0.81 ms, percentile(99%) = 0.98 ms
[06/17/2026-10:00:00] [I] Total Host Walltime: 3.0 s
[06/17/2026-10:00:00] [I] Total GPU Compute Time: 2500 ms
"""


def test_build_trtexec_command_includes_precision_profiles_and_extra_args() -> None:
    command = build_trtexec_command(
        "model.onnx",
        "model.engine",
        precision="fp16",
        profiles={
            "input": {
                "min": [1, 3, 224, 224],
                "opt": [8, 3, 224, 224],
                "max": [16, 3, 224, 224],
            }
        },
        extra_args=["--verbose"],
    )

    assert command == [
        "trtexec",
        "--onnx=model.onnx",
        "--saveEngine=model.engine",
        "--fp16",
        "--minShapes=input:1x3x224x224",
        "--optShapes=input:8x3x224x224",
        "--maxShapes=input:16x3x224x224",
        "--verbose",
    ]


def test_build_trtexec_command_rejects_invalid_precision_and_profile() -> None:
    with pytest.raises(ValueError, match="precision"):
        build_trtexec_command("model.onnx", "model.engine", precision="int2")

    with pytest.raises(ValueError, match="missing"):
        build_trtexec_command(
            "model.onnx",
            "model.engine",
            profiles={"input": {"min": [1], "opt": [1]}},
        )


def test_parse_trtexec_performance_summary() -> None:
    metrics = parse_trtexec_performance(TRTEXEC_OUTPUT)
    data = metrics.to_dict()

    assert data["throughput_qps"] == 1234.56
    assert data["latency_ms"]["mean"] == 0.95
    assert data["latency_ms"]["p99"] == 1.18
    assert data["enqueue_time_ms"]["mean"] == pytest.approx(0.02)
    assert data["gpu_compute_time_ms"]["p99"] == 0.98
    assert data["total_host_walltime_ms"] == 3000.0
    assert data["total_gpu_compute_time_ms"] == 2500.0


def test_evaluate_tensorrt_performance_thresholds() -> None:
    metrics = parse_trtexec_performance(TRTEXEC_OUTPUT)

    passing = evaluate_tensorrt_performance_thresholds(
        metrics,
        {
            "throughput_qps_min": 1000.0,
            "latency_mean_ms_max": 1.0,
            "gpu_compute_time_p99_ms_max": 1.0,
        },
    )
    failing = evaluate_tensorrt_performance_thresholds(
        metrics,
        {
            "throughput_qps_min": 2000.0,
            "latency_p99_ms_max": 1.0,
            "missing_metric_max": 1.0,
        },
    )

    assert passing.passed is True
    assert [check.passed for check in passing.checks] == [True, True, True]
    assert failing.passed is False
    assert [check.passed for check in failing.checks] == [False, False, False]
    assert failing.checks[-1].value is None


def test_evaluate_tensorrt_performance_thresholds_rejects_unknown_suffix() -> None:
    metrics = parse_trtexec_performance(TRTEXEC_OUTPUT)

    with pytest.raises(ValueError, match="_min or _max"):
        evaluate_tensorrt_performance_thresholds(metrics, {"throughput_qps": 1.0})


def test_build_tensorrt_engine_dry_run_returns_command(tmp_path) -> None:
    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_bytes(b"onnx")

    result = build_tensorrt_engine(
        onnx_path,
        tmp_path / "model.engine",
        precision="fp16",
        dry_run=True,
    )

    assert result.dry_run is True
    assert result.engine_path == tmp_path / "model.engine"
    assert "--fp16" in result.command
    assert result.checksum is None


def test_build_tensorrt_engine_parses_metrics_from_real_run(
    tmp_path,
    monkeypatch,
) -> None:
    onnx_path = tmp_path / "model.onnx"
    engine_path = tmp_path / "model.engine"
    onnx_path.write_bytes(b"onnx")

    def fake_which(command: str) -> str:
        assert command == "trtexec"
        return "/usr/bin/trtexec"

    def fake_run(*args, **kwargs):
        assert args[0][0] == "/usr/bin/trtexec"
        engine_path.write_bytes(b"engine")

        class Completed:
            returncode = 0
            stdout = TRTEXEC_OUTPUT
            stderr = ""

        return Completed()

    monkeypatch.setattr("xqt.export.tensorrt.shutil.which", fake_which)
    monkeypatch.setattr("xqt.export.tensorrt.subprocess.run", fake_run)

    result = build_tensorrt_engine(
        onnx_path,
        engine_path,
        precision="fp16",
        performance_thresholds={
            "throughput_qps_min": 1000.0,
            "latency_p99_ms_max": 1.2,
        },
    )

    assert result.dry_run is False
    assert result.returncode == 0
    assert result.checksum is not None
    assert result.metadata["performance"]["throughput_qps"] == 1234.56
    assert result.metadata["performance_threshold_report"]["passed"] is True


def test_build_tensorrt_engine_rejects_missing_inputs(tmp_path) -> None:
    with pytest.raises(XQTBackendError, match="ONNX file not found"):
        build_tensorrt_engine(tmp_path / "missing.onnx", tmp_path / "model.engine")

    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_bytes(b"onnx")
    with pytest.raises(XQTBackendError, match="trtexec executable not found"):
        build_tensorrt_engine(
            onnx_path,
            tmp_path / "model.engine",
            trtexec_path="definitely_missing_trtexec",
        )


def test_build_tensorrt_engine_python_api_dry_run_records_backend(tmp_path) -> None:
    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_bytes(b"onnx")

    result = build_tensorrt_engine(
        onnx_path,
        tmp_path / "model.engine",
        precision="int8",
        backend="python_api",
        dry_run=True,
        profiles={
            "images": {
                "min": [1, 3, 640, 640],
                "opt": [1, 3, 640, 640],
                "max": [1, 3, 640, 640],
            }
        },
    )

    assert result.dry_run is True
    assert result.metadata["backend"] == "python_api"
    assert result.metadata["precision"] == "int8"
    assert result.metadata["profiles"]["images"]["opt"] == [1, 3, 640, 640]


def test_materialize_tensorrt_compatible_onnx_rewrites_int32_bias_qdq(tmp_path) -> None:
    source = tmp_path / "source.onnx"
    output = tmp_path / "compat.onnx"
    model = helper.make_model(
        helper.make_graph(
            nodes=[
                helper.make_node(
                    "DequantizeLinear",
                    ["weight_q", "weight_scale", "weight_zp"],
                    ["weight_dq"],
                    name="weight_dq_node",
                ),
                helper.make_node(
                    "DequantizeLinear",
                    ["bias_q", "bias_scale", "bias_zp"],
                    ["bias_dq"],
                    name="bias_dq_node",
                ),
                helper.make_node(
                    "Conv",
                    ["input", "weight_dq", "bias_dq"],
                    ["output"],
                    name="conv_node",
                ),
            ],
            name="g",
            inputs=[
                helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1, 1, 1]),
            ],
            outputs=[
                helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1, 1, 1]),
            ],
            initializer=[
                numpy_helper.from_array(__import__("numpy").array([1], dtype="int8"), name="weight_q"),
                numpy_helper.from_array(__import__("numpy").array([0.5], dtype="float32"), name="weight_scale"),
                numpy_helper.from_array(__import__("numpy").array([0], dtype="int8"), name="weight_zp"),
                numpy_helper.from_array(__import__("numpy").array([6], dtype="int32"), name="bias_q"),
                numpy_helper.from_array(__import__("numpy").array([0.25], dtype="float32"), name="bias_scale"),
                numpy_helper.from_array(__import__("numpy").array(2, dtype="int32"), name="bias_zp"),
            ],
        )
    )
    onnx.save(model, str(source))

    compat_path, metadata = _materialize_tensorrt_compatible_onnx(source, output)

    assert compat_path == output
    assert metadata["applied"] is True
    assert metadata["rewritten_bias_count"] == 1
    rewritten = onnx.load(str(output))
    assert [node.name for node in rewritten.graph.node] == ["weight_dq_node", "conv_node"]
    conv = next(node for node in rewritten.graph.node if node.name == "conv_node")
    assert list(conv.input) == ["input", "weight_dq", "bias_dq"]
    init = {tensor.name: tensor for tensor in rewritten.graph.initializer}
    assert "bias_q" not in init
    assert "bias_scale" not in init
    assert "bias_zp" not in init
    bias = numpy_helper.to_array(init["bias_dq"])
    assert bias.tolist() == pytest.approx([1.0])


def test_summarize_tensorrt_engine_inspector_detects_quantization_and_fusion() -> None:
    summary = summarize_tensorrt_engine_inspector(
        {
            "ProfilingVerbosity": "DETAILED",
            "Layers": [
                "images_QuantizeLinear",
                "stem.weight_quantized + /stem/Conv",
                "/stem/Conv_output_0_DequantizeLinear",
                "/ReduceMean",
            ],
            "I/O Tensors": [{"Name": "images"}, {"Name": "predictions"}],
            "Engine Metadata": {"Multi-Device": "Disabled"},
        }
    ).to_dict()

    assert summary["profiling_verbosity"] == "DETAILED"
    assert summary["layer_count"] == 4
    assert summary["quantize_layer_count"] == 1
    assert summary["dequantize_layer_count"] == 1
    assert summary["quantized_conv_count"] == 1
    assert summary["fused_layer_count"] == 1
    assert summary["has_quantization_layers"] is True
    assert summary["has_fused_quantized_conv"] is True
    assert summary["fusion_signatures"] == ["stem.weight_quantized + /stem/Conv"]


def test_build_tensorrt_engine_rejects_unknown_backend(tmp_path) -> None:
    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_bytes(b"onnx")

    with pytest.raises(ValueError, match="backend must be one of trtexec, python_api"):
        build_tensorrt_engine(
            onnx_path,
            tmp_path / "model.engine",
            backend="unknown",
        )


def test_benchmark_tensorrt_engine_rejects_missing_runtime(tmp_path) -> None:
    engine_path = tmp_path / "model.engine"
    engine_path.write_bytes(b"engine")

    with pytest.raises(
        XQTBackendError,
        match="tensorrt is required|Failed to deserialize TensorRT engine",
    ):
        benchmark_tensorrt_engine(
            engine_path,
            input_shapes={"images": [1, 3, 640, 640]},
            iterations=1,
            warmup=0,
        )


def test_execute_tensorrt_engine_rejects_missing_runtime(tmp_path) -> None:
    engine_path = tmp_path / "model.engine"
    engine_path.write_bytes(b"engine")

    with pytest.raises(
        XQTBackendError,
        match="tensorrt is required|Failed to deserialize TensorRT engine",
    ):
        execute_tensorrt_engine(
            engine_path,
            inputs={"images": __import__("torch").rand(1, 3, 640, 640)},
        )
