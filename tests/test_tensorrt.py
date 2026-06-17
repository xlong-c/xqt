import pytest

from xqt.core.errors import XQTBackendError
from xqt.export import (
    build_tensorrt_engine,
    build_trtexec_command,
    evaluate_tensorrt_performance_thresholds,
    parse_trtexec_performance,
)


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
