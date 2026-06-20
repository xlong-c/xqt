import pytest

from xqt.benchmark.latency import benchmark_callable
from xqt.benchmark.memory import benchmark_memory
from xqt.benchmark.profiler import profile_callable


def test_benchmark_callable_reports_latency_samples() -> None:
    calls = {"count": 0}

    def fn() -> int:
        calls["count"] += 1
        return calls["count"]

    report = benchmark_callable(
        fn,
        warmup=2,
        iterations=5,
        sync_cuda=False,
        device="cpu",
    )

    assert calls["count"] == 7
    assert report.warmup == 2
    assert report.iterations == 5
    assert len(report.samples_ms) == 5
    assert report.mean_ms >= 0.0
    assert report.p50_ms >= 0.0
    assert report.p90_ms >= report.p50_ms
    assert report.p99_ms >= report.p90_ms
    assert report.to_dict()["iterations"] == 5


@pytest.mark.parametrize(
    ("warmup", "iterations", "message"),
    [
        (-1, 1, "warmup must be non-negative"),
        (0, 0, "iterations must be positive"),
    ],
)
def test_benchmark_callable_rejects_invalid_counts(
    warmup: int,
    iterations: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        benchmark_callable(
            lambda: None,
            warmup=warmup,
            iterations=iterations,
            sync_cuda=False,
            device="cpu",
        )


def test_benchmark_memory_reports_process_rss() -> None:
    calls = {"count": 0}

    def fn() -> int:
        calls["count"] += 1
        return calls["count"]

    report = benchmark_memory(fn, iterations=3, device="cpu", sync_cuda=False)
    data = report.to_dict()

    assert calls["count"] == 3
    assert report.backend == "process_rss"
    assert report.before_bytes >= 0
    assert report.after_bytes >= 0
    assert report.peak_bytes is not None
    assert data["backend"] == "process_rss"


def test_benchmark_helpers_accept_cuda_device_index(monkeypatch) -> None:
    sync_calls = {"count": 0}
    reset_calls = {"count": 0}

    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr(
        "torch.cuda.synchronize",
        lambda: sync_calls.__setitem__("count", sync_calls["count"] + 1),
    )
    monkeypatch.setattr(
        "torch.cuda.reset_peak_memory_stats",
        lambda: reset_calls.__setitem__("count", reset_calls["count"] + 1),
    )
    monkeypatch.setattr("torch.cuda.max_memory_allocated", lambda: 123)
    monkeypatch.setattr("torch.cuda.max_memory_reserved", lambda: 456)

    latency_report = benchmark_callable(
        lambda: None,
        warmup=0,
        iterations=1,
        sync_cuda=True,
        device="cuda:0",
    )
    memory_report = benchmark_memory(
        lambda: None,
        iterations=1,
        device="cuda:0",
        sync_cuda=True,
    )

    assert latency_report.iterations == 1
    assert sync_calls["count"] >= 2
    assert reset_calls["count"] == 1
    assert memory_report.backend == "cuda"
    assert memory_report.cuda_peak_allocated_bytes == 123
    assert memory_report.cuda_peak_reserved_bytes == 456


def test_benchmark_memory_rejects_invalid_iterations() -> None:
    with pytest.raises(ValueError, match="iterations must be positive"):
        benchmark_memory(lambda: None, iterations=0, device="cpu")


def test_profile_callable_reports_operator_rows() -> None:
    calls = {"count": 0}

    def fn() -> object:
        calls["count"] += 1
        return None

    report = profile_callable(
        fn,
        warmup=0,
        active=1,
        repeat=1,
        device="cpu",
        top_k=5,
    )

    assert calls["count"] == 1
    assert report.record_count >= 1
    assert report.kernel_count >= 1
    assert "cpu" in report.activities
    assert report.operators[0].count >= 1
    assert report.to_dict()["operators"]


@pytest.mark.parametrize(
    ("warmup", "active", "repeat", "message"),
    [
        (-1, 1, 1, "warmup must be non-negative"),
        (0, 0, 1, "active must be positive"),
        (0, 1, 0, "repeat must be positive"),
    ],
)
def test_profile_callable_rejects_invalid_counts(
    warmup: int,
    active: int,
    repeat: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        profile_callable(
            lambda: None,
            warmup=warmup,
            active=active,
            repeat=repeat,
            device="cpu",
        )
