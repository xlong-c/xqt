import pytest

from xqt.benchmark.latency import benchmark_callable
from xqt.benchmark.memory import benchmark_memory


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


def test_benchmark_memory_rejects_invalid_iterations() -> None:
    with pytest.raises(ValueError, match="iterations must be positive"):
        benchmark_memory(lambda: None, iterations=0, device="cpu")
