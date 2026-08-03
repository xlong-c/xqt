from __future__ import annotations

from typing import Any, Callable

import pytest

from xqt.analysis.compare import TensorDiff
from xqt.auto import (
    evaluate_stage_acceptance,
    rank_stage_benchmark_history,
    run_scheme_search,
    suggest_precision_actions,
    suggest_quant_backends,
)
from xqt.quant.sensitivity import LayerAnalysisRecord
from xqt.workflows import StageAcceptanceConfig


def _make_diff(
    *,
    mean_abs: float,
    max_abs: float,
    relative_error: float | None = None,
    cosine: float | None = None,
) -> TensorDiff:
    return TensorDiff(
        max_abs=max_abs,
        mean_abs=mean_abs,
        mean_squared=mean_abs,
        sqnr_db=None,
        relative_error=relative_error,
        cosine_similarity=cosine,
        correlation=None,
        argmax_mismatch_rate=None,
        allclose=False,
        atol=1e-5,
        rtol=1e-5,
        valid=True,
        message="ok",
    )


def _make_record(
    name: str,
    *,
    mean_abs: float,
    max_abs: float,
    relative_error: float | None = None,
    cosine: float | None = None,
) -> LayerAnalysisRecord:
    return LayerAnalysisRecord(
        name=name,
        module_type="Linear",
        diff=_make_diff(
            mean_abs=mean_abs,
            max_abs=max_abs,
            relative_error=relative_error,
            cosine=cosine,
        ),
        parameter_count=16,
        reference_summary={"shape": [4, 16]},
        candidate_summary={"shape": [4, 16]},
    )


def test_suggest_quant_backends_cpu_ordering() -> None:
    report = suggest_quant_backends(device="cpu", strategy="w8a8_int8")

    assert report.device == "cpu"
    names = [item.backend for item in report.suggestions]
    assert "torchao" in names
    assert "onnxruntime_qdq" in names
    for item in report.suggestions:
        assert not item.requires_cuda
    excluded_names = {item.backend for item in report.excluded}
    assert "bitsandbytes" in excluded_names
    assert all(item.rank >= 1 for item in report.suggestions)


def test_suggest_quant_backends_explicit_backend_precedence() -> None:
    report = suggest_quant_backends(
        device="cpu",
        strategy="w8a8_int8",
        explicit_backend="torchao",
    )

    assert report.explicit_backend == "torchao"
    assert report.suggestions[0].backend == "torchao"
    assert report.suggestions[0].rank == 1
    assert "matches the explicitly requested backend" in report.suggestions[0].reasons
    other_available = [
        item for item in report.excluded if item.status == "available"
    ]
    assert any("takes precedence" in " ".join(item.reasons) for item in other_available)


def test_suggest_quant_backends_cuda_constraint() -> None:
    report = suggest_quant_backends(device="cuda", strategy="w8a8_fp8_e4m3")

    assert report.device == "cuda"
    all_names = {
        item.backend
        for item in (*report.suggestions, *report.excluded)
    }
    assert "torchao" in all_names
    for item in report.suggestions:
        if item.maturity == "planned":
            assert any("status" in reason for reason in item.reasons) or True


def test_suggest_precision_actions_and_explicit_protection() -> None:
    records = [
        _make_record("fc1", mean_abs=1e-2, max_abs=5e-2),
        _make_record("fc2", mean_abs=1e-4, max_abs=1e-3, relative_error=0.2),
        _make_record("fc3", mean_abs=1e-6, max_abs=1e-5),
    ]

    report = suggest_precision_actions(
        records,
        max_mean_abs=1e-3,
        max_relative_error=0.1,
        explicit_keep_high_precision=["fc1"],
    )

    actions = {item.module_name: item.action for item in report.suggestions}
    assert "fc1" not in actions
    assert actions["fc2"] == "skip_quantize"
    assert actions["fc3"] == "quantize"
    assert report.protected_modules == ("fc1",)
    assert "fc1" not in report.policy_delta["keep_high_precision"]
    assert "fc1" not in report.policy_delta["skip_quantize"]
    assert "fc2" in report.policy_delta["skip_quantize"]


def test_suggest_precision_actions_never_mutates_inputs() -> None:
    records = [
        _make_record("fc1", mean_abs=1e-2, max_abs=5e-2),
    ]
    explicit = ["fc1"]
    report = suggest_precision_actions(
        records,
        max_mean_abs=1e-3,
        explicit_keep_high_precision=explicit,
    )

    assert explicit == ["fc1"]
    assert report.protected_modules == ("fc1",)
    assert report.suggestions == ()


def test_rank_stage_benchmark_history_best_and_rejected() -> None:
    stages = [
        {
            "name": "quant_slow",
            "kind": "quant",
            "accepted": False,
            "metrics": {"speedup": 0.8, "mean_abs": 0.01},
        },
        {
            "name": "quant_fast",
            "kind": "quant",
            "accepted": True,
            "metrics": {
                "speedup": 2.0,
                "peak_memory_bytes": 64 * 1024 * 1024,
                "mean_abs": 0.001,
            },
        },
        {
            "name": "prune_ok",
            "kind": "prune",
            "accepted": True,
            "metrics": {"speedup": 1.2, "mean_abs": 0.002},
        },
    ]

    leaderboard = rank_stage_benchmark_history(stages)

    assert leaderboard.best_stage == "quant_fast"
    assert leaderboard.rejected_stages == ("quant_slow",)
    by_name = {item.stage: item for item in leaderboard.rankings}
    assert by_name["quant_fast"].verdict == "best"
    assert by_name["quant_slow"].verdict == "rejected"
    assert by_name["prune_ok"].verdict == "neutral"
    assert by_name["quant_fast"].peak_memory_mb == pytest.approx(64.0)


def test_rank_stage_benchmark_history_memory_metric() -> None:
    stages = [
        {
            "name": "big",
            "kind": "quant",
            "accepted": True,
            "metrics": {"peak_memory_bytes": 256 * 1024 * 1024},
        },
        {
            "name": "small",
            "kind": "quant",
            "accepted": True,
            "metrics": {"peak_memory_bytes": 32 * 1024 * 1024},
        },
    ]

    leaderboard = rank_stage_benchmark_history(stages, metric="memory")

    assert leaderboard.best_stage == "small"


def test_evaluate_stage_acceptance_all_four_dimensions() -> None:
    accept = StageAcceptanceConfig(
        min_speedup=1.2,
        max_mean_abs=0.01,
        max_relative_error=0.05,
        max_memory_mb=128.0,
        max_accuracy_drop=0.02,
    )
    metrics = {
        "speedup": 1.5,
        "mean_abs": 0.005,
        "relative_error": 0.03,
        "peak_memory_bytes": 64 * 1024 * 1024,
        "accuracy_baseline": 0.91,
        "accuracy_candidate": 0.90,
    }

    evaluation = evaluate_stage_acceptance(accept, metrics)

    assert evaluation.accepted
    assert evaluation.message == "ok"
    assert all(check["passed"] for check in evaluation.checks.values())
    assert evaluation.checks["max_accuracy_drop"]["observed"] == pytest.approx(0.01)


def test_evaluate_stage_acceptance_missing_evidence_fails() -> None:
    accept = StageAcceptanceConfig(max_memory_mb=64.0, max_accuracy_drop=0.01)
    evaluation = evaluate_stage_acceptance(accept, {"speedup": 1.0})

    assert not evaluation.accepted
    assert "max_memory_mb" in evaluation.checks
    assert "max_accuracy_drop" in evaluation.checks
    assert "evidence is missing" in evaluation.message


def test_evaluate_stage_acceptance_speedup_from_reference() -> None:
    accept = StageAcceptanceConfig(min_speedup=1.5)
    evaluation = evaluate_stage_acceptance(
        accept,
        {"p50_ms": 2.0},
        reference_benchmark={"p50_ms": 5.0},
    )

    assert evaluation.accepted
    assert evaluation.checks["min_speedup"]["observed"] == pytest.approx(2.5)


def test_run_scheme_search_records_and_limits_attempts() -> None:
    calls: list[str] = []

    def quantize_fn(strategy: str | None, _scheme: Any) -> dict[str, Any]:
        calls.append(str(strategy))
        if str(strategy) == "w4a16_fp4":
            raise RuntimeError("simulated failure")
        return {"mean_abs": 0.01, "speedup": 1.2}

    report = run_scheme_search(
        quantize_fn,
        max_attempts=3,
        seed=7,
    )

    assert len(calls) == 3
    assert report.seed == 7
    assert report.max_attempts == 3
    assert report.reproducible
    statuses = {attempt.strategy: attempt.status for attempt in report.attempts}
    assert statuses["w4a16_fp4"] == "failed"
    assert report.failures[0].error_class == "RuntimeError"
    assert report.recommended


def test_run_scheme_search_reproducible_plan_hash() -> None:
    def quantize_fn(_strategy: str | None, _scheme: Any) -> dict[str, Any]:
        return {"mean_abs": 0.01}

    first = run_scheme_search(quantize_fn, max_attempts=2, seed=3)
    second = run_scheme_search(quantize_fn, max_attempts=2, seed=3)

    assert first.plan_hash == second.plan_hash
    assert first.attempts == second.attempts


def test_run_scheme_search_acceptance_callback() -> None:
    def quantize_fn(_strategy: str | None, _scheme: Any) -> dict[str, Any]:
        return {"mean_abs": 0.5, "speedup": 1.0}

    report = run_scheme_search(
        quantize_fn,
        max_attempts=2,
        seed=0,
        accept_fn=lambda metrics: metrics["mean_abs"] < 0.1,
    )

    assert all(attempt.status == "skipped" for attempt in report.attempts)
    assert report.recommended == ()
