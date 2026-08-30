"""Offline prefill/decode phase latency helpers (not serving TTFT/TPOT)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional

from .latency import LatencyReport, benchmark_callable


@dataclass(frozen=True, slots=True)
class PhaseLatencyReport:
    prefill: LatencyReport
    decode: LatencyReport
    offline_estimate: bool = True
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "offline_estimate": bool(self.offline_estimate),
            "prefill_mean_ms": self.prefill.mean_ms,
            "prefill_p50_ms": self.prefill.p50_ms,
            "prefill_p90_ms": self.prefill.p90_ms,
            "prefill_p99_ms": self.prefill.p99_ms,
            "decode_mean_ms": self.decode.mean_ms,
            "decode_p50_ms": self.decode.p50_ms,
            "decode_p90_ms": self.decode.p90_ms,
            "decode_p99_ms": self.decode.p99_ms,
            "prefill": self.prefill.to_dict(),
            "decode": self.decode.to_dict(),
            "notes": list(self.notes),
        }

    def llm_workload_fields(self) -> dict[str, float | bool]:
        return {
            "offline_estimate": True,
            "prefill_mean_ms": float(self.prefill.mean_ms),
            "decode_mean_ms": float(self.decode.mean_ms),
            "ttft_ms": float(self.prefill.mean_ms),
            "tpot_ms": float(self.decode.mean_ms),
        }


def benchmark_prefill_decode(
    prefill_fn: Callable[[], object],
    decode_fn: Callable[[], object],
    *,
    warmup: int = 2,
    iterations: int = 5,
    sync_cuda: bool = True,
    device: Optional[str] = None,
    notes: tuple[str, ...] | list[str] | None = None,
) -> PhaseLatencyReport:
    prefill = benchmark_callable(
        prefill_fn,
        warmup=warmup,
        iterations=iterations,
        sync_cuda=sync_cuda,
        device=device,
    )
    decode = benchmark_callable(
        decode_fn,
        warmup=warmup,
        iterations=iterations,
        sync_cuda=sync_cuda,
        device=device,
    )
    extra = tuple(str(n) for n in (notes or ()))
    base_notes = (
        "offline_estimate: not serving TTFT/TPOT; host or CUDA-event wall timing only",
    ) + extra
    return PhaseLatencyReport(
        prefill=prefill,
        decode=decode,
        offline_estimate=True,
        notes=base_notes,
    )


def merge_phase_into_metrics(
    metrics: Mapping[str, Any] | None,
    report: PhaseLatencyReport,
) -> dict[str, Any]:
    out = dict(metrics or {})
    out["phase_latency"] = report.to_dict()
    out.update(report.llm_workload_fields())
    return out


__all__ = [
    "PhaseLatencyReport",
    "benchmark_prefill_decode",
    "merge_phase_into_metrics",
]
