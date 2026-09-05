"""Persistent Timing Cache and Schedule Resolver for TileLang operators.

Provides a structured cache for storing, querying, and updating optimal kernel
schedules across architectures (SM89, SM90, etc.) and problem shapes.
Aligns with compiler autotuning best practices (e.g. TensorRT timing cache,
TileLang autotuner).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

DEFAULT_TIMING_CACHE_ENV = "XQT_TIMING_CACHE_PATH"
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "xqt"
DEFAULT_CACHE_FILE = DEFAULT_CACHE_DIR / "tilelang_timing_cache.json"


@dataclass(frozen=True)
class TimingCacheKey:
    """Canonical key identifying an operator problem signature."""

    op_type: str
    arch: str
    dtype: str
    shape: tuple[int, ...]
    extra: str = ""

    def to_key_string(self) -> str:
        shape_str = "x".join(str(s) for s in self.shape)
        extra_part = f"#{self.extra}" if self.extra else ""
        return f"{self.op_type}@{self.arch}@{self.dtype}@{shape_str}{extra_part}"

    @classmethod
    def from_key_string(cls, key_str: str) -> "TimingCacheKey":
        extra = ""
        if "#" in key_str:
            key_str, extra = key_str.split("#", 1)
        parts = key_str.split("@")
        if len(parts) != 4:
            raise ValueError(f"invalid timing cache key string: {key_str}")
        op_type, arch, dtype, shape_part = parts
        shape = tuple(int(s) for s in shape_part.split("x")) if shape_part else ()
        return cls(op_type=op_type, arch=arch, dtype=dtype, shape=shape, extra=extra)


@dataclass(frozen=True)
class TimingCacheEntry:
    """A cached schedule with execution diagnostics."""

    schedule: dict[str, Any]
    median_latency_us: float | None = None
    preset_name: str = "custom"
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "schedule": dict(self.schedule),
            "median_latency_us": self.median_latency_us,
            "preset_name": self.preset_name,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TimingCacheEntry":
        return cls(
            schedule=dict(data.get("schedule", {})),
            median_latency_us=(
                float(data["median_latency_us"])
                if data.get("median_latency_us") is not None
                else None
            ),
            preset_name=str(data.get("preset_name", "custom")),
            notes=str(data.get("notes", "")),
        )


class TileLangTimingCache:
    """File-backed cache of optimal TileLang operator execution schedules."""

    def __init__(self, cache_path: Path | str | None = None) -> None:
        if cache_path is not None:
            self._path = Path(cache_path)
        elif DEFAULT_TIMING_CACHE_ENV in os.environ:
            self._path = Path(os.environ[DEFAULT_TIMING_CACHE_ENV])
        else:
            self._path = DEFAULT_CACHE_FILE

        self._entries: dict[str, TimingCacheEntry] = {}
        self._load_from_disk()
        self._seed_builtin_presets()

    @property
    def path(self) -> Path:
        return self._path

    def _load_from_disk(self) -> None:
        if not self._path.is_file():
            return
        try:
            content = self._path.read_text(encoding="utf-8")
            data = json.loads(content)
            if isinstance(data, dict):
                for k, v in data.items():
                    if isinstance(v, dict):
                        self._entries[k] = TimingCacheEntry.from_dict(v)
        except (OSError, json.JSONDecodeError):
            pass

    def save_to_disk(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            payload = {k: v.to_dict() for k, v in self._entries.items()}
            temp_file = self._path.with_suffix(".tmp")
            temp_file.write_text(
                json.dumps(payload, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            temp_file.replace(self._path)
        except OSError:
            pass

    def _seed_builtin_presets(self) -> None:
        """Seed evidence-backed presets into the cache if not already set."""
        for m in (1, 2, 4):
            key_m = TimingCacheKey(
                op_type="linear",
                arch="sm_89",
                dtype="bfloat16",
                shape=(m, 4096, 4096),
                extra="bias=False,act=none",
            )
            if key_m.to_key_string() not in self._entries:
                self._entries[key_m.to_key_string()] = TimingCacheEntry(
                    schedule={
                        "block_m": 16,
                        "block_n": 64,
                        "block_k": 32,
                        "threads": 128,
                        "num_stages": 2,
                    },
                    preset_name="sm89_bf16_decode_m_le_4",
                    notes="Evidence-backed 16x64x32 schedule for decode",
                )

        for m in (1, 2, 4):
            key_fp16 = TimingCacheKey(
                op_type="linear",
                arch="sm_89",
                dtype="float16",
                shape=(m, 4096, 4096),
                extra="bias=False,act=none",
            )
            if key_fp16.to_key_string() not in self._entries:
                self._entries[key_fp16.to_key_string()] = TimingCacheEntry(
                    schedule={
                        "block_m": 16,
                        "block_n": 64,
                        "block_k": 32,
                        "threads": 128,
                        "num_stages": 2,
                    },
                    preset_name="sm89_fp16_decode_m_le_4_n4096",
                    notes="Evidence-backed 16x64x32 schedule for decode",
                )

        for dtype in ("float16", "bfloat16"):
            decode_key = TimingCacheKey(
                op_type="attention",
                arch="sm_89",
                dtype=dtype,
                shape=(1, 32, 1, 1024, 128),
                extra="causal=False",
            )
            if decode_key.to_key_string() not in self._entries:
                self._entries[decode_key.to_key_string()] = TimingCacheEntry(
                    schedule={
                        "block_m": 16,
                        "block_n": 64,
                        "threads": 128,
                        "num_stages": 2,
                    },
                    preset_name="sm89_attn_decode_preset",
                    notes="Optimized 16x64 tile schedule for decode attention",
                )
            prefill_key = TimingCacheKey(
                op_type="attention",
                arch="sm_89",
                dtype=dtype,
                shape=(1, 32, 1024, 1024, 128),
                extra="causal=True",
            )
            if prefill_key.to_key_string() not in self._entries:
                self._entries[prefill_key.to_key_string()] = TimingCacheEntry(
                    schedule={
                        "block_m": 64,
                        "block_n": 64,
                        "threads": 128,
                        "num_stages": 2,
                    },
                    preset_name="sm89_attn_prefill_preset",
                    notes="Standard 64x64 tile schedule for prefill attention",
                )

    def lookup(self, key: TimingCacheKey) -> TimingCacheEntry | None:
        """Query schedule by exact problem signature."""
        key_str = key.to_key_string()
        return self._entries.get(key_str)

    def record(
        self,
        key: TimingCacheKey,
        schedule: Mapping[str, Any],
        *,
        median_latency_us: float | None = None,
        preset_name: str = "custom",
        notes: str = "",
        persist: bool = True,
    ) -> None:
        """Record or update a schedule entry."""
        entry = TimingCacheEntry(
            schedule=dict(schedule),
            median_latency_us=median_latency_us,
            preset_name=preset_name,
            notes=notes,
        )
        self._entries[key.to_key_string()] = entry
        if persist:
            self.save_to_disk()

    def get_candidate_attention_schedules(
        self,
        seq_q: int,
        seq_kv: int,
        head_dim: int,
        arch: str | None = None,
    ) -> list[dict[str, int]]:
        """Generate candidate schedule search space for attention auto-tuning."""
        candidates: list[dict[str, int]] = []
        if seq_q <= 4:
            candidates.extend(
                [
                    {"block_m": 16, "block_n": 64, "threads": 128, "num_stages": 2},
                    {"block_m": 16, "block_n": 128, "threads": 128, "num_stages": 2},
                    {"block_m": 32, "block_n": 64, "threads": 128, "num_stages": 2},
                    {"block_m": 32, "block_n": 128, "threads": 128, "num_stages": 2},
                ]
            )
        elif seq_q <= 256:
            candidates.extend(
                [
                    {"block_m": 64, "block_n": 64, "threads": 128, "num_stages": 2},
                    {"block_m": 64, "block_n": 128, "threads": 128, "num_stages": 2},
                    {"block_m": 32, "block_n": 64, "threads": 128, "num_stages": 2},
                ]
            )
        else:
            candidates.extend(
                [
                    {"block_m": 128, "block_n": 64, "threads": 128, "num_stages": 2},
                    {"block_m": 64, "block_n": 64, "threads": 128, "num_stages": 2},
                    {"block_m": 64, "block_n": 128, "threads": 128, "num_stages": 2},
                ]
            )
        return candidates

    def get_candidate_linear_schedules(
        self,
        m: int,
        n: int,
        k: int,
        arch: str | None = None,
    ) -> list[dict[str, int]]:
        """Generate candidate schedule search space for auto-tuning."""
        candidates: list[dict[str, int]] = []
        if m <= 4:
            candidates.extend(
                [
                    {"block_m": 16, "block_n": 64, "block_k": 32, "threads": 128, "num_stages": 2},
                    {"block_m": 16, "block_n": 64, "block_k": 64, "threads": 128, "num_stages": 2},
                    {"block_m": 32, "block_n": 64, "block_k": 32, "threads": 128, "num_stages": 2},
                    {"block_m": 16, "block_n": 128, "block_k": 32, "threads": 128, "num_stages": 2},
                    {"block_m": 64, "block_n": 64, "block_k": 64, "threads": 128, "num_stages": 2},
                ]
            )
        elif m <= 64:
            candidates.extend(
                [
                    {"block_m": 64, "block_n": 64, "block_k": 32, "threads": 128, "num_stages": 2},
                    {"block_m": 64, "block_n": 64, "block_k": 64, "threads": 128, "num_stages": 2},
                    {"block_m": 64, "block_n": 128, "block_k": 32, "threads": 128, "num_stages": 3},
                    {"block_m": 32, "block_n": 128, "block_k": 32, "threads": 128, "num_stages": 2},
                ]
            )
        else:
            candidates.extend(
                [
                    {"block_m": 128, "block_n": 128, "block_k": 32, "threads": 128, "num_stages": 3},
                    {"block_m": 64, "block_n": 128, "block_k": 64, "threads": 128, "num_stages": 2},
                    {"block_m": 128, "block_n": 64, "block_k": 64, "threads": 128, "num_stages": 2},
                    {"block_m": 64, "block_n": 64, "block_k": 64, "threads": 128, "num_stages": 2},
                ]
            )

        valid = [c for c in candidates if k % c["block_k"] == 0]
        return valid or [{"block_m": 64, "block_n": 64, "block_k": 64, "threads": 128, "num_stages": 2}]


_GLOBAL_TIMING_CACHE: TileLangTimingCache | None = None


def get_tilelang_timing_cache() -> TileLangTimingCache:
    """Get the global TileLang timing cache singleton."""
    global _GLOBAL_TIMING_CACHE
    if _GLOBAL_TIMING_CACHE is None:
        _GLOBAL_TIMING_CACHE = TileLangTimingCache()
    return _GLOBAL_TIMING_CACHE


__all__ = [
    "DEFAULT_TIMING_CACHE_ENV",
    "TileLangTimingCache",
    "TimingCacheEntry",
    "TimingCacheKey",
    "get_tilelang_timing_cache",
]
