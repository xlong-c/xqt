"""Universal CUDA Graph caching and runtime scheduler for optimized blocks (XQT-013).

Provides rigorous cache keying, parameter-update invalidation, strict shape/stride/device
guards, LRU memory budget eviction, and observable execution reporting.
"""

from __future__ import annotations

import collections
import hashlib
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import torch
from torch import nn

from xqt.core.errors import XQTBackendError

_CUDA_GRAPH_LOCK = threading.RLock()


def compute_model_param_fingerprint(model: nn.Module) -> str:
    """Compute a lightweight hash over parameter shapes, dtypes, data pointers and in-place versions."""
    hasher = hashlib.sha256()
    for name, param in model.named_parameters():
        if param is not None:
            shape_str = ",".join(str(d) for d in param.shape)
            ptr = param.data_ptr()
            ver = getattr(param, "_version", 0)
            dev = str(param.device)
            dt = str(param.dtype)
            hasher.update(f"{name}:{shape_str}:{dt}:{dev}:{ptr}:{ver}\n".encode("utf-8"))
    return hasher.hexdigest()


@dataclass(frozen=True)
class CUDAGraphCacheKey:
    """Strict timing and execution cache key for CUDA Graph capture."""

    tensor_shapes: tuple[tuple[int, ...], ...]
    tensor_strides: tuple[tuple[int, ...], ...]
    tensor_dtypes: tuple[str, ...]
    tensor_devices: tuple[str, ...]
    requires_grads: tuple[bool, ...]
    param_fingerprint: str
    runtime_flags: tuple[tuple[str, str], ...] = ()

    @classmethod
    def from_inputs(
        cls,
        inputs: Sequence[torch.Tensor],
        model: nn.Module,
        runtime_flags: Mapping[str, Any] | None = None,
    ) -> CUDAGraphCacheKey:
        shapes = tuple(tuple(int(d) for d in t.shape) for t in inputs)
        strides = tuple(tuple(int(s) for s in t.stride()) for t in inputs)
        dtypes = tuple(str(t.dtype) for t in inputs)
        devices = tuple(str(t.device) for t in inputs)
        req_grads = tuple(bool(t.requires_grad) for t in inputs)
        param_fp = compute_model_param_fingerprint(model)
        flags = (
            tuple(sorted((str(k), str(v)) for k, v in runtime_flags.items()))
            if runtime_flags
            else ()
        )
        return cls(
            tensor_shapes=shapes,
            tensor_strides=strides,
            tensor_dtypes=dtypes,
            tensor_devices=devices,
            requires_grads=req_grads,
            param_fingerprint=param_fp,
            runtime_flags=flags,
        )


@dataclass
class CUDAGraphEntry:
    """Captured CUDA Graph instance with static buffers and lifecycle state."""

    key: CUDAGraphCacheKey
    graph: torch.cuda.CUDAGraph
    static_inputs: tuple[torch.Tensor, ...]
    static_output: torch.Tensor
    is_valid: bool = True
    evicted: bool = False

    def replay(self, runtime_args: Sequence[torch.Tensor]) -> torch.Tensor:
        if not self.is_valid or self.evicted:
            raise XQTBackendError(
                "Attempted to replay an invalid or evicted CUDA Graph"
            )
        for static_in, runtime_in in zip(self.static_inputs, runtime_args):
            static_in.copy_(runtime_in)
        self.graph.replay()
        return self.static_output


class CUDAGraphCache:
    """Bounded LRU cache for CUDA Graphs with parameter-invalidation support."""

    def __init__(self, max_entries: int = 4) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be at least 1")
        self.max_entries = max_entries
        self._cache: collections.OrderedDict[CUDAGraphCacheKey, CUDAGraphEntry] = (
            collections.OrderedDict()
        )
        self._lock = threading.RLock()

    def get(self, key: CUDAGraphCacheKey) -> CUDAGraphEntry | None:
        with self._lock:
            entry = self._cache.get(key)
            if entry is not None:
                if not entry.is_valid or entry.evicted:
                    self._cache.pop(key, None)
                    return None
                self._cache.move_to_end(key)
                return entry
            return None

    def put(self, entry: CUDAGraphEntry) -> CUDAGraphEntry | None:
        with self._lock:
            key = entry.key
            if key in self._cache:
                self._cache.move_to_end(key)
                self._cache[key] = entry
                return None

            evicted_entry: CUDAGraphEntry | None = None
            if len(self._cache) >= self.max_entries:
                _evicted_key, evicted_entry = self._cache.popitem(last=False)
                evicted_entry.evicted = True
                evicted_entry.is_valid = False

            self._cache[key] = entry
            return evicted_entry

    def invalidate_stale_parameters(self, current_fingerprint: str) -> list[CUDAGraphCacheKey]:
        with self._lock:
            stale_keys = [
                key
                for key, entry in self._cache.items()
                if key.param_fingerprint != current_fingerprint
            ]
            for key in stale_keys:
                entry = self._cache.pop(key)
                entry.is_valid = False
            return stale_keys

    def clear(self) -> None:
        with self._lock:
            for entry in self._cache.values():
                entry.is_valid = False
                entry.evicted = True
            self._cache.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._cache)


class CUDAGraphBlockRunner(nn.Module):
    """Execution wrapper for an optimized Block using managed CUDA Graph capture & caching."""

    def __init__(
        self,
        module: nn.Module,
        *,
        max_cache_size: int = 4,
        allow_eager_fallback: bool = False,
        warmup_steps: int = 2,
    ) -> None:
        super().__init__()
        self.module = module
        self.allow_eager_fallback = allow_eager_fallback
        self.warmup_steps = warmup_steps
        self.cache = CUDAGraphCache(max_entries=max_cache_size)
        self.last_execution_report: dict[str, Any] = {}

    def forward(self, *args: torch.Tensor) -> torch.Tensor:
        # Check inputs: CUDA Graph requires CUDA tensors
        if not args or not any(isinstance(a, torch.Tensor) for a in args):
            return self.module(*args)

        tensors = tuple(a for a in args if isinstance(a, torch.Tensor))
        if not all(t.is_cuda for t in tensors):
            if self.allow_eager_fallback:
                self.last_execution_report = {
                    "mode": "eager_fallback",
                    "reason": "inputs_not_cuda",
                }
                return self.module(*args)
            raise XQTBackendError(
                "CUDA Graph execution requires all inputs to be on CUDA device"
            )

        current_param_fp = compute_model_param_fingerprint(self.module)
        # Check and invalidate any stale cache entries whose parameter fingerprint no longer matches
        invalidated = self.cache.invalidate_stale_parameters(current_param_fp)

        key = CUDAGraphCacheKey.from_inputs(tensors, self.module)
        entry = self.cache.get(key)

        if entry is not None:
            with _CUDA_GRAPH_LOCK:
                output = entry.replay(tensors)
            self.last_execution_report = {
                "mode": "cuda_graph",
                "status": "cache_hit",
                "invalidated_stale_entries": len(invalidated),
            }
            return output

        # Cache miss: attempt capture
        with _CUDA_GRAPH_LOCK:
            try:
                # Prepare static buffers
                static_args = tuple(
                    torch.empty_strided(
                        size=tuple(int(d) for d in t.shape),
                        stride=tuple(int(s) for s in t.stride()),
                        dtype=t.dtype,
                        device=t.device,
                    )
                    for t in tensors
                )
                for s_arg, r_arg in zip(static_args, tensors):
                    s_arg.copy_(r_arg)

                # Warmup
                dev = tensors[0].device
                with torch.no_grad():
                    for _ in range(max(self.warmup_steps, 0)):
                        self.module(*static_args)
                    torch.cuda.synchronize(dev)

                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        static_output = self.module(*static_args)
                    torch.cuda.synchronize(dev)

                new_entry = CUDAGraphEntry(
                    key=key,
                    graph=graph,
                    static_inputs=static_args,
                    static_output=static_output,
                )
                evicted = self.cache.put(new_entry)

                # Replay with actual input
                output = new_entry.replay(tensors)
                self.last_execution_report = {
                    "mode": "cuda_graph",
                    "status": "captured",
                    "evicted_previous": evicted is not None,
                }
                return output

            except Exception as exc:
                if self.allow_eager_fallback:
                    self.last_execution_report = {
                        "mode": "eager_fallback",
                        "reason": f"capture_failed: {exc}",
                    }
                    return self.module(*args)
                raise XQTBackendError(
                    f"CUDA Graph capture failed: {exc}"
                ) from exc


__all__ = [
    "CUDAGraphBlockRunner",
    "CUDAGraphCache",
    "CUDAGraphCacheKey",
    "CUDAGraphEntry",
    "compute_model_param_fingerprint",
]
