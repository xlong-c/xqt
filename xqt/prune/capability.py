"""Backend capability matrix for XQT pruning runtimes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional


@dataclass(frozen=True)
class PruneRuntimeCapability:
    """Static capability description for one pruning method on one runtime."""

    method: str
    runtime: str
    supported: bool
    speedup_verified: bool
    pattern_present: bool
    artifact_kind: str
    notes: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    reason: str = ""
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "runtime": self.runtime,
            "supported": self.supported,
            "speedup_verified": self.speedup_verified,
            "pattern_present": self.pattern_present,
            "artifact_kind": self.artifact_kind,
            "notes": list(self.notes),
            "limitations": list(self.limitations),
            "reason": self.reason,
            "metadata": dict(self.metadata or {}),
        }


def _normalize_device(device: str | None) -> str:
    if device is None:
        return "cpu"
    text = str(device).lower()
    if text.startswith("cuda"):
        return "cuda"
    if text.startswith("xpu"):
        return "xpu"
    return "cpu"


def describe_prune_runtime_capability(
    *,
    method: str,
    device: str | None,
    pattern: Optional[tuple[int, int]] = None,
    block_shape: Optional[tuple[int, int]] = None,
) -> PruneRuntimeCapability:
    """Return runtime capability information for sparse pruning methods."""

    normalized_device = _normalize_device(device)

    if method == "nm_structured":
        pattern_tuple = tuple(pattern or (0, 0))
        if pattern_tuple == (2, 4) and normalized_device == "cuda":
            return PruneRuntimeCapability(
                method=method,
                runtime="cuda_sparse_candidate",
                supported=True,
                speedup_verified=False,
                pattern_present=True,
                artifact_kind="pytorch_model",
                notes=(
                    "2:4 semi-structured sparsity may map to NVIDIA sparse kernels on supported GPUs.",
                ),
                limitations=(
                    "This report does not prove the current model path is using a sparse runtime kernel.",
                ),
                reason="2:4 pattern is a candidate for hardware sparse execution on CUDA.",
                metadata={"pattern": list(pattern_tuple), "device": normalized_device},
            )
        return PruneRuntimeCapability(
            method=method,
            runtime="pytorch_eager",
            supported=False,
            speedup_verified=False,
            pattern_present=True,
            artifact_kind="pytorch_model",
            notes=("Pattern compliance can be reported even when runtime acceleration is unavailable.",),
            limitations=("No sparse runtime backend is configured for this environment.",),
            reason="No sparse runtime backend is configured for this N:M pattern.",
            metadata={"pattern": list(pattern_tuple), "device": normalized_device},
        )

    if method == "block_sparse":
        block = tuple(block_shape or (0, 0))
        return PruneRuntimeCapability(
            method=method,
            runtime="pytorch_eager",
            supported=False,
            speedup_verified=False,
            pattern_present=True,
            artifact_kind="pytorch_model",
            notes=("Block-sparse masks and reports are available.",),
            limitations=(
                "No block-sparse runtime backend is wired into XQT benchmark or export paths yet.",
            ),
            reason="Block-sparse pruning currently reports the pattern but does not provide a verified sparse runtime.",
            metadata={"block_shape": list(block), "device": normalized_device},
        )

    raise ValueError(f"Unsupported prune capability method: {method}")


def prune_runtime_capability_from_report(
    prune_metrics: Mapping[str, Any],
    *,
    device: str | None,
) -> dict[str, Any]:
    """Build a plain capability dictionary from prune metrics."""

    method = str(prune_metrics.get("method", ""))
    if method == "nm_structured":
        capability = describe_prune_runtime_capability(
            method=method,
            device=device,
            pattern=(
                int(prune_metrics.get("pattern_n", 0)),
                int(prune_metrics.get("pattern_m", 0)),
            ),
        )
        return capability.to_dict()
    if method == "block_sparse":
        block_shape = prune_metrics.get("block_shape", [0, 0])
        if isinstance(block_shape, (list, tuple)) and len(block_shape) == 2:
            block = (int(block_shape[0]), int(block_shape[1]))
        else:
            block = (0, 0)
        capability = describe_prune_runtime_capability(
            method=method,
            device=device,
            block_shape=block,
        )
        return capability.to_dict()
    raise ValueError(f"Unsupported prune metrics method for capability: {method}")


__all__ = [
    "PruneRuntimeCapability",
    "describe_prune_runtime_capability",
    "prune_runtime_capability_from_report",
]
