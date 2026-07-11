"""Candidate dataclass, builder, and report summarizer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional

from torch import fx


_PATTERN_TO_BACKEND = {
    "bias_gelu": "triton",
    "swiglu": "triton",
    "rmsnorm": "triton",
    "rmsnorm_residual": "triton",
    "rope": "triton",
    "attention": "tilelang",
    "dequant_gemm": "tilelang",
    "dequant_gemm_epilogue": "tilelang",
    "qdq_epilogue": "deployment_backend",
    "weight_only_matmul_epilogue": "tilelang",
    "fp8_scale_cast_matmul_epilogue": "tilelang",
    "bias_silu": "cutile",
    "gemm_epilogue": "cutlass",
    "grouped_gemm": "cutlass",
}


@dataclass
class OperatorPatternCandidate:
    """One discovered operator optimization candidate pattern."""

    pattern: str
    source: str
    anchor: str
    node_names: list[str]
    shape: list[list[int]]
    dtype: list[str]
    device: list[str]
    estimated_memory_io: Optional[int]
    estimated_kernel_count: int
    recommended_backend: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "pattern": self.pattern,
            "source": self.source,
            "anchor": self.anchor,
            "node_names": list(self.node_names),
            "shape": [list(item) for item in self.shape],
            "dtype": list(self.dtype),
            "device": list(self.device),
            "estimated_memory_io": self.estimated_memory_io,
            "estimated_kernel_count": self.estimated_kernel_count,
            "recommended_backend": self.recommended_backend,
        }


def _tensor_meta_to_lists(node: fx.Node) -> tuple[list[list[int]], list[str], list[str], Optional[int]]:
    tensor_meta = node.meta.get("tensor_meta")
    if tensor_meta is None:
        return [], [], [], None
    metas = tensor_meta if isinstance(tensor_meta, (list, tuple)) else [tensor_meta]
    shapes: list[list[int]] = []
    dtypes: list[str] = []
    devices: list[str] = []
    estimated_memory_io = 0
    for meta in metas:
        shape = [int(dim) for dim in getattr(meta, "shape", [])]
        shapes.append(shape)
        dtypes.append(str(getattr(meta, "dtype", "")))
        device = getattr(meta, "device", None)
        devices.append(str(device) if device is not None else "")
        numel = 1
        for dim in shape:
            numel *= dim
        estimated_memory_io += int(numel)
    return shapes, dtypes, devices, estimated_memory_io


def _node_list_to_names(nodes: Iterable[fx.Node]) -> list[str]:
    return [node.name for node in nodes]


def _build_candidate(
    pattern: str,
    source: str,
    anchor: fx.Node,
    nodes: list[fx.Node],
    *,
    estimated_kernel_count: int,
) -> OperatorPatternCandidate:
    shape, dtype, device, estimated_memory_io = _tensor_meta_to_lists(anchor)
    return OperatorPatternCandidate(
        pattern=pattern,
        source=source,
        anchor=anchor.name,
        node_names=_node_list_to_names(nodes),
        shape=shape,
        dtype=dtype,
        device=device,
        estimated_memory_io=estimated_memory_io,
        estimated_kernel_count=estimated_kernel_count,
        recommended_backend=_PATTERN_TO_BACKEND[pattern],
    )


def summarize_candidate_report(
    candidates: Iterable[OperatorPatternCandidate],
) -> dict[str, Any]:
    """Build a stable candidate report summary."""

    items = [candidate.to_dict() for candidate in candidates]
    return {
        "candidate_count": len(items),
        "patterns": [item["pattern"] for item in items],
        "recommended_backends": [item["recommended_backend"] for item in items],
        "candidates": items,
    }
