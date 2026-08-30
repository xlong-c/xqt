"""Candidate dataclass, builder, and report summarizer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional

from torch import fx

from xqt.kernels.engine_resolve import recommended_engine_for_pattern

_HIGH_FREQUENCY_OPERATOR_GROUPS = {
    "linear_gemm": {"linear_gemm"},
    "dequant_gemm": {
        "dequant_gemm",
        "dequant_gemm_epilogue",
        "weight_only_matmul_epilogue",
        "fp8_scale_cast_matmul_epilogue",
    },
    "attention": {"attention"},
    "norm": {"rmsnorm", "rmsnorm_residual"},
    "rope": {"rope"},
    "activation_epilogue": {
        "bias_gelu",
        "swiglu",
        "qdq_epilogue",
        "dequant_gemm_epilogue",
        "weight_only_matmul_epilogue",
        "fp8_scale_cast_matmul_epilogue",
    },
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

    def shape_signature(self) -> dict[str, Any]:
        return {
            "shape": [list(item) for item in self.shape],
            "dtype": list(self.dtype),
            "device": list(self.device),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "pattern": self.pattern,
            "source": self.source,
            "anchor": self.anchor,
            "node_names": list(self.node_names),
            "shape": [list(item) for item in self.shape],
            "dtype": list(self.dtype),
            "device": list(self.device),
            "shape_signature": self.shape_signature(),
            "estimated_memory_io": self.estimated_memory_io,
            "estimated_kernel_count": self.estimated_kernel_count,
            "recommended_backend": self.recommended_backend,
        }


def _tensor_meta_to_lists(node: fx.Node) -> tuple[list[list[int]], list[str], list[str], Optional[int]]:
    tensor_meta = node.meta.get("tensor_meta")
    if tensor_meta is None and "val" in node.meta:
        tensor_meta = node.meta["val"]
    if tensor_meta is None:
        return [], [], [], None
    if hasattr(tensor_meta, "shape") or hasattr(tensor_meta, "dtype"):
        metas = [tensor_meta]
    else:
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
        recommended_backend=recommended_engine_for_pattern(pattern),
    )


def summarize_candidate_report(
    candidates: Iterable[OperatorPatternCandidate],
) -> dict[str, Any]:
    """Build a stable candidate report summary."""

    items = [candidate.to_dict() for candidate in candidates]
    sources = list(dict.fromkeys(item["source"] for item in items))
    pattern_counts = {
        pattern: sum(1 for item in items if item["pattern"] == pattern)
        for pattern in dict.fromkeys(item["pattern"] for item in items)
    }
    source_counts = {
        source: sum(1 for item in items if item["source"] == source)
        for source in sources
    }
    return {
        "candidate_count": len(items),
        "sources": sources,
        "patterns": [item["pattern"] for item in items],
        "pattern_counts": pattern_counts,
        "source_counts": source_counts,
        "recommended_backends": [item["recommended_backend"] for item in items],
        "shape_signatures": [item["shape_signature"] for item in items],
        "devices": list(
            dict.fromkeys(
                device
                for item in items
                for device in item["device"]
                if device
            )
        ),
        "dtypes": list(
            dict.fromkeys(
                dtype
                for item in items
                for dtype in item["dtype"]
                if dtype
            )
        ),
        "candidates": items,
    }


def operator_pattern_coverage_report(
    candidates: Iterable[OperatorPatternCandidate],
) -> dict[str, Any]:
    """Summarize high-frequency operator-family coverage in scanned candidates."""

    items = [candidate.to_dict() for candidate in candidates]
    pattern_counts = {
        pattern: sum(1 for item in items if item["pattern"] == pattern)
        for pattern in dict.fromkeys(item["pattern"] for item in items)
    }
    groups: dict[str, dict[str, Any]] = {}
    for group_name, patterns in _HIGH_FREQUENCY_OPERATOR_GROUPS.items():
        matched_patterns = [
            pattern
            for pattern in sorted(patterns)
            if pattern_counts.get(pattern, 0) > 0
        ]
        groups[group_name] = {
            "covered": bool(matched_patterns),
            "patterns": matched_patterns,
            "candidate_count": sum(pattern_counts.get(pattern, 0) for pattern in patterns),
        }
    return {
        "groups": groups,
        "covered_groups": [name for name, group in groups.items() if group["covered"]],
        "missing_groups": [name for name, group in groups.items() if not group["covered"]],
    }
