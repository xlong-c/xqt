"""Deployment format capability matrix."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional


@dataclass(frozen=True)
class ExportCapability:
    """Capability summary for one deployment format."""

    format: str
    priority: str
    runtimes: tuple[str, ...]
    precisions: tuple[str, ...]
    dynamic_shapes: bool
    quantization: bool
    sparse_support: str = "unknown"
    status: str = "planned"
    notes: str = ""

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["runtimes"] = list(self.runtimes)
        data["precisions"] = list(self.precisions)
        return data


DEFAULT_EXPORT_CAPABILITIES: tuple[ExportCapability, ...] = (
    ExportCapability(
        format="torch_export",
        priority="P0",
        runtimes=("pytorch",),
        precisions=("fp32", "fp16", "bf16"),
        dynamic_shapes=True,
        quantization=False,
        status="implemented",
        notes="torch.export ExportedProgram save/load adapter is implemented.",
    ),
    ExportCapability(
        format="onnx",
        priority="P0",
        runtimes=("onnxruntime", "tensorrt", "openvino"),
        precisions=("fp32", "fp16", "bf16", "int8"),
        dynamic_shapes=True,
        quantization=True,
        sparse_support="backend-dependent",
        status="implemented",
        notes="ONNX export, checker, and ONNX Runtime diff are implemented.",
    ),
    ExportCapability(
        format="tensorrt",
        priority="P0",
        runtimes=("tensorrt",),
        precisions=("fp32", "fp16", "bf16", "int8", "fp8"),
        dynamic_shapes=True,
        quantization=True,
        sparse_support="2:4 on supported NVIDIA GPUs",
        status="adapter",
        notes="trtexec command adapter with dry-run and performance threshold support.",
    ),
    ExportCapability(
        format="torchscript",
        priority="P1",
        runtimes=("pytorch", "pnnx"),
        precisions=("fp32", "fp16", "bf16"),
        dynamic_shapes=False,
        quantization=False,
        status="implemented",
        notes="TorchScript trace/script fallback adapter.",
    ),
    ExportCapability(
        format="openvino",
        priority="P1",
        runtimes=("openvino",),
        precisions=("fp32", "fp16", "bf16", "int8"),
        dynamic_shapes=True,
        quantization=True,
        sparse_support="backend-dependent",
        status="adapter",
        notes="Optional dependency adapter.",
    ),
    ExportCapability(
        format="executorch",
        priority="P2",
        runtimes=("executorch",),
        precisions=("fp32", "fp16", "int8"),
        dynamic_shapes=False,
        quantization=True,
        status="adapter",
        notes="Optional ExecuTorch export adapter with dry-run support.",
    ),
    ExportCapability(
        format="ncnn",
        priority="P2",
        runtimes=("ncnn",),
        precisions=("fp32", "fp16", "int8"),
        dynamic_shapes=False,
        quantization=True,
        status="adapter",
        notes="pnnx and ONNX -> ncnn command adapters with dry-run support.",
    ),
    ExportCapability(
        format="mnn",
        priority="P2",
        runtimes=("mnn",),
        precisions=("fp32", "fp16", "int8"),
        dynamic_shapes=False,
        quantization=True,
        status="adapter",
        notes="ONNX -> MNN command adapter with dry-run support.",
    ),
)


def deployment_capability_matrix(
    *,
    priority: Optional[str] = None,
    implemented_only: bool = False,
) -> list[ExportCapability]:
    """Return deployment capability records, optionally filtered."""

    records = list(DEFAULT_EXPORT_CAPABILITIES)
    if priority is not None:
        records = [record for record in records if record.priority == priority]
    if implemented_only:
        records = [
            record
            for record in records
            if record.status in {"implemented", "adapter"}
        ]
    return records


__all__ = [
    "DEFAULT_EXPORT_CAPABILITIES",
    "ExportCapability",
    "deployment_capability_matrix",
]
