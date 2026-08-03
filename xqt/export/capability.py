"""Deployment format capability matrix."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Optional

from xqt.core.reporting import OptimizationCapability, normalize_capability_status


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
    maturity: str = "planned"
    notes: str = ""

    def to_optimization_capability(self) -> OptimizationCapability:
        """Project export capability onto the shared optimization schema."""

        unified_status = normalize_capability_status(self.status)
        available = unified_status in {"available", "adapter"}
        return OptimizationCapability(
            kind="export",
            name=self.format,
            engine=self.format,
            status=unified_status,
            maturity=self.maturity,
            runtime="/".join(self.runtimes) if self.runtimes else "unknown",
            artifact_kind=f"{self.format}_artifact",
            requires_exportable_graph=self.format not in {"torchscript"},
            available=available,
            supported=available,
            precisions=self.precisions,
            notes=(self.notes,) if self.notes else (),
            limitations=(),
            metadata={
                "priority": self.priority,
                "runtimes": list(self.runtimes),
                "dynamic_shapes": self.dynamic_shapes,
                "quantization": self.quantization,
                "sparse_support": self.sparse_support,
                "source_status": self.status,
            },
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["runtimes"] = list(self.runtimes)
        data["precisions"] = list(self.precisions)
        data["optimization_capability"] = self.to_optimization_capability().to_dict()
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
        maturity="executable",
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
        maturity="executable",
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
        maturity="reference_guarded",
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
        maturity="executable",
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
        maturity="reference_guarded",
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
        maturity="reference_guarded",
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
        maturity="reference_guarded",
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
        maturity="reference_guarded",
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
