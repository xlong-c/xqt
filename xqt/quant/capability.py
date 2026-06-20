"""Backend capability matrix for XQT quantization."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping, Optional


@dataclass(frozen=True)
class QuantBackendCapability:
    """Static and strategy-derived capability description for one backend."""

    backend: str
    status: str
    runtime: str
    artifact_kind: str
    model_families: tuple[str, ...]
    primary_module_types: tuple[str, ...]
    candidate_module_types: tuple[str, ...] = ()
    default_high_precision: tuple[str, ...] = ()
    preferred_devices: tuple[str, ...] = ()
    requires_calibration: bool = False
    requires_exportable_graph: bool = False
    requires_cuda: bool = False
    notes: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "status": self.status,
            "runtime": self.runtime,
            "artifact_kind": self.artifact_kind,
            "model_families": list(self.model_families),
            "primary_module_types": list(self.primary_module_types),
            "candidate_module_types": list(self.candidate_module_types),
            "default_high_precision": list(self.default_high_precision),
            "preferred_devices": list(self.preferred_devices),
            "requires_calibration": self.requires_calibration,
            "requires_exportable_graph": self.requires_exportable_graph,
            "requires_cuda": self.requires_cuda,
            "notes": list(self.notes),
            "limitations": list(self.limitations),
        }


_DEFAULT_HIGH_PRECISION = (
    "LayerNorm",
    "RMSNorm",
    "BatchNorm",
    "Embedding",
    "Softmax",
    "lm_head",
    "classifier",
    "head",
    "router",
    "gating",
)

_BASE_CAPABILITIES: dict[str, QuantBackendCapability] = {
    "torchao": QuantBackendCapability(
        backend="torchao",
        status="available",
        runtime="pytorch",
        artifact_kind="pytorch_model",
        model_families=(
            "linear_heavy",
            "vision_transformer",
            "transformer",
            "diffusion_transformer",
        ),
        primary_module_types=("Linear",),
        candidate_module_types=("Conv2d", "MultiheadAttention"),
        default_high_precision=_DEFAULT_HIGH_PRECISION,
        preferred_devices=("cuda", "cpu"),
        notes=(
            "Best current fit is PyTorch runtime quantization for Linear-heavy models.",
            "FP8 or float8 strategies should be treated as CUDA hardware paths.",
        ),
        limitations=(
            "Conv2d and MultiheadAttention are policy candidates, not a stable backend coverage promise.",
            "Deployment artifact remains a PyTorch model rather than a portable quantized graph.",
        ),
    ),
    "onnxruntime_qdq": QuantBackendCapability(
        backend="onnxruntime_qdq",
        status="available",
        runtime="onnxruntime",
        artifact_kind="onnx_qdq",
        model_families=(
            "cnn",
            "resnet",
            "mobile_cnn",
            "exportable_transformer",
        ),
        primary_module_types=("Conv2d", "Linear", "Gemm", "MatMul"),
        candidate_module_types=("Add", "Mul", "Relu"),
        default_high_precision=_DEFAULT_HIGH_PRECISION,
        preferred_devices=("cpu", "cuda"),
        requires_calibration=True,
        requires_exportable_graph=True,
        notes=(
            "Static QDQ quantization needs representative calibration data.",
            "The model or component must be exportable to ONNX before quantization.",
        ),
        limitations=(
            "Dynamic Python control flow and non-exportable custom ops need explicit handling before this backend.",
            "QDQ artifacts are runtime graph artifacts and do not replace the original PyTorch submodule in-place.",
        ),
    ),
    "gptq": QuantBackendCapability(
        backend="gptq",
        status="planned",
        runtime="transformers",
        artifact_kind="hf_weights",
        model_families=("llm", "decoder_only_transformer"),
        primary_module_types=("Linear",),
        default_high_precision=_DEFAULT_HIGH_PRECISION,
        preferred_devices=("cuda",),
        notes=("Planned large-language-model weight-only quantization path.",),
        limitations=("Not wired into XQT execution yet.",),
    ),
    "awq": QuantBackendCapability(
        backend="awq",
        status="planned",
        runtime="transformers",
        artifact_kind="hf_weights",
        model_families=("llm", "decoder_only_transformer", "vlm_decoder"),
        primary_module_types=("Linear",),
        default_high_precision=_DEFAULT_HIGH_PRECISION,
        preferred_devices=("cuda",),
        notes=("Planned activation-aware weight quantization path for LLM-style modules.",),
        limitations=("Not wired into XQT execution yet.",),
    ),
    "bitsandbytes": QuantBackendCapability(
        backend="bitsandbytes",
        status="planned",
        runtime="transformers",
        artifact_kind="hf_runtime_model",
        model_families=("llm", "vlm", "linear_heavy"),
        primary_module_types=("Linear",),
        default_high_precision=_DEFAULT_HIGH_PRECISION,
        preferred_devices=("cuda",),
        notes=("Planned HF runtime quantization path for 8-bit and 4-bit model loading.",),
        limitations=("Not wired into XQT execution yet.",),
    ),
}


def _strategy_requires_cuda(
    strategy: Optional[str],
    policy: Mapping[str, Any] | None,
) -> bool:
    if policy is not None and bool(policy.get("requires_cuda", False)):
        return True
    policy = policy or {}
    strategy_text = str(strategy or policy.get("strategy") or policy.get("dtype") or "").lower()
    return "fp8" in strategy_text or "float8" in strategy_text


def describe_quant_backend_capability(
    backend: str,
    *,
    strategy: Optional[str] = None,
    policy: Mapping[str, Any] | None = None,
) -> QuantBackendCapability:
    """Return a capability description for a quantization backend and strategy."""

    try:
        base = _BASE_CAPABILITIES[backend]
    except KeyError as exc:
        allowed = ", ".join(sorted(_BASE_CAPABILITIES))
        raise ValueError(f"Unsupported quantization backend: {backend}. Known: {allowed}") from exc

    requires_cuda = base.requires_cuda or _strategy_requires_cuda(strategy, policy)
    notes = list(base.notes)
    if backend == "torchao" and requires_cuda:
        notes.append("Configured strategy requires CUDA-capable hardware.")
    return replace(
        base,
        requires_cuda=requires_cuda,
        notes=tuple(notes),
    )


def list_quant_backend_capabilities(
    *,
    include_planned: bool = True,
) -> dict[str, dict[str, Any]]:
    """Return the quantization backend matrix as plain dictionaries."""

    capabilities = {
        name: capability.to_dict()
        for name, capability in sorted(_BASE_CAPABILITIES.items())
        if include_planned or capability.status != "planned"
    }
    return capabilities


__all__ = [
    "QuantBackendCapability",
    "describe_quant_backend_capability",
    "list_quant_backend_capabilities",
]
