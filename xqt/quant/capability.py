"""Backend capability matrix for XQT quantization."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping, Optional


@dataclass(frozen=True)
class QuantBackendCapability:
    """Static and method-derived capability description for one quantization backend."""

    backend: str
    status: str
    runtime: str
    artifact_kind: str
    methods: tuple[str, ...]
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
            "methods": list(self.methods),
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
        methods=(
            "dynamic_int8",
            "fp8_dynamic",
            "fp8_weight_only",
            "int4_weight_only",
            "int8_weight_only",
        ),
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
        methods=("static_int8",),
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
    "pytorch": QuantBackendCapability(
        backend="pytorch",
        status="planned",
        runtime="pytorch",
        artifact_kind="pytorch_model",
        methods=("awq", "gptq"),
        model_families=("linear_heavy", "llm", "decoder_only_transformer", "vlm_decoder"),
        primary_module_types=("Linear",),
        default_high_precision=_DEFAULT_HIGH_PRECISION,
        preferred_devices=("cuda", "cpu"),
        requires_calibration=True,
        notes=("Planned PyTorch reference path for method-driven weight-only quantization.",),
        limitations=("AWQ/GPTQ are methods, not backends, and are not wired into execution yet.",),
    ),
    "tilelang": QuantBackendCapability(
        backend="tilelang",
        status="planned",
        runtime="pytorch",
        artifact_kind="pytorch_model",
        methods=("awq",),
        model_families=("linear_heavy", "llm", "decoder_only_transformer", "vlm_decoder"),
        primary_module_types=("Linear",),
        default_high_precision=_DEFAULT_HIGH_PRECISION,
        preferred_devices=("cuda",),
        requires_calibration=True,
        requires_cuda=True,
        notes=("Planned TileLang runtime for packed weight-only kernels.",),
        limitations=("TileLang AWQ kernels are not wired into XQT execution yet.",),
    ),
    "bitsandbytes": QuantBackendCapability(
        backend="bitsandbytes",
        status="planned",
        runtime="transformers",
        artifact_kind="hf_runtime_model",
        methods=("int4_weight_only", "int8_weight_only"),
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
    method: Optional[str] = None,
    strategy: Optional[str] = None,
    policy: Mapping[str, Any] | None = None,
) -> QuantBackendCapability:
    """Return a capability description for a quantization backend and method."""

    try:
        base = _BASE_CAPABILITIES[backend]
    except KeyError as exc:
        allowed = ", ".join(sorted(_BASE_CAPABILITIES))
        raise ValueError(f"Unsupported quantization backend: {backend}. Known: {allowed}") from exc

    selected_method = method or (str(policy.get("method")) if policy and policy.get("method") else None)
    if selected_method is not None and selected_method not in base.methods:
        allowed = ", ".join(base.methods) if base.methods else "<none>"
        raise ValueError(
            f"Quantization method '{selected_method}' is not supported by backend "
            f"'{backend}'. Known methods: {allowed}"
        )

    requires_cuda = base.requires_cuda or _strategy_requires_cuda(strategy, policy)
    notes = list(base.notes)
    if backend == "torchao" and requires_cuda:
        notes.append("Configured strategy requires CUDA-capable hardware.")
    if selected_method is not None:
        notes.append(f"Configured quantization method: {selected_method}.")
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
