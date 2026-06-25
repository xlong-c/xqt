"""Backend capability matrix for XQT quantization."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping, Optional

from xqt.core.reporting import OptimizationCapability

from .strategy import normalize_quant_strategy
from .types import QuantizationNature


# ── strategy → nature mapping ─────────────────────────────────────────────
# TRUE  = native low-precision MMA (W8A8, K=32), compute speedup expected.
# PSEUDO = storage-only compression, dequant to fp16 before MMA (W8A16, K=16).
# UNKNOWN = not yet classified.
# ───────────────────────────────────────────────────────────────────────────
_STRATEGY_NATURE: dict[str, QuantizationNature] = {
    # fp8 native MMA: W8A8, K=32
    "fp8_dynamic": QuantizationNature.TRUE,
    "float8_dynamic_activation_float8_weight": QuantizationNature.TRUE,
    # fp8 weight-only: W8A16, dequant before mma, K=16
    "fp8_weight_only": QuantizationNature.PSEUDO,
    # int8 weight-only: W8A16, dequant before mma, K=16
    "weight_only_int8": QuantizationNature.PSEUDO,
    "int8_weight_only": QuantizationNature.PSEUDO,
    # int4 weight-only: W4A16, dequant before mma, K=16
    "weight_only_int4": QuantizationNature.PSEUDO,
    "int4_weight_only": QuantizationNature.PSEUDO,
    # fp4 weight-only: W4A16, packed storage with dequant before fp16 MMA
    "fp4_weight_only": QuantizationNature.PSEUDO,
    # dynamic int8: observer-based quantize/dequantize, not native mma
    "dynamic_int8": QuantizationNature.PSEUDO,
    "int8_dynamic_activation_int8_weight": QuantizationNature.PSEUDO,
    # onnxruntime QDQ INT8: runs QDQ ops on CPU integer backend → TRUE if backend hardware supports native int8 mma
    "static_qdq_int8": QuantizationNature.PSEUDO,
    "static_int8": QuantizationNature.PSEUDO,
    # awq / gptq: weight-only packing, dequant to fp16 before compute
    "awq": QuantizationNature.PSEUDO,
    "gptq": QuantizationNature.PSEUDO,
    # svdquant: low-rank fp16 branch + quantized residual
    # PSEUDO in Phase 1 (dequant to fp16 before MMA).
    # Upgraded to TRUE in Phase 2-3 when CuTe W4A4 MMA + SVDQuant fusion kernels ship.
    "svd_fp4": QuantizationNature.PSEUDO,
    "svd_int4": QuantizationNature.PSEUDO,
    "svdquant_fp4": QuantizationNature.PSEUDO,
    "svdquant_int4": QuantizationNature.PSEUDO,
}

_DEFAULT_NATURE = QuantizationNature.UNKNOWN


def _resolve_nature(
    strategy: Optional[str],
    policy: Mapping[str, Any] | None,
) -> QuantizationNature:
    """Resolve quantization nature from strategy name or policy metadata."""
    if policy is not None:
        explicit = policy.get("nature")
        if isinstance(explicit, str):
            try:
                return QuantizationNature(explicit)
            except ValueError:
                pass
    if strategy is not None:
        normalized = normalize_quant_strategy(strategy, policy)
        if normalized is not None:
            return _STRATEGY_NATURE.get(normalized, _DEFAULT_NATURE)
        return _DEFAULT_NATURE
    normalized = normalize_quant_strategy(None, policy)
    if normalized is not None:
        return _STRATEGY_NATURE.get(normalized, _DEFAULT_NATURE)
    return _DEFAULT_NATURE


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
    nature: QuantizationNature = QuantizationNature.UNKNOWN
    notes: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()

    def to_optimization_capability(self) -> OptimizationCapability:
        """Project quantization capability onto the shared optimization schema."""

        return OptimizationCapability(
            kind="quantization",
            name=self.backend,
            backend=self.backend,
            status=self.status,
            runtime=self.runtime,
            artifact_kind=self.artifact_kind,
            requires_cuda=self.requires_cuda,
            requires_calibration=self.requires_calibration,
            requires_exportable_graph=self.requires_exportable_graph,
            available=self.status == "available",
            supported=self.status == "available",
            methods=self.methods,
            model_families=self.model_families,
            target_module_types=self.primary_module_types,
            notes=self.notes,
            limitations=self.limitations,
            metadata={
                "candidate_module_types": list(self.candidate_module_types),
                "default_high_precision": list(self.default_high_precision),
                "preferred_devices": list(self.preferred_devices),
                "nature": self.nature.value,
            },
        )

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
            "nature": self.nature.value,
            "notes": list(self.notes),
            "limitations": list(self.limitations),
            "optimization_capability": self.to_optimization_capability().to_dict(),
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
            "weight_only_int4",
            "weight_only_int8",
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
        methods=("static_qdq_int8",),
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
        status="available",
        runtime="pytorch",
        artifact_kind="pytorch_model",
        methods=("awq", "gptq"),
        model_families=("linear_heavy", "llm", "decoder_only_transformer", "vlm_decoder"),
        primary_module_types=("Linear",),
        default_high_precision=_DEFAULT_HIGH_PRECISION,
        preferred_devices=("cuda", "cpu"),
        requires_calibration=True,
        notes=(
            "PyTorch backend can host reference method-driven weight-only quantization paths.",
        ),
        limitations=(
            "Current executable coverage is limited to reference fp4_weight_only Linear replacement.",
            "Other AWQ/GPTQ method combinations still fall back to planned capability/report only.",
        ),
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
        methods=("weight_only_int4", "weight_only_int8"),
        model_families=("llm", "vlm", "linear_heavy"),
        primary_module_types=("Linear",),
        default_high_precision=_DEFAULT_HIGH_PRECISION,
        preferred_devices=("cuda",),
        notes=("Planned HF runtime quantization path for 8-bit and 4-bit model loading.",),
        limitations=("Not wired into XQT execution yet.",),
    ),
    "svdquant": QuantBackendCapability(
        backend="svdquant",
        status="available",
        runtime="pytorch",
        artifact_kind="pytorch_model",
        methods=("svd_fp4", "svd_int4"),
        model_families=(
            "linear_heavy",
            "transformer",
            "vision_transformer",
            "diffusion_transformer",
            "moe",
            "llm",
            "vlm",
        ),
        primary_module_types=("Linear",),
        default_high_precision=_DEFAULT_HIGH_PRECISION,
        preferred_devices=("cuda", "cpu"),
        requires_cuda=False,
        notes=(
            "SVDQuant decomposes Linear weights via SVD into a low-rank FP16 branch "
            "and a quantized residual (INT4/FP4).",
            "Phase 1 (current): reference dequant+GEMM path. "
            "Phase 2-3: CuTe DSL W4A4 MMA + kernel fusion for TRUE compute speedup.",
        ),
        limitations=(
            "Phase 1 is PSEUDO quantization (storage compression only, dequant to "
            "fp16 before MMA). TRUE INT4 MMA + SVDQuant fusion kernels pending Phase 2-3.",
            "SVD decomposition cost is O(out × in × r) per layer — batch offline, not per-inference.",
        ),
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
    if selected_method is not None:
        selected_method = normalize_quant_strategy(selected_method) or selected_method
    if selected_method is not None and selected_method not in base.methods:
        allowed = ", ".join(base.methods) if base.methods else "<none>"
        raise ValueError(
            f"Quantization method '{selected_method}' is not supported by backend "
            f"'{backend}'. Known methods: {allowed}"
        )

    requires_cuda = base.requires_cuda or _strategy_requires_cuda(strategy, policy)
    resolved_nature = _resolve_nature(strategy, policy)
    notes = list(base.notes)
    if backend == "torchao" and requires_cuda:
        notes.append("Configured strategy requires CUDA-capable hardware.")
    if selected_method is not None:
        notes.append(f"Configured quantization method: {selected_method}.")
    if resolved_nature == QuantizationNature.PSEUDO:
        notes.append(
            "PSEUDO quantization: storage compression only. "
            "Weights dequantized to fp16 before MMA (K=16). "
            "Expect memory bandwidth savings, zero compute speedup."
        )
    elif resolved_nature == QuantizationNature.TRUE:
        notes.append(
            "TRUE quantization: native low-precision MMA (K=32). "
            "Expect compute speedup proportional to element packing density."
        )
    return replace(
        base,
        requires_cuda=requires_cuda,
        nature=resolved_nature,
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
