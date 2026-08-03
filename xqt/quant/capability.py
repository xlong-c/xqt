"""Quant backend capability matrix for XQT quantization.

AWQ / GPTQ / SVD (SVDQuant) are quant *methods*, not operator engines.
``tilelang`` is an operator engine only. ``svdquant`` is not a quant backend
name either - use ``backend='pytorch'`` with ``method='svd'`` and WxAy
``strategy`` plus optional ``compute``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping, Optional

from xqt.core.schema import CANONICAL_QUANT_STRATEGIES
from xqt.core.reporting import OptimizationCapability

from .strategy import normalize_quant_compute, normalize_quant_method, normalize_quant_strategy
from .types import QuantizationNature


# ── strategy → nature mapping ─────────────────────────────────────────────
# TRUE  = requested native low-precision MMA compute contract.
# PSEUDO = current XQT route uses dequantized/reference floating-point compute.
# UNKNOWN = strategy/storage alone cannot establish the runtime compute path.
# This is a quantization-time classification. Per-forward runtime metadata is the
# evidence for selected operands, native MMA, and fallback behavior.
# ───────────────────────────────────────────────────────────────────────────
_STRATEGY_NATURE: dict[str, QuantizationNature] = {
    "w4a16_int4": QuantizationNature.PSEUDO,
    "w8a16_int8": QuantizationNature.PSEUDO,
    "w4a16_fp4": QuantizationNature.PSEUDO,
    "w4a16_nvfp4": QuantizationNature.PSEUDO,
    "w4a16_mxfp4": QuantizationNature.PSEUDO,
    "w8a16_mxfp8": QuantizationNature.PSEUDO,
    "w8a16_fp8_e4m3": QuantizationNature.PSEUDO,
    "w8a16_fp8_e5m2": QuantizationNature.PSEUDO,
    "w8a8_int8": QuantizationNature.UNKNOWN,
    "w8a8_fp8_e4m3": QuantizationNature.UNKNOWN,
    "w8a8_fp8_e5m2": QuantizationNature.UNKNOWN,
    "w4a4_int4": QuantizationNature.PSEUDO,
    "w4a4_fp4": QuantizationNature.PSEUDO,
    "w4a4_nvfp4": QuantizationNature.PSEUDO,
    "w4a4_mxfp4": QuantizationNature.PSEUDO,
}

_COMPUTE_TRUE_NATURE = frozenset(
    {
        "w8a8_int8_mma",
        "fp8_mma",
    }
)


# Load-time online weight quant (T14): advertised as planned only - not an
# executable XQT algorithm yet. Distinct from runtime dynamic activation.
_PLANNED_ONLINE_WEIGHT_METHODS = frozenset(
    {
        "online",
        "online_weight_quant",
        "fp8_per_tensor",
        "fp8_per_block",
        "fp8_per_channel",
        "nvfp4_per_token",
    }
)

_DEFAULT_NATURE = QuantizationNature.UNKNOWN


def _resolve_route_for_capability(
    method: Optional[str],
    strategy: Optional[str],
    compute: Optional[str],
) -> Any:
    """Look up the executor route so capability maturity derives from the same
    route table the dispatcher uses (C2 single source of truth)."""

    from xqt.quant import quantizers as _quantizers  # noqa: F401  # register routes

    from .registry import RouteQuery, resolve_quant_route

    return resolve_quant_route(
        RouteQuery(
            backend="pytorch",
            method=(method or "none"),
            strategy=strategy or "",
            compute=compute or "dequant_fp16",
        )
    )


def _resolve_nature(
    strategy: Optional[str],
    policy: Mapping[str, Any] | None,
    *,
    compute: Optional[str] = None,
) -> QuantizationNature:
    if policy is not None:
        explicit = policy.get("nature")
        if isinstance(explicit, str):
            try:
                return QuantizationNature(explicit)
            except ValueError:
                pass
    normalized_compute = normalize_quant_compute(compute) if compute else None
    if normalized_compute is None and policy is not None:
        normalized_compute = normalize_quant_compute(policy.get("compute"))
    if normalized_compute in _COMPUTE_TRUE_NATURE:
        return QuantizationNature.TRUE
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
    maturity: str
    runtime: str
    artifact_kind: str
    methods: tuple[str, ...]
    model_families: tuple[str, ...]
    primary_module_types: tuple[str, ...]
    storage_strategies: tuple[str, ...] = ()
    compute_contracts: tuple[str, ...] = ()
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
            engine=self.backend,
            status=self.status,
            maturity=self.maturity,
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
                "storage_strategies": list(self.storage_strategies),
                "compute_contracts": list(self.compute_contracts),
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "status": self.status,
            "maturity": self.maturity,
            "runtime": self.runtime,
            "artifact_kind": self.artifact_kind,
            "methods": list(self.methods),
            "storage_strategies": list(self.storage_strategies),
            "compute_contracts": list(self.compute_contracts),
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
        maturity="executable",
        runtime="pytorch",
        artifact_kind="pytorch_model",
        methods=(
            "none",
        ),
        storage_strategies=(
            "w8a8_int8",
            "w8a8_fp8_e4m3",
            "w8a8_fp8_e5m2",
            "w8a16_fp8_e4m3",
            "w8a16_fp8_e5m2",
            "w4a16_int4",
            "w8a16_int8",
        ),
        compute_contracts=("dequant_fp16",),
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
        maturity="executable",
        runtime="onnxruntime",
        artifact_kind="onnx_qdq",
        methods=("none",),
        storage_strategies=("w8a8_int8",),
        compute_contracts=("qdq_static", "qdq_dynamic"),
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
        maturity="reference_guarded",
        runtime="pytorch",
        artifact_kind="pytorch_model",
        methods=(
            "none",
            "awq",
            "gptq",
            "svd",
            "convrot",
            "turboquant",
            "moe",
            "moe_weight_only",
        ),
        storage_strategies=CANONICAL_QUANT_STRATEGIES,
        compute_contracts=(
            "dequant_fp16",
            "w8a8_int8_mma",
            "fp8_mma",
            "dequant_gemm",
        ),
        model_families=("linear_heavy", "llm", "decoder_only_transformer", "vlm_decoder"),
        primary_module_types=("Linear",),
        default_high_precision=_DEFAULT_HIGH_PRECISION,
        preferred_devices=("cuda", "cpu"),
        requires_calibration=True,
        notes=(
            "PyTorch quant backend uses method x strategy(WxAy+format) x compute.",
            "awq/gptq/svd/convrot/turboquant are quant methods, not operator engines.",
            "TileLang is only an operator engine; never a quant backend.",
            "compute=w8a8_int8_mma retargets packed W4 residual or W8A8 paths to INT8 MMA.",
        ),
        limitations=(
            "Executable AWQ/GPTQ coverage currently targets W4/W8 weight-only Linear replacement.",
            "Methods need representative calibration inputs for algorithm-level execution.",
            "W4 storage + INT8 MMA is compute retarget, not bit-exact native FP4 MMA.",
        ),
    ),
    "bitsandbytes": QuantBackendCapability(
        backend="bitsandbytes",
        status="planned",
        maturity="planned",
        runtime="transformers",
        artifact_kind="hf_runtime_model",
        methods=("none",),
        storage_strategies=("w4a16_int4", "w8a16_int8"),
        compute_contracts=("dequant_fp16",),
        model_families=("llm", "vlm", "linear_heavy"),
        primary_module_types=("Linear",),
        default_high_precision=_DEFAULT_HIGH_PRECISION,
        preferred_devices=("cuda",),
        notes=("Planned HF runtime quantization path for 8-bit and 4-bit model loading.",),
        limitations=("Not wired into XQT execution yet.",),
    ),
}



def supported_quant_backends() -> tuple[str, ...]:
    """Return the canonical quant backend names (single capability fact source)."""

    return tuple(_BASE_CAPABILITIES)


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
    compute: Optional[str] = None,
    policy: Mapping[str, Any] | None = None,
) -> QuantBackendCapability:
    try:
        base = _BASE_CAPABILITIES[backend]
    except KeyError as exc:
        allowed = ", ".join(sorted(_BASE_CAPABILITIES))
        hint = ""
        backend_key = str(backend).strip().lower()
        if backend_key == "tilelang":
            hint = (
                " tilelang is an operator engine, not a quant backend; "
                "use backend='pytorch' with method/strategy/compute, "
                "then operator stage engine='tilelang'."
            )
        elif backend_key == "svdquant":
            hint = (
                " svdquant is a quant method, not a quant backend; "
                "use backend='pytorch' with method='svd' and strategy='w4a16_*'."
            )
        raise ValueError(
            f"Unsupported quantization backend: {backend}. Known: {allowed}.{hint}"
        ) from exc

    selected_method = method or (
        str(policy.get("method")) if policy and policy.get("method") else None
    )
    if selected_method is not None:
        selected_method = normalize_quant_method(selected_method)
    # T14: load-time online weight quant aliases are planned-only advertisements.
    if selected_method is not None and selected_method in _PLANNED_ONLINE_WEIGHT_METHODS:
        return QuantBackendCapability(
            backend=backend,
            status="planned",
            maturity="planned",
            runtime=base.runtime,
            artifact_kind=base.artifact_kind,
            methods=tuple(sorted(set(base.methods) | set(_PLANNED_ONLINE_WEIGHT_METHODS))),
            storage_strategies=base.storage_strategies,
            compute_contracts=base.compute_contracts,
            model_families=base.model_families,
            primary_module_types=base.primary_module_types,
            default_high_precision=base.default_high_precision,
            preferred_devices=base.preferred_devices,
            requires_cuda=True,
            requires_calibration=False,
            notes=(
                *base.notes,
                "online_weight_quant is load-time weight quantization (not per-forward).",
                "Not an executable XQT algorithm yet; capability advertisement only.",
            ),
            limitations=(
                *base.limitations,
                "Use offline PTQ / external checkpoint load instead of online weight quant.",
            ),
            nature=_DEFAULT_NATURE,
        )
    if selected_method is not None and selected_method not in base.methods:
        allowed = ", ".join(base.methods) if base.methods else "<none>"
        raise ValueError(
            f"Quantization method '{selected_method}' is not supported by backend "
            f"'{backend}'. Known methods: {allowed}"
        )

    normalized_strategy = normalize_quant_strategy(strategy, policy)
    normalized_compute = normalize_quant_compute(compute) if compute else None
    if normalized_compute is None and policy is not None:
        normalized_compute = normalize_quant_compute(policy.get("compute"))

    requires_cuda = base.requires_cuda or _strategy_requires_cuda(strategy, policy)
    resolved_nature = _resolve_nature(
        strategy,
        policy,
        compute=normalized_compute,
    )
    notes = list(base.notes)
    maturity = base.maturity
    if backend == "torchao" and requires_cuda:
        notes.append("Configured strategy requires CUDA-capable hardware.")
    if selected_method is not None:
        notes.append(f"Configured quantization method: {selected_method}.")
    if backend == "pytorch":
        route = _resolve_route_for_capability(
            selected_method,
            normalized_strategy,
            normalized_compute,
        )
        if route is not None:
            maturity = route.maturity
            notes.extend(route.notes)
        else:
            maturity = "planned"
            notes.append(
                "No executable quantization route for this method x strategy x "
                "compute combination; planned capability entry only."
            )
    if resolved_nature == QuantizationNature.PSEUDO:
        notes.append(
            "PSEUDO quantization: this XQT route currently uses dequantized or "
            "reference floating-point compute rather than native low-precision MMA. "
            "Do not infer a measured speedup or a specific dequant dtype from this label."
        )
    elif resolved_nature == QuantizationNature.TRUE:
        notes.append(
            "TRUE quantization: the configured compute contract requests native "
            "low-precision MMA. Confirm selected operands, engine, and fallback "
            "status from per-forward runtime metadata before claiming it executed."
        )
    return replace(
        base,
        maturity=maturity,
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
    "supported_quant_backends",
]
