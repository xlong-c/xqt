"""Capability-based operator engine resolution for Infer (DEBT-003 / G1).

Quant stages declare required_capabilities (+ optional preferred_engines hints).
This module selects an engine candidate; it does not run quantizers.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Iterator, Mapping, Sequence

from xqt.contracts.compute import ComputeConfig

# Default preference when preferred_engines is empty or contains only "auto".
# C10 auto chain: primary fusion DSL (tilelang) → portable (triton) → torch.
# ptx_sm89 / cuda_sm89 stay registered but dispatchable=False (module-local path).
_DEFAULT_INT8_MMA_ORDER: tuple[str, ...] = (
    "tilelang",
    "triton",
    "torch_int_mm",
    "torch",
)

# Quant route primary_kernel tokens → engine registry names (U4 alignment).
_PRIMARY_KERNEL_ENGINE_MAP: dict[str, str] = {
    "tilelang": "tilelang",
    "tilelang_fp4": "tilelang",
    "triton": "triton",
    "torch": "torch",
    "torch_int_mm": "torch_int_mm",
    "w8a8_int8_mma": "tilelang",
    "dequant_fp16": "torch",
    "dequant_gemm_epilogue": "tilelang",
    "onnxruntime_qdq": "torch",
    "torchao": "torch",
    "metadata_only": "torch",
    "ptx_sm89": "ptx_sm89",
    "cuda_sm89": "cuda_sm89",
    "reference": "reference",
}


@dataclass(frozen=True)
class EngineRegistration:
    """Static registration for one operator engine (C3 single source of truth).

    ``dispatchable=False`` marks engines that exist in XQT but are not wired
    into the generic dispatch chain (they need caller-side special handling
    such as prepacked operands or an explicit module API); the label is honest
    about reachability without erasing the capability. ``min_capability`` is
    the minimum SM version (``major * 10 + minor``) the engine's real kernels
    require; ``None`` means no hard floor. ``priority`` orders the auto
    preference chain; lower wins.
    """

    name: str
    provides: frozenset[str]
    min_capability: int | None = None
    priority: int = 100
    dispatchable: bool = True
    maturity: str = "executable"
    status: str = "available"
    runtime: str = "pytorch"
    artifact_kind: str = "pytorch_model"
    exportable: bool = False
    requires_cuda: bool = False
    requires_calibration: bool = False
    requires_exportable_graph: bool = False
    operator_visible: bool = False
    operator_order: int = 999
    convert_visible: bool = False
    convert_order: int = 999
    materializer: str | None = None
    recommended_patterns: frozenset[str] = frozenset()
    notes: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()


_ENGINE_REGISTRY: dict[str, EngineRegistration] = {
    "torch_compile": EngineRegistration(
        name="torch_compile",
        provides=frozenset({"generic", "graph_compile"}),
        priority=80,
        operator_visible=True,
        operator_order=10,
        materializer="torch_compile",
        recommended_patterns=frozenset({"linear_gemm"}),
        notes=("Uses torch.compile over the current PyTorch runtime module.",),
        limitations=(
            "Dynamic Python control flow or graph breaks can reduce optimization effectiveness.",
        ),
    ),
    "deployment_engine": EngineRegistration(
        name="deployment_engine",
        provides=frozenset({"deployment", "graph_export"}),
        priority=95,
        dispatchable=False,
        maturity="metadata_only",
        status="planned",
        runtime="deployment_engine",
        artifact_kind="deployment_artifact",
        exportable=True,
        requires_exportable_graph=True,
        operator_visible=True,
        operator_order=20,
        recommended_patterns=frozenset({"qdq_epilogue"}),
        notes=(
            "Represents deployment fusion through TensorRT, OpenVINO, or ONNX Runtime.",
        ),
        limitations=(
            "Built-in executor records capability only and does not rewrite the runtime module.",
        ),
    ),
    "cuda_sm89": EngineRegistration(
        name="cuda_sm89",
        provides=frozenset({"int8_mma", "true_int8_mma"}),
        min_capability=89,
        priority=20,
        dispatchable=False,
        maturity="executable",
        notes=(
            "CUTLASS-based sm_89 INT8 extension; reached through Int8MmaLinear "
            "explicit engine selection, not the generic auto chain.",
        ),
    ),
    "ptx_sm89": EngineRegistration(
        name="ptx_sm89",
        provides=frozenset({"int8_mma", "true_int8_mma"}),
        min_capability=89,
        priority=21,
        dispatchable=False,
        maturity="executable",
        notes=(
            "Hand-written Ada sm_89 PTX INT8 tensor-core kernel; the current "
            "W8A8 production path via Int8MmaLinear with a prepacked-B cache, "
            "not dispatchable as a stateless GEMM call.",
        ),
    ),
    "native_sm89": EngineRegistration(
        name="native_sm89",
        provides=frozenset({"int8_mma", "true_int8_mma"}),
        min_capability=89,
        priority=22,
        dispatchable=False,
        maturity="executable",
        notes=("Alias of ptx_sm89 kept for name compatibility.",),
    ),
    "tilelang": EngineRegistration(
        name="tilelang",
        provides=frozenset(
            {
                "int8_mma",
                "true_int8_mma",
                "fp4_mma",
                "dequant_gemm_epilogue",
                "w4_storage_int8_mma",
                "hadamard_groupwise",
            }
        ),
        min_capability=80,
        priority=10,
        dispatchable=True,
        maturity="executable",
        requires_cuda=True,
        operator_visible=True,
        operator_order=40,
        convert_visible=True,
        convert_order=30,
        materializer="tilelang",
        recommended_patterns=frozenset(
            {
                "attention",
                "conv",
                "conv3d_1x1x1",
                "dequant_gemm",
                "dequant_gemm_epilogue",
                "weight_only_matmul_epilogue",
                "fp8_scale_cast_matmul_epilogue",
            }
        ),
        notes=(
            "Primary fusion DSL line (C10 auto head): dequant GEMM epilogue, "
            "packed FP4/NVFP4 GEMM, linear_marlin, attention/conv kernels, "
            "groupwise hadamard static quant (C6 online path).",
            "Explicit TileLang dense Linear uses an exact FP16 M<=4,K=N=4096,activation=None schedule where validated.",
        ),
        limitations=(
            "Current built-in execution is limited to the attention, conv, linear, norm, and dequant_gemm_epilogue operator families.",
            "N=11008 and fused activation keep the default schedule.",
        ),
    ),
    "torch_int_mm": EngineRegistration(
        name="torch_int_mm",
        provides=frozenset({"int8_mma", "true_int8_mma"}),
        priority=40,
        dispatchable=True,
        maturity="executable",
    ),
    "cutlass": EngineRegistration(
        name="cutlass",
        provides=frozenset({"int8_mma", "fp4_mma", "int4_mma", "fp16_mma"}),
        min_capability=80,
        priority=50,
        dispatchable=False,
        maturity="metadata_only",
        status="planned",
        requires_cuda=True,
        operator_visible=True,
        operator_order=60,
        recommended_patterns=frozenset({"gemm_epilogue", "grouped_gemm"}),
        notes=(
            "Python adapter carries registry/metadata only; no real compile "
            "path yet. Evaluation line for CUTLASS scaled_mm.",
        ),
    ),
    "cute_dsl": EngineRegistration(
        name="cute_dsl",
        provides=frozenset({"int8_mma", "fp4_mma"}),
        min_capability=90,
        priority=60,
        dispatchable=False,
        maturity="reference_guarded",
        status="planned",
        requires_cuda=True,
        operator_visible=True,
        operator_order=70,
        convert_visible=True,
        convert_order=50,
        materializer="reference_guarded",
        notes=(
            "SM90/SM100 exploration line; gemm_epilogue/grouped_gemm on dense "
            "cache reference only; requires the cutlass.cute runtime.",
        ),
    ),
    "cutile": EngineRegistration(
        name="cutile",
        provides=frozenset({"generic", "fp16_mma"}),
        min_capability=90,
        priority=70,
        dispatchable=False,
        maturity="reference_guarded",
        status="planned",
        requires_cuda=True,
        operator_visible=True,
        operator_order=50,
        convert_visible=True,
        convert_order=40,
        materializer="reference_guarded",
        recommended_patterns=frozenset({"bias_silu"}),
        notes=(
            "Observation line overlapping tilelang's role; kept "
            "reference_guarded pending upstream cuda.tile maturity.",
        ),
    ),
    "torch": EngineRegistration(
        name="torch",
        provides=frozenset({"generic", "fp16_mma", "fp8_mma"}),
        priority=90,
        dispatchable=True,
        maturity="executable",
        convert_visible=True,
        convert_order=10,
        materializer="eager",
    ),
    "triton": EngineRegistration(
        name="triton",
        provides=frozenset(
            {
                "generic",
                "int8_mma",
                "true_int8_mma",
                "fp16_mma",
                "fp4_mma",
                "dequant_gemm_epilogue",
                "hadamard_groupwise",
            }
        ),
        priority=35,
        dispatchable=True,
        maturity="executable",
        requires_cuda=True,
        operator_visible=True,
        operator_order=30,
        convert_visible=True,
        convert_order=20,
        materializer="triton",
        recommended_patterns=frozenset(
            {"bias_gelu", "swiglu", "rmsnorm", "rmsnorm_residual", "rope"}
        ),
        notes=(
            "Portable reference and fallback line; every scheme keeps a "
            "triton/torch reference implementation as the numeric baseline. "
            "Auto chain places triton after tilelang (C10).",
        ),
    ),
    "reference": EngineRegistration(
        name="reference",
        provides=frozenset({"int8_mma", "generic"}),
        priority=100,
        dispatchable=True,
        maturity="reference_guarded",
    ),
    "custom_cuda": EngineRegistration(
        name="custom_cuda",
        provides=frozenset({"generic", "custom_cuda"}),
        min_capability=80,
        priority=75,
        dispatchable=False,
        maturity="planned",
        status="planned",
        requires_cuda=True,
        operator_visible=True,
        operator_order=80,
        notes=("Reserved for optional custom CUDA extensions.",),
        limitations=(
            "Built-in executor does not build or load arbitrary custom CUDA extensions.",
        ),
    ),
}

_OPERATOR_CONTRACT_PATTERNS: Mapping[str, frozenset[str]] = {
    "linear": frozenset(
        {
            "linear",
            "gemm_fp16",
            "gemm_bf16",
            "dense_linear_epilogue",
            "dequant_gemm_epilogue",
            "fp4_packed_dequant_gemm_epilogue",
            "mxfp4_packed_dequant_gemm_epilogue",
            "nvfp4_packed_dequant_gemm_epilogue",
            "gemm_int4_dequant",
            "gemm_mxfp8",
            "gemm_mxfp6",
            "gemm_mxfp4",
            "gemm_nvfp4_packed_dequant",
        }
    ),
    "conv2d": frozenset({"conv"}),
    "conv3d": frozenset({"conv3d_1x1x1"}),
    "feedforward": frozenset({"feedforward"}),
    "layernorm": frozenset({"norm"}),
    "attention": frozenset({"attention"}),
    "transformer_block": frozenset({"attention", "feedforward"}),
}

# Derived capability view; the registry above is the single source of truth.
_ENGINE_PROVIDES: dict[str, frozenset[str]] = {
    name: registration.provides for name, registration in _ENGINE_REGISTRY.items()
}

_ALIASES: dict[str, str] = {
    "native_sm89": "ptx_sm89",
    "auto": "auto",
}


@dataclass(frozen=True, kw_only=True)
class EngineResolveResult:
    """Outcome of one capability-based engine resolve."""

    engine: str
    required_capabilities: tuple[str, ...] = ()
    preferred_engines: tuple[str, ...] = ()
    candidates: tuple[str, ...] = ()
    reason: str = "resolved"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "engine": self.engine,
            "required_capabilities": list(self.required_capabilities),
            "preferred_engines": list(self.preferred_engines),
            "candidates": list(self.candidates),
            "reason": self.reason,
            "metadata": dict(self.metadata),
        }


def normalize_engine_name(engine: str | None) -> str:
    """Normalize engine token; ``None`` / empty → ``auto``."""
    if engine is None:
        return "auto"
    name = str(engine).strip().lower()
    if not name:
        return "auto"
    return _ALIASES.get(name, name)


def get_engine_registration(engine: str | None) -> EngineRegistration | None:
    """Return the registry entry for an engine name (alias-resolved)."""

    return _ENGINE_REGISTRY.get(normalize_engine_name(engine))


def map_primary_kernel_to_engine(primary_kernel: str | None) -> str | None:
    """Map a quant-route ``primary_kernel`` token onto an engine registry name.

    Returns ``None`` when the token is empty. Unknown tokens normalize as-is so
    callers can still attempt ``get_engine_registration``.
    """

    if primary_kernel is None:
        return None
    token = str(primary_kernel).strip().lower()
    if not token:
        return None
    mapped = _PRIMARY_KERNEL_ENGINE_MAP.get(token)
    if mapped is not None:
        return mapped
    return normalize_engine_name(token)


def default_auto_engine_order() -> tuple[str, ...]:
    """Return the documented C10 auto preference chain."""

    return _DEFAULT_INT8_MMA_ORDER


def iter_engine_registrations() -> Iterator[EngineRegistration]:
    """Iterate engine registrations in auto-chain priority order."""

    return iter(
        sorted(
            _ENGINE_REGISTRY.values(),
            key=lambda item: (item.priority, item.name),
        )
    )


def engine_registry_names() -> tuple[str, ...]:
    """Return the canonical implementation-side engine names (DEBT-001)."""

    return tuple(sorted(_ENGINE_REGISTRY))


def operator_engine_names() -> tuple[str, ...]:
    """Return config-visible operator engines in declaration order."""

    registrations = sorted(
        (item for item in _ENGINE_REGISTRY.values() if item.operator_visible),
        key=lambda item: (item.operator_order, item.name),
    )
    return tuple(item.name for item in registrations)


def convert_engine_names() -> tuple[str, ...]:
    """Return engines accepted as ``xqt.convert`` preferences."""

    registrations = sorted(
        (item for item in _ENGINE_REGISTRY.values() if item.convert_visible),
        key=lambda item: (item.convert_order, item.name),
    )
    return tuple(item.name for item in registrations)


def recommended_engine_for_pattern(pattern: str) -> str:
    """Return the registered recommendation for one discovered pattern."""

    normalized = str(pattern).strip()
    candidates = [
        registration
        for registration in _ENGINE_REGISTRY.values()
        if normalized in registration.recommended_patterns
    ]
    if not candidates:
        raise KeyError(f"no engine registration recommends pattern {normalized!r}")
    selected = min(candidates, key=lambda item: (item.priority, item.name))
    return selected.name


def operator_contract_patterns(operator_kind: str) -> frozenset[str] | None:
    """Return materializable patterns for one module contract kind."""

    return _OPERATOR_CONTRACT_PATTERNS.get(str(operator_kind))


def engines_providing(capability: str) -> list[str]:
    """List engines that declare a single capability."""
    cap = str(capability).strip()
    return sorted(
        engine
        for engine, caps in _ENGINE_PROVIDES.items()
        if cap in caps and engine not in {"native_sm89"}
    )


def engines_providing_all(capabilities: Sequence[str]) -> list[str]:
    """List engines that provide every required capability."""
    required = [str(item).strip() for item in capabilities if str(item).strip()]
    if not required:
        return sorted(
            engine
            for engine in _ENGINE_PROVIDES
            if engine not in {"native_sm89"}
        )
    result: list[str] = []
    for engine, caps in _ENGINE_PROVIDES.items():
        if engine in {"native_sm89"}:
            continue
        if all(cap in caps for cap in required):
            result.append(engine)
    return sorted(result)


def _stable_prefer(
    candidates: Sequence[str],
    preferred: Sequence[str],
) -> list[str]:
    preferred_norm = [normalize_engine_name(item) for item in preferred]
    preferred_norm = [item for item in preferred_norm if item != "auto"]
    ordered: list[str] = []
    seen: set[str] = set()
    for name in preferred_norm:
        if name in candidates and name not in seen:
            ordered.append(name)
            seen.add(name)
    for name in candidates:
        if name not in seen:
            ordered.append(name)
            seen.add(name)
    return ordered


def resolve_engine(
    *,
    required_capabilities: Sequence[str] | None = None,
    preferred_engines: Sequence[str] | None = None,
    default_order: Sequence[str] | None = None,
    fallback: str = "torch_int_mm",
) -> EngineResolveResult:
    """Resolve one operator engine from capabilities + optional preference hints.

    Never treats preferred_engines as a hard required_engine constraint:
    engines that lack capabilities are skipped even if preferred.
    """
    caps = tuple(
        str(item).strip()
        for item in (required_capabilities or ())
        if str(item).strip()
    )
    preferred_raw = [
        normalize_engine_name(item) for item in (preferred_engines or ())
    ]
    preferred = tuple(item for item in preferred_raw if item != "auto")
    providing = engines_providing_all(caps)
    if not providing:
        # No matrix hit: still allow explicit preferred if caller only passed hints.
        providing = list(preferred) if preferred else [normalize_engine_name(fallback)]
    order = tuple(default_order) if default_order is not None else _DEFAULT_INT8_MMA_ORDER
    if preferred:
        ranked = _stable_prefer(providing, preferred)
    else:
        ranked = _stable_prefer(providing, order)
    if not ranked:
        ranked = [normalize_engine_name(fallback)]
    chosen = ranked[0]
    return EngineResolveResult(
        engine=chosen,
        required_capabilities=caps,
        preferred_engines=preferred,
        candidates=tuple(ranked),
        reason="preferred" if preferred and chosen in preferred else "default_order",
        metadata={"fallback": normalize_engine_name(fallback)},
    )


def resolve_int8_mma_engine(
    engine: str | None = "auto",
    *,
    preferred_engines: Sequence[str] | None = None,
    fallback: str = "torch_int_mm",
) -> EngineResolveResult:
    """Resolve engine for int8_mma modules (quantizer / forward shared path)."""
    normalized = normalize_engine_name(engine)
    hints: list[str] = list(preferred_engines or [])
    if normalized != "auto":
        # Explicit request is a strong preference, not a schema required_engine key.
        hints = [normalized, *hints]
    return resolve_engine(
        required_capabilities=["int8_mma"],
        preferred_engines=hints,
        default_order=_DEFAULT_INT8_MMA_ORDER,
        fallback=fallback,
    )


def _capabilities_and_preferred(
    compute_config: Mapping[str, Any] | Any | None,
) -> tuple[list[str], list[str]] | None:
    """Extract (required_capabilities, preferred_engines) from a ComputeConfig."""

    if compute_config is None:
        return None
    if hasattr(compute_config, "modules"):
        modules: Sequence[Any] = list(getattr(compute_config, "modules") or [])
        caps = list(getattr(compute_config, "all_required_capabilities")())
        preferred: list[str] = []
        for module in modules:
            preferred.extend(list(getattr(module, "preferred_engines", []) or []))
        return caps, preferred
    if isinstance(compute_config, Mapping):
        parsed = ComputeConfig.from_mapping(compute_config)
        if parsed is None:
            return None
        preferred = []
        for module in parsed.modules:
            preferred.extend(module.preferred_engines)
        return list(parsed.all_required_capabilities()), preferred
    return None


def engine_resolve_from_compute_config(
    compute_config: Mapping[str, Any] | Any | None,
    *,
    fallback: str = "torch_int_mm",
) -> EngineResolveResult:
    """Resolve from a ComputeConfig object or mapping."""

    extracted = _capabilities_and_preferred(compute_config)
    if extracted is None:
        return resolve_engine(fallback=fallback)
    caps, preferred = extracted
    if not caps:
        caps = ["int8_mma"]
    return resolve_engine(
        required_capabilities=caps,
        preferred_engines=preferred,
        fallback=fallback,
    )


def resolve_compute_engine(
    compute_config: Mapping[str, Any] | Any | None,
    *,
    preferred_engines: Sequence[str] | None = None,
    fallback: str = "torch_int_mm",
) -> EngineResolveResult:
    """Unified compute-engine dispatch entry (C3).

    Collects required capabilities from the ComputeConfig (object or mapping),
    merges explicit preference hints, and resolves through the engine
    registry. The result metadata carries the chosen engine's registration
    facts (maturity, dispatchable, min SM capability, priority) so callers can
    report honestly instead of re-deriving them from a second matrix.
    """

    extracted = _capabilities_and_preferred(compute_config)
    caps: list[str] = []
    preferred: list[str] = []
    if extracted is not None:
        caps, preferred = extracted
    preferred.extend(
        normalize_engine_name(item) for item in (preferred_engines or ())
    )
    if not caps:
        caps = ["int8_mma"]
    result = resolve_engine(
        required_capabilities=caps,
        preferred_engines=preferred,
        fallback=fallback,
    )
    registration = get_engine_registration(result.engine)
    if registration is None:
        return result
    metadata = dict(result.metadata)
    metadata["engine_registration"] = {
        "maturity": registration.maturity,
        "dispatchable": registration.dispatchable,
        "min_capability": registration.min_capability,
        "priority": registration.priority,
    }
    return replace(result, metadata=metadata)


__all__ = [
    "EngineRegistration",
    "EngineResolveResult",
    "default_auto_engine_order",
    "convert_engine_names",
    "engine_registry_names",
    "engine_resolve_from_compute_config",
    "engines_providing",
    "engines_providing_all",
    "get_engine_registration",
    "iter_engine_registrations",
    "map_primary_kernel_to_engine",
    "normalize_engine_name",
    "operator_contract_patterns",
    "operator_engine_names",
    "recommended_engine_for_pattern",
    "resolve_compute_engine",
    "resolve_engine",
    "resolve_int8_mma_engine",
]
