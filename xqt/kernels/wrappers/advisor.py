"""Structured tuning advice helpers for XQT operator optimization."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import torch

from xqt.core.errors import XQTBackendError

from xqt.kernels.ops._impl.gemm_precision import describe_gemm_precision_capability


OperatorFamily = Literal["conv", "linear", "attn", "norm", "fusion", "megakernel"]
BottleneckKind = Literal[
    "unknown",
    "compute_bound",
    "memory_bandwidth",
    "launch_overhead",
    "occupancy",
    "register_pressure",
    "shared_memory",
    "numeric_stability",
]


def _normalize_sm(target_sm: str | int | None) -> str | None:
    if target_sm is None:
        return None
    if isinstance(target_sm, int):
        return f"sm_{target_sm}"
    value = str(target_sm).strip().lower()
    if not value:
        return None
    if value.startswith("sm_"):
        return value
    if value.isdigit():
        return f"sm_{value}"
    raise ValueError(f"unsupported target_sm format: {target_sm}")


def _sm_to_int(target_sm: str | int | None) -> int | None:
    normalized = _normalize_sm(target_sm)
    if normalized is None:
        return None
    return int(normalized.removeprefix("sm_"))


@dataclass(frozen=True)
class PrecisionRecommendation:
    """Recommended precision plan for one operator family on a target SM."""

    operator_family: str
    target_sm: str | None
    recommended_precision: str
    fallback_precision: str
    engine: str
    hardware_native: bool
    rationale: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    validation: dict[str, Any] = field(default_factory=dict)
    alternatives: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "operator_family": self.operator_family,
            "target_sm": self.target_sm,
            "recommended_precision": self.recommended_precision,
            "fallback_precision": self.fallback_precision,
            "engine": self.engine,
            "hardware_native": self.hardware_native,
            "rationale": list(self.rationale),
            "risks": list(self.risks),
            "validation": dict(self.validation),
            "alternatives": list(self.alternatives),
        }


@dataclass(frozen=True)
class ProfilingPlan:
    """Profiler-first diagnosis plan for one operator tuning task."""

    operator_family: str
    target_sm: str | None
    bottleneck: str
    tools: list[str] = field(default_factory=list)
    focus_areas: list[str] = field(default_factory=list)
    report_names: list[str] = field(default_factory=list)
    suggested_actions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "operator_family": self.operator_family,
            "target_sm": self.target_sm,
            "bottleneck": self.bottleneck,
            "tools": list(self.tools),
            "focus_areas": list(self.focus_areas),
            "report_names": list(self.report_names),
            "suggested_actions": list(self.suggested_actions),
        }


def _capability_from_sm(precision: str, sm: int | None) -> dict[str, Any]:
    capability = {
        "precision": precision,
        "device": "cuda",
        "available": False,
        "engine": "none",
        "hardware_native": False,
        "notes": [],
    }
    if sm is None:
        return capability
    if precision == "fp16":
        capability["available"] = True
        capability["engine"] = "triton"
        capability["hardware_native"] = sm >= 70
        if sm >= 70:
            capability["notes"].append("Tensor Core FP16 MMA available")
    elif precision == "bf16":
        capability["available"] = True
        capability["engine"] = "triton"
        capability["hardware_native"] = sm >= 80
        if sm >= 80:
            capability["notes"].append("Tensor Core BF16 MMA available")
    elif precision == "int8":
        capability["available"] = True
        capability["engine"] = "triton"
        capability["hardware_native"] = sm >= 75
        if sm >= 75:
            capability["notes"].append("Tensor Core INT8 MMA available")
    elif precision == "fp8":
        capability["available"] = sm >= 89
        capability["engine"] = "triton" if capability["available"] else "none"
        capability["hardware_native"] = sm >= 89
        if capability["available"]:
            capability["notes"].append("FP8 path available on Ada or newer NVIDIA architectures")
    elif precision == "int4":
        capability["available"] = True
        capability["engine"] = "triton"
        capability["hardware_native"] = False
        capability["notes"].append("INT4 uses unpack plus higher-precision MMA in current XQT paths")
    elif precision in {"mxfp8", "mxfp6", "mxfp4"}:
        capability["available"] = True
        capability["engine"] = "triton"
        capability["hardware_native"] = sm >= 100
        if capability["hardware_native"]:
            capability["notes"].append("Blackwell-class native microscaling support")
        else:
            capability["notes"].append("Microscaling path falls back to emulated block scaling")
    return capability


def _describe_precision(
    precision: str,
    *,
    target_sm: str | int | None = None,
    device: torch.device | None = None,
) -> dict[str, Any]:
    sm_value = _sm_to_int(target_sm)
    if sm_value is not None:
        return _capability_from_sm(precision, sm_value)
    return describe_gemm_precision_capability(
        precision,
        device=device or torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )


def recommend_precision_strategy(
    *,
    operator_family: OperatorFamily,
    target_sm: str | int | None = None,
    prefer_low_precision: bool = False,
    prioritize_accuracy: bool = False,
) -> PrecisionRecommendation:
    """Return a structured precision recommendation for an XQT operator family."""

    sm = _normalize_sm(target_sm)
    sm_value = _sm_to_int(sm)
    if prefer_low_precision and prioritize_accuracy:
        raise ValueError("prefer_low_precision and prioritize_accuracy cannot both be true")

    recommended = "fp16"
    fallback = "fp32"
    rationale: list[str] = []
    risks: list[str] = []
    validation = {"reference_precision": "fp32", "require_numeric_diff": True}
    alternatives: list[str] = []

    if operator_family == "conv":
        recommended = "bf16" if prioritize_accuracy and (sm_value or 0) >= 80 else "fp16"
        fallback = "fp16" if recommended == "bf16" else "fp32"
        rationale.append("Convolution kernels usually reach a stable fast path first in FP16 or BF16.")
        rationale.append("Do not push low-bit conv until memory or bandwidth is proven to dominate.")
        alternatives = ["bf16", "fp16", "int8"]
    elif operator_family == "linear":
        if prefer_low_precision and (sm_value or 0) >= 89:
            recommended = "fp8"
            fallback = "bf16"
        elif prioritize_accuracy:
            recommended = "bf16"
            fallback = "fp16"
        else:
            recommended = "fp16"
            fallback = "bf16"
        rationale.append("Linear and GEMM paths are the strongest current surface for Triton multi-precision tuning.")
        alternatives = ["fp16", "bf16", "fp8", "int8", "int4", "mxfp8", "mxfp4"]
    elif operator_family == "attn":
        recommended = "bf16" if (sm_value or 0) >= 80 else "fp16"
        fallback = "fp16"
        if prefer_low_precision and (sm_value or 0) >= 89:
            alternatives = ["fp8", "bf16", "fp16"]
        else:
            alternatives = ["bf16", "fp16"]
        rationale.append("Attention is usually more sensitive than GEMM, so start with BF16 or FP16.")
        rationale.append("Only move to FP8 after the fused higher-precision path is numerically stable.")
        risks.append("Softmax and accumulation paths are more sensitive to low-precision drift.")
    elif operator_family == "norm":
        recommended = "fp16"
        fallback = "fp16"
        alternatives = ["fp16", "bf16"]
        rationale.append("Current built-in norm fastpaths in XQT are centered on float16 runtime coverage.")
        rationale.append("Use BF16 as the next validation target after the FP16 path is stable on the target SM.")
        risks.append("Low-bit norm paths require tighter numeric drift checks than plain GEMM.")
    elif operator_family in {"fusion", "megakernel"}:
        recommended = "bf16" if prioritize_accuracy and (sm_value or 0) >= 80 else "fp16"
        fallback = "fp16" if recommended == "bf16" else "fp32"
        alternatives = ["fp16", "bf16", "fp8", "int8", "fp4", "nvfp4"]
        rationale.append("Start megakernels in FP16 or BF16, then add low precision only after the fused schedule is healthy.")
        risks.append("Aggressive fusion can inflate registers and collapse occupancy before low precision helps.")
    else:
        raise XQTBackendError(f"Unsupported operator family: {operator_family}")

    capability = _describe_precision(recommended, target_sm=sm)
    if not capability["available"] and fallback != recommended:
        fallback_cap = _describe_precision(fallback, target_sm=sm)
        if fallback_cap["available"]:
            recommended, fallback = fallback, "fp32"
            capability = fallback_cap
            rationale.append("Requested target SM does not make the original low-precision path available, so the recommendation falls back.")

    if recommended in {"fp8", "int8", "int4", "fp4", "nvfp4", "mxfp8", "mxfp6", "mxfp4"}:
        validation["reference_precision"] = "bf16"
        validation["require_scale_validation"] = True
        risks.append("Low precision requires explicit scale and dequant validation.")
    if operator_family in {"attn", "norm", "fusion", "megakernel"}:
        validation["tighten_runtime_validation"] = True

    return PrecisionRecommendation(
        operator_family=operator_family,
        target_sm=sm,
        recommended_precision=recommended,
        fallback_precision=fallback,
        engine=str(capability.get("engine", "none")),
        hardware_native=bool(capability.get("hardware_native", False)),
        rationale=rationale + [str(note) for note in capability.get("notes", [])],
        risks=risks,
        validation=validation,
        alternatives=alternatives,
    )


def build_profiling_plan(
    *,
    operator_family: OperatorFamily,
    target_sm: str | int | None = None,
    bottleneck: BottleneckKind = "unknown",
) -> ProfilingPlan:
    """Return a profiler-first plan for the requested bottleneck."""

    sm = _normalize_sm(target_sm)
    tools = ["benchmark"]
    focus_areas: list[str] = []
    report_names: list[str] = []
    suggested_actions: list[str] = []

    if bottleneck in {"unknown", "launch_overhead"}:
        tools.append("nsys")
        report_names.extend(["cuda_gpu_kern_sum", "cuda_api_sum", "cuda_gpu_trace"])
        focus_areas.extend(["kernel launch count", "CPU launch gaps", "memcpy overlap", "stream synchronization"])
    if bottleneck in {"memory_bandwidth", "occupancy", "register_pressure", "shared_memory", "compute_bound", "numeric_stability"}:
        tools.append("ncu")
    if bottleneck == "memory_bandwidth":
        report_names.extend(["MemoryWorkloadAnalysis", "Roofline"])
        focus_areas.extend(["global load/store efficiency", "L2 traffic", "arithmetic intensity"])
        suggested_actions.extend(["improve coalescing", "fuse producer-consumer edges", "stage reuse in shared memory"])
    elif bottleneck == "occupancy":
        report_names.extend(["Occupancy", "SchedulerStats"])
        focus_areas.extend(["active warps", "eligible warps", "CTA limits"])
        suggested_actions.extend(["reduce register pressure", "retune tile shape", "shrink shared-memory footprint"])
    elif bottleneck == "register_pressure":
        report_names.extend(["Occupancy", "SchedulerStats"])
        focus_areas.extend(["register-limited occupancy", "stall dependency chains"])
        suggested_actions.extend(["simplify epilogue fusion", "lower unroll pressure", "retune block size"])
    elif bottleneck == "shared_memory":
        report_names.extend(["MemoryWorkloadAnalysis", "SchedulerStats"])
        focus_areas.extend(["shared-memory bank conflicts", "short scoreboard stalls"])
        suggested_actions.extend(["change smem layout or swizzle", "reduce smem footprint", "vectorize smem transactions"])
    elif bottleneck == "compute_bound":
        report_names.extend(["Roofline", "SchedulerStats"])
        focus_areas.extend(["tensor-core use", "issue efficiency", "pipeline bubbles"])
        suggested_actions.extend(["move to tensor-core friendly tiles", "reduce branchy hot-loop logic", "adjust num_warps or num_stages"])
    elif bottleneck == "launch_overhead":
        suggested_actions.extend(["increase fusion", "batch tiny kernels", "consider CUDA Graph friendly execution"])
    elif bottleneck == "numeric_stability":
        report_names.extend(["benchmark"])
        focus_areas.extend(["numeric diff against higher precision", "scale loading cost", "reference comparison"])
        suggested_actions.extend(["step back one precision level", "tighten validation thresholds", "separate compile-only from runtime validation"])
    else:
        report_names.extend(["cuda_gpu_kern_sum", "Occupancy", "MemoryWorkloadAnalysis"])
        focus_areas.extend(["top kernels by time", "occupancy", "memory behavior"])
        suggested_actions.extend(["form one bottleneck hypothesis", "change one tuning variable", "remeasure"])

    if operator_family == "attn":
        suggested_actions.append("inspect softmax, scaling, and accumulation sensitivity before pushing lower precision")
    elif operator_family == "megakernel":
        suggested_actions.append("watch for occupancy collapse after adding more fused stages")
    elif operator_family == "linear":
        suggested_actions.append("compare fp16 or bf16 baseline before fp8 or int8 rollout")

    # Preserve order while removing duplicates.
    def _dedupe(items: list[str]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for item in items:
            if item not in seen:
                seen.add(item)
                result.append(item)
        return result

    return ProfilingPlan(
        operator_family=operator_family,
        target_sm=sm,
        bottleneck=bottleneck,
        tools=_dedupe(tools),
        focus_areas=_dedupe(focus_areas),
        report_names=_dedupe(report_names),
        suggested_actions=_dedupe(suggested_actions),
    )


__all__ = [
    "BottleneckKind",
    "OperatorFamily",
    "PrecisionRecommendation",
    "ProfilingPlan",
    "build_profiling_plan",
    "recommend_precision_strategy",
]
