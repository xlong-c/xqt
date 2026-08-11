"""Engine capability matrix for XQT operator optimization."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, replace
from typing import Any, Optional

import torch

from xqt.core.reporting import OptimizationCapability


def _package_available(package_name: str) -> bool:
    try:
        return importlib.util.find_spec(package_name) is not None
    except ModuleNotFoundError:
        return False


@dataclass(frozen=True)
class OperatorOptimizationEngineCapability:
    """Static and environment-derived capability description for one engine."""

    engine: str
    status: str
    maturity: str
    runtime: str
    exportable: bool
    artifact_kind: str = "pytorch_model"
    requires_cuda: bool = False
    requires_calibration: bool = False
    requires_exportable_graph: bool = False
    available: bool = False
    notes: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()

    def to_optimization_capability(self) -> OptimizationCapability:
        """Project operator engine capability onto the shared optimization schema."""

        return OptimizationCapability(
            kind="operator",
            name=self.engine,
            engine=self.engine,
            status=self.status,
            maturity=self.maturity,
            runtime=self.runtime,
            artifact_kind=self.artifact_kind,
            requires_cuda=self.requires_cuda,
            requires_calibration=self.requires_calibration,
            requires_exportable_graph=self.requires_exportable_graph,
            available=self.available,
            supported=self.available or self.status == "available",
            notes=self.notes,
            limitations=self.limitations,
            metadata={
                "exportable": self.exportable,
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "engine": self.engine,
            "status": self.status,
            "maturity": self.maturity,
            "runtime": self.runtime,
            "exportable": self.exportable,
            "artifact_kind": self.artifact_kind,
            "requires_cuda": self.requires_cuda,
            "requires_calibration": self.requires_calibration,
            "requires_exportable_graph": self.requires_exportable_graph,
            "available": self.available,
            "notes": list(self.notes),
            "limitations": list(self.limitations),
            "optimization_capability": self.to_optimization_capability().to_dict(),
        }


_BASE_CAPABILITIES: dict[str, OperatorOptimizationEngineCapability] = {
    "torch_compile": OperatorOptimizationEngineCapability(
        engine="torch_compile",
        status="available",
        maturity="executable",
        runtime="pytorch",
        exportable=False,
        notes=(
            "Uses torch.compile over the current PyTorch runtime module.",
            "Whole-model and component-level compile are supported in the built-in pass.",
        ),
        limitations=(
            "Dynamic Python control flow or graph breaks can reduce optimization effectiveness.",
        ),
    ),
    "deployment_engine": OperatorOptimizationEngineCapability(
        engine="deployment_engine",
        status="planned",
        maturity="metadata_only",
        runtime="deployment_engine",
        exportable=True,
        artifact_kind="deployment_artifact",
        requires_exportable_graph=True,
        notes=(
            "Represents TensorRT, OpenVINO, or ONNX Runtime deployment fusion rather than PyTorch custom kernels.",
        ),
        limitations=(
            "Built-in executor records capability only and does not rewrite the runtime module.",
        ),
    ),
    "triton": OperatorOptimizationEngineCapability(
        engine="triton",
        status="available",
        maturity="executable",
        runtime="pytorch",
        exportable=False,
        requires_cuda=True,
        notes=(
            "Built-in executor ships limited CUDA-only Triton fused kernels.",
            "Current built-in execution covers direct float16/bfloat16 Linear with zero-materialization transpose strides, explicit prepacked K,N weights, and an explicit fixed-signature CUDA Graph replay mode; standalone float16/bfloat16 RMSNorm; the xqt.nn.FeedForward Triton runtime composition; true W8A8 INT8 GEMM; and packed low-bit dequant GEMM wrappers for INT4/MXFP/NVFP4, all with eager reference fallback metadata where applicable.",
        ),
        limitations=(
            "Current built-in materialization is limited to direct half Linear, rmsnorm, xqt.nn.FeedForward, true W8A8 INT8 Linear, and low-bit linear dequant GEMM patterns; other registered kernel patterns do not yet have a general-purpose operator wrapper.",
            "Direct half Linear prepacked K,N weights duplicate the dense weight storage and are explicit opt-in; transpose-stride is the default zero-extra-memory layout.",
            "Direct half Linear CUDA Graph replay is explicit opt-in, requires a fixed input/layout/parameter contract, returns graph-owned output storage, and is not promoted as an all-shape automatic route.",
            "Current low-bit Triton execution is a composed runtime: packed weights are unpacked and dequantized on device before dispatching to Triton dense GEMM, rather than a single fused tensor-core kernel.",
        ),
    ),
    "tilelang": OperatorOptimizationEngineCapability(
        engine="tilelang",
        status="available",
        maturity="executable",
        runtime="pytorch",
        exportable=False,
        requires_cuda=True,
        notes=(
            "Built-in executor ships operator-family routing for attention, conv, direct FP16/BF16 linear, direct half norm, and dequant/dense linear targets with reference fallback metadata.",
            "Ada-class GPUs can use native runtime fastpaths under the TileLang engine for attention, conv, direct FP16/BF16 linear, direct half norm, and one-time-dequantized dense Linear paths.",
            "Explicit TileLang attention accepts matching float16 or bfloat16 Q/K/V tensors; Ada auto routing remains on native SDPA because the validated BF16 result is shape-dependent.",
            "Explicit TileLang dense Linear accepts matching float16 or bfloat16 activation, weight, bias, and output tensors with FP32 accumulation and partial M/N tiles. On sm_89, BF16 flattened M<=4 and the exact FP16 M<=4,K=N=4096,activation=None signature default to the validated 16x64x32 schedule; Linear auto routing remains unchanged.",
            "Packed FP4/MXFP4/NVFP4 TileLang kernels remain available for explicit pattern selection and future low-bit fused GEMM extensions.",
        ),
        limitations=(
            "Current built-in execution is limited to the attention, conv, linear, norm, and dequant_gemm_epilogue operator families/patterns.",
            "Current CUDA attention execution is limited to matching float16/bfloat16 Q/K/V with dropout_p=0 and seq_kv >= seq_q; bfloat16 head_dim must be divisible by 16. Direct dense Linear requires matching float16/bfloat16 tensors and K divisible by block_k; the sm_89 BF16 decode preset is validated only for M<=4, while the FP16 preset additionally requires K=N=4096 and activation=None. N=11008 and fused activation keep the default schedule. Half norm paths, dequant GEMM constraints, and packed FP4 runtime correctness/performance keep their existing hardware-specific limits.",
        ),
    ),
    "cutile": OperatorOptimizationEngineCapability(
        engine="cutile",
        status="planned",
        maturity="reference_guarded",
        runtime="pytorch",
        exportable=False,
        requires_cuda=True,
        notes=(
            "Reserved for CUDA-only nvvc CuTile Python DSL kernels.",
            "Built-in executor can materialize reference-guarded linear/dequant GEMM inference wrappers, including packed NVFP4 fallback paths.",
        ),
        limitations=(
            "CuTile linear/dequant execution is reference-guarded until real CuTile codegen is validated on target hardware.",
            "CuTile package availability and target architecture must be checked per environment.",
        ),
    ),
    "cutlass": OperatorOptimizationEngineCapability(
        engine="cutlass",
        status="planned",
        maturity="metadata_only",
        runtime="pytorch",
        exportable=False,
        requires_cuda=True,
        notes=("Reserved for CUDA-only CUTLASS Python/CuTe DSL kernels.",),
        limitations=(
            "Built-in executor records metadata and reference fallback only.",
            "CUTLASS Python DSL support is version and architecture sensitive.",
        ),
    ),
    "cute_dsl": OperatorOptimizationEngineCapability(
        engine="cute_dsl",
        status="planned",
        maturity="reference_guarded",
        runtime="pytorch",
        exportable=False,
        requires_cuda=True,
        notes=(
            "Reserved for CUDA-only CUTLASS CuTe DSL kernels through cutlass.cute.",
            "Built-in executor can materialize reference-guarded dense GEMM epilogue inference wrappers for NVFP4 dense-cache bridges.",
        ),
        limitations=(
            "CuTe DSL linear execution is reference-guarded and does not yet consume packed NVFP4 weights directly.",
            "CuTe DSL support is version, Python package, CUDA toolkit, and architecture sensitive.",
        ),
    ),
    "custom_cuda": OperatorOptimizationEngineCapability(
        engine="custom_cuda",
        status="planned",
        maturity="planned",
        runtime="pytorch",
        exportable=False,
        requires_cuda=True,
        notes=("Reserved for optional custom CUDA extensions.",),
        limitations=(
            "Built-in executor does not yet build or load custom CUDA extensions.",
        ),
    ),
}


def describe_operator_engine_capability(
    engine: str,
    *,
    torch_compile_available: Optional[bool] = None,
) -> OperatorOptimizationEngineCapability:
    """Return a capability description for an operator optimization engine."""

    try:
        base = _BASE_CAPABILITIES[engine]
    except KeyError as exc:
        allowed = ", ".join(sorted(_BASE_CAPABILITIES))
        raise ValueError(
            f"Unsupported operator optimization engine: {engine}. Known: {allowed}"
        ) from exc

    available = False
    status = base.status
    notes = list(base.notes)
    if engine == "torch_compile":
        available = (
            bool(torch_compile_available)
            if torch_compile_available is not None
            else hasattr(torch, "compile")
        )
        if not available:
            status = "unavailable"
            notes.append("torch.compile is not available in the current PyTorch build.")
    elif engine == "triton":
        available = _package_available("triton")
        try:
            from .backends.triton import list_triton_kernel_specs

            notes.append(
                "Registered patterns: " + ", ".join(sorted(list_triton_kernel_specs()))
            )
        except Exception:
            pass
    elif engine == "tilelang":
        available = True
        if not _package_available("tilelang"):
            notes.append(
                "tilelang package is not importable; built-in execution is limited to reference fallback."
            )
        else:
            try:
                from .kernels.tilelang._common import (
                    tilelang_runtime_unavailability_reason,
                    tilelang_runtime_usable,
                )

                if not tilelang_runtime_usable():
                    notes.append(
                        tilelang_runtime_unavailability_reason()
                        or "TileLang runtime is unavailable; built-in execution is limited to reference fallback."
                    )
            except Exception:
                pass
        try:
            from .backends.tilelang import list_tilelang_kernel_specs

            notes.append(
                "Registered patterns: "
                + ", ".join(sorted(list_tilelang_kernel_specs()))
            )
        except Exception:
            pass
    elif engine == "cutile":
        try:
            from .backends.cutile import cutile_available, list_cutile_kernel_specs

            available = cutile_available()
            notes.append(
                "Registered patterns: " + ", ".join(sorted(list_cutile_kernel_specs()))
            )
        except Exception:
            available = _package_available("cutile")
    elif engine == "cutlass":
        available = _package_available("cutlass")
        try:
            from .backends.cutlass import list_cutlass_kernel_specs

            notes.append(
                "Registered patterns: " + ", ".join(sorted(list_cutlass_kernel_specs()))
            )
        except Exception:
            pass
    elif engine == "cute_dsl":
        available = _package_available("cutlass.cute")
        try:
            from .backends.cute_dsl import list_cute_dsl_kernel_specs

            notes.append(
                "Registered patterns: "
                + ", ".join(sorted(list_cute_dsl_kernel_specs()))
            )
        except Exception:
            pass
    elif engine == "custom_cuda":
        try:
            from .cuda_extension import describe_custom_cuda_extension_capability

            extension = describe_custom_cuda_extension_capability()
            available = extension.available
            notes.extend(extension.notes)
            notes.append(
                "Registered custom ops: " + ", ".join(extension.registered_ops)
            )
            if not extension.compiled:
                notes.append("Optional nvcc extension module is not compiled.")
        except Exception:
            available = False
    elif engine == "deployment_engine":
        available = True

    return replace(base, status=status, available=available, notes=tuple(notes))


def list_operator_engine_capabilities() -> dict[str, dict[str, Any]]:
    """Return the operator optimization engine matrix as plain dictionaries."""

    return {
        name: describe_operator_engine_capability(name).to_dict()
        for name in sorted(_BASE_CAPABILITIES)
    }


__all__ = [
    "OperatorOptimizationEngineCapability",
    "describe_operator_engine_capability",
    "list_operator_engine_capabilities",
]
