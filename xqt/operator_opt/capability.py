"""Environment projection of the canonical operator engine registry."""

from __future__ import annotations

import importlib
import importlib.util
from dataclasses import dataclass
from typing import Any, Optional

import torch

from xqt.contracts.engine_resolve import (
    EngineRegistration,
    get_engine_registration,
    operator_engine_names,
)
from xqt.core.reporting import OptimizationCapability


def _package_available(package_name: str) -> bool:
    try:
        return importlib.util.find_spec(package_name) is not None
    except ModuleNotFoundError:
        return False


@dataclass(frozen=True)
class OperatorOptimizationEngineCapability:
    """Static registration plus environment availability for one engine."""

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
        """Project the engine onto the shared optimization schema."""

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
            metadata={"exportable": self.exportable},
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


def _registered_pattern_note(engine: str) -> str | None:
    module_functions = {
        "triton": ("xqt.operator_opt.backends.triton", "list_triton_kernel_specs"),
        "tilelang": (
            "xqt.operator_opt.backends.tilelang",
            "list_tilelang_kernel_specs",
        ),
        "cutile": ("xqt.operator_opt.backends.cutile", "list_cutile_kernel_specs"),
        "cutlass": (
            "xqt.operator_opt.backends.cutlass",
            "list_cutlass_kernel_specs",
        ),
        "cute_dsl": (
            "xqt.operator_opt.backends.cute_dsl",
            "list_cute_dsl_kernel_specs",
        ),
    }
    target = module_functions.get(engine)
    if target is None:
        return None
    try:
        module = importlib.import_module(target[0])
        patterns = getattr(module, target[1])()
    except Exception:
        return None
    return "Registered patterns: " + ", ".join(sorted(patterns))


def _environment_availability(
    registration: EngineRegistration,
    *,
    torch_compile_available: Optional[bool],
) -> tuple[bool, list[str]]:
    engine = registration.name
    notes: list[str] = []
    if engine == "torch_compile":
        available = (
            bool(torch_compile_available)
            if torch_compile_available is not None
            else hasattr(torch, "compile")
        )
        if not available:
            notes.append("torch.compile is not available in the current PyTorch build.")
        return available, notes
    if engine == "triton":
        return _package_available("triton"), notes
    if engine == "tilelang":
        if not _package_available("tilelang"):
            notes.append(
                "tilelang package is not importable; execution is limited to reference fallback."
            )
            return True, notes
        try:
            from .kernels.tilelang._common import (
                tilelang_runtime_unavailability_reason,
                tilelang_runtime_usable,
            )

            if not tilelang_runtime_usable():
                notes.append(
                    tilelang_runtime_unavailability_reason()
                    or "TileLang runtime is unavailable; execution is limited to reference fallback."
                )
        except Exception:
            pass
        return True, notes
    if engine == "cutile":
        try:
            from .backends.cutile import cutile_available

            return cutile_available(), notes
        except Exception:
            return _package_available("cutile"), notes
    if engine == "cutlass":
        return _package_available("cutlass"), notes
    if engine == "cute_dsl":
        available = _package_available("cutlass.cute")
        if not available:
            notes.append("cutlass.cute runtime is not importable.")
        return available, notes
    if engine == "custom_cuda":
        try:
            from .cuda_extension import describe_custom_cuda_extension_capability

            extension = describe_custom_cuda_extension_capability()
            notes.extend(extension.notes)
            notes.append("Registered custom ops: " + ", ".join(extension.registered_ops))
            if not extension.compiled:
                notes.append("Optional nvcc extension module is not compiled.")
            return extension.available, notes
        except Exception:
            return False, notes
    if engine == "deployment_engine":
        return True, notes
    return False, notes


def _capability_from_registration(
    registration: EngineRegistration,
    *,
    available: bool,
    notes: list[str],
) -> OperatorOptimizationEngineCapability:
    status = registration.status
    if registration.name == "torch_compile" and not available:
        status = "unavailable"
    return OperatorOptimizationEngineCapability(
        engine=registration.name,
        status=status,
        maturity=registration.maturity,
        runtime=registration.runtime,
        exportable=registration.exportable,
        artifact_kind=registration.artifact_kind,
        requires_cuda=registration.requires_cuda,
        requires_calibration=registration.requires_calibration,
        requires_exportable_graph=registration.requires_exportable_graph,
        available=available,
        notes=tuple([*registration.notes, *notes]),
        limitations=registration.limitations,
    )


def describe_operator_engine_capability(
    engine: str,
    *,
    torch_compile_available: Optional[bool] = None,
) -> OperatorOptimizationEngineCapability:
    """Project one config-visible engine registration into the environment."""

    registration = get_engine_registration(engine)
    if registration is None or not registration.operator_visible:
        allowed = ", ".join(operator_engine_names())
        raise ValueError(
            f"Unsupported operator optimization engine: {engine}. Known: {allowed}"
        )
    available, environment_notes = _environment_availability(
        registration,
        torch_compile_available=torch_compile_available,
    )
    pattern_note = _registered_pattern_note(registration.name)
    if pattern_note is not None:
        environment_notes.append(pattern_note)
    return _capability_from_registration(
        registration,
        available=available,
        notes=environment_notes,
    )


def list_operator_engine_capabilities() -> dict[str, dict[str, Any]]:
    """Return the operator engine matrix as plain dictionaries."""

    return {
        name: describe_operator_engine_capability(name).to_dict()
        for name in operator_engine_names()
    }


__all__ = [
    "OperatorOptimizationEngineCapability",
    "describe_operator_engine_capability",
    "list_operator_engine_capabilities",
]
