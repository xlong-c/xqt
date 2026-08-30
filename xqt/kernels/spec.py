"""Lightweight metadata for the unified ``xqt.kernels`` namespace.

This module is torch-free at import time so that ``import xqt.kernels`` stays
cheap and works on a CPU-only box.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, ClassVar, FrozenSet, Optional, Tuple, Union


class KernelBackend(str, Enum):
    """Provenance of a kernel implementation."""

    TORCH = "torch"
    TORCH_COMPILE = "torch_compile"
    TRITON = "triton"
    TILELANG = "tilelang"
    CUTILE = "cutile"
    CUTLASS = "cutlass"
    CUTE_DSL = "cute_dsl"
    CUSTOM_CUDA = "custom_cuda"
    FLASHINFER = "flashinfer"


class DeviceType(str, Enum):
    """Accelerator device family."""

    CUDA = "cuda"
    HIP = "hip"
    NPU = "npu"
    CPU = "cpu"


@dataclass(frozen=True, slots=True)
class PlatformInfo:
    """Minimal snapshot of the runtime accelerator platform."""

    device_type: str = "cpu"
    cuda_arch_major: Optional[int] = None
    cuda_arch_minor: Optional[int] = None

    @property
    def device(self) -> DeviceType:
        try:
            return DeviceType(self.device_type)
        except (ValueError, TypeError):
            return DeviceType.CPU

    @property
    def is_cuda(self) -> bool:
        return self.device_type == "cuda"

    @property
    def is_hip(self) -> bool:
        return self.device_type == "hip"

    @classmethod
    def detect(cls) -> PlatformInfo:
        try:
            import torch  # type: ignore[import]
        except Exception:
            return cls()
        try:
            if getattr(torch.version, "hip", None) is not None and torch.cuda.is_available():
                return cls(device_type="hip")
            npu = getattr(torch, "npu", None)
            if npu is not None and npu.is_available():  # type: ignore[union-attr]
                return cls(device_type="npu")
            if torch.cuda.is_available():
                major, minor = torch.cuda.get_device_capability()
                return cls(device_type="cuda", cuda_arch_major=major, cuda_arch_minor=minor)
        except Exception:
            pass
        return cls()


@dataclass(frozen=True, slots=True)
class CapabilityRequirement:
    """One device (plus optional CUDA arch window) a backend can run on."""

    device: DeviceType
    min_cuda_arch: Optional[Tuple[int, int]] = None
    max_cuda_arch: Optional[Tuple[int, int]] = None

    CUDA: ClassVar[CapabilityRequirement]  # type: ignore[misc]
    HIP: ClassVar[CapabilityRequirement]  # type: ignore[misc]
    NPU: ClassVar[CapabilityRequirement]  # type: ignore[misc]

    @classmethod
    def cuda(
        cls,
        min_sm: Optional[Tuple[int, int]] = None,
        max_sm: Optional[Tuple[int, int]] = None,
    ) -> CapabilityRequirement:
        return cls(device=DeviceType.CUDA, min_cuda_arch=min_sm, max_cuda_arch=max_sm)

    def is_satisfied_by(self, platform: PlatformInfo) -> bool:
        if self.device != platform.device:
            return False
        if self.device == DeviceType.CUDA and platform.cuda_arch_major is not None:
            arch = (platform.cuda_arch_major, platform.cuda_arch_minor or 0)
            if self.min_cuda_arch is not None and arch < self.min_cuda_arch:
                return False
            if self.max_cuda_arch is not None and arch > self.max_cuda_arch:
                return False
        return True


CapabilityRequirement.CUDA = CapabilityRequirement(device=DeviceType.CUDA)  # type: ignore[attr-defined]
CapabilityRequirement.HIP = CapabilityRequirement(device=DeviceType.HIP)  # type: ignore[attr-defined]
CapabilityRequirement.NPU = CapabilityRequirement(device=DeviceType.NPU)  # type: ignore[attr-defined]


def capabilities_satisfied(
    capabilities: Union[
        FrozenSet[CapabilityRequirement],
        Tuple[CapabilityRequirement, ...],
        CapabilityRequirement,
        frozenset[CapabilityRequirement],
    ],
    platform: PlatformInfo,
) -> bool:
    if isinstance(capabilities, CapabilityRequirement):
        capabilities = (capabilities,)  # type: ignore[assignment]
    if not capabilities:
        return True
    return any(c.is_satisfied_by(platform) for c in capabilities)


@dataclass(frozen=True, slots=True)
class FormatSignature:
    supported_dtypes: Tuple[str, ...] = ()
    in_place: bool = False
    description: str = ""


@dataclass(frozen=True, slots=True)
class KernelSpec:
    op: str
    backend: KernelBackend
    target: str
    capabilities: FrozenSet[CapabilityRequirement] = field(default_factory=frozenset)  # type: ignore[type-arg]
    format_signature: FormatSignature = field(default_factory=FormatSignature)
    description: str = ""

    @property
    def group(self) -> str:
        return self.op.split(".", 1)[0] if "." in self.op else self.op

    @property
    def name(self) -> str:
        return self.op.split(".", 1)[1] if "." in self.op else self.op

    def is_available(self, platform: PlatformInfo) -> bool:
        return capabilities_satisfied(self.capabilities, platform)

    def load(self) -> Callable[..., object]:
        from xqt.core.base.errors import XQTBackendError

        if ":" not in self.target:
            raise XQTBackendError(f"invalid kernel target {self.target!r} for op {self.op!r}")
        module_name, attr_path = self.target.split(":", 1)
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            raise XQTBackendError(f"kernel backend {self.backend.value!r} for op {self.op!r} not available: {exc}") from exc
        obj: object = module
        try:
            for part in attr_path.split("."):
                obj = getattr(obj, part)
        except AttributeError as exc:
            raise XQTBackendError(f"kernel target {self.target!r} not found for op {self.op!r}: {exc}") from exc
        if not callable(obj):
            raise XQTBackendError(f"kernel target {self.target!r} for op {self.op!r} is not callable")
        return obj  # type: ignore[return-value]
