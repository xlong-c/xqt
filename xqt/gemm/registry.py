"""Kernel capability registry for XQT GEMM.

Registry entries describe facts about an implementation.  They do not claim
that a CUTLASS metadata entry is executable: only an entry with
``maturity='executable'`` and a non-null executor can be selected as native.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

_MATURITIES = {"executable", "metadata_only", "planned", "reference_guarded"}

from .contracts import EpilogueSpec, GemmProblem, QuantSpec

@dataclass(frozen=True, slots=True)
class GemmCapability:
    """Declarative capability matrix for one kernel family.

    Defines which architectures, dtypes, scale modes, phases and epilogues
    a kernel supports. Used by registry to filter candidates during dispatch.
    """

    architectures: tuple[str, ...] = ("any",)
    weight_dtypes: tuple[str, ...] = ("fp32",)
    activation_dtypes: tuple[str, ...] = ("fp32",)
    scale_modes: tuple[str, ...] = ("w:none/a:none",)
    phases: tuple[str, ...] = ("generic", "prefill", "decode")
    epilogues: tuple[str, ...] = ("none",)
    min_sm: int | None = None

    def supports(
        self,
        problem: GemmProblem,
        quant: QuantSpec,
        epilogue: EpilogueSpec,
    ) -> bool:
        """Check if this capability matches the given problem/quant/epilogue."""
        architecture = problem.arch
        if "any" not in self.architectures and architecture not in self.architectures:
            return False
        if quant.weight_dtype not in self.weight_dtypes:
            return False
        if quant.activation_dtype not in self.activation_dtypes:
            return False
        if quant.scale_mode not in self.scale_modes and "any" not in self.scale_modes:
            return False
        if problem.phase not in self.phases and "any" not in self.phases:
            return False
        if epilogue.activation not in self.epilogues and "any" not in self.epilogues:
            return False
        if self.min_sm is not None and (problem.sm is None or problem.sm < self.min_sm):
            return False
        return True

    def to_dict(self) -> dict[str, Any]:
        """Return capability as dict for registry serialization."""


GemmExecutor = Callable[..., Any]


@dataclass(frozen=True, slots=True)
class GemmKernelRegistration:
    """One registry entry, including implementation maturity and tile metadata.

    Tracks kernel family, maturity level (executable/metadata_only/planned),
    capability matrix, tile params and executor. Used for dispatch filtering
    and fallback chain construction.
    """

    name: str
    backend: str
    maturity: str
    capability: GemmCapability
    kernel_family: str
    layout: str = "canonical"
    tile_shape: tuple[int, int, int] | None = None
    warp_count: int | None = None
    stage_count: int | None = None
    alignment: tuple[int, int, int] = (1, 1, 1)
    priority: int = 0
    implementation: str = "reference"
    executor: GemmExecutor | None = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("GemmKernelRegistration.name must be non-empty")
        if self.maturity not in _MATURITIES:
            raise ValueError(f"unsupported GEMM maturity: {self.maturity!r}")
        if self.priority < 0:
            raise ValueError("kernel priority must be non-negative")
        if any(int(item) <= 0 for item in self.alignment):
            raise ValueError("kernel alignment values must be positive")
        if self.tile_shape is not None and any(
            int(item) <= 0 for item in self.tile_shape
        ):
            raise ValueError("tile_shape values must be positive")
        if self.maturity == "executable" and self.executor is None:
            raise ValueError("executable GEMM registration requires an executor")

    @property
    def executable(self) -> bool:
        return self.maturity == "executable" and self.executor is not None

    def supports(
        self, problem: GemmProblem, quant: QuantSpec, epilogue: EpilogueSpec
    ) -> bool:
        return self.capability.supports(problem, quant, epilogue)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "backend": self.backend,
            "maturity": self.maturity,
            "kernel_family": self.kernel_family,
            "layout": self.layout,
            "tile_shape": None if self.tile_shape is None else list(self.tile_shape),
            "warp_count": self.warp_count,
            "stage_count": self.stage_count,
            "alignment": list(self.alignment),
            "priority": self.priority,
            "implementation": self.implementation,
            "executable": self.executable,
            "capability": self.capability.to_dict(),
        }


class GemmKernelRegistry:
    """Explicit registry used by dispatch and introspection."""

    def __init__(self, entries: Iterable[GemmKernelRegistration] = ()) -> None:
        self._entries: dict[str, GemmKernelRegistration] = {}
        for entry in entries:
            self.register(entry)

    def register(self, entry: GemmKernelRegistration) -> None:
        if entry.name in self._entries:
            raise ValueError(f"duplicate GEMM kernel registration: {entry.name!r}")
        self._entries[entry.name] = entry

    def replace(self, entry: GemmKernelRegistration) -> None:
        """Replace an existing entry after an artifact/correctness gate."""

        if entry.name not in self._entries:
            raise KeyError(
                f"cannot replace unknown GEMM kernel registration: {entry.name!r}"
            )
        self._entries[entry.name] = entry

    def get(self, name: str) -> GemmKernelRegistration:
        try:
            return self._entries[name]
        except KeyError as exc:
            raise KeyError(f"unknown GEMM kernel registration: {name!r}") from exc

    def entries(self) -> tuple[GemmKernelRegistration, ...]:
        return tuple(self._entries.values())

    def matching(
        self,
        problem: GemmProblem,
        quant: QuantSpec,
        epilogue: EpilogueSpec,
    ) -> tuple[GemmKernelRegistration, ...]:
        return tuple(
            sorted(
                (
                    entry
                    for entry in self._entries.values()
                    if entry.supports(problem, quant, epilogue)
                ),
                key=lambda entry: entry.priority,
                reverse=True,
            )
        )

    def to_dict(self) -> list[dict[str, Any]]:
        return [entry.to_dict() for entry in self.entries()]


def _reference_capability(
    *,
    weight_dtypes: tuple[str, ...],
    activation_dtypes: tuple[str, ...],
    scale_modes: tuple[str, ...],
    epilogues: tuple[str, ...] = ("any",),
    phases: tuple[str, ...] = ("generic", "prefill", "decode"),
) -> GemmCapability:
    return GemmCapability(
        architectures=("any",),
        weight_dtypes=weight_dtypes,
        activation_dtypes=activation_dtypes,
        scale_modes=scale_modes,
        phases=phases,
        epilogues=epilogues,
    )


def default_registry() -> GemmKernelRegistry:
    """Build a fresh registry with P0 reference and planned native entries."""

    dense_modes = ("w:per_tensor/a:per_tensor",)
    quantized_modes = (
        "w:per_tensor/a:per_tensor",
        "w:per_channel/a:per_tensor",
        "w:groupwise/a:per_tensor",
        "w:blockwise/a:per_tensor",
        "w:per_tensor/a:per_token",
        "w:per_channel/a:per_token",
        "w:groupwise/a:per_token",
        "w:blockwise/a:per_token",
    )
    fp8_modes = (
        "w:per_tensor/a:per_tensor",
        "w:per_channel/a:per_tensor",
        "w:per_tensor/a:per_token",
        "w:per_channel/a:per_token",
    )
    entries = [
        GemmKernelRegistration(
            name="dense_fp32_reference",
            backend="torch",
            maturity="reference_guarded",
            capability=_reference_capability(
                weight_dtypes=("fp32",),
                activation_dtypes=("fp32",),
                scale_modes=dense_modes,
            ),
            kernel_family="dense",
            priority=10,
        ),
        GemmKernelRegistration(
            name="dense_fp16_reference",
            backend="torch",
            maturity="reference_guarded",
            capability=_reference_capability(
                weight_dtypes=("fp16",),
                activation_dtypes=("fp16",),
                scale_modes=dense_modes,
            ),
            kernel_family="dense",
            priority=10,
        ),
        GemmKernelRegistration(
            name="dense_bf16_reference",
            backend="torch",
            maturity="reference_guarded",
            capability=_reference_capability(
                weight_dtypes=("bf16",),
                activation_dtypes=("bf16",),
                scale_modes=dense_modes,
            ),
            kernel_family="dense",
            priority=10,
        ),
        GemmKernelRegistration(
            name="sm89_dense_fp16_cutlass",
            backend="cutlass",
            maturity="metadata_only",
            capability=GemmCapability(
                architectures=("sm_89",),
                weight_dtypes=("fp16",),
                activation_dtypes=("fp16",),
                scale_modes=("w:per_tensor/a:per_tensor",),
                phases=("generic", "prefill", "decode"),
                epilogues=("any",),
                min_sm=89,
            ),
            kernel_family="dense_fp16",
            layout="sm89_dense_row_col_v1",
            tile_shape=(128, 128, 32),
            warp_count=4,
            stage_count=3,
            alignment=(8, 8, 8),
            priority=80,
            implementation="cutlass_artifact_pending",
        ),
        GemmKernelRegistration(
            name="sm89_dense_bf16_cutlass",
            backend="cutlass",
            maturity="metadata_only",
            capability=GemmCapability(
                architectures=("sm_89",),
                weight_dtypes=("bf16",),
                activation_dtypes=("bf16",),
                scale_modes=("w:per_tensor/a:per_tensor",),
                phases=("generic", "prefill", "decode"),
                epilogues=("any",),
                min_sm=89,
            ),
            kernel_family="dense_bf16",
            layout="sm89_dense_row_col_v1",
            tile_shape=(128, 128, 32),
            warp_count=4,
            stage_count=3,
            alignment=(8, 8, 8),
            priority=80,
            implementation="cutlass_artifact_pending",
        ),
        GemmKernelRegistration(
            name="sm90_dense_fp16_wgmma",
            backend="cutlass",
            maturity="metadata_only",
            capability=GemmCapability(
                architectures=("sm_90",),
                weight_dtypes=("fp16",),
                activation_dtypes=("fp16",),
                scale_modes=dense_modes,
                phases=("generic", "prefill", "decode"),
                epilogues=("any",),
                min_sm=90,
            ),
            kernel_family="dense_fp16_wgmma",
            layout="sm90_dense_row_col_v1",
            tile_shape=(128, 128, 64),
            warp_count=4,
            stage_count=4,
            alignment=(8, 8, 8),
            priority=90,
            implementation="cutlass_sm90_wgmma_collective_builder_pending",
        ),
        GemmKernelRegistration(
            name="sm90_dense_bf16_wgmma",
            backend="cutlass",
            maturity="metadata_only",
            capability=GemmCapability(
                architectures=("sm_90",),
                weight_dtypes=("bf16",),
                activation_dtypes=("bf16",),
                scale_modes=dense_modes,
                phases=("generic", "prefill", "decode"),
                epilogues=("any",),
                min_sm=90,
            ),
            kernel_family="dense_bf16_wgmma",
            layout="sm90_dense_row_col_v1",
            tile_shape=(128, 128, 64),
            warp_count=4,
            stage_count=4,
            alignment=(8, 8, 8),
            priority=90,
            implementation="cutlass_sm90_wgmma_collective_builder_pending",
        ),
        GemmKernelRegistration(
            name="quantized_dequant_reference",
            backend="torch",
            maturity="reference_guarded",
            capability=_reference_capability(
                weight_dtypes=("int4", "int8", "int3", "int2", "fp8_e4m3", "fp8_e5m2"),
                activation_dtypes=(
                    "fp16",
                    "bf16",
                    "fp32",
                    "int8",
                    "fp8_e4m3",
                    "fp8_e5m2",
                ),
                scale_modes=quantized_modes,
            ),
            kernel_family="quantized_dequant",
            priority=5,
        ),
        GemmKernelRegistration(
            name="fp8_blockwise_reference",
            backend="torch",
            maturity="reference_guarded",
            capability=_reference_capability(
                weight_dtypes=("fp8_e4m3", "fp8_e5m2"),
                activation_dtypes=("fp8_e4m3", "fp8_e5m2"),
                scale_modes=(
                    "w:blockwise/a:blockwise",
                    "w:blockwise/a:per_token",
                ),
            ),
            kernel_family="fp8_blockwise_reference",
            layout="xqt_fp8_rowmajor_v1",
            priority=6,
            implementation="reference",
        ),
        GemmKernelRegistration(
            name="w4a16_packed_reference",
            backend="torch",
            maturity="reference_guarded",
            capability=_reference_capability(
                weight_dtypes=("int4",),
                activation_dtypes=("fp16", "bf16"),
                scale_modes=("w:groupwise/a:per_tensor", "w:blockwise/a:per_tensor"),
            ),
            kernel_family="w4a16_reference",
            layout="xqt_int4_nk_v1",
            alignment=(1, 1, 1),
            priority=6,
            implementation="reference",
        ),
        GemmKernelRegistration(
            name="grouped_reference",
            backend="torch",
            maturity="reference_guarded",
            capability=_reference_capability(
                weight_dtypes=(
                    "fp16",
                    "bf16",
                    "fp32",
                    "int4",
                    "int8",
                    "fp8_e4m3",
                    "fp8_e5m2",
                ),
                activation_dtypes=(
                    "fp16",
                    "bf16",
                    "fp32",
                    "int8",
                    "fp8_e4m3",
                    "fp8_e5m2",
                ),
                scale_modes=("any",),
            ),
            kernel_family="grouped",
            priority=1,
        ),
        GemmKernelRegistration(
            name="sm89_fp8_e4m3_cutlass",
            backend="cutlass",
            maturity="metadata_only",
            capability=GemmCapability(
                architectures=("sm_89",),
                weight_dtypes=("fp8_e4m3",),
                activation_dtypes=("fp8_e4m3",),
                scale_modes=fp8_modes,
                phases=("generic", "prefill", "decode"),
                epilogues=("any",),
                min_sm=89,
            ),
            kernel_family="fp8_cutlass_mma",
            layout="xqt_fp8_rowmajor_v1",
            tile_shape=(128, 128, 32),
            warp_count=8,
            stage_count=3,
            alignment=(16, 8, 32),
            priority=100,
            implementation="cutlass_fp8_sm89_artifact_pending",
        ),
        GemmKernelRegistration(
            name="sm89_fp8_e5m2_cutlass",
            backend="cutlass",
            maturity="metadata_only",
            capability=GemmCapability(
                architectures=("sm_89",),
                weight_dtypes=("fp8_e5m2",),
                activation_dtypes=("fp8_e5m2",),
                scale_modes=fp8_modes,
                phases=("generic", "prefill", "decode"),
                epilogues=("any",),
                min_sm=89,
            ),
            kernel_family="fp8_cutlass_mma",
            layout="xqt_fp8_rowmajor_v1",
            tile_shape=(128, 128, 32),
            warp_count=8,
            stage_count=3,
            alignment=(16, 8, 32),
            priority=100,
            implementation="cutlass_fp8_sm89_artifact_pending",
        ),
        GemmKernelRegistration(
            name="sm90_fp8_e4m3_wgmma",
            backend="cutlass",
            maturity="metadata_only",
            capability=GemmCapability(
                architectures=("sm_90",),
                weight_dtypes=("fp8_e4m3",),
                activation_dtypes=("fp8_e4m3",),
                scale_modes=fp8_modes
                + ("w:blockwise/a:blockwise", "w:blockwise/a:per_token"),
                phases=("generic", "prefill", "decode"),
                epilogues=("any",),
                min_sm=90,
            ),
            kernel_family="fp8_wgmma",
            layout="xqt_fp8_rowmajor_v1",
            tile_shape=(128, 128, 128),
            warp_count=4,
            stage_count=4,
            alignment=(16, 16, 32),
            priority=100,
            implementation="cutlass_sm90_wgmma_tma_pending",
        ),
        GemmKernelRegistration(
            name="sm90_fp8_e5m2_wgmma",
            backend="cutlass",
            maturity="metadata_only",
            capability=GemmCapability(
                architectures=("sm_90",),
                weight_dtypes=("fp8_e5m2",),
                activation_dtypes=("fp8_e5m2",),
                scale_modes=fp8_modes
                + ("w:blockwise/a:blockwise", "w:blockwise/a:per_token"),
                phases=("generic", "prefill", "decode"),
                epilogues=("any",),
                min_sm=90,
            ),
            kernel_family="fp8_wgmma",
            layout="xqt_fp8_rowmajor_v1",
            tile_shape=(128, 128, 128),
            warp_count=4,
            stage_count=4,
            alignment=(16, 16, 32),
            priority=100,
            implementation="cutlass_sm90_wgmma_tma_pending",
        ),
        GemmKernelRegistration(
            name="sm89_int8_mma_cutlass",
            backend="cutlass",
            maturity="metadata_only",
            capability=GemmCapability(
                architectures=("sm_89",),
                weight_dtypes=("int8",),
                activation_dtypes=("int8",),
                scale_modes=(
                    "w:per_channel/a:per_token",
                    "w:per_channel/a:per_tensor",
                    "w:per_tensor/a:per_tensor",
                ),
                phases=("generic", "prefill", "decode"),
                epilogues=("any",),
                min_sm=89,
            ),
            kernel_family="w8a8_int8_mma",
            layout="sm89_int8_nk_v1",
            tile_shape=(64, 128, 64),
            warp_count=8,
            stage_count=3,
            alignment=(16, 16, 32),
            priority=100,
            implementation="cutlass_artifact_pending",
        ),
        GemmKernelRegistration(
            name="sm89_w4a16_cutlass",
            backend="cutlass",
            maturity="metadata_only",
            capability=GemmCapability(
                architectures=("sm_89",),
                weight_dtypes=("int4",),
                activation_dtypes=("fp16", "bf16"),
                scale_modes=("w:groupwise/a:per_tensor", "w:blockwise/a:per_tensor"),
                phases=("generic", "prefill", "decode"),
                epilogues=("any",),
                min_sm=89,
            ),
            kernel_family="w4a16",
            layout="sm89_int4_nk_v1",
            tile_shape=(64, 128, 64),
            warp_count=8,
            stage_count=3,
            alignment=(16, 16, 32),
            priority=90,
            implementation="cutlass_artifact_pending",
        ),
        GemmKernelRegistration(
            name="sm89_w4a16_cutlass_fused",
            backend="cutlass",
            maturity="metadata_only",
            capability=GemmCapability(
                architectures=("sm_89",),
                weight_dtypes=("int4",),
                activation_dtypes=("fp16",),
                scale_modes=("w:groupwise/a:per_tensor", "w:blockwise/a:per_tensor"),
                phases=("generic", "prefill", "decode"),
                epilogues=("none", "any"),
                min_sm=89,
            ),
            kernel_family="w4a16_fused_cutlass_mma",
            layout="xqt_int4_nk_v1",
            tile_shape=(16, 8, 16),
            warp_count=1,
            stage_count=1,
            alignment=(16, 8, 16),
            priority=95,
            implementation="custom_cuda_cutlass_mma_pending",
        ),
        GemmKernelRegistration(
            name="sm89_w4a16_cutlass_alt_tile",
            backend="cutlass",
            maturity="metadata_only",
            capability=GemmCapability(
                architectures=("sm_89",),
                weight_dtypes=("int4",),
                activation_dtypes=("fp16", "bf16"),
                scale_modes=("w:groupwise/a:per_tensor", "w:blockwise/a:per_tensor"),
                phases=("generic", "prefill", "decode"),
                epilogues=("any",),
                min_sm=89,
            ),
            kernel_family="w4a16",
            layout="sm89_int4_nk_v1_alt_tile",
            tile_shape=(32, 64, 32),
            warp_count=4,
            stage_count=2,
            alignment=(8, 8, 32),
            priority=85,
            implementation="cutlass_alternate_tile_pending",
        ),
        GemmKernelRegistration(
            name="sm89_w4a16_dequant_fallback",
            backend="cuda",
            maturity="metadata_only",
            capability=GemmCapability(
                architectures=("sm_89",),
                weight_dtypes=("int4",),
                activation_dtypes=("fp16", "bf16"),
                scale_modes=("w:groupwise/a:per_tensor", "w:blockwise/a:per_tensor"),
                phases=("generic", "prefill", "decode"),
                epilogues=("any",),
                min_sm=89,
            ),
            kernel_family="w4a16_dequant_fallback",
            layout="xqt_int4_nk_v1",
            tile_shape=(8, 16, 32),
            warp_count=4,
            stage_count=1,
            alignment=(1, 1, 1),
            priority=70,
            implementation="cuda_artifact_pending",
        ),
        GemmKernelRegistration(
            name="sm89_w4a16_triton_dequant",
            backend="triton",
            maturity="planned",
            capability=GemmCapability(
                architectures=("sm_89",),
                weight_dtypes=("int4",),
                activation_dtypes=("fp16", "bf16"),
                scale_modes=("w:groupwise/a:per_tensor", "w:blockwise/a:per_tensor"),
                phases=("generic", "prefill", "decode"),
                epilogues=("any",),
                min_sm=89,
            ),
            kernel_family="w4a16_triton_dequant",
            layout="xqt_int4_nk_v1",
            alignment=(1, 1, 1),
            priority=60,
            implementation="triton_dequant_pending",
        ),
        GemmKernelRegistration(
            name="w8a16_packed_reference",
            backend="torch",
            maturity="reference_guarded",
            capability=_reference_capability(
                weight_dtypes=("int8",),
                activation_dtypes=("fp16", "bf16"),
                scale_modes=(
                    "w:per_channel/a:per_tensor",
                    "w:groupwise/a:per_tensor",
                    "w:blockwise/a:per_tensor",
                ),
            ),
            kernel_family="w8a16_reference",
            layout="xqt_int8_nk_v1",
            alignment=(1, 1, 1),
            priority=7,
            implementation="reference",
        ),
        GemmKernelRegistration(
            name="sm89_w8a16_cutlass",
            backend="cutlass",
            maturity="metadata_only",
            capability=GemmCapability(
                architectures=("sm_89",),
                weight_dtypes=("int8",),
                activation_dtypes=("fp16", "bf16"),
                scale_modes=("w:per_channel/a:per_tensor",),
                phases=("generic", "prefill", "decode"),
                epilogues=("any",),
                min_sm=89,
            ),
            kernel_family="w8a16",
            layout="sm89_int8_nk_v1",
            tile_shape=(64, 128, 64),
            warp_count=8,
            stage_count=3,
            alignment=(16, 16, 32),
            priority=90,
            implementation="cutlass_artifact_pending",
        ),
        GemmKernelRegistration(
            name="w4a8_reference",
            backend="torch",
            maturity="reference_guarded",
            capability=_reference_capability(
                weight_dtypes=("int4",),
                activation_dtypes=("int8", "fp8_e4m3", "fp8_e5m2"),
                scale_modes=(
                    "w:groupwise/a:per_tensor",
                    "w:groupwise/a:per_token",
                    "w:groupwise/a:blockwise",
                ),
            ),
            kernel_family="w4a8_reference",
            layout="xqt_int4_nk_v1",
            alignment=(1, 1, 1),
            priority=8,
            implementation="reference",
        ),
        GemmKernelRegistration(
            name="sm89_w4a8_int8_cutlass",
            backend="cutlass",
            maturity="metadata_only",
            capability=GemmCapability(
                architectures=("sm_89",),
                weight_dtypes=("int4",),
                activation_dtypes=("int8",),
                scale_modes=(
                    "w:groupwise/a:per_tensor",
                    "w:groupwise/a:per_token",
                ),
                phases=("generic", "prefill", "decode"),
                epilogues=("any",),
                min_sm=89,
            ),
            kernel_family="w4a8_int8_mma",
            layout="sm89_w4a8_int4_int8_v1",
            tile_shape=(64, 128, 64),
            warp_count=8,
            stage_count=3,
            alignment=(16, 16, 32),
            priority=98,
            implementation="cutlass_sm89_w4a8_int8_artifact_pending",
        ),
        GemmKernelRegistration(
            name="sm89_w4a8_fp8_cutlass",
            backend="cutlass",
            maturity="metadata_only",
            capability=GemmCapability(
                architectures=("sm_89",),
                weight_dtypes=("int4",),
                activation_dtypes=("fp8_e4m3", "fp8_e5m2"),
                scale_modes=(
                    "w:groupwise/a:per_tensor",
                    "w:groupwise/a:per_token",
                    "w:groupwise/a:blockwise",
                ),
                phases=("generic", "prefill", "decode"),
                epilogues=("any",),
                min_sm=89,
            ),
            kernel_family="w4a8_fp8_mma",
            layout="sm89_w4a8_int4_fp8_v1",
            tile_shape=(64, 128, 64),
            warp_count=8,
            stage_count=3,
            alignment=(16, 16, 32),
            priority=98,
            implementation="cutlass_sm89_w4a8_fp8_artifact_pending",
        ),
        GemmKernelRegistration(
            name="w3a16_reference",
            backend="torch",
            maturity="reference_guarded",
            capability=_reference_capability(
                weight_dtypes=("int3",),
                activation_dtypes=("fp16", "bf16"),
                scale_modes=(
                    "w:groupwise/a:per_tensor",
                    "w:per_channel/a:per_tensor",
                ),
            ),
            kernel_family="w3a16_reference",
            layout="xqt_int3_nk_v1",
            priority=8,
            implementation="reference",
        ),
        GemmKernelRegistration(
            name="w2a16_reference",
            backend="torch",
            maturity="reference_guarded",
            capability=_reference_capability(
                weight_dtypes=("int2",),
                activation_dtypes=("fp16", "bf16"),
                scale_modes=(
                    "w:groupwise/a:per_tensor",
                    "w:per_channel/a:per_tensor",
                ),
            ),
            kernel_family="w2a16_reference",
            layout="xqt_int2_nk_v1",
            priority=8,
            implementation="reference",
        ),
        GemmKernelRegistration(
            name="sparse2_4_reference",
            backend="torch",
            maturity="reference_guarded",
            capability=_reference_capability(
                weight_dtypes=("fp16", "bf16", "int8"),
                activation_dtypes=("fp16", "bf16"),
                scale_modes=(
                    "w:per_tensor/a:per_tensor",
                    "w:per_channel/a:per_tensor",
                    "w:groupwise/a:per_tensor",
                ),
            ),
            kernel_family="sparse2_4_reference",
            layout="xqt_sparse2_4_v1",
            priority=3,
            implementation="reference",
        ),
        GemmKernelRegistration(
            name="vector_codebook_reference",
            backend="torch",
            maturity="reference_guarded",
            capability=_reference_capability(
                weight_dtypes=("codebook",),
                activation_dtypes=("fp16", "bf16"),
                scale_modes=(
                    "w:groupwise/a:per_tensor",
                    "w:per_channel/a:per_tensor",
                ),
            ),
            kernel_family="vector_codebook_reference",
            layout="xqt_codebook_v1",
            priority=8,
            implementation="reference",
        ),
        GemmKernelRegistration(
            name="fp4_e2m1_reference",
            backend="torch",
            maturity="reference_guarded",
            capability=_reference_capability(
                weight_dtypes=("fp4",),
                activation_dtypes=("fp16", "bf16", "fp32"),
                scale_modes=("w:groupwise/a:per_tensor",),
            ),
            kernel_family="fp4_e2m1_reference",
            layout="xqt_fp4_nk_v1",
            priority=8,
            implementation="reference",
        ),
        GemmKernelRegistration(
            name="mxfp4_reference",
            backend="torch",
            maturity="reference_guarded",
            capability=_reference_capability(
                weight_dtypes=("mxfp4",),
                activation_dtypes=("fp16", "bf16", "fp32"),
                scale_modes=("w:groupwise/a:per_tensor",),
            ),
            kernel_family="mxfp4_reference",
            layout="xqt_fp4_nk_v1",
            priority=8,
            implementation="reference",
        ),
        GemmKernelRegistration(
            name="nvfp4_reference",
            backend="torch",
            maturity="reference_guarded",
            capability=_reference_capability(
                weight_dtypes=("nvfp4",),
                activation_dtypes=("fp16", "bf16", "fp32"),
                scale_modes=("w:groupwise/a:per_tensor",),
            ),
            kernel_family="nvfp4_reference",
            layout="xqt_fp4_nk_v1",
            priority=8,
            implementation="reference",
        ),
        GemmKernelRegistration(
            name="sm100_fp4_e2m1_cutlass",
            backend="cutlass",
            maturity="planned",
            capability=GemmCapability(
                architectures=("sm_100",),
                weight_dtypes=("fp4",),
                activation_dtypes=("fp16", "bf16", "fp8_e4m3", "fp8_e5m2"),
                scale_modes=("w:groupwise/a:per_tensor", "w:groupwise/a:blockwise"),
                phases=("generic", "prefill", "decode"),
                epilogues=("any",),
                min_sm=100,
            ),
            kernel_family="fp4_e2m1_mma",
            layout="sm100_fp4_e2m1_v1",
            tile_shape=(128, 128, 64),
            warp_count=4,
            stage_count=4,
            alignment=(128, 128, 64),
            priority=110,
            implementation="cutlass_sm100_fp4_pending",
        ),
        GemmKernelRegistration(
            name="sm100_mxfp4_cutlass",
            backend="cutlass",
            maturity="planned",
            capability=GemmCapability(
                architectures=("sm_100",),
                weight_dtypes=("mxfp4",),
                activation_dtypes=("fp16", "bf16", "fp8_e4m3", "fp8_e5m2"),
                scale_modes=("w:groupwise/a:per_tensor", "w:groupwise/a:blockwise"),
                phases=("generic", "prefill", "decode"),
                epilogues=("any",),
                min_sm=100,
            ),
            kernel_family="mxfp4_mma",
            layout="sm100_mxfp4_v1",
            tile_shape=(128, 128, 64),
            warp_count=4,
            stage_count=4,
            alignment=(128, 128, 64),
            priority=110,
            implementation="cutlass_sm100_mxfp4_pending",
        ),
        GemmKernelRegistration(
            name="sm100_nvfp4_cutlass",
            backend="cutlass",
            maturity="planned",
            capability=GemmCapability(
                architectures=("sm_100",),
                weight_dtypes=("nvfp4",),
                activation_dtypes=("fp16", "bf16", "fp8_e4m3", "fp8_e5m2"),
                scale_modes=("w:groupwise/a:per_tensor", "w:groupwise/a:blockwise"),
                phases=("generic", "prefill", "decode"),
                epilogues=("any",),
                min_sm=100,
            ),
            kernel_family="nvfp4_mma",
            layout="sm100_nvfp4_v1",
            tile_shape=(128, 128, 64),
            warp_count=4,
            stage_count=4,
            alignment=(128, 128, 64),
            priority=110,
            implementation="cutlass_sm100_nvfp4_pending",
        ),
        GemmKernelRegistration(
            name="sm90_grouped_fp8_e4m3_wgmma",
            backend="cutlass",
            maturity="metadata_only",
            capability=GemmCapability(
                architectures=("sm_90",),
                weight_dtypes=("fp8_e4m3",),
                activation_dtypes=("fp8_e4m3",),
                scale_modes=(
                    "w:per_tensor/a:per_tensor",
                    "w:per_channel/a:per_token",
                    "w:blockwise/a:blockwise",
                ),
                phases=("generic", "prefill", "decode"),
                epilogues=("any",),
                min_sm=90,
            ),
            kernel_family="fp8_grouped_wgmma",
            layout="xqt_fp8_rowmajor_v1",
            tile_shape=(128, 128, 128),
            warp_count=4,
            stage_count=4,
            alignment=(16, 16, 32),
            priority=105,
            implementation="cutlass_sm90_grouped_wgmma_tma_pending",
        ),
        GemmKernelRegistration(
            name="sm90_grouped_fp8_e5m2_wgmma",
            backend="cutlass",
            maturity="metadata_only",
            capability=GemmCapability(
                architectures=("sm_90",),
                weight_dtypes=("fp8_e5m2",),
                activation_dtypes=("fp8_e5m2",),
                scale_modes=(
                    "w:per_tensor/a:per_tensor",
                    "w:per_channel/a:per_token",
                    "w:blockwise/a:blockwise",
                ),
                phases=("generic", "prefill", "decode"),
                epilogues=("any",),
                min_sm=90,
            ),
            kernel_family="fp8_grouped_wgmma",
            layout="xqt_fp8_rowmajor_v1",
            tile_shape=(128, 128, 128),
            warp_count=4,
            stage_count=4,
            alignment=(16, 16, 32),
            priority=105,
            implementation="cutlass_sm90_grouped_wgmma_tma_pending",
        ),
        GemmKernelRegistration(
            name="sm120_fp8_e4m3_tcgen05",
            backend="cutlass",
            maturity="metadata_only",
            capability=GemmCapability(
                architectures=("sm_120",),
                weight_dtypes=("fp8_e4m3",),
                activation_dtypes=("fp8_e4m3",),
                scale_modes=("w:blockwise/a:blockwise",),
                phases=("generic", "prefill", "decode"),
                epilogues=("any",),
                min_sm=120,
            ),
            kernel_family="fp8_tcgen05_blockwise",
            layout="xqt_fp8_sm120_blockwise_1x128x128_v1",
            tile_shape=(128, 128, 128),
            warp_count=4,
            stage_count=2,
            alignment=(16, 16, 32),
            priority=115,
            implementation="cutlass_sm120_fp8_blockwise_pending",
        ),
        GemmKernelRegistration(
            name="sm120_fp8_e5m2_tcgen05",
            backend="cutlass",
            maturity="metadata_only",
            capability=GemmCapability(
                architectures=("sm_120",),
                weight_dtypes=("fp8_e5m2",),
                activation_dtypes=("fp8_e5m2",),
                scale_modes=("w:blockwise/a:blockwise",),
                phases=("generic", "prefill", "decode"),
                epilogues=("any",),
                min_sm=120,
            ),
            kernel_family="fp8_tcgen05_blockwise",
            layout="xqt_fp8_sm120_blockwise_1x128x128_v1",
            tile_shape=(128, 128, 128),
            warp_count=4,
            stage_count=2,
            alignment=(16, 16, 32),
            priority=115,
            implementation="cutlass_sm120_fp8_blockwise_pending",
        ),
        GemmKernelRegistration(
            name="svd_dual_path_reference",
            backend="torch",
            maturity="reference_guarded",
            capability=_reference_capability(
                weight_dtypes=("int4", "int8", "fp4", "mxfp4", "nvfp4"),
                activation_dtypes=("fp16", "bf16", "fp32"),
                scale_modes=(
                    "w:per_channel/a:per_tensor",
                    "w:groupwise/a:per_tensor",
                    "w:blockwise/a:per_tensor",
                ),
            ),
            kernel_family="svd_dual_path",
            layout="xqt_svd_dual_v1",
            priority=2,
            implementation="reference",
        ),
    ]
    return GemmKernelRegistry(entries)


__all__ = [
    "GemmCapability",
    "GemmExecutor",
    "GemmKernelRegistration",
    "GemmKernelRegistry",
    "default_registry",
]
