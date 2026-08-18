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

        return {
            "architectures": list(self.architectures),
            "weight_dtypes": list(self.weight_dtypes),
            "activation_dtypes": list(self.activation_dtypes),
            "scale_modes": list(self.scale_modes),
            "phases": list(self.phases),
            "epilogues": list(self.epilogues),
            "min_sm": self.min_sm,
        }


GemmExecutor = Callable[..., Any]
GemmPrecisionScore = Callable[[GemmProblem, frozenset[str], str], int | None]
GemmPrecisionCaveats = Callable[[GemmProblem, frozenset[str]], tuple[str, ...]]


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
    scope: str = "contract"
    precision_mmas: tuple[str, ...] = ()
    precision_score: GemmPrecisionScore | None = field(
        default=None, compare=False, repr=False
    )
    precision_caveats: GemmPrecisionCaveats | None = field(
        default=None, compare=False, repr=False
    )
    selection_reason: str = ""
    dispatchable_by_precision: bool = False
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
        if self.scope not in {"contract", "precision"}:
            raise ValueError("GEMM registration scope must be contract or precision")
        if self.scope == "precision" and not self.precision_mmas:
            raise ValueError("precision GEMM registration requires precision_mmas")

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
            "scope": self.scope,
            "precision_mmas": list(self.precision_mmas),
            "dispatchable_by_precision": self.dispatchable_by_precision,
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
                    if entry.scope == "contract"
                    and entry.supports(problem, quant, epilogue)
                ),
                key=lambda entry: entry.priority,
                reverse=True,
            )
        )

    def matching_precision(
        self,
        *,
        mma: str,
        problem: GemmProblem,
        fused_ops: frozenset[str],
        goal: str,
    ) -> tuple[GemmKernelRegistration, ...]:
        """Return precision-dispatch entries in declaration-driven rank order."""

        ranked: list[tuple[int, GemmKernelRegistration]] = []
        for entry in self._entries.values():
            if entry.scope != "precision" or mma not in entry.precision_mmas:
                continue
            score = (
                entry.priority
                if entry.precision_score is None
                else entry.precision_score(problem, fused_ops, goal)
            )
            if score is not None:
                ranked.append((score, entry))
        ranked.sort(key=lambda item: item[0], reverse=True)
        return tuple(entry for _, entry in ranked)

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


def _precision_executor(engine: str) -> GemmExecutor:
    def execute(
        activation: Any,
        weight: Any,
        *,
        bias: Any = None,
        precision_policy: Any,
        activation_name: str | None,
        transpose_b: bool,
        runtime_kwargs: dict[str, Any],
        pattern: str | None,
        **_: Any,
    ) -> Any:
        from xqt.operator_opt.backends.gemm_precision import (
            _execute_registered_gemm_engine,
        )

        return _execute_registered_gemm_engine(
            engine,
            activation,
            weight,
            bias,
            precision_policy,
            activation_name,
            transpose_b,
            runtime_kwargs,
            pattern=pattern,
        )

    return execute


def _cuda_score(score: int) -> GemmPrecisionScore:
    def resolve(
        problem: GemmProblem, fused_ops: frozenset[str], goal: str
    ) -> int | None:
        del fused_ops, goal
        return score if problem.device.startswith("cuda") else None

    return resolve


def _cpu_score(score: int) -> GemmPrecisionScore:
    def resolve(
        problem: GemmProblem, fused_ops: frozenset[str], goal: str
    ) -> int | None:
        del fused_ops, goal
        return score if not problem.device.startswith("cuda") else None

    return resolve


def _int8_tilelang_score(
    problem: GemmProblem, fused_ops: frozenset[str], goal: str
) -> int | None:
    del goal
    if not problem.device.startswith("cuda"):
        return None
    aligned = all(dimension % 64 == 0 for dimension in problem.shape)
    return 120 if aligned and "activation_quant" not in fused_ops else 80


def _int8_tilelang_caveats(
    problem: GemmProblem, fused_ops: frozenset[str]
) -> tuple[str, ...]:
    aligned = all(dimension % 64 == 0 for dimension in problem.shape)
    if aligned and "activation_quant" not in fused_ops:
        return ()
    return (
        "requires pre-quantized int8 inputs and M/N/K aligned to 64",
    )


def _tilelang_dense_caveats(
    problem: GemmProblem, fused_ops: frozenset[str]
) -> tuple[str, ...]:
    del fused_ops
    if problem.k % 64 == 0:
        return ()
    return ("default TileLang dense schedule requires K aligned to 64",)


def _fp4_torch_score(
    problem: GemmProblem, fused_ops: frozenset[str], goal: str
) -> int | None:
    del fused_ops
    return 20 if problem.device.startswith("cuda") and goal == "accuracy" else None


def _precision_capability(weight_dtypes: tuple[str, ...]) -> GemmCapability:
    return GemmCapability(
        architectures=("any",),
        weight_dtypes=weight_dtypes,
        activation_dtypes=(
            "fp32",
            "fp16",
            "bf16",
            "int8",
            "fp8_e4m3",
            "fp8_e5m2",
        ),
        scale_modes=("any",),
        phases=("any",),
        epilogues=("any",),
    )


def _precision_registrations() -> list[GemmKernelRegistration]:
    dense = ("fp16", "bf16")
    triton_quant = ("int8", "int4", "fp8", "mxfp8", "mxfp6", "mxfp4")
    packed = ("fp4", "nvfp4")
    return [
        GemmKernelRegistration(
            name="precision_triton_dense",
            backend="triton",
            maturity="executable",
            capability=_precision_capability(("fp32", *dense)),
            kernel_family="operator_precision_dense",
            priority=100,
            implementation="operator_opt_registered",
            scope="precision",
            precision_mmas=dense,
            precision_score=_cuda_score(100),
            selection_reason="{mma} dense GEMM uses the registered Triton path",
            dispatchable_by_precision=True,
            executor=_precision_executor("triton"),
        ),
        GemmKernelRegistration(
            name="precision_tilelang_dense",
            backend="tilelang",
            maturity="executable",
            capability=_precision_capability(("fp32", *dense)),
            kernel_family="operator_precision_dense",
            priority=90,
            implementation="operator_opt_registered",
            scope="precision",
            precision_mmas=dense,
            precision_score=_cuda_score(90),
            precision_caveats=_tilelang_dense_caveats,
            selection_reason="registered TileLang dense GEMM alternative",
            dispatchable_by_precision=True,
            executor=_precision_executor("tilelang"),
        ),
        GemmKernelRegistration(
            name="precision_torch_dense_cpu",
            backend="torch",
            maturity="executable",
            capability=_precision_capability(("fp32", *dense, *triton_quant)),
            kernel_family="operator_precision_reference",
            priority=100,
            implementation="torch_reference",
            scope="precision",
            precision_mmas=("fp32", *dense, *triton_quant),
            precision_score=_cpu_score(100),
            selection_reason="non-CUDA device uses the registered torch reference",
            dispatchable_by_precision=True,
            executor=_precision_executor("torch"),
        ),
        GemmKernelRegistration(
            name="precision_triton_quantized",
            backend="triton",
            maturity="executable",
            capability=_precision_capability(
                ("int8", "int4", "fp8_e4m3", "mxfp8", "mxfp6", "mxfp4")
            ),
            kernel_family="operator_precision_quantized",
            priority=100,
            implementation="operator_opt_registered",
            scope="precision",
            precision_mmas=triton_quant,
            precision_score=_cuda_score(100),
            selection_reason="{mma} uses the registered Triton guarded path",
            dispatchable_by_precision=True,
            executor=_precision_executor("triton"),
        ),
        GemmKernelRegistration(
            name="precision_tilelang_int8",
            backend="tilelang",
            maturity="executable",
            capability=_precision_capability(("int8",)),
            kernel_family="operator_precision_w8a8",
            priority=80,
            implementation="operator_opt_registered",
            scope="precision",
            precision_mmas=("int8",),
            precision_score=_int8_tilelang_score,
            precision_caveats=_int8_tilelang_caveats,
            selection_reason="registered TileLang true-W8A8 path",
            dispatchable_by_precision=True,
            executor=_precision_executor("tilelang"),
        ),
        GemmKernelRegistration(
            name="precision_ptx_sm89_int8",
            backend="ptx_sm89",
            maturity="planned",
            capability=_precision_capability(("int8",)),
            kernel_family="operator_precision_w8a8",
            priority=110,
            implementation="stateful_prepack_required",
            scope="precision",
            precision_mmas=("int8",),
            precision_score=_cuda_score(110),
            selection_reason=(
                "SM89 PTX W8A8 exists as a stateful prepacked runtime path but "
                "is not callable from the stateless precision wrapper"
            ),
            dispatchable_by_precision=False,
        ),
        GemmKernelRegistration(
            name="precision_tilelang_packed_fp4",
            backend="tilelang",
            maturity="executable",
            capability=_precision_capability(packed),
            kernel_family="operator_precision_packed_fp4",
            priority=120,
            implementation="operator_opt_registered",
            scope="precision",
            precision_mmas=packed,
            precision_score=lambda problem, fused_ops, goal: 120,
            selection_reason="{mma} goal={goal} uses registered TileLang fused dequant GEMM",
            dispatchable_by_precision=True,
            executor=_precision_executor("tilelang"),
        ),
        GemmKernelRegistration(
            name="precision_tilelang_mxfp4_explicit",
            backend="tilelang",
            maturity="executable",
            capability=_precision_capability(("mxfp4",)),
            kernel_family="operator_precision_packed_fp4",
            priority=70,
            implementation="operator_opt_registered",
            scope="precision",
            precision_mmas=("mxfp4",),
            precision_score=lambda problem, fused_ops, goal: None,
            selection_reason="explicit TileLang MXFP4 packed dequant GEMM",
            dispatchable_by_precision=True,
            executor=_precision_executor("tilelang"),
        ),
        GemmKernelRegistration(
            name="precision_tilelang_int4_explicit",
            backend="tilelang",
            maturity="executable",
            capability=_precision_capability(("int4",)),
            kernel_family="operator_precision_packed_int4",
            priority=70,
            implementation="operator_opt_registered",
            scope="precision",
            precision_mmas=("int4",),
            precision_score=lambda problem, fused_ops, goal: None,
            selection_reason="explicit TileLang INT4 Marlin-style GEMM",
            dispatchable_by_precision=True,
            executor=_precision_executor("tilelang"),
        ),
        GemmKernelRegistration(
            name="precision_torch_fp4_accuracy",
            backend="torch",
            maturity="executable",
            capability=_precision_capability(packed),
            kernel_family="operator_precision_reference",
            priority=20,
            implementation="torch_reference",
            scope="precision",
            precision_mmas=packed,
            precision_score=_fp4_torch_score,
            selection_reason="{mma} goal=accuracy exposes the torch numeric reference",
            dispatchable_by_precision=True,
            executor=_precision_executor("torch"),
        ),
    ]


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
            name="sm120_nvfp4_tcgen05",
            backend="cutlass",
            maturity="metadata_only",
            capability=GemmCapability(
                architectures=("sm_120",),
                weight_dtypes=("nvfp4",),
                activation_dtypes=("fp16",),
                scale_modes=("w:groupwise/a:per_tensor",),
                phases=("generic", "prefill", "decode"),
                epilogues=("any",),
                min_sm=120,
            ),
            kernel_family="nvfp4_tcgen05",
            layout="xqt_nvfp4_sm120_sfvec16_v1",
            tile_shape=(128, 128, 128),
            warp_count=4,
            stage_count=4,
            alignment=(32, 32, 8),
            priority=115,
            implementation="cutlass_sm120_nvfp4_k128_k256_pending",
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
    entries.extend(_precision_registrations())
    return GemmKernelRegistry(entries)


__all__ = [
    "GemmCapability",
    "GemmExecutor",
    "GemmPrecisionCaveats",
    "GemmPrecisionScore",
    "GemmKernelRegistration",
    "GemmKernelRegistry",
    "default_registry",
]
