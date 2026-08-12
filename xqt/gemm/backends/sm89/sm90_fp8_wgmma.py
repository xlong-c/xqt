"""SM90 FP8 WGMMA/TMA capability boundary.

The repository does not currently have an SM90 validation device.  This
module therefore owns the independent contract and build/manifest path while
keeping execution reference-guarded.  It must not reuse the SM89 executor or
artifact.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import torch

from xqt.core.errors import XQTBackendError

from ..contracts import GemmSpec, GroupedGemmProblem, PackedWeight, QuantSpec
from ..preflight import (
    GemmArtifactManifest,
    build_compile_flags,
    default_cache_dir,
    probe_cuda_cutlass,
)
from ..reference import reference_gemm, reference_packed_grouped_gemm


_SOURCE = Path(__file__).with_name("sm90_fp8_wgmma_scaffold.cu")
_DEFAULT_ARTIFACT = default_cache_dir() / "sm90" / "fp8_wgmma_sm90.so"
_DEFAULT_GROUPED_ARTIFACT = default_cache_dir() / "sm90" / "fp8_grouped_wgmma_sm90.so"


@dataclass(frozen=True, slots=True)
class Sm90Fp8WgmmaContract:
    """Fine-grained FP8 scale contract for Hopper WGMMA/TMA."""

    format_name: str = "fp8_e4m3"
    weight_granularity: str = "blockwise"
    activation_granularity: str = "blockwise"
    block_k: int = 64
    output_dtype: str = "fp16"
    use_wgmma: bool = True
    use_tma: bool = True
    grouped: bool = False
    scale_mainloop: bool = True
    pack_version: str = "xqt-sm90-fp8-wgmma-v1"

    def __post_init__(self) -> None:
        if self.format_name not in {"fp8_e4m3", "fp8_e5m2"}:
            raise ValueError("SM90 WGMMA format must be fp8_e4m3 or fp8_e5m2")
        if self.weight_granularity not in {"per_tensor", "per_channel", "blockwise"}:
            raise ValueError("unsupported SM90 weight scale granularity")
        if self.activation_granularity not in {"per_tensor", "per_token", "blockwise"}:
            raise ValueError("unsupported SM90 activation scale granularity")
        if self.weight_granularity == "blockwise" or self.activation_granularity == "blockwise":
            if int(self.block_k) not in {32, 64, 128}:
                raise ValueError("SM90 blockwise FP8 block_k must be 32, 64, or 128")
        if (
            self.activation_granularity == "blockwise"
            and self.weight_granularity != "blockwise"
        ):
            raise ValueError(
                "SM90 blockwise activation requires blockwise weight under the shared group_size ABI"
            )
        if self.output_dtype not in {"fp16", "bf16", "fp32"}:
            raise ValueError("SM90 WGMMA output_dtype must be fp16, bf16, or fp32")
        if not all(
            isinstance(value, bool)
            for value in (self.use_wgmma, self.use_tma, self.grouped, self.scale_mainloop)
        ):
            raise TypeError("SM90 WGMMA/TMA flags must be bool")

    @property
    def capability(self) -> str:
        return "sm90_wgmma_tma"

    def quant_spec(self) -> QuantSpec:
        return QuantSpec(
            weight_dtype=self.format_name,
            activation_dtype=self.format_name,
            output_dtype=self.output_dtype,
            weight_granularity=self.weight_granularity,
            activation_granularity=self.activation_granularity,
            group_size=int(self.block_k) if (
                self.weight_granularity == "blockwise"
                or self.activation_granularity == "blockwise"
            ) else None,
            weight_scale_source="weight_offline",
            activation_scale_source="activation_static",
            storage_layout="xqt_fp8_rowmajor_v1",
            pack_version=self.pack_version,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "architecture": "sm_90",
            "format": self.format_name,
            "weight_granularity": self.weight_granularity,
            "activation_granularity": self.activation_granularity,
            "block_k": int(self.block_k),
            "output_dtype": self.output_dtype,
            "use_wgmma": self.use_wgmma,
            "use_tma": self.use_tma,
            "grouped": self.grouped,
            "scale_mainloop": self.scale_mainloop,
            "pack_version": self.pack_version,
            "maturity": "metadata_only",
        }


Sm90GroupedFp8WgmmaContract = Sm90Fp8WgmmaContract


@dataclass(frozen=True, slots=True)
class Sm90Fp8WgmmaBuildConfig:
    """Independent SM90 scaffold build inputs."""

    source: Path = _SOURCE
    output: Path = field(default_factory=lambda: _DEFAULT_ARTIFACT)
    target_arch: str = "sm_90"
    grouped: bool = False
    extra_flags: tuple[str, ...] = ("-lineinfo", "-lcudart")

    def __post_init__(self) -> None:
        if self.target_arch != "sm_90":
            raise ValueError("SM90 WGMMA build target_arch must be sm_90")


Sm90GroupedFp8WgmmaBuildConfig = Sm90Fp8WgmmaBuildConfig


def _build(config: Sm90Fp8WgmmaBuildConfig, *, kernel_name: str) -> GemmArtifactManifest:
    report = probe_cuda_cutlass(config.target_arch, require_device=False)
    if not report.ready_for_compile:
        raise XQTBackendError("SM90 WGMMA build preflight failed: " + "; ".join(report.reasons))
    source = config.source.expanduser().resolve()
    output = config.output.expanduser().resolve()
    if not source.is_file():
        raise XQTBackendError(f"SM90 WGMMA source not found: {source}")
    output.parent.mkdir(parents=True, exist_ok=True)
    flags = build_compile_flags(report, source=source, output=output, extra_flags=config.extra_flags)
    try:
        subprocess.run(list(flags), check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise XQTBackendError(f"SM90 WGMMA scaffold build failed: {exc}") from exc
    if not output.is_file():
        raise XQTBackendError(f"nvcc completed without producing artifact: {output}")
    manifest = GemmArtifactManifest(
        kernel_name=kernel_name,
        target_arch="sm_90",
        maturity="metadata_only",
        source=str(source),
        artifact=str(output),
        compile_flags=flags,
        tile_shape=(128, 128, 64),
        warp_count=4,
        stage_count=4,
        preflight=report,
        metadata={
            "build_status": "scaffold_compiled_pending_wgmma_validation",
            "correctness_verified": False,
            "native_wgmma_verified": False,
            "tma_verified": False,
            "scale_application": "fine_grained_mainloop_contract_only",
        },
    )
    manifest.write_json(output.with_suffix(output.suffix + ".manifest.json"))
    return manifest


def build_sm90_fp8_wgmma_artifact(
    config: Sm90Fp8WgmmaBuildConfig | None = None,
) -> GemmArtifactManifest:
    """Build the SM90 standalone scaffold and retain metadata-only maturity."""

    resolved = config or Sm90Fp8WgmmaBuildConfig()
    return _build(resolved, kernel_name="sm90_fp8_wgmma")


def build_sm90_grouped_fp8_wgmma_artifact(
    config: Sm90GroupedFp8WgmmaBuildConfig | None = None,
) -> GemmArtifactManifest:
    """Build the independent grouped SM90 scaffold."""

    resolved = config or Sm90Fp8WgmmaBuildConfig(
        output=_DEFAULT_GROUPED_ARTIFACT,
        grouped=True,
    )
    if not resolved.grouped:
        resolved = Sm90Fp8WgmmaBuildConfig(
            source=resolved.source,
            output=resolved.output,
            target_arch=resolved.target_arch,
            grouped=True,
            extra_flags=resolved.extra_flags,
        )
    return _build(resolved, kernel_name="sm90_grouped_fp8_wgmma")


def sm90_fp8_wgmma_artifact_available(artifact: str | Path | None = None) -> bool:
    path = Path(artifact) if artifact is not None else _DEFAULT_ARTIFACT
    return path.expanduser().is_file()


def sm90_grouped_fp8_wgmma_artifact_available(artifact: str | Path | None = None) -> bool:
    path = Path(artifact) if artifact is not None else _DEFAULT_GROUPED_ARTIFACT
    return path.expanduser().is_file()


def sm90_fp8_wgmma_executor(
    activation: torch.Tensor,
    weight: PackedWeight,
    *,
    spec: GemmSpec,
    weight_scales: torch.Tensor | None = None,
    activation_scales: torch.Tensor | None = None,
    artifact: str | Path | None = None,
) -> torch.Tensor:
    """Reject native execution until an SM90 WGMMA correctness gate exists."""

    del activation, weight, spec, weight_scales, activation_scales, artifact
    raise XQTBackendError(
        "SM90 FP8 WGMMA/TMA executor is metadata_only; target SM90 hardware "
        "and an independently validated WGMMA artifact are required"
    )


def sm90_fp8_wgmma_reference(
    activation: torch.Tensor,
    weight: PackedWeight,
    *,
    spec: GemmSpec,
    weight_scales: torch.Tensor | None = None,
    activation_scales: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reference result for the SM90 contract, for shape and scale gates."""

    return reference_gemm(
        activation,
        weight,
        spec=spec,
        weight_scales=weight_scales if weight_scales is not None else weight.scales,
        activation_scales=activation_scales,
    )


def sm90_grouped_fp8_wgmma_reference(
    grouped_problem: GroupedGemmProblem,
    activation: torch.Tensor,
    weights: Sequence[PackedWeight],
    *,
    quant: QuantSpec,
    weight_scales: Sequence[torch.Tensor | None] | None = None,
    activation_scales: torch.Tensor | Sequence[torch.Tensor | None] | None = None,
    bias: Sequence[torch.Tensor | None] | None = None,
) -> torch.Tensor:
    """Reference grouped SM90 result with empty-expert and scatter coverage."""

    return reference_packed_grouped_gemm(
        grouped_problem,
        activation,
        weights,
        quant_specs=quant,
        weight_scales=weight_scales,
        activation_scales=activation_scales,
        bias=bias,
    )


def install_sm90_fp8_wgmma_executor(*_: Any, **__: Any) -> bool:
    """Never promote the scaffold without target-specific native evidence."""

    return False


install_sm90_grouped_fp8_wgmma_executor = install_sm90_fp8_wgmma_executor


__all__ = [
    "Sm90Fp8WgmmaBuildConfig",
    "Sm90Fp8WgmmaContract",
    "Sm90GroupedFp8WgmmaBuildConfig",
    "Sm90GroupedFp8WgmmaContract",
    "build_sm90_fp8_wgmma_artifact",
    "build_sm90_grouped_fp8_wgmma_artifact",
    "install_sm90_fp8_wgmma_executor",
    "install_sm90_grouped_fp8_wgmma_executor",
    "sm90_fp8_wgmma_artifact_available",
    "sm90_fp8_wgmma_executor",
    "sm90_fp8_wgmma_reference",
    "sm90_grouped_fp8_wgmma_artifact_available",
    "sm90_grouped_fp8_wgmma_reference",
]
