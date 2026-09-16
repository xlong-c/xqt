"""CUDA/CUTLASS toolchain preflight and reproducible artifact metadata.

This module only probes and records build inputs.  It never compiles a kernel
and never labels a metadata artifact executable.  A later build step may use
the returned include path and compile flags after its own correctness gate.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Sequence


_ARCH_RE = re.compile(r"^sm_(?P<sm>[0-9]+)$")
_COMPILE_ARCH_RE = re.compile(r"^sm_(?P<sm>[0-9]+)(?P<variant>[af])?$")


def _normalize_arch(value: str | int) -> str:
    if isinstance(value, bool):
        raise ValueError("target SM must be an integer or sm_<number> string")
    if isinstance(value, int):
        sm = value
    else:
        match = _ARCH_RE.fullmatch(str(value).strip())
        if match is None:
            raise ValueError(f"target_arch must look like sm_89, got {value!r}")
        sm = int(match.group("sm"))
    if sm < 50:
        raise ValueError(f"target SM must be >= 50, got {sm}")
    return f"sm_{sm}"


def _normalize_compile_arch(value: str | int, *, target_arch: str) -> str:
    """Normalize a CUDA codegen target while preserving the logical SM."""

    if isinstance(value, bool):
        raise ValueError("compile target SM must be an integer or sm_<number>[a|f] string")
    if isinstance(value, int):
        compile_arch = f"sm_{value}"
    else:
        compile_arch = str(value).strip()
    match = _COMPILE_ARCH_RE.fullmatch(compile_arch)
    if match is None:
        raise ValueError(
            "compile target_arch must look like sm_89, sm_120a, or sm_120f"
        )
    if int(match.group("sm")) != int(target_arch[3:]):
        raise ValueError(
            f"compile target {compile_arch} does not match logical target {target_arch}"
        )
    return compile_arch


def _command_output(command: Sequence[str]) -> str | None:
    try:
        completed = subprocess.run(
            list(command),
            check=False,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    output = (completed.stdout or "") + (completed.stderr or "")
    return output.strip() or None


def _version_from_output(output: str | None) -> str | None:
    if not output:
        return None
    for line in output.splitlines():
        if "release" in line.lower() or "version" in line.lower():
            return line.strip()
    return output.splitlines()[0].strip()


def _find_cutlass_include(repo_root: Path | None = None) -> Path | None:
    """Locate a CUTLASS include directory without requiring a build.

    Resolution order: explicit ``XQT_CUTLASS_INCLUDE``, the caller-supplied
    repository root, this checkout's own ``third_party/cutlass``, then the
    CUTLASS bundled by an installed TileLang / CUDA-Tile package.
    """

    candidates: list[Path] = []
    env_path = os.environ.get("XQT_CUTLASS_INCLUDE")
    if env_path:
        candidates.append(Path(env_path))
    if repo_root is not None:
        candidates.append(repo_root / "third_party" / "cutlass" / "include")
    # parents[3] is the checkout root when xqt is a standalone repository;
    # parents[4] is the root when xqt is nested inside the XDL monorepo.
    for ancestor in Path(__file__).resolve().parents[3:5]:
        candidates.append(ancestor / "third_party" / "cutlass" / "include")
    for package_name in ("tilelang", "cuda_tile"):
        module_spec = importlib.util.find_spec(package_name)
        if module_spec is None or module_spec.origin is None:
            continue
        package_root = Path(module_spec.origin).resolve().parent
        candidates.append(package_root / "3rdparty" / "cutlass" / "include")
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if (resolved / "cutlass").is_dir() or (resolved / "cute").is_dir():
            return resolved
    return None


def _cutlass_python_version() -> tuple[str | None, str | None]:
    module_spec = importlib.util.find_spec("cutlass")
    if module_spec is None:
        return None, None
    module_path = None if module_spec.origin is None else str(Path(module_spec.origin).resolve())
    try:
        import cutlass  # type: ignore[import-not-found]
    except (ImportError, RuntimeError):
        return "import_error", module_path
    return str(getattr(cutlass, "__version__", "unknown")), module_path


def _host_compiler_version() -> str | None:
    compiler = os.environ.get("CXX") or shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        return None
    return _version_from_output(_command_output([compiler, "--version"]))


@dataclass(frozen=True, slots=True)
class GemmPreflightReport:
    """Structured probe result used by build scripts and dispatch diagnostics."""

    target_arch: str
    status: str
    nvcc_path: str | None
    nvcc_version: str | None
    cuda_runtime_version: str | None
    compiler_version: str | None
    cutlass_version: str | None
    cutlass_python_path: str | None
    cutlass_include: str | None
    device_name: str | None
    device_arch: str | None
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.status not in {"ready", "partial", "unavailable"}:
            raise ValueError("preflight status must be ready, partial, or unavailable")

    @property
    def ready_for_compile(self) -> bool:
        """Whether nvcc and CUTLASS headers are sufficient for compilation.

        A host GPU with a different SM is an execution limitation, not a
        cross-compilation limitation.  Executable promotion still requires a
        fully ready target-device preflight.
        """

        return self.nvcc_path is not None and self.cutlass_include is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_arch": self.target_arch,
            "status": self.status,
            "ready_for_compile": self.ready_for_compile,
            "nvcc_path": self.nvcc_path,
            "nvcc_version": self.nvcc_version,
            "cuda_runtime_version": self.cuda_runtime_version,
            "compiler_version": self.compiler_version,
            "cutlass_version": self.cutlass_version,
            "cutlass_python_path": self.cutlass_python_path,
            "cutlass_include": self.cutlass_include,
            "device_name": self.device_name,
            "device_arch": self.device_arch,
            "reasons": list(self.reasons),
        }


def probe_cuda_cutlass(
    target_arch: str | int = "sm_89",
    *,
    repo_root: str | Path | None = None,
    nvcc: str | None = None,
    require_device: bool = False,
) -> GemmPreflightReport:
    """Probe toolchain and optional device without importing a backend kernel."""

    normalized_arch = _normalize_arch(target_arch)
    root = None if repo_root is None else Path(repo_root).expanduser().resolve()
    nvcc_path = nvcc or os.environ.get("NVCC") or shutil.which("nvcc")
    nvcc_version = _version_from_output(
        _command_output([nvcc_path, "--version"]) if nvcc_path is not None else None
    )
    compiler_version = _host_compiler_version()
    cutlass_version, cutlass_python_path = _cutlass_python_version()
    include = _find_cutlass_include(root)
    cuda_runtime_version: str | None = None
    device_name: str | None = None
    device_arch: str | None = None
    reasons: list[str] = []
    try:
        import torch

        cuda_runtime_version = torch.version.cuda
        if torch.cuda.is_available():
            device_name = str(torch.cuda.get_device_name(0))
            major, minor = torch.cuda.get_device_capability(0)
            device_arch = f"sm_{major}{minor}"
    except (ImportError, RuntimeError) as exc:
        reasons.append(f"torch CUDA probe failed: {exc}")
    if nvcc_path is None:
        reasons.append("nvcc was not found")
    if nvcc_path is not None and nvcc_version is None:
        reasons.append("nvcc exists but --version returned no output")
    if include is None:
        reasons.append(
            "CUTLASS C++ headers were not found; set XQT_CUTLASS_INCLUDE to an include directory"
        )
    if cutlass_version is None:
        reasons.append("Python CUTLASS DSL package is not importable")
    if require_device and device_arch is None:
        reasons.append("a CUDA device is required but torch.cuda.is_available() is false")
    if device_arch is not None and device_arch != normalized_arch:
        reasons.append(f"device arch {device_arch} does not match target {normalized_arch}")
    mandatory_missing = nvcc_path is None or include is None
    status = "unavailable" if mandatory_missing else ("partial" if reasons else "ready")
    return GemmPreflightReport(
        target_arch=normalized_arch,
        status=status,
        nvcc_path=None if nvcc_path is None else str(Path(nvcc_path).resolve()),
        nvcc_version=nvcc_version,
        cuda_runtime_version=cuda_runtime_version,
        compiler_version=compiler_version,
        cutlass_version=cutlass_version,
        cutlass_python_path=cutlass_python_path,
        cutlass_include=None if include is None else str(include),
        device_name=device_name,
        device_arch=device_arch,
        reasons=tuple(reasons),
    )


@dataclass(frozen=True, slots=True)
class GemmArtifactManifest:
    """Build inputs and maturity for one compiled GEMM artifact."""

    kernel_name: str
    target_arch: str
    maturity: str
    source: str
    artifact: str | None
    compile_flags: tuple[str, ...]
    tile_shape: tuple[int, int, int] | None
    warp_count: int | None
    stage_count: int | None
    preflight: GemmPreflightReport
    schema_version: int = 1
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.maturity not in {"executable", "reference_guarded", "metadata_only", "planned"}:
            raise ValueError(f"unsupported artifact maturity: {self.maturity!r}")
        if self.maturity == "executable" and not self.artifact:
            raise ValueError("executable artifact must specify artifact path")
        if self.target_arch != self.preflight.target_arch:
            raise ValueError("manifest target_arch must match preflight target_arch")

    @property
    def correctness_verified(self) -> bool:
        """Whether an independent numeric gate has promoted this artifact."""

        value = self.metadata.get("correctness_verified", False)
        return value is True and isinstance(self.metadata.get("correctness"), Mapping)

    @property
    def executable_ready(self) -> bool:
        """Whether this manifest is safe to promote into the native registry."""

        if self.maturity != "executable" or not self.correctness_verified:
            return False
        if not self.artifact or not Path(self.artifact).expanduser().is_file():
            return False
        return self.preflight.status == "ready"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "kernel_name": self.kernel_name,
            "target_arch": self.target_arch,
            "maturity": self.maturity,
            "source": self.source,
            "artifact": self.artifact,
            "compile_flags": list(self.compile_flags),
            "tile_shape": None if self.tile_shape is None else list(self.tile_shape),
            "warp_count": self.warp_count,
            "stage_count": self.stage_count,
            "preflight": self.preflight.to_dict(),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "GemmArtifactManifest":
        """Load a manifest without inferring or repairing missing fields."""

        if not isinstance(payload, Mapping):
            raise TypeError("GemmArtifactManifest.from_dict expects a mapping")
        preflight_payload = payload.get("preflight")
        if not isinstance(preflight_payload, Mapping):
            raise ValueError("artifact manifest requires a preflight mapping")
        preflight = GemmPreflightReport(
            target_arch=str(preflight_payload["target_arch"]),
            status=str(preflight_payload["status"]),
            nvcc_path=preflight_payload.get("nvcc_path"),
            nvcc_version=preflight_payload.get("nvcc_version"),
            cuda_runtime_version=preflight_payload.get("cuda_runtime_version"),
            compiler_version=preflight_payload.get("compiler_version"),
            cutlass_version=preflight_payload.get("cutlass_version"),
            cutlass_python_path=preflight_payload.get("cutlass_python_path"),
            cutlass_include=preflight_payload.get("cutlass_include"),
            device_name=preflight_payload.get("device_name"),
            device_arch=preflight_payload.get("device_arch"),
            reasons=tuple(str(item) for item in preflight_payload.get("reasons", [])),
        )
        tile_shape = payload.get("tile_shape")
        return cls(
            schema_version=int(payload.get("schema_version", 1)),
            kernel_name=str(payload["kernel_name"]),
            target_arch=str(payload["target_arch"]),
            maturity=str(payload["maturity"]),
            source=str(payload["source"]),
            artifact=None if payload.get("artifact") is None else str(payload["artifact"]),
            compile_flags=tuple(str(item) for item in payload.get("compile_flags", [])),
            tile_shape=None if tile_shape is None else tuple(int(item) for item in tile_shape),
            warp_count=None if payload.get("warp_count") is None else int(payload["warp_count"]),
            stage_count=None
            if payload.get("stage_count") is None
            else int(payload["stage_count"]),
            preflight=preflight,
            metadata=dict(payload.get("metadata", {})),
        )

    @classmethod
    def load_json(cls, path: str | Path) -> "GemmArtifactManifest":
        """Read one manifest from JSON."""

        source = Path(path).expanduser()
        with source.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return cls.from_dict(payload)

    def promote_correctness(self, evidence: Mapping[str, Any]) -> "GemmArtifactManifest":
        """Return an executable manifest after recording numeric evidence."""

        if not isinstance(evidence, Mapping) or not evidence:
            raise ValueError("correctness evidence must be a non-empty mapping")
        metadata = dict(self.metadata)
        metadata["correctness_verified"] = True
        metadata["correctness"] = dict(evidence)
        metadata["build_status"] = "compiled_and_correctness_verified"
        promoted = replace(self, maturity="executable", metadata=metadata)
        if not promoted.executable_ready:
            raise ValueError("correctness promotion requires an existing artifact and ready preflight")
        return promoted

    def write_json(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        return output


def artifact_manifest_path(artifact: str | Path) -> Path:
    """Return the sidecar path used by all GEMM build artifacts."""

    return Path(f"{Path(artifact).expanduser()}.manifest.json")


def load_artifact_manifest(path_or_artifact: str | Path) -> GemmArtifactManifest:
    """Load a manifest from a sidecar path or from its artifact path."""

    candidate = Path(path_or_artifact).expanduser()
    if candidate.name.endswith(".manifest.json"):
        manifest_path = candidate
    else:
        manifest_path = artifact_manifest_path(candidate)
    return GemmArtifactManifest.load_json(manifest_path)


def artifact_ready_for_execution(
    path_or_manifest: str | Path | GemmArtifactManifest,
    *,
    kernel_name: str | None = None,
    target_arch: str | None = None,
) -> bool:
    """Check artifact existence, manifest identity and correctness promotion."""

    try:
        manifest = (
            path_or_manifest
            if isinstance(path_or_manifest, GemmArtifactManifest)
            else load_artifact_manifest(path_or_manifest)
        )
    except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError):
        return False
    if kernel_name is not None and manifest.kernel_name != kernel_name:
        return False
    if target_arch is not None and manifest.target_arch != target_arch:
        return False
    return manifest.executable_ready


def promote_artifact_manifest(
    path_or_artifact: str | Path,
    *,
    evidence: Mapping[str, Any],
    output_path: str | Path | None = None,
) -> GemmArtifactManifest:
    """Persist numeric correctness evidence and return the promoted manifest."""

    manifest = load_artifact_manifest(path_or_artifact)
    promoted = manifest.promote_correctness(evidence)
    destination = (
        Path(output_path).expanduser()
        if output_path is not None
        else artifact_manifest_path(manifest.artifact or path_or_artifact)
    )
    promoted.write_json(destination)
    return promoted


def default_cache_dir() -> Path:
    """Return the user-local cache path without mutating it."""

    configured = os.environ.get("XQT_GEMM_CACHE")
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".cache" / "xqt" / "gemm"


def build_compile_flags(
    preflight: GemmPreflightReport,
    *,
    source: str | Path,
    output: str | Path,
    extra_flags: Sequence[str] = (),
    compile_target_arch: str | int | None = None,
) -> tuple[str, ...]:
    """Build deterministic nvcc flags for a logical target and its codegen."""

    if preflight.nvcc_path is None or preflight.cutlass_include is None:
        raise RuntimeError("cannot build compile flags before CUDA/CUTLASS preflight is ready")
    codegen_arch = _normalize_compile_arch(
        preflight.target_arch if compile_target_arch is None else compile_target_arch,
        target_arch=preflight.target_arch,
    )
    cutlass_include = Path(preflight.cutlass_include)
    utility_include = cutlass_include.parent / "tools" / "util" / "include"
    include_flags = ["-I", str(cutlass_include)]
    if utility_include.is_dir():
        include_flags.extend(["-I", str(utility_include)])
    return tuple(
        [
            preflight.nvcc_path,
            "-O3",
            "-std=c++17",
            "-shared",
            "-Xcompiler",
            "-fPIC",
            f"-gencode=arch=compute_{codegen_arch[3:]},code={codegen_arch}",
            *include_flags,
            str(source),
            "-o",
            str(output),
            *[str(flag) for flag in extra_flags],
        ]
    )


__all__ = [
    "artifact_manifest_path",
    "artifact_ready_for_execution",
    "GemmArtifactManifest",
    "GemmPreflightReport",
    "build_compile_flags",
    "default_cache_dir",
    "load_artifact_manifest",
    "probe_cuda_cutlass",
    "promote_artifact_manifest",
]
