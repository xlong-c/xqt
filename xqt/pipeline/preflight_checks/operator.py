"""Preflight checks for operator optimization stages."""

from __future__ import annotations

from typing import Any

import torch

from xqt.kernels.wrappers.capability import describe_operator_engine_capability
from xqt.kernels.wrappers.cuda_extension import describe_custom_cuda_extension_capability

from ._base import (
    PreflightReport,
    _check_dependency,
    _cutile_available,
    _cutile_metadata,
    _module_metadata,
    _package_available,
)
from .model import _check_cuda


def _check_operator_targets(
    report: PreflightReport,
    operator_targets: list[Any],
    *,
    default_engine: str,
    prefix: str = "operator_optimization",
) -> None:
    if not operator_targets:
        report.add(
            f"{prefix}.targets",
            False,
            "operator optimization is enabled but no targets are configured",
            level="error",
        )
        return
    report.add(
        f"{prefix}.targets",
        True,
        "operator optimization targets configured",
        count=len(operator_targets),
        default_engine=default_engine,
    )
    torch_compile_available = hasattr(torch, "compile")
    report.add(
        f"{prefix}.torch_compile",
        torch_compile_available,
        "torch.compile available"
        if torch_compile_available
        else "torch.compile unavailable",
        torch_version=torch.__version__,
    )
    target_engines = {
        getattr(target, "engine", None) or default_engine for target in operator_targets
    }
    if target_engines & {
        "triton",
        "tilelang",
        "cutile",
        "cutlass",
        "cute_dsl",
        "custom_cuda",
    }:
        _check_cuda(report, f"{prefix}.hardware.cuda")
    for package_engine in ("triton", "tilelang", "cutlass"):
        if package_engine in target_engines:
            _check_dependency(report, package_engine)
    if "cutile" in target_engines:
        available = _cutile_available()
        report.add(
            "dependency.cuda.tile",
            available,
            "available" if available else "missing optional dependency",
            package="cuda.tile",
            legacy_package="cutile",
        )
    if "cute_dsl" in target_engines:
        _check_dependency(report, "cutlass.cute")
    for index, target in enumerate(operator_targets):
        target_prefix = f"{prefix}.targets.{index}"
        target_engine = getattr(target, "engine", None) or default_engine
        capability = describe_operator_engine_capability(
            target_engine,
            torch_compile_available=torch_compile_available,
        )
        fallback_policy = str(getattr(target, "fallback_policy", "prefer_fallback"))
        fallback_policy_ok = fallback_policy in {"strict", "prefer_fallback"}
        report.add(
            f"{target_prefix}.capability",
            capability.available or capability.status == "planned",
            "operator optimization engine capability described",
            level="info"
            if capability.available or capability.status == "available"
            else "warning",
            target_name=getattr(target, "name", None),
            module_path=getattr(target, "target", None),
            **capability.to_dict(),
        )
        report.add(
            f"{target_prefix}.fallback_policy",
            fallback_policy_ok,
            "operator fallback policy recorded"
            if fallback_policy_ok
            else "operator fallback_policy must be 'strict' or 'prefer_fallback'",
            level="info" if fallback_policy_ok else "error",
            target_name=getattr(target, "name", None),
            module_path=getattr(target, "target", None),
            fallback_policy=fallback_policy,
        )
        if target_engine == "tilelang" and not _package_available("tilelang"):
            report.add(
                f"{target_prefix}.tilelang.runtime",
                True,
                "tilelang package is not importable; built-in executor will be limited to reference fallback",
                level="warning",
                target_name=getattr(target, "name", None),
                module_path=getattr(target, "target", None),
            )
        if target_engine == "cutile" and not capability.available:
            report.add(
                f"{target_prefix}.cutile.runtime",
                False,
                "cutile engine is configured but cuda.tile is not importable",
                level="warning",
                target_name=getattr(target, "name", None),
                module_path=getattr(target, "target", None),
            )
        if target_engine == "cutlass" and not capability.available:
            report.add(
                f"{target_prefix}.cutlass.runtime",
                False,
                "cutlass engine is configured but cutlass is not importable",
                level="warning",
                target_name=getattr(target, "name", None),
                module_path=getattr(target, "target", None),
            )
        if target_engine == "cute_dsl" and not capability.available:
            report.add(
                f"{target_prefix}.cute_dsl.runtime",
                False,
                "cute_dsl engine is configured but cutlass.cute is not importable",
                level="warning",
                target_name=getattr(target, "name", None),
                module_path=getattr(target, "target", None),
            )
        if target_engine == "tilelang":
            tilelang_config = getattr(target, "tilelang")
            tilelang_metadata = {
                "target_name": getattr(target, "name", None),
                "module_path": getattr(target, "target", None),
                "target": tilelang_config.target,
                "target_arch": tilelang_config.target_arch,
                "cache_dir": tilelang_config.cache_dir,
                "threads": tilelang_config.threads,
                "num_stages": tilelang_config.num_stages,
                "pass_configs": dict(tilelang_config.pass_configs),
            }
            tilelang_metadata.update(_module_metadata("tilelang"))
            report.add(
                f"{target_prefix}.tilelang.config",
                True,
                "tilelang compile configuration recorded",
                **tilelang_metadata,
            )
        if target_engine == "cutile":
            cutile_config = getattr(target, "cutile")
            cutile_metadata = {
                "target_name": getattr(target, "name", None),
                "module_path": getattr(target, "target", None),
                "target": cutile_config.target,
                "target_arch": cutile_config.target_arch,
                "cache_dir": cutile_config.cache_dir,
                "threads": cutile_config.threads,
                "pass_configs": dict(cutile_config.pass_configs),
            }
            cutile_metadata.update(_cutile_metadata())
            report.add(
                f"{target_prefix}.cutile.config",
                True,
                "cutile compile configuration recorded",
                **cutile_metadata,
            )
        if target_engine == "cutlass":
            cutlass_config = getattr(target, "cutlass")
            cutlass_metadata = {
                "target_name": getattr(target, "name", None),
                "module_path": getattr(target, "target", None),
                "target_arch": cutlass_config.target_arch,
                "cache_dir": cutlass_config.cache_dir,
                "tile_shape": list(cutlass_config.tile_shape),
                "cluster_shape": (
                    list(cutlass_config.cluster_shape)
                    if cutlass_config.cluster_shape is not None
                    else None
                ),
                "pass_configs": dict(cutlass_config.pass_configs),
            }
            cutlass_metadata.update(_module_metadata("cutlass"))
            report.add(
                f"{target_prefix}.cutlass.config",
                True,
                "cutlass compile configuration recorded",
                **cutlass_metadata,
            )
        if target_engine == "cute_dsl":
            cute_dsl_config = getattr(target, "cute_dsl")
            cute_dsl_metadata = {
                "target_name": getattr(target, "name", None),
                "module_path": getattr(target, "target", None),
                "target_arch": cute_dsl_config.target_arch,
                "cache_dir": cute_dsl_config.cache_dir,
                "tile_shape": list(cute_dsl_config.tile_shape),
                "cluster_shape": (
                    list(cute_dsl_config.cluster_shape)
                    if cute_dsl_config.cluster_shape is not None
                    else None
                ),
                "pass_configs": dict(cute_dsl_config.pass_configs),
            }
            cute_dsl_metadata.update(_module_metadata("cutlass.cute"))
            report.add(
                f"{target_prefix}.cute_dsl.config",
                True,
                "cute_dsl compile configuration recorded",
                **cute_dsl_metadata,
            )
        if target_engine == "custom_cuda":
            extension = describe_custom_cuda_extension_capability()
            report.add(
                f"{target_prefix}.custom_cuda.extension",
                extension.available,
                "custom CUDA extension capability described",
                level="info" if extension.available else "warning",
                target_name=getattr(target, "name", None),
                module_path=getattr(target, "target", None),
                **extension.to_dict(),
            )
