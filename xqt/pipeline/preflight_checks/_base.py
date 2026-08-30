"""Preflight base types and generic check helpers."""

from __future__ import annotations

import importlib
import importlib.util
import shutil
from dataclasses import dataclass, field
from typing import Any


@dataclass
class PreflightCheck:
    """Single preflight check result."""

    name: str
    passed: bool
    message: str
    level: str = "info"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "message": self.message,
            "level": self.level,
            "metadata": dict(self.metadata),
        }


@dataclass
class PreflightReport:
    """Preflight report for one XQT recipe."""

    checks: list[PreflightCheck] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)

    def add(self, name: str, passed: bool, message: str, **metadata: Any) -> None:
        level = str(metadata.pop("level", "info"))
        self.checks.append(
            PreflightCheck(
                name=name,
                passed=passed,
                message=message,
                level=level,
                metadata=metadata,
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "checks": [check.to_dict() for check in self.checks],
        }


def _package_available(package_name: str) -> bool:
    try:
        return importlib.util.find_spec(package_name) is not None
    except ModuleNotFoundError:
        return False


def _check_target(report: PreflightReport, name: str, target: str | None) -> None:
    if not target:
        report.add(name, True, "target is not configured")
        return
    try:
        from xqt.core.imports import resolve_target

        resolve_target(target)
    except Exception as exc:
        report.add(name, False, f"failed to resolve target: {exc}", target=target)
        return
    report.add(name, True, "target resolved", target=target)


def _check_dependency(report: PreflightReport, package_name: str) -> None:
    available = _package_available(package_name)
    report.add(
        f"dependency.{package_name}",
        available,
        "available" if available else "missing optional dependency",
        package=package_name,
    )


def _module_metadata(module_name: str) -> dict[str, Any]:
    try:
        module = importlib.import_module(module_name)
    except Exception:
        return {}
    version = getattr(module, "__version__", None)
    metadata: dict[str, Any] = {}
    if version is not None:
        metadata["version"] = str(version)
    return metadata


def _cutile_available() -> bool:
    try:
        from xqt.kernels.ops._impl.engines.cutile import cutile_available
    except Exception:
        return _package_available("cutile")
    return cutile_available()


def _cutile_metadata() -> dict[str, Any]:
    try:
        from xqt.kernels.ops._impl.cutile._common import cutile_module_metadata
    except Exception:
        return _module_metadata("cutile")
    return cutile_module_metadata()


def _check_executable(report: PreflightReport, executable: str, name: str) -> None:
    path = shutil.which(executable)
    report.add(
        name,
        path is not None,
        f"found: {path}" if path is not None else "missing optional executable",
        executable=executable,
    )


def _check_optional_executable(
    report: PreflightReport,
    executable: str,
    name: str,
    *,
    dry_run: bool,
) -> None:
    path = shutil.which(executable)
    if path is not None:
        report.add(name, True, f"found: {path}", executable=executable, dry_run=dry_run)
        return
    report.add(
        name,
        bool(dry_run),
        "missing optional executable; dry-run command construction only"
        if dry_run
        else "missing optional executable",
        level="warning" if dry_run else "info",
        executable=executable,
        dry_run=dry_run,
    )


def _check_optional_dependency(
    report: PreflightReport,
    package_name: str,
    name: str,
    *,
    dry_run: bool,
) -> None:
    available = _package_available(package_name)
    report.add(
        name,
        available or bool(dry_run),
        "available"
        if available
        else "missing optional dependency; dry-run command construction only"
        if dry_run
        else "missing optional dependency",
        level="info" if available else "warning" if dry_run else "info",
        package=package_name,
        dry_run=dry_run,
    )
