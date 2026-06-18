"""Pre-export PyTorch module fusion helpers."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from torch import nn

from xqt.core.errors import XQTBackendError


@dataclass
class PreExportFusionResult:
    """Result of applying pre-export module fusion."""

    model: nn.Module
    applied: bool
    mode: str
    inplace: bool
    fused_groups: list[list[str]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def apply_pre_export_fusion(
    model: nn.Module,
    config: Mapping[str, Any] | None,
) -> PreExportFusionResult:
    """Apply optional PyTorch-side fusion before export."""

    if not config or not bool(config.get("enabled", False)):
        return PreExportFusionResult(
            model=model,
            applied=False,
            mode="disabled",
            inplace=True,
            metadata={"enabled": False},
        )

    mode = str(config.get("mode", "eager"))
    inplace = bool(config.get("inplace", False))
    target_model = model if inplace else deepcopy(model)
    target_model.eval()

    if mode == "eager":
        groups = config.get("modules_to_fuse")
        if not isinstance(groups, Sequence) or not groups:
            raise XQTBackendError(
                "pre_export_fusion.modules_to_fuse must be a non-empty list when mode=eager"
            )
        normalized: list[list[str]] = []
        for group in groups:
            if not isinstance(group, Sequence) or isinstance(group, (str, bytes)):
                raise XQTBackendError(
                    "each pre_export_fusion.modules_to_fuse entry must be a list of module names"
                )
            names = [str(name) for name in group]
            if len(names) < 2:
                raise XQTBackendError(
                    "each pre_export_fusion.modules_to_fuse entry must contain at least two module names"
                )
            normalized.append(names)
        try:
            from torch.ao.quantization import fuse_modules
        except ImportError as exc:
            raise XQTBackendError(
                "torch.ao.quantization.fuse_modules is required for eager pre-export fusion"
            ) from exc
        fused_model = fuse_modules(target_model, normalized, inplace=True)
        return PreExportFusionResult(
            model=fused_model,
            applied=True,
            mode=mode,
            inplace=inplace,
            fused_groups=normalized,
            metadata={
                "enabled": True,
                "mode": mode,
                "inplace": inplace,
                "fused_groups": normalized,
            },
        )

    if mode == "fx":
        try:
            from torch.ao.quantization.quantize_fx import fuse_fx
        except ImportError as exc:
            raise XQTBackendError(
                "torch.ao.quantization.quantize_fx.fuse_fx is required for fx pre-export fusion"
            ) from exc
        fused_model = fuse_fx(target_model)
        return PreExportFusionResult(
            model=fused_model,
            applied=True,
            mode=mode,
            inplace=inplace,
            metadata={
                "enabled": True,
                "mode": mode,
                "inplace": inplace,
                "fused_groups": [],
            },
        )

    raise XQTBackendError(f"Unsupported pre_export_fusion mode: {mode}")


__all__ = [
    "PreExportFusionResult",
    "apply_pre_export_fusion",
]
