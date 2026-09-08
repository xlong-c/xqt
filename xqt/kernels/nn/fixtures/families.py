"""Model family classification and component grouping helpers.

These helpers are shared by quant / prune / export / analysis stages so that
model-family-specific logic lives in one place and recipe entrypoints stay
thin. The smoke report explicitly marks synthetic verification and never
claims real-world performance gains.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from torch import nn

from xqt.contracts.model_structure import (
    COMPONENT_ROLES,
    ComponentSpec,
    ModelStructureContract,
    build_structure_contract,
    classify_model_family,
    component_grouping,
    model_family_names,
)
from xqt.core.base import XQTConfigError


ModelFamily = Literal[
    "transformer",
    "vit",
    "detection",
    "llm",
    "diffusion",
    "moe",
    "multimodal",
    "convnet",
    "unknown",
]

_FAMILY_NAMES = model_family_names()


@dataclass(frozen=True)
class FamilySmokeReport:
    """Honest model-family smoke verification report."""

    model_family: str
    task_type: str | None
    synthetic: bool = True
    speedup_claimed: bool = False
    performance_verified: bool = False
    module_counts: dict[str, int] = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_family": self.model_family,
            "task_type": self.task_type,
            "synthetic": self.synthetic,
            "speedup_claimed": self.speedup_claimed,
            "performance_verified": self.performance_verified,
            "module_counts": dict(self.module_counts),
            "notes": list(self.notes),
        }


def family_smoke_report(
    model: nn.Module,
    *,
    family: str | None = None,
    task_type: str | None = None,
) -> FamilySmokeReport:
    """Build a smoke verification report for one model family.

    The report always marks the verification as synthetic and never claims
    real speedup; real practice requires target data / model and hardware
    evidence outside this helper.
    """

    resolved = family or classify_model_family(model, task_type=task_type)
    counts: dict[str, int] = {}
    for _name, module in model.named_modules():
        key = type(module).__name__
        counts[key] = counts.get(key, 0) + 1
    return FamilySmokeReport(
        model_family=resolved,
        task_type=task_type,
        synthetic=True,
        speedup_claimed=False,
        performance_verified=False,
        module_counts=counts,
        notes=(
            "smoke recipe 只验证模型侧链路可运行 (量化 / 剪枝 / 导出 / benchmark), 不冒充真实性能收益.",
            "真实模型族 practice 需要目标数据与真实模型, 并在目标硬件上验证.",
        ),
    )


__all__ = [
    "FamilySmokeReport",
    "build_structure_contract",
    "classify_model_family",
    "component_grouping",
    "family_smoke_report",
    "model_family_names",
]
