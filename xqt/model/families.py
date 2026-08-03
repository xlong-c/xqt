"""Model family classification and component grouping helpers.

These helpers are shared by quant / prune / export / analysis stages so that
model-family-specific logic lives in one place and recipe entrypoints stay
thin. The smoke report explicitly marks synthetic verification and never
claims real-world performance gains.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Sequence

from torch import nn


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


_FAMILY_NAMES: tuple[str, ...] = (
    "transformer",
    "vit",
    "detection",
    "llm",
    "diffusion",
    "moe",
    "multimodal",
    "convnet",
    "unknown",
)

_TASK_TO_FAMILY: dict[str, str] = {
    "detection": "detection",
    "llm": "llm",
    "text_generation": "llm",
    "diffusion": "diffusion",
    "image_generation": "diffusion",
    "multimodal": "multimodal",
    "vlm": "multimodal",
    "classification": "transformer",
}

_DETECTION_HEAD_MARKERS = ("head", "detect", "yolo", "rtdetr")
_EXPERT_MARKERS = ("experts", "expert", "moe")
_ROUTER_MARKERS = ("router", "gating", "gate")
_DIFFUSION_MARKERS = ("unet", "dit", "vae", "text_encoder", "denoiser")
_MULTIMODAL_MARKERS = ("vision_encoder", "visual_encoder", "encoder_cache", "cross_attn")


def model_family_names() -> tuple[str, ...]:
    """Return the canonical model family vocabulary."""

    return _FAMILY_NAMES


def _module_paths(model: nn.Module) -> list[str]:
    return [name for name, _module in model.named_modules()]


def _module_types(model: nn.Module) -> list[str]:
    return [type(module).__name__ for _name, module in model.named_modules()]


def _contains_attention(paths: Sequence[str]) -> bool:
    return any(
        marker in path.lower()
        for marker in (
            "attention",
            "attn",
            "encoder",
            "q_proj",
            "k_proj",
            "v_proj",
            "out_proj",
        )
        for path in paths
    )


def _contains_conv(model: nn.Module) -> bool:
    return any(
        isinstance(module, nn.Conv2d)
        for _name, module in model.named_modules()
    )


def classify_model_family(
    model: nn.Module,
    *,
    task_type: str | None = None,
) -> str:
    """Classify a model into a canonical XQT model family.

    ``task_type`` (from ``TaskConfig.type``) takes precedence when it maps to a
    known family; otherwise structural heuristics on module names are used.
    """

    if task_type:
        normalized = str(task_type).strip().lower()
        if normalized in _TASK_TO_FAMILY:
            return _TASK_TO_FAMILY[normalized]

    paths = _module_paths(model)
    lowered = [path.lower() for path in paths]
    joined = " ".join(lowered)

    if any(marker in joined for marker in _EXPERT_MARKERS) and any(
        marker in joined for marker in _ROUTER_MARKERS
    ):
        return "moe"
    if any(marker in joined for marker in _DIFFUSION_MARKERS):
        return "diffusion"
    if any(marker in joined for marker in _MULTIMODAL_MARKERS):
        return "multimodal"
    has_detection_head = any(
        marker in joined for marker in ("detect", "yolo", "rtdetr", "detection_head")
    ) or (
        "head" in joined
        and "patch_embed" not in joined
        and "lm_head" not in joined
        and "blocks" not in joined
    )
    if has_detection_head and _contains_conv(model):
        return "detection"
    if "lm_head" in joined or (
        "q_proj" in joined and "k_proj" in joined and "v_proj" in joined
    ):
        return "llm"
    if "patch_embed" in joined or ("vit" in joined and _contains_attention(paths)):
        return "vit"
    if _contains_attention(paths) and "norm" in joined:
        return "transformer"
    if _contains_conv(model):
        return "convnet"
    return "unknown"


def component_grouping(
    model: nn.Module,
    *,
    family: str | None = None,
    task_type: str | None = None,
) -> dict[str, list[str]]:
    """Group module paths by model-family component role.

    The returned mapping is a suggestion for quant / prune policy defaults
    (for example keep the head and router in high precision); it is not an
    automatic policy rewrite.
    """

    resolved = family or classify_model_family(model, task_type=task_type)
    groups: dict[str, list[str]] = {
        "attention": [],
        "ffn": [],
        "norm": [],
        "head": [],
        "backbone": [],
        "expert": [],
        "router": [],
        "embedding": [],
        "encoder": [],
        "cross_attention": [],
        "diffusion_component": [],
        "other": [],
    }
    for name, module in model.named_modules():
        if not name:
            continue
        lowered = name.lower()
        module_type = type(module).__name__
        if any(marker in lowered for marker in _ROUTER_MARKERS):
            groups["router"].append(name)
        elif any(marker in lowered for marker in _DIFFUSION_MARKERS):
            groups["diffusion_component"].append(name)
        elif any(
            marker in lowered
            for marker in ("vision_encoder", "visual_encoder", "encoder_cache")
        ):
            groups["encoder"].append(name)
        elif any(marker in lowered for marker in ("cross_attn", "cross_attention")):
            groups["cross_attention"].append(name)
        elif any(marker in lowered for marker in _EXPERT_MARKERS):
            groups["expert"].append(name)
        elif isinstance(module, (nn.Embedding,)):
            groups["embedding"].append(name)
        elif any(marker in lowered for marker in _DETECTION_HEAD_MARKERS):
            groups["head"].append(name)
        elif isinstance(module, (nn.LayerNorm,)) or "rmsnorm" in module_type.lower():
            groups["norm"].append(name)
        elif any(
            marker in lowered
            for marker in ("attention", "attn", "q_proj", "k_proj", "v_proj", "out_proj")
        ):
            groups["attention"].append(name)
        elif any(marker in lowered for marker in ("ffn", "mlp", "feed_forward")):
            groups["ffn"].append(name)
        elif isinstance(module, (nn.Conv2d,)):
            groups["backbone"].append(name)
        else:
            groups["other"].append(name)
    return groups


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
    "classify_model_family",
    "component_grouping",
    "family_smoke_report",
    "model_family_names",
]
