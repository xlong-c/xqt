"""KV cache scale calibration (C7) - model-side artifact only.

Produces per-tensor k_scale / v_scale for attention projections. Field names
align with vLLM ``.attn.k_scale`` / ``.attn.v_scale``. Does not manage a KV
cache or attach a serving engine.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch import nn

from xqt.core.types import XQTContext
from xqt.quant.calibration.scale_artifact import run_calibration_batches
from xqt.quant.execution.reporting import optional_calibration_summary
from xqt.quant.types import QuantizationComponentPlan, QuantizationNature, QuantizationReport


_K_NAME_PATTERNS = ("k_proj", "wk", "to_k", "key_proj", "qkv")
_V_NAME_PATTERNS = ("v_proj", "wv", "to_v", "value_proj", "qkv")


@dataclass(frozen=True)
class KvScaleArtifact:
    """Per-tensor K/V scales for one attention layer (or shared prefix)."""

    layer_path: str
    k_scale: float
    v_scale: float
    num_samples: int
    observer: str = "minmax"
    qmax: int = 127

    def __post_init__(self) -> None:
        if not str(self.layer_path).strip() and self.layer_path != "":
            raise ValueError("KvScaleArtifact.layer_path must be a string")
        if float(self.k_scale) <= 0.0 or float(self.v_scale) <= 0.0:
            raise ValueError("k_scale and v_scale must be positive")
        if int(self.num_samples) <= 0:
            raise ValueError("num_samples must be positive")

    def to_dict(self) -> dict[str, Any]:
        """Serialize with vLLM-aligned field names."""

        return {
            "layer_path": self.layer_path,
            "k_scale": float(self.k_scale),
            "v_scale": float(self.v_scale),
            "attn.k_scale": float(self.k_scale),
            "attn.v_scale": float(self.v_scale),
            "num_samples": int(self.num_samples),
            "observer": self.observer,
            "qmax": int(self.qmax),
        }

    def buffer_names(self) -> dict[str, str]:
        """Module-relative buffer names matching vLLM conventions."""

        prefix = f"{self.layer_path}." if self.layer_path else ""
        return {
            "k_scale": f"{prefix}attn.k_scale",
            "v_scale": f"{prefix}attn.v_scale",
        }


def _parent_path(name: str) -> str:
    if "." not in name:
        return ""
    return name.rsplit(".", 1)[0]


def _leaf_name(name: str) -> str:
    return name.rsplit(".", 1)[-1].lower()


def _is_k_proj(name: str) -> bool:
    leaf = _leaf_name(name)
    return any(token in leaf for token in _K_NAME_PATTERNS)


def _is_v_proj(name: str) -> bool:
    leaf = _leaf_name(name)
    return any(token in leaf for token in _V_NAME_PATTERNS)


def discover_kv_projection_modules(
    model: nn.Module,
    *,
    module_names: Sequence[str] | None = None,
) -> dict[str, dict[str, str]]:
    """Map layer prefix -> {k: module_path, v: module_path}.

    Supports separate ``k_proj``/``v_proj`` and fused ``qkv`` Linear modules.
    """

    named = dict(model.named_modules())
    candidates = [
        name
        for name, module in named.items()
        if name and isinstance(module, nn.Linear)
        and (module_names is None or name in module_names)
    ]
    layers: dict[str, dict[str, str]] = {}
    for name in candidates:
        leaf = _leaf_name(name)
        parent = _parent_path(name)
        if leaf in {"qkv", "wqkv", "query_key_value"}:
            layers.setdefault(parent, {})["k"] = name
            layers.setdefault(parent, {})["v"] = name
            layers.setdefault(parent, {})["fused_qkv"] = name
            continue
        if _is_k_proj(name):
            layers.setdefault(parent, {})["k"] = name
        if _is_v_proj(name):
            layers.setdefault(parent, {})["v"] = name
    return {
        path: roles
        for path, roles in layers.items()
        if "k" in roles and "v" in roles
    }


def _symmetric_scale(max_abs: float, *, qmax: int, eps: float) -> float:
    return max(float(max_abs), float(eps)) / float(qmax)


def calibrate_kv_scales(
    model: nn.Module,
    calibration_inputs: Iterable[Any],
    *,
    module_names: Sequence[str] | None = None,
    qmax: int = 127,
    eps: float = 1e-6,
    observer: str = "minmax",
    forward_kwargs: Mapping[str, Any] | None = None,
) -> dict[str, KvScaleArtifact]:
    """Calibrate per-tensor K/V scales from projection *outputs*.

    Only ``observer="minmax"`` is supported in the first batch.
    """

    if observer != "minmax":
        raise ValueError(
            "calibrate_kv_scales currently supports observer='minmax' only; "
            f"got {observer!r}"
        )
    if int(qmax) <= 0:
        raise ValueError("qmax must be positive")

    layers = discover_kv_projection_modules(model, module_names=module_names)
    if not layers:
        return {}

    hook_targets: dict[str, str] = {}
    for layer_path, roles in layers.items():
        hook_targets[roles["k"]] = f"{layer_path}|k"
        hook_targets[roles["v"]] = f"{layer_path}|v"

    max_abs: dict[str, float] = {key: 0.0 for key in hook_targets.values()}
    sample_counts: dict[str, int] = {key: 0 for key in hook_targets.values()}
    modules = dict(model.named_modules())
    handles: list[Any] = []

    try:
        for module_path, key in hook_targets.items():
            module = modules[module_path]

            def make_hook(stat_key: str, fused: bool, role: str):
                def hook(
                    _module: nn.Module,
                    _inputs: tuple[Any, ...],
                    output: Any,
                ) -> None:
                    value = output
                    if isinstance(value, (tuple, list)):
                        value = value[0] if value else value
                    if not isinstance(value, torch.Tensor):
                        return
                    tensor = value.detach().to(dtype=torch.float32, device="cpu")
                    if fused and tensor.ndim >= 1 and tensor.shape[-1] % 3 == 0:
                        third = int(tensor.shape[-1]) // 3
                        if role == "k":
                            tensor = tensor[..., third : 2 * third]
                        else:
                            tensor = tensor[..., 2 * third :]
                    flat = tensor.reshape(-1)
                    if flat.numel() == 0:
                        return
                    current = float(flat.abs().amax().item())
                    if current > max_abs[stat_key]:
                        max_abs[stat_key] = current
                    sample_counts[stat_key] += int(flat.numel())

                return hook

            fused = "fused_qkv" in layers.get(_parent_path(module_path), {}) or (
                _leaf_name(module_path) in {"qkv", "wqkv", "query_key_value"}
            )
            role = key.rsplit("|", 1)[-1]
            handles.append(
                module.register_forward_hook(make_hook(key, fused, role))
            )

        batch_count = run_calibration_batches(
            model,
            calibration_inputs,
            forward_kwargs=forward_kwargs,
        )
    finally:
        while handles:
            handles.pop().remove()

    if batch_count <= 0:
        raise ValueError("calibrate_kv_scales requires at least one calibration batch")

    artifacts: dict[str, KvScaleArtifact] = {}
    for layer_path, roles in layers.items():
        k_key = f"{layer_path}|k"
        v_key = f"{layer_path}|v"
        if sample_counts.get(k_key, 0) <= 0 or sample_counts.get(v_key, 0) <= 0:
            continue
        artifacts[layer_path] = KvScaleArtifact(
            layer_path=layer_path,
            k_scale=_symmetric_scale(max_abs[k_key], qmax=int(qmax), eps=float(eps)),
            v_scale=_symmetric_scale(max_abs[v_key], qmax=int(qmax), eps=float(eps)),
            num_samples=min(sample_counts[k_key], sample_counts[v_key]),
            observer=observer,
            qmax=int(qmax),
        )
    return artifacts


def attach_kv_scale_buffers(
    model: nn.Module,
    artifacts: Mapping[str, KvScaleArtifact],
) -> list[str]:
    """Register ``attn.k_scale`` / ``attn.v_scale`` buffers on layer modules."""

    attached: list[str] = []
    named = dict(model.named_modules())
    for layer_path, artifact in artifacts.items():
        module = named.get(layer_path, model if not layer_path else None)
        if module is None:
            continue
        module.register_buffer(
            "k_scale",
            torch.tensor(float(artifact.k_scale), dtype=torch.float32),
        )
        module.register_buffer(
            "v_scale",
            torch.tensor(float(artifact.v_scale), dtype=torch.float32),
        )
        # vLLM-aligned aliases when the module is an attention block.
        if not hasattr(module, "attn"):
            module.register_buffer(
                "attn_k_scale",
                torch.tensor(float(artifact.k_scale), dtype=torch.float32),
            )
            module.register_buffer(
                "attn_v_scale",
                torch.tensor(float(artifact.v_scale), dtype=torch.float32),
            )
        attached.append(layer_path)
    return attached


def kv_scales_to_compute_metadata(
    artifacts: Mapping[str, KvScaleArtifact],
) -> dict[str, Any]:
    """Build ``kv_cache_quant`` metadata for ComputeConfig / quant pair."""

    return {
        "kv_cache_quant": {
            "mode": "per_tensor_scale",
            "dtype": "int8",
            "field_convention": "vllm.attn.k_scale",
            "layers": {
                path: artifact.to_dict() for path, artifact in artifacts.items()
            },
        }
    }


def evaluate_kv_scale_cosine(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    *,
    threshold: float = 0.99,
) -> dict[str, Any]:
    """Offline cos-similarity gate vs FP baseline attention outputs."""

    ref = reference.detach().to(dtype=torch.float32).reshape(-1)
    cand = candidate.detach().to(dtype=torch.float32).reshape(-1)
    if ref.numel() != cand.numel():
        raise ValueError("reference and candidate must have the same number of elements")
    if ref.numel() == 0:
        return {"cosine": 1.0, "threshold": threshold, "passed": True}
    cos = float(
        torch.nn.functional.cosine_similarity(
            ref.unsqueeze(0),
            cand.unsqueeze(0),
            dim=1,
        ).item()
    )
    return {
        "cosine": cos,
        "threshold": float(threshold),
        "passed": cos >= float(threshold),
    }


def execute_kv_scale_component(
    context: XQTContext,
    root_model: nn.Module,
    component: QuantizationComponentPlan,
) -> tuple[nn.Module, QuantizationReport]:
    """Calibrate KV scales and attach buffers; no weight quantization."""

    from xqt.quant.execution.component import (
        resolve_component_model,
        replace_component_model,
    )

    target = resolve_component_model(root_model, component.target_path)
    calibration = context.calibration_inputs
    if calibration is None:
        calibration = context.example_inputs
    if calibration is None:
        raise ValueError(
            "KV scale calibration requires context.calibration_inputs or example_inputs"
        )
    if isinstance(calibration, torch.Tensor) or not isinstance(
        calibration, (list, tuple)
    ):
        batches: list[Any] = [calibration]
    else:
        batches = list(calibration)
    sample_limit = component.policy.get("sample_limit")
    if sample_limit is not None:
        batches = batches[: int(sample_limit)]

    qmax = int(component.policy.get("qmax", 127))
    eps = float(component.policy.get("eps", 1e-6))
    artifacts = calibrate_kv_scales(
        target,
        batches,
        qmax=qmax,
        eps=eps,
        observer=str(component.policy.get("observer", "minmax")),
    )
    attached = attach_kv_scale_buffers(target, artifacts)
    updated = replace_component_model(root_model, component.target_path, target)
    calibration_samples, calibration_summary = optional_calibration_summary(
        context, component
    )
    lineage = {
        "kv_scales": {path: art.to_dict() for path, art in artifacts.items()},
        "attached_layers": attached,
        **kv_scales_to_compute_metadata(artifacts),
    }
    if calibration_summary is None:
        calibration_summary = {}
    else:
        calibration_summary = dict(calibration_summary)
    calibration_summary["kv_scale_lineage"] = lineage
    report = QuantizationReport(
        component_name=component.name,
        backend="pytorch",
        runtime="pytorch",
        method=component.method or "kv_scale",
        strategy=component.strategy or "kv_scale",
        target_path=component.target_path,
        quantized_modules=attached,
        calibration_samples=calibration_samples or len(batches),
        calibration_summary=calibration_summary,
        nature=QuantizationNature.PSEUDO,
        algorithm_executable=True,
        method_semantics="kv_cache_scale_artifact_only",
        metadata={
            "executed": True,
            "kv_scale_lineage": lineage,
            "field_convention": "vllm.attn.k_scale",
        },
    )
    return updated, report


__all__ = [
    "KvScaleArtifact",
    "attach_kv_scale_buffers",
    "calibrate_kv_scales",
    "discover_kv_projection_modules",
    "evaluate_kv_scale_cosine",
    "execute_kv_scale_component",
    "kv_scales_to_compute_metadata",
]
