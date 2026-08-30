"""Unlimited-OCR Half baseline and ConvRot W8A8 model-side helpers.

The upstream Unlimited-OCR repository owns image preprocessing and generation.
This module deliberately does not duplicate those APIs.  Callers provide one
real ``model.infer(...)`` invocation for static activation calibration, then
this module replaces only the selected ``nn.Linear`` modules with XQT's
``ConvRotInt8Linear`` implementation.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
import re
from typing import Any, Callable, Mapping

import torch
from torch import nn

from xqt.compression.quant.policy import QuantizationPolicy, should_quantize_module
from xqt.compression.quant.quantizers.convrot_4bit import (
    _apply_groupwise_rotation,
    _normalize_rot_size,
)
from xqt.compression.quant.quantizers.convrot_int8 import (
    ConvRotInt8QuantizationResult,
    quantize_with_convrot_int8,
)


UNLIMITED_OCR_REPO_ID = "baidu/Unlimited-OCR"
UNLIMITED_OCR_CONVROT_INT8_STRATEGY = "w8a8_int8"

# The vision stack and output head are kept at Half by default.  The remaining
# Linear modules include the language model's attention, MLP, and routed-expert
# projections, whose input feature sizes satisfy ConvRot's regular Hadamard
# group constraints on the published checkpoint.
_DEFAULT_EXCLUDE_NAME_PATTERNS = (
    r"^model\.vision_model(?:\.|$)",
    r"^model\.sam_model(?:\.|$)",
    r"^model\.projector(?:\.|$)",
    r"lm_head$",
    r"\.gate$",
)

CalibrationCall = Callable[[nn.Module], Any]


@dataclass(frozen=True)
class UnlimitedOcrConvRotCalibration:
    """Static rotated-activation scales collected from one real OCR call."""

    activation_scales: dict[str, float]
    target_module_names: tuple[str, ...]
    observed_module_names: tuple[str, ...]
    rot_size_by_module: dict[str, int]
    min_input_rows_by_module: dict[str, int]
    max_input_rows_by_module: dict[str, int]

    @property
    def missing_module_names(self) -> tuple[str, ...]:
        """Return selected modules not reached by the calibration invocation."""

        observed = set(self.observed_module_names)
        return tuple(name for name in self.target_module_names if name not in observed)

    def int8_eligible_module_names(self, min_input_rows: int) -> tuple[str, ...]:
        """Return modules that never fell below the true-INT8 row threshold."""

        threshold = max(0, int(min_input_rows))
        return tuple(
            name
            for name in self.target_module_names
            if name in self.min_input_rows_by_module
            and int(self.min_input_rows_by_module[name]) >= threshold
        )

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe calibration metadata without serializing all scales."""

        rot_size_counts: dict[str, int] = {}
        max_input_row_counts: dict[str, int] = {}
        min_input_row_counts: dict[str, int] = {}
        for rot_size in self.rot_size_by_module.values():
            key = str(rot_size)
            rot_size_counts[key] = rot_size_counts.get(key, 0) + 1
        for rows in self.max_input_rows_by_module.values():
            key = str(rows)
            max_input_row_counts[key] = max_input_row_counts.get(key, 0) + 1
        for rows in self.min_input_rows_by_module.values():
            key = str(rows)
            min_input_row_counts[key] = min_input_row_counts.get(key, 0) + 1
        return {
            "scale_mode": "static_signed_int8_per_tensor_rotated",
            "target_module_count": len(self.target_module_names),
            "observed_module_count": len(self.observed_module_names),
            "missing_module_count": len(self.missing_module_names),
            "missing_module_names": list(self.missing_module_names),
            "rot_size_counts": rot_size_counts,
            "min_input_row_counts": min_input_row_counts,
            "max_input_row_counts": max_input_row_counts,
        }


@dataclass(frozen=True)
class UnlimitedOcrConvRotInt8Result:
    """Materialized Unlimited-OCR ConvRot model and its calibration lineage."""

    model: nn.Module
    quantization: ConvRotInt8QuantizationResult
    calibration: UnlimitedOcrConvRotCalibration | None

    @property
    def quantized_modules(self) -> list[str]:
        """Expose replaced module paths from the underlying XQT quantizer."""

        return self.quantization.quantized_modules

    def to_dict(self) -> dict[str, Any]:
        """Return compact JSON-safe model optimization metadata."""

        return {
            "repo_id": UNLIMITED_OCR_REPO_ID,
            "method": "convrot",
            "strategy": UNLIMITED_OCR_CONVROT_INT8_STRATEGY,
            "quantized_module_count": len(self.quantized_modules),
            "quantization_metadata": dict(self.quantization.metadata),
            "calibration": None
            if self.calibration is None
            else self.calibration.to_dict(),
        }


class _FP16TorchProxy:
    """Expose ``torch.float16`` where upstream remote code requests bf16."""

    def __init__(self, torch_module: Any) -> None:
        self._torch_module = torch_module

    def __getattr__(self, name: str) -> Any:
        if name == "bfloat16":
            return torch.float16
        return getattr(self._torch_module, name)


def unlimited_ocr_convrot_default_policy() -> dict[str, Any]:
    """Return the documented Half + ConvRot W8A8 selection policy."""

    return {
        "dtype": "int8",
        "scheme": "convrot_w8a8",
        "include_module_types": ["Linear"],
        "exclude_name_patterns": list(_DEFAULT_EXCLUDE_NAME_PATTERNS),
        "min_parameters": 65536,
    }


def _merged_policy(policy: Mapping[str, Any] | None) -> dict[str, Any]:
    merged = unlimited_ocr_convrot_default_policy()
    if policy is not None:
        merged.update(dict(policy))
    return merged


def _selection_policy(policy: Mapping[str, Any]) -> QuantizationPolicy:
    """Convert the public mapping into the generic XQT module filter."""

    return QuantizationPolicy(
        dtype=str(policy.get("dtype", "int8")),
        scheme=str(policy.get("scheme", "convrot_w8a8")),
        include_module_types=tuple(
            str(item) for item in policy.get("include_module_types", ())
        ),
        exclude_module_types=tuple(
            str(item) for item in policy.get("exclude_module_types", ())
        ),
        include_name_patterns=tuple(
            str(item) for item in policy.get("include_name_patterns", ())
        ),
        exclude_name_patterns=tuple(
            str(item) for item in policy.get("exclude_name_patterns", ())
        ),
        include_module_names=tuple(
            str(item) for item in policy.get("include_module_names", ())
        ),
        exclude_module_names=tuple(
            str(item) for item in policy.get("exclude_module_names", ())
        ),
        min_parameters=int(policy.get("min_parameters", 0)),
    )


def select_unlimited_ocr_convrot_modules(
    model: nn.Module,
    *,
    policy: Mapping[str, Any] | None = None,
) -> tuple[str, ...]:
    """Select published Unlimited-OCR Linear modules eligible for ConvRot."""

    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    policy_mapping = _merged_policy(policy)
    effective_policy = _selection_policy(policy_mapping)
    selection_mode = (
        str(policy_mapping.get("selection_mode", "default")).strip().lower()
    )
    if selection_mode not in {"default", "include_only"}:
        raise ValueError("selection_mode must be 'default' or 'include_only'")
    include_names = set(effective_policy.include_module_names)
    include_patterns = tuple(effective_policy.include_name_patterns)
    selected: list[str] = []
    for name, module in model.named_modules():
        if not name or not isinstance(module, nn.Linear):
            continue
        if selection_mode == "include_only":
            included = name in include_names or any(
                re.search(pattern, name) for pattern in include_patterns
            )
            if not included:
                continue
        if should_quantize_module(name, module, effective_policy):
            selected.append(name)
    return tuple(selected)


def calibrate_unlimited_ocr_convrot_activation_scales(
    model: nn.Module,
    *,
    calibration_call: CalibrationCall,
    policy: Mapping[str, Any] | None = None,
    rot_size: int = 256,
    eps: float = 1e-6,
) -> UnlimitedOcrConvRotCalibration:
    """Collect static per-tensor scales after the same rotation used at runtime.

    ``calibration_call`` must invoke the official model API, normally one real
    ``model.infer(...)`` call.  It gives the remote checkpoint ownership of
    image preprocessing, prompt construction, cache progression, and decode
    semantics while XQT observes only selected Linear inputs.
    """

    if not callable(calibration_call):
        raise TypeError("calibration_call must be callable")
    if float(eps) <= 0.0:
        raise ValueError("eps must be positive")

    target_names = select_unlimited_ocr_convrot_modules(model, policy=policy)
    if not target_names:
        raise ValueError("Unlimited-OCR ConvRot policy selected no nn.Linear modules")
    rot_size_by_module = {
        name: _normalize_rot_size(
            int(rot_size),
            int(model.get_submodule(name).in_features),
        )
        for name in target_names
    }
    maxima: dict[str, float] = {}
    min_input_rows_by_module: dict[str, int] = {}
    max_input_rows_by_module: dict[str, int] = {}
    handles: list[torch.utils.hooks.RemovableHandle] = []

    def make_hook(name: str) -> Callable[[nn.Module, tuple[Any, ...], Any], None]:
        def hook(_module: nn.Module, inputs: tuple[Any, ...], _output: Any) -> None:
            if not inputs or not isinstance(inputs[0], torch.Tensor):
                return
            values = inputs[0].detach()
            if values.numel() == 0:
                return
            rows = int(values.reshape(-1, values.shape[-1]).shape[0])
            previous_min = min_input_rows_by_module.get(name)
            min_input_rows_by_module[name] = (
                rows if previous_min is None else min(previous_min, rows)
            )
            max_input_rows_by_module[name] = max(
                max_input_rows_by_module.get(name, 0),
                rows,
            )
            rotated = _apply_groupwise_rotation(
                values.to(torch.float32),
                rot_size=rot_size_by_module[name],
            )
            maximum = float(rotated.abs().amax().item())
            maxima[name] = max(maxima.get(name, 0.0), maximum)

        return hook

    try:
        for name in target_names:
            module = model.get_submodule(name)
            handles.append(module.register_forward_hook(make_hook(name)))
        with torch.inference_mode():
            calibration_call(model)
    finally:
        for handle in handles:
            handle.remove()

    activation_scales = {
        name: max(float(value) / 127.0, float(eps)) for name, value in maxima.items()
    }
    return UnlimitedOcrConvRotCalibration(
        activation_scales=activation_scales,
        target_module_names=target_names,
        observed_module_names=tuple(activation_scales),
        rot_size_by_module=rot_size_by_module,
        min_input_rows_by_module=min_input_rows_by_module,
        max_input_rows_by_module=max_input_rows_by_module,
    )


def force_unlimited_ocr_fp16_runtime(model: nn.Module) -> None:
    """Patch only the loaded remote module's hard-coded bf16 requests to fp16.

    The published remote code builds image tensors and autocast scopes using
    ``torch.bfloat16`` even when the checkpoint itself is loaded as fp16.  The
    XQT sm_89 CUTLASS ConvRot route requires fp16 output, so the Half baseline
    and quantized model must share this correction.  The proxy is scoped to the
    remote module object, not the process-wide ``torch`` module.
    """

    module_name = type(model).__module__
    remote_module = sys.modules.get(module_name)
    if remote_module is None:
        raise RuntimeError(f"Unlimited-OCR remote module is not loaded: {module_name}")
    remote_torch = getattr(remote_module, "torch", None)
    if remote_torch is None:
        raise RuntimeError(
            f"Unlimited-OCR remote module has no torch binding: {module_name}"
        )
    if isinstance(remote_torch, _FP16TorchProxy):
        return
    setattr(remote_module, "torch", _FP16TorchProxy(remote_torch))


def load_unlimited_ocr(
    *,
    repo_id: str = UNLIMITED_OCR_REPO_ID,
    revision: str | None = None,
    dtype: torch.dtype = torch.float16,
    device: str | torch.device | None = None,
    local_files_only: bool = False,
) -> nn.Module:
    """Load the remote Unlimited-OCR model for a Half or ConvRot comparison."""

    try:
        from transformers import AutoModel
    except ImportError as exc:
        raise RuntimeError("transformers is required to load Unlimited-OCR") from exc

    kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "use_safetensors": True,
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True,
        "local_files_only": local_files_only,
    }
    if revision is not None:
        kwargs["revision"] = revision
    model = AutoModel.from_pretrained(repo_id, **kwargs)
    if not isinstance(model, nn.Module):
        raise TypeError("Unlimited-OCR AutoModel loader did not return torch.nn.Module")
    if dtype == torch.float16:
        force_unlimited_ocr_fp16_runtime(model)
    if device is not None:
        model = model.to(device)
    return model.eval()


def quantize_unlimited_ocr_convrot_int8(
    model: nn.Module,
    *,
    calibration_call: CalibrationCall | None = None,
    policy: Mapping[str, Any] | None = None,
    activation_scales: Mapping[str, torch.Tensor | float] | None = None,
    activation_scale_mode: str = "static",
    rot_size: int = 256,
    engine: str = "auto",
    fallback_engine: str = "torch_int_mm",
    mse_clip: bool = True,
    mse_clip_grid: int = 80,
    min_int8_rows: int = 32,
    only_static_int8_eligible_modules: bool = False,
    eps: float = 1e-6,
    inplace: bool = True,
) -> UnlimitedOcrConvRotInt8Result:
    """Apply ConvRot W8A8 to Unlimited-OCR after optional real-call calibration."""

    if activation_scale_mode not in {"static", "dynamic"}:
        raise ValueError("activation_scale_mode must be 'static' or 'dynamic'")
    effective_policy = _merged_policy(policy)
    effective_policy.update(
        {
            "rot_size": int(rot_size),
            "engine": str(engine),
            "fallback_engine": str(fallback_engine),
            "activation_scale_mode": str(activation_scale_mode),
            "mse_clip": bool(mse_clip),
            "mse_clip_grid": int(mse_clip_grid),
            "min_int8_rows": int(min_int8_rows),
            "eps": float(eps),
        }
    )
    calibration: UnlimitedOcrConvRotCalibration | None = None
    scales = dict(activation_scales or {})
    if activation_scale_mode == "static" and not scales:
        if calibration_call is None:
            raise ValueError(
                "static ConvRot requires activation_scales or a real calibration_call"
            )
        calibration = calibrate_unlimited_ocr_convrot_activation_scales(
            model,
            calibration_call=calibration_call,
            policy=effective_policy,
            rot_size=int(rot_size),
            eps=float(eps),
        )
        scales.update(calibration.activation_scales)
    if only_static_int8_eligible_modules:
        if calibration is None:
            raise ValueError(
                "only_static_int8_eligible_modules requires calibration_call, not "
                "precomputed activation_scales"
            )
        eligible_modules = calibration.int8_eligible_module_names(min_int8_rows)
        if not eligible_modules:
            raise ValueError(
                "calibration found no module reaching min_int8_rows for ConvRot"
            )
        effective_policy["include_module_names"] = list(eligible_modules)
        effective_policy["include_name_patterns"] = []
        effective_policy["selection_mode"] = "include_only"
        scales = {name: scales[name] for name in eligible_modules if name in scales}

    quantization = quantize_with_convrot_int8(
        model,
        policy=effective_policy,
        strategy=UNLIMITED_OCR_CONVROT_INT8_STRATEGY,
        inplace=inplace,
        engine=str(engine),
        fallback_engine=str(fallback_engine),
        activation_scale_mode=str(activation_scale_mode),
        activation_scales=scales,
        eps=float(eps),
        mse_clip=bool(mse_clip),
        mse_clip_grid=int(mse_clip_grid),
        min_int8_rows=int(min_int8_rows),
    )
    return UnlimitedOcrConvRotInt8Result(
        model=quantization.model,
        quantization=quantization,
        calibration=calibration,
    )


__all__ = [
    "UNLIMITED_OCR_CONVROT_INT8_STRATEGY",
    "UNLIMITED_OCR_REPO_ID",
    "UnlimitedOcrConvRotCalibration",
    "UnlimitedOcrConvRotInt8Result",
    "calibrate_unlimited_ocr_convrot_activation_scales",
    "force_unlimited_ocr_fp16_runtime",
    "load_unlimited_ocr",
    "quantize_unlimited_ocr_convrot_int8",
    "select_unlimited_ocr_convrot_modules",
    "unlimited_ocr_convrot_default_policy",
]
