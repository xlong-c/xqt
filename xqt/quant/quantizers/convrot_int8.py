"""ConvRot-inspired group-wise regular Hadamard INT8 W8A8 quantization.

Quantization produces rotated INT8 weights (per-output-channel scale) and an
online activation rotation. Compute reuses ``Int8MmaLinear`` (torch._int_mm /
tilelang / reference). This is the Comfy-ecosystem production path counterpart
to the PSEUDO W4A4 path in ``convrot_4bit.py``.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import re
from typing import Any, Iterable, Mapping, Optional

import torch
from torch import nn

from xqt.contracts import ComputeConfig, QuantizedModel
from xqt.core.types import XQTContext
from xqt.quant.comfy_quant import (
    DEFAULT_CONVROT_GROUP_SIZE,
    encode_int8_tensorwise_marker,
)

from ..execution.component import (
    ordered_unique,
    prefix_module_names,
    replace_component_model,
    resolve_component_model,
)
from ..execution.reporting import optional_calibration_summary
from ..execution.selection import (
    build_effective_selection_policy,
    module_selection_reason_metadata,
    selection_policy_metadata,
)
from ..policy import QuantizationPolicy, should_quantize_module
from ..strategy import normalize_quant_strategy
from ..types import QuantizationComponentPlan, QuantizationNature, QuantizationReport
from .convrot_4bit import (
    _apply_groupwise_rotation,
    _collect_convrot_activation_stats,
    _normalize_rot_size,
    _normalized_regular_hadamard,
    build_regular_hadamard_matrix,
)
from .int8_mma import Int8MmaLinear


@dataclass
class ConvRotInt8QuantizationResult(QuantizedModel):
    """Result returned by the ConvRot W8A8 helper."""

    backend: str = "pytorch"
    method: str | None = "convrot"
    strategy: str = "w8a8_int8"
    compute: str = "w8a8_int8_mma"


def _target_arch_from_device(device: torch.device) -> str | None:
    if device.type != "cuda":
        return None
    major, minor = torch.cuda.get_device_capability(device)
    return f"sm_{major}{minor}"


class ConvRotInt8Linear(nn.Module):
    """Rotated-weight INT8 Linear with online activation rotation + W8A8 GEMM."""

    def __init__(
        self,
        qweight_t: torch.Tensor,
        weight_scale: torch.Tensor,
        *,
        bias: torch.Tensor | None,
        input_features: int,
        output_features: int,
        rot_size: int,
        rotation_matrix: torch.Tensor,
        engine: str = "auto",
        fallback_engine: str = "torch_int_mm",
        activation_scale_mode: str = "dynamic",
        activation_scale: torch.Tensor | float | None = None,
        eps: float = 1e-6,
        preferred_engines: list[str] | tuple[str, ...] | None = None,
        min_int8_rows: int = 0,
        output_dtype: torch.dtype = torch.float32,
        comfy_quant_marker: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.input_features = int(input_features)
        self.output_features = int(output_features)
        self.rot_size = int(rot_size)
        self.register_buffer(
            "rotation_matrix",
            rotation_matrix.to(torch.float32).contiguous(),
        )
        runtime_rotation_dtype = (
            output_dtype
            if output_dtype in {torch.float16, torch.bfloat16}
            else torch.float32
        )
        self.register_buffer(
            "runtime_rotation_matrix",
            rotation_matrix.to(runtime_rotation_dtype).contiguous(),
        )
        if comfy_quant_marker is None:
            marker = encode_int8_tensorwise_marker(
                convrot=True,
                convrot_groupsize=self.rot_size,
            )
        else:
            marker = comfy_quant_marker.to(torch.uint8).contiguous()
        self.register_buffer("comfy_quant", marker)
        self.int8_compute = Int8MmaLinear(
            qweight_t,
            weight_scale,
            bias=bias,
            input_features=self.input_features,
            output_features=self.output_features,
            engine=engine,
            fallback_engine=fallback_engine,
            activation_scale_mode=activation_scale_mode,
            activation_scale=activation_scale,
            eps=eps,
            preferred_engines=preferred_engines,
            min_int8_rows=min_int8_rows,
            output_dtype=output_dtype,
        )
        self._last_fused_static_fallback_reason: str | None = None
        self._unrotated_dense_weight: torch.Tensor | None = None

    def _dense_unrotated_weight(
        self,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Dequantize to a dense weight with the rotation folded out.

        The regular Hadamard is symmetric and self-inverse, so rotating the
        activation and using the rotated weight is equivalent to using the
        un-rotated weight directly. Folding it here keeps the float fallback off
        the per-forward rotation, which otherwise costs more than the fp16 GEMM.
        """

        cached = self._unrotated_dense_weight
        if cached is not None and cached.device == device and cached.dtype == dtype:
            return cached
        compute = self.int8_compute
        rotated = compute.qweight_t.to(torch.float32).t() * compute.weight_scale.to(
            torch.float32
        ).reshape(-1, 1)
        weight = _apply_groupwise_rotation(
            rotated,
            rot_size=self.rot_size,
            rotation_matrix=self.rotation_matrix.to(device=rotated.device),
        ).to(device=device, dtype=dtype)
        self._unrotated_dense_weight = weight
        return weight

    def _forward_float_fallback(self, inputs: torch.Tensor) -> torch.Tensor:
        compute = self.int8_compute
        compute_dtype = (
            inputs.dtype
            if inputs.dtype in {torch.float16, torch.bfloat16, torch.float32}
            else torch.float32
        )
        weight = self._dense_unrotated_weight(compute_dtype, inputs.device)
        bias = (
            None
            if compute.bias is None
            else compute.bias.to(device=inputs.device, dtype=compute_dtype)
        )
        output = nn.functional.linear(inputs.to(compute_dtype), weight, bias)
        compute.last_execution = {
            "engine": "float_fallback",
            "reason": "rows_below_min_int8_rows",
            "true_int8_mma": False,
            "min_int8_rows": compute.min_int8_rows,
            "rotation_folded_into_weight": True,
            "input_rows": int(inputs.reshape(-1, self.input_features).shape[0]),
            "input_features": self.input_features,
            "output_features": self.output_features,
        }
        return output.to(compute.output_dtype)

    @classmethod
    def from_linear(
        cls,
        module: nn.Linear,
        *,
        rot_size: int = DEFAULT_CONVROT_GROUP_SIZE,
        engine: str = "auto",
        fallback_engine: str = "torch_int_mm",
        activation_scale_mode: str = "dynamic",
        activation_scale: torch.Tensor | float | None = None,
        eps: float = 1e-6,
        preferred_engines: list[str] | tuple[str, ...] | None = None,
        min_int8_rows: int = 0,
        mse_clip: bool = False,
        mse_clip_grid: int = 80,
    ) -> "ConvRotInt8Linear":
        weight = module.weight.detach().to(torch.float32)
        input_features = int(module.in_features)
        output_features = int(module.out_features)
        normalized_rot_size = _normalize_rot_size(rot_size, input_features)
        if input_features % normalized_rot_size != 0:
            raise ValueError(
                "ConvRot INT8 linear requires input_features divisible by rot_size"
            )
        rotation = _normalized_regular_hadamard(
            normalized_rot_size,
            device=weight.device,
        )
        # Offline weight rotation: W_rot = group(W) @ H^T (H is symmetric).
        rotated_weight = _apply_groupwise_rotation(
            weight,
            rot_size=normalized_rot_size,
            rotation_matrix=rotation,
        )
        qweight, scale = _quantize_int8_per_row(
            rotated_weight,
            eps=eps,
            mse_clip=mse_clip,
            mse_clip_grid=mse_clip_grid,
        )
        bias = None if module.bias is None else module.bias.detach().to(torch.float32)
        return cls(
            qweight.t().contiguous(),
            scale.reshape(-1),
            bias=bias,
            input_features=input_features,
            output_features=output_features,
            rot_size=normalized_rot_size,
            rotation_matrix=rotation,
            engine=engine,
            fallback_engine=fallback_engine,
            activation_scale_mode=activation_scale_mode,
            activation_scale=activation_scale,
            eps=eps,
            preferred_engines=preferred_engines,
            min_int8_rows=min_int8_rows,
            output_dtype=module.weight.dtype,
        )

    def _rotate_inputs(self, inputs: torch.Tensor) -> torch.Tensor:
        return _apply_groupwise_rotation(
            inputs.to(torch.float32),
            rot_size=self.rot_size,
            rotation_matrix=self.rotation_matrix.to(device=inputs.device),
        ).to(dtype=inputs.dtype)

    def _can_use_tilelang_hadamard_static_quant(
        self, flat_inputs: torch.Tensor
    ) -> bool:
        min_rows = self.int8_compute.min_int8_rows
        if 0 < min_rows and int(flat_inputs.shape[0]) < min_rows:
            # Let Int8MmaLinear apply its float fallback instead of the fused path.
            return False
        if self.int8_compute.engine not in {"auto", "triton", "cuda_sm89"}:
            return False
        if self.int8_compute.activation_scale_mode != "static":
            return False
        if not self.int8_compute._has_static_activation_scale:
            return False
        if not flat_inputs.is_cuda or not self.int8_compute.qweight_t.is_cuda:
            return False
        if flat_inputs.dtype not in {torch.float16, torch.bfloat16, torch.float32}:
            return False
        if self.int8_compute.fallback_engine != "torch_int_mm":
            return False
        block_n = min(128, self.rot_size)
        block_k = min(64, self.rot_size)
        if self.rot_size % block_n != 0 or self.rot_size % block_k != 0:
            return False
        if self.int8_compute.engine == "cuda_sm89":
            allowed, _ = self.int8_compute._can_use_cuda_sm89(flat_inputs)
            return allowed
        return True

    def _forward_tilelang_hadamard_static(
        self,
        flat_inputs: torch.Tensor,
        original_shape: tuple[int, ...],
    ) -> torch.Tensor:
        from xqt.operator_opt.kernels.tilelang.int8_mma import (
            groupwise_hadamard_static_quantize_tilelang,
        )

        activation_scale = self.int8_compute._static_activation_scale(
            flat_inputs.device
        )
        block_n = min(128, self.rot_size)
        block_k = min(64, self.rot_size)
        qactivation, padded_quant_rows = groupwise_hadamard_static_quantize_tilelang(
            flat_inputs,
            self.runtime_rotation_matrix.to(device=flat_inputs.device),
            activation_scale,
            rot_size=self.rot_size,
            block_m=32,
            block_n=block_n,
            block_k=block_k,
            threads=128,
            num_stages=2,
            target_arch=_target_arch_from_device(flat_inputs.device),
        )
        prefer_cuda_sm89 = self.int8_compute.engine == "cuda_sm89" or (
            self.int8_compute.engine == "auto" and int(flat_inputs.shape[0]) >= 32
        )
        cuda_allowed, _ = self.int8_compute._can_use_cuda_sm89(qactivation)
        if prefer_cuda_sm89 and cuda_allowed:
            output = self.int8_compute._run_cuda_sm89(qactivation, activation_scale)
            used_engine = "cuda_sm89"
            reason = "true_int8_mma_cuda_sm89_tilelang_hadamard_static_quant"
            fused_static_status = "tilelang_rotation_quant_then_cuda_gemm"
            padded_rows = int(qactivation.shape[0])
            target_arch = _target_arch_from_device(qactivation.device) or "auto"
        else:
            output, padded_rows, target_arch = self.int8_compute._run_triton(
                qactivation,
                activation_scale,
                self.int8_compute.output_dtype,
            )
            used_engine = "triton"
            reason = "true_int8_mma_triton_tilelang_hadamard_static_quant"
            fused_static_status = "tilelang_rotation_quant_then_triton_gemm"
        self.int8_compute.last_execution = {
            "engine": used_engine,
            "reason": reason,
            "true_int8_mma": True,
            "activation_dtype": "int8",
            "weight_dtype": "int8",
            "accumulation_dtype": "int32",
            "activation_scale_mode": self.int8_compute.activation_scale_mode,
            "activation_quant_engine": "tilelang_hadamard_static",
            "fused_static_status": fused_static_status,
            "rotation_fused": True,
            "input_rows": int(flat_inputs.shape[0]),
            "padded_rows": padded_rows,
            "padded_quant_rows": int(padded_quant_rows),
            "input_features": self.input_features,
            "output_features": self.output_features,
            "target_arch": target_arch,
            "prepacked_b": (
                self.int8_compute._qweight_prepacked_b is not None
                if used_engine == "cuda_sm89"
                else False
            ),
        }
        return output.to(self.int8_compute.output_dtype).reshape(
            *original_shape,
            self.output_features,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.shape[-1] != self.input_features:
            raise ValueError(
                "ConvRotInt8Linear input trailing dimension does not match input_features"
            )
        original_shape = tuple(int(dim) for dim in inputs.shape[:-1])
        flat_inputs = inputs.reshape(-1, self.input_features)
        min_rows = self.int8_compute.min_int8_rows
        if 0 < min_rows and int(flat_inputs.shape[0]) < min_rows:
            return self._forward_float_fallback(inputs)
        if self._can_use_tilelang_hadamard_static_quant(flat_inputs):
            try:
                self._last_fused_static_fallback_reason = None
                return self._forward_tilelang_hadamard_static(
                    flat_inputs,
                    original_shape,
                )
            except Exception as exc:
                self._last_fused_static_fallback_reason = str(exc)
        rotated = self._rotate_inputs(inputs)
        return self.int8_compute(rotated)

    def execution_metadata(self) -> dict[str, Any]:
        base = self.int8_compute.execution_metadata()
        return {
            **base,
            "rotation_kind": "regular_hadamard",
            "rotation_scope": "groupwise",
            "rot_size": self.rot_size,
            "method": "convrot",
            "strategy": "w8a8_int8",
            "fused_rotation_quant_fallback_reason": self._last_fused_static_fallback_reason,
        }


def _quantize_int8_per_row(
    weight: torch.Tensor,
    *,
    eps: float,
    mse_clip: bool,
    mse_clip_grid: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-output-channel INT8 quant of a dense weight matrix."""

    absmax = weight.abs().amax(dim=1, keepdim=True).clamp_min(float(eps))
    if not mse_clip:
        scale = (absmax / 127.0).clamp_min(float(eps))
        qweight = torch.round(weight / scale).clamp(-127, 127).to(torch.int8)
        return qweight, scale.reshape(-1)

    grid = max(2, int(mse_clip_grid))
    alphas = torch.linspace(0.55, 1.0, grid, device=weight.device, dtype=torch.float32)
    best_mse = torch.full_like(absmax, float("inf"))
    best_scale = absmax / 127.0
    best_q = torch.round(weight / best_scale).clamp(-127, 127)
    for alpha in alphas.tolist():
        scale = (absmax * float(alpha) / 127.0).clamp_min(float(eps))
        qweight = torch.round(weight / scale).clamp(-127, 127)
        mse = ((qweight * scale - weight) ** 2).mean(dim=1, keepdim=True)
        better = mse < best_mse
        best_mse = torch.where(better, mse, best_mse)
        best_scale = torch.where(better, scale, best_scale)
        best_q = torch.where(better.expand_as(qweight), qweight, best_q)
    return best_q.to(torch.int8), best_scale.reshape(-1)


def _policy_from_mapping(policy: Mapping[str, Any]) -> QuantizationPolicy:
    kwargs: dict[str, Any] = {}
    for key, value in policy.items():
        if key == "dtype":
            kwargs["dtype"] = str(value)
        elif key == "scheme":
            kwargs["scheme"] = str(value)
        elif key in {
            "include_module_types",
            "exclude_module_types",
            "include_name_patterns",
            "exclude_name_patterns",
            "include_module_names",
            "exclude_module_names",
        }:
            kwargs[key] = tuple(str(item) for item in value)
        elif key == "min_parameters":
            kwargs[key] = int(value)
    return QuantizationPolicy(**kwargs)


def _replace_submodule(root: nn.Module, path: str, replacement: nn.Module) -> None:
    parent_path, _, attribute = path.rpartition(".")
    parent = root.get_submodule(parent_path) if parent_path else root
    if attribute.isdigit() and isinstance(parent, (nn.Sequential, nn.ModuleList)):
        parent[int(attribute)] = replacement
        return
    setattr(parent, attribute, replacement)


def _select_convrot_candidate_names(
    model: nn.Module,
    *,
    policy: QuantizationPolicy,
    policy_mapping: Mapping[str, Any],
) -> list[str]:
    """Select Linear modules, with an explicit include-only mode for hybrids."""

    selection_mode = (
        str(policy_mapping.get("selection_mode", "default")).strip().lower()
    )
    if selection_mode not in {"default", "include_only"}:
        raise ValueError("selection_mode must be 'default' or 'include_only'")
    include_names = set(policy.include_module_names)
    include_patterns = tuple(policy.include_name_patterns)
    candidates: list[str] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if selection_mode == "include_only":
            included = name in include_names or any(
                re.search(pattern, name) for pattern in include_patterns
            )
            if not included:
                continue
        if should_quantize_module(name, module, policy):
            candidates.append(name)
    return candidates


def quantize_with_convrot_int8(
    model: nn.Module,
    *,
    policy: Optional[Mapping[str, Any] | QuantizationPolicy] = None,
    strategy: Optional[str] = None,
    calibration_inputs: Iterable[Any] | None = None,
    inplace: bool = True,
    engine: str = "auto",
    fallback_engine: str = "torch_int_mm",
    activation_scale_mode: str = "dynamic",
    activation_scales: Optional[Mapping[str, torch.Tensor | float]] = None,
    eps: float = 1e-6,
    mse_clip: bool = False,
    mse_clip_grid: int = 80,
    min_int8_rows: int = 0,
) -> ConvRotInt8QuantizationResult:
    """Quantize Linear modules with ConvRot rotation + W8A8 INT8 storage/compute."""

    quant_policy = (
        policy
        if isinstance(policy, QuantizationPolicy)
        else _policy_from_mapping(policy or {})
    )
    policy_mapping = dict(policy) if isinstance(policy, Mapping) else {}
    configured_rot_size = int(
        policy_mapping.get(
            "rot_size",
            policy_mapping.get("convrot_groupsize", DEFAULT_CONVROT_GROUP_SIZE),
        )
        or DEFAULT_CONVROT_GROUP_SIZE
    )
    selected_strategy = (
        normalize_quant_strategy(
            strategy,
            {
                "dtype": "int8",
                "scheme": "convrot_w8a8",
            },
        )
        or "w8a8_int8"
    )
    if selected_strategy not in {"w8a8_int8", "convrot_w8a8"}:
        # Keep alias friendly; canonical storage/compute remains w8a8_int8.
        if selected_strategy.startswith("w8a8"):
            pass
        else:
            selected_strategy = "w8a8_int8"
    else:
        selected_strategy = "w8a8_int8"

    engine_name = str(policy_mapping.get("engine", engine) or engine)
    fallback = str(
        policy_mapping.get("fallback_engine", fallback_engine) or fallback_engine
    )
    act_mode = (
        str(
            policy_mapping.get("activation_scale_mode", activation_scale_mode)
            or activation_scale_mode
        )
        .strip()
        .lower()
    )
    if act_mode not in {"dynamic", "static"}:
        raise ValueError("activation_scale_mode must be dynamic or static")
    use_mse = bool(policy_mapping.get("mse_clip", mse_clip))
    grid = int(policy_mapping.get("mse_clip_grid", mse_clip_grid) or mse_clip_grid)
    min_rows = int(policy_mapping.get("min_int8_rows", min_int8_rows) or 0)
    static_scales = dict(activation_scales or {})
    if isinstance(policy_mapping.get("activation_scales"), Mapping):
        static_scales.update(dict(policy_mapping["activation_scales"]))
    sample_limit = (
        int(policy_mapping["sample_limit"])
        if "sample_limit" in policy_mapping
        else None
    )

    target_model = model if inplace else copy.deepcopy(model)
    candidate_names = _select_convrot_candidate_names(
        target_model,
        policy=quant_policy,
        policy_mapping=policy_mapping,
    )
    calibrated_scales, _ = _collect_convrot_activation_stats(
        target_model,
        module_names=candidate_names,
        calibration_inputs=calibration_inputs,
        sample_limit=sample_limit,
        rot_size=configured_rot_size,
        quant_max=127.0,
    )
    for name, scale in calibrated_scales.items():
        static_scales.setdefault(name, scale)
    if (
        calibrated_scales
        and act_mode == "dynamic"
        and "activation_scale_mode" not in policy_mapping
    ):
        act_mode = "static"

    quantized_modules: list[str] = []
    static_scale_modules = 0
    dynamic_fallback_modules = 0
    for name, module in list(target_model.named_modules()):
        if name not in candidate_names:
            continue
        module_act_mode = act_mode
        module_scale = static_scales.get(name)
        if module_act_mode == "static":
            if module_scale is None:
                module_act_mode = "dynamic"
                dynamic_fallback_modules += 1
            else:
                static_scale_modules += 1
        replacement = ConvRotInt8Linear.from_linear(
            module,
            rot_size=configured_rot_size,
            engine=engine_name,
            fallback_engine=fallback,
            activation_scale_mode=module_act_mode,
            activation_scale=module_scale,
            eps=float(policy_mapping.get("eps", eps) or eps),
            min_int8_rows=min_rows,
            mse_clip=use_mse,
            mse_clip_grid=grid,
        )
        if name:
            _replace_submodule(target_model, name, replacement)
        else:
            target_model = replacement
        quantized_modules.append(name)

    preferred_hint = [] if str(engine_name) == "auto" else [str(engine_name)]
    compute_config = ComputeConfig.from_modules(
        module_names=quantized_modules,
        compute_contract="int8_mma",
        precision="w8a8",
        required_capabilities=["int8_mma"],
        preferred_engines=preferred_hint,
        default_precision="w8a8",
        storage={
            "format": "int8_per_out_channel_rotated",
            "layout": "qweight_t",
            "rotation": "regular_hadamard_groupwise",
            "rot_size": configured_rot_size,
        },
        metadata={
            "method": "convrot",
            "activation_scale_mode": act_mode,
            "fallback_engine": fallback,
            "comfy_quant_format": "int8_tensorwise",
        },
    )
    return ConvRotInt8QuantizationResult(
        model=target_model,
        method="convrot",
        strategy=selected_strategy,
        quantized_modules=quantized_modules,
        compute_config=compute_config,
        metadata={
            "implementation": "convrot_groupwise_regular_hadamard_w8a8",
            "quantization_nature": "true",
            "quantization_nature_scope": "requested_compute_contract_not_runtime_observation",
            "weight_encoding": "signed_int8_per_output_channel_rotated",
            "activation_encoding": f"{act_mode}_signed_int8_per_tensor_rotated",
            "activation_scale_mode": act_mode,
            "static_scale_module_count": static_scale_modules,
            "dynamic_fallback_module_count": dynamic_fallback_modules,
            "calibrated_static_scale_module_count": len(calibrated_scales),
            "rot_size": configured_rot_size,
            "mse_clip": use_mse,
            "comfy_quant": {
                "format": "int8_tensorwise",
                "convrot": True,
                "convrot_groupsize": configured_rot_size,
            },
            "precision_description": {
                "quantization_time": {
                    "weight": (
                        "offline group-wise regular Hadamard rotation then "
                        "static signed INT8 per output channel"
                    ),
                    "activation": (
                        "not stored as an activation artifact; each forward rotates "
                        f"then applies {act_mode} signed INT8 encoding"
                    ),
                },
                "runtime": {
                    "requested_compute": "W8A8 INT8 MMA after online activation rotation",
                    "actual_execution_source": (
                        "ConvRotInt8Linear.execution_metadata -> Int8MmaLinear.runtime_precision"
                    ),
                    "native_mma": "only true when the per-forward metadata reports it",
                },
            },
            "algorithm_metadata": {
                "rotation_kind": "regular_hadamard",
                "rotation_scope": "groupwise",
                "rot_size": configured_rot_size,
                "weight_bits": 8,
                "activation_bits": 8,
                "activation_scale_mode": act_mode,
            },
            "engine_preference": engine_name,
            "preferred_engines": preferred_hint,
            "fallback_engine": fallback,
            "compute_config": compute_config.to_dict(),
            "policy": {
                "dtype": quant_policy.dtype,
                "scheme": quant_policy.scheme,
                "include_module_types": list(quant_policy.include_module_types),
                "exclude_module_types": list(quant_policy.exclude_module_types),
                "include_name_patterns": list(quant_policy.include_name_patterns),
                "exclude_name_patterns": list(quant_policy.exclude_name_patterns),
                "include_module_names": list(quant_policy.include_module_names),
                "exclude_module_names": list(quant_policy.exclude_module_names),
                "min_parameters": quant_policy.min_parameters,
                "rot_size": configured_rot_size,
                "mse_clip": use_mse,
                "selection_mode": str(policy_mapping.get("selection_mode", "default")),
            },
        },
    )


def execute_convrot_int8_component(
    context: XQTContext,
    root_model: nn.Module,
    component: QuantizationComponentPlan,
    *,
    quantize_fn: Any = quantize_with_convrot_int8,
) -> tuple[nn.Module, QuantizationReport]:
    """Execute the ConvRot W8A8 quantizer for one component."""

    target_model = resolve_component_model(root_model, component.target_path)
    effective_policy = build_effective_selection_policy(component)
    result = quantize_fn(
        target_model,
        policy=effective_policy,
        strategy=component.strategy or effective_policy.get("strategy"),
        calibration_inputs=context.calibration_inputs,
        inplace=True,
        engine=str(component.policy.get("engine", "auto")),
        fallback_engine=str(component.policy.get("fallback_engine", "torch_int_mm")),
        activation_scale_mode=str(
            component.policy.get("activation_scale_mode", "dynamic")
        ),
        activation_scales=component.policy.get("activation_scales"),
        eps=float(component.policy.get("eps", 1e-6)),
        mse_clip=bool(component.policy.get("mse_clip", False)),
        mse_clip_grid=int(component.policy.get("mse_clip_grid", 80)),
        min_int8_rows=int(component.policy.get("min_int8_rows", 0)),
    )
    updated_model = replace_component_model(
        root_model, component.target_path, result.model
    )
    high_precision_modules = prefix_module_names(
        component.keep_high_precision,
        component.target_path,
    )
    skipped_modules = ordered_unique(
        [
            *prefix_module_names(component.skip_quantize, component.target_path),
            *high_precision_modules,
        ]
    )
    quantized_modules = prefix_module_names(
        result.quantized_modules, component.target_path
    )
    module_selection_reasons = module_selection_reason_metadata(
        component,
        quantized_modules=quantized_modules,
        skipped_modules=skipped_modules,
        high_precision_modules=high_precision_modules,
    )
    calibration_samples, calibration_summary = optional_calibration_summary(
        context,
        component,
    )
    report = QuantizationReport(
        component_name=component.name,
        backend=component.backend,
        runtime="pytorch",
        method=component.method or "convrot",
        strategy=result.strategy,
        target_path=component.target_path,
        quantized_modules=quantized_modules,
        skipped_modules=skipped_modules,
        high_precision_modules=high_precision_modules,
        calibration_samples=calibration_samples,
        calibration_summary=calibration_summary,
        nature=QuantizationNature.TRUE,
        algorithm_executable=True,
        method_semantics="groupwise_regular_hadamard_rotation_w8a8_int8_mma",
        compute_speedup_expected=None,
        metadata={
            **dict(result.metadata),
            "execution_state": "convrot_int8",
            "analysis_only": component.analysis_only,
            "policy": effective_policy,
            "selection_policy": selection_policy_metadata(component),
            "module_selection_reasons": module_selection_reasons,
            "executed": True,
            "algorithm_executable": True,
        },
    )
    return updated_model, report


__all__ = [
    "ConvRotInt8Linear",
    "ConvRotInt8QuantizationResult",
    "build_regular_hadamard_matrix",
    "execute_convrot_int8_component",
    "quantize_with_convrot_int8",
]
