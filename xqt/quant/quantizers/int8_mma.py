"""True W8A8 INT8 MMA runtime quantization for Linear modules."""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

import torch
from torch import nn

from xqt.core.errors import XQTBackendError
from xqt.core.types import XQTContext
from xqt.operator_opt.kernels.tilelang.int8_mma import (
    int8_linear_tilelang,
    int8_linear_static_activation_tilelang,
    int8_mma_reference,
    pad_rows_to_block,
    static_activation_quantize_tilelang,
)

_PTX_SM89_ENGINES = frozenset({"ptx_sm89", "native_sm89"})
_VALID_ENGINES = frozenset({"auto", "tilelang", "torch_int_mm"}) | _PTX_SM89_ENGINES

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


_ACTIVATION_SCALE_MODES = {"dynamic", "static"}
_OUTPUT_DTYPES = {torch.float16, torch.bfloat16, torch.float32}
_SELECTION_MODES = {"default", "include_only"}


@dataclass
class Int8MmaQuantizationResult:
    """Result returned by the true INT8 MMA quantization helper."""

    model: nn.Module
    backend: str = "pytorch"
    strategy: str = "dynamic_int8_mma"
    quantized_modules: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


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


def _matches_name_patterns(name: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(pattern, name) for pattern in patterns)


def _module_parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters(recurse=False))


def _should_quantize_int8_mma_module(
    name: str,
    module: nn.Module,
    policy: QuantizationPolicy,
    *,
    selection_mode: str,
) -> bool:
    if selection_mode == "default":
        return should_quantize_module(name, module, policy)
    if selection_mode != "include_only":
        raise ValueError("selection_mode must be default or include_only")

    included = name in policy.include_module_names or _matches_name_patterns(
        name,
        policy.include_name_patterns,
    )
    if not included:
        return False
    if name in policy.exclude_module_names:
        return False
    if _matches_name_patterns(name, policy.exclude_name_patterns):
        return False
    if type(module).__name__ in policy.exclude_module_types:
        return False
    if policy.include_module_types and type(module).__name__ not in policy.include_module_types:
        return False
    return _module_parameter_count(module) >= policy.min_parameters


def _replace_submodule(root: nn.Module, path: str, replacement: nn.Module) -> None:
    parent_path, _, attribute = path.rpartition(".")
    parent = root.get_submodule(parent_path) if parent_path else root
    if attribute.isdigit() and isinstance(parent, (nn.Sequential, nn.ModuleList)):
        parent[int(attribute)] = replacement
        return
    setattr(parent, attribute, replacement)


def _target_arch_from_device(device: torch.device) -> str | None:
    if device.type != "cuda":
        return None
    major, minor = torch.cuda.get_device_capability(device)
    return f"sm_{major}{minor}"


class Int8MmaLinear(nn.Module):
    """Linear replacement backed by true W8A8 INT8 MMA."""

    def __init__(
        self,
        qweight_t: torch.Tensor,
        weight_scale: torch.Tensor,
        *,
        bias: torch.Tensor | None,
        input_features: int,
        output_features: int,
        engine: str = "tilelang",
        fallback_engine: str = "torch_int_mm",
        block_m: int = 64,
        block_n: int = 64,
        block_k: int = 64,
        threads: int = 128,
        num_stages: int = 2,
        output_dtype: torch.dtype = torch.float32,
        activation_scale_mode: str = "dynamic",
        activation_scale: torch.Tensor | float | None = None,
        activation_quant_block_size: int = 256,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        normalized_engine = str(engine)
        if normalized_engine == "native_sm89":
            normalized_engine = "ptx_sm89"
        if normalized_engine not in _VALID_ENGINES:
            raise ValueError(
                "engine must be one of auto, tilelang, torch_int_mm, ptx_sm89"
            )
        if str(fallback_engine) not in {"torch_int_mm", "reference"}:
            raise ValueError("fallback_engine must be torch_int_mm or reference")
        self.input_features = int(input_features)
        self.output_features = int(output_features)
        self.engine = normalized_engine
        self.fallback_engine = str(fallback_engine)
        self.block_m = int(block_m)
        self.block_n = int(block_n)
        self.block_k = int(block_k)
        self.threads = int(threads)
        self.num_stages = int(num_stages)
        if output_dtype not in _OUTPUT_DTYPES:
            raise ValueError("output_dtype must be float16, bfloat16, or float32")
        self.output_dtype = output_dtype
        if str(activation_scale_mode) not in _ACTIVATION_SCALE_MODES:
            raise ValueError("activation_scale_mode must be dynamic or static")
        self.activation_scale_mode = str(activation_scale_mode)
        self.activation_quant_block_size = int(activation_quant_block_size)
        self.eps = float(eps)
        self.last_execution: dict[str, Any] = {"engine": "not_run"}
        self.register_buffer("qweight_t", qweight_t.to(torch.int8).contiguous())
        self.register_buffer("weight_scale", weight_scale.to(torch.float32).contiguous())
        self._qweight_prepacked_b: torch.Tensor | None = None
        if normalized_engine == "ptx_sm89":
            self._ensure_ptx_prepacked_b()
        initial_activation_scale = torch.tensor(0.0, dtype=torch.float32)
        self._has_static_activation_scale = activation_scale is not None
        if activation_scale is not None:
            initial_activation_scale = torch.as_tensor(activation_scale, dtype=torch.float32).reshape(())
        self.register_buffer("static_activation_scale", initial_activation_scale)
        if bias is None:
            self.register_buffer("bias", None)
        else:
            self.register_buffer("bias", bias.detach().to(torch.float32).contiguous())

    @classmethod
    def from_linear(
        cls,
        module: nn.Linear,
        *,
        engine: str = "tilelang",
        fallback_engine: str = "torch_int_mm",
        block_m: int = 64,
        block_n: int = 64,
        block_k: int = 64,
        threads: int = 128,
        num_stages: int = 2,
        activation_scale_mode: str = "dynamic",
        activation_scale: torch.Tensor | float | None = None,
        activation_quant_block_size: int = 256,
        eps: float = 1e-6,
    ) -> "Int8MmaLinear":
        weight = module.weight.detach().to(torch.float32)
        max_abs = weight.abs().amax(dim=1, keepdim=True)
        scale = torch.where(
            max_abs > float(eps),
            max_abs / 127.0,
            torch.ones_like(max_abs),
        )
        qweight = torch.round(weight / scale).clamp(-127, 127).to(torch.int8)
        bias = None if module.bias is None else module.bias.detach().to(torch.float32)
        return cls(
            qweight.t().contiguous(),
            scale.reshape(-1),
            bias=bias,
            input_features=module.in_features,
            output_features=module.out_features,
            engine=engine,
            fallback_engine=fallback_engine,
            block_m=block_m,
            block_n=block_n,
            block_k=block_k,
            threads=threads,
            num_stages=num_stages,
            output_dtype=module.weight.dtype,
            activation_scale_mode=activation_scale_mode,
            activation_scale=activation_scale,
            activation_quant_block_size=activation_quant_block_size,
            eps=eps,
        )

    @staticmethod
    def activation_scale_from_inputs(inputs: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """Return the symmetric per-tensor INT8 activation scale for sample inputs."""

        max_abs = inputs.detach().reshape(-1).to(torch.float32).abs().amax().clamp_min(float(eps))
        return (max_abs / 127.0).reshape(())

    def set_static_activation_scale(self, scale: torch.Tensor | float) -> None:
        """Set a calibrated static activation scale used by the static fast path."""

        value = torch.as_tensor(scale, dtype=torch.float32, device=self.static_activation_scale.device).reshape(())
        if float(value.detach().cpu().item()) <= 0.0:
            raise ValueError("static activation scale must be positive")
        self.static_activation_scale.copy_(value)
        self._has_static_activation_scale = True
        self.activation_scale_mode = "static"

    def calibrate_static_activation_scale(self, sample_inputs: torch.Tensor) -> torch.Tensor:
        """Calibrate this layer's static activation scale from representative inputs."""

        scale = self.activation_scale_from_inputs(sample_inputs, eps=self.eps).to(
            device=self.static_activation_scale.device,
            dtype=torch.float32,
        )
        self.set_static_activation_scale(scale)
        return self.static_activation_scale.detach().clone()

    def _static_activation_scale(self, device: torch.device) -> torch.Tensor:
        if not self._has_static_activation_scale:
            raise XQTBackendError(
                "static activation scale mode requires calibrate_static_activation_scale() "
                "or set_static_activation_scale() before forward"
            )
        return self.static_activation_scale.to(device=device, dtype=torch.float32)

    def _quantize_activation(
        self,
        inputs: torch.Tensor,
        *,
        prefer_tilelang: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, str]:
        flat_input = inputs.reshape(-1, self.input_features)
        if self.activation_scale_mode == "static":
            scale = self._static_activation_scale(flat_input.device)
            if prefer_tilelang and flat_input.is_cuda and inputs.dtype in {
                torch.float16,
                torch.bfloat16,
                torch.float32,
            }:
                qactivation = static_activation_quantize_tilelang(
                    flat_input,
                    scale,
                    block_size=self.activation_quant_block_size,
                    target_arch=_target_arch_from_device(flat_input.device),
                )
                return qactivation.contiguous(), scale, "tilelang_static"
        else:
            flat = flat_input.to(torch.float32)
            scale = self.activation_scale_from_inputs(flat, eps=self.eps).to(device=flat.device)
            qactivation = torch.round(flat / scale).clamp(-127, 127).to(torch.int8)
            return qactivation.contiguous(), scale, f"torch_{self.activation_scale_mode}"
        qactivation = torch.round(flat_input.to(torch.float32) / scale).clamp(-127, 127).to(torch.int8)
        return qactivation.contiguous(), scale, f"torch_{self.activation_scale_mode}"

    def _can_use_tilelang(self, qactivation: torch.Tensor) -> tuple[bool, str]:
        if not qactivation.is_cuda or not self.qweight_t.is_cuda:
            return False, "TileLang INT8 MMA requires CUDA tensors"
        if self.output_features % self.block_n != 0:
            return False, "output_features is not block_n aligned"
        if self.input_features % self.block_k != 0:
            return False, "input_features is not block_k aligned"
        return True, "ok"

    def _run_torch_int_mm(self, qactivation: torch.Tensor) -> torch.Tensor:
        int_mm = getattr(torch, "_int_mm", None)
        if qactivation.is_cuda and self.qweight_t.is_cuda and callable(int_mm):
            return int_mm(qactivation, self.qweight_t)
        return int8_mma_reference(qactivation, self.qweight_t)

    def _can_use_ptx_sm89(self, qactivation: torch.Tensor) -> tuple[bool, str]:
        if not qactivation.is_cuda or not self.qweight_t.is_cuda:
            return False, "ptx_sm89 INT8 MMA requires CUDA tensors"
        major, minor = torch.cuda.get_device_capability(qactivation.device)
        if (major, minor) != (8, 9):
            return False, f"ptx_sm89 requires sm_89, got sm_{major}{minor}"
        try:
            from xqt.operator_opt.kernels.cute.int8mma_binding import int8mma_available
        except Exception as exc:
            return False, f"ptx_sm89 import failed: {exc}"
        if not int8mma_available():
            return False, "ptx_sm89 shared library not built"
        return True, "ok"

    def _ensure_ptx_prepacked_b(self) -> torch.Tensor | None:
        if self.engine not in {"ptx_sm89", "auto"}:
            return None
        if self._qweight_prepacked_b is not None:
            if self._qweight_prepacked_b.device != self.qweight_t.device:
                self._qweight_prepacked_b = self._qweight_prepacked_b.to(
                    device=self.qweight_t.device
                )
            return self._qweight_prepacked_b
        try:
            from xqt.operator_opt.kernels.cute.int8mma_binding import (
                prepack_qweight_t_for_ptx_sm89,
            )
        except Exception:
            return None
        packed = prepack_qweight_t_for_ptx_sm89(self.qweight_t)
        self._qweight_prepacked_b = packed.to(device=self.qweight_t.device)
        return self._qweight_prepacked_b

    def _run_ptx_sm89(
        self,
        qactivation: torch.Tensor,
        activation_scale: torch.Tensor,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        from xqt.operator_opt.kernels.cute.int8mma_binding import int8_linear_ptx_sm89

        prepacked = self._ensure_ptx_prepacked_b()
        return int8_linear_ptx_sm89(
            qactivation,
            self.qweight_t,
            activation_scale,
            self.weight_scale,
            self.bias,
            output_dtype=output_dtype,
            prepacked_b=prepacked,
        )

    def _can_use_ptx_sm89_static_fused(self, flat_inputs: torch.Tensor) -> tuple[bool, str]:
        if flat_inputs.dtype != torch.float16:
            return False, "ptx_sm89 fused static path requires float16 activations"
        return self._can_use_ptx_sm89(flat_inputs)

    def _run_ptx_sm89_static_fused(
        self,
        flat_inputs: torch.Tensor,
        activation_scale: torch.Tensor,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        from xqt.operator_opt.kernels.cute.int8mma_binding import (
            int8_linear_fused_static_ptx_sm89,
        )

        prepacked = self._ensure_ptx_prepacked_b()
        return int8_linear_fused_static_ptx_sm89(
            flat_inputs,
            self.qweight_t,
            activation_scale,
            self.weight_scale,
            self.bias,
            output_dtype=output_dtype,
            prepacked_b=prepacked,
        )

    def _run_tilelang(
        self,
        qactivation: torch.Tensor,
        activation_scale: torch.Tensor,
        output_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, int, str]:
        padded, original_rows = pad_rows_to_block(qactivation, self.block_m)
        target_arch = _target_arch_from_device(padded.device)
        output = int8_linear_tilelang(
            padded,
            self.qweight_t,
            activation_scale,
            self.weight_scale,
            self.bias,
            output_dtype=output_dtype,
            block_m=self.block_m,
            block_n=self.block_n,
            block_k=self.block_k,
            threads=self.threads,
            num_stages=self.num_stages,
            target_arch=target_arch,
        )
        return output[:original_rows], int(padded.shape[0]), target_arch or "auto"

    def _can_use_tilelang_static_fused(self, flat_inputs: torch.Tensor) -> tuple[bool, str]:
        if not flat_inputs.is_cuda or not self.qweight_t.is_cuda:
            return False, "TileLang fused INT8 Linear requires CUDA tensors"
        if flat_inputs.dtype not in {torch.float16, torch.bfloat16, torch.float32}:
            return False, "TileLang fused INT8 Linear requires fp16, bf16, or fp32 activations"
        if self.output_features % self.block_n != 0:
            return False, "output_features is not block_n aligned"
        if self.input_features % self.block_k != 0:
            return False, "input_features is not block_k aligned"
        return True, "ok"

    def _run_tilelang_static_fused(
        self,
        flat_inputs: torch.Tensor,
        activation_scale: torch.Tensor,
        output_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, int, str]:
        padded, original_rows = pad_rows_to_block(flat_inputs, self.block_m)
        target_arch = _target_arch_from_device(padded.device)
        output = int8_linear_static_activation_tilelang(
            padded,
            self.qweight_t,
            activation_scale,
            self.weight_scale,
            self.bias,
            output_dtype=output_dtype,
            block_m=self.block_m,
            block_n=self.block_n,
            block_k=self.block_k,
            threads=self.threads,
            num_stages=self.num_stages,
            target_arch=target_arch,
        )
        return output[:original_rows], int(padded.shape[0]), target_arch or "auto"

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.shape[-1] != self.input_features:
            raise XQTBackendError(
                "Int8MmaLinear input trailing dimension does not match input_features"
            )
        original_shape = tuple(int(dim) for dim in inputs.shape[:-1])
        selected_engine = self.engine
        prefer_tilelang = selected_engine == "tilelang" or (
            selected_engine == "auto" and inputs.is_cuda and self.qweight_t.is_cuda
        )
        flat_inputs = inputs.reshape(-1, self.input_features)
        if selected_engine == "auto" and inputs.is_cuda and self.qweight_t.is_cuda:
            m_rows = int(flat_inputs.shape[0])
            ptx_ok, _ = self._can_use_ptx_sm89(flat_inputs)
            if ptx_ok and m_rows >= 192:
                selected_engine = "ptx_sm89"
            else:
                selected_engine = "tilelang"
        prefer_ptx = selected_engine == "ptx_sm89"
        prefer_tilelang = selected_engine == "tilelang" or (
            selected_engine == "auto" and inputs.is_cuda and self.qweight_t.is_cuda
        )
        if (
            prefer_ptx
            and self.activation_scale_mode == "static"
            and self.fallback_engine == "torch_int_mm"
        ):
            activation_scale = self._static_activation_scale(flat_inputs.device)
            used_engine = "ptx_sm89"
            reason = "true_int8_mma_ptx_sm89_fused_static"
            fused_static_status = "not_attempted"
            padded_rows = int(flat_inputs.shape[0])
            target_arch = _target_arch_from_device(flat_inputs.device)
            allowed, reason = self._can_use_ptx_sm89_static_fused(flat_inputs)
            if allowed:
                try:
                    output = self._run_ptx_sm89_static_fused(
                        flat_inputs,
                        activation_scale,
                        self.output_dtype,
                    )
                    self.last_execution = {
                        "engine": used_engine,
                        "reason": "true_int8_mma_ptx_sm89_fused_static_prepacked_b",
                        "true_int8_mma": True,
                        "activation_dtype": "int8",
                        "weight_dtype": "int8",
                        "accumulation_dtype": "int32",
                        "activation_scale_mode": self.activation_scale_mode,
                        "activation_quant_engine": "ptx_sm89_fused_static",
                        "fused_static_status": "used",
                        "input_rows": int(flat_inputs.shape[0]),
                        "padded_rows": padded_rows,
                        "input_features": self.input_features,
                        "output_features": self.output_features,
                        "target_arch": target_arch,
                        "prepacked_b": self._qweight_prepacked_b is not None,
                    }
                    output = output.to(self.output_dtype)
                    return output.reshape(*original_shape, self.output_features)
                except Exception as exc:
                    reason = f"ptx_sm89_fused_static_fallback: {exc}"
                    fused_static_status = reason
            else:
                reason = f"ptx_sm89_fused_static_unavailable: {reason}"
                fused_static_status = reason
        if (
            prefer_tilelang
            and not prefer_ptx
            and self.activation_scale_mode == "static"
            and self.fallback_engine == "torch_int_mm"
        ):
            activation_scale = self._static_activation_scale(flat_inputs.device)
            used_engine = "tilelang"
            reason = "true_int8_mma_fused_static_activation"
            fused_static_status = "not_attempted"
            padded_rows = int(flat_inputs.shape[0])
            target_arch = _target_arch_from_device(flat_inputs.device)
            allowed, reason = self._can_use_tilelang_static_fused(flat_inputs)
            if allowed:
                try:
                    output, padded_rows, target_arch = self._run_tilelang_static_fused(
                        flat_inputs,
                        activation_scale,
                        self.output_dtype,
                    )
                    self.last_execution = {
                        "engine": used_engine,
                        "reason": reason,
                        "true_int8_mma": True,
                        "activation_dtype": "int8",
                        "weight_dtype": "int8",
                        "accumulation_dtype": "int32",
                        "activation_scale_mode": self.activation_scale_mode,
                        "activation_quant_engine": "tilelang_fused_static",
                        "fused_static_status": "used",
                        "input_rows": int(flat_inputs.shape[0]),
                        "padded_rows": padded_rows,
                        "input_features": self.input_features,
                        "output_features": self.output_features,
                        "target_arch": target_arch,
                    }
                    output = output.to(self.output_dtype)
                    return output.reshape(*original_shape, self.output_features)
                except Exception as exc:
                    reason = f"tilelang_fused_static_fallback: {exc}"
                    fused_static_status = reason
            else:
                reason = f"tilelang_fused_static_unavailable: {reason}"
                fused_static_status = reason
        qactivation, activation_scale, activation_quant_engine = self._quantize_activation(
            inputs,
            prefer_tilelang=prefer_tilelang and not prefer_ptx,
        )
        if selected_engine == "auto":
            if qactivation.is_cuda:
                ptx_ok, _ = self._can_use_ptx_sm89(qactivation)
                selected_engine = "ptx_sm89" if ptx_ok else "tilelang"
            else:
                selected_engine = "torch_int_mm"

        used_engine = selected_engine
        reason = reason if "reason" in locals() else "true_int8_mma"
        padded_rows = int(qactivation.shape[0])
        target_arch = _target_arch_from_device(qactivation.device)
        try:
            if selected_engine == "ptx_sm89":
                allowed, reason = self._can_use_ptx_sm89(qactivation)
                if not allowed:
                    raise XQTBackendError(reason)
                output = self._run_ptx_sm89(
                    qactivation,
                    activation_scale,
                    self.output_dtype,
                )
                reason = (
                    "true_int8_mma_ptx_sm89_prepacked_b"
                    if self._qweight_prepacked_b is not None
                    else "true_int8_mma_ptx_sm89"
                )
            elif selected_engine == "tilelang":
                allowed, reason = self._can_use_tilelang(qactivation)
                if not allowed:
                    raise XQTBackendError(reason)
                output, padded_rows, target_arch = self._run_tilelang(
                    qactivation,
                    activation_scale,
                    self.output_dtype,
                )
            else:
                acc = self._run_torch_int_mm(qactivation)
                output = acc.to(torch.float32) * (
                    activation_scale.to(torch.float32) * self.weight_scale.to(
                        device=acc.device,
                        dtype=torch.float32,
                    )
                )
        except Exception as exc:
            if self.fallback_engine != "torch_int_mm":
                raise
            used_engine = "torch_int_mm"
            reason = f"int8_mma_fallback: {exc}"
            acc = self._run_torch_int_mm(qactivation)
            output = acc.to(torch.float32) * (
                activation_scale.to(torch.float32) * self.weight_scale.to(
                    device=acc.device,
                    dtype=torch.float32,
                )
            )
        if self.bias is not None and used_engine not in {"tilelang", "ptx_sm89"}:
            output = output + self.bias.to(device=output.device, dtype=output.dtype)
        self.last_execution = {
            "engine": used_engine,
            "reason": reason,
            "true_int8_mma": bool(qactivation.is_cuda and self.qweight_t.is_cuda),
            "activation_dtype": "int8",
            "weight_dtype": "int8",
            "accumulation_dtype": "int32",
            "activation_scale_mode": self.activation_scale_mode,
            "activation_quant_engine": activation_quant_engine,
            "fused_static_status": fused_static_status if "fused_static_status" in locals() else "not_attempted",
            "input_rows": int(qactivation.shape[0]),
            "padded_rows": padded_rows,
            "input_features": self.input_features,
            "output_features": self.output_features,
            "target_arch": target_arch,
        }
        output = output.to(self.output_dtype)
        return output.reshape(*original_shape, self.output_features)

    def execution_metadata(self) -> dict[str, Any]:
        return dict(self.last_execution)


def quantize_with_int8_mma(
    model: nn.Module,
    *,
    policy: Optional[Mapping[str, Any] | QuantizationPolicy] = None,
    strategy: Optional[str] = None,
    inplace: bool = True,
    engine: str = "tilelang",
    fallback_engine: str = "torch_int_mm",
    block_m: int = 64,
    block_n: int = 64,
    block_k: int = 64,
    threads: int = 128,
    num_stages: int = 2,
    activation_scale_mode: str = "dynamic",
    activation_scales: Optional[Mapping[str, torch.Tensor | float]] = None,
    activation_quant_block_size: int = 256,
    eps: float = 1e-6,
) -> Int8MmaQuantizationResult:
    """Replace Linear modules with true W8A8 INT8 MMA runtime modules."""

    quant_policy = (
        policy
        if isinstance(policy, QuantizationPolicy)
        else _policy_from_mapping(policy or {})
    )
    selection_mode = "default"
    if isinstance(policy, Mapping):
        selection_mode = str(policy.get("selection_mode", "default"))
    if selection_mode not in _SELECTION_MODES:
        raise ValueError("selection_mode must be default or include_only")
    selected_strategy = (
        normalize_quant_strategy(
            strategy,
            {
                "dtype": "int8",
                "scheme": "dynamic_mma",
                "engine": engine,
            },
        )
        or "dynamic_int8_mma"
    )
    target_model = model if inplace else copy.deepcopy(model)
    quantized_modules: list[str] = []
    static_scales = dict(activation_scales or {})
    static_scale_modules = 0
    dynamic_fallback_modules = 0

    for name, module in list(target_model.named_modules()):
        if not name or not isinstance(module, nn.Linear):
            continue
        if not _should_quantize_int8_mma_module(
            name,
            module,
            quant_policy,
            selection_mode=selection_mode,
        ):
            continue
        module_activation_scale_mode = activation_scale_mode
        module_activation_scale = static_scales.get(name)
        if activation_scale_mode == "static":
            if module_activation_scale is None:
                module_activation_scale_mode = "dynamic"
                dynamic_fallback_modules += 1
            else:
                static_scale_modules += 1
        _replace_submodule(
            target_model,
            name,
            Int8MmaLinear.from_linear(
                module,
                engine=engine,
                fallback_engine=fallback_engine,
                block_m=block_m,
                block_n=block_n,
                block_k=block_k,
                threads=threads,
                num_stages=num_stages,
                activation_scale_mode=module_activation_scale_mode,
                activation_scale=module_activation_scale,
                activation_quant_block_size=activation_quant_block_size,
                eps=eps,
            ),
        )
        quantized_modules.append(name)

    return Int8MmaQuantizationResult(
        model=target_model,
        strategy=selected_strategy,
        quantized_modules=quantized_modules,
        metadata={
            "implementation": "dynamic_w8a8_int8_mma_linear",
            "quantization_nature": "true",
            "activation_encoding": f"{activation_scale_mode}_signed_int8_per_tensor",
            "weight_encoding": "signed_int8_per_output_channel",
            "accumulation": "int32",
            "engine": engine,
            "fallback_engine": fallback_engine,
            "activation_scale_mode": activation_scale_mode,
            "static_scale_module_count": static_scale_modules,
            "dynamic_fallback_module_count": dynamic_fallback_modules,
            "activation_quant_block_size": int(activation_quant_block_size),
            "block_m": int(block_m),
            "block_n": int(block_n),
            "block_k": int(block_k),
            "threads": int(threads),
            "num_stages": int(num_stages),
            "selection_mode": selection_mode,
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
            },
        },
    )


def execute_int8_mma_component(
    context: XQTContext,
    root_model: nn.Module,
    component: QuantizationComponentPlan,
    *,
    quantize_fn: Any = quantize_with_int8_mma,
) -> tuple[nn.Module, QuantizationReport]:
    """Execute the true W8A8 INT8 MMA quantizer for a component."""

    target_model = resolve_component_model(root_model, component.target_path)
    effective_policy = build_effective_selection_policy(component)
    result = quantize_fn(
        target_model,
        policy=effective_policy,
        strategy=component.strategy or effective_policy.get("strategy"),
        inplace=True,
        engine=str(component.policy.get("engine", "tilelang")),
        fallback_engine=str(component.policy.get("fallback_engine", "torch_int_mm")),
        block_m=int(component.policy.get("block_m", 64)),
        block_n=int(component.policy.get("block_n", 64)),
        block_k=int(component.policy.get("block_k", 64)),
        threads=int(component.policy.get("threads", 128)),
        num_stages=int(component.policy.get("num_stages", 2)),
        activation_scale_mode=str(component.policy.get("activation_scale_mode", "dynamic")),
        activation_scales=component.policy.get("activation_scales"),
        activation_quant_block_size=int(component.policy.get("activation_quant_block_size", 256)),
        eps=float(component.policy.get("eps", 1e-6)),
    )
    updated_model = replace_component_model(root_model, component.target_path, result.model)
    high_precision_modules = prefix_module_names(component.keep_high_precision, component.target_path)
    skipped_modules = ordered_unique(
        [
            *prefix_module_names(component.skip_quantize, component.target_path),
            *high_precision_modules,
        ]
    )
    quantized_modules = prefix_module_names(result.quantized_modules, component.target_path)
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
    method_semantics = "true_w8a8_int8_mma_runtime_quantization"
    report = QuantizationReport(
        component_name=component.name,
        backend=result.backend,
        runtime="pytorch",
        method=component.method,
        strategy=result.strategy,
        target_path=component.target_path,
        quantized_modules=quantized_modules,
        skipped_modules=skipped_modules,
        high_precision_modules=high_precision_modules,
        calibration_samples=calibration_samples,
        calibration_summary=calibration_summary,
        nature=QuantizationNature.TRUE,
        algorithm_executable=True,
        method_semantics=method_semantics,
        compute_speedup_expected=None,
        metadata={
            **dict(result.metadata),
            "analysis_only": component.analysis_only,
            "policy": effective_policy,
            "selection_policy": selection_policy_metadata(component),
            "module_selection_reasons": module_selection_reasons,
            "executed": True,
            "execution_state": result.strategy,
            "algorithm_executable": True,
            "method_semantics": method_semantics,
        },
    )
    return updated_model, report


__all__ = [
    "execute_int8_mma_component",
    "Int8MmaLinear",
    "Int8MmaQuantizationResult",
    "quantize_with_int8_mma",
]
