"""Reference storage semantics for W8A8 INT8 linear artifacts."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from xqt.core.base import XQTBackendError


_ACTIVATION_SCALE_MODES = frozenset({"dynamic", "static"})
_OUTPUT_DTYPES = frozenset({torch.float16, torch.bfloat16, torch.float32})


class Int8MmaLinear(nn.Module):
    """INT8 storage artifact with a backend-neutral reference forward."""

    quant_max: float = 127.0

    def __init__(
        self,
        qweight_t: torch.Tensor,
        weight_scale: torch.Tensor,
        *,
        bias: torch.Tensor | None,
        input_features: int,
        output_features: int,
        engine: str = "auto",
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
        preferred_engines: list[str] | tuple[str, ...] | None = None,
        min_int8_rows: int = 0,
    ) -> None:
        super().__init__()
        if output_dtype not in _OUTPUT_DTYPES:
            raise ValueError("output_dtype must be float16, bfloat16, or float32")
        if activation_scale_mode not in _ACTIVATION_SCALE_MODES:
            raise ValueError("activation_scale_mode must be dynamic or static")
        if min_int8_rows < 0:
            raise ValueError("min_int8_rows must be >= 0")
        self.input_features = int(input_features)
        self.output_features = int(output_features)
        self.engine = str(engine)
        self.fallback_engine = str(fallback_engine)
        self.block_m = int(block_m)
        self.block_n = int(block_n)
        self.block_k = int(block_k)
        self.threads = int(threads)
        self.num_stages = int(num_stages)
        self.output_dtype = output_dtype
        self.activation_scale_mode = str(activation_scale_mode)
        self.activation_quant_block_size = int(activation_quant_block_size)
        self.eps = float(eps)
        self.preferred_engines = [str(item) for item in preferred_engines or ()]
        self.min_int8_rows = int(min_int8_rows)
        self.last_execution: dict[str, Any] = {"engine": "not_run"}
        self.register_buffer("qweight_t", qweight_t.to(torch.int8).contiguous())
        self.register_buffer(
            "weight_scale", weight_scale.to(torch.float32).contiguous()
        )
        if bias is None:
            self.register_buffer("bias", None)
        else:
            self.register_buffer("bias", bias.detach().to(torch.float32).contiguous())
        initial_scale = torch.tensor(0.0, dtype=torch.float32)
        self._has_static_activation_scale = activation_scale is not None
        if activation_scale is not None:
            initial_scale = torch.as_tensor(
                activation_scale, dtype=torch.float32
            ).reshape(())
        self.register_buffer("static_activation_scale", initial_scale)

    @classmethod
    def from_linear(
        cls,
        module: nn.Linear,
        *,
        engine: str = "auto",
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
        preferred_engines: list[str] | tuple[str, ...] | None = None,
        min_int8_rows: int = 0,
    ) -> "Int8MmaLinear":
        weight = module.weight.detach().to(torch.float32)
        max_abs = weight.abs().amax(dim=1, keepdim=True)
        scale = torch.where(
            max_abs > float(eps),
            max_abs / cls.quant_max,
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
            preferred_engines=preferred_engines,
            min_int8_rows=min_int8_rows,
        )

    @staticmethod
    def activation_scale_from_inputs(
        inputs: torch.Tensor, eps: float = 1e-6
    ) -> torch.Tensor:
        """Return the symmetric per-tensor INT8 activation scale."""

        max_abs = (
            inputs.detach()
            .reshape(-1)
            .to(torch.float32)
            .abs()
            .amax()
            .clamp_min(float(eps))
        )
        return (max_abs / 127.0).reshape(())

    def set_static_activation_scale(self, scale: torch.Tensor | float) -> None:
        """Set the calibrated reference activation scale."""

        value = torch.as_tensor(
            scale,
            dtype=torch.float32,
            device=self.static_activation_scale.device,
        ).reshape(())
        if float(value.detach().cpu().item()) <= 0.0:
            raise ValueError("static activation scale must be positive")
        self.static_activation_scale.copy_(value)
        self._has_static_activation_scale = True
        self.activation_scale_mode = "static"

    def calibrate_static_activation_scale(
        self, sample_inputs: torch.Tensor
    ) -> torch.Tensor:
        """Calibrate the reference activation scale from representative inputs."""

        scale = self.activation_scale_from_inputs(sample_inputs, eps=self.eps).to(
            device=self.static_activation_scale.device,
            dtype=torch.float32,
        )
        self.set_static_activation_scale(scale)
        return self.static_activation_scale.detach().clone()

    def _static_activation_scale(self, device: torch.device) -> torch.Tensor:
        if not self._has_static_activation_scale:
            raise XQTBackendError(
                "static activation scale mode requires "
                "calibrate_static_activation_scale() or "
                "set_static_activation_scale() before forward"
            )
        return self.static_activation_scale.to(device=device, dtype=torch.float32)

    def dequantize_weight(self) -> torch.Tensor:
        """Materialize the reference floating-point weight."""

        return self.qweight_t.to(torch.float32).t() * self.weight_scale.reshape(-1, 1)

    def _run_float_fallback(self, inputs: torch.Tensor) -> torch.Tensor:
        compute_dtype = (
            inputs.dtype if inputs.dtype in _OUTPUT_DTYPES else torch.float32
        )
        weight = self.dequantize_weight().to(
            device=inputs.device, dtype=compute_dtype
        )
        bias = (
            None
            if self.bias is None
            else self.bias.to(device=inputs.device, dtype=compute_dtype)
        )
        output = F.linear(inputs.to(compute_dtype), weight, bias)
        self.last_execution = {
            "engine": "bf16_fallback",
            "reason": "rows_below_min_int8_rows",
            "true_int8_mma": False,
            "min_int8_rows": self.min_int8_rows,
            "input_rows": int(inputs.numel() // self.input_features),
            "input_features": self.input_features,
            "output_features": self.output_features,
        }
        return output.to(self.output_dtype)

    def _activation_scale(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.activation_scale_mode == "static":
            return self._static_activation_scale(inputs.device)
        return self.activation_scale_from_inputs(inputs, eps=self.eps).to(inputs.device)

    def run_quantized_activation(
        self,
        qactivation: torch.Tensor,
        activation_scale: torch.Tensor | float,
    ) -> torch.Tensor:
        """Run reference INT8 accumulation for an already quantized activation."""

        flat = qactivation.reshape(-1, self.input_features).to(torch.int8)
        original_rows = int(flat.shape[0])
        padded = flat
        if flat.is_cuda:
            aligned_rows = max(32, ((original_rows + 31) // 32) * 32)
            if aligned_rows > original_rows:
                padded = torch.cat(
                    [
                        flat,
                        torch.zeros(
                            aligned_rows - original_rows,
                            self.input_features,
                            dtype=torch.int8,
                            device=flat.device,
                        ),
                    ],
                    dim=0,
                )
        accum = torch._int_mm(padded.contiguous(), self.qweight_t)
        accum = accum[:original_rows]
        scale = torch.as_tensor(
            activation_scale, dtype=torch.float32, device=flat.device
        ).reshape(())
        output = accum.to(torch.float32) * (
            scale * self.weight_scale.to(flat.device)
        ).reshape(1, -1)
        if self.bias is not None:
            output = output + self.bias.to(flat.device).reshape(1, -1)
        return output.to(self.output_dtype).reshape(
            *qactivation.shape[:-1], self.output_features
        )

    def _runtime_precision_summary(self) -> dict[str, Any]:
        execution = self.last_execution
        engine = str(execution.get("engine", "not_run"))
        float_fallback = engine == "bf16_fallback"
        int8_operands = (
            not float_fallback
            and execution.get("activation_dtype") == "int8"
            and execution.get("weight_dtype") == "int8"
        )
        execution_kind = (
            "not_run"
            if engine == "not_run"
            else "floating_point_fallback"
            if float_fallback
            else "w8a8_int8_reference"
            if int8_operands
            else "unclassified"
        )
        return {
            "requested": "w8a8_int8_mma",
            "weight_storage": "signed_int8_per_output_channel",
            "activation_encoding": "signed_int8_per_tensor",
            "activation_granularity": "per_tensor",
            "execution_kind": execution_kind,
            "int8_operands_executed": bool(int8_operands),
            "native_mma_executed": False,
            "float_fallback_taken": float_fallback,
            "float_fallback_reason": (
                execution.get("reason") if float_fallback else None
            ),
            "small_batch_float_fallback": {
                "enabled": self.min_int8_rows > 0,
                "condition": "input_rows < min_int8_rows",
                "min_int8_rows": self.min_int8_rows,
                "status": (
                    "disabled"
                    if self.min_int8_rows == 0
                    else "taken"
                    if float_fallback
                    else "not_taken"
                ),
            },
            "engine_error_int8_fallback": {
                "enabled": self.fallback_engine == "torch_int_mm",
                "condition": "selected non-reference INT8 engine raises or fails its runtime check",
                "status": "not_taken",
            },
        }

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.shape[-1] != self.input_features:
            raise XQTBackendError(
                "Int8MmaLinear input trailing dimension does not match input_features"
            )
        rows = int(inputs.numel() // self.input_features)
        if 0 < self.min_int8_rows and rows < self.min_int8_rows:
            return self._run_float_fallback(inputs)
        activation_scale = self._activation_scale(inputs)
        qactivation = torch.round(inputs.to(torch.float32) / activation_scale).clamp(
            -127, 127
        ).to(torch.int8)
        output = self.run_quantized_activation(qactivation, activation_scale)
        self.last_execution = {
            "engine": "torch_int_mm",
            "reason": "contracts_reference_semantics",
            "true_int8_mma": False,
            "activation_dtype": "int8",
            "weight_dtype": "int8",
            "accumulation_dtype": "int32",
            "activation_scale_mode": self.activation_scale_mode,
            "activation_quant_engine": "torch_per_tensor_reference",
            "input_rows": rows,
            "input_features": self.input_features,
            "output_features": self.output_features,
        }
        return output

    def execution_metadata(self) -> dict[str, Any]:
        metadata = dict(self.last_execution)
        metadata["runtime_precision"] = self._runtime_precision_summary()
        metadata["artifact_view"] = "contracts_reference"
        return metadata


__all__ = ["Int8MmaLinear"]
