"""TileLang decode pipeline for one quantized HunyuanOCR text decoder block."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from xqt.benchmark import benchmark_callable
from xqt.benchmark.latency import LatencyReport
from xqt.core.errors import XQTBackendError
from xqt.operator_opt.kernels.tilelang.hunyuan_block import (
    gqa_decode_attention_tilelang,
    residual_add_tilelang,
    residual_rmsnorm_tilelang,
    rmsnorm_tilelang,
    swiglu_tilelang,
)
from xqt.runtime.modules import Int8MmaLinear, SVDQuantInt8MmaLinear


QKPositionTransform = Callable[
    [torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]
]
W8A8Projection = Int8MmaLinear | SVDQuantInt8MmaLinear


@dataclass(frozen=True)
class HunyuanOcrTileLangDecodeSpec:
    """Static decode shape required by the HunyuanOCR TileLang block pipeline.

    Real tencent/HunyuanOCR text decoder uses independent attention width:
    ``attention_dim = query_heads * head_dim`` may differ from ``hidden_size``.
    Released checkpoint: hidden=1024, heads=16/8, head_dim=128 so
    ``q_proj`` is 1024->2048 and ``o_proj`` is 2048->1024.
    """

    batch_size: int
    hidden_size: int
    query_heads: int
    key_value_heads: int
    head_dim: int
    kv_cache_length: int
    intermediate_size: int
    target_arch: str = "sm_89"
    int8_block_m: int = 16
    gqa_query_tile_rows: int = 1
    decode_min_int8_rows: int = 16

    @property
    def attention_dim(self) -> int:
        """Flattened Q/O attention width: query_heads * head_dim."""

        return self.query_heads * self.head_dim

    @property
    def key_value_dim(self) -> int:
        """Flattened K/V projection width: key_value_heads * head_dim."""

        return self.key_value_heads * self.head_dim

    def validate(self) -> None:
        """Validate shape constraints imposed by the static TileLang kernels."""

        if self.batch_size != 1:
            raise ValueError(
                "HunyuanOCR TileLang decode currently requires batch_size=1"
            )
        if self.query_heads % self.key_value_heads != 0:
            raise ValueError("query_heads must be divisible by key_value_heads")
        if self.head_dim <= 0:
            raise ValueError("head_dim must be positive")
        if self.attention_dim % 64 != 0:
            raise ValueError(
                "query_heads * head_dim (attention_dim) must be a multiple of 64"
            )
        if self.key_value_dim % 64 != 0:
            raise ValueError(
                "key_value_heads * head_dim must be a multiple of 64"
            )
        if self.kv_cache_length <= 0 or self.kv_cache_length % 64 != 0:
            raise ValueError("kv_cache_length must be a positive multiple of 64")
        if self.hidden_size % 64 != 0 or self.intermediate_size % 64 != 0:
            raise ValueError(
                "hidden_size and intermediate_size must be multiples of 64"
            )
        if self.int8_block_m <= 0 or self.int8_block_m % 16 != 0:
            raise ValueError(
                "int8_block_m must be a positive multiple of 16 (sm_89 MMA floor)"
            )
        if self.gqa_query_tile_rows not in {1, 64}:
            raise ValueError("gqa_query_tile_rows must be 1 (exact) or 64 (legacy MMA)")
        if self.decode_min_int8_rows < 0:
            raise ValueError("decode_min_int8_rows must be >= 0")


def _module_weight(module: nn.Module, name: str) -> torch.Tensor:
    weight = getattr(module, "weight", None)
    if not isinstance(weight, torch.Tensor):
        raise XQTBackendError(f"HunyuanOCR {name} module must expose a Tensor weight")
    return weight


def _rmsnorm_eps(module: nn.Module, name: str) -> float:
    eps = getattr(module, "variance_epsilon", getattr(module, "eps", None))
    if eps is None:
        raise XQTBackendError(f"HunyuanOCR {name} module must expose RMSNorm epsilon")
    return float(eps)


class HunyuanOcrTileLangDecodeBlock(nn.Module):
    """Static-cache TileLang decode pipeline for a HunyuanOCR decoder layer.

    ``position_transform`` owns Hunyuan's XD-RoPE semantics. It receives and
    returns tensors shaped ``[B, heads, 1, head_dim]`` before Q/K RMSNorm.
    The caller owns cache-length progression: one block instance is built for a
    fixed exact cache length and writes the current K/V token into its last slot.
    """

    def __init__(
        self,
        layer: nn.Module,
        spec: HunyuanOcrTileLangDecodeSpec,
        *,
        position_transform: QKPositionTransform,
    ) -> None:
        super().__init__()
        spec.validate()
        self.spec = spec
        self.position_transform = position_transform
        self.input_layernorm = self._require_module(layer, "input_layernorm")
        self.post_attention_layernorm = self._require_module(
            layer,
            "post_attention_layernorm",
        )
        self.self_attn = self._require_module(layer, "self_attn")
        self.mlp = self._require_module(layer, "mlp")
        self.q_proj = self._require_w8a8_projection(
            self.self_attn,
            "q_proj",
            input_features=spec.hidden_size,
            output_features=spec.attention_dim,
        )
        self.k_proj = self._require_w8a8_projection(
            self.self_attn,
            "k_proj",
            input_features=spec.hidden_size,
            output_features=spec.key_value_dim,
        )
        self.v_proj = self._require_w8a8_projection(
            self.self_attn,
            "v_proj",
            input_features=spec.hidden_size,
            output_features=spec.key_value_dim,
        )
        self.o_proj = self._require_w8a8_projection(
            self.self_attn,
            "o_proj",
            input_features=spec.attention_dim,
            output_features=spec.hidden_size,
        )
        self.gate_proj = self._require_w8a8_projection(
            self.mlp,
            "gate_proj",
            input_features=spec.hidden_size,
            output_features=spec.intermediate_size,
        )
        self.up_proj = self._require_w8a8_projection(
            self.mlp,
            "up_proj",
            input_features=spec.hidden_size,
            output_features=spec.intermediate_size,
        )
        self.down_proj = self._require_w8a8_projection(
            self.mlp,
            "down_proj",
            input_features=spec.intermediate_size,
            output_features=spec.hidden_size,
        )
        self.input_norm_weight = _module_weight(self.input_layernorm, "input_layernorm")
        self.post_attention_norm_weight = _module_weight(
            self.post_attention_layernorm,
            "post_attention_layernorm",
        )
        self.query_norm_weight = _module_weight(
            self._require_module(self.self_attn, "query_layernorm"), "query_layernorm"
        )
        self.key_norm_weight = _module_weight(
            self._require_module(self.self_attn, "key_layernorm"), "key_layernorm"
        )
        if int(self.query_norm_weight.shape[0]) != spec.head_dim:
            raise XQTBackendError(
                "HunyuanOCR query_layernorm weight must match head_dim "
                f"({spec.head_dim}), got {tuple(self.query_norm_weight.shape)}"
            )
        if int(self.key_norm_weight.shape[0]) != spec.head_dim:
            raise XQTBackendError(
                "HunyuanOCR key_layernorm weight must match head_dim "
                f"({spec.head_dim}), got {tuple(self.key_norm_weight.shape)}"
            )
        if int(self.input_norm_weight.shape[0]) != spec.hidden_size:
            raise XQTBackendError(
                "HunyuanOCR input_layernorm weight must match hidden_size "
                f"({spec.hidden_size}), got {tuple(self.input_norm_weight.shape)}"
            )
        if int(self.post_attention_norm_weight.shape[0]) != spec.hidden_size:
            raise XQTBackendError(
                "HunyuanOCR post_attention_layernorm weight must match hidden_size "
                f"({spec.hidden_size}), got {tuple(self.post_attention_norm_weight.shape)}"
            )
        self.input_norm_eps = _rmsnorm_eps(self.input_layernorm, "input_layernorm")
        self.post_attention_norm_eps = _rmsnorm_eps(
            self.post_attention_layernorm,
            "post_attention_layernorm",
        )
        self.query_norm_eps = _rmsnorm_eps(
            self._require_module(self.self_attn, "query_layernorm"),
            "query_layernorm",
        )
        self.key_norm_eps = _rmsnorm_eps(
            self._require_module(self.self_attn, "key_layernorm"),
            "key_layernorm",
        )
        self._apply_decode_projection_schedule(
            block_m=spec.int8_block_m,
            min_int8_rows=spec.decode_min_int8_rows,
        )
        self._qkv_fusion_modules = self._resolve_fusion_modules(
            [self.q_proj, self.k_proj, self.v_proj]
        )
        self._gate_up_fusion_modules = self._resolve_fusion_modules(
            [self.gate_proj, self.up_proj]
        )

    @staticmethod
    def _require_module(root: nn.Module, name: str) -> nn.Module:
        module = getattr(root, name, None)
        if not isinstance(module, nn.Module):
            raise XQTBackendError(
                f"HunyuanOCR decoder layer is missing module '{name}'"
            )
        return module

    def _apply_decode_projection_schedule(
        self,
        *,
        block_m: int,
        min_int8_rows: int,
    ) -> None:
        tile = int(block_m)
        row_threshold = int(min_int8_rows)
        for module in (
            self.q_proj,
            self.k_proj,
            self.v_proj,
            self.o_proj,
            self.gate_proj,
            self.up_proj,
            self.down_proj,
        ):
            targets: list[Any] = []
            if isinstance(module, Int8MmaLinear):
                targets.append(module)
            residual = getattr(module, "residual_int8", None)
            if residual is not None:
                targets.append(residual)
                compute = getattr(residual, "_compute", None)
                if compute is not None:
                    targets.append(compute)
            for target in targets:
                if hasattr(target, "block_m"):
                    object.__setattr__(target, "block_m", tile)
                if hasattr(target, "min_int8_rows"):
                    object.__setattr__(target, "min_int8_rows", row_threshold)
                if row_threshold > 0 and isinstance(target, Int8MmaLinear):
                    qweight = target.qweight_t
                    if isinstance(qweight, torch.Tensor) and qweight.is_cuda:
                        target._dense_bf16_weight(target.output_dtype, qweight.device)

    @staticmethod
    def _require_w8a8_projection(
        root: nn.Module,
        name: str,
        *,
        input_features: int,
        output_features: int,
    ) -> W8A8Projection:
        module = getattr(root, name, None)
        if not isinstance(module, (Int8MmaLinear, SVDQuantInt8MmaLinear)):
            raise XQTBackendError(
                "HunyuanOCR TileLang block requires a W8A8 projection "
                f"(Int8MmaLinear or SVDQuantInt8MmaLinear) at '{name}'"
            )
        module_in = int(getattr(module, "input_features"))
        module_out = int(getattr(module, "output_features"))
        if module_in != int(input_features) or module_out != int(output_features):
            raise XQTBackendError(
                "HunyuanOCR TileLang projection dimension mismatch at "
                f"'{name}': expected ({input_features}, {output_features}), "
                f"got ({module_in}, {module_out})"
            )
        return module

    def _reshape_q(self, projection: torch.Tensor) -> torch.Tensor:
        expected = (
            self.spec.batch_size,
            1,
            self.spec.attention_dim,
        )
        if tuple(projection.shape) != expected:
            raise XQTBackendError(
                f"HunyuanOCR TileLang q_proj expects shape {expected}, "
                f"got {tuple(projection.shape)}"
            )
        return projection.reshape(
            self.spec.batch_size,
            1,
            self.spec.query_heads,
            self.spec.head_dim,
        ).transpose(1, 2)

    def _reshape_kv(self, projection: torch.Tensor) -> torch.Tensor:
        expected = (
            self.spec.batch_size,
            1,
            self.spec.key_value_dim,
        )
        if tuple(projection.shape) != expected:
            raise XQTBackendError(
                f"HunyuanOCR TileLang k/v_proj expects shape {expected}, "
                f"got {tuple(projection.shape)}"
            )
        return projection.reshape(
            self.spec.batch_size,
            1,
            self.spec.key_value_heads,
            self.spec.head_dim,
        ).transpose(1, 2)

    def _flatten_attention(self, attention: torch.Tensor) -> torch.Tensor:
        """Reshape attention heads to the o_proj input width (attention_dim)."""

        return attention.transpose(1, 2).reshape(
            self.spec.batch_size,
            1,
            self.spec.attention_dim,
        )

    @staticmethod
    def _resolve_fusion_modules(
        modules: list[W8A8Projection],
    ) -> list[Int8MmaLinear] | None:
        resolved: list[Int8MmaLinear] = []
        for module in modules:
            if not isinstance(module, Int8MmaLinear):
                return None
            if module.bias is not None:
                return None
            if module.activation_scale_mode != "static":
                return None
            if not module._has_static_activation_scale:
                return None
            resolved.append(module)
        if not resolved:
            return None
        first_buffer = resolved[0].static_activation_scale
        if not isinstance(first_buffer, torch.Tensor):
            return None
        first_scale = float(first_buffer.detach().cpu().reshape(-1)[0])
        for module in resolved[1:]:
            buffer = module.static_activation_scale
            if not isinstance(buffer, torch.Tensor):
                return None
            scale = float(buffer.detach().cpu().reshape(-1)[0])
            if scale != first_scale:
                return None
        return resolved

    def _fused_int8_projections(
        self,
        inputs: torch.Tensor,
        modules: list[Int8MmaLinear],
    ) -> torch.Tensor:
        weights: list[torch.Tensor] = []
        scales: list[torch.Tensor] = []
        for module in modules:
            qweight = module.qweight_t
            weight_scale = module.weight_scale
            if not isinstance(qweight, torch.Tensor) or not isinstance(
                weight_scale, torch.Tensor
            ):
                raise XQTBackendError("fused INT8 projection requires Tensor buffers")
            weights.append(qweight)
            scales.append(weight_scale)
        packed_weight = torch.cat(weights, dim=1)
        packed_scale = torch.cat(scales, dim=0)
        activation_scale = modules[0]._static_activation_scale(inputs.device)
        flat = inputs.reshape(-1, modules[0].input_features)
        from xqt.operator_opt.kernels.cute.int8mma_binding import (
            int8_gemv_m1_fused_static_ptx_sm89,
            int8mma_available,
            prepack_qweight_t_for_ptx_sm89,
        )
        from xqt.operator_opt.kernels.tilelang.int8_mma import (
            int8_linear_static_activation_m1_tilelang,
        )

        backend = "torch_m1_int8_products"
        if flat.is_cuda and flat.dtype == torch.float16 and int8mma_available():
            try:
                packed_b = prepack_qweight_t_for_ptx_sm89(packed_weight)
                output = int8_gemv_m1_fused_static_ptx_sm89(
                    flat,
                    packed_weight,
                    activation_scale,
                    packed_scale,
                    None,
                    output_dtype=modules[0].output_dtype,
                    prepacked_b=packed_b,
                )
                backend = "ptx_sm89_m1_dp4a_gemv"
            except Exception:
                output = int8_linear_static_activation_m1_tilelang(
                    flat,
                    packed_weight,
                    activation_scale,
                    packed_scale,
                    None,
                    output_dtype=modules[0].output_dtype,
                )
        else:
            output = int8_linear_static_activation_m1_tilelang(
                flat,
                packed_weight,
                activation_scale,
                packed_scale,
                None,
                output_dtype=modules[0].output_dtype,
            )
        for module in modules:
            module.last_execution = {
                "engine": "ptx_sm89" if backend.startswith("ptx") else "tilelang",
                "reason": "true_int8_m1_dp4a_gemv_fused",
                "true_int8_mma": True,
                "activation_dtype": "int8",
                "weight_dtype": "int8",
                "accumulation_dtype": "int32",
                "activation_scale_mode": "static",
                "activation_quant_engine": backend,
                "fused_static_status": "used",
                "input_rows": 1,
                "padded_rows": 1,
                "input_features": module.input_features,
                "output_features": module.output_features,
            }
        return output.reshape(*inputs.shape[:-1], output.shape[-1])

    def _fused_qkv_projections(
        self,
        normed: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        modules = self._qkv_fusion_modules
        rows = int(normed.reshape(-1, self.spec.hidden_size).shape[0])
        use_int8_fusion = (
            modules is not None
            and rows == 1
            and modules[0].min_int8_rows == 0
        )
        if not use_int8_fusion:
            return {
                "query": self.q_proj(normed),
                "key": self.k_proj(normed),
                "value": self.v_proj(normed),
            }
        projected = self._fused_int8_projections(normed, modules)
        query_end = self.spec.attention_dim
        key_end = query_end + self.spec.key_value_dim
        return {
            "query": projected[..., :query_end],
            "key": projected[..., query_end:key_end],
            "value": projected[..., key_end:],
        }

    def _fused_gate_up_projections(
        self,
        normed: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        modules = self._gate_up_fusion_modules
        rows = int(normed.reshape(-1, self.spec.hidden_size).shape[0])
        use_int8_fusion = (
            modules is not None
            and rows == 1
            and modules[0].min_int8_rows == 0
        )
        if not use_int8_fusion:
            return {
                "gate": self.gate_proj(normed),
                "up": self.up_proj(normed),
            }
        projected = self._fused_int8_projections(normed, modules)
        mid = self.spec.intermediate_size
        return {
            "gate": projected[..., :mid],
            "up": projected[..., mid:],
        }

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
    ) -> torch.Tensor:
        """Run one token through the complete static-cache decoder block pipeline."""

        expected_hidden = (self.spec.batch_size, 1, self.spec.hidden_size)
        expected_cache = (
            self.spec.batch_size,
            self.spec.key_value_heads,
            self.spec.kv_cache_length,
            self.spec.head_dim,
        )
        if tuple(hidden_states.shape) != expected_hidden:
            raise XQTBackendError(
                f"HunyuanOCR TileLang decode expects hidden_states {expected_hidden}"
            )
        if (
            tuple(key_cache.shape) != expected_cache
            or tuple(value_cache.shape) != expected_cache
        ):
            raise XQTBackendError(
                f"HunyuanOCR TileLang decode expects KV cache {expected_cache}"
            )
        if hidden_states.dtype not in {torch.float16, torch.bfloat16}:
            raise XQTBackendError("HunyuanOCR TileLang decode requires fp16 or bf16")

        normed = rmsnorm_tilelang(
            hidden_states,
            self.input_norm_weight,
            eps=self.input_norm_eps,
            target_arch=self.spec.target_arch,
        )
        query_key_value = self._fused_qkv_projections(normed)
        query = self._reshape_q(query_key_value["query"])
        key = self._reshape_kv(query_key_value["key"])
        value = self._reshape_kv(query_key_value["value"])
        query, key = self.position_transform(query, key)
        query = rmsnorm_tilelang(
            query,
            self.query_norm_weight,
            eps=self.query_norm_eps,
            target_arch=self.spec.target_arch,
        )
        key = rmsnorm_tilelang(
            key,
            self.key_norm_weight,
            eps=self.key_norm_eps,
            target_arch=self.spec.target_arch,
        )
        key_cache[:, :, -1, :].copy_(key.squeeze(2))
        value_cache[:, :, -1, :].copy_(value.squeeze(2))
        attention = gqa_decode_attention_tilelang(
            query.squeeze(2),
            key_cache,
            value_cache,
            query_tile_rows=self.spec.gqa_query_tile_rows,
            target_arch=self.spec.target_arch,
        )
        attention_output = self.o_proj(self._flatten_attention(attention))
        residual, normed = residual_rmsnorm_tilelang(
            attention_output,
            hidden_states,
            self.post_attention_norm_weight,
            eps=self.post_attention_norm_eps,
            target_arch=self.spec.target_arch,
        )
        gate_up = self._fused_gate_up_projections(normed)
        activated = swiglu_tilelang(
            gate_up["gate"],
            gate_up["up"],
            target_arch=self.spec.target_arch,
        )
        mlp_output = self.down_proj(activated)
        return residual_add_tilelang(
            mlp_output,
            residual,
            target_arch=self.spec.target_arch,
        )


class HunyuanOcrTileLangCudaGraphRunner:
    """CUDA Graph runner for one exact-KV-length HunyuanOCR decode block.

    ``replay`` returns graph-owned output storage. It remains valid until the
    next replay, which lets the steady-state path avoid a graph-external clone.
    """

    def __init__(
        self,
        block: HunyuanOcrTileLangDecodeBlock,
        *,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
    ) -> None:
        if not torch.cuda.is_available():
            raise XQTBackendError("HunyuanOCR TileLang CUDA Graph requires CUDA")
        if not key_cache.is_cuda or not value_cache.is_cuda:
            raise XQTBackendError(
                "HunyuanOCR TileLang CUDA Graph requires CUDA KV cache"
            )
        if key_cache.device != value_cache.device:
            raise XQTBackendError("HunyuanOCR TileLang KV caches must share a device")
        if key_cache.dtype != value_cache.dtype:
            raise XQTBackendError("HunyuanOCR TileLang KV caches must share a dtype")
        self.block = block.eval()
        self.key_cache = key_cache
        self.value_cache = value_cache
        self.static_hidden = torch.empty(
            (1, 1, block.spec.hidden_size),
            device=key_cache.device,
            dtype=key_cache.dtype,
        )
        self.static_output: torch.Tensor | None = None
        self.graph = torch.cuda.CUDAGraph()
        self._captured = False

    def _validate_hidden_states(self, hidden_states: torch.Tensor) -> None:
        expected_shape = (1, 1, self.block.spec.hidden_size)
        if tuple(hidden_states.shape) != expected_shape:
            raise XQTBackendError(
                f"HunyuanOCR TileLang CUDA Graph expects hidden_states {expected_shape}"
            )
        if hidden_states.device != self.static_hidden.device:
            raise XQTBackendError(
                "HunyuanOCR TileLang CUDA Graph hidden_states must use the cache device"
            )
        if hidden_states.dtype != self.static_hidden.dtype:
            raise XQTBackendError(
                "HunyuanOCR TileLang CUDA Graph hidden_states must use the cache dtype"
            )

    def capture(self, warmup_hidden: torch.Tensor) -> None:
        """Warm and capture the complete multi-kernel decoder block invocation."""

        if self._captured:
            raise RuntimeError("HunyuanOCR TileLang CUDA Graph is already captured")
        self._validate_hidden_states(warmup_hidden)
        self.static_hidden.copy_(warmup_hidden)
        stream = torch.cuda.Stream(device=self.static_hidden.device)
        stream.wait_stream(torch.cuda.current_stream(self.static_hidden.device))
        with torch.no_grad(), torch.cuda.stream(stream):
            for _ in range(3):
                self.static_output = self.block(
                    self.static_hidden,
                    self.key_cache,
                    self.value_cache,
                )
        torch.cuda.current_stream(self.static_hidden.device).wait_stream(stream)
        torch.cuda.synchronize(self.static_hidden.device)
        with torch.no_grad(), torch.cuda.graph(self.graph):
            self.static_output = self.block(
                self.static_hidden,
                self.key_cache,
                self.value_cache,
            )
        self._captured = True

    def replay(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Replay and return graph-owned output for the same static KV cache length."""

        if not self._captured or self.static_output is None:
            raise RuntimeError("capture() must run before CUDA Graph replay")
        self._validate_hidden_states(hidden_states)
        with torch.no_grad():
            self.static_hidden.copy_(hidden_states)
            self.graph.replay()
        return self.static_output


def benchmark_hunyuan_ocr_tilelang_decode_graph(
    runner: HunyuanOcrTileLangCudaGraphRunner,
    hidden_states: torch.Tensor,
    *,
    warmup: int = 20,
    iterations: int = 100,
) -> LatencyReport:
    """Measure steady-state replay latency for one captured Hunyuan decoder block."""

    return benchmark_callable(
        lambda: runner.replay(hidden_states),
        warmup=warmup,
        iterations=iterations,
        sync_cuda=True,
        device=str(hidden_states.device),
    )


__all__ = [
    "HunyuanOcrTileLangCudaGraphRunner",
    "HunyuanOcrTileLangDecodeBlock",
    "HunyuanOcrTileLangDecodeSpec",
    "QKPositionTransform",
    "benchmark_hunyuan_ocr_tilelang_decode_graph",
]
