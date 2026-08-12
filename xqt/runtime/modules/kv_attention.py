"""Model-side KV-scale attention reference entity (C9 residual).

This module consumes the canonical KV scale artifacts (``KvScaleArtifact``
payloads, i.e. per-tensor k/v scales) and an optional ``RuntimeQuantContract``,
and runs a torch SDPA reference path with int8 quantize/dequantize of K and V.

It is a model-side *reference entity*: there is no page table, block pool,
cache eviction, or serving scheduler here. ``selected_kernel`` and fallback
reasons are recorded for every forward. The default path stays the torch SDPA
reference; an opt-in CUDA fused path (``preferred_kernel="auto"|"tilelang"``)
runs one packed QKV projection followed by the TileLang KV-int8 quantize-layout
and fused attention kernels. The attention kernel consumes int8 K/V storage,
reads Q directly from the packed projection, and dequantizes inside the kernel.
An independent ``attention_fastpath="graph"`` option captures the complete
packed forward and replays it for matching fixed tensor contracts.
The fused path is only reported as verified after a fused forward actually ran
on device; any unavailability or kernel error falls back to the reference path
with an honest ``fallback_reason``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch import nn

from xqt.contracts.contract_consume import (
    ContractConsumeReport,
    consume_runtime_quant_contract,
)
from xqt.contracts.runtime_quant import (
    RUNTIME_QUANT_CONTRACT_KEY,
    RuntimeQuantContract,
)
from xqt.core.errors import XQTBackendError, XQTConfigError
from xqt.operator_opt.runtime import target_arch_mismatch


@dataclass(frozen=True, slots=True)
class KvCacheMetadata:
    """Model-side KV cache metadata; no cache storage or management."""

    dtype: str
    mode: str
    layer_scales: tuple[tuple[str, float, float], ...] = ()
    cache_block_size: int | None = None
    hit_rate: float | None = None
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "dtype": self.dtype,
            "mode": self.mode,
            "layer_scales": [
                {
                    "layer_path": layer_path,
                    "k_scale": float(k_scale),
                    "v_scale": float(v_scale),
                }
                for layer_path, k_scale, v_scale in self.layer_scales
            ],
            "cache_block_size": self.cache_block_size,
            "hit_rate": self.hit_rate,
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> KvCacheMetadata:
        if not isinstance(payload, Mapping):
            raise XQTConfigError(
                "KvCacheMetadata.from_dict expects a mapping; "
                f"got {type(payload).__name__}"
            )
        dtype = str(payload.get("dtype", "int8")).strip().lower()
        mode = str(payload.get("mode", "per_tensor_scale")).strip().lower()
        scales: list[tuple[str, float, float]] = []
        raw_scales = payload.get("layer_scales", ())
        if not isinstance(raw_scales, (list, tuple)):
            raise XQTConfigError("KvCacheMetadata.layer_scales must be a sequence")
        for item in raw_scales:
            if not isinstance(item, Mapping):
                raise XQTConfigError(
                    "KvCacheMetadata.layer_scales items must be mappings"
                )
            layer_path = str(item.get("layer_path", ""))
            k_scale = item.get("k_scale", item.get("attn.k_scale"))
            v_scale = item.get("v_scale", item.get("attn.v_scale"))
            if k_scale is None or v_scale is None:
                raise XQTConfigError(
                    f"KvCacheMetadata layer {layer_path!r} requires k_scale/v_scale"
                )
            scales.append(
                (layer_path, float(k_scale), float(v_scale))
            )
        raw_block_size = payload.get("cache_block_size")
        raw_hit_rate = payload.get("hit_rate")
        raw_notes = payload.get("notes", ())
        if not isinstance(raw_notes, (list, tuple)):
            raise XQTConfigError("KvCacheMetadata.notes must be a sequence")
        return cls(
            dtype=dtype,
            mode=mode,
            layer_scales=tuple(scales),
            cache_block_size=(
                None if raw_block_size is None else int(raw_block_size)
            ),
            hit_rate=None if raw_hit_rate is None else float(raw_hit_rate),
            notes=tuple(str(item) for item in raw_notes),
        )

    @classmethod
    def from_artifacts(
        cls,
        artifacts: Mapping[str, Any],
        *,
        dtype: str | None = None,
        mode: str | None = None,
    ) -> KvCacheMetadata:
        """Extract per-layer K/V scales from canonical KV quant metadata."""

        if not isinstance(artifacts, Mapping):
            raise XQTConfigError(
                "KvCacheMetadata.from_artifacts expects a mapping"
            )
        raw_layers = artifacts.get("layers")
        layer_source: Mapping[str, Any]
        if isinstance(raw_layers, Mapping):
            layer_source = raw_layers
        else:
            layer_source = artifacts
        scales: list[tuple[str, float, float]] = []
        for layer_path, raw_artifact in layer_source.items():
            if not isinstance(raw_artifact, Mapping):
                continue
            k_scale = raw_artifact.get(
                "k_scale", raw_artifact.get("attn.k_scale")
            )
            v_scale = raw_artifact.get(
                "v_scale", raw_artifact.get("attn.v_scale")
            )
            if k_scale is None or v_scale is None:
                continue
            scales.append(
                (str(layer_path), float(k_scale), float(v_scale))
            )
        if not scales:
            raise XQTConfigError(
                "KvCacheMetadata.from_artifacts found no k_scale/v_scale layers"
            )
        return cls(
            dtype=str(dtype or artifacts.get("dtype", "int8")).strip().lower(),
            mode=str(
                mode or artifacts.get("mode", "per_tensor_scale")
            ).strip().lower(),
            layer_scales=tuple(scales),
            notes=(
                "model-side metadata only; no cache storage or management",
            ),
        )

    def scales_for(self, layer_path: str) -> tuple[float, float] | None:
        for candidate_path, k_scale, v_scale in self.layer_scales:
            if candidate_path == layer_path:
                return k_scale, v_scale
        return None


class KvScaleAttention(nn.Module):
    """Reference self-attention entity consuming KV scale artifacts."""

    def __init__(
        self,
        dim: int,
        *,
        heads: int,
        head_dim: int | None = None,
        k_scale: float,
        v_scale: float,
        qmax: int = 127,
        dtype: str = "int8",
        causal: bool = False,
        dropout: float = 0.0,
        qkv_bias: bool = False,
        out_bias: bool = False,
        layer_path: str = "",
        contract: RuntimeQuantContract | Mapping[str, Any] | None = None,
        cache_block_size: int | None = None,
        hit_rate: float | None = None,
        preferred_kernel: str = "reference",
        fused_block_m: int = 64,
        fused_block_n: int = 64,
        fused_quant_block_size: int = 256,
        attention_fastpath: str = "eager",
        cuda_graph_warmup: int = 2,
        target_arch: str | None = None,
    ) -> None:
        super().__init__()
        if dim <= 0 or heads <= 0:
            raise ValueError("dim and heads must be positive")
        resolved_head_dim = int(dim // heads if head_dim is None else head_dim)
        if resolved_head_dim <= 0:
            raise ValueError("head_dim must be positive")
        if head_dim is None and dim % heads != 0:
            raise ValueError("dim must be divisible by heads when head_dim is omitted")
        if float(k_scale) <= 0.0 or float(v_scale) <= 0.0:
            raise ValueError("k_scale and v_scale must be positive")
        if int(qmax) <= 0:
            raise ValueError("qmax must be positive")
        normalized_kernel = str(preferred_kernel).strip().lower()
        if normalized_kernel not in {"reference", "auto", "tilelang"}:
            raise ValueError(
                "preferred_kernel must be one of 'reference', 'auto', 'tilelang'"
            )
        normalized_fastpath = str(attention_fastpath).strip().lower()
        if normalized_fastpath not in {"eager", "graph"}:
            raise ValueError(
                "attention_fastpath must be one of 'eager', 'graph'"
            )
        if int(cuda_graph_warmup) < 0:
            raise ValueError("cuda_graph_warmup must be non-negative")
        if (
            int(fused_block_m) <= 0
            or int(fused_block_n) <= 0
            or int(fused_quant_block_size) <= 0
        ):
            raise ValueError(
                "fused_block_m, fused_block_n, and fused_quant_block_size must be positive"
            )

        self.dim = int(dim)
        self.heads = int(heads)
        self.head_dim = resolved_head_dim
        self.inner_dim = self.heads * self.head_dim
        self.qmax = int(qmax)
        self.kv_cache_dtype = str(dtype).strip().lower()
        self.causal = bool(causal)
        self.dropout_p = float(dropout)
        self.layer_path = str(layer_path)
        self._contract = contract
        self._cache_block_size = (
            None if cache_block_size is None else int(cache_block_size)
        )
        self._hit_rate = None if hit_rate is None else float(hit_rate)
        self.preferred_kernel = normalized_kernel
        self._fused_block_m = int(fused_block_m)
        self._fused_block_n = int(fused_block_n)
        self._fused_quant_block_size = int(fused_quant_block_size)
        self.attention_fastpath = normalized_fastpath
        self.cuda_graph_warmup = int(cuda_graph_warmup)
        self.target_arch = (
            None if target_arch is None else str(target_arch).strip()
        )
        self._last_phase = "forward"
        self._fallback_reason: str | None = None
        self._selected_kernel = "torch_sdpa_kv_scale_reference"
        self._selected_kernels = ("torch_sdpa_kv_scale_reference",)
        self._selected_fastpath = "torch_sdpa_reference"
        self._cuda_fused_verified = False
        self._cuda_graph_state = "disabled"
        self._cuda_graph_reason: str | None = None
        self._cuda_graph_cache: dict[tuple[Any, ...], dict[str, Any]] = {}
        self._cuda_graph_last_state: dict[str, Any] | None = None

        self.qkv = nn.Linear(self.dim, 3 * self.inner_dim, bias=qkv_bias)
        self.out_proj = nn.Linear(self.inner_dim, self.dim, bias=out_bias)
        self.register_buffer(
            "k_scale",
            torch.tensor(float(k_scale), dtype=torch.float32),
        )
        self.register_buffer(
            "v_scale",
            torch.tensor(float(v_scale), dtype=torch.float32),
        )
        self.register_buffer(
            "attn_k_scale",
            torch.tensor(float(k_scale), dtype=torch.float32),
        )
        self.register_buffer(
            "attn_v_scale",
            torch.tensor(float(v_scale), dtype=torch.float32),
        )

    def _apply(
        self,
        fn: Callable[[torch.Tensor], torch.Tensor],
        recurse: bool = True,
    ) -> KvScaleAttention:
        module = super()._apply(fn, recurse=recurse)
        self._cuda_graph_cache.clear()
        self._cuda_graph_last_state = None
        self._cuda_graph_state = "disabled"
        self._cuda_graph_reason = "module device or dtype changed"
        return module

    @property
    def metadata(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if isinstance(self._contract, RuntimeQuantContract):
            payload[RUNTIME_QUANT_CONTRACT_KEY] = self._contract.to_dict()
        elif isinstance(self._contract, Mapping):
            payload[RUNTIME_QUANT_CONTRACT_KEY] = dict(self._contract)
        return payload

    def _consume_contract(self) -> ContractConsumeReport:
        if self._contract is None:
            return ContractConsumeReport(
                ok=False,
                contract=None,
                errors=("runtime_quant_contract_absent",),
                notes=(
                    "reference entity can run without a contract; a contract is "
                    "required to claim kernel/layout semantics",
                ),
            )
        report = consume_runtime_quant_contract(
            self._contract,
            require_kernels=False,
            require_shapes=False,
        )
        contract = report.contract
        if (
            report.ok
            and contract is not None
            and contract.kv_cache_dtype is not None
            and str(contract.kv_cache_dtype).strip().lower() != self.kv_cache_dtype
        ):
            return ContractConsumeReport(
                ok=False,
                contract=contract,
                errors=("kv_cache_dtype_mismatch",),
                notes=report.notes
                + (
                    f"contract kv_cache_dtype={contract.kv_cache_dtype}, "
                    f"entity dtype={self.kv_cache_dtype}",
                ),
            )
        return report

    def _reshape(self, tensor: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = tensor.shape
        return (
            tensor.reshape(batch, seq, self.heads, self.head_dim)
            .permute(0, 2, 1, 3)
            .contiguous()
        )

    def _merge_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch, _heads, seq, head_dim = tensor.shape
        return (
            tensor.permute(0, 2, 1, 3)
            .contiguous()
            .reshape(batch, seq, self.heads * head_dim)
        )

    def _split_qkv(
        self,
        qkv: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """将等宽 packed QKV projection 切分为 reference 路径所需的 views."""

        return torch.split(qkv, self.inner_dim, dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                "KvScaleAttention expects 3D input shaped [batch, seq, dim]"
            )
        if int(x.shape[-1]) != self.dim:
            raise ValueError(
                f"input last dim {int(x.shape[-1])} does not match dim {self.dim}"
            )
        use_graph = (
            self.preferred_kernel != "reference"
            and self.attention_fastpath == "graph"
        )
        if use_graph:
            graph_output = self._try_graph_forward(x)
            if graph_output is not None:
                return graph_output
        else:
            self._cuda_graph_state = "disabled"
            self._cuda_graph_reason = (
                "preferred_kernel is reference"
                if self.preferred_kernel == "reference"
                else "attention_fastpath is eager"
            )
        qkv = self.qkv(x)
        if self.preferred_kernel != "reference":
            fused_attn = self._try_fused_attention(qkv)
            if fused_attn is not None:
                return self.out_proj(fused_attn)
        else:
            self._selected_kernel = "torch_sdpa_kv_scale_reference"
            self._selected_kernels = ("torch_sdpa_kv_scale_reference",)
            self._selected_fastpath = "torch_sdpa_reference"
            self._cuda_fused_verified = False
            self._fallback_reason = None
        q, k, v = self._split_qkv(qkv)
        q = self._reshape(q)
        k_int8 = self._quantize_kv_int8(k, self.k_scale)
        v_int8 = self._quantize_kv_int8(v, self.v_scale)
        k = self._reshape(self._dequantize_kv(k_int8, self.k_scale, x.dtype))
        v = self._reshape(self._dequantize_kv(v_int8, self.v_scale, x.dtype))
        attn = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout_p,
            is_causal=self.causal,
        )
        return self.out_proj(self._merge_heads(attn))

    def _quantize_kv_int8(
        self,
        tensor: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        """reference 路径的 per-tensor INT8 量化."""

        return (
            torch.round(tensor / scale)
            .clamp(-self.qmax, self.qmax)
            .to(dtype=torch.int8)
        )

    def _dequantize_kv(
        self,
        quantized: torch.Tensor,
        scale: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return quantized.to(dtype=dtype) * scale

    def _quantize_kv(
        self,
        tensor: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        return self._dequantize_kv(
            self._quantize_kv_int8(tensor, scale),
            scale,
            tensor.dtype,
        )

    def _fused_unavailability_reason(self, qkv: torch.Tensor) -> str | None:
        """返回 fused 路径不可用的原因; 可用时返回 None."""

        if not qkv.is_cuda:
            return "cuda_unavailable"
        mismatch = target_arch_mismatch(self.target_arch, qkv)
        if mismatch is not None:
            return mismatch
        if torch.is_grad_enabled() and qkv.requires_grad:
            return "autograd_unsupported"
        if qkv.dtype != torch.float16:
            return "dtype_not_fp16"
        if self.dropout_p != 0.0:
            return "dropout_unsupported"
        if self.head_dim % 16 != 0:
            return "head_dim_not_multiple_of_16"
        from xqt.operator_opt.kernels.tilelang._common import (
            tilelang_runtime_unavailability_reason,
            tilelang_runtime_usable,
        )

        if not tilelang_runtime_usable():
            return tilelang_runtime_unavailability_reason() or "tilelang_unusable"
        return None

    def _run_packed_tilelang_attention(self, qkv: torch.Tensor) -> torch.Tensor:
        """Run packed-QKV quantize-layout and attention without report mutation."""

        from xqt.operator_opt.kernels.tilelang.kv_int8_attention import (
            fused_kv_int8_attention_packed_qkv_forward_tilelang,
            quantize_packed_qkv_int8_layout_tilelang,
        )

        k_int8, v_int8 = quantize_packed_qkv_int8_layout_tilelang(
            qkv,
            self.k_scale,
            self.v_scale,
            heads=self.heads,
            head_dim=self.head_dim,
            qmax=self.qmax,
            block_size=self._fused_quant_block_size,
        )
        return fused_kv_int8_attention_packed_qkv_forward_tilelang(
            qkv,
            k_int8,
            v_int8,
            self.k_scale,
            self.v_scale,
            heads=self.heads,
            head_dim=self.head_dim,
            causal=self.causal,
            block_m=self._fused_block_m,
            block_n=self._fused_block_n,
        )

    def _run_packed_tilelang_full_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run projection, packed TileLang attention, and output projection."""

        return self.out_proj(self._run_packed_tilelang_attention(self.qkv(x)))

    def _mark_packed_tilelang_success(self, *, graph: bool) -> None:
        selected_fastpath = (
            "packed_qkv_tilelang_cuda_graph"
            if graph
            else "packed_qkv_tilelang_eager"
        )
        if (
            self._selected_fastpath == selected_fastpath
            and self._cuda_fused_verified
            and self._fallback_reason is None
        ):
            return
        from xqt.operator_opt.kernels.tilelang.kv_int8_attention import (
            KV_INT8_PACKED_QKV_ATTENTION_KERNEL_NAME,
            KV_INT8_PACKED_QKV_QUANTIZE_LAYOUT_KERNEL_NAME,
        )

        self._selected_kernel = KV_INT8_PACKED_QKV_ATTENTION_KERNEL_NAME
        self._selected_kernels = (
            KV_INT8_PACKED_QKV_QUANTIZE_LAYOUT_KERNEL_NAME,
            KV_INT8_PACKED_QKV_ATTENTION_KERNEL_NAME,
        )
        self._selected_fastpath = selected_fastpath
        self._cuda_fused_verified = True
        self._fallback_reason = None

    def _mark_reference_fallback(self, reason: str) -> None:
        self._selected_kernel = "torch_sdpa_kv_scale_reference"
        self._selected_kernels = ("torch_sdpa_kv_scale_reference",)
        self._selected_fastpath = "torch_sdpa_reference_fallback"
        self._cuda_fused_verified = False
        self._fallback_reason = reason

    @staticmethod
    def _tensor_storage_signature(tensor: torch.Tensor | None) -> tuple[Any, ...]:
        if tensor is None:
            return (None,)
        return (
            int(tensor.data_ptr()),
            tuple(int(dim) for dim in tensor.shape),
            tuple(int(stride) for stride in tensor.stride()),
            str(tensor.dtype),
            str(tensor.device),
        )

    def _graph_cache_key(self, x: torch.Tensor) -> tuple[Any, ...]:
        from xqt.operator_opt.runtime import cuda_graph_tensor_signature

        device_index = x.device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        compute_capability = tuple(
            int(value) for value in torch.cuda.get_device_capability(device_index)
        )
        return (
            cuda_graph_tensor_signature(x),
            bool(self.causal),
            float(self.dropout_p),
            int(self.heads),
            int(self.head_dim),
            int(self.qmax),
            int(self._fused_quant_block_size),
            int(self._fused_block_m),
            int(self._fused_block_n),
            compute_capability,
            self._tensor_storage_signature(self.qkv.weight),
            self._tensor_storage_signature(self.qkv.bias),
            self._tensor_storage_signature(self.out_proj.weight),
            self._tensor_storage_signature(self.out_proj.bias),
            self._tensor_storage_signature(self.k_scale),
            self._tensor_storage_signature(self.v_scale),
        )

    def _capture_graph(self, x: torch.Tensor) -> dict[str, Any]:
        from xqt.operator_opt.runtime import capture_cuda_graph_with_static_state

        state = capture_cuda_graph_with_static_state(
            (x,),
            body=self._run_packed_tilelang_full_forward,
            warmup=self.cuda_graph_warmup,
        )
        state["kind"] = "kv_int8_packed_qkv_full_forward"
        state["output_storage"] = "graph_owned"
        return state

    def _annotate_graph_state(
        self,
        state: dict[str, Any],
        x: torch.Tensor,
        cache_key: tuple[Any, ...],
    ) -> None:
        from xqt.operator_opt.runtime import replay_cuda_graph_tensor_callable

        state.update(
            {
                "cache_key": cache_key,
                "fast_signature": self._graph_fast_signature(x),
                "replay_callable": replay_cuda_graph_tensor_callable,
            }
        )

    def _graph_fast_signature(
        self,
        x: torch.Tensor,
    ) -> tuple[Any, ...]:
        return (
            x.shape,
            x.stride(),
            x.dtype,
            x.device,
            self.causal,
            self.dropout_p,
            self.heads,
            self.head_dim,
            self.qmax,
            self._fused_quant_block_size,
            self._fused_block_m,
            self._fused_block_n,
            id(self.qkv.weight),
            id(self.qkv.bias),
            id(self.out_proj.weight),
            id(self.out_proj.bias),
            id(self.k_scale),
            id(self.v_scale),
        )

    def _graph_state_matches(
        self,
        state: Mapping[str, Any],
        x: torch.Tensor,
    ) -> bool:
        return state.get("fast_signature") == self._graph_fast_signature(x)

    def _replay_graph_state(
        self,
        state: dict[str, Any],
        x: torch.Tensor,
        *,
        graph_state: str,
    ) -> torch.Tensor | None:
        try:
            replay_callable = state["replay_callable"]
            output = replay_callable(state, (x,))
        except Exception as exc:
            cache_key = state.get("cache_key")
            if isinstance(cache_key, tuple):
                self._cuda_graph_cache.pop(cache_key, None)
            if self._cuda_graph_last_state is state:
                self._cuda_graph_last_state = None
            self._cuda_graph_state = "fallback_eager"
            self._cuda_graph_reason = (
                f"CUDA Graph replay failed: {type(exc).__name__}: {exc}"
            )
            return None
        self._cuda_graph_last_state = state
        self._cuda_graph_state = graph_state
        self._cuda_graph_reason = None
        self._mark_packed_tilelang_success(graph=True)
        return output

    def _try_graph_forward(self, x: torch.Tensor) -> torch.Tensor | None:
        last_state = self._cuda_graph_last_state
        if last_state is not None and self._graph_state_matches(last_state, x):
            return self._replay_graph_state(
                last_state,
                x,
                graph_state="replayed",
            )

        reason = self._fused_unavailability_reason(x)
        if reason is not None:
            self._cuda_graph_state = "fallback_eager"
            self._cuda_graph_reason = reason
            return None

        cache_key = self._graph_cache_key(x)
        state = self._cuda_graph_cache.get(cache_key)
        if state is None:
            try:
                state = self._capture_graph(x)
            except Exception as exc:
                self._cuda_graph_state = "fallback_eager"
                self._cuda_graph_reason = (
                    f"CUDA Graph capture failed: {type(exc).__name__}: {exc}"
                )
                return None
            self._annotate_graph_state(state, x, cache_key)
            self._cuda_graph_cache[cache_key] = state
            return self._replay_graph_state(
                state,
                x,
                graph_state="captured",
            )
        self._cuda_graph_last_state = state
        return self._replay_graph_state(
            state,
            x,
            graph_state="replayed",
        )

    def _try_fused_attention(
        self,
        qkv: torch.Tensor,
    ) -> torch.Tensor | None:
        """尝试 packed-QKV TileLang 路径; 不可用时记录并返回 None."""

        reason = self._fused_unavailability_reason(qkv)
        if reason is not None:
            self._mark_reference_fallback(reason)
            return None

        try:
            attn = self._run_packed_tilelang_attention(qkv)
        except (XQTBackendError, RuntimeError) as exc:
            self._mark_reference_fallback(
                f"tilelang_kernel_error:{type(exc).__name__}"
            )
            return None
        self._mark_packed_tilelang_success(graph=False)
        return attn

    def prefill(self, x: torch.Tensor) -> torch.Tensor:
        """Offline prefill phase wrapper (full sequence)."""
        self._last_phase = "prefill"
        return self.forward(x)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """Offline decode phase wrapper (single token)."""
        self._last_phase = "decode"
        return self.forward(x)

    def kv_cache_metadata(self) -> KvCacheMetadata:
        fused_note = (
            f"CUDA fused kernel verified on device: {self._selected_kernel}"
            if self._cuda_fused_verified
            else "CUDA fused-kernel verification pending"
        )
        return KvCacheMetadata(
            dtype=self.kv_cache_dtype,
            mode="per_tensor_scale",
            layer_scales=(
                (self.layer_path, float(self.k_scale.item()), float(self.v_scale.item())),
            ),
            cache_block_size=self._cache_block_size,
            hit_rate=self._hit_rate,
            notes=(
                "model-side metadata only; no cache storage or management",
                fused_note,
            ),
        )

    def report(self) -> dict[str, Any]:
        consume = self._consume_contract()
        # contract 缺失不覆盖已记录的 kernel fallback 原因, 也不把已成功
        # 运行的 fused 路径标成 fallback.
        if (
            not consume.ok
            and not self._cuda_fused_verified
            and self._fallback_reason is None
        ):
            self._fallback_reason = ";".join(consume.errors) or "contract_invalid"
        note = (
            f"torch SDPA reference entity; CUDA fused kernel verified on device: "
            f"{self._selected_kernel}"
            if self._cuda_fused_verified
            else "torch SDPA reference entity; CUDA fused-kernel verification pending"
        )
        return {
            "entity": "kv_scale_attention_reference",
            "layer_path": self.layer_path,
            "preferred_kernel": self.preferred_kernel,
            "attention_fastpath": self.attention_fastpath,
            "phase": self._last_phase,
            "target_arch": self.target_arch,
            "serving_cache": {
                "implemented": False,
                "reason": "model_side_entity_does_not_manage_serving_cache",
            },
            "projection_mode": "packed_qkv",
            "selected_kernel": self._selected_kernel,
            "selected_kernels": list(self._selected_kernels),
            "selected_fastpath": self._selected_fastpath,
            "cuda_fused_verified": self._cuda_fused_verified,
            "fallback_reason": self._fallback_reason,
            "cuda_graph": {
                "state": self._cuda_graph_state,
                "reason": self._cuda_graph_reason,
                "cache_size": len(self._cuda_graph_cache),
                "output_storage": (
                    "graph_owned"
                    if self._cuda_graph_state in {"captured", "replayed"}
                    else None
                ),
            },
            "contract_ok": consume.ok,
            "contract_errors": list(consume.errors),
            "contract_notes": list(consume.notes),
            "kv_cache_metadata": self.kv_cache_metadata().to_dict(),
            "notes": (note,),
        }

    @classmethod
    def from_artifacts(
        cls,
        artifacts: Mapping[str, Any],
        *,
        dim: int,
        heads: int,
        head_dim: int | None = None,
        layer_path: str = "",
        contract: RuntimeQuantContract | Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> KvScaleAttention:
        """Build one layer entity from canonical KV quant metadata."""

        metadata = KvCacheMetadata.from_artifacts(artifacts)
        scales = metadata.scales_for(layer_path)
        if scales is None:
            if metadata.layer_scales:
                first_path, k_scale, v_scale = metadata.layer_scales[0]
                layer_path = first_path
                scales = (k_scale, v_scale)
            else:
                raise XQTConfigError("no K/V scale layer available")
        k_scale, v_scale = scales
        return cls(
            dim,
            heads=heads,
            head_dim=head_dim,
            k_scale=k_scale,
            v_scale=v_scale,
            layer_path=layer_path,
            contract=contract,
            dtype=str(metadata.dtype),
            **kwargs,
        )


__all__ = [
    "KvCacheMetadata",
    "KvScaleAttention",
]
