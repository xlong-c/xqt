"""CUDA-graph greedy decode session for Llama-family models.

The session keeps a static KV cache and captures **one** CUDA graph that
serves every decode step, independent of the sequence length. Two device
scalars drive the graph at replay time:

- ``position``: the cache slot the next token is written to.
- ``valid_len``: the number of written cache slots the attention kernel reads.

``xqt.kernels.ops._impl.triton.decode_kernels`` provides the two decode
kernels that make this possible on a torch-only runtime:

- a single-pass GQA decode attention whose loop bound comes from
  ``valid_len`` (no padding correction, no per-length re-capture), with a
  SIMT default and an opt-in tensor-core variant (``attention_impl="tc"``), and
- a fused HF-Llama RoPE + KV-cache scatter.

Everything else in the decode body is a fused op: one AWQ W4A16 GEMV for the
fused q/k/v and gate/up projections, ``F.rms_norm`` for both normalizations,
and a Triton SwiGLU. The W4A16 LM head can be supplied through ``lm_head``.

Greedy decoding still needs every sampled token as the next input, but that
input only has to reach the device: the graph writes it in place. ``generate``
therefore replays a run of ``readback_chunk`` steps and reads the tokens back
once, which trades a small overrun past EOS for one fewer device-to-host
synchronization per token.

With ``int8_activations=True`` the decode path additionally rounds every
linear-layer activation through a per-row symmetric INT8 quantize/dequantize
step (norm outputs, the fused SwiGLU output, and the attention output). The
weights stay 4-bit, so the resulting route is the QuaRot-style
"INT8 activation + 4-bit weight" configuration (W4A8).

This is a runtime capability, not a serving engine: it is single-request,
greedy, and Llama-family only. It intentionally does not own scheduling,
batching, tokenization, or sampling.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from xqt.core.errors import XQTBackendError


@dataclass(frozen=True)
class GraphDecodeResult:
    """Tokens produced by one ``generate`` call plus runtime counters.

    ``decode_steps`` counts replayed steps, which can exceed
    ``generated_tokens`` by up to ``readback_chunk - 1`` when the run stops on
    EOS in the middle of a readback chunk.
    """

    token_ids: list[int]
    prefill_tokens: int
    decode_steps: int
    captured_graphs: int
    hit_eos: bool

    @property
    def generated_tokens(self) -> int:
        return len(self.token_ids)


def _validate_model(model: nn.Module) -> tuple[int, int, int, int, int]:
    """Return (layers, heads, kv_heads, head_dim, hidden) or raise."""

    config = getattr(model, "config", None)
    inner = getattr(model, "model", None)
    if config is None or inner is None or getattr(model, "lm_head", None) is None:
        raise XQTBackendError(
            "CudaGraphDecodeSession requires a Llama-family causal LM "
            "(model.model.* plus lm_head)"
        )
    required = (
        "embed_tokens",
        "rotary_emb",
        "layers",
        "norm",
    )
    missing = [name for name in required if not hasattr(inner, name)]
    if missing:
        raise XQTBackendError(
            f"CudaGraphDecodeSession requires model.model attributes {required}; "
            f"missing {missing}"
        )
    try:
        layers = int(config.num_hidden_layers)
        heads = int(config.num_attention_heads)
        kv_heads = int(config.num_key_value_heads)
        hidden = int(config.hidden_size)
    except AttributeError as exc:
        raise XQTBackendError(
            "CudaGraphDecodeSession requires num_hidden_layers, "
            "num_attention_heads, num_key_value_heads and hidden_size in config"
        ) from exc
    if hidden % heads != 0:
        raise XQTBackendError("hidden_size must be divisible by num_attention_heads")
    return layers, heads, kv_heads, hidden // heads, hidden


def _fuse_awq_projections(parts: Sequence[nn.Module]) -> "_FusedProjection | None":
    """Fuse AWQ decode views that share one input into a single GEMV.

    Returns ``None`` when the modules are not fusable (not materialized AWQ
    runtime views, mismatched input features, or missing decode bindings);
    callers then keep the unfused path.
    """

    from xqt.runtime.modules.awq_w4a16_linear import AWQW4A16Linear

    storages: list[Any] = []
    input_features: int | None = None
    for part in parts:
        storage = getattr(part, "storage", None)
        decode = getattr(part, "decode", None)
        if storage is None or not isinstance(decode, AWQW4A16Linear):
            return None
        features = int(getattr(part, "input_features", 0) or 0)
        if input_features is None:
            input_features = features
        elif features != input_features:
            return None
        storages.append(storage)
    try:
        decode = AWQW4A16Linear.from_signed_groupwise_storages(storages)
    except XQTBackendError:
        return None
    return _FusedProjection(parts, decode)


class _FusedProjection(nn.Module):
    """One decode GEMV for several projections that share the same input.

    Decode rows (M <= 8) go through a single fused AWQ W4A16 GEMV; prefill
    rows keep the per-projection path so no concatenated dense weight is
    materialized. Fusing does not change values: every output row is computed
    from its own weights and the same input vector.
    """

    def __init__(self, parts: Sequence[nn.Module], decode: nn.Module) -> None:
        super().__init__()
        self.input_features = int(getattr(parts[0], "input_features"))
        self.split_sizes = [int(getattr(part, "output_features")) for part in parts]
        self.parts = nn.ModuleList(list(parts))
        self.decode = decode
        # AWQ native decode views refuse inference in training mode.
        self.eval()

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        flat = inputs.reshape(-1, self.input_features)
        if (
            int(flat.shape[0]) <= 8
            and flat.is_cuda
            and flat.dtype in {torch.float16, torch.bfloat16}
        ):
            pieces = self.decode(flat).split(self.split_sizes, dim=-1)
            leading = inputs.shape[:-1]
            return tuple(
                piece.reshape(*leading, size)
                for piece, size in zip(pieces, self.split_sizes)
            )
        return tuple(part(inputs) for part in self.parts)

    def forward_w4a8(
        self, inputs: torch.Tensor, scale_a: torch.Tensor | float
    ) -> tuple[torch.Tensor, ...]:
        flat = inputs.reshape(-1, self.input_features)
        pieces = self.decode.forward_w4a8(flat, scale_a).split(self.split_sizes, dim=-1)
        leading = inputs.shape[:-1]
        return tuple(
            piece.reshape(*leading, size)
            for piece, size in zip(pieces, self.split_sizes)
        )


def _rms_norm(module: nn.Module, hidden: torch.Tensor, *, fused: bool) -> torch.Tensor:
    """Apply an RMSNorm module, optionally through the fused ATen op.

    HF's ``LlamaRMSNorm`` composes upcast/pow/mean/rsqrt/mul/downcast in about
    seven small kernels per call; across 84 calls per decode step that is the
    largest remaining shared cost. The fused op keeps the same math with one
    kernel but rounds differently, so callers can opt out for bit-faithful HF
    parity.
    """

    weight = getattr(module, "weight", None)
    eps = getattr(module, "variance_epsilon", None)
    if fused and isinstance(weight, torch.Tensor) and eps is not None:
        return F.rms_norm(hidden, (hidden.shape[-1],), weight, float(eps))
    return module(hidden)


def _supports_prequant(module: nn.Module) -> bool:
    """Return whether a projection exposes the optional INT8 prefill view."""

    probe = getattr(module, "supports_prequant", None)
    return bool(probe is not None and probe())


def _apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embeddings the same way HF LlamaAttention does (prefill)."""

    def rotate(x: torch.Tensor) -> torch.Tensor:
        half = x.shape[-1] // 2
        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat((-x2, x1), dim=-1)

    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return q * cos + rotate(q) * sin, k * cos + rotate(k) * sin


def _sdpa_prefill_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    """Default prefill attention: the historical SDPA call, unchanged."""

    return F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)


def _resolve_prefill_attention(
    impl: str, *, heads: int, kv_heads: int
) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    """Return the prefill attention callable for ``impl``.

    ``"sdpa"`` keeps the historical SDPA call. ``"triton"`` routes to the
    GQA-capable Triton forward kernel; ``"tilelang"`` routes to the TileLang
    kernel, which does not implement GQA, so mismatched head counts are
    rejected up front instead of silently producing wrong results. The Triton
    and TileLang kernels require contiguous BHSD inputs, so their wrappers
    materialize a contiguous copy when the caller passes a transposed view.
    """

    if impl == "sdpa":
        return _sdpa_prefill_attention
    if impl == "triton":
        from xqt.kernels.ops._impl.triton.attention import (
            fused_attention_forward_triton,
        )

        def _triton(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
            return fused_attention_forward_triton(
                q.contiguous(),
                k.contiguous(),
                v.contiguous(),
                causal=True,
            )

        return _triton
    if impl == "tilelang":
        if heads != kv_heads:
            raise XQTBackendError(
                "prefill_attention_impl='tilelang' does not support GQA: "
                f"the model has {heads} query heads and {kv_heads} key/value "
                "heads; use 'sdpa' or 'triton'"
            )
        from xqt.kernels.ops._impl.tilelang.attention import (
            fused_attention_forward_tilelang,
        )

        def _tilelang(
            q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
        ) -> torch.Tensor:
            return fused_attention_forward_tilelang(
                q.contiguous(),
                k.contiguous(),
                v.contiguous(),
                causal=True,
            )

        return _tilelang
    raise ValueError("prefill_attention_impl must be 'sdpa', 'triton' or 'tilelang'")


class CudaGraphDecodeSession:
    """Greedy single-request decode behind one length-agnostic CUDA graph.

    ``max_cache_len`` fixes the static cache and therefore the graph shapes;
    the number of valid slots is a device tensor updated before every replay.
    Capture happens lazily on the first decode step, or eagerly through
    :meth:`capture`.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        max_cache_len: int = 4096,
        fuse_projections: bool = True,
        fuse_norms: bool = True,
        lm_head: nn.Module | None = None,
        attention_splits: int = 16,
        attention_impl: str = "simt",
        prefill_attention_impl: str = "sdpa",
        int8_activations: bool = False,
        readback_chunk: int = 16,
        int8_prefill: bool = False,
        min_int8_prefill_rows: int = 256,
        kv_quant: str = "none",
    ) -> None:
        if not torch.cuda.is_available():
            raise XQTBackendError("CudaGraphDecodeSession requires CUDA")
        if max_cache_len < 1:
            raise ValueError("max_cache_len must be positive")
        if kv_quant not in {"none", "int8", "int4"}:
            raise ValueError("kv_quant must be 'none', 'int8', or 'int4'")
        try:
            from xqt.kernels.ops._impl.triton.decode_kernels import (
                decode_attention_forward_triton,
                decode_attention_forward_triton_tc,
                hadamard_matrix,
                pack_write_kv_int4_triton,
                quantize_row_true_int8_triton,
                rmsnorm_int8_triton,
                rmsnorm_true_int8_triton,
                rope_decode_triton,
                rope_write_qkv_int8_triton,
                rope_write_qkv_triton,
                swiglu_int8_triton,
                swiglu_true_int8_triton,
            )
            from xqt.kernels.ops._impl.triton.gemm import (
                quantize_int8_rowwise_triton,
            )
            from xqt.kernels.ops._impl.triton.pointwise import fused_swiglu_triton
        except Exception as exc:
            raise XQTBackendError(
                "CudaGraphDecodeSession requires the Triton decode and INT8 "
                "quantization kernels (triton must be importable)"
            ) from exc

        layers, heads, kv_heads, head_dim, hidden = _validate_model(model)
        self.model = model
        self.max_cache_len = int(max_cache_len)
        self.num_layers = layers
        self.num_q_heads = heads
        self.num_kv_heads = kv_heads
        self.head_dim = head_dim
        self.hidden_size = hidden
        self._kv_quant = kv_quant
        if kv_quant != "none" and attention_impl != "tc":
            # Quantized KV cache defaults to TC kernel for optimal throughput
            attention_impl = "tc"
        if attention_impl not in {"simt", "tc"}:
            raise ValueError("attention_impl must be 'simt' or 'tc'")
        self._attention_impl = attention_impl
        self._decode_attention = (
            decode_attention_forward_triton_tc
            if attention_impl == "tc"
            else decode_attention_forward_triton
        )
        if prefill_attention_impl not in {"sdpa", "triton", "tilelang"}:
            raise ValueError(
                "prefill_attention_impl must be 'sdpa', 'triton' or 'tilelang'"
            )
        self._prefill_attention_impl = prefill_attention_impl
        self._prefill_attention = _resolve_prefill_attention(
            prefill_attention_impl, heads=heads, kv_heads=kv_heads
        )
        self._rope_write_qkv = rope_write_qkv_triton
        self._rope_write_qkv_int8 = rope_write_qkv_int8_triton
        self._rope_decode = rope_decode_triton
        self._pack_write_kv_int4 = pack_write_kv_int4_triton
        self._swiglu = fused_swiglu_triton
        self._rmsnorm_int8 = rmsnorm_int8_triton
        self._swiglu_int8 = swiglu_int8_triton
        self._rmsnorm_true_int8 = rmsnorm_true_int8_triton
        self._swiglu_true_int8 = swiglu_true_int8_triton
        self._quantize_row_true_int8 = quantize_row_true_int8_triton
        self._int8_activations = bool(int8_activations)
        self._attention_splits = int(attention_splits)
        if self._attention_splits < 1:
            raise ValueError("attention_splits must be positive")
        self._readback_chunk = int(readback_chunk)
        if self._readback_chunk < 1:
            raise ValueError("readback_chunk must be positive")
        self._int8_prefill = bool(int8_prefill)
        self._min_int8_prefill_rows = int(min_int8_prefill_rows)
        if self._min_int8_prefill_rows < 1:
            raise ValueError("min_int8_prefill_rows must be positive")
        self._quantize_int8_rowwise = quantize_int8_rowwise_triton

        dtype = next(model.parameters()).dtype
        self.hadamard_h: torch.Tensor | None = None
        if kv_quant == "int4":
            self.hadamard_h = hadamard_matrix(head_dim, device="cuda", dtype=dtype)
            shape = (1, kv_heads, self.max_cache_len, head_dim // 2)
            cache_dtype = torch.int8
        elif kv_quant == "int8":
            shape = (1, kv_heads, self.max_cache_len, head_dim)
            cache_dtype = torch.int8
        else:
            shape = (1, kv_heads, self.max_cache_len, head_dim)
            cache_dtype = dtype

        self.k_cache = [
            torch.zeros(shape, dtype=cache_dtype, device="cuda") for _ in range(layers)
        ]
        self.v_cache = [
            torch.zeros(shape, dtype=cache_dtype, device="cuda") for _ in range(layers)
        ]
        if kv_quant in {"int8", "int4"}:
            scale_shape = (1, kv_heads, self.max_cache_len)
            self.k_scale = [
                torch.zeros(scale_shape, dtype=torch.float32, device="cuda")
                for _ in range(layers)
            ]
            self.v_scale = [
                torch.zeros(scale_shape, dtype=torch.float32, device="cuda")
                for _ in range(layers)
            ]
        else:
            self.k_scale = []
            self.v_scale = []

        if kv_quant == "int4":
            self._q_rope_buf = torch.zeros(
                (1, heads, 1, head_dim), dtype=dtype, device="cuda"
            )
            self._k_rope_buf = torch.zeros(
                (1, kv_heads, 1, head_dim), dtype=dtype, device="cuda"
            )
        self.token = torch.zeros(1, 1, dtype=torch.long, device="cuda")
        self.position = torch.zeros(1, dtype=torch.long, device="cuda")
        self.valid_len = torch.zeros(1, dtype=torch.int32, device="cuda")
        self.rope_cos, self.rope_sin = self._build_rope_tables(dtype)

        self._graphs: list[torch.cuda.CUDAGraph] = []
        self._warmed = False
        self.length = 0
        self.decode_steps = 0
        self._residual_bindings: dict[
            tuple[int, torch.dtype], Callable[[torch.Tensor, torch.Tensor], None]
        ] = {}
        self._fused_qkv, self._fused_gate_up = self._build_fused_projections(
            model, fuse_projections
        )
        self._lm_head_override = lm_head
        self._fuse_norms = bool(fuse_norms)
        if self._int8_activations:
            cfg = getattr(model, "config", None)
            intermediate_size = getattr(cfg, "intermediate_size", self.hidden_size * 3)
            self._norm_int8_buf = torch.zeros(
                (1, self.hidden_size), dtype=torch.int8, device="cuda"
            )
            self._norm_scale_buf = torch.zeros((1,), dtype=torch.float32, device="cuda")
            self._mlp_norm_int8_buf = torch.zeros(
                (1, self.hidden_size), dtype=torch.int8, device="cuda"
            )
            self._mlp_norm_scale_buf = torch.zeros(
                (1,), dtype=torch.float32, device="cuda"
            )
            self._swiglu_int8_buf = torch.zeros(
                (1, intermediate_size), dtype=torch.int8, device="cuda"
            )
            self._swiglu_scale_buf = torch.zeros(
                (1,), dtype=torch.float32, device="cuda"
            )
            self._attn_int8_buf = torch.zeros(
                (1, self.hidden_size), dtype=torch.int8, device="cuda"
            )
            self._attn_scale_buf = torch.zeros((1,), dtype=torch.float32, device="cuda")

    # ------------------------------------------------------------ rope tables
    def _build_rope_tables(
        self, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Precompute the HF rotary cos/sin tables for every cache position."""

        model = self.model
        device = next(model.parameters()).device
        probe = torch.zeros(1, 1, self.hidden_size, dtype=dtype, device=device)
        positions = torch.arange(self.max_cache_len, device=device).unsqueeze(0)
        cos, sin = model.model.rotary_emb(probe, positions)
        cos = cos.reshape(self.max_cache_len, self.head_dim).contiguous()
        sin = sin.reshape(self.max_cache_len, self.head_dim).contiguous()
        return cos, sin

    def _logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """Project ``[1, 1, hidden]`` to logits, using the override head if built."""

        if self._lm_head_override is None:
            return self.model.lm_head(hidden)
        out = self._lm_head_override(hidden.reshape(1, self.hidden_size))
        return out.reshape(1, 1, -1)

    def _build_fused_projections(
        self, model: nn.Module, enabled: bool
    ) -> tuple[list["_FusedProjection | None"], list["_FusedProjection | None"]]:
        layers = list(model.model.layers)
        if not enabled:
            return [None] * len(layers), [None] * len(layers)
        qkv: list["_FusedProjection | None"] = []
        gate_up: list["_FusedProjection | None"] = []
        for layer in layers:
            attention = layer.self_attn
            qkv.append(
                _fuse_awq_projections(
                    [attention.q_proj, attention.k_proj, attention.v_proj]
                )
            )
            mlp = layer.mlp
            gate_up.append(_fuse_awq_projections([mlp.gate_proj, mlp.up_proj]))
        return qkv, gate_up

    @property
    def fused_projection_count(self) -> int:
        return sum(1 for item in self._fused_qkv if item is not None) + sum(
            1 for item in self._fused_gate_up if item is not None
        )

    @property
    def captured_graphs(self) -> int:
        return len(self._graphs)

    # ---------------------------------------------------------------- prefill
    @torch.inference_mode()
    def prefill(self, input_ids: torch.Tensor) -> int:
        """Fill the cache with ``input_ids`` and return the first token."""

        if input_ids.dim() != 2 or int(input_ids.shape[0]) != 1:
            raise ValueError("input_ids must be [1, seq_len]")
        seq_len = int(input_ids.shape[1])
        if seq_len < 1 or seq_len > self.max_cache_len:
            raise ValueError(
                f"prompt length must be in [1, {self.max_cache_len}], got {seq_len}"
            )
        model = self.model
        hidden = model.model.embed_tokens(input_ids)
        position_ids = torch.arange(seq_len, device="cuda").unsqueeze(0)
        cos, sin = model.model.rotary_emb(hidden, position_ids)
        for index, layer in enumerate(model.model.layers):
            residual = hidden
            normed = _rms_norm(layer.input_layernorm, hidden, fused=self._fuse_norms)
            attention = layer.self_attn
            q, k, v = self._project_qkv(index, layer, normed, seq_len)
            q, k = _apply_rope(q, k, cos, sin)
            if self._kv_quant == "int4":
                assert self.hadamard_h is not None
                q_rot = torch.matmul(q, self.hadamard_h)
                k_rot = torch.matmul(k, self.hadamard_h)
                v_rot = torch.matmul(v, self.hadamard_h)

                k_fp = k_rot.to(torch.float32)
                k_scale = (k_fp.abs().amax(dim=-1) / 7.0).clamp_min(1e-8)
                q_k = (k_fp / k_scale.unsqueeze(-1)).round().clamp(-8, 7).to(torch.int8)
                half = self.head_dim // 2
                k1, k2 = q_k[..., :half], q_k[..., half:]
                k_packed = (k1 & 0x0F) | ((k2 & 0x0F) << 4)
                self.k_cache[index][:, :, :seq_len] = k_packed
                self.k_scale[index][:, :, :seq_len] = k_scale

                v_fp = v_rot.to(torch.float32)
                v_scale = (v_fp.abs().amax(dim=-1) / 7.0).clamp_min(1e-8)
                q_v = (v_fp / v_scale.unsqueeze(-1)).round().clamp(-8, 7).to(torch.int8)
                v1, v2 = q_v[..., :half], q_v[..., half:]
                v_packed = (v1 & 0x0F) | ((v2 & 0x0F) << 4)
                self.v_cache[index][:, :, :seq_len] = v_packed
                self.v_scale[index][:, :, :seq_len] = v_scale

                out = self._prefill_attention(q_rot, k_rot, v_rot)
                out = torch.matmul(out, self.hadamard_h)
            elif self._kv_quant == "int8":
                k_fp = k.to(torch.float32)
                k_scale = (k_fp.abs().amax(dim=-1) / 127.0).clamp_min(1e-8)
                k_int8 = (
                    (k_fp / k_scale.unsqueeze(-1))
                    .round()
                    .clamp(-128, 127)
                    .to(torch.int8)
                )
                self.k_cache[index][:, :, :seq_len] = k_int8
                self.k_scale[index][:, :, :seq_len] = k_scale

                v_fp = v.to(torch.float32)
                v_scale = (v_fp.abs().amax(dim=-1) / 127.0).clamp_min(1e-8)
                v_int8 = (
                    (v_fp / v_scale.unsqueeze(-1))
                    .round()
                    .clamp(-128, 127)
                    .to(torch.int8)
                )
                self.v_cache[index][:, :, :seq_len] = v_int8
                self.v_scale[index][:, :, :seq_len] = v_scale
                out = self._prefill_attention(q, k, v)
            else:
                self.k_cache[index][:, :, :seq_len] = k
                self.v_cache[index][:, :, :seq_len] = v
                out = self._prefill_attention(q, k, v)

            out = out.transpose(1, 2).reshape(1, seq_len, -1)
            hidden = residual + attention.o_proj(out)
            residual = hidden
            if self._int8_prefill:
                hidden = self._prefill_mlp_int8(index, layer, hidden, residual)
            else:
                # Non-int8 prefill keeps the HF-parity MLP: the fused SwiGLU
                # rounds differently and would break exact greedy reproduction
                # (SpecDecodeSession's correctness gate) and the documented
                # per-token HF equivalence of the default prefill path.
                hidden = residual + self._mlp_forward(
                    index,
                    layer,
                    layer.post_attention_layernorm(hidden),
                    fused_ops=False,
                )
        logits = self._logits(
            _rms_norm(model.model.norm, hidden, fused=self._fuse_norms)[:, -1:, :]
        )
        self.token.copy_(logits.argmax(-1))
        self.length = seq_len
        return int(self.token.item())

    def _prefill_mlp_int8(
        self,
        index: int,
        layer: nn.Module,
        hidden: torch.Tensor,
        residual: torch.Tensor,
    ) -> torch.Tensor:
        """Prefill MLP with an MLP-only INT8 W8A8 fast path.

        Attention and q/k/v stay bf16 exactly as in :meth:`prefill`; only the
        gated MLP is routed through the INT8 Tensor Core GEMM when the row count
        clears ``min_int8_prefill_rows`` and the projections expose the AWQ
        pre-quantized view. Every other case falls back to the existing bf16
        fused MLP, so the returned tensor is ``residual + mlp(normed)``.
        """

        mlp = layer.mlp
        normed = _rms_norm(
            layer.post_attention_layernorm, hidden, fused=self._fuse_norms
        )
        rows = int(normed.numel() // self.hidden_size)
        if (
            rows >= self._min_int8_prefill_rows
            and _supports_prequant(mlp.gate_proj)
            and _supports_prequant(mlp.up_proj)
            and _supports_prequant(mlp.down_proj)
        ):
            normed_2d = normed.reshape(rows, self.hidden_size)
            qactivation, activation_scale = self._quantize_int8_rowwise(normed_2d)
            gate = mlp.gate_proj.forward_prequant(qactivation, activation_scale)
            up = mlp.up_proj.forward_prequant(qactivation, activation_scale)
            activation = self._swiglu(gate, up)
            qactivation, activation_scale = self._quantize_int8_rowwise(activation)
            down = mlp.down_proj.forward_prequant(qactivation, activation_scale)
            return residual + down.reshape(normed.shape)
        hidden_mlp = self._mlp_forward(
            index, layer, normed, fused_ops=True, residual=None
        )
        return residual + hidden_mlp

    # ------------------------------------------------------------ graph steps
    def _decode_body(self) -> None:
        """One decode step; ``self.token`` advances in place.

        The caller owns ``position``/``valid_len`` (``_fill_step_state``): both
        eager callers and the captured graph read them from device tensors, and
        ``decode_batch`` must set them *outside* the replayed graph so every
        replay can advance to a fresh slot without re-capture.
        """

        model = self.model
        hidden = model.model.embed_tokens(self.token)
        quantize = self._int8_activations
        for index, layer in enumerate(model.model.layers):
            residual = hidden
            attention = layer.self_attn
            if quantize:
                eps = float(layer.input_layernorm.variance_epsilon)
                self._rmsnorm_true_int8(
                    hidden,
                    layer.input_layernorm.weight,
                    eps=eps,
                    out_q=self._norm_int8_buf,
                    out_scale=self._norm_scale_buf,
                )
                q, k, v = self._project_qkv_w4a8(
                    index, layer, self._norm_int8_buf, self._norm_scale_buf
                )
            else:
                normed = self._norm(layer.input_layernorm, hidden, quantize)
                q, k, v = self._project_qkv(index, layer, normed, 1)

            if self._kv_quant == "int4":
                assert self.hadamard_h is not None
                self._rope_decode(
                    q,
                    k,
                    self.rope_cos,
                    self.rope_sin,
                    self._q_rope_buf,
                    self._k_rope_buf,
                    self.position,
                )
                q_rot = torch.matmul(self._q_rope_buf, self.hadamard_h)
                k_rot = torch.matmul(self._k_rope_buf, self.hadamard_h)
                v_rot = torch.matmul(v, self.hadamard_h)
                self._pack_write_kv_int4(
                    k_rot,
                    v_rot,
                    self.k_cache[index],
                    self.v_cache[index],
                    self.k_scale[index],
                    self.v_scale[index],
                    self.position,
                )
                out = self._decode_attention(
                    q_rot,
                    self.k_cache[index],
                    self.v_cache[index],
                    self.valid_len,
                    k_scale=self.k_scale[index],
                    v_scale=self.v_scale[index],
                    splits=self._attention_splits,
                    quantize_out=False,
                )
                out = torch.matmul(out, self.hadamard_h)
            elif self._kv_quant == "int8":
                self._rope_write_qkv_int8(
                    q,
                    k,
                    v,
                    self.rope_cos,
                    self.rope_sin,
                    q,
                    self.k_cache[index],
                    self.v_cache[index],
                    self.k_scale[index],
                    self.v_scale[index],
                    self.position,
                )
                out = self._decode_attention(
                    q,
                    self.k_cache[index],
                    self.v_cache[index],
                    self.valid_len,
                    k_scale=self.k_scale[index],
                    v_scale=self.v_scale[index],
                    splits=self._attention_splits,
                    quantize_out=False,
                )
            else:
                self._rope_write_qkv(
                    q,
                    k,
                    v,
                    self.rope_cos,
                    self.rope_sin,
                    q,
                    self.k_cache[index],
                    self.v_cache[index],
                    self.position,
                )
                out = self._decode_attention(
                    q,
                    self.k_cache[index],
                    self.v_cache[index],
                    self.valid_len,
                    splits=self._attention_splits,
                    quantize_out=False,
                )
            out = out.transpose(1, 2).reshape(1, 1, -1)

            if quantize:
                self._quantize_row_true_int8(
                    out,
                    out_q=self._attn_int8_buf,
                    out_scale=self._attn_scale_buf,
                )
                hidden = self._residual_project_w4a8(
                    attention.o_proj,
                    self._attn_int8_buf,
                    self._attn_scale_buf,
                    residual,
                )
                residual = hidden
                eps_mlp = float(layer.post_attention_layernorm.variance_epsilon)
                self._rmsnorm_true_int8(
                    hidden,
                    layer.post_attention_layernorm.weight,
                    eps=eps_mlp,
                    out_q=self._mlp_norm_int8_buf,
                    out_scale=self._mlp_norm_scale_buf,
                )
                hidden = self._mlp_forward_w4a8(
                    index,
                    layer,
                    self._mlp_norm_int8_buf,
                    self._mlp_norm_scale_buf,
                    residual=residual,
                )
            else:
                hidden = self._residual_project(attention.o_proj, out, residual)
                residual = hidden
                hidden = self._mlp_forward(
                    index,
                    layer,
                    self._norm(layer.post_attention_layernorm, hidden, quantize),
                    fused_ops=True,
                    quantize=quantize,
                    residual=residual,
                )
        logits = self._logits(
            _rms_norm(model.model.norm, hidden, fused=self._fuse_norms)
        )
        self.token.copy_(logits.argmax(-1))

    def _linear_w4a8_or_fallback(
        self, module: nn.Module, inputs: torch.Tensor, scale_a: torch.Tensor
    ) -> torch.Tensor:
        forward_w4a8 = getattr(module, "forward_w4a8", None)
        if forward_w4a8 is not None:
            return forward_w4a8(inputs, scale_a)
        weight = getattr(module, "weight", None)
        dtype = weight.dtype if weight is not None else torch.bfloat16
        dequant = inputs.to(dtype) * scale_a.to(dtype)
        return module(dequant)

    def _project_qkv_w4a8(
        self,
        index: int,
        layer: nn.Module,
        inputs: torch.Tensor,
        scale_a: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        fused = self._fused_qkv[index]
        if fused is not None:
            forward_w4a8 = getattr(fused, "forward_w4a8", None)
            if forward_w4a8 is not None:
                q, k, v = forward_w4a8(inputs, scale_a)
            else:
                dequant = inputs.to(torch.bfloat16) * scale_a.to(torch.bfloat16)
                q, k, v = fused(dequant)
        else:
            attention = layer.self_attn
            q = self._linear_w4a8_or_fallback(attention.q_proj, inputs, scale_a)
            k = self._linear_w4a8_or_fallback(attention.k_proj, inputs, scale_a)
            v = self._linear_w4a8_or_fallback(attention.v_proj, inputs, scale_a)
        q = q.view(1, 1, self.num_q_heads, self.head_dim).transpose(1, 2)
        k = k.view(1, 1, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(1, 1, self.num_kv_heads, self.head_dim).transpose(1, 2)
        return q, k, v

    def _mlp_forward_w4a8(
        self,
        index: int,
        layer: nn.Module,
        inputs: torch.Tensor,
        scale_a: torch.Tensor,
        *,
        residual: torch.Tensor,
    ) -> torch.Tensor:
        mlp = layer.mlp
        fused = self._fused_gate_up[index]
        if fused is not None:
            forward_w4a8 = getattr(fused, "forward_w4a8", None)
            if forward_w4a8 is not None:
                gate, up = forward_w4a8(inputs, scale_a)
            else:
                dequant = inputs.to(torch.bfloat16) * scale_a.to(torch.bfloat16)
                gate, up = fused(dequant)
        else:
            gate = self._linear_w4a8_or_fallback(mlp.gate_proj, inputs, scale_a)
            up = self._linear_w4a8_or_fallback(mlp.up_proj, inputs, scale_a)
        self._swiglu_true_int8(
            gate, up, out_q=self._swiglu_int8_buf, out_scale=self._swiglu_scale_buf
        )
        return self._residual_project_w4a8(
            mlp.down_proj, self._swiglu_int8_buf, self._swiglu_scale_buf, residual
        )

    def _residual_project_w4a8(
        self,
        module: nn.Module,
        inputs: torch.Tensor,
        scale_a: torch.Tensor,
        residual: torch.Tensor,
    ) -> torch.Tensor:
        binder = getattr(module, "bind_residual_w4a8", None)
        if binder is None:
            out = self._linear_w4a8_or_fallback(module, inputs, scale_a)
            return residual + out
        key = (id(module), "w4a8", residual.dtype)
        runner = self._residual_bindings.get(key)
        if runner is None:
            features = int(getattr(module, "input_features"))
            flat = inputs.reshape(-1, features)
            runner = binder(rows=int(flat.shape[0]), dtype=residual.dtype)
            self._residual_bindings[key] = runner
        features = int(getattr(module, "input_features"))
        runner(
            inputs.reshape(-1, features),
            scale_a,
            residual.reshape(-1, self.hidden_size),
        )
        return residual

    def _norm(
        self, module: nn.Module, hidden: torch.Tensor, quantize: bool
    ) -> torch.Tensor:
        """Apply a normalization, optionally with an INT8 activation round-trip."""

        if not quantize:
            return _rms_norm(module, hidden, fused=self._fuse_norms)
        weight = getattr(module, "weight", None)
        eps = getattr(module, "variance_epsilon", None)
        if not isinstance(weight, torch.Tensor) or eps is None:
            raise XQTBackendError(
                "INT8 activation mode requires RMSNorm modules with weight and eps"
            )
        return self._rmsnorm_int8(hidden, weight, eps=float(eps))

    def _project_qkv(
        self,
        index: int,
        layer: nn.Module,
        inputs: torch.Tensor,
        tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project hidden states to q/k/v, fused into one GEMV when possible."""

        fused = self._fused_qkv[index]
        if fused is not None:
            q, k, v = fused(inputs)
        else:
            attention = layer.self_attn
            q = attention.q_proj(inputs)
            k = attention.k_proj(inputs)
            v = attention.v_proj(inputs)
        q = q.view(1, tokens, self.num_q_heads, self.head_dim).transpose(1, 2)
        k = k.view(1, tokens, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(1, tokens, self.num_kv_heads, self.head_dim).transpose(1, 2)
        return q, k, v

    def _residual_project(
        self, module: nn.Module, inputs: torch.Tensor, residual: torch.Tensor
    ) -> torch.Tensor:
        """Fold ``residual + module(inputs)`` into the projection epilogue.

        AWQ decode modules expose :meth:`bind_residual`, whose output-dtype
        epilogue add matches the unfused expression exactly; anything else keeps
        the separate elementwise add. The residual row is updated in place, so
        the returned tensor is ``residual`` itself.
        """

        binder = getattr(module, "bind_residual", None)
        if binder is None:
            return residual + module(inputs)
        key = (id(module), inputs.dtype)
        runner = self._residual_bindings.get(key)
        if runner is None:
            features = int(getattr(module, "input_features"))
            flat = inputs.reshape(-1, features)
            runner = binder(rows=int(flat.shape[0]), dtype=flat.dtype)
            self._residual_bindings[key] = runner
        features = int(getattr(module, "input_features"))
        runner(inputs.reshape(-1, features), residual.reshape(-1, self.hidden_size))
        return residual

    def _mlp_forward(
        self,
        index: int,
        layer: nn.Module,
        inputs: torch.Tensor,
        *,
        fused_ops: bool,
        quantize: bool = False,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the gated MLP, fusing gate/up and the SwiGLU when possible."""

        mlp = layer.mlp
        fused = self._fused_gate_up[index]
        if fused is not None:
            gate, up = fused(inputs)
        else:
            gate = mlp.gate_proj(inputs)
            up = mlp.up_proj(inputs)
        if fused_ops:
            if quantize:
                activation = self._swiglu_int8(gate, up)
            else:
                activation = self._swiglu(gate, up)
            if residual is None:
                return mlp.down_proj(activation)
            return self._residual_project(mlp.down_proj, activation, residual)
        return mlp.down_proj(mlp.act_fn(gate) * up)

    def _fill_step_state(self, length: int) -> None:
        self.position.fill_(length - 1)
        self.valid_len.fill_(length)

    def _warm(self) -> None:
        """Run one eager decode step so the first capture sees bound kernels.

        Quantized runtime modules build their kernels/runners on first use, and
        that setup is not always capture-safe. The eager step is a real decode
        step, so it overwrites one KV-cache slot; that slot is snapshotted and
        restored because a caller may capture after a prefill.
        """

        if self._warmed:
            return
        saved_token = self.token.clone()
        slot = min(max(self.length, 1), self.max_cache_len) - 1
        saved_k = [cache[:, :, slot].clone() for cache in self.k_cache]
        saved_v = [cache[:, :, slot].clone() for cache in self.v_cache]
        saved_ks = (
            [scale[:, :, slot].clone() for scale in self.k_scale]
            if self._kv_quant in {"int8", "int4"}
            else []
        )
        saved_vs = (
            [scale[:, :, slot].clone() for scale in self.v_scale]
            if self._kv_quant in {"int8", "int4"}
            else []
        )
        self._fill_step_state(slot + 1)
        with torch.inference_mode():
            self._decode_body()
            torch.cuda.synchronize()
        for index in range(self.num_layers):
            self.k_cache[index][:, :, slot] = saved_k[index]
            self.v_cache[index][:, :, slot] = saved_v[index]
            if self._kv_quant in {"int8", "int4"}:
                self.k_scale[index][:, :, slot] = saved_ks[index]
                self.v_scale[index][:, :, slot] = saved_vs[index]
        self.token.copy_(saved_token)
        self._warmed = True

    def capture(self) -> None:
        """Capture (or reuse) the single decode graph."""

        if self._graphs:
            return
        self._warm()
        saved_token = self.token.clone()
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.inference_mode():
                with torch.cuda.graph(graph):
                    self._decode_body()
                torch.cuda.synchronize()
        finally:
            # Capture advanced the token buffer once; restore it so the first
            # replay starts from the same token.
            self.token.copy_(saved_token)
        self._graphs.append(graph)

    def capture_until(self, max_length: int) -> int:
        """Pre-capture the decode graph; the parameter is a compatibility bound."""

        if int(max_length) > self.max_cache_len:
            raise ValueError(
                f"max_length {max_length} exceeds max_cache_len {self.max_cache_len}"
            )
        self.capture()
        return self.captured_graphs

    @torch.inference_mode()
    def decode_batch(self, steps: int) -> torch.Tensor:
        """Advance ``steps`` tokens and return them as one device tensor.

        Greedy decoding needs each sampled token to pick the next input, but
        that input only has to reach the *device*: the graph already writes it
        in place. Replaying a batch and reading the tokens back once therefore
        removes one device-to-host synchronization per token at the cost of a
        small readback overrun past EOS. Callers that stop on EOS truncate the
        returned tokens; ``decode_steps`` still counts every replay.
        """

        if steps < 1:
            raise ValueError("decode_batch steps must be positive")
        if self.length + steps > self.max_cache_len:
            raise ValueError(f"decode exceeds max_cache_len {self.max_cache_len}")
        if not self._graphs:
            self.capture()
        tokens = torch.empty(steps, dtype=torch.long, device=self.token.device)
        graph = self._graphs[0]
        for index in range(steps):
            length = self.length + 1
            self._fill_step_state(length)
            graph.replay()
            self.length = length
            self.decode_steps += 1
            tokens[index].copy_(self.token.reshape(()))
        return tokens

    @torch.inference_mode()
    def decode_step(self) -> int:
        """Advance exactly one token and return it."""

        return int(self.decode_batch(1)[0].item())

    # --------------------------------------------------------------- generate
    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        *,
        max_new_tokens: int,
        eos_token_ids: Iterable[int] = (),
        warm_buckets: bool = False,
    ) -> GraphDecodeResult:
        """Greedy decode; ``token_ids`` excludes the prompt tokens."""

        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        eos = {int(token) for token in eos_token_ids}
        prompt_tokens = int(input_ids.shape[1])
        if prompt_tokens + max_new_tokens > self.max_cache_len:
            raise ValueError(
                "prompt plus max_new_tokens exceeds max_cache_len "
                f"({prompt_tokens} + {max_new_tokens} > {self.max_cache_len})"
            )
        if warm_buckets or not self._graphs:
            self.capture()
        steps_before = self.decode_steps
        first = self.prefill(input_ids)
        tokens: list[int] = [first]
        hit_eos = first in eos
        while len(tokens) < max_new_tokens and not hit_eos:
            take = min(self._readback_chunk, max_new_tokens - len(tokens))
            for token in self.decode_batch(take).tolist():
                tokens.append(token)
                if token in eos:
                    hit_eos = True
                    break
        return GraphDecodeResult(
            token_ids=tokens,
            prefill_tokens=prompt_tokens,
            decode_steps=self.decode_steps - steps_before,
            captured_graphs=self.captured_graphs,
            hit_eos=hit_eos,
        )


__all__ = ["CudaGraphDecodeSession", "GraphDecodeResult"]
