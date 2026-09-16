"""MiniCPM5 model loading and quantization helpers for XQT.

The model uses the standard Transformers LlamaForCausalLM implementation.
Tokenizer and generation remain owned by the caller; this module owns model
selection, quantization policy, and model-side metadata.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Callable, Iterable, Mapping

import torch
from torch import nn

from xqt.compression.quant import (
    QuantizationPolicy,
    quantize_with_awq_weight_only,
    quantize_with_fp4_weight_only,
    quantize_with_gptq_weight_only,
    quantize_with_int8_mma,
)

from .minicpm5_quarot import (
    QuaRotTransformReport,
    apply_quarot_minicpm5,
    build_quarot_rotation,
)

MINICPM5_2B_REPO_ID = "openbmb/MiniCPM5-2B"
MINICPM5_2B_PROFILE_ID = "hf.minicpm5-2b"


def load_minicpm5(
    checkpoint: str | None = None,
    *,
    repo_id: str = MINICPM5_2B_REPO_ID,
    dtype: torch.dtype = torch.bfloat16,
    device: str | torch.device | None = None,
    local_files_only: bool = False,
    revision: str | None = None,
) -> nn.Module:
    """Load MiniCPM5 through the standard Transformers causal-LM loader."""

    try:
        from transformers import AutoModelForCausalLM
    except ImportError as exc:
        raise RuntimeError("transformers is required to load MiniCPM5") from exc

    source = checkpoint or repo_id
    kwargs: dict[str, Any] = {
        "dtype": dtype,
        "low_cpu_mem_usage": True,
        "local_files_only": local_files_only,
    }
    if revision is not None:
        kwargs["revision"] = revision
    model = AutoModelForCausalLM.from_pretrained(source, **kwargs)
    if not isinstance(model, nn.Module):
        raise TypeError("MiniCPM5 loader did not return torch.nn.Module")
    if device is not None:
        model = model.to(device)
    return model.eval()


def minicpm5_quantization_policy(
    *,
    min_parameters: int = 0,
    include_name_patterns: Iterable[str] = (),
    exclude_name_patterns: Iterable[str] = (r"(^|\.)lm_head$",),
) -> QuantizationPolicy:
    """Select transformer Linear projections while protecting the LM head."""

    return QuantizationPolicy(
        dtype="int8",
        scheme="weight_only",
        include_module_types=("Linear",),
        exclude_module_types=("LayerNorm", "Embedding"),
        include_name_patterns=tuple(str(value) for value in include_name_patterns),
        exclude_name_patterns=tuple(str(value) for value in exclude_name_patterns),
        min_parameters=int(min_parameters),
    )


def minicpm5_mlp_only_quantization_policy(
    *,
    min_parameters: int = 0,
) -> QuantizationPolicy:
    """Quantize MLP projections while keeping attention and LM head in BF16."""

    return minicpm5_quantization_policy(
        min_parameters=min_parameters,
        exclude_name_patterns=(r"(^|\.)lm_head$", r"\.self_attn\."),
    )


def minicpm5_edge_protected_quantization_policy(
    *,
    min_parameters: int = 0,
    edge_layers: int = 2,
) -> QuantizationPolicy:
    """Quantize the middle layers while protecting attention and edge layers."""

    if edge_layers < 0 or edge_layers > 21:
        raise ValueError("edge_layers must be between 0 and 21")
    protected = [r"(^|\.)lm_head$", r"\.self_attn\."]
    if edge_layers:
        first_layers = [str(index) for index in range(edge_layers)]
        last_layers = [str(index) for index in range(42 - edge_layers, 42)]
        layer_pattern = "|".join(first_layers + last_layers)
        protected.append(rf"^model\.layers\.({layer_pattern})\.")
    return minicpm5_quantization_policy(
        min_parameters=min_parameters,
        exclude_name_patterns=tuple(protected),
    )


class _MiniCPM5W4A16HybridLinear(nn.Module):
    """Decode (M <= 8) via the native SM89 W4A16 GEMV; prefill via dense cache.

    The native kernel only accepts row counts 1..8, while prefill batches are
    larger. The dense branch reuses the storage module's cached dequantized
    weight so prefill never repacks anything per forward.
    """

    def __init__(self, storage: Any, decode: Any) -> None:
        super().__init__()
        self.storage = storage
        self.decode = decode
        self.input_features = int(storage.input_features)
        self.output_features = int(storage.output_features)
        self._max_native_rows = 8
        self._int8_weight: torch.Tensor | None = None
        self._int8_scale: torch.Tensor | None = None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        flat = inputs.reshape(-1, self.input_features)
        rows = int(flat.shape[0])
        if (
            rows <= self._max_native_rows
            and flat.is_cuda
            and flat.dtype in {torch.float16, torch.bfloat16}
        ):
            return self.decode(flat)
        weight = self.storage.dense_weight(dtype=inputs.dtype, device=inputs.device)
        bias = self.storage.dense_bias(dtype=inputs.dtype, device=inputs.device)
        return nn.functional.linear(flat, weight, bias)

    def supports_prequant(self) -> bool:
        """Whether this view can run the INT8 W8A8 prefill GEMM.

        Cheap and side-effect free: only checks that the backing storage is the
        4-bit group-64 AWQ weight-only shell (so ``dense_weight`` exists) and
        that CUDA is available. No weights are materialized here.
        """

        if not torch.cuda.is_available():
            return False
        storage = self.storage
        return (
            int(getattr(storage, "bits", 0) or 0) == 4
            and int(getattr(storage, "group_size", 0) or 0) == 64
            and callable(getattr(storage, "dense_weight", None))
        )

    def _int8_view(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Lazily build the per-output-channel symmetric INT8 weight view.

        Derives the INT8 codes from the storage's dense bf16 weight
        ``[N, K]``. The bf16 dense cache is materialized anyway by the existing
        dense prefill branch, so this reuses it instead of unpacking the INT4
        codes by hand.
        """

        if self._int8_weight is not None and self._int8_scale is not None:
            if self._int8_weight.device == device:
                return self._int8_weight, self._int8_scale
        weight = self.storage.dense_weight(dtype=torch.bfloat16, device=device)
        weight_fp32 = weight.float()
        scale = (weight_fp32.abs().amax(dim=1, keepdim=True) / 127.0).clamp_min(1e-8)
        quantized = (
            (weight_fp32 / scale).round().clamp(-127, 127).to(torch.int8).contiguous()
        )
        weight_scale = scale.reshape(-1).contiguous()
        # Store the K-major [K, N] layout the GEMM consumes so the transpose is
        # paid once here instead of on every prefill call. `gemm_int8_triton`
        # materializes `b.t().contiguous()` internally when `transpose_b=True`,
        # which costs ~0.7 ms/layer at MiniCPM5-2B MLP shapes.
        weight_kn = quantized.t().contiguous()
        self._int8_scale = weight_scale
        self._int8_weight = weight_kn
        return weight_kn, weight_scale

    def forward_prequant(
        self, qactivation: torch.Tensor, activation_scale: torch.Tensor
    ) -> torch.Tensor:
        """Run the INT8 W8A8 prefill GEMM for a pre-quantized activation.

        ``qactivation`` is ``[rows, K]`` INT8 and ``activation_scale`` is
        ``[rows]`` FP32. Returns ``[rows, N]`` bf16, matching the module's
        flattened dense contract.
        """

        from xqt.kernels.ops._impl.triton.gemm import gemm_int8_triton

        qweight, weight_scale = self._int8_view(qactivation.device)
        bias = self.storage.dense_bias(dtype=torch.bfloat16, device=qactivation.device)
        return gemm_int8_triton(
            qactivation,
            qweight,
            activation_scale,
            weight_scale,
            bias=bias,
            transpose_b=False,
            block_m=128,
            block_n=64,
            block_k=64,
            group_m=8,
            num_warps=4,
            num_stages=3,
            output_dtype=torch.bfloat16,
        )

    def bind_residual(
        self, *, rows: int, dtype: torch.dtype
    ) -> Callable[[torch.Tensor, torch.Tensor], None]:
        """Bind the native decode GEMV with a fused in-place residual add.

        The residual epilogue exists only on the native decode path; prefill
        rows keep the dense branch. ``residual`` is ``[rows, output_features]``
        and is updated in place.
        """

        return self.decode.bind_residual(rows=rows, dtype=dtype)

    def bind_residual_w4a8(
        self, *, rows: int, dtype: torch.dtype
    ) -> Callable[[torch.Tensor, torch.Tensor | float, torch.Tensor], None]:
        return self.decode.bind_residual_w4a8(rows=rows, dtype=dtype)

    def bind_w4a8(
        self, *, rows: int, dtype: torch.dtype
    ) -> Callable[[torch.Tensor, torch.Tensor | float], torch.Tensor]:
        return self.decode.bind_w4a8(rows=rows, dtype=dtype)

    def forward_w4a8(
        self,
        inputs: torch.Tensor,
        scale_a: torch.Tensor | float,
        *,
        output_dtype: torch.dtype = torch.bfloat16,
    ) -> torch.Tensor:
        return self.decode.forward_w4a8(inputs, scale_a, output_dtype=output_dtype)

    def execution_metadata(self) -> dict[str, Any]:
        prefill_engine = (
            "int8_w8a8" if self.supports_prequant() else "dense_cache_f_linear"
        )
        return {
            "implementation": "minicpm5_w4a16_hybrid",
            "decode_engine": "native_sm89_awq_w4a16_gemv",
            "prefill_engine": prefill_engine,
            "max_native_rows": self._max_native_rows,
        }


def materialize_minicpm5_weight_only_runtime(
    model: nn.Module,
    *,
    target_arch: str = "sm_89",
    block_m: int = 16,
    block_n: int = 64,
    block_k: int = 128,
    engine: str = "cuda",
) -> nn.Module:
    """Wrap XQT packed INT4 Linear artifacts with a fast execution view.

    engine="cuda" (default, sm_89): each storage Linear is replaced by a hybrid
    view that routes decode rows (M <= 8) through the native FasterTransformer-
    style W4A16 GEMV (register-level nibble decode, measured ~2x dense bf16 at
    M=1) and larger batches through the dense dequantized cache + F.linear.
    engine="tilelang": use the K-blocked TileLang fused dequant GEMM instead
    (works on sm_89, requires fp16 activations, currently slower than the
    dense-cache path). engine="dense": keep the reference dense-cache view.
    """

    from xqt.contracts.weight_only import AWQGPTQWeightOnlyLinear

    for name, module in list(model.named_modules()):
        if not name or not isinstance(module, AWQGPTQWeightOnlyLinear):
            continue
        parent_path, _, attribute = name.rpartition(".")
        parent = model.get_submodule(parent_path) if parent_path else model
        if engine == "cuda":
            from xqt.runtime.modules.awq_w4a16_linear import AWQW4A16Linear

            replacement: nn.Module = _MiniCPM5W4A16HybridLinear(
                module,
                AWQW4A16Linear.from_signed_groupwise_storage(module),
            )
        elif engine == "tilelang":
            from xqt.kernels.wrappers.dequant_gemm import (
                _TileLangDequantGemmWrapper,
            )

            replacement = _TileLangDequantGemmWrapper(
                module,
                fallback="eager",
                settings={
                    "target_arch": target_arch,
                    "preferred_patterns": ["fp4_packed_dequant_gemm_epilogue"],
                    "linear_fastpath": "packed",
                    "block_m": int(block_m),
                    "block_n": int(block_n),
                    "block_k": int(block_k),
                    "threads": 256,
                    "num_stages": 3,
                },
            )
        elif engine == "dense":
            replacement = module
        else:
            raise ValueError("engine must be one of cuda, tilelang, dense")
        if attribute.isdigit() and isinstance(parent, (nn.Sequential, nn.ModuleList)):
            parent[int(attribute)] = replacement
        else:
            setattr(parent, attribute, replacement)
        replacement.eval()
    return model


def materialize_minicpm5_int8_runtime(
    model: nn.Module,
    *,
    engine: str = "auto",
) -> nn.Module:
    """Materialize XQT's infer-facing INT8 execution views in a model."""

    from xqt.contracts.int8_mma import Int8MmaLinear as StorageInt8MmaLinear
    from xqt.runtime.modules.int8_mma_linear import (
        Int8MmaLinear as RuntimeInt8MmaLinear,
    )

    for name, module in list(model.named_modules()):
        if not name or type(module) is not StorageInt8MmaLinear:
            continue
        parent_path, _, attribute = name.rpartition(".")
        parent = model.get_submodule(parent_path) if parent_path else model
        replacement = RuntimeInt8MmaLinear.from_storage(module, engine=engine)
        if attribute.isdigit() and isinstance(parent, (nn.Sequential, nn.ModuleList)):
            parent[int(attribute)] = replacement
        else:
            setattr(parent, attribute, replacement)
    return model


def minicpm5_linear_summary(model: nn.Module) -> dict[str, Any]:
    """Summarize Linear coverage and parameter sizes for a model snapshot."""

    rows = [
        {
            "name": name,
            "in_features": module.in_features,
            "out_features": module.out_features,
            "parameters": sum(parameter.numel() for parameter in module.parameters()),
        }
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear)
    ]
    return {
        "linear_module_count": len(rows),
        "linear_parameter_count": sum(int(row["parameters"]) for row in rows),
        "modules": rows,
    }


def quantize_minicpm5(
    model: nn.Module,
    *,
    strategy: str,
    policy: Mapping[str, Any] | QuantizationPolicy | None = None,
    calibration_inputs: Iterable[Any] | None = None,
    inplace: bool = False,
    engine: str = "auto",
    group_size: int | None = None,
    min_int8_rows: int = 0,
    materialize_runtime: bool = False,
    target_arch: str = "sm_89",
) -> Any:
    """Apply one existing XQT quantizer to MiniCPM5 Linear projections.

    ``materialize_runtime=True`` wraps INT4 modules with the TileLang packed
    dequant-GEMM bridge. That kernel allocates the full K dimension in shared
    memory and only fits sm_90+ shapes; on sm_89 LLM shapes it raises during
    construction and every forward would fall back to a slow eager reference
    (~1500x dense bf16 at M=1). Keep it False unless the target kernel is
    known to support the layer shapes.
    """

    effective_policy: Mapping[str, Any] | QuantizationPolicy
    if policy is None:
        effective_policy = minicpm5_quantization_policy()
        if group_size is not None:
            effective_policy = asdict(effective_policy)
            effective_policy["group_size"] = int(group_size)
    elif isinstance(policy, QuantizationPolicy) and group_size is not None:
        effective_policy = asdict(policy)
        effective_policy["group_size"] = int(group_size)
    elif isinstance(policy, Mapping) and group_size is not None:
        effective_policy = dict(policy)
        effective_policy["group_size"] = int(group_size)
    else:
        effective_policy = policy
    if strategy == "w8a8_int8":
        result = quantize_with_int8_mma(
            model,
            policy=effective_policy,
            strategy=strategy,
            inplace=inplace,
            engine=engine,
            fallback_engine="torch_int_mm",
            allow_dynamic_fallback=True,
            min_int8_rows=min_int8_rows,
        )
        if materialize_runtime:
            result.model = materialize_minicpm5_int8_runtime(
                result.model,
                engine=engine,
            )
        return result
    if strategy == "w4a16_int4":
        result = quantize_with_awq_weight_only(
            model,
            policy=effective_policy,
            calibration_inputs=calibration_inputs,
            strategy=strategy,
            inplace=inplace,
        )
        if materialize_runtime:
            result.model = materialize_minicpm5_weight_only_runtime(
                result.model,
                target_arch=target_arch,
                block_m=16,
                block_n=64,
                block_k=128,
            )
        return result
    if strategy == "w4a16_gptq":
        return quantize_with_gptq_weight_only(
            model,
            policy=effective_policy,
            calibration_inputs=calibration_inputs,
            strategy="w4a16_int4",
            inplace=inplace,
        )
    if strategy == "w4a16_fp4":
        return quantize_with_fp4_weight_only(
            model,
            policy=effective_policy,
            strategy=strategy,
            inplace=inplace,
        )
    raise ValueError(
        "unsupported MiniCPM5 strategy; expected one of "
        "w8a8_int8, w4a16_int4, w4a16_gptq, w4a16_fp4"
    )


def quantize_minicpm5_lm_head_w4(
    model: nn.Module,
    *,
    group_size: int = 64,
) -> nn.Module:
    """Quantize the MiniCPM5 LM head to W4A16 group-64 and return a decode view.

    The default Linear policies protect ``lm_head`` because it is
    quality-sensitive; quantizing it is a deliberate deployment choice. The
    BF16 head reads about 0.53 GB per decode token, so a W4 head cuts that to
    about 0.13 GB. The returned module serves decode rows (M <= 8) through the
    native SM89 AWQ GEMV and keeps a dense view for larger batches.
    """

    from xqt.contracts.weight_only import AWQGPTQWeightOnlyLinear
    from xqt.runtime.modules.awq_w4a16_linear import AWQW4A16Linear

    head = getattr(model, "lm_head", None)
    if not isinstance(head, nn.Linear):
        raise TypeError(
            "quantize_minicpm5_lm_head_w4 expects model.lm_head to be nn.Linear"
        )
    storage = AWQGPTQWeightOnlyLinear.from_linear(
        head,
        bits=4,
        group_size=group_size,
        method="rtn",
    )
    device = next(model.parameters()).device
    storage = storage.to(device)
    return AWQW4A16Linear.from_signed_groupwise_storage(storage).eval()


__all__ = [
    "MINICPM5_2B_PROFILE_ID",
    "MINICPM5_2B_REPO_ID",
    "load_minicpm5",
    "minicpm5_linear_summary",
    "minicpm5_quantization_policy",
    "quantize_minicpm5",
    "quantize_minicpm5_lm_head_w4",
]
