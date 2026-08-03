"""FLUX.2 klein loading, modelopt mapping, and inference helpers."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
from torch import nn

from xqt.core.errors import XQTBackendError
from xqt.quant.quantizers.convrot_4bit import (
    ConvRot4BitQuantizationResult,
    quantize_with_convrot_4bit,
)
from xqt.runtime import apply_execution_policy

from .types import (
    FLUX2_KLEIN_4B_NVFP4_FILENAME,
    FLUX2_KLEIN_4B_NVFP4_REPO_ID,
    FLUX2_KLEIN_4B_REPO_ID,
    _Flux2KleinNVFP4Linear,
    _MODEL_OPT_NVFP4_PARAMS,
    _NON_QUANT_KEY_RENAMES,
    _MappedNVFP4Layer,
    _cuda_arch,
    flux2_klein_nvfp4_single_file_url,
    normalize_flux2_klein_nvfp4_engine,
)
from .targets import materialize_flux2_klein_nvfp4_engine


def _resolve_flux2_klein_nvfp4_model_file(
    model_file: str | None,
    *,
    local_files_only: bool,
) -> str:
    if model_file is not None:
        return model_file
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        return flux2_klein_nvfp4_single_file_url()
    return hf_hub_download(
        repo_id=FLUX2_KLEIN_4B_NVFP4_REPO_ID,
        filename=FLUX2_KLEIN_4B_NVFP4_FILENAME,
        local_files_only=local_files_only,
    )


def _local_safetensors_path(source: str) -> Path | None:
    try:
        path = Path(source)
    except TypeError:
        return None
    if path.is_file() and path.suffix == ".safetensors":
        return path
    return None


def _modelopt_nvfp4_layers(source: str) -> tuple[str, ...]:
    path = _local_safetensors_path(source)
    if path is None:
        return ()
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise XQTBackendError("safetensors is required to inspect FLUX.2 NVFP4 weights") from exc
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        raw_metadata = (handle.metadata() or {}).get("_quantization_metadata")
    if raw_metadata is None:
        return ()
    metadata = json.loads(raw_metadata)
    layers = metadata.get("layers")
    if not isinstance(layers, Mapping):
        return ()
    return tuple(
        str(name)
        for name, entry in layers.items()
        if isinstance(entry, Mapping) and entry.get("format") == "nvfp4"
    )


def _map_modelopt_nvfp4_layer(source_name: str) -> list[_MappedNVFP4Layer]:
    parts = source_name.split(".")
    if len(parts) < 4:
        return []
    if parts[0] == "double_blocks" and len(parts) == 4:
        block_index = parts[1]
        stream = parts[2]
        leaf = parts[3]
        block_prefix = f"transformer_blocks.{block_index}"
        if stream == "img_attn" and leaf == "proj":
            return [
                _MappedNVFP4Layer(
                    source_name=source_name,
                    target_path=f"{block_prefix}.attn.to_out.0",
                )
            ]
        if stream == "txt_attn" and leaf == "proj":
            return [
                _MappedNVFP4Layer(
                    source_name=source_name,
                    target_path=f"{block_prefix}.attn.to_add_out",
                )
            ]
        if stream == "img_attn" and leaf == "qkv":
            return [
                _MappedNVFP4Layer(
                    source_name=source_name,
                    target_path=f"{block_prefix}.attn.to_q",
                    chunk_index=0,
                    chunk_count=3,
                ),
                _MappedNVFP4Layer(
                    source_name=source_name,
                    target_path=f"{block_prefix}.attn.to_k",
                    chunk_index=1,
                    chunk_count=3,
                ),
                _MappedNVFP4Layer(
                    source_name=source_name,
                    target_path=f"{block_prefix}.attn.to_v",
                    chunk_index=2,
                    chunk_count=3,
                ),
            ]
        if stream == "txt_attn" and leaf == "qkv":
            return [
                _MappedNVFP4Layer(
                    source_name=source_name,
                    target_path=f"{block_prefix}.attn.add_q_proj",
                    chunk_index=0,
                    chunk_count=3,
                ),
                _MappedNVFP4Layer(
                    source_name=source_name,
                    target_path=f"{block_prefix}.attn.add_k_proj",
                    chunk_index=1,
                    chunk_count=3,
                ),
                _MappedNVFP4Layer(
                    source_name=source_name,
                    target_path=f"{block_prefix}.attn.add_v_proj",
                    chunk_index=2,
                    chunk_count=3,
                ),
            ]
        if stream == "img_mlp" and leaf == "0":
            return [
                _MappedNVFP4Layer(
                    source_name=source_name,
                    target_path=f"{block_prefix}.ff.linear_in",
                )
            ]
        if stream == "img_mlp" and leaf == "2":
            return [
                _MappedNVFP4Layer(
                    source_name=source_name,
                    target_path=f"{block_prefix}.ff.linear_out",
                )
            ]
        if stream == "txt_mlp" and leaf == "0":
            return [
                _MappedNVFP4Layer(
                    source_name=source_name,
                    target_path=f"{block_prefix}.ff_context.linear_in",
                )
            ]
        if stream == "txt_mlp" and leaf == "2":
            return [
                _MappedNVFP4Layer(
                    source_name=source_name,
                    target_path=f"{block_prefix}.ff_context.linear_out",
                )
            ]
        return []
    if parts[0] == "single_blocks" and len(parts) == 3:
        block_index = parts[1]
        block_prefix = f"single_transformer_blocks.{block_index}.attn"
        if parts[2] == "linear1":
            return [
                _MappedNVFP4Layer(
                    source_name=source_name,
                    target_path=f"{block_prefix}.to_qkv_mlp_proj",
                )
            ]
        if parts[2] == "linear2":
            return [
                _MappedNVFP4Layer(
                    source_name=source_name,
                    target_path=f"{block_prefix}.to_out",
                )
            ]
    return []


def _replace_submodule(root: nn.Module, target_path: str, module: nn.Module) -> None:
    parent_path, _, child_name = target_path.rpartition(".")
    parent = root.get_submodule(parent_path) if parent_path else root
    setattr(parent, child_name, module)


def _map_modelopt_non_quant_key(source_key: str) -> str | None:
    renamed = _NON_QUANT_KEY_RENAMES.get(source_key)
    if renamed is not None:
        return renamed
    parts = source_key.split(".")
    if (
        len(parts) == 6
        and parts[0] == "double_blocks"
        and parts[3] == "norm"
        and parts[5] == "scale"
    ):
        block_prefix = f"transformer_blocks.{parts[1]}.attn"
        if parts[2] == "img_attn" and parts[4] == "query_norm":
            return f"{block_prefix}.norm_q.weight"
        if parts[2] == "img_attn" and parts[4] == "key_norm":
            return f"{block_prefix}.norm_k.weight"
        if parts[2] == "txt_attn" and parts[4] == "query_norm":
            return f"{block_prefix}.norm_added_q.weight"
        if parts[2] == "txt_attn" and parts[4] == "key_norm":
            return f"{block_prefix}.norm_added_k.weight"
    if (
        len(parts) == 5
        and parts[0] == "single_blocks"
        and parts[2] == "norm"
        and parts[4] == "scale"
    ):
        block_prefix = f"single_transformer_blocks.{parts[1]}.attn"
        if parts[3] == "query_norm":
            return f"{block_prefix}.norm_q.weight"
        if parts[3] == "key_norm":
            return f"{block_prefix}.norm_k.weight"
    return None


def _slice_rows(
    tensor: torch.Tensor | None,
    *,
    chunk_index: int | None,
    chunk_count: int | None,
) -> torch.Tensor | None:
    if tensor is None or chunk_index is None or chunk_count is None:
        return tensor
    if tensor.ndim == 0:
        return tensor
    rows = int(tensor.shape[0])
    if rows % int(chunk_count) != 0:
        raise XQTBackendError(
            f"cannot split FLUX.2 NVFP4 tensor with rows={rows} into {chunk_count} chunks"
        )
    chunk_rows = rows // int(chunk_count)
    start = int(chunk_index) * chunk_rows
    return tensor[start : start + chunk_rows]


def _load_modelopt_nvfp4_linear(handle: Any, mapped: _MappedNVFP4Layer) -> nn.Module:
    keys = set(handle.keys())
    source = mapped.source_name
    missing = [
        name
        for name in _MODEL_OPT_NVFP4_PARAMS
        if f"{source}.{name}" not in keys
    ]
    if missing:
        raise XQTBackendError(
            f"FLUX.2 NVFP4 layer {source} is missing tensors: {', '.join(sorted(missing))}"
        )
    bias_key = f"{source}.bias"
    bias = handle.get_tensor(bias_key) if bias_key in keys else None
    return _Flux2KleinNVFP4Linear(
        packed_weight=_slice_rows(
            handle.get_tensor(f"{source}.weight"),
            chunk_index=mapped.chunk_index,
            chunk_count=mapped.chunk_count,
        ),
        weight_scale=_slice_rows(
            handle.get_tensor(f"{source}.weight_scale"),
            chunk_index=mapped.chunk_index,
            chunk_count=mapped.chunk_count,
        ),
        weight_scale_2=handle.get_tensor(f"{source}.weight_scale_2"),
        input_scale=handle.get_tensor(f"{source}.input_scale"),
        bias=_slice_rows(
            bias,
            chunk_index=mapped.chunk_index,
            chunk_count=mapped.chunk_count,
        ),
        source_name=source,
    )


def _load_modelopt_nvfp4_transformer(
    source: str,
    *,
    transformer_cls: type[Any],
    config: str | None,
    config_subfolder: str | None,
    dtype: torch.dtype,
    device: str | torch.device | None,
    local_files_only: bool,
    **kwargs: Any,
) -> nn.Module:
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise XQTBackendError("safetensors is required to load FLUX.2 NVFP4 weights") from exc
    if config is None:
        raise XQTBackendError(
            "modelopt FLUX.2 NVFP4 loading requires a Diffusers transformer config"
        )
    config_kwargs = {
        "pretrained_model_name_or_path": config,
        "local_files_only": local_files_only,
    }
    if config_subfolder is not None:
        config_kwargs["subfolder"] = config_subfolder
    config_revision = kwargs.get("config_revision")
    if config_revision is not None:
        config_kwargs["revision"] = config_revision
    model_config = transformer_cls.load_config(**config_kwargs)
    transformer = transformer_cls.from_config(model_config)
    layers = _modelopt_nvfp4_layers(source)
    replaced_targets = 0
    with safe_open(source, framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        for layer in layers:
            for mapped in _map_modelopt_nvfp4_layer(layer):
                target_module = _load_modelopt_nvfp4_linear(handle, mapped)
                _replace_submodule(transformer, mapped.target_path, target_module)
                replaced_targets += 1
        state_dict = {}
        for source_key in keys:
            target_key = _map_modelopt_non_quant_key(source_key)
            if target_key is not None:
                state_dict[target_key] = handle.get_tensor(source_key)
    if replaced_targets == 0:
        raise XQTBackendError("no FLUX.2 modelopt NVFP4 Linear layers were mapped")
    transformer.load_state_dict(state_dict, strict=False)
    transformer.eval()
    if dtype is not None:
        transformer.to(dtype=dtype)
    if device is not None:
        transformer.to(device=device)
    return transformer


def load_flux2_klein_nvfp4_transformer(
    *,
    model_file: str | None = None,
    config: str | None = FLUX2_KLEIN_4B_REPO_ID,
    config_subfolder: str | None = "transformer",
    dtype: torch.dtype = torch.float16,
    device: str | torch.device | None = None,
    local_files_only: bool = False,
    **kwargs: Any,
) -> nn.Module:
    """Lazy-load the FLUX.2 klein NVFP4 transformer single-file checkpoint."""

    try:
        from diffusers import Flux2Transformer2DModel
    except ImportError as exc:
        raise XQTBackendError(
            "diffusers is required to load FLUX.2 klein NVFP4. Install a version exposing Flux2Transformer2DModel."
        ) from exc
    source = _resolve_flux2_klein_nvfp4_model_file(
        model_file,
        local_files_only=local_files_only,
    )
    load_kwargs = dict(kwargs)
    if config is not None and "original_config" not in load_kwargs:
        load_kwargs["config"] = config
    if (
        config_subfolder is not None
        and "subfolder" not in load_kwargs
        and "original_config" not in load_kwargs
    ):
        load_kwargs["subfolder"] = config_subfolder
    if _modelopt_nvfp4_layers(source):
        return _load_modelopt_nvfp4_transformer(
            source,
            transformer_cls=Flux2Transformer2DModel,
            config=config,
            config_subfolder=config_subfolder,
            dtype=dtype,
            device=device,
            local_files_only=local_files_only,
            **kwargs,
        )
    transformer = Flux2Transformer2DModel.from_single_file(
        source,
        torch_dtype=dtype,
        local_files_only=local_files_only,
        **load_kwargs,
    )
    transformer.eval()
    if device is not None:
        transformer.to(device=device)
    return transformer


def load_flux2_klein_nvfp4_pipeline(
    *,
    model_file: str | None = None,
    base_repo_id: str = FLUX2_KLEIN_4B_REPO_ID,
    dtype: torch.dtype = torch.float16,
    device: str | torch.device | None = None,
    local_files_only: bool = False,
    **kwargs: Any,
) -> Any:
    """Lazy-load a Diffusers FLUX.2 klein pipeline with the NVFP4 transformer."""

    try:
        from diffusers import Flux2KleinPipeline
    except ImportError as exc:
        raise XQTBackendError(
            "diffusers is required to load FLUX.2 klein pipelines. Install a version exposing Flux2KleinPipeline."
        ) from exc
    transformer = load_flux2_klein_nvfp4_transformer(
        model_file=model_file,
        config=base_repo_id,
        config_subfolder="transformer",
        dtype=dtype,
        device=device,
        local_files_only=local_files_only,
    )
    pipeline = Flux2KleinPipeline.from_pretrained(
        base_repo_id,
        transformer=transformer,
        torch_dtype=dtype,
        local_files_only=local_files_only,
        **kwargs,
    )
    if device is not None:
        pipeline.to(device)
    return pipeline


def load_flux2_klein_bf16_transformer(
    *,
    repo_id: str = FLUX2_KLEIN_4B_REPO_ID,
    subfolder: str = "transformer",
    dtype: torch.dtype = torch.bfloat16,
    device: str | torch.device | None = None,
    local_files_only: bool = False,
    **kwargs: Any,
) -> nn.Module:
    """Lazy-load the FLUX.2 klein BF16 transformer from Diffusers."""

    try:
        from diffusers import Flux2Transformer2DModel
    except ImportError as exc:
        raise XQTBackendError(
            "diffusers is required to load FLUX.2 klein BF16. Install a version exposing Flux2Transformer2DModel."
        ) from exc
    transformer = Flux2Transformer2DModel.from_pretrained(
        repo_id,
        subfolder=subfolder,
        torch_dtype=dtype,
        local_files_only=local_files_only,
        **kwargs,
    )
    transformer.eval()
    if device is not None:
        transformer.to(device=device)
    return transformer


def load_flux2_klein_bf16_pipeline(
    *,
    repo_id: str = FLUX2_KLEIN_4B_REPO_ID,
    dtype: torch.dtype = torch.bfloat16,
    device: str | torch.device | None = None,
    local_files_only: bool = False,
    **kwargs: Any,
) -> Any:
    """Lazy-load a Diffusers FLUX.2 klein pipeline with the BF16 transformer."""

    try:
        from diffusers import Flux2KleinPipeline
    except ImportError as exc:
        raise XQTBackendError(
            "diffusers is required to load FLUX.2 klein pipelines. Install a version exposing Flux2KleinPipeline."
        ) from exc
    transformer = load_flux2_klein_bf16_transformer(
        repo_id=repo_id,
        subfolder="transformer",
        dtype=dtype,
        device=device,
        local_files_only=local_files_only,
    )
    pipeline = Flux2KleinPipeline.from_pretrained(
        repo_id,
        transformer=transformer,
        torch_dtype=dtype,
        local_files_only=local_files_only,
        **kwargs,
    )
    if device is not None:
        pipeline.to(device)
    return pipeline


def quantize_flux2_klein_bf16_transformer_to_convrot_4bit(
    model: nn.Module,
    *,
    policy: Mapping[str, Any] | None = None,
    calibration_inputs: Iterable[Any] | None = None,
    inplace: bool = False,
    materialize_mixed_precision: bool = True,
) -> ConvRot4BitQuantizationResult:
    """Quantize one FLUX.2 klein BF16 transformer with ConvRot W4A4 and optional policy materialization."""

    result = quantize_with_convrot_4bit(
        model,
        policy=policy,
        strategy="convrot_w4a4",
        calibration_inputs=calibration_inputs,
        inplace=inplace,
    )
    if not materialize_mixed_precision:
        return result
    execution_policies = result.metadata.get("execution_policies", [])
    if not execution_policies:
        return result
    first_policy = execution_policies[0]
    if not isinstance(first_policy, Mapping):
        return result
    precision_overrides = first_policy.get("precision_overrides", [])
    model_with_policy = apply_execution_policy(
        result.model,
        precision_overrides=precision_overrides if isinstance(precision_overrides, list) else None,
        default_precision="w4a4",
        inplace=False,
    )
    return ConvRot4BitQuantizationResult(
        model=model_with_policy,
        backend=result.backend,
        method=result.method,
        strategy=result.strategy,
        quantized_modules=list(result.quantized_modules),
        metadata=dict(result.metadata),
    )


def quantize_flux2_klein_bf16_pipeline_to_convrot_4bit(
    pipeline: Any,
    *,
    transformer_attr: str = "transformer",
    policy: Mapping[str, Any] | None = None,
    calibration_inputs: Iterable[Any] | None = None,
    inplace: bool = False,
    materialize_mixed_precision: bool = True,
) -> tuple[Any, ConvRot4BitQuantizationResult]:
    """Quantize the BF16 FLUX.2 klein transformer attached to one pipeline."""

    if not hasattr(pipeline, transformer_attr):
        raise XQTBackendError(
            f"pipeline does not expose transformer attribute {transformer_attr!r}"
        )
    target_pipeline = pipeline if inplace else copy.deepcopy(pipeline)
    transformer = getattr(target_pipeline, transformer_attr)
    result = quantize_flux2_klein_bf16_transformer_to_convrot_4bit(
        transformer,
        policy=policy,
        calibration_inputs=calibration_inputs,
        inplace=True,
        materialize_mixed_precision=materialize_mixed_precision,
    )
    setattr(target_pipeline, transformer_attr, result.model)
    return target_pipeline, result


def load_and_quantize_flux2_klein_bf16_pipeline_to_convrot_4bit(
    *,
    repo_id: str = FLUX2_KLEIN_4B_REPO_ID,
    dtype: torch.dtype = torch.bfloat16,
    device: str | torch.device | None = None,
    local_files_only: bool = False,
    transformer_attr: str = "transformer",
    policy: Mapping[str, Any] | None = None,
    calibration_inputs: Iterable[Any] | None = None,
    materialize_mixed_precision: bool = True,
    **kwargs: Any,
) -> tuple[Any, ConvRot4BitQuantizationResult]:
    """Load one BF16 FLUX.2 klein pipeline and quantize its transformer to ConvRot W4A4."""

    pipeline = load_flux2_klein_bf16_pipeline(
        repo_id=repo_id,
        dtype=dtype,
        device=device,
        local_files_only=local_files_only,
        **kwargs,
    )
    return quantize_flux2_klein_bf16_pipeline_to_convrot_4bit(
        pipeline,
        transformer_attr=transformer_attr,
        policy=policy,
        calibration_inputs=calibration_inputs,
        inplace=True,
        materialize_mixed_precision=materialize_mixed_precision,
    )


def run_flux2_klein_bf16_convrot_4bit_inference(
    pipeline: Any,
    *,
    prompt: str | list[str],
    transformer_attr: str = "transformer",
    policy: Mapping[str, Any] | None = None,
    calibration_inputs: Iterable[Any] | None = None,
    materialize_mixed_precision: bool = True,
    inplace: bool = False,
    **kwargs: Any,
) -> dict[str, Any]:
    """Quantize one BF16 FLUX.2 klein pipeline to ConvRot W4A4 and execute it once."""

    quantized_pipeline, result = quantize_flux2_klein_bf16_pipeline_to_convrot_4bit(
        pipeline,
        transformer_attr=transformer_attr,
        policy=policy,
        calibration_inputs=calibration_inputs,
        inplace=inplace,
        materialize_mixed_precision=materialize_mixed_precision,
    )
    with torch.inference_mode():
        output = quantized_pipeline(prompt=prompt, **kwargs)
    return {
        "pipeline": quantized_pipeline,
        "quantization": result.to_dict(),
        "output": output,
    }


def run_flux2_klein_nvfp4_inference(
    pipeline: Any,
    *,
    prompt: str | list[str],
    engine: str | None = None,
    target_arch: str | None = None,
    max_targets: int | None = None,
    **kwargs: Any,
) -> Any:
    """Run a Diffusers FLUX.2 klein pipeline after XQT engine materialization."""

    result = materialize_flux2_klein_nvfp4_engine(
        pipeline,
        engine=engine,
        target_arch=target_arch,
        max_targets=max_targets,
        inplace=True,
    )
    with torch.inference_mode():
        return result.model(prompt=prompt, **kwargs)
