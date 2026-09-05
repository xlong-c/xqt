"""vLLM-style HF compressed-tensors export for XQT quantized models.

Optional external format adapter only; not an internal runtime fact source.

Writes a checkpoint directory consumable by vLLM-style external quantization
loaders and by ``probe_external_quant_config`` / the C4 load path. Validation is
offline only (format self-check + probe round-trip); real vLLM/SGLang load is
out of scope.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from xqt.contracts import QuantizedModel
from xqt.core.errors import XQTArtifactError
from xqt.contracts.external import probe_external_quant_config
from xqt.contracts.int8_mma import Int8MmaLinear
from xqt.contracts.weight_only import AWQGPTQWeightOnlyLinear
from xqt.export.base import ExportResultBase


@dataclass(frozen=True)
class HFQuantExportReport(ExportResultBase):
    """Result of exporting one XQT quantized model to HF quant layout."""

    output_dir: str
    format: str
    module_count: int
    files: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def artifact_paths(self) -> tuple[Path, ...]:
        return (Path(self.output_dir),)

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_dir": self.output_dir,
            "format": self.format,
            "module_count": int(self.module_count),
            "files": list(self.files),
            "metadata": dict(self.metadata),
        }


def _module_state_prefix(name: str) -> str:
    return f"{name}." if name else ""


def _pack_awq_gptq_module(
    name: str,
    module: AWQGPTQWeightOnlyLinear,
    state: dict[str, torch.Tensor],
) -> None:
    prefix = _module_state_prefix(name)
    state[f"{prefix}qweight"] = module.quantized_weight.detach().cpu()
    state[f"{prefix}scales"] = module.weight_scale.detach().cpu()
    if module.bias is not None:
        state[f"{prefix}bias"] = module.bias.detach().cpu()


def _pack_int8_mma_module(
    name: str,
    module: Int8MmaLinear,
    state: dict[str, torch.Tensor],
) -> None:
    prefix = _module_state_prefix(name)
    state[f"{prefix}qweight"] = module.qweight_t.t().contiguous().detach().cpu()
    state[f"{prefix}scales"] = module.weight_scale.detach().cpu()
    if module.bias is not None:
        state[f"{prefix}bias"] = module.bias.detach().cpu()
    if getattr(module, "activation_scale_mode", None) == "static" and getattr(module, "activation_scale", None) is not None:
        scale = module.activation_scale
        if isinstance(scale, torch.Tensor):
            state[f"{prefix}input_scale"] = scale.detach().cpu()
        elif isinstance(scale, (int, float)):
            state[f"{prefix}input_scale"] = torch.tensor(scale, dtype=torch.float32)


def _is_kv_scale_buffer(name: str) -> bool:
    leaf = name.rsplit(".", 1)[-1].lower()
    return (
        leaf in {"k_scale", "v_scale", "attn_k_scale", "attn_v_scale"}
        or name.endswith(".attn.k_scale")
        or name.endswith(".attn.v_scale")
        or leaf.endswith("k_scale")
        or leaf.endswith("v_scale")
    )


def _normalize_kv_scale_key(name: str) -> str:
    if name.endswith(".attn.k_scale") or name.endswith(".attn.v_scale"):
        return name
    if name.endswith(".k_scale"):
        prefix = name[:-len(".k_scale")]
        return f"{prefix}.attn.k_scale" if prefix else "attn.k_scale"
    if name.endswith(".v_scale"):
        prefix = name[:-len(".v_scale")]
        return f"{prefix}.attn.v_scale" if prefix else "attn.v_scale"
    if name.endswith(".attn_k_scale"):
        prefix = name[:-len(".attn_k_scale")]
        return f"{prefix}.attn.k_scale" if prefix else "attn.k_scale"
    if name.endswith(".attn_v_scale"):
        prefix = name[:-len(".attn_v_scale")]
        return f"{prefix}.attn.v_scale" if prefix else "attn.v_scale"
    return name


def _collect_packed_state(model: nn.Module) -> tuple[dict[str, torch.Tensor], list[str], list[str]]:
    state: dict[str, torch.Tensor] = {}
    packed: list[str] = []
    kv_scales: list[str] = []

    for name, module in model.named_modules():
        if isinstance(module, AWQGPTQWeightOnlyLinear):
            _pack_awq_gptq_module(name, module, state)
            packed.append(name)
            continue
        if isinstance(module, Int8MmaLinear):
            _pack_int8_mma_module(name, module, state)
            packed.append(name)
            continue
        if name and hasattr(module, "state_dict") and not any(
            child is module for child in model.children()
        ):
            continue

    # Collect kv scale buffers from all named_buffers
    for buf_name, buf in model.named_buffers():
        if _is_kv_scale_buffer(buf_name):
            tensor_val = buf.detach().cpu()
            state[buf_name] = tensor_val
            norm_key = _normalize_kv_scale_key(buf_name)
            if norm_key != buf_name:
                state[norm_key] = tensor_val
            kv_scales.append(norm_key)

    if not packed:
        # Fall back to full state_dict so the export is still loadable.
        for k, v in model.state_dict().items():
            if k not in state:
                state[k] = v.detach().cpu()
    else:
        # Also collect unquantized parameters that do not belong to packed modules
        packed_prefixes = tuple(f"{p}." for p in packed)
        for param_name, param in model.named_parameters():
            if not any(param_name.startswith(p) for p in packed_prefixes):
                if param_name not in state:
                    state[param_name] = param.detach().cpu()
        for buf_name, buf in model.named_buffers():
            if not any(buf_name.startswith(p) for p in packed_prefixes):
                if buf_name not in state:
                    state[buf_name] = buf.detach().cpu()

    return state, packed, sorted(set(kv_scales))


def _quantization_config_payload(
    *,
    format_name: str,
    bits: int,
    group_size: int,
    method: str | None,
    strategy: str | None,
    kv_cache_scheme: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "quant_method": format_name if format_name != "compressed_tensors" else "compressed-tensors",
        "format": "compressed-tensors" if format_name == "compressed_tensors" else format_name,
        "bits": int(bits),
        "group_size": int(group_size),
        "sym": True,
        "producer": "xqt",
    }
    if method is not None:
        payload["xqt_method"] = str(method)
    if strategy is not None:
        payload["xqt_strategy"] = str(strategy)
    if kv_cache_scheme is not None:
        payload["kv_cache_scheme"] = dict(kv_cache_scheme)
    if format_name == "compressed_tensors":
        payload["config_groups"] = {
            "group_0": {
                "weights": {
                    "num_bits": int(bits),
                    "type": "int",
                    "strategy": "group",
                    "group_size": int(group_size),
                    "symmetric": True,
                }
            }
        }
    if extra:
        payload.update({str(k): v for k, v in extra.items()})
    return payload


def export_compressed_tensors(
    quantized: QuantizedModel | nn.Module,
    out_dir: str | Path,
    *,
    format_name: str = "compressed_tensors",
    bits: int | None = None,
    group_size: int | None = None,
    kv_cache_scheme: Mapping[str, Any] | None = None,
    config_extra: Mapping[str, Any] | None = None,
    weights_format: str = "safetensors",
    weights_name: str | None = None,
) -> HFQuantExportReport:
    """Export an XQT quantized model as a vLLM-style HF quant checkpoint directory.

    Layout:
    - ``config.json`` with ``quantization_config``
    - ``model.safetensors`` (or ``model.pt``) packed / state tensors
    - ``xqt_export.json`` lineage sidecar
    """

    if isinstance(quantized, QuantizedModel):
        model = quantized.model
        method = quantized.method
        strategy = quantized.strategy
        backend = quantized.backend
        if not isinstance(model, nn.Module):
            raise XQTArtifactError(
                "export_compressed_tensors requires QuantizedModel.model to be nn.Module"
            )
    elif isinstance(quantized, nn.Module):
        model = quantized
        method = None
        strategy = None
        backend = "pytorch"
    else:
        raise TypeError(
            "export_compressed_tensors expects QuantizedModel or nn.Module; "
            f"got {type(quantized).__name__}"
        )

    resolved_bits = int(bits) if bits is not None else 4
    resolved_group = int(group_size) if group_size is not None else 128
    for module in model.modules():
        if isinstance(module, AWQGPTQWeightOnlyLinear):
            resolved_bits = int(module.bits)
            resolved_group = int(module.group_size)
            break
        if isinstance(module, Int8MmaLinear):
            resolved_bits = 8
            resolved_group = int(getattr(module, "block_k", 64))
            break

    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    state, packed_modules, detected_kv_scales = _collect_packed_state(model)

    resolved_kv_cache_scheme: dict[str, Any] | None = None
    if kv_cache_scheme is not None:
        resolved_kv_cache_scheme = dict(kv_cache_scheme)
    elif detected_kv_scales:
        resolved_kv_cache_scheme = {
            "type": "fp8",
            "num_bits": 8,
            "strategy": "tensor",
            "symmetric": True,
        }

    if weights_name is not None:
        resolved_weights_name = weights_name
        use_safetensors = resolved_weights_name.endswith(".safetensors")
    elif weights_format == "safetensors":
        resolved_weights_name = "model.safetensors"
        use_safetensors = True
    elif weights_format in ("pt", "torch_state_dict"):
        resolved_weights_name = "model.pt"
        use_safetensors = False
    else:
        raise XQTArtifactError(
            f"unsupported weights_format={weights_format!r}; supported are 'safetensors', 'pt'"
        )

    weights_path = output / resolved_weights_name
    if use_safetensors:
        try:
            from safetensors.torch import save_file
        except ImportError as exc:
            raise XQTArtifactError(
                "exporting safetensors requires safetensors package"
            ) from exc
        tensor_dict = {
            str(k): (v.detach().cpu().contiguous() if isinstance(v, torch.Tensor) else v)
            for k, v in state.items()
        }
        save_file(tensor_dict, str(weights_path))
        actual_format = "safetensors"
    else:
        torch.save(state, weights_path)
        actual_format = "torch_state_dict"

    quant_config = _quantization_config_payload(
        format_name=format_name,
        bits=resolved_bits,
        group_size=resolved_group,
        method=method,
        strategy=strategy,
        kv_cache_scheme=resolved_kv_cache_scheme,
        extra=config_extra,
    )
    config_payload = {
        "model_type": "xqt_exported",
        "architectures": ["XQTExportedModel"],
        "quantization_config": quant_config,
    }
    config_path = output / "config.json"
    config_path.write_text(
        json.dumps(config_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    lineage = {
        "producer": "xqt.export.hf_quant",
        "backend": backend,
        "method": method,
        "strategy": strategy,
        "packed_modules": packed_modules,
        "format": format_name,
        "weights_format": actual_format,
        "kv_scales": detected_kv_scales,
        "kv_cache_scheme": resolved_kv_cache_scheme,
    }
    lineage_path = output / "xqt_export.json"
    lineage_path.write_text(
        json.dumps(lineage, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    probed = probe_external_quant_config(output)
    if probed is None:
        raise XQTArtifactError(
            f"export_compressed_tensors self-check failed: probe returned None for {output}"
        )

    files = ("config.json", resolved_weights_name, "xqt_export.json")
    return HFQuantExportReport(
        output_dir=str(output),
        format=format_name,
        module_count=len(packed_modules),
        files=files,
        metadata={
            "bits": resolved_bits,
            "group_size": resolved_group,
            "probe_format": probed.format,
            "packed_modules": packed_modules,
            "weights_format": actual_format,
            "weights_name": resolved_weights_name,
            "kv_scales": detected_kv_scales,
            "kv_cache_scheme": resolved_kv_cache_scheme,
        },
    )


__all__ = [
    "HFQuantExportReport",
    "export_compressed_tensors",
]
