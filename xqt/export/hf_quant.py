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
from xqt.quant.external import probe_external_quant_config
from xqt.quant.quantizers.awq_gptq_weight_only import AWQGPTQWeightOnlyLinear


@dataclass(frozen=True)
class HFQuantExportReport:
    """Result of exporting one XQT quantized model to HF quant layout."""

    output_dir: str
    format: str
    module_count: int
    files: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

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


def _collect_packed_state(model: nn.Module) -> tuple[dict[str, torch.Tensor], list[str]]:
    state: dict[str, torch.Tensor] = {}
    packed: list[str] = []
    for name, module in model.named_modules():
        if isinstance(module, AWQGPTQWeightOnlyLinear):
            _pack_awq_gptq_module(name, module, state)
            packed.append(name)
            continue
        if name and hasattr(module, "state_dict") and not any(
            child is module for child in model.children()
        ):
            continue
    if not packed:
        # Fall back to full state_dict so the export is still loadable.
        state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    return state, packed


def _quantization_config_payload(
    *,
    format_name: str,
    bits: int,
    group_size: int,
    method: str | None,
    strategy: str | None,
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
    config_extra: Mapping[str, Any] | None = None,
) -> HFQuantExportReport:
    """Export an XQT quantized model as a vLLM-style HF quant checkpoint directory.

    Layout:
    - ``config.json`` with ``quantization_config``
    - ``model.pt`` packed / state tensors
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

    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    state, packed_modules = _collect_packed_state(model)
    weights_path = output / "model.pt"
    torch.save(state, weights_path)

    quant_config = _quantization_config_payload(
        format_name=format_name,
        bits=resolved_bits,
        group_size=resolved_group,
        method=method,
        strategy=strategy,
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

    files = ("config.json", "model.pt", "xqt_export.json")
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
        },
    )


__all__ = [
    "HFQuantExportReport",
    "export_compressed_tensors",
]
