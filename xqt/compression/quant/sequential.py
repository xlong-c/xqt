"""Layer-Sequential Quantization Pipeline for Large Transformer Models.

Enables layer-by-layer calibration, quantization, and memory release for
large models (e.g. 70B, 32B, 7B) on memory-constrained devices (e.g. 24GB GPUs).
Captures activations passing from prefix embeddings into successive blocks,
quantizes block linear layers locally, passes forward activations to the next block,
and offloads processed blocks back to CPU.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence

import torch
from torch import nn

from xqt.contracts.model_structure import (
    ModelStructureContract,
    is_module_path_within,
    structure_contract_keep_high_precision_paths,
)
from xqt.contracts.packing_int4 import _normalize_group_size
from xqt.contracts.weight_only import (
    AWQGPTQWeightOnlyLinear,
    _quantize_grouped_weight,
)
from xqt.compression.quant.policy import (
    QuantizationPolicy,
    should_quantize_module,
)
from xqt.compression.quant.quantizers.fp4_weight_only import (
    _LinearCalibrationStats,
)
from xqt.compression.quant.quantizers.awq_gptq_weight_only import (
    _from_linear_awq,
    _from_linear_gptq,
)
from xqt.compression.quant.quantizers.adaptive_rounding import (
    optimize_linear_rounding,
)
from xqt.compression.quant.quantizers.base import (
    call_model,
    iter_calibration_batches,
    move_batch_to_device,
)


@dataclass(frozen=True)
class SequentialBlockSpec:
    """Specification of a single sequential block."""

    index: int
    name: str
    module: nn.Module
    quantizable_linears: tuple[str, ...]


@dataclass(frozen=True)
class SequentialPartition:
    """Decomposition of a model into prefix, sequential blocks, and postfix."""

    prefix_module_names: tuple[str, ...]
    blocks: tuple[SequentialBlockSpec, ...]
    postfix_module_names: tuple[str, ...]

    def block_names(self) -> tuple[str, ...]:
        return tuple(b.name for b in self.blocks)


@dataclass(frozen=True)
class LayerSequentialConfig:
    """Execution options for layer-sequential quantization."""

    target_device: str = "cuda"
    offload_device: str = "cpu"
    clean_memory_each_block: bool = True
    sample_limit: int | None = 16
    pass_kwargs_through: bool = True
    bits: int = 4  # Supports 2 (W2A16), 3 (W3A16), 4 (W4A16), 8 (W8A16)
    group_size: int = 128
    method: str = "awq"  # "awq", "gptq", "rtn", "adaround", "autoround"

    def __post_init__(self) -> None:
        if self.bits not in (2, 3, 4, 8):
            raise ValueError(f"LayerSequentialConfig.bits must be 2, 3, 4, or 8; got {self.bits}")
        if self.group_size <= 0:
            raise ValueError(f"LayerSequentialConfig.group_size must be positive; got {self.group_size}")


@dataclass(frozen=True)
class LayerSequentialReport:
    """Execution diagnostics and summary for layer-sequential quantization."""

    total_blocks: int
    quantized_blocks: int
    quantized_linear_count: int
    prefix_modules: tuple[str, ...]
    postfix_modules: tuple[str, ...]
    device_transitions: int
    peak_memory_bytes: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_blocks": self.total_blocks,
            "quantized_blocks": self.quantized_blocks,
            "quantized_linear_count": self.quantized_linear_count,
            "prefix_modules": list(self.prefix_modules),
            "postfix_modules": list(self.postfix_modules),
            "device_transitions": self.device_transitions,
            "peak_memory_bytes": self.peak_memory_bytes,
        }


def _get_submodule_by_path(root: nn.Module, path: str) -> nn.Module:
    """Retrieve a nested submodule by dot-separated path."""
    curr = root
    if not path:
        return curr
    for part in path.split("."):
        if part.isdigit():
            curr = curr[int(part)]
        else:
            curr = getattr(curr, part)
    return curr


def replace_submodule_in_block(
    block: nn.Module,
    rel_path: str,
    new_module: nn.Module,
) -> None:
    """Replace one nested module inside a block using a relative path."""
    parts = rel_path.split(".")
    parent = block
    for p in parts[:-1]:
        if p.isdigit():
            parent = parent[int(p)]
        else:
            parent = getattr(parent, p)
    last = parts[-1]
    if last.isdigit():
        parent[int(last)] = new_module
    else:
        setattr(parent, last, new_module)


def discover_sequential_partition(
    model: nn.Module,
    *,
    contract: ModelStructureContract | None = None,
    policy: QuantizationPolicy | None = None,
) -> SequentialPartition:
    """Discover sequential blocks, prefix, and postfix submodules in a model.

    Supports contract-driven block discovery as well as common transformer
    structural heuristics (e.g. model.layers, transformer.h, blocks).
    """
    effective_policy = policy or QuantizationPolicy()

    # Heuristic search candidate container paths
    candidate_container_names = (
        "model.layers",
        "layers",
        "transformer.h",
        "transformer.blocks",
        "blocks",
        "model.decoder.layers",
        "decoder.layers",
        "encoder.layer",
        "encoder.layers",
    )

    container_path: str | None = None
    container_module: nn.Module | None = None

    for name in candidate_container_names:
        try:
            sub = _get_submodule_by_path(model, name)
            if isinstance(sub, (nn.ModuleList, nn.Sequential)) and len(sub) >= 2:
                container_path = name
                container_module = sub
                break
        except (AttributeError, IndexError, KeyError):
            continue

    if container_module is None:
        # Generic search for any ModuleList / Sequential with >= 2 children
        for mod_name, mod in model.named_modules():
            if isinstance(mod, (nn.ModuleList, nn.Sequential)) and len(mod) >= 2:
                # Check if elements are structured blocks (composite modules)
                if all(len(list(child.children())) > 0 for child in mod):
                    container_path = mod_name
                    container_module = mod
                    break

    if container_module is None or container_path is None:
        raise ValueError(
            "Could not automatically locate sequential block container in model. "
            "Please ensure model has a standard ModuleList/Sequential block structure."
        )

    # Build SequentialBlockSpecs
    block_specs: list[SequentialBlockSpec] = []
    for i, child in enumerate(container_module):
        block_name = f"{container_path}.{i}" if container_path else str(i)
        quantizable: list[str] = []
        for rel_name, sub_m in child.named_modules():
            if isinstance(sub_m, nn.Linear):
                full_name = f"{block_name}.{rel_name}" if rel_name else block_name
                if should_quantize_module(
                    full_name,
                    sub_m,
                    effective_policy,
                    structure_contract=contract,
                ):
                    quantizable.append(rel_name)

        block_specs.append(
            SequentialBlockSpec(
                index=i,
                name=block_name,
                module=child,
                quantizable_linears=tuple(quantizable),
            )
        )

    # Classify prefix and postfix modules
    top_children = list(model.named_children())
    prefix_names: list[str] = []
    postfix_names: list[str] = []
    found_container = False

    container_root = container_path.split(".")[0]

    for name, _ in top_children:
        if name == container_root:
            found_container = True
            continue
        if not found_container:
            prefix_names.append(name)
        else:
            postfix_names.append(name)

    return SequentialPartition(
        prefix_module_names=tuple(prefix_names),
        blocks=tuple(block_specs),
        postfix_module_names=tuple(postfix_names),
    )


def capture_block_inputs(
    model: nn.Module,
    first_block: nn.Module,
    calibration_inputs: Iterable[Any],
    *,
    sample_limit: int | None = 16,
    target_device: torch.device,
) -> list[tuple[tuple[Any, ...], dict[str, Any]]]:
    """Run prefix modules and capture activations entering the first block."""
    captured: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def _pre_hook(
        module: nn.Module,
        args: tuple[Any, ...],
        kwargs: dict[str, Any] | None = None,
    ) -> None:
        del module
        if sample_limit is not None and len(captured) >= sample_limit:
            return
        # Store captured tensors on CPU to keep memory footprint minimal
        detached_args = tuple(
            a.detach().cpu() if isinstance(a, torch.Tensor) else a for a in args
        )
        detached_kwargs = {
            k: (v.detach().cpu() if isinstance(v, torch.Tensor) else v)
            for k, v in (kwargs or {}).items()
        }
        captured.append((detached_args, detached_kwargs))

    handle = first_block.register_forward_pre_hook(_pre_hook, with_kwargs=True)
    was_training = model.training
    model.eval()

    try:
        with torch.no_grad():
            for batch in iter_calibration_batches(calibration_inputs, sample_limit=sample_limit):
                batch_on_device = move_batch_to_device(batch, target_device)
                try:
                    call_model(model, batch_on_device)
                except Exception:
                    # Some models may fail forward if hooks stop execution early;
                    # if inputs are captured, proceed safely.
                    if captured:
                        break
                    raise
                if sample_limit is not None and len(captured) >= sample_limit:
                    break
    finally:
        handle.remove()
        if was_training:
            model.train()

    return captured


def quantize_single_block(
    block: nn.Module,
    block_spec: SequentialBlockSpec,
    layer_inputs: list[tuple[tuple[Any, ...], dict[str, Any]]],
    *,
    config: LayerSequentialConfig,
    target_device: torch.device,
) -> int:
    """Collect local activation statistics and quantize linear modules within a block."""
    if not block_spec.quantizable_linears:
        return 0

    method = config.method.lower()
    bits = config.bits
    group_size = config.group_size

    if method in {"awq", "gptq"} and layer_inputs:
        # Collect activation stats locally on block
        wanted = set(block_spec.quantizable_linears)
        sums: dict[str, torch.Tensor] = {}
        sq_sums: dict[str, torch.Tensor] = {}
        counts: dict[str, int] = {}
        handles = []

        def _make_hook(rel_name: str) -> Any:
            def _hook(mod: nn.Module, inp: tuple[Any, ...], _: Any) -> None:
                if not inp or not isinstance(inp[0], torch.Tensor):
                    return
                act = inp[0].detach().to(torch.float32)
                if act.ndim == 0:
                    return
                flattened = act.reshape(-1, act.shape[-1])
                in_feat = getattr(mod, "in_features", flattened.shape[-1])
                if flattened.shape[-1] != in_feat:
                    return
                sums[rel_name] = (
                    sums.get(rel_name, torch.zeros(flattened.shape[-1]))
                    + flattened.abs().sum(dim=0).cpu()
                )
                sq_sums[rel_name] = (
                    sq_sums.get(rel_name, torch.zeros(flattened.shape[-1]))
                    + flattened.square().sum(dim=0).cpu()
                )
                counts[rel_name] = counts.get(rel_name, 0) + int(flattened.shape[0])

            return _hook

        for rel_name in wanted:
            sub = _get_submodule_by_path(block, rel_name)
            if isinstance(sub, nn.Linear):
                handles.append(sub.register_forward_hook(_make_hook(rel_name)))

        try:
            with torch.no_grad():
                for args, kwargs in layer_inputs:
                    dev_args = tuple(
                        a.to(target_device) if isinstance(a, torch.Tensor) else a for a in args
                    )
                    dev_kwargs = {
                        k: (v.to(target_device) if isinstance(v, torch.Tensor) else v)
                        for k, v in kwargs.items()
                    }
                    block(*dev_args, **dev_kwargs)
        finally:
            for h in handles:
                h.remove()

        stats: dict[str, _LinearCalibrationStats] = {}
        for rel_name in wanted:
            sample_count = counts.get(rel_name, 0)
            if sample_count > 0:
                stats[rel_name] = _LinearCalibrationStats(
                    activation_abs_mean=sums[rel_name] / float(sample_count),
                    activation_hessian_diag=sq_sums[rel_name] / float(sample_count),
                    sample_count=sample_count,
                )

        # Replace linear modules with quantized modules
        quantized_count = 0
        for rel_name in block_spec.quantizable_linears:
            sub = _get_submodule_by_path(block, rel_name)
            if not isinstance(sub, nn.Linear):
                continue
            lin_stat = stats.get(rel_name)
            if method == "awq":
                replacement = _from_linear_awq(
                    sub,
                    bits=bits,
                    group_size=group_size,
                    stats=lin_stat,
                )
            else:
                replacement = _from_linear_gptq(
                    sub,
                    bits=bits,
                    group_size=group_size,
                    stats=lin_stat,
                )
            replace_submodule_in_block(block, rel_name, replacement)
            quantized_count += 1
        return quantized_count

    if method in {"adaround", "autoround"} and layer_inputs:
        wanted = set(block_spec.quantizable_linears)
        inputs_dict: dict[str, list[torch.Tensor]] = {name: [] for name in wanted}
        handles = []

        def _make_act_hook(rel_name: str) -> Any:
            def _hook(m: nn.Module, inps: tuple[Any, ...], _: Any) -> None:
                if inps and isinstance(inps[0], torch.Tensor):
                    act = inps[0].detach()
                    flat = act.reshape(-1, act.shape[-1])[:256].cpu()
                    inputs_dict[rel_name].append(flat)

            return _hook

        for rel_name in wanted:
            sub = _get_submodule_by_path(block, rel_name)
            if isinstance(sub, nn.Linear):
                handles.append(sub.register_forward_hook(_make_act_hook(rel_name)))

        try:
            with torch.no_grad():
                for args, kwargs in layer_inputs:
                    dev_args = tuple(
                        a.to(target_device) if isinstance(a, torch.Tensor) else a for a in args
                    )
                    dev_kwargs = {
                        k: (v.to(target_device) if isinstance(v, torch.Tensor) else v)
                        for k, v in kwargs.items()
                    }
                    block(*dev_args, **dev_kwargs)
        finally:
            for h in handles:
                h.remove()

        quantized_count = 0
        for rel_name in block_spec.quantizable_linears:
            sub = _get_submodule_by_path(block, rel_name)
            if not isinstance(sub, nn.Linear):
                continue
            acts = inputs_dict.get(rel_name, [])
            if acts:
                cat_acts = torch.cat(acts, dim=0).to(target_device)
            else:
                cat_acts = torch.randn(16, sub.in_features, device=target_device)
            replacement = optimize_linear_rounding(
                sub,
                cat_acts,
                bits=bits,
                group_size=group_size,
                steps=30,
            )
            replace_submodule_in_block(block, rel_name, replacement)
            quantized_count += 1
        return quantized_count

    # RTN or fallback without calibration stats
    quantized_count = 0
    for rel_name in block_spec.quantizable_linears:
        sub = _get_submodule_by_path(block, rel_name)
        if isinstance(sub, nn.Linear):
            replacement = AWQGPTQWeightOnlyLinear.from_linear(
                sub,
                bits=bits,
                group_size=group_size,
                method="rtn",
            )
            replace_submodule_in_block(block, rel_name, replacement)
            quantized_count += 1
    return quantized_count


def quantize_layer_sequential(
    model: nn.Module,
    calibration_inputs: Iterable[Any],
    *,
    config: LayerSequentialConfig | None = None,
    contract: ModelStructureContract | None = None,
    policy: QuantizationPolicy | None = None,
    block_quantizer_fn: Callable[[nn.Module, SequentialBlockSpec, list[Any]], int] | None = None,
) -> tuple[nn.Module, LayerSequentialReport]:
    """Execute layer-by-layer sequential quantization with memory offloading."""
    cfg = config or LayerSequentialConfig()
    target_dev = torch.device(
        cfg.target_device
        if (cfg.target_device == "cpu" or torch.cuda.is_available())
        else "cpu"
    )
    offload_dev = torch.device(cfg.offload_device)

    partition = discover_sequential_partition(model, contract=contract, policy=policy)
    if not partition.blocks:
        raise ValueError("No sequential blocks discovered in model.")

    # 1. Capture activations entering block 0
    # Temporarily move model or prefix to target_dev if needed
    was_training = model.training
    model.eval()

    # If model is on CPU and target is CUDA, move prefix and first block to target_dev
    first_block = partition.blocks[0].module
    model_dev = next(model.parameters(), torch.empty((), device="cpu")).device

    # Ensure model parameters are on target device for calibration capture if needed
    if model_dev != target_dev and target_dev.type == "cuda":
        model.to(target_dev)

    layer_inputs = capture_block_inputs(
        model,
        first_block,
        calibration_inputs,
        sample_limit=cfg.sample_limit,
        target_device=target_dev,
    )

    device_transitions = 0
    # Offload all blocks to offload_dev initially to reclaim memory
    if offload_dev != target_dev:
        for b in partition.blocks:
            b.module.to(offload_dev)
            device_transitions += 1

    total_quantized_linears = 0
    processed_blocks = 0

    # 2. Iterate block by block
    for block_spec in partition.blocks:
        block = block_spec.module
        if offload_dev != target_dev:
            block.to(target_dev)
            device_transitions += 1

        if block_quantizer_fn is not None:
            q_count = block_quantizer_fn(block, block_spec, layer_inputs)
        else:
            q_count = quantize_single_block(
                block,
                block_spec,
                layer_inputs,
                config=cfg,
                target_device=target_dev,
            )
        total_quantized_linears += q_count
        processed_blocks += 1

        # Propagate activations through quantized block to get inputs for next block
        next_inputs: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        with torch.no_grad():
            for args, kwargs in layer_inputs:
                dev_args = tuple(
                    a.to(target_dev) if isinstance(a, torch.Tensor) else a for a in args
                )
                dev_kwargs = {
                    k: (v.to(target_dev) if isinstance(v, torch.Tensor) else v)
                    for k, v in kwargs.items()
                }
                out = block(*dev_args, **dev_kwargs)
                if isinstance(out, torch.Tensor):
                    next_args = (out.detach().to(offload_dev),)
                elif isinstance(out, (tuple, list)):
                    # Transformer blocks typically return (hidden_states, ...)
                    next_args = (out[0].detach().to(offload_dev),)
                else:
                    next_args = (out,)
                next_inputs.append((next_args, kwargs))

        layer_inputs = next_inputs

        if offload_dev != target_dev:
            block.to(offload_dev)
            device_transitions += 1

        if cfg.clean_memory_each_block:
            gc.collect()
            if torch.cuda.is_available() and target_dev.type == "cuda":
                torch.cuda.empty_cache()

    if was_training:
        model.train()

    peak_memory: int | None = None
    if torch.cuda.is_available() and target_dev.type == "cuda":
        peak_memory = torch.cuda.max_memory_allocated(target_dev)

    report = LayerSequentialReport(
        total_blocks=len(partition.blocks),
        quantized_blocks=processed_blocks,
        quantized_linear_count=total_quantized_linears,
        prefix_modules=partition.prefix_module_names,
        postfix_modules=partition.postfix_module_names,
        device_transitions=device_transitions,
        peak_memory_bytes=peak_memory,
    )

    return model, report


__all__ = [
    "LayerSequentialConfig",
    "LayerSequentialReport",
    "SequentialBlockSpec",
    "SequentialPartition",
    "capture_block_inputs",
    "discover_sequential_partition",
    "quantize_layer_sequential",
    "quantize_single_block",
    "replace_submodule_in_block",
]
