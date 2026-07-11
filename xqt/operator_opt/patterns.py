"""FX and torch.export candidate pattern discovery for operator optimization."""

from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import Any, Iterable, Optional

import torch
from torch import fx, nn


_PATTERN_TO_BACKEND = {
    "bias_gelu": "triton",
    "swiglu": "triton",
    "rmsnorm": "triton",
    "rmsnorm_residual": "triton",
    "rope": "triton",
    "attention": "tilelang",
    "dequant_gemm": "tilelang",
    "dequant_gemm_epilogue": "tilelang",
    "qdq_epilogue": "deployment_backend",
    "weight_only_matmul_epilogue": "tilelang",
    "fp8_scale_cast_matmul_epilogue": "tilelang",
    "bias_silu": "cutile",
    "gemm_epilogue": "cutlass",
    "grouped_gemm": "cutlass",
}

_RESHAPE_SUFFIXES = (
    "reshape.default",
    "view.default",
)
_ADD_SUFFIXES = (
    "add",
    "add.Tensor",
    "add_.Tensor",
)
_ACTIVATION_SUFFIXES = (
    "gelu.default",
    "relu.default",
    "silu.default",
)
_MUL_SUFFIXES = (
    "mul",
    "mul.Tensor",
)
_POW_SUFFIXES = (
    "pow",
    "pow.Tensor_Scalar",
)
_MEAN_SUFFIXES = (
    "mean",
    "mean.dim",
)
_RSQRT_SUFFIXES = (
    "rsqrt",
    "rsqrt.default",
)
_SLICE_SUFFIXES = (
    "getitem",
    "slice.Tensor",
)
_STACK_SUFFIXES = (
    "stack",
    "stack.default",
)
_FLATTEN_SUFFIXES = (
    "flatten",
    "flatten.using_ints",
)


def _is_target(node: fx.Node, suffix: str) -> bool:
    return _normalize_target(node.target).endswith(suffix)


def _matches_any_suffix(node: fx.Node, suffixes: Iterable[str]) -> bool:
    target_name = _normalize_target(node.target)
    return any(target_name.endswith(suffix) for suffix in suffixes)


def _find_node_users(nodes: Iterable[fx.Node], value: fx.Node) -> list[fx.Node]:
    return [node for node in nodes if value in node.all_input_nodes]


def _subclass_key(node: fx.Node) -> Optional[str]:
    if not _is_target(node, "export.access_subclass_inner_tensor.default"):
        return None
    if len(node.args) < 2 or not isinstance(node.args[1], str):
        return None
    return str(node.args[1])


def _ancestor_subclass_keys(node: fx.Node) -> set[str]:
    keys: set[str] = set()
    pending = list(node.all_input_nodes)
    visited: set[fx.Node] = set()
    while pending:
        current = pending.pop()
        if current in visited:
            continue
        visited.add(current)
        key = _subclass_key(current)
        if key is not None:
            keys.add(key)
        pending.extend(current.all_input_nodes)
    return keys


def _ancestor_support_nodes(
    nodes: Iterable[fx.Node],
    anchor: fx.Node,
    keys: set[str],
) -> list[fx.Node]:
    if not keys:
        return []
    pending = list(anchor.all_input_nodes)
    ancestors: set[fx.Node] = set()
    while pending:
        current = pending.pop()
        if current in ancestors:
            continue
        ancestors.add(current)
        pending.extend(current.all_input_nodes)
    return [
        node
        for node in nodes
        if node in ancestors and _subclass_key(node) in keys
    ]


def _find_first_user_with_suffix(
    nodes: Iterable[fx.Node],
    value: fx.Node,
    suffix: str,
) -> Optional[fx.Node]:
    for node in nodes:
        if value in node.all_input_nodes and _is_target(node, suffix):
            return node
    return None


def _find_qdq_epilogue_nodes(
    nodes: list[fx.Node],
    linear_node: fx.Node,
) -> Optional[list[fx.Node]]:
    div_node = _find_first_user_with_suffix(nodes, linear_node, "div.Tensor")
    if div_node is None:
        return None
    round_node = _find_first_user_with_suffix(nodes, div_node, "round.default")
    if round_node is None:
        return None
    clamp_node = _find_first_user_with_suffix(nodes, round_node, "clamp.default")
    if clamp_node is None:
        return None
    requant_mul = _find_first_user_with_suffix(nodes, clamp_node, "mul.Tensor")
    if requant_mul is None:
        return None
    epilogue = next(
        (
            node
            for node in nodes
            if requant_mul in node.all_input_nodes
            and (
                _is_target(node, "add.Tensor")
                or _is_target(node, "gelu.default")
                or _is_target(node, "relu.default")
                or _is_target(node, "silu.default")
            )
        ),
        None,
    )
    if epilogue is None:
        return None
    return [div_node, round_node, clamp_node, requant_mul, epilogue]


def _find_user_path(
    nodes: list[fx.Node],
    start: fx.Node,
    *,
    target_suffixes: Iterable[str],
    passthrough_suffixes: Iterable[str] = (),
) -> Optional[list[fx.Node]]:
    pending: list[tuple[fx.Node, list[fx.Node]]] = [(start, [])]
    visited: set[fx.Node] = {start}
    while pending:
        current, path = pending.pop(0)
        for user in _find_node_users(nodes, current):
            if user in visited:
                continue
            visited.add(user)
            next_path = [*path, user]
            if _matches_any_suffix(user, target_suffixes):
                return next_path
            if _matches_any_suffix(user, passthrough_suffixes):
                pending.append((user, next_path))
    return None


def _find_epilogue_path(
    nodes: list[fx.Node],
    start: fx.Node,
    *,
    passthrough_suffixes: Iterable[str] = (),
) -> list[fx.Node]:
    chain: list[fx.Node] = []
    current = start
    add_path = _find_user_path(
        nodes,
        current,
        target_suffixes=_ADD_SUFFIXES,
        passthrough_suffixes=passthrough_suffixes,
    )
    if add_path is not None:
        chain.extend(add_path)
        current = add_path[-1]
    activation_path = _find_user_path(
        nodes,
        current,
        target_suffixes=_ACTIVATION_SUFFIXES,
        passthrough_suffixes=passthrough_suffixes,
    )
    if activation_path is not None:
        chain.extend(activation_path)
    return chain


def _find_rmsnorm_residual_nodes(anchor: fx.Node) -> Optional[list[fx.Node]]:
    if not _matches_any_suffix(anchor, _MUL_SUFFIXES):
        return None
    rms_mul = next(
        (
            source
            for source in anchor.all_input_nodes
            if _matches_any_suffix(source, _MUL_SUFFIXES)
        ),
        None,
    )
    if rms_mul is None:
        return None
    merged_node = next(
        (
            source
            for source in rms_mul.all_input_nodes
            if _matches_any_suffix(source, _ADD_SUFFIXES)
        ),
        None,
    )
    rsqrt_node = next(
        (
            source
            for source in rms_mul.all_input_nodes
            if _matches_any_suffix(source, _RSQRT_SUFFIXES)
        ),
        None,
    )
    if merged_node is None or rsqrt_node is None:
        return None
    eps_add = next(
        (
            source
            for source in rsqrt_node.all_input_nodes
            if _matches_any_suffix(source, _ADD_SUFFIXES)
        ),
        None,
    )
    if eps_add is None:
        return None
    mean_node = next(
        (
            source
            for source in eps_add.all_input_nodes
            if _matches_any_suffix(source, _MEAN_SUFFIXES)
        ),
        None,
    )
    if mean_node is None:
        return None
    pow_node = next(
        (
            source
            for source in mean_node.all_input_nodes
            if _matches_any_suffix(source, _POW_SUFFIXES)
        ),
        None,
    )
    if pow_node is None or merged_node not in pow_node.all_input_nodes:
        return None
    return [merged_node, pow_node, mean_node, eps_add, rsqrt_node, rms_mul, anchor]


def _find_rmsnorm_nodes(anchor: fx.Node) -> Optional[list[fx.Node]]:
    if not _matches_any_suffix(anchor, _MUL_SUFFIXES):
        return None
    rms_mul = next(
        (
            source
            for source in anchor.all_input_nodes
            if _matches_any_suffix(source, _MUL_SUFFIXES)
        ),
        None,
    )
    if rms_mul is None:
        return None
    x_node = next(
        (
            source
            for source in rms_mul.all_input_nodes
            if not _matches_any_suffix(source, _RSQRT_SUFFIXES)
        ),
        None,
    )
    rsqrt_node = next(
        (
            source
            for source in rms_mul.all_input_nodes
            if _matches_any_suffix(source, _RSQRT_SUFFIXES)
        ),
        None,
    )
    if x_node is None or rsqrt_node is None:
        return None
    eps_add = next(
        (
            source
            for source in rsqrt_node.all_input_nodes
            if _matches_any_suffix(source, _ADD_SUFFIXES)
        ),
        None,
    )
    if eps_add is None:
        return None
    mean_node = next(
        (
            source
            for source in eps_add.all_input_nodes
            if _matches_any_suffix(source, _MEAN_SUFFIXES)
        ),
        None,
    )
    if mean_node is None:
        return None
    pow_node = next(
        (
            source
            for source in mean_node.all_input_nodes
            if _matches_any_suffix(source, _POW_SUFFIXES)
        ),
        None,
    )
    if pow_node is None or x_node not in pow_node.all_input_nodes:
        return None
    if any(_matches_any_suffix(source, _ADD_SUFFIXES) for source in x_node.all_input_nodes):
        return None
    return [x_node, pow_node, mean_node, eps_add, rsqrt_node, rms_mul, anchor]


def _find_rope_nodes(anchor: fx.Node) -> Optional[list[fx.Node]]:
    if not _matches_any_suffix(anchor, _FLATTEN_SUFFIXES):
        return None
    stack_node = next(
        (
            source
            for source in anchor.all_input_nodes
            if _matches_any_suffix(source, _STACK_SUFFIXES)
        ),
        None,
    )
    if stack_node is None:
        return None
    rotation_nodes = list(stack_node.all_input_nodes)
    if len(rotation_nodes) != 2:
        return None
    if not any(_matches_any_suffix(node, ("sub", "sub.Tensor")) for node in rotation_nodes):
        return None
    if not any(_matches_any_suffix(node, _ADD_SUFFIXES) for node in rotation_nodes):
        return None
    mul_nodes = [
        source
        for node in rotation_nodes
        for source in node.all_input_nodes
        if _matches_any_suffix(source, _MUL_SUFFIXES)
    ]
    if len(mul_nodes) != 4:
        return None
    slice_nodes: list[fx.Node] = []
    for mul_node in mul_nodes:
        slice_nodes.extend(
            source
            for source in mul_node.all_input_nodes
            if _matches_any_suffix(source, _SLICE_SUFFIXES)
        )
    unique_slice_nodes = list(dict.fromkeys(slice_nodes))
    if len(unique_slice_nodes) != 2:
        return None
    if not all(slice_nodes.count(node) == 2 for node in unique_slice_nodes):
        return None
    return [*unique_slice_nodes, *mul_nodes, *rotation_nodes, stack_node, anchor]


@dataclass
class OperatorPatternCandidate:
    """One discovered operator optimization candidate pattern."""

    pattern: str
    source: str
    anchor: str
    node_names: list[str]
    shape: list[list[int]]
    dtype: list[str]
    device: list[str]
    estimated_memory_io: Optional[int]
    estimated_kernel_count: int
    recommended_backend: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "pattern": self.pattern,
            "source": self.source,
            "anchor": self.anchor,
            "node_names": list(self.node_names),
            "shape": [list(item) for item in self.shape],
            "dtype": list(self.dtype),
            "device": list(self.device),
            "estimated_memory_io": self.estimated_memory_io,
            "estimated_kernel_count": self.estimated_kernel_count,
            "recommended_backend": self.recommended_backend,
        }


def _normalize_target(target: object) -> str:
    if isinstance(target, str):
        return target
    if hasattr(target, "__name__"):
        module_name = getattr(target, "__module__", "")
        qualname = getattr(target, "__name__", str(target))
        return f"{module_name}.{qualname}" if module_name else qualname
    return str(target)


def _tensor_meta_to_lists(node: fx.Node) -> tuple[list[list[int]], list[str], list[str], Optional[int]]:
    tensor_meta = node.meta.get("tensor_meta")
    if tensor_meta is None:
        return [], [], [], None
    metas = tensor_meta if isinstance(tensor_meta, (list, tuple)) else [tensor_meta]
    shapes: list[list[int]] = []
    dtypes: list[str] = []
    devices: list[str] = []
    estimated_memory_io = 0
    for meta in metas:
        shape = [int(dim) for dim in getattr(meta, "shape", [])]
        shapes.append(shape)
        dtypes.append(str(getattr(meta, "dtype", "")))
        device = getattr(meta, "device", None)
        devices.append(str(device) if device is not None else "")
        numel = 1
        for dim in shape:
            numel *= dim
        estimated_memory_io += int(numel)
    return shapes, dtypes, devices, estimated_memory_io


def _node_list_to_names(nodes: Iterable[fx.Node]) -> list[str]:
    return [node.name for node in nodes]


def _build_candidate(
    pattern: str,
    source: str,
    anchor: fx.Node,
    nodes: list[fx.Node],
    *,
    estimated_kernel_count: int,
) -> OperatorPatternCandidate:
    shape, dtype, device, estimated_memory_io = _tensor_meta_to_lists(anchor)
    return OperatorPatternCandidate(
        pattern=pattern,
        source=source,
        anchor=anchor.name,
        node_names=_node_list_to_names(nodes),
        shape=shape,
        dtype=dtype,
        device=device,
        estimated_memory_io=estimated_memory_io,
        estimated_kernel_count=estimated_kernel_count,
        recommended_backend=_PATTERN_TO_BACKEND[pattern],
    )


def _match_fx_patterns(graph_module: fx.GraphModule) -> list[OperatorPatternCandidate]:
    candidates: list[OperatorPatternCandidate] = []
    nodes = list(graph_module.graph.nodes)
    for index, node in enumerate(nodes):
        target_name = _normalize_target(node.target)
        if node.op == "call_function" and target_name.endswith("gelu"):
            previous = nodes[index - 1] if index > 0 else None
            if previous is not None and previous.op == "call_function":
                previous_target = _normalize_target(previous.target)
                if previous.target is operator.add or previous_target.endswith("add"):
                    candidates.append(
                        _build_candidate(
                            "bias_gelu",
                            "fx",
                            node,
                            [previous, node],
                            estimated_kernel_count=2,
                        )
                    )
        if node.op == "call_function" and target_name.endswith("mul"):
            rmsnorm_nodes = _find_rmsnorm_nodes(node)
            if rmsnorm_nodes is not None:
                candidates.append(
                    _build_candidate(
                        "rmsnorm",
                        "fx",
                        node,
                        rmsnorm_nodes,
                        estimated_kernel_count=len(rmsnorm_nodes),
                    )
                )
            rmsnorm_nodes = _find_rmsnorm_residual_nodes(node)
            if rmsnorm_nodes is not None:
                candidates.append(
                    _build_candidate(
                        "rmsnorm_residual",
                        "fx",
                        node,
                        rmsnorm_nodes,
                        estimated_kernel_count=len(rmsnorm_nodes),
                    )
                )
            mul_args = [arg for arg in node.args if isinstance(arg, fx.Node)]
            silu_node = next(
                (
                    arg
                    for arg in mul_args
                    if arg.op == "call_function"
                    and _normalize_target(arg.target).endswith("silu")
                ),
                None,
            )
            if silu_node is not None:
                candidates.append(
                    _build_candidate(
                        "swiglu",
                        "fx",
                        node,
                        [silu_node, node],
                        estimated_kernel_count=2,
                    )
                )
        if node.op in {"call_function", "call_method"} and _matches_any_suffix(
            node,
            _FLATTEN_SUFFIXES,
        ):
            rope_nodes = _find_rope_nodes(node)
            if rope_nodes is not None:
                candidates.append(
                    _build_candidate(
                        "rope",
                        "fx",
                        node,
                        rope_nodes,
                        estimated_kernel_count=len(rope_nodes),
                    )
                )
        if node.op == "call_function" and (
            target_name.endswith("linear") or target_name.endswith("matmul")
        ):
            dequant_nodes = [
                arg
                for arg in node.args
                if isinstance(arg, fx.Node)
                and _normalize_target(arg.target).endswith("mul")
                and any(
                    isinstance(source, fx.Node)
                    and _normalize_target(source.target).endswith("to")
                    for source in arg.all_input_nodes
                )
            ]
            if dequant_nodes:
                candidates.append(
                    _build_candidate(
                        "dequant_gemm",
                        "fx",
                        node,
                        [*dequant_nodes, node],
                        estimated_kernel_count=3,
                    )
                )
        if node.op == "call_module":
            module = graph_module.get_submodule(str(node.target))
            try:
                from xqt import nn as xqt_nn

                is_xqt_attention = isinstance(module, xqt_nn.Attention)
            except Exception:
                is_xqt_attention = False
            if isinstance(module, nn.MultiheadAttention) or is_xqt_attention:
                candidates.append(
                    _build_candidate(
                        "attention",
                        "fx",
                        node,
                        [node],
                        estimated_kernel_count=1,
                    )
                )
        if node.op == "call_function" and (
            target_name.endswith("gelu")
            or target_name.endswith("relu")
            or target_name.endswith("silu")
            or target_name.endswith("add")
        ):
            previous = nodes[index - 1] if index > 0 else None
            if previous is None:
                continue
            if not _normalize_target(previous.target).endswith("mul"):
                continue
            clamp_node = next(
                (
                    source
                    for source in previous.all_input_nodes
                    if isinstance(source, fx.Node)
                    and _normalize_target(source.target).endswith("clamp")
                ),
                None,
            )
            if clamp_node is None:
                continue
            round_node = next(
                (
                    source
                    for source in clamp_node.all_input_nodes
                    if isinstance(source, fx.Node)
                    and _normalize_target(source.target).endswith("round")
                ),
                None,
            )
            if round_node is None:
                continue
            div_node = next(
                (
                    source
                    for source in round_node.all_input_nodes
                    if isinstance(source, fx.Node)
                    and _normalize_target(source.target).endswith("truediv")
                ),
                None,
            )
            if div_node is None:
                continue
            linear_node = next(
                (
                    source
                    for source in div_node.all_input_nodes
                    if isinstance(source, fx.Node)
                    and _normalize_target(source.target).endswith("linear")
                ),
                None,
            )
            if linear_node is None:
                continue
            candidates.append(
                _build_candidate(
                    "qdq_epilogue",
                    "fx",
                    node,
                    [linear_node, div_node, round_node, clamp_node, previous, node],
                    estimated_kernel_count=6,
                )
            )
    return candidates


def scan_fx_candidates(
    model: nn.Module,
    example_input: Any,
) -> list[OperatorPatternCandidate]:
    """Scan a symbolic FX graph for operator optimization candidates."""

    traced = fx.symbolic_trace(model.eval())
    try:
        from torch.fx.passes.shape_prop import ShapeProp

        args = example_input if isinstance(example_input, tuple) else (example_input,)
        ShapeProp(traced).propagate(*args)
    except Exception:
        pass
    return _match_fx_patterns(traced)


def _match_export_patterns(graph_module: fx.GraphModule) -> list[OperatorPatternCandidate]:
    candidates: list[OperatorPatternCandidate] = []
    nodes = list(graph_module.graph.nodes)
    for index, node in enumerate(nodes):
        target_name = _normalize_target(node.target)
        if node.op == "call_function" and target_name.endswith("aten.gelu.default"):
            previous = nodes[index - 1] if index > 0 else None
            if previous is not None and _normalize_target(previous.target).endswith("aten.add.Tensor"):
                candidates.append(
                    _build_candidate(
                        "bias_gelu",
                        "torch_export",
                        node,
                        [previous, node],
                        estimated_kernel_count=2,
                    )
                )
        if node.op == "call_function" and target_name.endswith("aten.mul.Tensor"):
            rmsnorm_nodes = _find_rmsnorm_nodes(node)
            if rmsnorm_nodes is not None:
                candidates.append(
                    _build_candidate(
                        "rmsnorm",
                        "torch_export",
                        node,
                        rmsnorm_nodes,
                        estimated_kernel_count=len(rmsnorm_nodes),
                    )
                )
            rmsnorm_nodes = _find_rmsnorm_residual_nodes(node)
            if rmsnorm_nodes is not None:
                candidates.append(
                    _build_candidate(
                        "rmsnorm_residual",
                        "torch_export",
                        node,
                        rmsnorm_nodes,
                        estimated_kernel_count=len(rmsnorm_nodes),
                    )
                )
            previous = nodes[index - 1] if index > 0 else None
            if previous is not None and _normalize_target(previous.target).endswith("aten.silu.default"):
                candidates.append(
                    _build_candidate(
                        "swiglu",
                        "torch_export",
                        node,
                        [previous, node],
                        estimated_kernel_count=2,
                    )
                )
        if node.op == "call_function" and _matches_any_suffix(node, _FLATTEN_SUFFIXES):
            rope_nodes = _find_rope_nodes(node)
            if rope_nodes is not None:
                candidates.append(
                    _build_candidate(
                        "rope",
                        "torch_export",
                        node,
                        rope_nodes,
                        estimated_kernel_count=len(rope_nodes),
                    )
                )
        if node.op == "call_function" and target_name.endswith("aten.scaled_dot_product_attention.default"):
            candidates.append(
                _build_candidate(
                    "attention",
                    "torch_export",
                    node,
                    [node],
                    estimated_kernel_count=1,
                )
            )
        if node.op == "call_function" and (
            target_name.endswith("aten.linear.default")
            or target_name.endswith("aten.matmul.default")
            or target_name.endswith("aten.mm.default")
        ):
            dequant_nodes = [
                arg
                for arg in node.args
                if isinstance(arg, fx.Node)
                and _is_target(arg, "aten.mul.Tensor")
                and any(
                    isinstance(source, fx.Node) and _is_target(source, "aten.to.dtype")
                    for source in arg.all_input_nodes
                )
            ]
            if dequant_nodes:
                candidates.append(
                    _build_candidate(
                        "dequant_gemm",
                        "torch_export",
                        node,
                        [*dequant_nodes, node],
                        estimated_kernel_count=3,
                    )
                )
                epilogue_nodes = _find_epilogue_path(
                    nodes,
                    node,
                    passthrough_suffixes=_RESHAPE_SUFFIXES,
                )
                if epilogue_nodes:
                    candidates.append(
                        _build_candidate(
                            "dequant_gemm_epilogue",
                            "torch_export",
                            epilogue_nodes[-1],
                            [*dequant_nodes, node, *epilogue_nodes],
                            estimated_kernel_count=3 + len(epilogue_nodes),
                        )
                    )
                qdq_nodes = _find_qdq_epilogue_nodes(nodes, node)
                if qdq_nodes is not None:
                    candidates.append(
                        _build_candidate(
                            "qdq_epilogue",
                            "torch_export",
                            qdq_nodes[-1],
                            [node, *qdq_nodes],
                            estimated_kernel_count=1 + len(qdq_nodes),
                        )
                    )
            ancestry_keys = _ancestor_subclass_keys(node)
            if "int_data" in ancestry_keys:
                scale_path = _find_user_path(
                    nodes,
                    node,
                    target_suffixes=("mul.Tensor",),
                )
                if scale_path is not None:
                    epilogue_nodes = _find_epilogue_path(
                        nodes,
                        scale_path[-1],
                        passthrough_suffixes=_RESHAPE_SUFFIXES,
                    )
                    if epilogue_nodes:
                        support_nodes = _ancestor_support_nodes(
                            nodes,
                            node,
                            {"int_data", "scale"},
                        )
                        candidates.append(
                            _build_candidate(
                                "weight_only_matmul_epilogue",
                                "torch_export",
                                epilogue_nodes[-1],
                                [*support_nodes, node, *scale_path, *epilogue_nodes],
                                estimated_kernel_count=1 + len(scale_path) + len(epilogue_nodes),
                            )
                        )
            if "qdata" in ancestry_keys and "scale" in ancestry_keys:
                epilogue_nodes = _find_epilogue_path(
                    nodes,
                    node,
                    passthrough_suffixes=_RESHAPE_SUFFIXES,
                )
                if epilogue_nodes:
                    support_nodes = _ancestor_support_nodes(
                        nodes,
                        node,
                        {"qdata", "scale"},
                    )
                    candidates.append(
                        _build_candidate(
                            "fp8_scale_cast_matmul_epilogue",
                            "torch_export",
                            epilogue_nodes[-1],
                            [*support_nodes, node, *epilogue_nodes],
                            estimated_kernel_count=1 + len(epilogue_nodes),
                        )
                    )
    return candidates


def scan_export_candidates(
    model: nn.Module,
    example_input: Any,
) -> list[OperatorPatternCandidate]:
    """Scan a torch.export graph for operator optimization candidates."""

    if not hasattr(torch, "export"):
        raise RuntimeError("torch.export is not available in the current PyTorch build")
    args = example_input if isinstance(example_input, tuple) else (example_input,)
    exported = torch.export.export(model.eval(), args)
    return _match_export_patterns(exported.module())


def summarize_candidate_report(
    candidates: Iterable[OperatorPatternCandidate],
) -> dict[str, Any]:
    """Build a stable candidate report summary."""

    items = [candidate.to_dict() for candidate in candidates]
    return {
        "candidate_count": len(items),
        "patterns": [item["pattern"] for item in items],
        "recommended_backends": [item["recommended_backend"] for item in items],
        "candidates": items,
    }


__all__ = [
    "OperatorPatternCandidate",
    "scan_export_candidates",
    "scan_fx_candidates",
    "summarize_candidate_report",
]
