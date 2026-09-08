"""Model structure contract (declarative model-family description).

One contract per model family declares: component grouping (with roles),
merged projections (qkv / gate_up) and the declarative checkpoint weight
mapping. Quantizer component plans, prune candidates and block-level
materialization consume this contract instead of grepping module names.

This is the XQT counterpart of SGLang ``models/*.py`` structure + weight_loader
declarations. It is a structure declaration only: it never selects engines,
never owns forward semantics and never rewrites the model.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Iterable, Literal, Mapping, Protocol, Sequence

from xqt.core.base import XQTConfigError

MODEL_STRUCTURE_CONTRACT_SCHEMA_VERSION = 1

ComponentRole = Literal[
    "attention",
    "ffn",
    "norm",
    "router",
    "expert",
    "embedding",
    "head",
    "backbone",
    "encoder",
    "cross_attention",
    "diffusion_component",
    "other",
]

# 与 xqt.kernels.nn.fixtures.component_grouping 的分组词表保持一致;
# 该 helper 是本词表的启发式派生源, 禁止两表漂移 (回归测试锁定).
COMPONENT_ROLES: tuple[str, ...] = (
    "attention",
    "ffn",
    "norm",
    "router",
    "expert",
    "embedding",
    "head",
    "backbone",
    "encoder",
    "cross_attention",
    "diffusion_component",
    "other",
)

MODEL_FAMILY_NAMES: tuple[str, ...] = (
    "transformer",
    "vit",
    "detection",
    "llm",
    "diffusion",
    "moe",
    "multimodal",
    "convnet",
    "unknown",
)

WEIGHT_MAPPING_KINDS: tuple[str, ...] = (
    "direct",        # checkpoint 参数名 -> 模型内参数, 一对一
    "merged_split",  # checkpoint 单个 merged 权重 -> 模型内 merged 模块 (拆分见 MergedProjectionSpec)
    "stacked_slice", # checkpoint 多个参数 -> 模型内一个 stacked 参数
)

# keep_high_precision 是 xqt.auto.precision_suggestion 已有词表;
# 其他组件不写 precision_hint, 不在契约里发明第二种策略语法.
PRECISION_HINTS: tuple[str, ...] = ("keep_high_precision",)


@dataclass(frozen=True, slots=True)
class MergedProjectionSpec:
    """One physical Linear carrying several logical projections.

    ``parts`` 与 ``split_out_features`` 等长且非空, 例如
    ``MergedProjectionSpec("blocks.0.attn.qkv_proj", ("q", "k", "v"), (1024, 512, 1024))``.
    契约只声明拆分事实; 合并/拆分动作由消费方 (weight loading, block fusion)
    按需执行.
    """

    module_path: str
    parts: tuple[str, ...]
    split_out_features: tuple[int, ...]

    def __post_init__(self) -> None:
        if not str(self.module_path).strip():
            raise XQTConfigError(
                "MergedProjectionSpec.module_path must be a non-empty str"
            )
        if len(self.parts) < 2:
            raise XQTConfigError(
                "MergedProjectionSpec.parts must name at least two logical projections"
            )
        if len(self.parts) != len(self.split_out_features):
            raise XQTConfigError(
                "MergedProjectionSpec.split_out_features must match parts length"
            )
        if any(int(dim) <= 0 for dim in self.split_out_features):
            raise XQTConfigError(
                "MergedProjectionSpec.split_out_features entries must be positive ints"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "module_path": self.module_path,
            "parts": list(self.parts),
            "split_out_features": [int(dim) for dim in self.split_out_features],
        }


@dataclass(frozen=True, slots=True)
class ComponentSpec:
    """One component group of the model, declared by module paths."""

    role: str
    paths: tuple[str, ...]
    precision_hint: str | None = None
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.role not in COMPONENT_ROLES:
            raise XQTConfigError(
                f"ComponentSpec.role must be one of {COMPONENT_ROLES}; got {self.role!r}"
            )
        if not self.paths or any(not str(path).strip() for path in self.paths):
            raise XQTConfigError(
                "ComponentSpec.paths must be a non-empty sequence of non-empty module paths"
            )
        if self.precision_hint is not None and self.precision_hint not in PRECISION_HINTS:
            raise XQTConfigError(
                f"ComponentSpec.precision_hint must be one of {PRECISION_HINTS}; "
                f"got {self.precision_hint!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "paths": list(self.paths),
            "precision_hint": self.precision_hint,
            "notes": list(self.notes),
        }


@dataclass(frozen=True, slots=True)
class WeightMappingEntry:
    """One declarative checkpoint -> model parameter mapping.

    对标 SGLang per-model ``weight_loader``: checkpoint 参数名到组件路径的
    显式映射. 通配符与模式展开是消费方能力, 契约保持字面声明 (显式优于隐式).
    """

    checkpoint_name: str
    component_path: str
    kind: str = "direct"

    def __post_init__(self) -> None:
        if not str(self.checkpoint_name).strip():
            raise XQTConfigError(
                "WeightMappingEntry.checkpoint_name must be a non-empty str"
            )
        if not str(self.component_path).strip():
            raise XQTConfigError(
                "WeightMappingEntry.component_path must be a non-empty str"
            )
        if self.kind not in WEIGHT_MAPPING_KINDS:
            raise XQTConfigError(
                f"WeightMappingEntry.kind must be one of {WEIGHT_MAPPING_KINDS}; "
                f"got {self.kind!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_name": self.checkpoint_name,
            "component_path": self.component_path,
            "kind": self.kind,
        }


@dataclass(frozen=True, slots=True)
class ModelStructureContract:
    """Declarative structure contract for one model family.

    合法性: 组件路径与 merged projection 路径必须存在于目标模型
    (用 :func:`structure_contract_mismatches` 校验, 消费方决定报错时机).
    契约不覆盖全部模块也不是错误; 未声明模块归 "other".
    """

    family: str
    components: tuple[ComponentSpec, ...]
    merged_projections: tuple[MergedProjectionSpec, ...] = ()
    weight_mapping: tuple[WeightMappingEntry, ...] = ()
    schema_version: int = MODEL_STRUCTURE_CONTRACT_SCHEMA_VERSION
    topology_fingerprint: str | None = None

    def __post_init__(self) -> None:
        if self.family not in MODEL_FAMILY_NAMES:
            raise XQTConfigError(
                f"ModelStructureContract.family must be one of {MODEL_FAMILY_NAMES}; "
                f"got {self.family!r}"
            )
        if not self.components:
            raise XQTConfigError(
                "ModelStructureContract.components must be a non-empty tuple"
            )
        seen_roles: set[str] = set()
        seen_paths: set[str] = set()
        for component in self.components:
            if component.role in seen_roles:
                raise XQTConfigError(
                    f"ModelStructureContract has duplicate role: {component.role!r}"
                )
            seen_roles.add(component.role)
            for path in component.paths:
                if path in seen_paths:
                    raise XQTConfigError(
                        f"ModelStructureContract declares module path twice: {path!r}"
                    )
                seen_paths.add(path)
        seen_merged: set[str] = set()
        for merged in self.merged_projections:
            if merged.module_path in seen_merged:
                raise XQTConfigError(
                    f"ModelStructureContract has duplicate merged projection: "
                    f"{merged.module_path!r}"
                )
            seen_merged.add(merged.module_path)

    def component_by_role(self, role: str) -> ComponentSpec | None:
        for component in self.components:
            if component.role == role:
                return component
        return None

    def role_for_path(self, path: str) -> str | None:
        """Return the declared role for a module path, if any."""
        for component in self.components:
            if path in component.paths:
                return component.role
        return None

    def paths_for_role(self, role: str) -> tuple[str, ...]:
        """Return all declared module paths for a role."""
        comp = self.component_by_role(role)
        return comp.paths if comp is not None else ()

    def with_topology_fingerprint(
        self, fingerprint: str | None
    ) -> "ModelStructureContract":
        """Return a copy of this contract bound to a specific topology fingerprint."""
        return ModelStructureContract(
            family=self.family,
            components=self.components,
            merged_projections=self.merged_projections,
            weight_mapping=self.weight_mapping,
            schema_version=self.schema_version,
            topology_fingerprint=fingerprint,
        )

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "schema_version": self.schema_version,
            "family": self.family,
            "components": [component.to_dict() for component in self.components],
            "merged_projections": [merged.to_dict() for merged in self.merged_projections],
            "weight_mapping": [entry.to_dict() for entry in self.weight_mapping],
        }
        if self.topology_fingerprint is not None:
            data["topology_fingerprint"] = self.topology_fingerprint
        return data

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ModelStructureContract":
        if payload.get("schema_version") != MODEL_STRUCTURE_CONTRACT_SCHEMA_VERSION:
            raise XQTConfigError(
                "ModelStructureContract.schema_version must be "
                f"{MODEL_STRUCTURE_CONTRACT_SCHEMA_VERSION}; "
                f"got {payload.get('schema_version')!r}"
            )
        try:
            components = tuple(
                ComponentSpec(
                    role=str(item["role"]),
                    paths=tuple(str(path) for path in item["paths"]),
                    precision_hint=(
                        None if item.get("precision_hint") is None
                        else str(item["precision_hint"])
                    ),
                    notes=tuple(str(note) for note in item.get("notes", ())),
                )
                for item in payload["components"]
            )
            merged_projections = tuple(
                MergedProjectionSpec(
                    module_path=str(item["module_path"]),
                    parts=tuple(str(part) for part in item["parts"]),
                    split_out_features=tuple(int(dim) for dim in item["split_out_features"]),
                )
                for item in payload.get("merged_projections", ())
            )
            weight_mapping = tuple(
                WeightMappingEntry(
                    checkpoint_name=str(item["checkpoint_name"]),
                    component_path=str(item["component_path"]),
                    kind=str(item.get("kind", "direct")),
                )
                for item in payload.get("weight_mapping", ())
            )
            return cls(
                family=str(payload["family"]),
                components=components,
                merged_projections=merged_projections,
                weight_mapping=weight_mapping,
                topology_fingerprint=(
                    str(payload["topology_fingerprint"])
                    if payload.get("topology_fingerprint") is not None
                    else None
                ),
            )
        except KeyError as exc:
            raise XQTConfigError(
                f"ModelStructureContract payload missing field: {exc.args[0]}"
            ) from exc
        except (TypeError, ValueError) as exc:
            raise XQTConfigError(
                f"ModelStructureContract payload invalid: {exc}"
            ) from exc


@dataclass(frozen=True, slots=True)
class StructureMismatchReport:
    """Pure view of contract-vs-model mismatches; consumers decide policy."""

    missing_module_paths: tuple[str, ...]
    missing_merged_projections: tuple[str, ...]
    unknown_weight_mapping_targets: tuple[str, ...]
    declared_module_count: int
    merged_projection_count: int

    @property
    def is_consistent(self) -> bool:
        return not (
            self.missing_module_paths
            or self.missing_merged_projections
            or self.unknown_weight_mapping_targets
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_consistent": self.is_consistent,
            "missing_module_paths": list(self.missing_module_paths),
            "missing_merged_projections": list(self.missing_merged_projections),
            "unknown_weight_mapping_targets": list(self.unknown_weight_mapping_targets),
            "declared_module_count": self.declared_module_count,
            "merged_projection_count": self.merged_projection_count,
        }


class _NamedModuleLike(Protocol):
    """Minimal torch.nn.Module duck type, keeps contracts torch-free."""

    def named_modules(
        self, prefix: str = "", remove_duplicate: bool = True
    ) -> Iterable[tuple[str, Any]]: ...

    def named_parameters(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ) -> Iterable[tuple[str, Any]]: ...


def structure_contract_mismatches(
    model: _NamedModuleLike,
    contract: ModelStructureContract,
) -> StructureMismatchReport:
    """Compare declared paths against actual ``named_modules`` of a model.

    纯视图, 不改模型也不抛错; 校验失败要变成硬错误的消费方自行 raise.
    weight_mapping 的 ``component_path`` 统一指向参数路径 (merged_split 的
    拆分事实由 ``merged_projections`` 声明), 因此全部对照 ``named_parameters``;
    组件与 merged projection 的模块路径对照 ``named_modules``.
    """

    module_paths = {name for name, _module in model.named_modules() if name}
    parameter_paths = {name for name, _param in model.named_parameters() if name}
    declared = {path for component in contract.components for path in component.paths}
    merged = {merged.module_path for merged in contract.merged_projections}
    unknown_mapping_targets = tuple(
        sorted(
            entry.component_path
            for entry in contract.weight_mapping
            if entry.component_path not in parameter_paths
        )
    )
    return StructureMismatchReport(
        missing_module_paths=tuple(sorted(declared - module_paths)),
        missing_merged_projections=tuple(sorted(merged - module_paths)),
        unknown_weight_mapping_targets=tuple(sorted(unknown_mapping_targets)),
        declared_module_count=len(declared),
        merged_projection_count=len(merged),
    )


def resolve_weight_mapping(
    contract: ModelStructureContract,
    checkpoint_keys: Iterable[str],
    *,
    checkpoint_prefix: str = "",
) -> dict[str, str]:
    """Expand the declared weight mapping against concrete checkpoint keys.

    只做字面匹配与 prefix 剥离. 声明式映射的验收语义是全覆盖: 契约必须为每个
    checkpoint key 声明条目 (stacked_slice 为每个源参数各声明一条), 未覆盖的
    key 直接报错, 不静默跳过也不猜测补齐.
    """

    mapping: dict[str, str] = {}
    for entry in contract.weight_mapping:
        candidate = entry.checkpoint_name
        if checkpoint_prefix and candidate.startswith(checkpoint_prefix):
            candidate = candidate[len(checkpoint_prefix):]
        if candidate in mapping and mapping[candidate] != entry.component_path:
            raise XQTConfigError(
                f"Weight mapping declares checkpoint key {candidate!r} twice "
                f"with different targets"
            )
        mapping[candidate] = entry.component_path
    present = set(checkpoint_keys)
    if checkpoint_prefix:
        present = {
            key[len(checkpoint_prefix):] if key.startswith(checkpoint_prefix) else key
            for key in present
        }
    unmapped = present - set(mapping)
    if unmapped:
        raise XQTConfigError(
            "Weight mapping does not cover checkpoint keys: "
            f"{sorted(unmapped)[:8]}{'...' if len(unmapped) > 8 else ''}"
        )
    return mapping


def structure_contract_keep_high_precision_paths(
    contract: ModelStructureContract,
) -> tuple[str, ...]:
    """Return sorted module paths of components declared ``keep_high_precision``.

    这是保护路径的唯一派生入口: 候选发现 (prune) 与模块选择 (quant) 都用它
    判定受保护范围, 不各自实现第二份 hint 解释. 保护范围覆盖组件模块本身
    及其全部子模块 (前缀匹配).
    """

    paths: set[str] = set()
    for component in contract.components:
        if component.precision_hint == "keep_high_precision":
            paths.update(component.paths)
    return tuple(sorted(paths))


def resolve_structure_role(
    contract: ModelStructureContract,
    module_name: str,
) -> str | None:
    """Resolve the declared component role for one module path.

    精确匹配或组件路径前缀 (``is_module_path_within``) 匹配; 契约校验保证
    路径不重复声明, 因此至多命中一个 role. 未声明返回 ``None``, "undeclared"
    语义由调用方决定.
    """

    name = str(module_name)
    for component in contract.components:
        for path in component.paths:
            if is_module_path_within(name, path):
                return component.role
    return None


def is_module_path_within(path: str, root: str) -> bool:
    """Return whether ``path`` equals ``root`` or lies under it (module boundary)."""

    return path == root or path.startswith(f"{root}.")


def compute_topology_fingerprint(model: _NamedModuleLike) -> str:
    """Compute a deterministic hash representing the model's structural topology.

    Covers module hierarchy names, module type names, parameter names and tensor shapes.
    """
    hasher = hashlib.sha256()
    for name, mod in model.named_modules():
        if name:
            cls_name = getattr(type(mod), "__name__", "")
            hasher.update(f"m:{name}:{cls_name}\n".encode("utf-8"))
    for name, param in model.named_parameters():
        if name and param is not None:
            shape = getattr(param, "shape", ())
            shape_str = ",".join(str(int(d)) for d in shape)
            hasher.update(f"p:{name}:{shape_str}\n".encode("utf-8"))
    return hasher.hexdigest()


def is_structure_contract_valid_for_model(
    contract: ModelStructureContract,
    model: _NamedModuleLike,
) -> bool:
    """Check if the structure contract matches the current model topology."""
    if contract.topology_fingerprint is not None:
        current_fp = compute_topology_fingerprint(model)
        if contract.topology_fingerprint != current_fp:
            return False
    return True


def update_structure_contract_for_model(
    contract: ModelStructureContract,
    model: _NamedModuleLike,
) -> ModelStructureContract:
    """Update dimensions and fingerprint of a structure contract after pruning or rewriting."""
    module_dict = dict(model.named_modules())
    updated_merged: list[MergedProjectionSpec] = []
    for spec in contract.merged_projections:
        mod = module_dict.get(spec.module_path)
        if mod is not None and hasattr(mod, "out_features"):
            total_out = getattr(mod, "out_features", None)
            if isinstance(total_out, int) and total_out > 0:
                if total_out != sum(spec.split_out_features):
                    num_parts = len(spec.parts)
                    if total_out % num_parts == 0:
                        part_dim = total_out // num_parts
                        new_dims = tuple(part_dim for _ in range(num_parts))
                    else:
                        old_total = sum(spec.split_out_features)
                        ratios = [d / old_total for d in spec.split_out_features]
                        new_dims = tuple(max(1, int(round(r * total_out))) for r in ratios)
                    updated_merged.append(
                        MergedProjectionSpec(
                            module_path=spec.module_path,
                            parts=spec.parts,
                            split_out_features=new_dims,
                        )
                    )
                    continue
        updated_merged.append(spec)

    new_fp = compute_topology_fingerprint(model)
    return ModelStructureContract(
        family=contract.family,
        components=contract.components,
        merged_projections=tuple(updated_merged),
        weight_mapping=contract.weight_mapping,
        schema_version=contract.schema_version,
        topology_fingerprint=new_fp,
    )


_TASK_TO_FAMILY: dict[str, str] = {
    "detection": "detection",
    "llm": "llm",
    "text_generation": "llm",
    "diffusion": "diffusion",
    "image_generation": "diffusion",
    "multimodal": "multimodal",
    "vlm": "multimodal",
    "classification": "transformer",
}

_DETECTION_HEAD_MARKERS = ("head", "detect", "yolo", "rtdetr")
_EXPERT_MARKERS = ("experts", "expert", "moe")
_ROUTER_MARKERS = ("router", "gating", "gate")
_DIFFUSION_MARKERS = ("unet", "dit", "vae", "text_encoder", "denoiser")
_MULTIMODAL_MARKERS = ("vision_encoder", "visual_encoder", "encoder_cache", "cross_attn")


def model_family_names() -> tuple[str, ...]:
    """Return the canonical model family vocabulary."""
    return MODEL_FAMILY_NAMES


def _module_paths(model: _NamedModuleLike) -> list[str]:
    return [name for name, _module in model.named_modules()]


def _contains_attention(paths: Sequence[str]) -> bool:
    return any(
        marker in path.lower()
        for marker in (
            "attention",
            "attn",
            "encoder",
            "q_proj",
            "k_proj",
            "v_proj",
            "out_proj",
        )
        for path in paths
    )


def _contains_conv(model: _NamedModuleLike) -> bool:
    return any(
        type(module).__name__ == "Conv2d"
        for _name, module in model.named_modules()
    )


def classify_model_family(
    model: _NamedModuleLike,
    *,
    task_type: str | None = None,
) -> str:
    """Classify a model into a canonical XQT model family."""
    if task_type:
        normalized = str(task_type).strip().lower()
        if normalized in _TASK_TO_FAMILY:
            return _TASK_TO_FAMILY[normalized]

    paths = _module_paths(model)
    lowered = [path.lower() for path in paths]
    joined = " ".join(lowered)

    if any(marker in joined for marker in _EXPERT_MARKERS) and any(
        marker in joined for marker in _ROUTER_MARKERS
    ):
        return "moe"
    if any(marker in joined for marker in _DIFFUSION_MARKERS):
        return "diffusion"
    if any(marker in joined for marker in _MULTIMODAL_MARKERS):
        return "multimodal"
    has_detection_head = any(
        marker in joined for marker in ("detect", "yolo", "rtdetr", "detection_head")
    ) or (
        "head" in joined
        and "patch_embed" not in joined
        and "lm_head" not in joined
        and "blocks" not in joined
    )
    if has_detection_head and _contains_conv(model):
        return "detection"
    if "lm_head" in joined or (
        "q_proj" in joined and "k_proj" in joined and "v_proj" in joined
    ):
        return "llm"
    if "patch_embed" in joined or ("vit" in joined and _contains_attention(paths)):
        return "vit"
    if _contains_attention(paths) and "norm" in joined:
        return "transformer"
    if _contains_conv(model):
        return "convnet"
    return "unknown"


def component_grouping(
    model: _NamedModuleLike,
    *,
    family: str | None = None,
    task_type: str | None = None,
) -> dict[str, list[str]]:
    """Group module paths by model-family component role."""
    resolved = family or classify_model_family(model, task_type=task_type)
    groups: dict[str, list[str]] = {
        "attention": [],
        "ffn": [],
        "norm": [],
        "head": [],
        "backbone": [],
        "expert": [],
        "router": [],
        "embedding": [],
        "encoder": [],
        "cross_attention": [],
        "diffusion_component": [],
        "other": [],
    }
    for name, module in model.named_modules():
        if not name:
            continue
        lowered = name.lower()
        module_type = type(module).__name__
        if any(marker in lowered for marker in _ROUTER_MARKERS):
            groups["router"].append(name)
        elif any(marker in lowered for marker in _DIFFUSION_MARKERS):
            groups["diffusion_component"].append(name)
        elif any(
            marker in lowered
            for marker in ("vision_encoder", "visual_encoder", "encoder_cache")
        ):
            groups["encoder"].append(name)
        elif any(marker in lowered for marker in ("cross_attn", "cross_attention")):
            groups["cross_attention"].append(name)
        elif any(marker in lowered for marker in _EXPERT_MARKERS):
            groups["expert"].append(name)
        elif module_type in ("Embedding", "VocabParallelEmbedding"):
            groups["embedding"].append(name)
        elif any(marker in lowered for marker in _DETECTION_HEAD_MARKERS):
            groups["head"].append(name)
        elif module_type in ("LayerNorm", "RMSNorm") or "norm" in module_type.lower():
            groups["norm"].append(name)
        elif any(
            marker in lowered
            for marker in ("attention", "attn", "q_proj", "k_proj", "v_proj", "out_proj")
        ):
            groups["attention"].append(name)
        elif any(marker in lowered for marker in ("ffn", "mlp", "feed_forward")):
            groups["ffn"].append(name)
        elif module_type == "Conv2d":
            groups["backbone"].append(name)
        else:
            groups["other"].append(name)
    return groups


def build_structure_contract(
    model: _NamedModuleLike,
    *,
    family: str | None = None,
    task_type: str | None = None,
) -> ModelStructureContract:
    """Derive a draft ModelStructureContract from grouping heuristics."""
    resolved = family or classify_model_family(model, task_type=task_type)
    groups = component_grouping(model, family=resolved, task_type=task_type)
    unknown_roles = set(groups) - set(COMPONENT_ROLES)
    if unknown_roles:
        raise XQTConfigError(
            f"component_grouping produced roles outside COMPONENT_ROLES: "
            f"{sorted(unknown_roles)}"
        )
    components = tuple(
        ComponentSpec(role=role, paths=tuple(sorted(paths)))
        for role, paths in groups.items()
        if paths
    )
    return ModelStructureContract(family=resolved, components=components)


def resolve_and_validate_structure_contract(
    model: _NamedModuleLike,
    profile: Any | None = None,
    adapter: Any | None = None,
    *,
    strict: bool = True,
) -> ModelStructureContract | None:
    """Resolve, validate and bind a structure contract to a model."""
    contract: ModelStructureContract | None = None
    if adapter is not None and hasattr(adapter, "structure_contract"):
        contract = adapter.structure_contract(model)

    if contract is None and profile is not None:
        raw_contract = getattr(profile, "structure_contract", None)
        if raw_contract is not None:
            if isinstance(raw_contract, ModelStructureContract):
                contract = raw_contract
            elif isinstance(raw_contract, Mapping):
                contract = ModelStructureContract.from_mapping(raw_contract)
            elif isinstance(raw_contract, str):
                from xqt.core.imports import resolve_target
                target_obj = resolve_target(raw_contract)
                if callable(target_obj):
                    target_res = target_obj(model)
                    contract = (
                        target_res
                        if isinstance(target_res, ModelStructureContract)
                        else None
                    )
                elif isinstance(target_obj, ModelStructureContract):
                    contract = target_obj
        if contract is None:
            family = getattr(profile, "family", None)
            try:
                contract = build_structure_contract(model, family=family)
            except Exception:
                pass

    if contract is None:
        try:
            contract = build_structure_contract(model)
        except Exception:
            contract = None

    if contract is not None:
        mismatches = structure_contract_mismatches(model, contract)
        if not mismatches.is_consistent:
            details = []
            if mismatches.missing_module_paths:
                details.append(f"missing modules: {mismatches.missing_module_paths}")
            if mismatches.missing_merged_projections:
                details.append(f"missing merged projections: {mismatches.missing_merged_projections}")
            if mismatches.unknown_weight_mapping_targets:
                details.append(f"unknown weight mapping targets: {mismatches.unknown_weight_mapping_targets}")
            if strict:
                raise XQTConfigError(
                    f"ModelStructureContract validation failed: {'; '.join(details)}"
                )
        fp = compute_topology_fingerprint(model)
        contract = contract.with_topology_fingerprint(fp)
    elif strict and profile is not None:
        raise XQTConfigError(
            f"Failed to resolve model structure contract for profile {getattr(profile, 'profile_id', profile)!r}"
        )

    return contract


__all__ = [
    "COMPONENT_ROLES",
    "ComponentRole",
    "ComponentSpec",
    "MODEL_FAMILY_NAMES",
    "MODEL_STRUCTURE_CONTRACT_SCHEMA_VERSION",
    "MergedProjectionSpec",
    "ModelStructureContract",
    "PRECISION_HINTS",
    "StructureMismatchReport",
    "WEIGHT_MAPPING_KINDS",
    "WeightMappingEntry",
    "build_structure_contract",
    "classify_model_family",
    "component_grouping",
    "compute_topology_fingerprint",
    "is_module_path_within",
    "is_structure_contract_valid_for_model",
    "model_family_names",
    "resolve_and_validate_structure_contract",
    "resolve_structure_role",
    "resolve_weight_mapping",
    "structure_contract_keep_high_precision_paths",
    "structure_contract_mismatches",
    "update_structure_contract_for_model",
]
