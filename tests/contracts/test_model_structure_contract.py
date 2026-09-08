from __future__ import annotations

import pytest
from torch import Tensor, nn

from xqt.contracts import (
    COMPONENT_ROLES,
    ComponentSpec,
    MergedProjectionSpec,
    ModelStructureContract,
    WeightMappingEntry,
    resolve_weight_mapping,
    structure_contract_mismatches,
)
from xqt.core.base import XQTConfigError
from xqt.kernels.nn.fixtures import build_smoke_llm
from xqt.kernels.nn.fixtures.families import (
    build_structure_contract,
    component_grouping,
)


class _MergedQKVBlock(nn.Module):
    def __init__(self, hidden_dim: int = 16) -> None:
        super().__init__()
        self.qkv_proj = nn.Linear(hidden_dim, hidden_dim * 3)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.out_proj(self.norm(self.qkv_proj(x)))


def _merged_qkv_contract() -> ModelStructureContract:
    return ModelStructureContract(
        family="transformer",
        components=(
            ComponentSpec(role="attention", paths=("out_proj", "qkv_proj")),
            ComponentSpec(role="norm", paths=("norm",)),
        ),
        merged_projections=(
            MergedProjectionSpec("qkv_proj", ("q", "k", "v"), (16, 16, 16)),
        ),
        weight_mapping=(
            WeightMappingEntry(
                "blocks.0.qkv_proj.weight",
                "qkv_proj.weight",
                kind="merged_split",
            ),
            WeightMappingEntry("blocks.0.qkv_proj.bias", "qkv_proj.bias", kind="merged_split"),
            WeightMappingEntry("blocks.0.out_proj.weight", "out_proj.weight"),
            WeightMappingEntry("blocks.0.out_proj.bias", "out_proj.bias"),
            WeightMappingEntry("blocks.0.norm.weight", "norm.weight"),
            WeightMappingEntry("blocks.0.norm.bias", "norm.bias"),
        ),
    )


def test_component_role_vocabulary_matches_grouping_keys() -> None:
    assert set(component_grouping(build_smoke_llm())) == set(COMPONENT_ROLES)


def test_builder_derives_consistent_draft_contract() -> None:
    model = build_smoke_llm()
    contract = build_structure_contract(model, family="llm")
    assert contract.family == "llm"
    roles = {component.role for component in contract.components}
    assert {"attention", "norm"} <= roles
    mismatches = structure_contract_mismatches(model, contract)
    assert mismatches.is_consistent, mismatches.to_dict()


def test_contract_roundtrip_preserves_declarations() -> None:
    contract = _merged_qkv_contract()
    restored = ModelStructureContract.from_mapping(contract.to_dict())
    assert restored == contract


def test_mismatch_report_detects_missing_and_unknown_paths() -> None:
    model = _MergedQKVBlock()
    contract = ModelStructureContract(
        family="transformer",
        components=(
            ComponentSpec(role="attention", paths=("out_proj", "missing_proj")),
            ComponentSpec(role="norm", paths=("norm",)),
        ),
        merged_projections=(
            MergedProjectionSpec("qkv_proj", ("q", "k", "v"), (16, 16, 16)),
        ),
        weight_mapping=(
            WeightMappingEntry("ckpt.weight", "not_a_param.weight"),
        ),
    )
    report = structure_contract_mismatches(model, contract)
    assert not report.is_consistent
    assert report.missing_module_paths == ("missing_proj",)
    assert report.missing_merged_projections == ()
    assert report.unknown_weight_mapping_targets == ("not_a_param.weight",)

    ok = structure_contract_mismatches(model, _merged_qkv_contract())
    assert ok.is_consistent, ok.to_dict()


def test_resolve_weight_mapping_direct_and_prefix() -> None:
    contract = _merged_qkv_contract()
    keys = [
        f"model.blocks.0.{name}"
        for name in (
            "qkv_proj.weight",
            "qkv_proj.bias",
            "out_proj.weight",
            "out_proj.bias",
            "norm.weight",
            "norm.bias",
        )
    ]
    mapping = resolve_weight_mapping(contract, keys, checkpoint_prefix="model.")
    assert mapping["blocks.0.qkv_proj.weight"] == "qkv_proj.weight"
    assert mapping["blocks.0.out_proj.weight"] == "out_proj.weight"


def test_resolve_weight_mapping_rejects_uncovered_keys() -> None:
    contract = _merged_qkv_contract()
    with pytest.raises(XQTConfigError, match="does not cover"):
        resolve_weight_mapping(contract, ["blocks.0.qkv_proj.weight", "extra.weight"])


def test_contract_validation_failures_are_explicit() -> None:
    with pytest.raises(XQTConfigError, match="role must be one of"):
        ComponentSpec(role="not_a_role", paths=("m",))
    with pytest.raises(XQTConfigError, match="duplicate role"):
        ModelStructureContract(
            family="llm",
            components=(
                ComponentSpec(role="norm", paths=("a",)),
                ComponentSpec(role="norm", paths=("b",)),
            ),
        )
    with pytest.raises(XQTConfigError, match="must match parts length"):
        MergedProjectionSpec("qkv_proj", ("q", "k", "v"), (16, 16))
    with pytest.raises(XQTConfigError, match="schema_version"):
        ModelStructureContract.from_mapping({"schema_version": 99, "family": "llm"})


def test_topology_fingerprint_and_validity_lifecycle() -> None:
    from xqt.contracts import (
        compute_topology_fingerprint,
        is_structure_contract_valid_for_model,
        update_structure_contract_for_model,
    )

    model = _MergedQKVBlock(hidden_dim=16)
    contract = _merged_qkv_contract()
    fp_before = compute_topology_fingerprint(model)
    bound_contract = contract.with_topology_fingerprint(fp_before)

    assert is_structure_contract_valid_for_model(bound_contract, model)

    # 改变模型参数形状（模拟剪枝）
    model.qkv_proj = nn.Linear(16, 24)
    fp_after = compute_topology_fingerprint(model)
    assert fp_before != fp_after

    # 验证旧契约指纹失效
    assert not is_structure_contract_valid_for_model(bound_contract, model)

    # 通过 update_structure_contract_for_model 刷新契约与维度
    updated_contract = update_structure_contract_for_model(bound_contract, model)
    assert updated_contract.topology_fingerprint == fp_after
    assert updated_contract.merged_projections[0].split_out_features == (8, 8, 8)
    assert is_structure_contract_valid_for_model(updated_contract, model)


def test_resolve_and_validate_structure_contract_detects_mismatch() -> None:
    from xqt.contracts import resolve_and_validate_structure_contract

    class _BadModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(4, 4)

    bad_model = _BadModel()
    contract = _merged_qkv_contract()

    # 通过 profile 传入带有不匹配声明的 contract
    from xqt.model.config import ModelProfile

    profile = ModelProfile(
        profile_id="test.bad",
        family="transformer",
        structure_contract=contract,
    )

    with pytest.raises(XQTConfigError) as exc_info:
        resolve_and_validate_structure_contract(bad_model, profile=profile, strict=True)
    assert "ModelStructureContract validation failed" in str(exc_info.value)
    assert "missing modules" in str(exc_info.value)


from xqt.model.adapter import ModelAdapter


class _MockAdapter(ModelAdapter):
    def load(self, checkpoint: str | None, **params: object) -> nn.Module:
        return _MergedQKVBlock(hidden_dim=16)

    def adapt(self, model: nn.Module, **params: object) -> nn.Module:
        return model

    def structure_contract(self, model: nn.Module) -> ModelStructureContract:
        return _merged_qkv_contract()

    def inference_contract(self, model: nn.Module) -> object:
        return None


def test_load_model_pass_direct_inject_and_contract_parity() -> None:
    from xqt.core.types import XQTContext
    from xqt.model.config import ModelProfile
    from xqt.model.registry import register_model_profile
    from xqt.pipeline.model_pass import LoadModelPass

    profile = ModelProfile(
        profile_id="test.mock_qkv",
        family="transformer",
        adapter_target="tests.xqt.contracts.test_model_structure_contract._MockAdapter",
    )
    register_model_profile(profile, replace=True)

    # 1. 方式 A: 通过 checkpoint + profile 加载
    ctx_loaded = XQTContext()
    ctx_loaded.model_profile = profile
    LoadModelPass().run(ctx_loaded)

    # 2. 方式 B: 直接通过 model 注入
    ctx_injected = XQTContext()
    ctx_injected.model = _MergedQKVBlock(hidden_dim=16)
    ctx_injected.model_profile = profile
    LoadModelPass().run(ctx_injected)

    assert ctx_loaded.model is not None
    assert ctx_injected.model is not None
    assert ctx_loaded.structure_contract is not None
    assert ctx_injected.structure_contract is not None
    assert (
        ctx_loaded.structure_contract.to_dict()
        == ctx_injected.structure_contract.to_dict()
    )
    assert (
        ctx_loaded.structure_contract.topology_fingerprint
        == ctx_injected.structure_contract.topology_fingerprint
    )

