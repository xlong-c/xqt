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
