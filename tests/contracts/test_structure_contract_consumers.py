"""Structure-contract consumer behavior tests (prune discovery + quant policy)."""

from __future__ import annotations

import pytest
from torch import Tensor, nn

from xqt.compression.prune import (
    find_structured_pruning_targets,
    plan_structured_pruning,
)
from xqt.compression.quant.policy import (
    QuantizationPolicy,
    list_quantizable_modules,
    should_quantize_module,
)
from xqt.contracts import (
    ComponentSpec,
    ModelStructureContract,
    is_module_path_within,
    resolve_structure_role,
    structure_contract_keep_high_precision_paths,
)


class _MoEBlock(nn.Module):
    def __init__(self, experts_count: int = 4, hidden: int = 8) -> None:
        super().__init__()
        self.router = nn.Linear(hidden, experts_count)
        self.experts = nn.ModuleList(
            nn.Linear(hidden, hidden) for _index in range(experts_count)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.experts[0](x)


class _TwoMoEModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.moe_a = _MoEBlock()
        self.moe_b = _MoEBlock()

    def forward(self, x: Tensor) -> Tensor:
        return self.moe_a(x) + self.moe_b(x)


def _moe_contract() -> ModelStructureContract:
    return ModelStructureContract(
        family="moe",
        components=(
            ComponentSpec(
                role="router",
                paths=("moe_a",),
                precision_hint="keep_high_precision",
            ),
            ComponentSpec(role="expert", paths=("moe_b",)),
        ),
    )


def test_contract_helper_matching() -> None:
    contract = _moe_contract()
    assert structure_contract_keep_high_precision_paths(contract) == ("moe_a",)
    assert resolve_structure_role(contract, "moe_b.experts.0") == "expert"
    assert resolve_structure_role(contract, "moe_b") == "expert"
    assert resolve_structure_role(contract, "other.module") is None
    assert is_module_path_within("moe_a.experts.2", "moe_a")
    assert not is_module_path_within("moe_ax", "moe_a")


def test_prune_targets_without_contract_keep_baseline_behavior() -> None:
    model = _TwoMoEModel()
    targets = find_structured_pruning_targets(model, granularity="expert")
    assert {target.module_name for target in targets} == {"moe_a", "moe_b"}
    assert all("structure_role" not in target.metadata for target in targets)


def test_prune_targets_skip_contract_protected_component() -> None:
    model = _TwoMoEModel()
    targets = find_structured_pruning_targets(
        model, granularity="expert", structure_contract=_moe_contract()
    )
    assert {target.module_name for target in targets} == {"moe_b"}
    assert targets[0].metadata["structure_role"] == "expert"


def test_prune_targets_annotate_undeclared_paths_without_dropping() -> None:
    model = _TwoMoEModel()
    contract = ModelStructureContract(
        family="moe",
        components=(
            ComponentSpec(
                role="router",
                paths=("moe_a",),
                precision_hint="keep_high_precision",
            ),
        ),
    )
    targets = find_structured_pruning_targets(
        model, granularity="expert", structure_contract=contract
    )
    roles = {target.module_name: target.metadata["structure_role"] for target in targets}
    assert roles == {"moe_b": "undeclared"}


def test_prune_plan_records_contract_protection() -> None:
    model = _TwoMoEModel()
    plan = plan_structured_pruning(
        model,
        0.5,
        granularity="expert",
        structure_contract=_moe_contract(),
    )
    assert plan.blocked_modules == ["moe_a"]
    assert {action.module_name for action in plan.actions} == {"moe_b"}


def test_quant_policy_without_contract_keeps_baseline_behavior() -> None:
    model = nn.Module()
    model.body = nn.Linear(8, 8)
    model.classifier = nn.Linear(8, 4)
    candidates = {
        candidate.name: candidate
        for candidate in list_quantizable_modules(model, QuantizationPolicy())
    }
    assert candidates["body"].quantize is True
    assert candidates["body"].reason == "matched policy"
    assert candidates["body"].structure_role is None
    assert candidates["classifier"].quantize is False
    assert candidates["classifier"].reason == "filtered by policy"


def test_quant_policy_contract_protection_overrides_include() -> None:
    body = nn.Linear(8, 8)
    classifier = nn.Linear(8, 4)
    contract = ModelStructureContract(
        family="transformer",
        components=(
            ComponentSpec(
                role="head",
                paths=("classifier",),
                precision_hint="keep_high_precision",
            ),
            ComponentSpec(role="ffn", paths=("body",)),
        ),
    )
    policy = QuantizationPolicy(include_module_names=("classifier",))
    assert should_quantize_module("classifier", classifier, policy)
    assert not should_quantize_module(
        "classifier", classifier, policy, structure_contract=contract
    )
    assert should_quantize_module("body", body, policy, structure_contract=contract)


def test_list_quantizable_modules_annotates_roles_and_protection() -> None:
    class _Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.body = nn.Linear(8, 8)
            self.classifier = nn.Linear(8, 4)

    model = _Model()
    contract = ModelStructureContract(
        family="transformer",
        components=(
            ComponentSpec(
                role="head",
                paths=("classifier",),
                precision_hint="keep_high_precision",
            ),
            ComponentSpec(role="ffn", paths=("body",)),
        ),
    )
    candidates = {
        candidate.name: candidate
        for candidate in list_quantizable_modules(
            model, QuantizationPolicy(), structure_contract=contract
        )
    }
    assert candidates["classifier"].quantize is False
    assert candidates["classifier"].reason == "structure contract keep_high_precision"
    assert candidates["classifier"].structure_role == "head"
    assert candidates["body"].quantize is True
    assert candidates["body"].reason == "matched policy"
    assert candidates["body"].structure_role == "ffn"


def test_prune_targets_reject_inconsistent_contract_consumer_model() -> None:
    model = _TwoMoEModel()
    contract = ModelStructureContract(
        family="moe",
        components=(ComponentSpec(role="expert", paths=("missing_moe",)),),
    )
    targets = find_structured_pruning_targets(
        model, granularity="expert", structure_contract=contract
    )
    assert {target.module_name for target in targets} == {"moe_a", "moe_b"}
    assert all(
        target.metadata["structure_role"] == "undeclared" for target in targets
    )


@pytest.mark.parametrize("protected_name", ["moe_a", "moe_a.experts.0"])
def test_protected_prefix_matching_covers_submodules(protected_name: str) -> None:
    model = _TwoMoEModel()
    contract = ModelStructureContract(
        family="moe",
        components=(
            ComponentSpec(
                role="router",
                paths=(protected_name,),
                precision_hint="keep_high_precision",
            ),
        ),
    )
    targets = find_structured_pruning_targets(
        model, granularity="expert", structure_contract=contract
    )
    assert {target.module_name for target in targets} == {"moe_b"}


def test_run_prune_stage_consumes_context_structure_contract() -> None:
    import copy

    import torch

    from xqt.core.types import XQTContext
    from xqt.pipeline.passes import run_prune_stage
    from xqt.core.stage_specs import PruneStageSpec

    model = _TwoMoEModel()
    context = XQTContext(
        model=model,
        reference_model=copy.deepcopy(model),
        example_inputs=torch.randn(2, 8),
        device="cpu",
        artifact_dir="artifacts/xqt/tests/structure_contract_stage",
        project_name="structure_contract_stage",
    )
    context.structure_contract = _moe_contract()
    output = run_prune_stage(
        context,
        PruneStageSpec(
            method="structured",
            target_sparsity=0.5,
            granularity="expert",
        ),
    )
    assert output.metrics["prune"]["structure_contract_family"] == "moe"
    assert len(model.moe_a.experts) == 4
    assert len(model.moe_b.experts) == 2
    assert model.moe_b.router.out_features == 2
    assert output.structure_contract is not None
    assert output.structure_contract.topology_fingerprint is not None
    from xqt.contracts import is_structure_contract_valid_for_model

    assert is_structure_contract_valid_for_model(output.structure_contract, output.model)


def test_structure_contract_stale_fingerprint_rejected_by_consumer() -> None:
    from xqt.core.base import XQTConfigError

    model = _TwoMoEModel()
    contract = _moe_contract().with_topology_fingerprint("deadbeef" * 8)
    with pytest.raises(XQTConfigError, match="invalid or expired"):
        find_structured_pruning_targets(
            model, granularity="expert", structure_contract=contract
        )


def test_run_prune_stage_without_contract_keeps_baseline_rewrite() -> None:
    import copy

    import torch

    from xqt.core.stage_specs import PruneStageSpec
    from xqt.core.types import XQTContext
    from xqt.pipeline.passes import run_prune_stage

    model = _TwoMoEModel()
    context = XQTContext(
        model=model,
        reference_model=copy.deepcopy(model),
        example_inputs=torch.randn(2, 8),
        device="cpu",
        artifact_dir="artifacts/xqt/tests/structure_contract_stage",
        project_name="structure_contract_stage",
    )
    run_prune_stage(
        context,
        PruneStageSpec(
            method="structured",
            target_sparsity=0.5,
            granularity="expert",
            scope="per_layer",
        ),
    )
    assert "structure_contract_family" not in context.metrics["prune"]
    assert len(model.moe_a.experts) == 2
    assert len(model.moe_b.experts) == 2
