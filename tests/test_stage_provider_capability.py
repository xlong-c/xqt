from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Mapping

import torch.nn as nn

from xqt.contracts import module as module_contracts
from xqt.core.stage_specs import PruneStageSpec, QuantStageSpec
from xqt.workflows.stage_provider import (
    ModelPrunerProvider,
    ModelQuantizerProvider,
    StagePayloadBuildContext,
)


def _build_context(
    stage: SimpleNamespace,
    metrics: Mapping[str, Any],
    model: nn.Module,
) -> StagePayloadBuildContext:
    return StagePayloadBuildContext(
        state=SimpleNamespace(
            context=SimpleNamespace(model=model, device="cpu"),
        ),
        stage=stage,
        source_stage_name="baseline",
        accepted=True,
        message="accepted",
        metrics=metrics,
        new_artifacts={},
    )


def test_quant_provider_uses_capability_written_in_nested_metrics() -> None:
    # Given
    capability = {
        "backend": "pass-owned",
        "optimization_capability": {"status": "available", "runtime": "eager"},
    }
    stage = SimpleNamespace(
        name="quant",
        kind="quant",
        params={},
        spec=QuantStageSpec(backend="pytorch"),
        from_stage=None,
        compare_to=None,
    )
    context = _build_context(
        stage,
        {"components": [{"metadata": {"capability": capability}}]},
        nn.Linear(4, 2),
    )

    # When
    output = ModelQuantizerProvider().build(context)

    # Then
    assert output.payload_value.capability == capability


def test_prune_provider_uses_optimization_capability_written_in_metrics() -> None:
    # Given
    capability = {"status": "fallback", "runtime": "sparse_reference"}
    stage = SimpleNamespace(
        name="prune",
        kind="prune",
        params={},
        spec=PruneStageSpec(method="nm_structured"),
        from_stage=None,
        compare_to=None,
    )
    context = _build_context(
        stage,
        {"runtime_capability": {"optimization_capability": capability}},
        nn.Linear(4, 2),
    )

    # When
    output = ModelPrunerProvider().build(context)

    # Then
    assert output.payload_value.capability == capability


def test_quant_provider_does_not_infer_capability_when_metrics_omit_it() -> None:
    # Given
    stage = SimpleNamespace(
        name="quant",
        kind="quant",
        params={},
        spec=QuantStageSpec(backend="pytorch"),
        from_stage=None,
        compare_to=None,
    )
    context = _build_context(stage, {"backend": "pytorch"}, nn.Linear(4, 2))

    # When
    output = ModelQuantizerProvider().build(context)

    # Then
    assert output.payload_value.capability is None


def test_get_module_contract_returns_detached_mapping() -> None:
    # Given
    model = nn.Linear(4, 2)
    raw_contract = {"operator_kind": "linear"}
    setattr(model, "_xqt_module_contract", raw_contract)

    # When
    contract = module_contracts.get_module_contract(model)

    # Then
    assert contract == raw_contract
    assert contract is not raw_contract
