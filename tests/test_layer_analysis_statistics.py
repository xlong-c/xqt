from __future__ import annotations

import torch

from xqt.analysis.layer_analysis import (
    build_layer_analysis_events,
    build_layer_analysis_payload,
)


class _TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = torch.nn.Linear(8, 8)
        self.act = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(8, 4)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(inputs)))


def test_layer_analysis_statistics_include_output_and_weight_rows() -> None:
    torch.manual_seed(0)
    reference_model = _TinyModel().eval()
    candidate_model = _TinyModel().eval()
    candidate_model.load_state_dict(reference_model.state_dict())
    with torch.no_grad():
        candidate_model.fc1.weight.mul_(0.9)
        candidate_model.fc2.bias.add_(0.05)
    example_input = torch.randn(2, 8)

    payload = build_layer_analysis_payload(
        reference_model,
        candidate_model,
        example_input,
        module_names=["fc1", "fc2"],
        include_statistics=True,
        sample_budget=64,
    )

    rows = payload["layer_statistics"]
    assert isinstance(rows, list)
    assert len(rows) == 4
    variable_pairs = {(row["layer"], row["variable"]) for row in rows}
    assert ("fc1", "output") in variable_pairs
    assert ("fc1", "weight") in variable_pairs
    assert ("fc2", "output") in variable_pairs
    assert ("fc2", "weight") in variable_pairs
    for row in rows:
        assert "error" in row
        assert "quantized" in row
        assert "float" in row


def test_layer_analysis_events_emit_weight_statistics_rows() -> None:
    torch.manual_seed(1)
    reference_model = _TinyModel().eval()
    candidate_model = _TinyModel().eval()
    candidate_model.load_state_dict(reference_model.state_dict())
    with torch.no_grad():
        candidate_model.fc1.weight.add_(0.1)
    example_input = torch.randn(1, 8)

    payload = build_layer_analysis_payload(
        reference_model,
        candidate_model,
        example_input,
        module_names=["fc1"],
        include_statistics=True,
    )
    events = build_layer_analysis_events("quant_eval", payload)

    statistics_events = [event for event in events if event.get("event") == "layer_statistics"]
    assert len(statistics_events) == 2
    assert {event["variable"] for event in statistics_events} == {"output", "weight"}
