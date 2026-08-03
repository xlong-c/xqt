from pathlib import Path

import torch

from xqt.quant import (
    TurboQuantCodec,
    TurboQuantWeightOnlyLinear,
    quantize_with_turboquant,
)
from xqt.workflows import optimize_model


class _TinyLinearModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = torch.nn.Linear(64, 32, bias=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.fc(inputs)


def test_codec_round_trip_error_decreases_with_bits() -> None:
    rows = torch.randn(128, 64)
    prev = None
    for bits in (2, 3, 4):
        codec = TurboQuantCodec(dim=64, bits=bits, mode="mse")
        encoding = codec.encode(rows)
        recon = codec.decode(encoding, device=rows.device)
        mse = (rows - recon).pow(2).mean().item()
        assert encoding.codebook.numel() == (1 << bits)
        assert int(encoding.codes.max()) < (1 << bits)
        if prev is not None:
            # 每加 1 bit, 失真应显著下降 (Panter-Dite ~ 1/4)
            assert mse < prev * 0.6
        prev = mse


def test_rotation_matrix_is_orthonormal() -> None:
    codec = TurboQuantCodec(dim=64, bits=3, rotation_kind="randomized_hadamard")
    rotation = codec.rotation_matrix(torch.device("cpu"))
    gram = rotation.t() @ rotation
    assert torch.allclose(gram, torch.eye(64), atol=1e-5)


def test_prod_mode_removes_inner_product_bias() -> None:
    # y 与 x 相关时, MSE 量化对内积有收缩偏差; QJL 残差把偏差修成近无偏.
    torch.manual_seed(0)
    trials = 200

    def mean_bias(mode: str) -> float:
        codec = TurboQuantCodec(dim=64, bits=3, mode=mode)
        total = 0.0
        for _ in range(trials):
            x = torch.randn(1, 64)
            y = 0.7 * x.reshape(-1) + 0.3 * torch.randn(64)
            encoding = codec.encode(x)
            est = codec.estimate_inner_product(encoding, y, device=x.device).item()
            total += est - (x.reshape(-1) @ y).item()
        return total / trials

    bias_mse = abs(mean_bias("mse"))
    bias_prod = abs(mean_bias("prod"))
    assert bias_prod < bias_mse * 0.5


def test_quantize_with_turboquant_replaces_linear_on_cpu() -> None:
    model = _TinyLinearModel().eval()

    result = quantize_with_turboquant(
        model,
        policy={
            "bits": 4,
            "mode": "mse",
            "rotation_kind": "randomized_hadamard",
            "include_module_names": ["fc"],
        },
        inplace=False,
    )
    output = result.model(torch.randn(4, 64))

    assert isinstance(result.model.fc, TurboQuantWeightOnlyLinear)
    assert result.quantized_modules == ["fc"]
    assert output.shape == (4, 32)
    assert result.strategy == "w4a16_fp4"
    assert result.method == "turboquant"
    assert result.compute == "dequant_fp16"
    assert result.metadata["data_oblivious"] is True
    assert result.metadata["algorithm_metadata"]["rotation_kind"] == "randomized_hadamard"


def test_turboquant_weight_only_forward_matches_dequantized_weight() -> None:
    linear = torch.nn.Linear(64, 32, bias=True)
    qlinear = TurboQuantWeightOnlyLinear.from_linear(linear, bits=4, mode="mse")
    inputs = torch.randn(8, 64)

    weight = qlinear.dequantized_weight(dtype=torch.float32, device=inputs.device)
    manual = torch.nn.functional.linear(inputs, weight, qlinear.bias)

    assert torch.allclose(qlinear(inputs), manual, atol=1e-5)


def test_turboquant_workflow_runs_via_optimize_model() -> None:
    recipe = (
        Path(__file__).resolve().parents[3]
        / "xqt"
        / "recipes"
        / "quant"
        / "turboquant"
        / "turboquant_smoke.yaml"
    )

    result = optimize_model(
        recipe,
        example_inputs=torch.randn(2, 16, dtype=torch.float32),
        write_outputs=False,
    )

    assert [stage.name for stage in result.stages] == [
        "turboquant_quant",
        "benchmark_model",
    ]
    quant_stage = result.stages[0]
    assert quant_stage.accepted is True
    assert quant_stage.metrics["strategy"] == "w4a16_fp4"
    assert quant_stage.metrics["metadata"]["execution_state"] == "turboquant"
    assert quant_stage.metrics["quantized_modules"] == ["linear", "proj"]
