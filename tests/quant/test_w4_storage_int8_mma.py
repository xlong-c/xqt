import torch

from examples.mlp_w4_int8_mma_acceptance import (
    _select_best_path,
    weight_retarget_error,
)
from xqt.quant.quantizers.fp4_weight_only import FP4WeightOnlyLinear, quantize_with_fp4_weight_only
from xqt.quant.quantizers.w4_storage_int8_mma import (
    W4StorageInt8MmaLinear,
    quantize_with_w4_storage_int8_mma,
)
from xqt.workflows import XQTOptimizationSession


class _TinyLinearModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = torch.nn.Linear(16, 32, bias=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.fc(inputs)


def test_quantize_with_w4_storage_int8_mma_from_linear_on_cpu() -> None:
    model = _TinyLinearModel().eval()

    result = quantize_with_w4_storage_int8_mma(
        model,
        policy={"include_module_types": ["Linear"], "exclude_name_patterns": []},
        engine="torch_int_mm",
        inplace=False,
        group_size=8,
    )
    output = result.model(torch.randn(4, 16))

    assert isinstance(result.model.fc, W4StorageInt8MmaLinear)
    assert result.quantized_modules == ["fc"]
    assert output.shape == (4, 32)
    assert result.metadata["quantization_nature"] == "true"
    assert result.metadata["storage_encoding"] == "packed_signed_int4_group_scale"
    assert result.metadata["compute_encoding"] == "w8a8_int8_mma"
    assert result.model.fc.packed_weight.dtype == torch.uint8
    assert result.model.fc.storage_nbytes() < 16 * 32 * 4


def test_w4_storage_int8_mma_retargets_existing_fp4_weight_only() -> None:
    model = _TinyLinearModel().eval()
    fp4 = quantize_with_fp4_weight_only(
        model,
        policy={"include_module_types": ["Linear"], "exclude_name_patterns": [], "group_size": 8},
        inplace=False,
    )
    assert isinstance(fp4.model.fc, FP4WeightOnlyLinear)

    result = quantize_with_w4_storage_int8_mma(
        fp4.model,
        policy={"include_module_types": ["Linear"], "exclude_name_patterns": []},
        engine="torch_int_mm",
        inplace=False,
        source="fp4_weight_only",
    )
    output = result.model(torch.randn(3, 16))
    metadata = result.model.fc.execution_metadata()

    assert isinstance(result.model.fc, W4StorageInt8MmaLinear)
    assert result.metadata["source_fp4_module_count"] == 1
    assert result.metadata["source_linear_module_count"] == 0
    assert output.shape == (3, 32)
    assert metadata["retarget"] == "w4_storage_int8_mma"
    assert metadata["storage_dtype"] == "packed_signed_int4"
    assert metadata["compute_dtype"] == "int8"


def test_session_quant_w4_storage_int8_mma_replaces_linear(tmp_path) -> None:
    session = XQTOptimizationSession(
        project={
            "name": "session_w4_storage_int8_mma",
            "artifact_dir": str(tmp_path / "artifacts"),
        },
        model=_TinyLinearModel().eval(),
        example_inputs=torch.randn(4, 16),
    )

    stage = session.quant(
        name="w4_int8",
        backend="pytorch",
        strategy="w4_storage_int8_mma",
        policy={
            "dtype": "int8",
            "scheme": "w4_storage_int8_mma",
            "include_module_names": ["fc"],
            "engine": "torch_int_mm",
            "group_size": 8,
        },
    )
    output = session.model(torch.randn(4, 16))

    assert stage.accepted is True
    assert isinstance(session.model.fc, W4StorageInt8MmaLinear)
    assert output.shape == (4, 32)
    assert stage.metrics["nature"] == "true"
    assert stage.metrics["algorithm_executable"] is True
    assert stage.metrics["method_semantics"] == "w4_storage_int8_mma_compute_retarget"
    assert stage.metrics["metadata"]["execution_state"] == "w4_storage_int8_mma"
    assert stage.metrics["quantized_modules"] == ["fc"]


def test_w4_storage_release_int8_compute_view_keeps_packed_storage() -> None:
    source = torch.nn.Linear(16, 32, bias=False).eval()
    module = W4StorageInt8MmaLinear.from_linear(
        source,
        group_size=8,
        engine="torch_int_mm",
        cache_int8_compute_view=True,
    ).eval()

    assert module._compute is not None
    packed_before = module.packed_weight.clone()
    module.release_int8_compute_view()
    assert module._compute is None
    assert torch.equal(module.packed_weight, packed_before)

    output = module(torch.randn(2, 16))
    assert output.shape == (2, 32)
    assert module._compute is not None


def test_w4_storage_weight_retarget_error_compares_fp4_to_int8_view() -> None:
    model = _TinyLinearModel().eval()
    fp4 = quantize_with_fp4_weight_only(
        model,
        policy={"include_module_types": ["Linear"], "exclude_name_patterns": [], "group_size": 8},
        inplace=False,
    )
    result = quantize_with_w4_storage_int8_mma(
        fp4.model,
        policy={"include_module_types": ["Linear"], "exclude_name_patterns": []},
        engine="torch_int_mm",
        inplace=False,
        source="fp4_weight_only",
    )

    error = weight_retarget_error(fp4.model, result.model, chunk_rows=5)
    aggregate = error["aggregate"]

    assert aggregate["layers"] == 1
    assert aggregate["numel"] == 16 * 32
    assert aggregate["max_abs"] >= 0.0
    assert aggregate["mean_abs"] < 1e-3
    assert "fc" in error["per_layer"]


def test_w4_storage_benchmark_best_path_prefers_fastest_passing_candidate() -> None:
    candidates = [
        {
            "name": "too_slow",
            "status": "ok",
            "latency_ms": 2.0,
            "gates": {"overall": True},
        },
        {
            "name": "fast_but_failed_gate",
            "status": "ok",
            "latency_ms": 0.5,
            "gates": {"overall": False},
        },
        {
            "name": "fastest_passing",
            "status": "ok",
            "latency_ms": 1.0,
            "gates": {"overall": True},
        },
    ]

    best = _select_best_path(candidates)

    assert best is not None
    assert best["name"] == "fastest_passing"
