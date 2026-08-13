import pytest
import torch

from xqt.core.errors import XQTBackendError
from xqt.quant.quantizers.int8_mma import Int8MmaLinear, quantize_with_int8_mma
from xqt.workflows import XQTOptimizationSession

from examples.unlimited_ocr_int8_inference import _ocr_text_quality_gate


class _TinyLinearModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = torch.nn.Linear(16, 32, bias=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.fc(inputs)


def test_quantize_with_int8_mma_replaces_linear_on_cpu() -> None:
    model = _TinyLinearModel().eval()

    result = quantize_with_int8_mma(
        model,
        policy={"include_module_types": ["Linear"], "exclude_name_patterns": []},
        engine="torch_int_mm",
        inplace=False,
    )
    output = result.model(torch.randn(4, 16))

    assert isinstance(result.model.fc, Int8MmaLinear)
    assert result.quantized_modules == ["fc"]
    assert output.shape == (4, 32)
    assert result.metadata["quantization_nature"] == "true"
    description = result.metadata["precision_description"]
    assert description["quantization_time"]["weight"] == (
        "offline static signed INT8 per output channel"
    )
    assert description["runtime"]["small_batch_float_fallback"] == {
        "enabled": False,
        "condition": "input_rows < min_int8_rows",
        "note": "quantize_with_int8_mma constructs min_int8_rows=0",
    }


def test_session_quant_dynamic_int8_mma_replaces_linear(tmp_path) -> None:
    session = XQTOptimizationSession(
        project={
            "name": "session_dynamic_int8_mma",
            "artifact_dir": str(tmp_path / "artifacts"),
        },
        model=_TinyLinearModel().eval(),
        example_inputs=torch.randn(4, 16),
    )

    stage = session.quant(
        name="int8_mma",
        backend="pytorch",
        method=None,
        strategy="w8a8_int8",
        compute="w8a8_int8_mma",
        policy={
            "dtype": "int8",
            "scheme": "dynamic_mma",
            "include_module_names": ["fc"],
            "engine": "torch_int_mm",
        },
    )
    output = session.model(torch.randn(4, 16))

    assert stage.accepted is True
    assert isinstance(session.model.fc, Int8MmaLinear)
    assert output.shape == (4, 32)
    assert stage.metrics["nature"] == "true"
    assert stage.metrics["algorithm_executable"] is True
    assert stage.metrics["method_semantics"] == (
        "w8a8_int8_mma_runtime_quantization_contract"
    )
    assert stage.metrics["metadata"]["execution_state"] == "w8a8_int8"
    assert stage.metrics["metadata"]["algorithm_executable"] is True
    assert stage.metrics["quantized_modules"] == ["fc"]


def test_int8_mma_linear_cpu_metadata_reports_reference_not_true_mma() -> None:
    source = torch.nn.Linear(16, 32, bias=False).eval()
    qlinear = Int8MmaLinear.from_linear(source, engine="torch_int_mm").eval()

    _ = qlinear(torch.randn(3, 16))
    metadata = qlinear.execution_metadata()

    assert metadata["engine"] == "torch_int_mm"
    assert metadata["true_int8_mma"] is False
    assert metadata["activation_dtype"] == "int8"
    assert metadata["weight_dtype"] == "int8"
    assert metadata["activation_scale_mode"] == "dynamic"
    assert metadata["runtime_precision"] == {
        "requested": "w8a8_int8_mma",
        "weight_storage": "signed_int8_per_output_channel",
        "activation_encoding": "signed_int8_per_tensor",
        "activation_granularity": "per_tensor",
        "execution_kind": "w8a8_int8_reference",
        "int8_operands_executed": True,
        "native_mma_executed": False,
        "float_fallback_taken": False,
        "float_fallback_reason": None,
        "small_batch_float_fallback": {
            "enabled": False,
            "condition": "input_rows < min_int8_rows",
            "min_int8_rows": 0,
            "status": "disabled",
        },
        "engine_error_int8_fallback": {
            "enabled": True,
            "condition": "selected non-reference INT8 engine raises or fails its runtime check",
            "status": "not_taken",
        },
    }


def test_int8_mma_linear_reports_taken_small_batch_float_fallback() -> None:
    source = torch.nn.Linear(16, 32, bias=False).eval()
    qlinear = Int8MmaLinear.from_linear(
        source,
        engine="torch_int_mm",
        min_int8_rows=4,
    ).eval()

    _ = qlinear(torch.randn(3, 16))
    metadata = qlinear.execution_metadata()

    assert metadata["engine"] == "bf16_fallback"
    assert metadata["runtime_precision"]["execution_kind"] == "floating_point_fallback"
    assert metadata["runtime_precision"]["int8_operands_executed"] is False
    assert metadata["runtime_precision"]["native_mma_executed"] is False
    assert metadata["runtime_precision"]["float_fallback_taken"] is True
    assert metadata["runtime_precision"]["float_fallback_reason"] == (
        "rows_below_min_int8_rows"
    )
    assert metadata["runtime_precision"]["small_batch_float_fallback"]["status"] == "taken"


def test_int8_mma_linear_static_activation_scale_cpu() -> None:
    source = torch.nn.Linear(16, 32, bias=False).eval()
    qlinear = Int8MmaLinear.from_linear(
        source,
        engine="torch_int_mm",
        activation_scale_mode="static",
    ).eval()
    inputs = torch.randn(3, 16)

    with pytest.raises(XQTBackendError, match="static activation scale mode requires"):
        qlinear(inputs)

    scale = qlinear.calibrate_static_activation_scale(inputs)
    output = qlinear(inputs)
    metadata = qlinear.execution_metadata()

    assert output.shape == (3, 32)
    assert scale.item() > 0.0
    assert metadata["activation_scale_mode"] == "static"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_int8_mma_auto_static_uses_fastpath_cuda() -> None:
    source = torch.nn.Linear(128, 96, bias=True, dtype=torch.float16, device="cuda").eval()
    qlinear = Int8MmaLinear.from_linear(
        source,
        engine="auto",
        activation_scale_mode="static",
        activation_scale=0.02,
    ).eval()

    output = qlinear(torch.randn(64, 128, device="cuda", dtype=torch.float16))
    torch.cuda.synchronize()
    metadata = qlinear.execution_metadata()

    assert output.dtype == torch.float16
    from xqt.operator_opt.kernels.cute.int8mma_binding import int8mma_available

    expected_engine = (
        "cuda_sm89"
        if torch.cuda.get_device_capability() == (8, 9) and int8mma_available()
        else "triton"
    )
    assert metadata["engine"] == expected_engine
    assert metadata["runtime_precision"]["native_mma_executed"] is True
    assert metadata["fused_static_status"] == (
        "two_kernel_quant_then_cuda_gemm"
        if expected_engine == "cuda_sm89"
        else "two_kernel_quant_then_gemm"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_int8_mma_cuda_sm89_static_path_caches_fused_scale_bias() -> None:
    from xqt.operator_opt.kernels.cute.int8mma_binding import int8mma_available

    if torch.cuda.get_device_capability() != (8, 9) or not int8mma_available():
        pytest.skip("cuda_sm89 CUTLASS extension unavailable")

    torch.manual_seed(41)
    source = torch.nn.Linear(256, 256, bias=True, dtype=torch.float16, device="cuda").eval()
    qlinear = Int8MmaLinear.from_linear(
        source,
        engine="cuda_sm89",
        activation_scale_mode="static",
        activation_scale=0.02,
    ).eval()
    inputs = torch.randn(64, 256, device="cuda", dtype=torch.float16).clamp(-2, 2)

    output = qlinear(inputs)
    cached = qlinear._cuda_sm89_scale_bias_cache
    assert cached is not None
    cached_ptr = cached.data_ptr()
    output_again = qlinear(inputs)
    torch.cuda.synchronize()
    metadata = qlinear.execution_metadata()
    qactivation = torch.round(inputs.float() / 0.02).clamp(-127, 127).to(torch.int8)
    expected = (
        torch._int_mm(qactivation, qlinear.qweight_t).float()
        * (0.02 * qlinear.weight_scale).view(1, -1)
        + qlinear.bias.view(1, -1)
    ).half()

    assert metadata["engine"] == "cuda_sm89"
    assert metadata["activation_quant_engine"] == "tilelang_static"
    assert metadata["fused_static_status"] == "two_kernel_quant_then_cuda_gemm"
    assert metadata["prepacked_b"] is True
    assert qlinear._cuda_sm89_scale_bias_cache is not None
    assert qlinear._cuda_sm89_scale_bias_cache.data_ptr() == cached_ptr
    assert (output.float() - expected.float()).abs().max().item() < 2e-3
    assert torch.equal(output, output_again)


def test_quantize_with_int8_mma_accepts_static_activation_scales() -> None:
    model = _TinyLinearModel().eval()

    result = quantize_with_int8_mma(
        model,
        policy={"include_module_types": ["Linear"], "exclude_name_patterns": []},
        engine="torch_int_mm",
        inplace=False,
        activation_scale_mode="static",
        activation_scales={"fc": 0.02},
    )
    output = result.model(torch.randn(4, 16))

    assert output.shape == (4, 32)
    assert result.metadata["activation_scale_mode"] == "static"
    assert result.metadata["activation_encoding"] == "static_signed_int8_per_tensor"
    assert result.metadata["static_scale_module_count"] == 1
    assert result.metadata["dynamic_fallback_module_count"] == 0


def test_quantize_with_int8_mma_static_mode_falls_back_without_scale() -> None:
    model = _TinyLinearModel().eval()

    result = quantize_with_int8_mma(
        model,
        policy={"include_module_types": ["Linear"], "exclude_name_patterns": []},
        engine="torch_int_mm",
        inplace=False,
        activation_scale_mode="static",
        activation_scales={},
    )
    output = result.model(torch.randn(4, 16))
    metadata = result.model.fc.execution_metadata()

    assert output.shape == (4, 32)
    assert result.metadata["static_scale_module_count"] == 0
    assert result.metadata["dynamic_fallback_module_count"] == 1
    assert metadata["activation_scale_mode"] == "dynamic"


def test_quantize_with_int8_mma_include_only_restricts_name_patterns() -> None:
    model = torch.nn.Sequential(
        torch.nn.Linear(16, 32, bias=False),
        torch.nn.Linear(32, 32, bias=False),
    ).eval()

    result = quantize_with_int8_mma(
        model,
        policy={
            "include_module_types": ["Linear"],
            "include_name_patterns": [r"^0$"],
            "exclude_name_patterns": [],
            "selection_mode": "include_only",
        },
        engine="torch_int_mm",
        inplace=False,
    )

    assert isinstance(result.model[0], Int8MmaLinear)
    assert isinstance(result.model[1], torch.nn.Linear)
    assert result.quantized_modules == ["0"]
    assert result.metadata["selection_mode"] == "include_only"


def test_unlimited_ocr_text_quality_gate_allows_minor_layout_drift() -> None:
    reference = (
        "<|det|>text [90, 84, 275, 99]<|/det|>"
        "Unlimited-OCRINT8 MMA smoke image "
        "<|det|>text [90, 141, 263, 156]<|/det|>"
        "This file is generated under artifacts."
    )
    candidate = (
        "<|det|>text [90, 84, 275, 99]<|/det|>"
        "Unlimited-OCR INT8 MMA smoke image "
        "<|det|>text [90, 141, 263, 155]<|/det|>"
        "This file is generated under artifacts/"
    )

    result = _ocr_text_quality_gate(
        reference=reference,
        candidate=candidate,
        min_text_similarity=0.98,
        required_keywords=(
            "Unlimited-OCR INT8 MMA smoke image",
            "This file is generated under artifacts",
        ),
    )

    assert result["passed"] is True
    assert result["exact_match"] is False
    assert result["keywords_passed"] is True
    assert result["text_similarity"] >= 0.98


def test_unlimited_ocr_text_quality_gate_rejects_missing_content() -> None:
    result = _ocr_text_quality_gate(
        reference="Unlimited-OCR INT8 MMA smoke image",
        candidate="</td></tr></table>",
        min_text_similarity=0.98,
        required_keywords=("Unlimited-OCR INT8 MMA smoke image",),
    )

    assert result["passed"] is False
    assert result["keywords_passed"] is False
    assert result["missing_keywords"] == ["Unlimited-OCR INT8 MMA smoke image"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for true INT8 MMA")
def test_int8_mma_linear_tilelang_cuda_reports_true_mma() -> None:
    source = torch.nn.Linear(64, 64, bias=False, dtype=torch.bfloat16, device="cuda").eval()
    qlinear = Int8MmaLinear.from_linear(source, engine="tilelang").cuda().eval()

    output = qlinear(torch.randn(5, 64, device="cuda", dtype=torch.bfloat16))
    torch.cuda.synchronize()
    metadata = qlinear.execution_metadata()

    assert output.shape == (5, 64)
    assert metadata["engine"] == "tilelang"
    assert metadata["true_int8_mma"] is True
    assert metadata["padded_rows"] == 64
