import importlib.util

import pytest
import torch

from xqt.quant import FP4DynamicLinear, quantize_with_mxfp4_dynamic, quantize_with_nvfp4_dynamic
from xqt.operator_opt.kernels.fp4_quant_common import unpack_nvfp4_weight_from_tilelang
from xqt.operator_opt.kernels.tilelang._common import tilelang_runtime_usable
from xqt.workflows import XQTOptimizationSession


requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for dynamic FP4 runtime tests",
)

requires_triton = pytest.mark.skipif(
    importlib.util.find_spec("triton") is None,
    reason="triton package is required for Triton dynamic FP4 tests",
)

requires_tilelang = pytest.mark.skipif(
    not tilelang_runtime_usable(),
    reason="a runtime-compatible TileLang adapter is required for TileLang dynamic FP4 tests",
)


class _TinyLinearModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = torch.nn.Linear(16, 32, bias=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.fc(inputs)


class _WideLinearModel(torch.nn.Module):
    def __init__(
        self,
        input_features: int,
        output_features: int,
        *,
        device: str,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.fc = torch.nn.Linear(
            input_features,
            output_features,
            bias=True,
            device=device,
            dtype=dtype,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.fc(inputs)


def _build_cuda_dynamic_fp4_pair(
    *,
    fp4_format: str,
    engine: str,
    batch_size: int,
    input_features: int,
    output_features: int,
) -> tuple[FP4DynamicLinear, FP4DynamicLinear, torch.Tensor]:
    torch.manual_seed(7)
    source = torch.nn.Linear(
        input_features,
        output_features,
        bias=True,
        dtype=torch.float16,
        device="cuda",
    ).eval()
    candidate = FP4DynamicLinear.from_linear(source, fp4_format=fp4_format, engine=engine).cuda().eval()
    reference = FP4DynamicLinear.from_linear(source, fp4_format=fp4_format, engine="torch").cuda().eval()
    inputs = torch.randn(batch_size, input_features, device="cuda", dtype=torch.float16)
    return candidate, reference, inputs


def test_quantize_with_nvfp4_dynamic_replaces_linear_on_cpu() -> None:
    model = _TinyLinearModel().eval()

    result = quantize_with_nvfp4_dynamic(
        model,
        policy={"include_module_types": ["Linear"], "exclude_name_patterns": []},
        engine="torch",
        inplace=False,
    )
    output = result.model(torch.randn(4, 16))

    assert isinstance(result.model.fc, FP4DynamicLinear)
    assert result.model.fc.fp4_format == "nvfp4"
    assert result.quantized_modules == ["fc"]
    assert output.shape == (4, 32)
    assert result.metadata["quantization_nature"] == "pseudo"


def test_quantize_with_mxfp4_dynamic_replaces_linear_on_cpu() -> None:
    model = _TinyLinearModel().eval()

    result = quantize_with_mxfp4_dynamic(
        model,
        policy={"include_module_types": ["Linear"], "exclude_name_patterns": []},
        engine="torch",
        inplace=False,
    )
    output = result.model(torch.randn(4, 16))

    assert isinstance(result.model.fc, FP4DynamicLinear)
    assert result.model.fc.fp4_format == "mxfp4"
    assert result.quantized_modules == ["fc"]
    assert output.shape == (4, 32)
    assert result.metadata["quantization_nature"] == "pseudo"


def test_nvfp4_dynamic_linear_cpu_metadata_reports_reference_path() -> None:
    source = torch.nn.Linear(16, 32, bias=False).eval()
    qlinear = FP4DynamicLinear.from_linear(source, fp4_format="nvfp4", engine="torch").eval()

    _ = qlinear(torch.randn(3, 16))
    metadata = qlinear.execution_metadata()

    assert metadata["engine"] == "torch"
    assert metadata["fp4_format"] == "nvfp4"
    assert metadata["activation_quant_engine"] == "reference_nvfp4_quant"
    assert metadata["weight_gemm_engine"] == "reference_packed_nvfp4_weight_gemm"


def test_mxfp4_dynamic_linear_cpu_metadata_reports_reference_path() -> None:
    source = torch.nn.Linear(16, 32, bias=False).eval()
    qlinear = FP4DynamicLinear.from_linear(source, fp4_format="mxfp4", engine="torch").eval()

    _ = qlinear(torch.randn(3, 16))
    metadata = qlinear.execution_metadata()

    assert metadata["engine"] == "torch"
    assert metadata["fp4_format"] == "mxfp4"
    assert metadata["activation_quant_engine"] == "reference_mxfp4_quant"
    assert metadata["weight_gemm_engine"] == "reference_packed_mxfp4_weight_gemm"
    assert metadata["weight_storage"] == "packed_mxfp4_int4_uint8"


def test_mxfp4_dynamic_linear_reuses_mxfp_weight_storage_contract() -> None:
    source = torch.nn.Linear(16, 32, bias=False).eval()
    qlinear = FP4DynamicLinear.from_linear(source, fp4_format="mxfp4", engine="torch").eval()

    assert qlinear.packed_weight.dtype == torch.uint8
    assert qlinear.weight_scale.dtype == torch.float32
    assert qlinear.group_size == 32
    assert qlinear.padded_input_features == 32


def test_nvfp4_dynamic_linear_prepacks_aligned_weight_for_tilelang_once() -> None:
    source = torch.nn.Linear(128, 64, bias=False).eval()
    qlinear = FP4DynamicLinear.from_linear(
        source,
        fp4_format="nvfp4",
        engine="tilelang",
    ).eval()

    assert qlinear.tilelang_packed_weight is not None
    assert qlinear.tilelang_weight_scale is not None
    assert qlinear.tilelang_packed_weight.shape == (4, 1, 16, 64)
    assert qlinear.tilelang_weight_scale.shape == (4, 1, 16, 8)
    restored_weight, restored_scale = unpack_nvfp4_weight_from_tilelang(
        qlinear.tilelang_packed_weight,
        qlinear.tilelang_weight_scale,
    )
    assert torch.equal(restored_weight, qlinear.packed_weight)
    assert torch.equal(restored_scale, qlinear.weight_scale)


def test_nvfp4_dynamic_linear_skips_tilelang_prepack_when_k_is_unaligned() -> None:
    source = torch.nn.Linear(64, 64, bias=False).eval()
    qlinear = FP4DynamicLinear.from_linear(
        source,
        fp4_format="nvfp4",
        engine="tilelang",
    ).eval()

    assert qlinear.tilelang_packed_weight is None
    assert qlinear.tilelang_weight_scale is None


def test_session_quant_nvfp4_dynamic_replaces_linear(tmp_path) -> None:
    session = XQTOptimizationSession(
        project={
            "name": "session_nvfp4_dynamic",
            "artifact_dir": str(tmp_path / "artifacts"),
        },
        model=_TinyLinearModel().eval(),
        example_inputs=torch.randn(4, 16),
    )

    stage = session.quant(
        name="nvfp4_dynamic",
        backend="pytorch",
        strategy="nvfp4_dynamic",
        policy={
            "dtype": "nvfp4",
            "scheme": "dynamic",
            "include_module_names": ["fc"],
            "engine": "torch",
        },
    )
    output = session.model(torch.randn(4, 16))

    assert stage.accepted is True
    assert isinstance(session.model.fc, FP4DynamicLinear)
    assert output.shape == (4, 32)
    assert stage.metrics["metadata"]["execution_state"] == "nvfp4_dynamic"
    assert stage.metrics["metadata"]["algorithm_executable"] is True


def test_session_quant_mxfp4_dynamic_replaces_linear(tmp_path) -> None:
    session = XQTOptimizationSession(
        project={
            "name": "session_mxfp4_dynamic",
            "artifact_dir": str(tmp_path / "artifacts"),
        },
        model=_TinyLinearModel().eval(),
        example_inputs=torch.randn(4, 16),
    )

    stage = session.quant(
        name="mxfp4_dynamic",
        backend="pytorch",
        strategy="mxfp4_dynamic",
        policy={
            "dtype": "mxfp4",
            "scheme": "dynamic",
            "include_module_names": ["fc"],
            "engine": "torch",
        },
    )
    output = session.model(torch.randn(4, 16))

    assert stage.accepted is True
    assert isinstance(session.model.fc, FP4DynamicLinear)
    assert output.shape == (4, 32)
    assert stage.metrics["metadata"]["execution_state"] == "mxfp4_dynamic"
    assert stage.metrics["metadata"]["algorithm_executable"] is True


@requires_cuda
@requires_tilelang
def test_session_quant_nvfp4_dynamic_cuda_tilelang_executes_runtime_fastpath(tmp_path) -> None:
    session = XQTOptimizationSession(
        project={
            "name": "session_nvfp4_dynamic_cuda_tilelang",
            "artifact_dir": str(tmp_path / "artifacts"),
        },
        model=_WideLinearModel(128, 64, device="cuda", dtype=torch.float16).eval(),
        example_inputs=torch.randn(64, 128, device="cuda", dtype=torch.float16),
    )

    stage = session.quant(
        name="nvfp4_dynamic_cuda_tilelang",
        backend="pytorch",
        strategy="nvfp4_dynamic",
        policy={
            "dtype": "nvfp4",
            "scheme": "dynamic",
            "include_module_names": ["fc"],
            "engine": "tilelang",
        },
    )
    output = session.model(torch.randn(64, 128, device="cuda", dtype=torch.float16))
    torch.cuda.synchronize()
    metadata = session.model.fc.execution_metadata()

    assert stage.accepted is True
    assert isinstance(session.model.fc, FP4DynamicLinear)
    assert output.shape == (64, 64)
    assert metadata["engine"] == "tilelang"
    assert metadata["activation_quant_engine"] == "tilelang_nvfp4_quant"
    assert metadata["weight_gemm_engine"] == "tilelang_prepacked_nvfp4_weight_gemm"
    assert metadata["weight_gemm_entry"] == "packed_activation_api"
    assert metadata["fallback_count"] == 0
    assert metadata["tilelang_weight_layout"] == "cta_tiled_n16_k128"


@requires_cuda
@requires_triton
def test_session_quant_mxfp4_dynamic_cuda_triton_executes_runtime_fastpath(tmp_path) -> None:
    session = XQTOptimizationSession(
        project={
            "name": "session_mxfp4_dynamic_cuda_triton",
            "artifact_dir": str(tmp_path / "artifacts"),
        },
        model=_WideLinearModel(64, 64, device="cuda", dtype=torch.float16).eval(),
        example_inputs=torch.randn(64, 64, device="cuda", dtype=torch.float16),
    )

    stage = session.quant(
        name="mxfp4_dynamic_cuda_triton",
        backend="pytorch",
        strategy="mxfp4_dynamic",
        policy={
            "dtype": "mxfp4",
            "scheme": "dynamic",
            "include_module_names": ["fc"],
            "engine": "triton",
        },
    )
    output = session.model(torch.randn(64, 64, device="cuda", dtype=torch.float16))
    torch.cuda.synchronize()
    metadata = session.model.fc.execution_metadata()

    assert stage.accepted is True
    assert isinstance(session.model.fc, FP4DynamicLinear)
    assert output.shape == (64, 64)
    assert metadata["engine"] == "triton"
    assert metadata["activation_quant_engine"] == "triton_mxfp4_quant"
    assert metadata["weight_gemm_engine"] == "triton_packed_mxfp4_weight_gemm"
    assert metadata["weight_gemm_entry"] == "packed_activation_api"
    assert metadata["fallback_count"] == 0


@requires_cuda
@requires_triton
@pytest.mark.parametrize(
    ("fp4_format", "input_features", "output_features", "expected_quant_engine", "expected_weight_engine"),
    [
        ("nvfp4", 128, 64, "triton_nvfp4_quant", "triton_packed_nvfp4_weight_gemm"),
        ("mxfp4", 64, 64, "triton_mxfp4_quant", "triton_packed_mxfp4_weight_gemm"),
    ],
)
def test_fp4_dynamic_linear_triton_cuda_fastpath_matches_reference(
    fp4_format: str,
    input_features: int,
    output_features: int,
    expected_quant_engine: str,
    expected_weight_engine: str,
) -> None:
    qlinear, reference, inputs = _build_cuda_dynamic_fp4_pair(
        fp4_format=fp4_format,
        engine="triton",
        batch_size=64,
        input_features=input_features,
        output_features=output_features,
    )

    actual = qlinear(inputs)
    expected = reference(inputs)
    torch.cuda.synchronize()
    metadata = qlinear.execution_metadata()

    assert actual.shape == (64, output_features)
    torch.testing.assert_close(actual.float(), expected.float(), atol=2e-2, rtol=2e-2)
    assert metadata["engine"] == "triton"
    assert metadata["requested_engine"] == "triton"
    assert metadata["execution_mode"] == "cuda_triton_entry"
    assert metadata["activation_quant_engine"] == expected_quant_engine
    assert metadata["weight_gemm_engine"] == expected_weight_engine
    assert metadata["weight_gemm_entry"] == "packed_activation_api"
    assert metadata["fallback_count"] == 0
    assert metadata["runtime_fallbacks"] == []


@requires_cuda
@requires_tilelang
@pytest.mark.parametrize(
    ("fp4_format", "input_features", "output_features", "expected_quant_engine", "expected_weight_engine"),
    [
        ("nvfp4", 128, 64, "tilelang_nvfp4_quant", "tilelang_prepacked_nvfp4_weight_gemm"),
        ("mxfp4", 64, 64, "tilelang_mxfp4_quant", "tilelang_packed_mxfp4_weight_gemm"),
    ],
)
def test_fp4_dynamic_linear_tilelang_cuda_fastpath_matches_reference(
    fp4_format: str,
    input_features: int,
    output_features: int,
    expected_quant_engine: str,
    expected_weight_engine: str,
) -> None:
    qlinear, reference, inputs = _build_cuda_dynamic_fp4_pair(
        fp4_format=fp4_format,
        engine="tilelang",
        batch_size=64,
        input_features=input_features,
        output_features=output_features,
    )

    actual = qlinear(inputs)
    expected = reference(inputs)
    torch.cuda.synchronize()
    metadata = qlinear.execution_metadata()

    assert actual.shape == (64, output_features)
    torch.testing.assert_close(actual.float(), expected.float(), atol=2e-2, rtol=2e-2)
    assert metadata["engine"] == "tilelang"
    assert metadata["requested_engine"] == "tilelang"
    assert metadata["execution_mode"] == "cuda_tilelang_entry"
    assert metadata["activation_quant_engine"] == expected_quant_engine
    assert metadata["weight_gemm_engine"] == expected_weight_engine
    assert metadata["weight_gemm_entry"] == "packed_activation_api"
    assert metadata["fallback_count"] == 0
    assert metadata["runtime_fallbacks"] == []
    if fp4_format == "nvfp4":
        assert metadata["tilelang_weight_layout"] == "cta_tiled_n16_k128"


@requires_cuda
@requires_tilelang
@pytest.mark.parametrize(
    ("fp4_format", "input_features", "output_features", "expected_quant_engine", "expected_weight_engine"),
    [
        ("nvfp4", 128, 64, "triton_nvfp4_quant", "triton_packed_nvfp4_weight_gemm"),
        ("mxfp4", 64, 64, "triton_mxfp4_quant", "triton_packed_mxfp4_weight_gemm"),
    ],
)
def test_fp4_dynamic_linear_tilelang_cuda_falls_back_to_triton_when_shape_unaligned(
    fp4_format: str,
    input_features: int,
    output_features: int,
    expected_quant_engine: str,
    expected_weight_engine: str,
) -> None:
    qlinear, reference, inputs = _build_cuda_dynamic_fp4_pair(
        fp4_format=fp4_format,
        engine="tilelang",
        batch_size=5,
        input_features=input_features,
        output_features=output_features,
    )

    actual = qlinear(inputs)
    expected = reference(inputs)
    torch.cuda.synchronize()
    metadata = qlinear.execution_metadata()

    assert actual.shape == (5, output_features)
    torch.testing.assert_close(actual.float(), expected.float(), atol=2e-2, rtol=2e-2)
    assert metadata["engine"] == "triton"
    assert metadata["requested_engine"] == "tilelang"
    assert metadata["engine_candidates"] == ["tilelang", "triton", "torch"]
    assert metadata["execution_mode"] == "cuda_triton_entry"
    assert metadata["activation_quant_engine"] == expected_quant_engine
    assert metadata["weight_gemm_engine"] == expected_weight_engine
    assert metadata["weight_gemm_entry"] == "packed_activation_api"
    assert metadata["fallback_count"] == 1
    assert metadata["runtime_fallbacks"][0]["engine"] == "tilelang"
    assert metadata["runtime_fallbacks"][0]["stage"] == "weight_gemm"
    assert "tilelang weight_gemm runtime fallback" in metadata["fallback_reason"]


@requires_cuda
@requires_triton
@requires_tilelang
@pytest.mark.parametrize(
    ("fp4_format", "input_features", "output_features", "expected_quant_engine", "expected_weight_engine"),
    [
        ("nvfp4", 128, 64, "triton_nvfp4_quant", "triton_packed_nvfp4_weight_gemm"),
        ("mxfp4", 64, 64, "triton_mxfp4_quant", "triton_packed_mxfp4_weight_gemm"),
    ],
)
def test_fp4_dynamic_linear_auto_cuda_falls_back_from_tilelang_to_triton(
    fp4_format: str,
    input_features: int,
    output_features: int,
    expected_quant_engine: str,
    expected_weight_engine: str,
) -> None:
    qlinear, reference, inputs = _build_cuda_dynamic_fp4_pair(
        fp4_format=fp4_format,
        engine="auto",
        batch_size=5,
        input_features=input_features,
        output_features=output_features,
    )

    actual = qlinear(inputs)
    expected = reference(inputs)
    torch.cuda.synchronize()
    metadata = qlinear.execution_metadata()

    assert actual.shape == (5, output_features)
    torch.testing.assert_close(actual.float(), expected.float(), atol=2e-2, rtol=2e-2)
    assert metadata["engine"] == "triton"
    assert metadata["requested_engine"] == "auto"
    assert metadata["engine_candidates"] == ["tilelang", "triton", "torch"]
    assert metadata["execution_mode"] == "cuda_triton_entry"
    assert metadata["activation_quant_engine"] == expected_quant_engine
    assert metadata["weight_gemm_engine"] == expected_weight_engine
    assert metadata["weight_gemm_entry"] == "packed_activation_api"
    assert metadata["fallback_count"] == 1
    assert metadata["runtime_fallbacks"][0]["engine"] == "tilelang"
    assert metadata["runtime_fallbacks"][0]["stage"] == "weight_gemm"
    assert "tilelang weight_gemm runtime fallback" in metadata["fallback_reason"]
