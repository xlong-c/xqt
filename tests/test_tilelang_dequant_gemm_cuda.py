from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path

import pytest
import torch

import xqt.operator_opt.kernels.tilelang.gemm as tilelang_gemm_module
from xqt.operator_opt.backends.tilelang import list_tilelang_kernel_specs
from xqt.operator_opt.backends.tilelang_validation import (
    TileLangFP4ValidationResult,
    validate_tilelang_packed_fp4_fused_gemm,
)
from xqt.operator_opt.kernels.tilelang import (
    build_tilelang_fp4_fused_dequant_gemm_kernel,
)
from xqt.operator_opt.kernels.tilelang._common import tilelang_runtime_usable
from xqt.operator_opt.kernels.tilelang.gemm import (
    dequant_gemm_epilogue_reference,
    dequant_gemm_epilogue_tilelang,
    fp4_packed_dequant_gemm_epilogue_reference,
    fp4_packed_dequant_gemm_epilogue_tilelang,
    mxfp4_packed_activation_gemm_epilogue_reference,
    mxfp4_packed_activation_gemm_epilogue_tilelang,
    nvfp4_packed_activation_gemm_epilogue_reference,
    nvfp4_packed_activation_gemm_epilogue_tilelang,
)
from xqt.operator_opt.kernels.tilelang.fp4_quant import (
    scaled_mxfp4_quant_reference,
    scaled_nvfp4_quant_reference,
)
from xqt.quant.quantizers.fp4_weight_only import FP4WeightOnlyLinear
from xqt.quant import MXFPWeightOnlyLinear, bridge_module_to_nvfp4_linear


class _ExternalCompressedNVFP4Linear(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.in_features = 64
        self.out_features = 64
        self.register_parameter(
            "weight_packed",
            torch.nn.Parameter(torch.full((64, 32), 0x21, dtype=torch.uint8), requires_grad=False),
        )
        self.register_parameter(
            "weight_scale",
            torch.nn.Parameter(
                torch.ones((64, 4), dtype=torch.float32).to(torch.float8_e4m3fn),
                requires_grad=False,
            ),
        )
        self.register_parameter(
            "weight_global_scale",
            torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float32), requires_grad=False),
        )
        self.register_parameter(
            "bias",
            torch.nn.Parameter(torch.zeros(64, dtype=torch.float32), requires_grad=False),
        )


requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for TileLang dequant GEMM CUDA test",
)

requires_tilelang = pytest.mark.skipif(
    not tilelang_runtime_usable(),
    reason="a runtime-compatible TileLang adapter is required for TileLang dequant GEMM CUDA test",
)

requires_nvcc = pytest.mark.skipif(
    shutil.which("nvcc") is None and not Path("/usr/local/cuda/bin/nvcc").exists(),
    reason="nvcc is required for TileLang compile-only tests",
)


def test_tilelang_fp4_registry_reports_tilelang_unpack_stage() -> None:
    specs = list_tilelang_kernel_specs()
    metadata = specs["fp4_packed_dequant_gemm_epilogue"]["metadata"]

    assert metadata["unpack_stage"] == "tilelang_fused_gemm_kernel"
    assert (
        metadata["fusion_status"]
        == "single_tilelang_kernel_for_unpack_dequant_gemm_epilogue"
    )
    assert metadata["epilogue_stage"] == "tilelang_fused_bias_activation"


def test_tilelang_packed_fp4_validation_reports_skipped_without_cuda() -> None:
    result = validate_tilelang_packed_fp4_fused_gemm(
        target_arch="sm_80",
        compile_only=False,
        warmup=0,
        iterations=1,
    )

    assert isinstance(result, TileLangFP4ValidationResult)
    assert result.compile_only is False
    assert result.shape["m"] == 64
    assert result.shape["out_features"] == 64
    if torch.cuda.is_available():
        assert result.status in {"ok", "error"}
        assert result.device == "cuda"
        assert result.compile_status in {"ok", "error"}
    else:
        assert result.status == "skipped"
        assert result.compile_status == "not_requested"
        assert "CUDA runtime is not available" in str(result.reason)


@requires_tilelang
@requires_nvcc
def test_tilelang_packed_fp4_fused_epilogue_kernel_compiles_with_explicit_arch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("TILELANG_CACHE_DIR", str(tmp_path / "tilelang-cache"))

    kernel = build_tilelang_fp4_fused_dequant_gemm_kernel(
        64,
        64,
        32,
        16,
        target_arch="sm_80",
        has_bias=True,
        activation="silu",
    )

    assert callable(kernel)


@requires_tilelang
@requires_nvcc
def test_tilelang_packed_fp4_validation_compile_only(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("TILELANG_CACHE_DIR", str(tmp_path / "tilelang-cache"))

    result = validate_tilelang_packed_fp4_fused_gemm(
        target_arch="sm_80",
        compile_only=True,
        warmup=0,
        iterations=1,
    )

    assert result.status == "ok"
    assert result.compile_only is True
    assert result.compile_status == "ok"
    assert result.target_arch == "sm_80"
    assert result.allclose is None


@requires_cuda
@requires_tilelang
def test_tilelang_dequant_gemm_cuda_matches_reference() -> None:
    torch.manual_seed(0)
    x = torch.randn(64, 32, device="cuda", dtype=torch.float16)
    qweight = torch.randn(64, 32, device="cuda", dtype=torch.float16)
    scale = torch.randn(64, device="cuda", dtype=torch.float16).abs() + 0.01
    bias = torch.randn(64, device="cuda", dtype=torch.float16)

    output = dequant_gemm_epilogue_tilelang(
        x,
        qweight,
        scale,
        bias,
        activation="silu",
        block_m=64,
        block_n=64,
        threads=128,
        num_stages=2,
    )
    reference = dequant_gemm_epilogue_reference(
        x,
        qweight,
        scale,
        bias,
        activation="silu",
    )

    assert output.shape == reference.shape
    assert output.dtype == torch.float16
    assert torch.allclose(output.float(), reference.float(), atol=1e-2, rtol=1e-2)


@requires_cuda
@requires_tilelang
def test_tilelang_packed_fp4_validation_cuda_runtime() -> None:
    result = validate_tilelang_packed_fp4_fused_gemm(
        target_arch=None,
        compile_only=False,
        warmup=1,
        iterations=2,
    )

    assert result.status == "ok"
    assert result.device == "cuda"
    assert result.compile_status == "ok"
    assert result.allclose is True
    assert result.max_abs_error is not None
    assert result.mean_abs_error is not None
    assert result.latency_ms_tilelang is not None
    assert result.latency_ms_reference is not None


@requires_cuda
@requires_tilelang
def test_tilelang_packed_fp4_dequant_gemm_cuda_matches_reference() -> None:
    torch.manual_seed(1)
    linear = torch.nn.Linear(32, 64)
    fp4_linear = FP4WeightOnlyLinear.from_linear(linear, group_size=16).to("cuda")
    x = torch.randn(64, 32, device="cuda", dtype=torch.float16)
    packed_weight = fp4_linear.packed_weight
    scale = fp4_linear.weight_scale.to(dtype=torch.float16)
    bias = (
        fp4_linear.bias.to(dtype=torch.float16) if fp4_linear.bias is not None else None
    )

    output = fp4_packed_dequant_gemm_epilogue_tilelang(
        x,
        packed_weight,
        scale,
        bias,
        input_features=fp4_linear.input_features,
        group_size=fp4_linear.group_size,
        block_m=64,
        block_n=64,
        threads=128,
        num_stages=2,
    )
    reference = fp4_packed_dequant_gemm_epilogue_reference(
        x,
        packed_weight,
        scale,
        bias,
        input_features=fp4_linear.input_features,
        group_size=fp4_linear.group_size,
    )

    assert output.shape == reference.shape
    assert output.dtype == torch.float16
    assert torch.allclose(output.float(), reference.float(), atol=1e-2, rtol=1e-2)


@requires_cuda
@requires_tilelang
def test_tilelang_nvfp4_packed_activation_gemm_cuda_matches_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(0)
    module = _ExternalCompressedNVFP4Linear().to("cuda").eval()
    bridge = bridge_module_to_nvfp4_linear(module)
    assert bridge is not None
    x = torch.randn(64, 64, device="cuda", dtype=torch.float16)
    activation_global_scale = (
        2688.0 / x.detach().to(torch.float32).abs().amax().clamp_min(1e-8)
    ).reshape(1).to(device="cuda", dtype=torch.float32)
    packed_activation, activation_scale = scaled_nvfp4_quant_reference(
        x,
        activation_global_scale,
        group_size=16,
    )
    expected = nvfp4_packed_activation_gemm_epilogue_reference(
        packed_activation,
        activation_scale,
        module.weight_packed.detach(),
        module.weight_scale.detach().to(torch.float16),
        module.bias.detach().to(torch.float16),
        input_features=module.in_features,
        group_size=16,
        activation_global_scale=activation_global_scale,
        weight_global_scale=module.weight_global_scale.detach(),
        activation=None,
        output_dtype=torch.float16,
    )

    def _forbidden_dequant(*args: object, **kwargs: object) -> torch.Tensor:
        raise AssertionError("common dequantize_nvfp4_codes should not be used in TileLang packed-activation fastpath")

    def _forbidden_tilelang_unpack(
        *args: object, **kwargs: object
    ) -> torch.Tensor:
        raise AssertionError(
            "two-stage TileLang unpack should not be used in packed-activation fastpath"
        )

    monkeypatch.setattr(tilelang_gemm_module, "dequantize_nvfp4_codes", _forbidden_dequant)
    monkeypatch.setattr(
        tilelang_gemm_module,
        "_tilelang_unpack_dequant_fp4_codes",
        _forbidden_tilelang_unpack,
    )
    actual = nvfp4_packed_activation_gemm_epilogue_tilelang(
        packed_activation,
        activation_scale,
        module.weight_packed.detach(),
        module.weight_scale.detach().to(torch.float16),
        module.bias.detach().to(torch.float16),
        input_features=module.in_features,
        group_size=16,
        activation_global_scale=activation_global_scale,
        weight_global_scale=module.weight_global_scale.detach(),
        activation=None,
        block_m=64,
        block_n=16,
        block_k=64,
        threads=128,
        num_stages=2,
        target_arch=None,
        output_dtype=torch.float16,
    )

    assert torch.allclose(actual.float(), expected.float(), atol=2e-2, rtol=2e-2)


@requires_cuda
@requires_tilelang
def test_tilelang_mxfp4_packed_activation_gemm_cuda_matches_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(0)
    module = MXFPWeightOnlyLinear.from_linear(
        torch.nn.Linear(64, 64),
        mx_precision=4,
        block_size=32,
    ).to("cuda").eval()
    x = torch.randn(64, 64, device="cuda", dtype=torch.float16)
    packed_activation, activation_scale = scaled_mxfp4_quant_reference(
        x,
        group_size=32,
    )
    expected = mxfp4_packed_activation_gemm_epilogue_reference(
        packed_activation,
        activation_scale,
        module.packed_weight,
        module.weight_scale,
        None if module.bias is None else module.bias.to(device="cuda", dtype=torch.float16),
        input_features=module.input_features,
        group_size=module.block_size,
        activation=None,
        output_dtype=torch.float16,
    )

    def _forbidden_dequant(*args: object, **kwargs: object) -> torch.Tensor:
        raise AssertionError("common dequantize_nvfp4_codes should not be used in TileLang packed-activation fastpath")

    def _forbidden_tilelang_unpack(
        *args: object, **kwargs: object
    ) -> torch.Tensor:
        raise AssertionError(
            "two-stage TileLang unpack should not be used in packed-activation fastpath"
        )

    monkeypatch.setattr(tilelang_gemm_module, "dequantize_nvfp4_codes", _forbidden_dequant)
    monkeypatch.setattr(
        tilelang_gemm_module,
        "_tilelang_unpack_dequant_fp4_codes",
        _forbidden_tilelang_unpack,
    )
    actual = mxfp4_packed_activation_gemm_epilogue_tilelang(
        packed_activation,
        activation_scale,
        module.packed_weight,
        module.weight_scale,
        None if module.bias is None else module.bias.to(device="cuda", dtype=torch.float16),
        input_features=module.input_features,
        group_size=module.block_size,
        activation=None,
        block_m=64,
        block_n=64,
        threads=128,
        num_stages=2,
        target_arch=None,
        output_dtype=torch.float16,
    )

    assert torch.allclose(actual.float(), expected.float(), atol=2e-2, rtol=2e-2)
