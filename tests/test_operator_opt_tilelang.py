import pytest
import torch
import torch.nn.functional as F

from xqt.core.errors import XQTBackendError
from xqt.operator_opt.backends.tilelang import (
    TileLangCompileSettings,
    build_tilelang_artifact_metadata,
    get_tilelang_kernel_spec,
    list_tilelang_kernel_specs,
    run_tilelang_kernel,
    tilelang_validation_thresholds,
)
from xqt.operator_opt.kernels.tilelang import (
    dequant_gemm_epilogue_reference,
    dequant_gemm_epilogue_tilelang,
    fused_attention_forward_reference,
    fused_attention_forward_tilelang,
)


def test_tilelang_attention_reference_matches_sdpa() -> None:
    q = torch.randn(1, 2, 4, 8)
    k = torch.randn(1, 2, 4, 8)
    v = torch.randn(1, 2, 4, 8)

    output = fused_attention_forward_reference(q, k, v)

    assert torch.allclose(output, F.scaled_dot_product_attention(q, k, v))


def test_tilelang_dequant_gemm_epilogue_reference_matches_pytorch() -> None:
    x = torch.randn(2, 4)
    qweight = torch.randint(-8, 8, (3, 4), dtype=torch.int8)
    scale = torch.full((3, 1), 0.125)
    bias = torch.randn(3)

    output = dequant_gemm_epilogue_reference(
        x,
        qweight,
        scale,
        bias,
        activation="gelu",
    )
    expected = F.gelu(x.matmul((qweight.float() * scale).t()) + bias)

    assert torch.allclose(output, expected)


def test_tilelang_registry_and_artifact_metadata_are_stable(tmp_path) -> None:
    specs = list_tilelang_kernel_specs()
    settings = TileLangCompileSettings(
        target="cuda",
        target_arch="sm_89",
        cache_dir=str(tmp_path / "tilelang-cache"),
        threads=128,
        num_stages=2,
        pass_configs={"TL_ENABLE_FAST_MATH": True},
    )
    metadata = build_tilelang_artifact_metadata("attention", settings)

    assert set(specs) == {"attention", "dequant_gemm_epilogue"}
    assert specs["attention"]["metadata"]["kernel_name"] == "fused_attention_forward"
    assert specs["attention"]["metadata"]["baseline"].endswith("scaled_dot_product_attention")
    assert get_tilelang_kernel_spec("dequant_gemm_epilogue").fallback == "eager"
    assert metadata["backend"] == "tilelang"
    assert metadata["pattern"] == "attention"
    assert metadata["exportable"] is False
    assert metadata["compile"]["target_arch"] == "sm_89"
    assert metadata["compile"]["pass_configs"] == {"TL_ENABLE_FAST_MATH": True}
    assert metadata["artifact_path"] == str(tmp_path / "tilelang-cache" / "attention.tilelang.json")
    assert metadata["artifact_exists"] is False
    assert metadata["compile_status"] == "metadata_only"
    assert metadata["compile_latency_ms"] is None
    assert metadata["execution_latency_ms"] is None
    assert metadata["validation_thresholds"]["torch.float16"] == {"atol": 1e-3, "rtol": 1e-3}


def test_tilelang_validation_thresholds_are_dtype_specific() -> None:
    assert tilelang_validation_thresholds(torch.float32) == {"atol": 1e-5, "rtol": 1e-5}
    assert tilelang_validation_thresholds(torch.float16) == {"atol": 1e-3, "rtol": 1e-3}
    assert tilelang_validation_thresholds(torch.bfloat16) == {"atol": 1e-2, "rtol": 1e-2}
    assert tilelang_validation_thresholds("unknown") == {"atol": 1e-5, "rtol": 1e-5}


def test_tilelang_backend_uses_eager_fallback_on_cpu() -> None:
    q = torch.randn(1, 2, 4, 8)
    k = torch.randn(1, 2, 4, 8)
    v = torch.randn(1, 2, 4, 8)

    output = run_tilelang_kernel("attention", q, k, v, fallback="eager")

    assert torch.allclose(output, F.scaled_dot_product_attention(q, k, v))


def test_tilelang_backend_requires_cuda_without_fallback() -> None:
    q = torch.randn(1, 2, 4, 8)
    k = torch.randn(1, 2, 4, 8)
    v = torch.randn(1, 2, 4, 8)

    with pytest.raises(XQTBackendError, match="requires CUDA tensors"):
        run_tilelang_kernel("attention", q, k, v, fallback="raise")


def test_tilelang_kernel_entries_reject_cpu_tensors() -> None:
    q = torch.randn(1, 2, 4, 8)
    k = torch.randn(1, 2, 4, 8)
    v = torch.randn(1, 2, 4, 8)
    x = torch.randn(2, 4)
    qweight = torch.randint(-8, 8, (3, 4), dtype=torch.int8)
    scale = torch.full((3, 1), 0.125)

    with pytest.raises(XQTBackendError, match="CUDA tensors"):
        fused_attention_forward_tilelang(q, k, v)
    with pytest.raises(XQTBackendError, match="CUDA tensors"):
        dequant_gemm_epilogue_tilelang(x, qweight, scale)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_tilelang_backend_cuda_attention_smoke_matches_reference() -> None:
    q = torch.randn(1, 2, 4, 8, device="cuda")
    k = torch.randn(1, 2, 4, 8, device="cuda")
    v = torch.randn(1, 2, 4, 8, device="cuda")

    output = run_tilelang_kernel("attention", q, k, v)

    assert torch.allclose(output, F.scaled_dot_product_attention(q, k, v))
