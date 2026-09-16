"""Unit and integration tests for TVM FFI JIT compilation."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from xqt.kernels.jit.utils.compile import (
    CompileSpec,
    compile_cache_key,
    load_extension,
    load_tvm_ffi_extension,
)


def test_compile_spec_backend_validation(tmp_path: Path) -> None:
    source = tmp_path / "dummy.cpp"
    source.write_text("// dummy\n", encoding="utf-8")

    spec_default = CompileSpec(name="test_default", sources=(source,))
    assert spec_default.backend == "tvm_ffi"

    spec_torch = CompileSpec(name="test_torch", sources=(source,), backend="torch_cpp")
    assert spec_torch.backend == "torch_cpp"
    assert spec_torch.to_dict()["backend"] == "torch_cpp"

    spec_tvm = CompileSpec(name="test_tvm", sources=(source,), backend="tvm_ffi")
    assert spec_tvm.backend == "tvm_ffi"
    assert spec_tvm.to_dict()["backend"] == "tvm_ffi"

    # Backend differences should produce distinct cache keys
    assert compile_cache_key(spec_default) != compile_cache_key(spec_tvm)

    with pytest.raises(ValueError, match="unsupported JIT backend"):
        CompileSpec(name="test_bad", sources=(source,), backend="invalid_backend")


def test_tvm_ffi_cpp_compile_and_execute(tmp_path: Path) -> None:
    source = tmp_path / "add_scalar.cpp"
    source.write_text(
        """
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/function.h>

void add_scalar(tvm::ffi::TensorView x, tvm::ffi::TensorView y, float val) {
    float* px = static_cast<float*>(x.data_ptr());
    float* py = static_cast<float*>(y.data_ptr());
    for (int i = 0; i < x.size(0); ++i) {
        py[i] = px[i] + val;
    }
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(add_scalar, add_scalar);
""",
        encoding="utf-8",
    )

    spec = CompileSpec(
        name="test_tvm_ffi_add_scalar",
        sources=(source,),
        cache_dir=tmp_path / "cache",
        with_cuda=False,
        backend="tvm_ffi",
    )

    module = load_extension(spec)
    x = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
    y = torch.empty_like(x)
    module.add_scalar(x, y, 10.0)

    torch.testing.assert_close(y, x + 10.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_tvm_ffi_cuda_compile_and_execute(tmp_path: Path) -> None:
    source = tmp_path / "scale_cuda.cu"
    source.write_text(
        """
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/function.h>
#include <cuda_runtime.h>

__global__ void scale_kernel(const float* x, float* y, float scale, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        y[idx] = x[idx] * scale;
    }
}

void scale_gpu(tvm::ffi::TensorView x, tvm::ffi::TensorView y, float scale) {
    int n = static_cast<int>(x.size(0));
    const float* px = static_cast<const float*>(x.data_ptr());
    float* py = static_cast<float*>(y.data_ptr());
    int threads = 128;
    int blocks = (n + threads - 1) / threads;
    scale_kernel<<<blocks, threads>>>(px, py, scale, n);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(scale_gpu, scale_gpu);
""",
        encoding="utf-8",
    )

    spec = CompileSpec(
        name="test_tvm_ffi_scale_cuda",
        sources=(source,),
        cache_dir=tmp_path / "cache",
        target_arch="sm_89",
        with_cuda=True,
        backend="tvm_ffi",
    )

    module = load_tvm_ffi_extension(spec)
    x = torch.tensor([2.0, 4.0, 6.0, 8.0], dtype=torch.float32, device="cuda")
    y = torch.empty_like(x)
    module.scale_gpu(x, y, 0.5)
    torch.cuda.synchronize()

    torch.testing.assert_close(y, x * 0.5)


def test_sglang_quantization_ops_registered_with_tvm_ffi() -> None:
    from xqt.kernels.selector import get_kernel
    from xqt.kernels.spec import KernelBackend
    import xqt.kernels.ops.quantization as ops_quant

    assert callable(ops_quant.convrot_w8a8_linear)
    assert callable(ops_quant.svdq_w8a8_linear)
    assert callable(ops_quant.awq_w4a16_decode)
    assert callable(ops_quant.convrot_w4a4_rowwise_linear)
    assert callable(ops_quant.svdq_w4a4_linear)

    # Verify each op is in the registry with TVM_FFI backend
    for op_name in [
        "quantization.convrot_w8a8_linear",
        "quantization.svdq_w8a8_linear",
        "quantization.awq_w4a16_decode",
        "quantization.convrot_w4a4_rowwise_linear",
        "quantization.svdq_w4a4_linear",
    ]:
        fn = get_kernel(op_name, backend=KernelBackend.TVM_FFI)
        assert callable(fn), f"Op {op_name} not found with TVM_FFI backend"

