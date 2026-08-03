// SM89 dense FP16/BF16 CUTLASS seed.
// Logical A[M,K] and W[N,K] are passed as row-major A and column-major B[K,N]
// respectively; the column-major B pointer is therefore the canonical W[N,K]
// storage without a runtime transpose. C is the beta source and D is output.

#include <cuda_runtime.h>

#include "cutlass/arch/arch.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/device/gemm.h"

template <typename Element, typename Gemm>
static int run_dense(void const* a, void const* b, void const* c, void* d, int m, int n, int k) {
  if (!a || !b || !c || !d || m <= 0 || n <= 0 || k <= 0) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  Gemm gemm;
  typename Gemm::Arguments args(
      {m, n, k},
      {static_cast<Element const*>(a), k},
      {static_cast<Element const*>(b), k},
      {static_cast<Element const*>(c), n},
      {static_cast<Element*>(d), n},
      {1.0f, 1.0f});
  cutlass::Status status = gemm.can_implement(args);
  if (status != cutlass::Status::kSuccess) {
    return static_cast<int>(cudaErrorNotSupported);
  }
  status = gemm.initialize(args);
  if (status != cutlass::Status::kSuccess) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  status = gemm();
  return status == cutlass::Status::kSuccess ? static_cast<int>(cudaSuccess)
                                              : static_cast<int>(cudaErrorLaunchFailure);
}

using F16Gemm = cutlass::gemm::device::Gemm<
    cutlass::half_t,
    cutlass::layout::RowMajor,
    cutlass::half_t,
    cutlass::layout::ColumnMajor,
    cutlass::half_t,
    cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 8, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    3,
    8,
    8>;

using BF16Gemm = cutlass::gemm::device::Gemm<
    cutlass::bfloat16_t,
    cutlass::layout::RowMajor,
    cutlass::bfloat16_t,
    cutlass::layout::ColumnMajor,
    cutlass::bfloat16_t,
    cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<cutlass::bfloat16_t, 8, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    3,
    8,
    8>;

extern "C" int dense_sm89_fp16_run(
    void const* a, void const* b, void const* c, void* d, int m, int n, int k) {
  return run_dense<cutlass::half_t, F16Gemm>(a, b, c, d, m, n, k);
}

extern "C" int dense_sm89_bf16_run(
    void const* a, void const* b, void const* c, void* d, int m, int n, int k) {
  return run_dense<cutlass::bfloat16_t, BF16Gemm>(a, b, c, d, m, n, k);
}

extern "C" const char* dense_sm89_version() {
  return "dense-sm89-cutlass-v1 tile=128x128x32 warp=4x64x64 stages=3";
}
