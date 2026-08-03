// SM89 FP8 CUTLASS GEMM.
//
// The input buffers use the canonical one-byte FP8 representation defined by
// xqt.gemm.fp8.  The CUTLASS column-major B view intentionally aliases the
// logical [N,K] row-major weight buffer, so no runtime transpose is required.
// This artifact owns two explicit paths:
// - tensorwise CUTLASS FP8 MMA, where A/W scales collapse to one alpha scalar.
// - blockwise SIMT FP32 accumulation, where A/W scales are loaded inside the K
//   mainloop per block_k tile.  This is intentionally not labeled MMA.
//
// Row/column scale modes that do not match either ABI are handled by the Python
// adapter and are never silently called tensorwise native.

#include <cuda_runtime.h>

#include "cutlass/arch/arch.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/device/gemm.h"

template <typename ElementF8, typename ElementOutput>
using Fp8Gemm = cutlass::gemm::device::Gemm<
    ElementF8,
    cutlass::layout::RowMajor,
    ElementF8,
    cutlass::layout::ColumnMajor,
    ElementOutput,
    cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm89,
    cutlass::gemm::GemmShape<128, 256, 64>,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<16, 8, 32>,
    cutlass::epilogue::thread::LinearCombination<
        ElementOutput,
        128 / cutlass::sizeof_bits<ElementOutput>::value,
        float,
        float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    3,
    8,
    8>;

template <typename ElementF8, typename ElementOutput>
static int run_fp8(
    void const* a,
    void const* b,
    void const* c,
    void* d,
    int m,
    int n,
    int k,
    float alpha,
    float beta) {
  if (!a || !b || !c || !d || m <= 0 || n <= 0 || k <= 0) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  Fp8Gemm<ElementF8, ElementOutput> gemm;
  typename Fp8Gemm<ElementF8, ElementOutput>::Arguments args(
      {m, n, k},
      {static_cast<ElementF8 const*>(a), k},
      {static_cast<ElementF8 const*>(b), k},
      {static_cast<ElementOutput const*>(c), n},
      {static_cast<ElementOutput*>(d), n},
      {alpha, beta});
  cutlass::Status status = gemm.can_implement(args);
  if (status != cutlass::Status::kSuccess) {
    return static_cast<int>(cudaErrorNotSupported);
  }
  status = gemm.initialize(args);
  if (status != cutlass::Status::kSuccess) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  status = gemm();
  return status == cutlass::Status::kSuccess
             ? static_cast<int>(cudaSuccess)
             : static_cast<int>(cudaErrorLaunchFailure);
}

template <typename ElementF8, typename ElementOutput, int BlockK>
__global__ void fp8_blockwise_kernel(
    ElementF8 const* __restrict__ a,
    ElementF8 const* __restrict__ b,
    float const* __restrict__ a_scales,
    float const* __restrict__ b_scales,
    ElementOutput const* __restrict__ c,
    ElementOutput* __restrict__ d,
    int m,
    int n,
    int k,
    float beta) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= m || col >= n) {
    return;
  }
  int scale_blocks = (k + BlockK - 1) / BlockK;
  float acc = 0.0f;
  for (int kk = 0; kk < k; ++kk) {
    int block = kk / BlockK;
    float a_value = static_cast<float>(a[row * k + kk]) *
                    a_scales[row * scale_blocks + block];
    float b_value = static_cast<float>(b[col * k + kk]) *
                    b_scales[col * scale_blocks + block];
    acc += a_value * b_value;
  }
  if (beta != 0.0f && c != nullptr) {
    acc += beta * static_cast<float>(c[row * n + col]);
  }
  d[row * n + col] = static_cast<ElementOutput>(acc);
}

template <typename ElementF8, typename ElementOutput, int BlockK>
static int run_fp8_blockwise_impl(
    void const* a,
    void const* b,
    float const* a_scales,
    float const* b_scales,
    void const* c,
    void* d,
    int m,
    int n,
    int k,
    float beta) {
  if (!a || !b || !a_scales || !b_scales || !d || m <= 0 || n <= 0 || k <= 0) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  if (beta != 0.0f && c == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  dim3 block(16, 16);
  dim3 grid((n + block.x - 1) / block.x, (m + block.y - 1) / block.y);
  fp8_blockwise_kernel<ElementF8, ElementOutput, BlockK><<<grid, block>>>(
      static_cast<ElementF8 const*>(a),
      static_cast<ElementF8 const*>(b),
      a_scales,
      b_scales,
      static_cast<ElementOutput const*>(c),
      static_cast<ElementOutput*>(d),
      m,
      n,
      k,
      beta);
  return static_cast<int>(cudaGetLastError());
}

template <typename ElementF8, typename ElementOutput>
static int run_fp8_blockwise(
    void const* a,
    void const* b,
    float const* a_scales,
    float const* b_scales,
    void const* c,
    void* d,
    int m,
    int n,
    int k,
    int block_k,
    float beta) {
  switch (block_k) {
    case 32:
      return run_fp8_blockwise_impl<ElementF8, ElementOutput, 32>(
          a, b, a_scales, b_scales, c, d, m, n, k, beta);
    case 64:
      return run_fp8_blockwise_impl<ElementF8, ElementOutput, 64>(
          a, b, a_scales, b_scales, c, d, m, n, k, beta);
    case 128:
      return run_fp8_blockwise_impl<ElementF8, ElementOutput, 128>(
          a, b, a_scales, b_scales, c, d, m, n, k, beta);
    default:
      return static_cast<int>(cudaErrorInvalidValue);
  }
}

extern "C" int fp8_sm89_e4m3_fp16_run(
    void const* a, void const* b, void const* c, void* d,
    int m, int n, int k, float alpha, float beta) {
  return run_fp8<cutlass::float_e4m3_t, cutlass::half_t>(
      a, b, c, d, m, n, k, alpha, beta);
}

extern "C" int fp8_sm89_e4m3_bf16_run(
    void const* a, void const* b, void const* c, void* d,
    int m, int n, int k, float alpha, float beta) {
  return run_fp8<cutlass::float_e4m3_t, cutlass::bfloat16_t>(
      a, b, c, d, m, n, k, alpha, beta);
}

extern "C" int fp8_sm89_e5m2_fp16_run(
    void const* a, void const* b, void const* c, void* d,
    int m, int n, int k, float alpha, float beta) {
  return run_fp8<cutlass::float_e5m2_t, cutlass::half_t>(
      a, b, c, d, m, n, k, alpha, beta);
}

extern "C" int fp8_sm89_e5m2_bf16_run(
    void const* a, void const* b, void const* c, void* d,
    int m, int n, int k, float alpha, float beta) {
  return run_fp8<cutlass::float_e5m2_t, cutlass::bfloat16_t>(
      a, b, c, d, m, n, k, alpha, beta);
}

extern "C" int fp8_sm89_e4m3_fp16_run_blockwise(
    void const* a, void const* b, float const* a_scales, float const* b_scales,
    void const* c, void* d, int m, int n, int k, int block_k, float beta) {
  return run_fp8_blockwise<cutlass::float_e4m3_t, cutlass::half_t>(
      a, b, a_scales, b_scales, c, d, m, n, k, block_k, beta);
}

extern "C" int fp8_sm89_e4m3_bf16_run_blockwise(
    void const* a, void const* b, float const* a_scales, float const* b_scales,
    void const* c, void* d, int m, int n, int k, int block_k, float beta) {
  return run_fp8_blockwise<cutlass::float_e4m3_t, cutlass::bfloat16_t>(
      a, b, a_scales, b_scales, c, d, m, n, k, block_k, beta);
}

extern "C" int fp8_sm89_e5m2_fp16_run_blockwise(
    void const* a, void const* b, float const* a_scales, float const* b_scales,
    void const* c, void* d, int m, int n, int k, int block_k, float beta) {
  return run_fp8_blockwise<cutlass::float_e5m2_t, cutlass::half_t>(
      a, b, a_scales, b_scales, c, d, m, n, k, block_k, beta);
}

extern "C" int fp8_sm89_e5m2_bf16_run_blockwise(
    void const* a, void const* b, float const* a_scales, float const* b_scales,
    void const* c, void* d, int m, int n, int k, int block_k, float beta) {
  return run_fp8_blockwise<cutlass::float_e5m2_t, cutlass::bfloat16_t>(
      a, b, a_scales, b_scales, c, d, m, n, k, block_k, beta);
}

extern "C" const char* fp8_sm89_cutlass_version() {
  return "sm89-fp8-cutlass-gemm-v2 tensorwise_mma=16x8x32 blockwise_simt=16x16 block_k=32/64/128";
}
