// SM89 FP8 CUTLASS GEMM.
//
// The input buffers use the canonical one-byte FP8 representation defined by
// xqt.gemm.fp8.  The CUTLASS column-major B view intentionally aliases the
// logical [N,K] row-major weight buffer, so no runtime transpose is required.
// This artifact owns two explicit paths:
// - tensorwise CUTLASS FP8 MMA, where A/W scales collapse to one alpha scalar.
// - blockwise FP8 warp MMA, where [M,Kb]/[N,Kb] scales are applied inside the
//   K mainloop: raw FP8 MMA products accumulate per K block and are promoted
//   into the FP32 total at every block_k boundary.  Scales are never deferred
//   to a final scalar epilogue.  An opt-in split-K workspace/reduction ABI
//   mirrors the fused W4A16 artifact; splits are block_k aligned.
//
// Row/column scale modes that do not match either ABI are handled by the Python
// adapter and are never silently called tensorwise native.

#include <cuda_runtime.h>
#include <cstdint>

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

// SM89 FP8 K-blockwise warp MMA.
//
// Each warp owns one 16x8 output tile and walks K in 32-column instruction
// tiles through the SM89 mma.sync m16n8k32 FP8 primitive (cutlass::arch::Mma).
// The raw FP8 products accumulate in a per-block fragment; at every block_k
// boundary the block fragment is promoted into the final FP32 total with
// scale_a[m, block] * scale_w[col, block], so K-block scales live inside the
// mainloop by construction.  A/B fragments are loaded straight from global
// memory as uint32 words: the adapter pads K to a multiple of 32, so every
// fragment row segment is 4-byte aligned and needs no shared-memory staging,
// transpose or decode step.
//
// Fragment register mapping for mma.m16n8k32 8-bit operands (PTX ISA):
// - A row-major m16k32: word0/1 hold rows quad and quad+8 at columns
//   lane_in_quad*4..+3, word2/3 hold the same rows at columns +16.
// - B col-major k32n8 (the [N,K] row-major weight): word0/1 hold column quad
//   at rows lane_in_quad*4..+3 and +16.
// - C row-major m16n8 f32: [0,1] map to (quad, lane_in_quad*2 +{0,1}),
//   [2,3] map to (quad+8, lane_in_quad*2 +{0,1}).

template <typename ElementF8>
using Fp8WarpMma = cutlass::arch::Mma<
    cutlass::gemm::GemmShape<16, 8, 32>,
    32,
    ElementF8,
    cutlass::layout::RowMajor,
    ElementF8,
    cutlass::layout::ColumnMajor,
    float,
    cutlass::layout::RowMajor,
    cutlass::arch::OpMultiplyAdd>;

template <typename ElementF8, int BlockK>
__device__ __forceinline__ void fp8_blockwise_mainloop(
    ElementF8 const* __restrict__ a,
    ElementF8 const* __restrict__ b,
    float const* __restrict__ a_scales,
    float const* __restrict__ b_scales,
    int row_base,
    int col_base,
    int k_begin,
    int k_end,
    int k,
    int scale_blocks,
    float (&total)[4]) {
  static_assert(BlockK % 32 == 0, "BlockK must be a multiple of the 32-wide instruction");
  constexpr int k_tiles_per_block = BlockK / 32;
  int lane = static_cast<int>(threadIdx.x) & 31;
  int quad = lane >> 2;
  int lane_in_quad = lane & 3;
  Fp8WarpMma<ElementF8> mma;
  typename Fp8WarpMma<ElementF8>::FragmentC block_accum;
  block_accum.clear();
  ElementF8 const* a_row0 = a + (row_base + quad) * k;
  ElementF8 const* a_row1 = a + (row_base + quad + 8) * k;
  ElementF8 const* b_row = b + (col_base + quad) * k;
  for (int k_base = k_begin; k_base < k_end; k_base += 32) {
    int tile_in_block = ((k_base - k_begin) >> 5) % k_tiles_per_block;
    if (tile_in_block == 0) {
      block_accum.clear();
    }
    typename Fp8WarpMma<ElementF8>::FragmentA fragment_a;
    uint32_t* a_words = reinterpret_cast<uint32_t*>(&fragment_a);
    int k_lane = k_base + lane_in_quad * 4;
    a_words[0] = *reinterpret_cast<uint32_t const*>(a_row0 + k_lane);
    a_words[1] = *reinterpret_cast<uint32_t const*>(a_row1 + k_lane);
    a_words[2] = *reinterpret_cast<uint32_t const*>(a_row0 + k_lane + 16);
    a_words[3] = *reinterpret_cast<uint32_t const*>(a_row1 + k_lane + 16);
    typename Fp8WarpMma<ElementF8>::FragmentB fragment_b;
    uint32_t* b_words = reinterpret_cast<uint32_t*>(&fragment_b);
    b_words[0] = *reinterpret_cast<uint32_t const*>(b_row + k_lane);
    b_words[1] = *reinterpret_cast<uint32_t const*>(b_row + k_lane + 16);
    mma(block_accum, fragment_a, fragment_b, block_accum);
    bool block_done =
        tile_in_block == k_tiles_per_block - 1 || k_base + 32 >= k_end;
    if (block_done) {
      int block = k_base / BlockK;
      float scale_a0 = a_scales[(row_base + quad) * scale_blocks + block];
      float scale_a1 = a_scales[(row_base + quad + 8) * scale_blocks + block];
      float scale_w0 =
          b_scales[(col_base + lane_in_quad * 2) * scale_blocks + block];
      float scale_w1 =
          b_scales[(col_base + lane_in_quad * 2 + 1) * scale_blocks + block];
      total[0] += block_accum[0] * scale_a0 * scale_w0;
      total[1] += block_accum[1] * scale_a0 * scale_w1;
      total[2] += block_accum[2] * scale_a1 * scale_w0;
      total[3] += block_accum[3] * scale_a1 * scale_w1;
    }
  }
}

template <typename ElementF8, typename ElementOutput, int BlockK>
__global__ void fp8_blockwise_mma_kernel(
    ElementF8 const* __restrict__ a,
    ElementF8 const* __restrict__ b,
    float const* __restrict__ a_scales,
    float const* __restrict__ b_scales,
    ElementOutput const* __restrict__ c,
    ElementOutput* __restrict__ d,
    int m,
    int n,
    int k,
    int scale_blocks,
    float beta) {
  int row_base = static_cast<int>(blockIdx.y) * 16;
  int col_base = static_cast<int>(blockIdx.x) * 8;
  int lane = static_cast<int>(threadIdx.x) & 31;
  int quad = lane >> 2;
  int lane_in_quad = lane & 3;
  float total[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  fp8_blockwise_mainloop<ElementF8, BlockK>(
      a, b, a_scales, b_scales, row_base, col_base, 0, k, k, scale_blocks, total);
  for (int row = 0; row < 2; ++row) {
    for (int col = 0; col < 2; ++col) {
      int index = row * 2 + col;
      int global_row = row_base + quad + row * 8;
      int global_column = col_base + lane_in_quad * 2 + col;
      float value = total[index];
      if (beta != 0.0f && c != nullptr) {
        value += beta * static_cast<float>(c[global_row * n + global_column]);
      }
      d[global_row * n + global_column] = static_cast<ElementOutput>(value);
    }
  }
}

// Split-K partial pass: blockIdx.z selects a block_k-aligned slice of K and
// the warp promotes exactly the blocks it owns into a per-split FP32
// workspace.  Block scales are applied inside the slice mainloop, so the
// reduction pass is a plain deterministic sum with no scale semantics of its
// own.  A split boundary can never divide a scale block.
template <typename ElementF8, int BlockK>
__global__ void fp8_blockwise_mma_splitk_partial_kernel(
    ElementF8 const* __restrict__ a,
    ElementF8 const* __restrict__ b,
    float const* __restrict__ a_scales,
    float const* __restrict__ b_scales,
    float* __restrict__ workspace,
    int n,
    int k,
    int scale_blocks,
    int k_per_split) {
  int row_base = static_cast<int>(blockIdx.y) * 16;
  int col_base = static_cast<int>(blockIdx.x) * 8;
  int split = static_cast<int>(blockIdx.z);
  int k_begin = split * k_per_split;
  int k_end = min(k_begin + k_per_split, k);
  float total[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  if (k_begin < k_end) {
    fp8_blockwise_mainloop<ElementF8, BlockK>(
        a, b, a_scales, b_scales, row_base, col_base, k_begin, k_end, k,
        scale_blocks, total);
  }
  int lane = static_cast<int>(threadIdx.x) & 31;
  int quad = lane >> 2;
  int lane_in_quad = lane & 3;
  float* split_workspace =
      workspace + static_cast<long long>(split) * gridDim.y * 16 * n;
  for (int row = 0; row < 2; ++row) {
    for (int col = 0; col < 2; ++col) {
      int index = row * 2 + col;
      int global_row = row_base + quad + row * 8;
      int global_column = col_base + lane_in_quad * 2 + col;
      split_workspace[global_row * n + global_column] = total[index];
    }
  }
}

// Reduction pass: sum the per-split FP32 partials, apply beta*C exactly once,
// and emit the output.  A second kernel plus a [split, M, N] float workspace
// is chosen over atomicAdd so the accumulation order stays deterministic and
// reproducible run to run (same ABI contract as the fused W4A16 artifact).
template <typename ElementOutput>
__global__ void fp8_blockwise_mma_splitk_reduce_kernel(
    float const* __restrict__ workspace,
    ElementOutput const* __restrict__ c,
    ElementOutput* __restrict__ d,
    int element_count,
    int split_count,
    float beta) {
  int index = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) +
              static_cast<int>(threadIdx.x);
  if (index >= element_count) {
    return;
  }
  float value = 0.0f;
  for (int split = 0; split < split_count; ++split) {
    value += workspace[static_cast<long long>(split) * element_count + index];
  }
  if (beta != 0.0f && c != nullptr) {
    value += beta * static_cast<float>(c[index]);
  }
  d[index] = static_cast<ElementOutput>(value);
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
  if (m % 16 != 0 || n % 8 != 0 || k % 32 != 0) {
    return static_cast<int>(cudaErrorNotSupported);
  }
  int scale_blocks = (k + BlockK - 1) / BlockK;
  dim3 grid(static_cast<unsigned>(n / 8), static_cast<unsigned>(m / 16));
  fp8_blockwise_mma_kernel<ElementF8, ElementOutput, BlockK>
      <<<grid, 32>>>(
          static_cast<ElementF8 const*>(a),
          static_cast<ElementF8 const*>(b),
          a_scales,
          b_scales,
          static_cast<ElementOutput const*>(c),
          static_cast<ElementOutput*>(d),
          m,
          n,
          k,
          scale_blocks,
          beta);
  return static_cast<int>(cudaGetLastError());
}

template <typename ElementF8, typename ElementOutput, int BlockK>
static int run_fp8_blockwise_splitk_impl(
    void const* a,
    void const* b,
    float const* a_scales,
    float const* b_scales,
    void const* c,
    void* d,
    void* workspace,
    int m,
    int n,
    int k,
    int split_k,
    float beta) {
  if (!a || !b || !a_scales || !b_scales || !d || !workspace || m <= 0 ||
      n <= 0 || k <= 0 || split_k < 2) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  if (beta != 0.0f && c == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  if (m % 16 != 0 || n % 8 != 0 || k % 32 != 0) {
    return static_cast<int>(cudaErrorNotSupported);
  }
  int execution_blocks = (k + BlockK - 1) / BlockK;
  int blocks_per_split = (execution_blocks + split_k - 1) / split_k;
  int k_per_split = blocks_per_split * BlockK;
  int split_count = (k + k_per_split - 1) / k_per_split;
  dim3 partial_grid(
      static_cast<unsigned>(n / 8),
      static_cast<unsigned>(m / 16),
      static_cast<unsigned>(split_count));
  fp8_blockwise_mma_splitk_partial_kernel<ElementF8, BlockK>
      <<<partial_grid, 32>>>(
          static_cast<ElementF8 const*>(a),
          static_cast<ElementF8 const*>(b),
          a_scales,
          b_scales,
          static_cast<float*>(workspace),
          n,
          k,
          execution_blocks,
          k_per_split);
  int element_count = m * n;
  int reduce_threads = 256;
  int reduce_blocks = (element_count + reduce_threads - 1) / reduce_threads;
  fp8_blockwise_mma_splitk_reduce_kernel<ElementOutput>
      <<<reduce_blocks, reduce_threads>>>(
          static_cast<float const*>(workspace),
          static_cast<ElementOutput const*>(c),
          static_cast<ElementOutput*>(d),
          element_count,
          split_count,
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

template <typename ElementF8, typename ElementOutput>
static int run_fp8_blockwise_splitk(
    void const* a,
    void const* b,
    float const* a_scales,
    float const* b_scales,
    void const* c,
    void* d,
    void* workspace,
    int m,
    int n,
    int k,
    int block_k,
    int split_k,
    float beta) {
  switch (block_k) {
    case 32:
      return run_fp8_blockwise_splitk_impl<ElementF8, ElementOutput, 32>(
          a, b, a_scales, b_scales, c, d, workspace, m, n, k, split_k, beta);
    case 64:
      return run_fp8_blockwise_splitk_impl<ElementF8, ElementOutput, 64>(
          a, b, a_scales, b_scales, c, d, workspace, m, n, k, split_k, beta);
    case 128:
      return run_fp8_blockwise_splitk_impl<ElementF8, ElementOutput, 128>(
          a, b, a_scales, b_scales, c, d, workspace, m, n, k, split_k, beta);
    default:
      return static_cast<int>(cudaErrorInvalidValue);
  }
}

// Compile-time resource attributes and device occupancy for one blockwise
// variant.  The query never launches a kernel; profiling and benchmark reports
// inspect a loaded artifact without enqueueing a fake operation.  block_k
// changes the unrolled mainloop, so each legal block_k is queried separately.
template <typename ElementF8, typename ElementOutput>
static cudaError_t query_fp8_blockwise_variant(
    int block_k,
    int* registers_per_thread,
    int* static_shared_bytes,
    int* max_active_blocks) {
  cudaFuncAttributes attributes{};
  cudaError_t status = cudaErrorInvalidValue;
  switch (block_k) {
    case 32:
      status = cudaFuncGetAttributes(
          &attributes, fp8_blockwise_mma_kernel<ElementF8, ElementOutput, 32>);
      break;
    case 64:
      status = cudaFuncGetAttributes(
          &attributes, fp8_blockwise_mma_kernel<ElementF8, ElementOutput, 64>);
      break;
    case 128:
      status = cudaFuncGetAttributes(
          &attributes, fp8_blockwise_mma_kernel<ElementF8, ElementOutput, 128>);
      break;
    default:
      return cudaErrorInvalidValue;
  }
  if (status != cudaSuccess) {
    return status;
  }
  switch (block_k) {
    case 32:
      status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          max_active_blocks,
          fp8_blockwise_mma_kernel<ElementF8, ElementOutput, 32>,
          32,
          0);
      break;
    case 64:
      status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          max_active_blocks,
          fp8_blockwise_mma_kernel<ElementF8, ElementOutput, 64>,
          32,
          0);
      break;
    case 128:
      status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          max_active_blocks,
          fp8_blockwise_mma_kernel<ElementF8, ElementOutput, 128>,
          32,
          0);
      break;
    default:
      return cudaErrorInvalidValue;
  }
  if (status != cudaSuccess) {
    return status;
  }
  *registers_per_thread = attributes.numRegs;
  *static_shared_bytes = static_cast<int>(attributes.sharedSizeBytes);
  return cudaSuccess;
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

extern "C" int fp8_sm89_e4m3_fp16_run_blockwise_splitk(
    void const* a, void const* b, float const* a_scales, float const* b_scales,
    void const* c, void* d, void* workspace, int m, int n, int k, int block_k,
    int split_k, float beta) {
  return run_fp8_blockwise_splitk<cutlass::float_e4m3_t, cutlass::half_t>(
      a, b, a_scales, b_scales, c, d, workspace, m, n, k, block_k, split_k,
      beta);
}

extern "C" int fp8_sm89_e4m3_bf16_run_blockwise_splitk(
    void const* a, void const* b, float const* a_scales, float const* b_scales,
    void const* c, void* d, void* workspace, int m, int n, int k, int block_k,
    int split_k, float beta) {
  return run_fp8_blockwise_splitk<cutlass::float_e4m3_t, cutlass::bfloat16_t>(
      a, b, a_scales, b_scales, c, d, workspace, m, n, k, block_k, split_k,
      beta);
}

extern "C" int fp8_sm89_e5m2_fp16_run_blockwise_splitk(
    void const* a, void const* b, float const* a_scales, float const* b_scales,
    void const* c, void* d, void* workspace, int m, int n, int k, int block_k,
    int split_k, float beta) {
  return run_fp8_blockwise_splitk<cutlass::float_e5m2_t, cutlass::half_t>(
      a, b, a_scales, b_scales, c, d, workspace, m, n, k, block_k, split_k,
      beta);
}

extern "C" int fp8_sm89_e5m2_bf16_run_blockwise_splitk(
    void const* a, void const* b, float const* a_scales, float const* b_scales,
    void const* c, void* d, void* workspace, int m, int n, int k, int block_k,
    int split_k, float beta) {
  return run_fp8_blockwise_splitk<cutlass::float_e5m2_t, cutlass::bfloat16_t>(
      a, b, a_scales, b_scales, c, d, workspace, m, n, k, block_k, split_k,
      beta);
}

extern "C" int fp8_sm89_blockwise_resource_query(
    int format,
    int output,
    int block_k,
    int* registers_per_thread,
    int* static_shared_bytes,
    int* max_active_blocks) {
  if (format < 0 || format > 1 || output < 0 || output > 1 ||
      !registers_per_thread || !static_shared_bytes || !max_active_blocks) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  if (format == 0 && output == 0) {
    return static_cast<int>(
        query_fp8_blockwise_variant<cutlass::float_e4m3_t, cutlass::half_t>(
            block_k, registers_per_thread, static_shared_bytes,
            max_active_blocks));
  }
  if (format == 0 && output == 1) {
    return static_cast<int>(
        query_fp8_blockwise_variant<cutlass::float_e4m3_t, cutlass::bfloat16_t>(
            block_k, registers_per_thread, static_shared_bytes,
            max_active_blocks));
  }
  if (format == 1 && output == 0) {
    return static_cast<int>(
        query_fp8_blockwise_variant<cutlass::float_e5m2_t, cutlass::half_t>(
            block_k, registers_per_thread, static_shared_bytes,
            max_active_blocks));
  }
  return static_cast<int>(
      query_fp8_blockwise_variant<cutlass::float_e5m2_t, cutlass::bfloat16_t>(
          block_k, registers_per_thread, static_shared_bytes,
          max_active_blocks));
}

extern "C" const char* fp8_sm89_cutlass_version() {
  return "sm89-fp8-cutlass-gemm-v3 tensorwise_mma=128x256x64 blockwise_warp_mma=16x8x32 block_k=32/64/128 splitk_workspace";
}
