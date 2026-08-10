// SM89 grouped FP8 decode kernel.
//
// The task grid matches grouped W8A8: one CTA owns one routed row tile and a
// configurable number of eight-column MMA tiles.  Warps share the A staging
// tile, while each warp reads one [8,32] B fragment directly from the static
// [expert,N,K] weight payload.  Tensorwise scales are applied after the full K
// reduction; blockwise scales are promoted into FP32 at each block boundary.

#include <cuda_runtime.h>
#include <cstdint>

#include "cutlass/arch/arch.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/device/gemm.h"

namespace {

constexpr int kRows = 16;
constexpr int kCols = 8;
constexpr int kKTile = 32;
constexpr int kALd = 64;

__device__ __forceinline__ int a_offset(int row, int col) {
  return row * kALd + (col ^ ((row & 3) << 4));
}

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

template <typename ElementF8, int BlockK, int Warps>
__global__ void grouped_fp8_kernel(
    const uint8_t* __restrict__ activation,
    const uint8_t* __restrict__ weights_nk,
    const float* __restrict__ activation_scales,
    const float* __restrict__ weight_scales,
    const float* __restrict__ bias,
    const int32_t* __restrict__ task_table,
    const int32_t* __restrict__ output_rows,
    cutlass::half_t* __restrict__ output,
    int task_count,
    int n,
    int k,
    int expert_count,
    int scale_blocks,
    int has_bias,
    int has_output_rows) {
  static_assert(BlockK == 0 || BlockK == 32 || BlockK == 64 || BlockK == 128);
  static_assert(Warps == 1 || Warps == 2 || Warps == 4 || Warps == 8);
  constexpr int kThreads = Warps * 32;
  constexpr int kBlockCols = Warps * kCols;
  constexpr int kTilesPerBlock = BlockK == 0 ? 1 : BlockK / kKTile;
  const int task_index = static_cast<int>(blockIdx.y);
  const int block_n = static_cast<int>(blockIdx.x) * kBlockCols;
  if (task_index >= task_count || block_n >= n) {
    return;
  }
  const int expert = task_table[task_index * 3 + 0];
  const int row_base = task_table[task_index * 3 + 1];
  const int row_count = task_table[task_index * 3 + 2];
  if (expert < 0 || expert >= expert_count || row_count <= 0 || row_count > 8) {
    return;
  }

  __shared__ __align__(16) uint8_t shared_a[kRows * kALd];
  const int thread = static_cast<int>(threadIdx.x);
  const int lane = thread & 31;
  const int warp = thread >> 5;
  const int quad = lane >> 2;
  const int lane_in_quad = lane & 3;
  const int n_tile = block_n + warp * kCols;
  const int64_t weight_base = static_cast<int64_t>(expert) * n * k;
  Fp8WarpMma<ElementF8> mma;
  typename Fp8WarpMma<ElementF8>::FragmentC block_accum;
  block_accum.clear();
  float total[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  const int k_tiles = (k + kKTile - 1) / kKTile;

  for (int kt = 0; kt < k_tiles; ++kt) {
    const int k_base = kt * kKTile;
    if constexpr (BlockK != 0) {
      if (kt % kTilesPerBlock == 0) {
        block_accum.clear();
      }
    }
    for (int index = thread; index < kRows * kKTile; index += kThreads) {
      const int row = index / kKTile;
      const int col = index - row * kKTile;
      const int global_col = k_base + col;
      uint8_t value = 0;
      if (row < row_count && global_col < k) {
        value = activation[static_cast<int64_t>(row_base + row) * k + global_col];
      }
      shared_a[a_offset(row, col)] = value;
    }
    __syncthreads();

    typename Fp8WarpMma<ElementF8>::FragmentA fragment_a;
    uint32_t* a_words = reinterpret_cast<uint32_t*>(&fragment_a);
    const int fragment_col = lane_in_quad * 4;
    a_words[0] = *reinterpret_cast<const uint32_t*>(
        shared_a + a_offset(quad, fragment_col));
    a_words[1] = *reinterpret_cast<const uint32_t*>(
        shared_a + a_offset(quad + 8, fragment_col));
    a_words[2] = *reinterpret_cast<const uint32_t*>(
        shared_a + a_offset(quad, fragment_col + 16));
    a_words[3] = *reinterpret_cast<const uint32_t*>(
        shared_a + a_offset(quad + 8, fragment_col + 16));

    typename Fp8WarpMma<ElementF8>::FragmentB fragment_b;
    uint32_t* b_words = reinterpret_cast<uint32_t*>(&fragment_b);
    const int weight_column = n_tile + quad;
    if (weight_column < n) {
      const uint8_t* weight_row =
          weights_nk + weight_base + static_cast<int64_t>(weight_column) * k;
      b_words[0] = *reinterpret_cast<const uint32_t*>(
          weight_row + k_base + lane_in_quad * 4);
      b_words[1] = *reinterpret_cast<const uint32_t*>(
          weight_row + k_base + lane_in_quad * 4 + 16);
    } else {
      b_words[0] = 0;
      b_words[1] = 0;
    }
    mma(block_accum, fragment_a, fragment_b, block_accum);

    if constexpr (BlockK != 0) {
      const bool block_done =
          (kt % kTilesPerBlock == kTilesPerBlock - 1) || (kt + 1 == k_tiles);
      if (block_done) {
        const int block = k_base / BlockK;
        const int source_row0 = row_base + quad;
        const int source_row1 = source_row0 + 8;
        const int column0 = n_tile + lane_in_quad * 2;
        const int column1 = column0 + 1;
        const float scale_a0 = quad < row_count
                                   ? activation_scales[source_row0 * scale_blocks + block]
                                   : 0.0f;
        const float scale_a1 = quad + 8 < row_count
                                   ? activation_scales[source_row1 * scale_blocks + block]
                                   : 0.0f;
        const float scale_w0 = column0 < n
                                   ? weight_scales[
                                         (static_cast<int64_t>(expert) * n + column0) *
                                             scale_blocks +
                                         block]
                                   : 0.0f;
        const float scale_w1 = column1 < n
                                   ? weight_scales[
                                         (static_cast<int64_t>(expert) * n + column1) *
                                             scale_blocks +
                                         block]
                                   : 0.0f;
        total[0] += block_accum[0] * scale_a0 * scale_w0;
        total[1] += block_accum[1] * scale_a0 * scale_w1;
        total[2] += block_accum[2] * scale_a1 * scale_w0;
        total[3] += block_accum[3] * scale_a1 * scale_w1;
      }
    }
    __syncthreads();
  }

  if constexpr (BlockK == 0) {
    const float alpha = activation_scales[expert] * weight_scales[expert];
    total[0] = block_accum[0] * alpha;
    total[1] = block_accum[1] * alpha;
    total[2] = block_accum[2] * alpha;
    total[3] = block_accum[3] * alpha;
  }

  const int source_rows[2] = {row_base + quad, row_base + quad + 8};
  const int values_per_row = 2;
  const int column0 = n_tile + lane_in_quad * values_per_row;
  const float* expert_bias =
      has_bias ? bias + static_cast<int64_t>(expert) * n : nullptr;
  for (int row = 0; row < 2; ++row) {
    const int local_row = quad + row * 8;
    if (local_row >= row_count) {
      continue;
    }
    const int source_row = source_rows[row];
    const int destination_row =
        has_output_rows ? output_rows[source_row] : source_row;
    if (column0 < n) {
      const float value = total[row * 2] + (has_bias ? expert_bias[column0] : 0.0f);
      output[static_cast<int64_t>(destination_row) * n + column0] =
          static_cast<cutlass::half_t>(value);
    }
    if (column0 + 1 < n) {
      const float value =
          total[row * 2 + 1] + (has_bias ? expert_bias[column0 + 1] : 0.0f);
      output[static_cast<int64_t>(destination_row) * n + column0 + 1] =
          static_cast<cutlass::half_t>(value);
    }
  }
}

template <typename ElementF8, int BlockK>
int launch_grouped_fp8(
    const void* activation,
    const void* weights_nk,
    const float* activation_scales,
    const float* weight_scales,
    const float* bias,
    const int32_t* task_table,
    const int32_t* output_rows,
    void* output,
    int task_count,
    int n,
    int k,
    int expert_count,
    int has_bias,
    int has_output_rows,
    int warps_per_block,
    cudaStream_t stream) {
  const int scale_blocks = BlockK == 0 ? 1 : (k + BlockK - 1) / BlockK;
#define XQT_LAUNCH_GROUPED_FP8(WARPS)                                                       \
  do {                                                                                     \
    const dim3 grid((n + (WARPS) * kCols - 1) / ((WARPS) * kCols), task_count, 1);         \
    grouped_fp8_kernel<ElementF8, BlockK, WARPS><<<grid, (WARPS) * 32, 0, stream>>>(        \
        static_cast<const uint8_t*>(activation),                                            \
        static_cast<const uint8_t*>(weights_nk),                                            \
        activation_scales,                                                                 \
        weight_scales,                                                                     \
        bias,                                                                              \
        task_table,                                                                        \
        output_rows,                                                                       \
        static_cast<cutlass::half_t*>(output),                                              \
        task_count, n, k, expert_count, scale_blocks, has_bias, has_output_rows);           \
  } while (false)
  switch (warps_per_block) {
    case 1:
      XQT_LAUNCH_GROUPED_FP8(1);
      break;
    case 2:
      XQT_LAUNCH_GROUPED_FP8(2);
      break;
    case 4:
      XQT_LAUNCH_GROUPED_FP8(4);
      break;
    case 8:
      XQT_LAUNCH_GROUPED_FP8(8);
      break;
    default:
      return static_cast<int>(cudaErrorInvalidValue);
  }
#undef XQT_LAUNCH_GROUPED_FP8
  return static_cast<int>(cudaGetLastError());
}

template <typename ElementF8>
int dispatch_grouped_fp8(
    const void* activation,
    const void* weights_nk,
    const float* activation_scales,
    const float* weight_scales,
    const float* bias,
    const int32_t* task_table,
    const int32_t* output_rows,
    void* output,
    int task_count,
    int n,
    int k,
    int expert_count,
    int block_k,
    int has_bias,
    int has_output_rows,
    int warps_per_block,
    cudaStream_t stream) {
  switch (block_k) {
    case 0:
      return launch_grouped_fp8<ElementF8, 0>(
          activation, weights_nk, activation_scales, weight_scales, bias,
          task_table, output_rows, output, task_count, n, k, expert_count,
          has_bias, has_output_rows, warps_per_block, stream);
    case 32:
      return launch_grouped_fp8<ElementF8, 32>(
          activation, weights_nk, activation_scales, weight_scales, bias,
          task_table, output_rows, output, task_count, n, k, expert_count,
          has_bias, has_output_rows, warps_per_block, stream);
    case 64:
      return launch_grouped_fp8<ElementF8, 64>(
          activation, weights_nk, activation_scales, weight_scales, bias,
          task_table, output_rows, output, task_count, n, k, expert_count,
          has_bias, has_output_rows, warps_per_block, stream);
    case 128:
      return launch_grouped_fp8<ElementF8, 128>(
          activation, weights_nk, activation_scales, weight_scales, bias,
          task_table, output_rows, output, task_count, n, k, expert_count,
          has_bias, has_output_rows, warps_per_block, stream);
    default:
      return static_cast<int>(cudaErrorInvalidValue);
  }
}

template <typename ElementF8, int BlockK, int Warps>
cudaError_t query_variant(
    int* registers_per_thread,
    int* static_shared_bytes,
    int* max_active_blocks_per_sm) {
  cudaFuncAttributes attributes{};
  cudaError_t status = cudaFuncGetAttributes(
      &attributes, grouped_fp8_kernel<ElementF8, BlockK, Warps>);
  if (status != cudaSuccess) {
    return status;
  }
  *registers_per_thread = attributes.numRegs;
  *static_shared_bytes = attributes.sharedSizeBytes;
  return cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      max_active_blocks_per_sm,
      grouped_fp8_kernel<ElementF8, BlockK, Warps>,
      Warps * 32,
      0);
}

template <typename ElementF8, int BlockK>
cudaError_t query_warps(
    int warps_per_block,
    int* registers_per_thread,
    int* static_shared_bytes,
    int* max_active_blocks_per_sm) {
  switch (warps_per_block) {
    case 1:
      return query_variant<ElementF8, BlockK, 1>(
          registers_per_thread, static_shared_bytes, max_active_blocks_per_sm);
    case 2:
      return query_variant<ElementF8, BlockK, 2>(
          registers_per_thread, static_shared_bytes, max_active_blocks_per_sm);
    case 4:
      return query_variant<ElementF8, BlockK, 4>(
          registers_per_thread, static_shared_bytes, max_active_blocks_per_sm);
    case 8:
      return query_variant<ElementF8, BlockK, 8>(
          registers_per_thread, static_shared_bytes, max_active_blocks_per_sm);
    default:
      return cudaErrorInvalidValue;
  }
}

template <typename ElementF8>
cudaError_t query_block_k(
    int block_k,
    int warps_per_block,
    int* registers_per_thread,
    int* static_shared_bytes,
    int* max_active_blocks_per_sm) {
  switch (block_k) {
    case 0:
      return query_warps<ElementF8, 0>(
          warps_per_block, registers_per_thread, static_shared_bytes,
          max_active_blocks_per_sm);
    case 32:
      return query_warps<ElementF8, 32>(
          warps_per_block, registers_per_thread, static_shared_bytes,
          max_active_blocks_per_sm);
    case 64:
      return query_warps<ElementF8, 64>(
          warps_per_block, registers_per_thread, static_shared_bytes,
          max_active_blocks_per_sm);
    case 128:
      return query_warps<ElementF8, 128>(
          warps_per_block, registers_per_thread, static_shared_bytes,
          max_active_blocks_per_sm);
    default:
      return cudaErrorInvalidValue;
  }
}

}  // namespace

extern "C" int xqt_fp8_grouped_sm89_fp16_run(
    const void* activation,
    const void* weights_nk,
    const void* activation_scales,
    const void* weight_scales,
    const void* bias,
    const void* task_table,
    const void* output_rows,
    void* output,
    int task_count,
    int n,
    int k,
    int expert_count,
    int format_id,
    int block_k,
    int has_bias,
    int has_output_rows,
    int warps_per_block,
    void* stream_ptr) {
  if (!activation || !weights_nk || !activation_scales || !weight_scales ||
      !task_table || !output || task_count <= 0 || n <= 0 || k <= 0 ||
      expert_count <= 0) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  if ((has_bias && !bias) || (has_output_rows && !output_rows)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  if (format_id == 0) {
    return dispatch_grouped_fp8<cutlass::float_e4m3_t>(
        activation, weights_nk, static_cast<const float*>(activation_scales),
        static_cast<const float*>(weight_scales), static_cast<const float*>(bias),
        static_cast<const int32_t*>(task_table),
        static_cast<const int32_t*>(output_rows), output, task_count, n, k,
        expert_count, block_k, has_bias, has_output_rows, warps_per_block, stream);
  }
  if (format_id == 1) {
    return dispatch_grouped_fp8<cutlass::float_e5m2_t>(
        activation, weights_nk, static_cast<const float*>(activation_scales),
        static_cast<const float*>(weight_scales), static_cast<const float*>(bias),
        static_cast<const int32_t*>(task_table),
        static_cast<const int32_t*>(output_rows), output, task_count, n, k,
        expert_count, block_k, has_bias, has_output_rows, warps_per_block, stream);
  }
  return static_cast<int>(cudaErrorInvalidValue);
}

extern "C" int xqt_fp8_grouped_sm89_resource_query(
    int format_id,
    int block_k,
    int warps_per_block,
    int* registers_per_thread,
    int* static_shared_bytes,
    int* max_active_blocks_per_sm) {
  if (!registers_per_thread || !static_shared_bytes || !max_active_blocks_per_sm) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  cudaError_t status = cudaErrorInvalidValue;
  if (format_id == 0) {
    status = query_block_k<cutlass::float_e4m3_t>(
        block_k, warps_per_block, registers_per_thread, static_shared_bytes,
        max_active_blocks_per_sm);
  } else if (format_id == 1) {
    status = query_block_k<cutlass::float_e5m2_t>(
        block_k, warps_per_block, registers_per_thread, static_shared_bytes,
        max_active_blocks_per_sm);
  }
  return static_cast<int>(status);
}

extern "C" const char* xqt_fp8_grouped_sm89_version() {
  return "fp8-grouped-sm89-e4m3-e5m2-tensor-block32-64-128-warps1-2-4-8-v1";
}
