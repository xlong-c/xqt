// SM89 grouped W8A8 decode kernel.
//
// One warp computes one 16x8 MMA tile for one routed task.  The logical task
// has at most eight rows; the second half of the 16-row operand is zero-filled.
// Weights are stored once in contiguous [expert,N,K] order, so one launch can
// cover all experts and all N tiles without a Python expert loop.

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace {

constexpr int kRows = 16;
constexpr int kCols = 8;
constexpr int kKTile = 32;
// The row XOR reaches byte offsets 32..63 for row classes 2 and 3.
// A 64-byte stride is therefore part of the fragment layout, not padding.
constexpr int kALd = 64;
constexpr int kBLd = 32;

__device__ __forceinline__ int a_offset(int row, int col) {
  return row * kALd + (col ^ ((row & 3) << 4));
}

__device__ __forceinline__ void mma_s8s8s32_m16n8k32(
    int& d0,
    int& d1,
    int& d2,
    int& d3,
    uint32_t a0,
    uint32_t a1,
    uint32_t a2,
    uint32_t a3,
    uint32_t b0,
    uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void load_a_fragment(
    uint32_t& a0,
    uint32_t& a1,
    uint32_t& a2,
    uint32_t& a3,
    const int8_t* shared_a,
    int lane) {
  const int group = lane >> 2;
  const int quad_lane = lane & 3;
  const int col = quad_lane * 4;
  auto load4 = [&](int row, int column) -> uint32_t {
    return *reinterpret_cast<const uint32_t*>(shared_a + a_offset(row, column));
  };
  a0 = load4(group, col);
  a1 = load4(group + 8, col);
  a2 = load4(group, col + 16);
  a3 = load4(group + 8, col + 16);
}

__device__ __forceinline__ void load_b_fragment(
    uint32_t& b0,
    uint32_t& b1,
    const int8_t* shared_b,
    int lane) {
  const int group = lane >> 2;
  const int quad_lane = lane & 3;
  const int column = group;
  b0 = *reinterpret_cast<const uint32_t*>(shared_b + column * kBLd + quad_lane * 4);
  b1 = *reinterpret_cast<const uint32_t*>(shared_b + column * kBLd + quad_lane * 4 + 16);
}

template <int Warps>
__global__ void grouped_w8a8_kernel(
    const int8_t* __restrict__ activation,
    const int8_t* __restrict__ weights_nk,
    const float* __restrict__ weight_scales,
    const float* __restrict__ activation_scales,
    const float* __restrict__ bias,
    const int32_t* __restrict__ task_table,
    const int32_t* __restrict__ output_rows,
    half* __restrict__ output,
    int task_count,
    int n,
    int k,
    int expert_count,
    int has_bias,
    int has_output_rows,
    int activation_per_token) {
  static_assert(Warps == 1 || Warps == 2 || Warps == 4 || Warps == 8);
  constexpr int kThreads = Warps * 32;
  constexpr int kBlockCols = Warps * kCols;
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

  __shared__ __align__(16) int8_t shared_a[kRows * kALd];
  __shared__ __align__(16) int8_t shared_b[Warps * kCols * kBLd];
  const int thread = static_cast<int>(threadIdx.x);
  const int lane = thread & 31;
  const int warp = thread >> 5;
  const int n_tile = block_n + warp * kCols;
  int8_t* warp_b = shared_b + warp * kCols * kBLd;
  int acc[4] = {0, 0, 0, 0};
  const int k_tiles = (k + kKTile - 1) / kKTile;
  const int64_t weight_base = static_cast<int64_t>(expert) * n * k;

  for (int kt = 0; kt < k_tiles; ++kt) {
    for (int index = thread; index < kRows * kKTile; index += kThreads) {
      const int row = index / kKTile;
      const int col = index - row * kKTile;
      const int global_row = row_base + row;
      const int global_col = kt * kKTile + col;
      int8_t value = 0;
      if (row < row_count && global_col < k) {
        value = activation[static_cast<int64_t>(global_row) * k + global_col];
      }
      shared_a[a_offset(row, col)] = value;
    }
    for (int index = lane; index < kCols * kKTile; index += 32) {
      const int column = index / kKTile;
      const int col = index - column * kKTile;
      const int global_col = n_tile + column;
      const int global_k = kt * kKTile + col;
      int8_t value = 0;
      if (global_col < n && global_k < k) {
        value = weights_nk[weight_base + static_cast<int64_t>(global_col) * k + global_k];
      }
      warp_b[column * kBLd + col] = value;
    }
    __syncthreads();
    uint32_t a0, a1, a2, a3, b0, b1;
    load_a_fragment(a0, a1, a2, a3, shared_a, lane);
    load_b_fragment(b0, b1, warp_b, lane);
    mma_s8s8s32_m16n8k32(acc[0], acc[1], acc[2], acc[3], a0, a1, a2, a3, b0, b1);
    __syncthreads();
  }

  const int group = lane >> 2;
  const int quad_lane = lane & 3;
  const int col0 = n_tile + quad_lane * 2;
  const float* expert_scales = weight_scales + static_cast<int64_t>(expert) * n;
  const float* expert_bias = has_bias ? bias + static_cast<int64_t>(expert) * n : nullptr;
  const int row0 = row_base + group;
  const int row1 = row_base + group + 8;
  const int rows[2] = {row0, row1};
  const int values[4] = {acc[0], acc[1], acc[2], acc[3]};
  for (int half = 0; half < 2; ++half) {
    const int local_row = group + half * 8;
    if (local_row >= row_count) {
      continue;
    }
    const int source_row = rows[half];
    const int destination_row = has_output_rows ? output_rows[source_row] : source_row;
    const float activation_scale = activation_per_token ? activation_scales[source_row]
                                                        : activation_scales[0];
    if (col0 < n) {
      const float scale0 = activation_scale * expert_scales[col0];
      const float bias0 = has_bias ? expert_bias[col0] : 0.0f;
      output[static_cast<int64_t>(destination_row) * n + col0] =
          __float2half(static_cast<float>(values[half * 2]) * scale0 + bias0);
    }
    if (col0 + 1 < n) {
      const float scale1 = activation_scale * expert_scales[col0 + 1];
      const float bias1 = has_bias ? expert_bias[col0 + 1] : 0.0f;
      output[static_cast<int64_t>(destination_row) * n + col0 + 1] =
          __float2half(static_cast<float>(values[half * 2 + 1]) * scale1 + bias1);
    }
  }
}

}  // namespace

extern "C" int xqt_w8a8_grouped_sm89_fp16_run(
    const void* activation,
    const void* weights_nk,
    const void* weight_scales,
    const void* activation_scales,
    const void* bias,
    const void* task_table,
    const void* output_rows,
    void* output,
    int task_count,
    int n,
    int k,
    int expert_count,
    int has_bias,
    int has_output_rows,
    int activation_per_token,
    int warps_per_block,
    void* stream_ptr) {
  if (!activation || !weights_nk || !weight_scales || !activation_scales || !task_table || !output ||
      task_count <= 0 || n <= 0 || k <= 0 || expert_count <= 0) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  if (has_bias && !bias) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  if (has_output_rows && !output_rows) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
#define XQT_LAUNCH_GROUPED_W8A8(WARPS)                                                     \
  do {                                                                                     \
    const dim3 grid((n + (WARPS) * kCols - 1) / ((WARPS) * kCols), task_count, 1);         \
    grouped_w8a8_kernel<WARPS><<<grid, (WARPS) * 32, 0, stream>>>(                         \
        static_cast<const int8_t*>(activation),                                             \
        static_cast<const int8_t*>(weights_nk),                                             \
        static_cast<const float*>(weight_scales),                                           \
        static_cast<const float*>(activation_scales),                                       \
        static_cast<const float*>(bias),                                                    \
        static_cast<const int32_t*>(task_table),                                            \
        static_cast<const int32_t*>(output_rows),                                           \
        static_cast<half*>(output),                                                         \
        task_count, n, k, expert_count, has_bias, has_output_rows, activation_per_token);   \
  } while (false)
  switch (warps_per_block) {
    case 1:
      XQT_LAUNCH_GROUPED_W8A8(1);
      break;
    case 2:
      XQT_LAUNCH_GROUPED_W8A8(2);
      break;
    case 4:
      XQT_LAUNCH_GROUPED_W8A8(4);
      break;
    case 8:
      XQT_LAUNCH_GROUPED_W8A8(8);
      break;
    default:
      return static_cast<int>(cudaErrorInvalidValue);
  }
#undef XQT_LAUNCH_GROUPED_W8A8
  return static_cast<int>(cudaGetLastError());
}

extern "C" int xqt_w8a8_grouped_sm89_resource_query(
    int warps_per_block,
    int* registers_per_thread,
    int* static_shared_bytes,
    int* max_active_blocks_per_sm) {
  if (!registers_per_thread || !static_shared_bytes || !max_active_blocks_per_sm) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  cudaFuncAttributes attributes{};
  cudaError_t status = cudaSuccess;
  switch (warps_per_block) {
    case 1:
      status = cudaFuncGetAttributes(&attributes, grouped_w8a8_kernel<1>);
      break;
    case 2:
      status = cudaFuncGetAttributes(&attributes, grouped_w8a8_kernel<2>);
      break;
    case 4:
      status = cudaFuncGetAttributes(&attributes, grouped_w8a8_kernel<4>);
      break;
    case 8:
      status = cudaFuncGetAttributes(&attributes, grouped_w8a8_kernel<8>);
      break;
    default:
      return static_cast<int>(cudaErrorInvalidValue);
  }
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  *registers_per_thread = attributes.numRegs;
  *static_shared_bytes = attributes.sharedSizeBytes;
  int blocks = 0;
  switch (warps_per_block) {
    case 1:
      status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &blocks, grouped_w8a8_kernel<1>, 32, 0);
      break;
    case 2:
      status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &blocks, grouped_w8a8_kernel<2>, 64, 0);
      break;
    case 4:
      status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &blocks, grouped_w8a8_kernel<4>, 128, 0);
      break;
    case 8:
      status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &blocks, grouped_w8a8_kernel<8>, 256, 0);
      break;
  }
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  *max_active_blocks_per_sm = blocks;
  return static_cast<int>(cudaSuccess);
}

extern "C" const char* xqt_w8a8_grouped_sm89_version() {
  return "w8a8-grouped-sm89-mma-m16n8k32-shared-a-warps-1-2-4-8-v2";
}
