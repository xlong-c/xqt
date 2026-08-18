// SM89 experimental fused W4A16 mainloop built on a CUTLASS warp MMA.
//
// This is deliberately a separate artifact from w4a16_sm89.cu.  It decodes
// only the current K tile, applies [N,G] scale/zero-point before the MMA, and
// invokes CUTLASS's half Tensor Core warp primitive.  It is not the standard
// CUTLASS device::Gemm INT4 path: the group scale transform is owned by this
// custom mainloop and the entry remains manifest-gated until correctness and
// performance evidence are recorded.

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "cutlass/arch/arch.h"
#include "cutlass/gemm/warp/default_mma_tensor_op.h"
#include "cutlass/gemm/warp/default_mma_tensor_op_sm80.h"

using FusedMma = cutlass::gemm::warp::DefaultMmaTensorOp<
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::half_t,
    cutlass::layout::RowMajor,
    cutlass::half_t,
    cutlass::layout::ColumnMajor,
    float,
    cutlass::layout::RowMajor,
    cutlass::arch::OpMultiplyAdd>::Type;

template <typename T>
__device__ float load_activation(T value);

template <>
__device__ float load_activation<cutlass::half_t>(cutlass::half_t value) {
  return float(value);
}

template <typename T>
__device__ void store_output(T* destination, float value);

template <>
__device__ void store_output<cutlass::half_t>(cutlass::half_t* destination, float value) {
  *destination = __float2half_rn(value);
}

__global__ void w4a16_cutlass_fused_kernel(
    cutlass::half_t const* activation,
    unsigned char const* qweight,
    float const* scales,
    float const* zero_points,
    float const* bias,
    cutlass::half_t* output,
    int m,
    int n,
    int k,
    int padded_k,
    int group_size,
    int signed_nibble,
    int has_zero_points,
    int has_bias) {
  constexpr int tile_m = 16;
  constexpr int tile_n = 8;
  constexpr int tile_k = 16;
  __shared__ cutlass::half_t activation_tile[tile_m * tile_k];
  __shared__ cutlass::half_t weight_tile[tile_k * tile_n];

  int lane = threadIdx.x & 31;
  int row_base = static_cast<int>(blockIdx.y) * tile_m;
  int column_base = static_cast<int>(blockIdx.x) * tile_n;
  int packed_columns = (padded_k + 1) / 2;
  int group_count = (padded_k + group_size - 1) / group_size;

  FusedMma mma;
  using IteratorA = FusedMma::IteratorA;
  using IteratorB = FusedMma::IteratorB;
  typename IteratorA::Layout layout_a(tile_k);
  typename IteratorB::Layout layout_b(tile_k);
  FusedMma::FragmentC accumulators;
  accumulators.clear();
  for (int k_base = 0; k_base < k; k_base += tile_k) {
    for (int index = lane; index < tile_m * tile_k; index += 32) {
      int row = index / tile_k;
      int local_k = index - row * tile_k;
      int global_row = row_base + row;
      int global_k = k_base + local_k;
      activation_tile[layout_a({row, local_k})] = activation[global_row * k + global_k];
    }
    for (int index = lane; index < tile_k * tile_n; index += 32) {
      int local_k = index / tile_n;
      int column = index - local_k * tile_n;
      int global_column = column_base + column;
      int global_k = k_base + local_k;
      unsigned char packed = qweight[global_column * packed_columns + global_k / 2];
      int code = (global_k & 1) == 0 ? static_cast<int>(packed & 0x0f)
                                    : static_cast<int>((packed >> 4) & 0x0f);
      if (signed_nibble && code >= 8) {
        code -= 16;
      }
      int group = global_k / group_size;
      float zero = has_zero_points
                       ? zero_points[global_column * group_count + group]
                       : 0.0f;
      float value = (static_cast<float>(code) - zero) *
                    scales[global_column * group_count + group];
      weight_tile[layout_b({local_k, column})] = cutlass::half_t(value);
    }
    __syncthreads();

    IteratorA iterator_a(
        typename IteratorA::TensorRef(activation_tile, layout_a),
        lane);
    IteratorB iterator_b(
        typename IteratorB::TensorRef(weight_tile, layout_b),
        lane);
    FusedMma::FragmentA fragment_a;
    FusedMma::FragmentB fragment_b;
    FusedMma::TransformedFragmentA transformed_a;
    FusedMma::TransformedFragmentB transformed_b;
    iterator_a.load(fragment_a);
    iterator_b.load(fragment_b);
    mma.transform(transformed_a, transformed_b, fragment_a, fragment_b);
    mma(accumulators, transformed_a, transformed_b, accumulators);
    __syncthreads();
  }

  // CUTLASS's row-major accumulator iterator maps each quad to one output
  // row and each lane to two consecutive columns.  Keep the mapping local so
  // the fused entry can write half output without materializing a float tile.
  int quad = lane >> 2;
  int lane_in_quad = lane & 3;
  for (int row = 0; row < 2; ++row) {
    for (int col = 0; col < 2; ++col) {
      int accumulator_index = row * 2 + col;
      int global_row = row_base + quad + row * 8;
      int global_column = column_base + lane_in_quad * 2 + col;
      float value = accumulators[accumulator_index];
      if (has_bias) {
        value += bias[global_column];
      }
      output[global_row * n + global_column] = cutlass::half_t(value);
    }
  }
}

// Split-K partial kernel: blockIdx.z selects a contiguous slice of K tiles and
// the warp accumulates only that slice into a per-split float workspace.  The
// group scale/zero-point transform stays inside the K-tile decode, so group
// semantics do not change when a group straddles a split boundary.
__global__ void w4a16_cutlass_fused_splitk_partial_kernel(
    cutlass::half_t const* activation,
    unsigned char const* qweight,
    float const* scales,
    float const* zero_points,
    float* workspace,
    int n,
    int k,
    int padded_k,
    int group_size,
    int signed_nibble,
    int has_zero_points,
    int k_per_split) {
  constexpr int tile_m = 16;
  constexpr int tile_n = 8;
  constexpr int tile_k = 16;
  __shared__ cutlass::half_t activation_tile[tile_m * tile_k];
  __shared__ cutlass::half_t weight_tile[tile_k * tile_n];

  int lane = threadIdx.x & 31;
  int row_base = static_cast<int>(blockIdx.y) * tile_m;
  int column_base = static_cast<int>(blockIdx.x) * tile_n;
  int split = static_cast<int>(blockIdx.z);
  int k_begin = split * k_per_split;
  int k_end = min(k_begin + k_per_split, k);
  int packed_columns = (padded_k + 1) / 2;
  int group_count = (padded_k + group_size - 1) / group_size;

  FusedMma mma;
  using IteratorA = FusedMma::IteratorA;
  using IteratorB = FusedMma::IteratorB;
  typename IteratorA::Layout layout_a(tile_k);
  typename IteratorB::Layout layout_b(tile_k);
  FusedMma::FragmentC accumulators;
  accumulators.clear();
  for (int k_base = k_begin; k_base < k_end; k_base += tile_k) {
    for (int index = lane; index < tile_m * tile_k; index += 32) {
      int row = index / tile_k;
      int local_k = index - row * tile_k;
      int global_row = row_base + row;
      int global_k = k_base + local_k;
      activation_tile[layout_a({row, local_k})] = activation[global_row * k + global_k];
    }
    for (int index = lane; index < tile_k * tile_n; index += 32) {
      int local_k = index / tile_n;
      int column = index - local_k * tile_n;
      int global_column = column_base + column;
      int global_k = k_base + local_k;
      unsigned char packed = qweight[global_column * packed_columns + global_k / 2];
      int code = (global_k & 1) == 0 ? static_cast<int>(packed & 0x0f)
                                    : static_cast<int>((packed >> 4) & 0x0f);
      if (signed_nibble && code >= 8) {
        code -= 16;
      }
      int group = global_k / group_size;
      float zero = has_zero_points
                       ? zero_points[global_column * group_count + group]
                       : 0.0f;
      float value = (static_cast<float>(code) - zero) *
                    scales[global_column * group_count + group];
      weight_tile[layout_b({local_k, column})] = cutlass::half_t(value);
    }
    __syncthreads();

    IteratorA iterator_a(
        typename IteratorA::TensorRef(activation_tile, layout_a),
        lane);
    IteratorB iterator_b(
        typename IteratorB::TensorRef(weight_tile, layout_b),
        lane);
    FusedMma::FragmentA fragment_a;
    FusedMma::FragmentB fragment_b;
    FusedMma::TransformedFragmentA transformed_a;
    FusedMma::TransformedFragmentB transformed_b;
    iterator_a.load(fragment_a);
    iterator_b.load(fragment_b);
    mma.transform(transformed_a, transformed_b, fragment_a, fragment_b);
    mma(accumulators, transformed_a, transformed_b, accumulators);
    __syncthreads();
  }

  int quad = lane >> 2;
  int lane_in_quad = lane & 3;
  float* split_workspace = workspace + static_cast<long long>(split) * gridDim.y * tile_m * n;
  for (int row = 0; row < 2; ++row) {
    for (int col = 0; col < 2; ++col) {
      int accumulator_index = row * 2 + col;
      int global_row = row_base + quad + row * 8;
      int global_column = column_base + lane_in_quad * 2 + col;
      split_workspace[global_row * n + global_column] = accumulators[accumulator_index];
    }
  }
}

// Reduction pass: sum the per-split float partials, apply bias exactly once,
// and emit the fp16 result.  A second kernel plus a [split, M, N] float
// workspace is chosen over atomicAdd for two reasons: accumulation order is
// deterministic so results are reproducible run to run, and the workspace
// traffic (split_k * M * N * 8 bytes written+read) is small next to the K
// serial mainloop it replaces for the prefill shapes this path targets.
// Atomics would save one launch and the workspace but serialize on conflicting
// output addresses and make numerics nondeterministic.
__global__ void w4a16_cutlass_fused_splitk_reduce_kernel(
    float const* workspace,
    float const* bias,
    cutlass::half_t* output,
    int element_count,
    int n,
    int split_count,
    int has_bias) {
  int index = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= element_count) {
    return;
  }
  float value = 0.0f;
  for (int split = 0; split < split_count; ++split) {
    value += workspace[static_cast<long long>(split) * element_count + index];
  }
  if (has_bias) {
    value += bias[index % n];
  }
  output[index] = cutlass::half_t(value);
}

extern "C" int xqt_w4a16_cutlass_fused_sm89_fp16_run_splitk(
    void const* activation,
    void const* qweight,
    void const* scales,
    void const* zero_points,
    void const* bias,
    void* output,
    void* workspace,
    int m,
    int n,
    int k,
    int padded_k,
    int group_size,
    int signed_nibble,
    int has_zero_points,
    int has_bias,
    int split_k,
    void* stream) {
  if (!activation || !qweight || !scales || !output || !workspace || m <= 0 ||
      n <= 0 || k <= 0 || m % 16 != 0 || n % 8 != 0 || k % 16 != 0 ||
      padded_k < k || group_size <= 0 || split_k < 2) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  int k_tiles = (k + 15) / 16;
  int tiles_per_split = (k_tiles + split_k - 1) / split_k;
  int k_per_split = tiles_per_split * 16;
  int split_count = (k + k_per_split - 1) / k_per_split;
  cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
  dim3 partial_blocks(
      static_cast<unsigned>(n / 8),
      static_cast<unsigned>(m / 16),
      static_cast<unsigned>(split_count));
  w4a16_cutlass_fused_splitk_partial_kernel<<<partial_blocks, 32, 0, cuda_stream>>>(
      static_cast<cutlass::half_t const*>(activation),
      static_cast<unsigned char const*>(qweight),
      static_cast<float const*>(scales),
      static_cast<float const*>(zero_points),
      static_cast<float*>(workspace),
      n,
      k,
      padded_k,
      group_size,
      signed_nibble,
      has_zero_points,
      k_per_split);
  int element_count = m * n;
  int reduce_threads = 256;
  int reduce_blocks = (element_count + reduce_threads - 1) / reduce_threads;
  w4a16_cutlass_fused_splitk_reduce_kernel<<<reduce_blocks, reduce_threads, 0, cuda_stream>>>(
      static_cast<float const*>(workspace),
      static_cast<float const*>(bias),
      static_cast<cutlass::half_t*>(output),
      element_count,
      n,
      split_count,
      has_bias);
  return static_cast<int>(cudaGetLastError());
}

// Decode-oriented fused path for M=1..8.  The m16n8k16 warp MMA above would
// pad a single decode row to a 16-row tile and waste 15/16 of its compute, so
// this kernel stays SIMT: one block per output column, threads stride K in
// packed-byte steps (one byte load feeds two nibbles, halving byte traffic
// versus striding K element-wise), each thread keeps one accumulator per row
// and reuses the decoded weight across all M rows.  K partials are reduced
// with a warp shuffle followed by a shared-memory tree across the 8 warps.
// MAX_ROWS is a compile-time row bound (1/2/4/8, the host rounds M up to the
// next bound) so M=1 pays no accumulator or shuffle cost for rows it does not
// have; a runtime 8-row upper bound measurably loses to the legacy m1_gemv at
// M=1.  Weight traffic matches the old m1_gemv/small_m variants (each packed
// byte is read once per column), but K parallelism is 256-wide instead of one
// thread per column.  Nibble decode, signed/unsigned handling and the group
// scale/zero-point transform are byte-identical to the prefill mainloop, so
// the whole artifact still has exactly one nibble semantics.  uint32 vector
// loads are rejected on purpose: a packed row starts at column*packed_columns,
// which is not guaranteed 4-byte aligned for arbitrary padded_k.
template <int max_rows>
__global__ void w4a16_cutlass_fused_decode_kernel(
    cutlass::half_t const* activation,
    unsigned char const* qweight,
    float const* scales,
    float const* zero_points,
    float const* bias,
    cutlass::half_t* output,
    int m,
    int n,
    int k,
    int padded_k,
    int group_size,
    int signed_nibble,
    int has_zero_points,
    int has_bias) {
  int column = static_cast<int>(blockIdx.x);
  if (column >= n) {
    return;
  }
  int packed_columns = (padded_k + 1) / 2;
  int group_count = (padded_k + group_size - 1) / group_size;
  unsigned char const* column_weight = qweight + column * packed_columns;
  float accumulators[max_rows];
#pragma unroll
  for (int row = 0; row < max_rows; ++row) {
    accumulators[row] = 0.0f;
  }
  int byte_count = k / 2;
  for (int byte = threadIdx.x; byte < byte_count; byte += blockDim.x) {
    unsigned char packed = column_weight[byte];
#pragma unroll
    for (int half_index = 0; half_index < 2; ++half_index) {
      int global_k = byte * 2 + half_index;
      int code = half_index == 0 ? static_cast<int>(packed & 0x0f)
                                 : static_cast<int>((packed >> 4) & 0x0f);
      if (signed_nibble && code >= 8) {
        code -= 16;
      }
      int group = global_k / group_size;
      float zero = has_zero_points
                       ? zero_points[column * group_count + group]
                       : 0.0f;
      float weight_value = (static_cast<float>(code) - zero) *
                           scales[column * group_count + group];
#pragma unroll
      for (int row = 0; row < max_rows; ++row) {
        if (row < m) {
          accumulators[row] += float(activation[row * k + global_k]) * weight_value;
        }
      }
    }
  }

  // 256 threads = 8 warps.  Each warp shuffles its 32 lane partials down to
  // lane 0, which writes one float per (warp, row); the first m threads then
  // reduce the 8 warp partials per row and store the fp16 result.
  __shared__ float partial[8][max_rows];
  int lane = threadIdx.x & 31;
  int warp = threadIdx.x >> 5;
#pragma unroll
  for (int row = 0; row < max_rows; ++row) {
    float value = accumulators[row];
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    if (lane == 0) {
      partial[warp][row] = value;
    }
  }
  __syncthreads();
  if (threadIdx.x < m) {
    int row = threadIdx.x;
    float value = 0.0f;
#pragma unroll
    for (int warp_index = 0; warp_index < 8; ++warp_index) {
      value += partial[warp_index][row];
    }
    if (has_bias) {
      value += bias[column];
    }
    output[row * n + column] = cutlass::half_t(value);
  }
}

template <int max_rows>
static int run_decode_bound(
    void const* activation,
    void const* qweight,
    void const* scales,
    void const* zero_points,
    void const* bias,
    void* output,
    int m,
    int n,
    int k,
    int padded_k,
    int group_size,
    int signed_nibble,
    int has_zero_points,
    int has_bias,
    void* stream) {
  w4a16_cutlass_fused_decode_kernel<max_rows><<<
      static_cast<unsigned>(n),
      256,
      0,
      static_cast<cudaStream_t>(stream)>>>(
      static_cast<cutlass::half_t const*>(activation),
      static_cast<unsigned char const*>(qweight),
      static_cast<float const*>(scales),
      static_cast<float const*>(zero_points),
      static_cast<float const*>(bias),
      static_cast<cutlass::half_t*>(output),
      m,
      n,
      k,
      padded_k,
      group_size,
      signed_nibble,
      has_zero_points,
      has_bias);
  return static_cast<int>(cudaGetLastError());
}

extern "C" int xqt_w4a16_cutlass_fused_sm89_fp16_run_decode(
    void const* activation,
    void const* qweight,
    void const* scales,
    void const* zero_points,
    void const* bias,
    void* output,
    int m,
    int n,
    int k,
    int padded_k,
    int group_size,
    int signed_nibble,
    int has_zero_points,
    int has_bias,
    void* stream) {
  if (!activation || !qweight || !scales || !output || m < 1 || m > 8 || n <= 0 ||
      k <= 0 || k % 16 != 0 || padded_k < k || group_size <= 0) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  if (m == 1) {
    return run_decode_bound<1>(activation, qweight, scales, zero_points, bias, output,
                               m, n, k, padded_k, group_size, signed_nibble,
                               has_zero_points, has_bias, stream);
  }
  if (m == 2) {
    return run_decode_bound<2>(activation, qweight, scales, zero_points, bias, output,
                               m, n, k, padded_k, group_size, signed_nibble,
                               has_zero_points, has_bias, stream);
  }
  if (m <= 4) {
    return run_decode_bound<4>(activation, qweight, scales, zero_points, bias, output,
                               m, n, k, padded_k, group_size, signed_nibble,
                               has_zero_points, has_bias, stream);
  }
  return run_decode_bound<8>(activation, qweight, scales, zero_points, bias, output,
                             m, n, k, padded_k, group_size, signed_nibble,
                             has_zero_points, has_bias, stream);
}

extern "C" int xqt_w4a16_cutlass_fused_sm89_fp16_run(
    void const* activation,
    void const* qweight,
    void const* scales,
    void const* zero_points,
    void const* bias,
    void* output,
    int m,
    int n,
    int k,
    int padded_k,
    int group_size,
    int signed_nibble,
    int has_zero_points,
    int has_bias,
    void* stream) {
  if (!activation || !qweight || !scales || !output || m <= 0 || n <= 0 || k <= 0 ||
      m % 16 != 0 || n % 8 != 0 || k % 16 != 0 || padded_k < k || group_size <= 0) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  dim3 blocks(static_cast<unsigned>(n / 8), static_cast<unsigned>(m / 16), 1);
  w4a16_cutlass_fused_kernel<<<blocks, 32, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<cutlass::half_t const*>(activation),
      static_cast<unsigned char const*>(qweight),
      static_cast<float const*>(scales),
      static_cast<float const*>(zero_points),
      static_cast<float const*>(bias),
      static_cast<cutlass::half_t*>(output),
      m,
      n,
      k,
      padded_k,
      group_size,
      signed_nibble,
      has_zero_points,
      has_bias);
  return static_cast<int>(cudaGetLastError());
}

extern "C" const char* xqt_w4a16_cutlass_fused_sm89_version() {
  return "sm89-w4a16-cutlass-fused-mma-v3 tile=16x8x16 groupwise-scale splitk-workspace decode-m1-8";
}
