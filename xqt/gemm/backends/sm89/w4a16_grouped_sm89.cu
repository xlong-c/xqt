// SM89 grouped W4A16 decode GEMM.
//
// The host pre-packs expert weights as contiguous [E, N, packed_K] / [E, N, G]
// buffers and materializes a device task table. One task is one contiguous
// expert row tile. A single direct-grid launch covers every (task, output
// column) pair, so it does not dispatch one Python call or one CUDA launch per
// expert. The optional output-row permutation writes routed tokens directly
// back into their original order without a follow-up scatter kernel.

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <limits.h>

namespace {

constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;
constexpr int kPersistentMaxRows = 8;

template <int MaxRows>
__global__ void grouped_w4a16_decode_kernel(
    __half const* activation,
    unsigned char const* qweight,
    float const* scales,
    float const* zero_points,
    float const* bias,
    int const* tasks,
    int const* output_rows,
    __half* output,
    int n,
    int k,
    int padded_k,
    int group_size,
    int signed_nibble,
    int has_zero_points,
    int has_bias) {
  int work_item = static_cast<int>(blockIdx.x);
  int task_index = work_item / n;
  int column = work_item - task_index * n;
  int const* task = tasks + task_index * 3;
  int expert = task[0];
  int packed_row_base = task[1];
  int row_count = task[2];
  int packed_columns = (padded_k + 1) / 2;
  int group_count = (padded_k + group_size - 1) / group_size;

  float accumulators[MaxRows];
#pragma unroll
  for (int row = 0; row < MaxRows; ++row) {
    accumulators[row] = 0.0f;
  }

  unsigned char const* column_weight =
      qweight + (static_cast<size_t>(expert) * n + column) * packed_columns;
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
      size_t scale_index =
          (static_cast<size_t>(expert) * n + column) * group_count + group;
      float zero = has_zero_points ? zero_points[scale_index] : 0.0f;
      float weight_value = (static_cast<float>(code) - zero) * scales[scale_index];
#pragma unroll
      for (int row = 0; row < MaxRows; ++row) {
        if (row < row_count) {
          accumulators[row] +=
              __half2float(activation[(packed_row_base + row) * k + global_k]) * weight_value;
        }
      }
    }
  }

  __shared__ float partial[kWarps][MaxRows];
  int lane = threadIdx.x & 31;
  int warp = threadIdx.x >> 5;
#pragma unroll
  for (int row = 0; row < MaxRows; ++row) {
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

  if (threadIdx.x < row_count) {
    int row = threadIdx.x;
    float value = 0.0f;
#pragma unroll
    for (int warp_index = 0; warp_index < kWarps; ++warp_index) {
      value += partial[warp_index][row];
    }
    if (has_bias) {
      value += bias[static_cast<size_t>(expert) * n + column];
    }
    int packed_row = packed_row_base + row;
    int output_row = output_rows == nullptr ? packed_row : output_rows[packed_row];
    output[static_cast<size_t>(output_row) * n + column] = __float2half_rn(value);
  }
}

template <int MaxRows>
__device__ __forceinline__ void persistent_grouped_w4a16_work_item(
    __half const* activation,
    unsigned char const* qweight,
    float const* scales,
    float const* zero_points,
    float const* bias,
    int const* tasks,
    int const* output_rows,
    __half* output,
    int work_item,
    int n,
    int k,
    int padded_k,
    int group_size,
    int signed_nibble,
    int has_zero_points,
    int has_bias,
    float* partial) {
  int task_index = work_item / n;
  int column = work_item - task_index * n;
  int const* task = tasks + task_index * 3;
  int expert = task[0];
  int packed_row_base = task[1];
  int row_count = task[2];
  int packed_columns = (padded_k + 1) / 2;
  int group_count = (padded_k + group_size - 1) / group_size;

  float accumulators[MaxRows];
#pragma unroll
  for (int row = 0; row < MaxRows; ++row) {
    accumulators[row] = 0.0f;
  }

  unsigned char const* column_weight =
      qweight + (static_cast<size_t>(expert) * n + column) * packed_columns;
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
      size_t scale_index =
          (static_cast<size_t>(expert) * n + column) * group_count + group;
      float zero = has_zero_points ? zero_points[scale_index] : 0.0f;
      float weight_value = (static_cast<float>(code) - zero) * scales[scale_index];
#pragma unroll
      for (int row = 0; row < MaxRows; ++row) {
        if (row < row_count) {
          accumulators[row] +=
              __half2float(activation[(packed_row_base + row) * k + global_k]) * weight_value;
        }
      }
    }
  }

  int lane = threadIdx.x & 31;
  int warp = threadIdx.x >> 5;
#pragma unroll
  for (int row = 0; row < MaxRows; ++row) {
    float value = accumulators[row];
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    if (lane == 0) {
      partial[warp * kPersistentMaxRows + row] = value;
    }
  }
  __syncthreads();

  if (threadIdx.x < row_count) {
    int row = threadIdx.x;
    float value = 0.0f;
#pragma unroll
    for (int warp_index = 0; warp_index < kWarps; ++warp_index) {
      value += partial[warp_index * kPersistentMaxRows + row];
    }
    if (has_bias) {
      value += bias[static_cast<size_t>(expert) * n + column];
    }
    int packed_row = packed_row_base + row;
    int output_row = output_rows == nullptr ? packed_row : output_rows[packed_row];
    output[static_cast<size_t>(output_row) * n + column] = __float2half_rn(value);
  }
  __syncthreads();
}

__global__ void persistent_grouped_w4a16_decode_kernel(
    __half const* activation,
    unsigned char const* qweight,
    float const* scales,
    float const* zero_points,
    float const* bias,
    int const* tasks,
    int const* output_rows,
    __half* output,
    int work_items,
    int n,
    int k,
    int padded_k,
    int group_size,
    int signed_nibble,
    int has_zero_points,
    int has_bias) {
  __shared__ float partial[kWarps * kPersistentMaxRows];
  for (int work_item = static_cast<int>(blockIdx.x); work_item < work_items;
       work_item += static_cast<int>(gridDim.x)) {
    int task_index = work_item / n;
    int row_count = tasks[task_index * 3 + 2];
    if (row_count <= 1) {
      persistent_grouped_w4a16_work_item<1>(
          activation, qweight, scales, zero_points, bias, tasks, output_rows, output,
          work_item, n, k, padded_k, group_size, signed_nibble, has_zero_points,
          has_bias, partial);
    } else if (row_count <= 2) {
      persistent_grouped_w4a16_work_item<2>(
          activation, qweight, scales, zero_points, bias, tasks, output_rows, output,
          work_item, n, k, padded_k, group_size, signed_nibble, has_zero_points,
          has_bias, partial);
    } else if (row_count <= 4) {
      persistent_grouped_w4a16_work_item<4>(
          activation, qweight, scales, zero_points, bias, tasks, output_rows, output,
          work_item, n, k, padded_k, group_size, signed_nibble, has_zero_points,
          has_bias, partial);
    } else if (row_count <= 8) {
      persistent_grouped_w4a16_work_item<8>(
          activation, qweight, scales, zero_points, bias, tasks, output_rows, output,
          work_item, n, k, padded_k, group_size, signed_nibble, has_zero_points,
          has_bias, partial);
    }
  }
}

template <int MaxRows>
int run_grouped_bound(
    void const* activation,
    void const* qweight,
    void const* scales,
    void const* zero_points,
    void const* bias,
    void const* tasks,
    void const* output_rows,
    void* output,
    int work_items,
    int n,
    int k,
    int padded_k,
    int group_size,
    int signed_nibble,
    int has_zero_points,
    int has_bias,
    void* stream) {
  grouped_w4a16_decode_kernel<MaxRows><<<
      static_cast<unsigned>(work_items),
      kThreads,
      0,
      static_cast<cudaStream_t>(stream)>>>(
      static_cast<__half const*>(activation),
      static_cast<unsigned char const*>(qweight),
      static_cast<float const*>(scales),
      static_cast<float const*>(zero_points),
      static_cast<float const*>(bias),
      static_cast<int const*>(tasks),
      static_cast<int const*>(output_rows),
      static_cast<__half*>(output),
      n,
      k,
      padded_k,
      group_size,
      signed_nibble,
      has_zero_points,
      has_bias);
  return static_cast<int>(cudaGetLastError());
}

template <int MaxRows>
cudaError_t query_grouped_resources(
    int* registers_per_thread,
    int* static_shared_bytes,
    int* max_active_blocks_per_sm) {
  cudaFuncAttributes attributes{};
  cudaError_t status = cudaFuncGetAttributes(&attributes, grouped_w4a16_decode_kernel<MaxRows>);
  if (status != cudaSuccess) {
    return status;
  }
  int active_blocks = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks, grouped_w4a16_decode_kernel<MaxRows>, kThreads, 0);
  if (status != cudaSuccess) {
    return status;
  }
  *registers_per_thread = attributes.numRegs;
  *static_shared_bytes = static_cast<int>(attributes.sharedSizeBytes);
  *max_active_blocks_per_sm = active_blocks;
  return cudaSuccess;
}

cudaError_t query_persistent_resources(
    int* registers_per_thread,
    int* static_shared_bytes,
    int* max_active_blocks_per_sm) {
  cudaFuncAttributes attributes{};
  cudaError_t status =
      cudaFuncGetAttributes(&attributes, persistent_grouped_w4a16_decode_kernel);
  if (status != cudaSuccess) {
    return status;
  }
  int active_blocks = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks, persistent_grouped_w4a16_decode_kernel, kThreads, 0);
  if (status != cudaSuccess) {
    return status;
  }
  *registers_per_thread = attributes.numRegs;
  *static_shared_bytes = static_cast<int>(attributes.sharedSizeBytes);
  *max_active_blocks_per_sm = active_blocks;
  return cudaSuccess;
}

}  // namespace

extern "C" int xqt_w4a16_grouped_sm89_fp16_run(
    void const* activation,
    void const* qweight,
    void const* scales,
    void const* zero_points,
    void const* bias,
    void const* tasks,
    void const* output_rows,
    void* output,
    int task_count,
    int n,
    int k,
    int padded_k,
    int group_size,
    int expert_count,
    int signed_nibble,
    int has_zero_points,
    int has_bias,
    int row_bound,
    void* stream) {
  if (!activation || !qweight || !scales || !tasks || !output || task_count <= 0 || n <= 0 ||
      k <= 0 || k % 16 != 0 || padded_k < k || group_size <= 0 || expert_count <= 0 ||
      (has_zero_points && !zero_points) || (has_bias && !bias)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  long long work_items_64 = static_cast<long long>(task_count) * n;
  if (work_items_64 <= 0 || work_items_64 > INT_MAX) {
    return static_cast<int>(cudaErrorInvalidConfiguration);
  }
  int work_items = static_cast<int>(work_items_64);
  switch (row_bound) {
    case 1:
      return run_grouped_bound<1>(activation, qweight, scales, zero_points, bias, tasks,
                                  output_rows, output, work_items, n, k, padded_k, group_size,
                                  signed_nibble, has_zero_points, has_bias, stream);
    case 2:
      return run_grouped_bound<2>(activation, qweight, scales, zero_points, bias, tasks,
                                  output_rows, output, work_items, n, k, padded_k, group_size,
                                  signed_nibble, has_zero_points, has_bias, stream);
    case 4:
      return run_grouped_bound<4>(activation, qweight, scales, zero_points, bias, tasks,
                                  output_rows, output, work_items, n, k, padded_k, group_size,
                                  signed_nibble, has_zero_points, has_bias, stream);
    case 8:
      return run_grouped_bound<8>(activation, qweight, scales, zero_points, bias, tasks,
                                  output_rows, output, work_items, n, k, padded_k, group_size,
                                  signed_nibble, has_zero_points, has_bias, stream);
    default:
      return static_cast<int>(cudaErrorInvalidValue);
  }
}

extern "C" int xqt_w4a16_grouped_sm89_fp16_run_persistent(
    void const* activation,
    void const* qweight,
    void const* scales,
    void const* zero_points,
    void const* bias,
    void const* tasks,
    void const* output_rows,
    void* output,
    int task_count,
    int n,
    int k,
    int padded_k,
    int group_size,
    int expert_count,
    int signed_nibble,
    int has_zero_points,
    int has_bias,
    int grid_blocks,
    void* stream) {
  if (!activation || !qweight || !scales || !tasks || !output || task_count <= 0 || n <= 0 ||
      k <= 0 || k % 16 != 0 || padded_k < k || group_size <= 0 || expert_count <= 0 ||
      grid_blocks <= 0 || (has_zero_points && !zero_points) || (has_bias && !bias)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  long long work_items_64 = static_cast<long long>(task_count) * n;
  if (work_items_64 <= 0 || work_items_64 > INT_MAX || grid_blocks > work_items_64) {
    return static_cast<int>(cudaErrorInvalidConfiguration);
  }
  persistent_grouped_w4a16_decode_kernel<<<
      static_cast<unsigned>(grid_blocks),
      kThreads,
      0,
      static_cast<cudaStream_t>(stream)>>>(
      static_cast<__half const*>(activation),
      static_cast<unsigned char const*>(qweight),
      static_cast<float const*>(scales),
      static_cast<float const*>(zero_points),
      static_cast<float const*>(bias),
      static_cast<int const*>(tasks),
      static_cast<int const*>(output_rows),
      static_cast<__half*>(output),
      static_cast<int>(work_items_64),
      n,
      k,
      padded_k,
      group_size,
      signed_nibble,
      has_zero_points,
      has_bias);
  return static_cast<int>(cudaGetLastError());
}

extern "C" int xqt_w4a16_grouped_sm89_resource_query(
    int row_bound,
    int* registers_per_thread,
    int* static_shared_bytes,
    int* max_active_blocks_per_sm) {
  if (!registers_per_thread || !static_shared_bytes || !max_active_blocks_per_sm) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  cudaError_t status = cudaErrorInvalidValue;
  switch (row_bound) {
    case 1:
      status = query_grouped_resources<1>(registers_per_thread, static_shared_bytes,
                                          max_active_blocks_per_sm);
      break;
    case 2:
      status = query_grouped_resources<2>(registers_per_thread, static_shared_bytes,
                                          max_active_blocks_per_sm);
      break;
    case 4:
      status = query_grouped_resources<4>(registers_per_thread, static_shared_bytes,
                                          max_active_blocks_per_sm);
      break;
    case 8:
      status = query_grouped_resources<8>(registers_per_thread, static_shared_bytes,
                                          max_active_blocks_per_sm);
      break;
  }
  return static_cast<int>(status);
}

extern "C" int xqt_w4a16_grouped_sm89_persistent_resource_query(
    int* registers_per_thread,
    int* static_shared_bytes,
    int* max_active_blocks_per_sm) {
  if (!registers_per_thread || !static_shared_bytes || !max_active_blocks_per_sm) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  return static_cast<int>(query_persistent_resources(
      registers_per_thread, static_shared_bytes, max_active_blocks_per_sm));
}

extern "C" const char* xqt_w4a16_grouped_sm89_version() {
  return "sm89-grouped-w4a16-v2 direct bucketed persistent-grid-stride in-kernel-scatter";
}
