// SM89 W4A16 correctness/fallback kernel.
//
// This is intentionally a separate dequant_fallback artifact. It reads the
// canonical low/high nibble layout and applies group scales in the K loop, but
// it is not the production CUTLASS INT4 mainloop. The registry must not label
// this path as native W4A16 CUTLASS.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

template <typename T>
__device__ float load_value(T value);

template <>
__device__ float load_value<half>(half value) {
  return __half2float(value);
}

template <>
__device__ float load_value<__nv_bfloat16>(__nv_bfloat16 value) {
  return __bfloat162float(value);
}

template <typename T>
__device__ void store_value(T* destination, float value);

template <>
__device__ void store_value<half>(half* destination, float value) {
  *destination = __float2half_rn(value);
}

template <>
__device__ void store_value<__nv_bfloat16>(__nv_bfloat16* destination, float value) {
  *destination = __float2bfloat16(value);
}

template <typename T>
__global__ void w4a16_dequant_kernel(
    T const* activation,
    unsigned char const* qweight,
    float const* scales,
    float const* zero_points,
    float const* bias,
    T* output,
    int m,
    int n,
    int k,
    int padded_k,
    int group_size,
    int signed_nibble,
    int has_zero_points,
    int has_bias) {
  int linear = blockIdx.x * blockDim.x + threadIdx.x;
  int total = m * n;
  if (linear >= total) {
    return;
  }
  int row = linear / n;
  int column = linear - row * n;
  int packed_columns = (padded_k + 1) / 2;
  int group_count = (padded_k + group_size - 1) / group_size;
  float accumulator = 0.0f;
  for (int kk = 0; kk < k; ++kk) {
    unsigned char packed = qweight[column * packed_columns + kk / 2];
    int code = (kk & 1) == 0 ? static_cast<int>(packed & 0x0f)
                              : static_cast<int>((packed >> 4) & 0x0f);
    if (signed_nibble && code >= 8) {
      code -= 16;
    }
    int group = kk / group_size;
    float zero = has_zero_points ? zero_points[column * group_count + group] : 0.0f;
    float weight_value = (static_cast<float>(code) - zero) *
                         scales[column * group_count + group];
    accumulator += load_value<T>(activation[row * k + kk]) * weight_value;
  }
  if (has_bias) {
    accumulator += bias[column];
  }
  store_value<T>(&output[row * n + column], accumulator);
}

// Decode one output column per block.  This is the decode-oriented M=1 path:
// threads cooperate over K, so each weight nibble is read once per reduction
// lane instead of launching one thread for the whole dot product.
template <typename T>
__global__ void w4a16_m1_gemv_kernel(
    T const* activation,
    unsigned char const* qweight,
    float const* scales,
    float const* zero_points,
    float const* bias,
    T* output,
    int n,
    int k,
    int padded_k,
    int group_size,
    int signed_nibble,
    int has_zero_points,
    int has_bias) {
  int column = blockIdx.x;
  if (column >= n) {
    return;
  }
  int packed_columns = (padded_k + 1) / 2;
  int group_count = (padded_k + group_size - 1) / group_size;
  float accumulator = 0.0f;
  for (int kk = threadIdx.x; kk < k; kk += blockDim.x) {
    unsigned char packed = qweight[column * packed_columns + kk / 2];
    int code = (kk & 1) == 0 ? static_cast<int>(packed & 0x0f)
                              : static_cast<int>((packed >> 4) & 0x0f);
    if (signed_nibble && code >= 8) {
      code -= 16;
    }
    int group = kk / group_size;
    float zero = has_zero_points ? zero_points[column * group_count + group] : 0.0f;
    float weight_value = (static_cast<float>(code) - zero) *
                         scales[column * group_count + group];
    accumulator += load_value<T>(activation[kk]) * weight_value;
  }

  __shared__ float partial[256];
  partial[threadIdx.x] = accumulator;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      partial[threadIdx.x] += partial[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    float value = partial[0] + (has_bias ? bias[column] : 0.0f);
    store_value<T>(&output[column], value);
  }
}

// One thread owns an output column and accumulates up to eight rows.  The
// weight decode and group-scale lookup are shared across those rows, which is
// the useful reuse pattern for decode and small prefill batches.
template <typename T>
__global__ void w4a16_small_m_kernel(
    T const* activation,
    unsigned char const* qweight,
    float const* scales,
    float const* zero_points,
    float const* bias,
    T* output,
    int m,
    int n,
    int k,
    int padded_k,
    int group_size,
    int signed_nibble,
    int has_zero_points,
    int has_bias) {
  int column = blockIdx.x * blockDim.x + threadIdx.x;
  if (column >= n) {
    return;
  }
  float accumulators[8] = {0.0f, 0.0f, 0.0f, 0.0f,
                           0.0f, 0.0f, 0.0f, 0.0f};
  int packed_columns = (padded_k + 1) / 2;
  int group_count = (padded_k + group_size - 1) / group_size;
  for (int kk = 0; kk < k; ++kk) {
    unsigned char packed = qweight[column * packed_columns + kk / 2];
    int code = (kk & 1) == 0 ? static_cast<int>(packed & 0x0f)
                              : static_cast<int>((packed >> 4) & 0x0f);
    if (signed_nibble && code >= 8) {
      code -= 16;
    }
    int group = kk / group_size;
    float zero = has_zero_points ? zero_points[column * group_count + group] : 0.0f;
    float weight_value = (static_cast<float>(code) - zero) *
                         scales[column * group_count + group];
#pragma unroll
    for (int row = 0; row < 8; ++row) {
      if (row < m) {
        accumulators[row] += load_value<T>(activation[row * k + kk]) * weight_value;
      }
    }
  }
  for (int row = 0; row < m; ++row) {
    float value = accumulators[row] + (has_bias ? bias[column] : 0.0f);
    store_value<T>(&output[row * n + column], value);
  }
}

// Prefill-oriented tile kernel.  It keeps only one 8x32 activation tile and
// one 16x32 decoded weight tile in shared memory, so a decoded weight is reused
// by eight output rows and an activation by sixteen output columns.
template <typename T>
__global__ void w4a16_tiled_kernel(
    T const* activation,
    unsigned char const* qweight,
    float const* scales,
    float const* zero_points,
    float const* bias,
    T* output,
    int m,
    int n,
    int k,
    int padded_k,
    int group_size,
    int signed_nibble,
    int has_zero_points,
    int has_bias,
    int persistent) {
  constexpr int tile_m = 8;
  constexpr int tile_n = 16;
  constexpr int tile_k = 32;
  extern __shared__ unsigned char shared_bytes[];
  T* activation_tile = reinterpret_cast<T*>(shared_bytes);
  float* weight_tile = reinterpret_cast<float*>(
      shared_bytes + tile_m * tile_k * static_cast<int>(sizeof(T)));
  int packed_columns = (padded_k + 1) / 2;
  int group_count = (padded_k + group_size - 1) / group_size;
  int tile_count_x = (n + tile_n - 1) / tile_n;
  int tile_count_y = (m + tile_m - 1) / tile_m;
  int total_tiles = tile_count_x * tile_count_y;
  int tile_id = persistent ? static_cast<int>(blockIdx.x)
                           : static_cast<int>(blockIdx.y) * tile_count_x + blockIdx.x;
  int tile_stride = persistent ? static_cast<int>(gridDim.x) : total_tiles;

  for (; tile_id < total_tiles; tile_id += tile_stride) {
    int row_base = (tile_id / tile_count_x) * tile_m;
    int column_base = (tile_id - (tile_id / tile_count_x) * tile_count_x) * tile_n;
    float accumulator = 0.0f;
    for (int k_base = 0; k_base < k; k_base += tile_k) {
      for (int index = threadIdx.x; index < tile_m * tile_k; index += blockDim.x) {
        int row = index / tile_k;
        int local_k = index - row * tile_k;
        int global_row = row_base + row;
        int global_k = k_base + local_k;
        activation_tile[index] = (global_row < m && global_k < k)
                                    ? activation[global_row * k + global_k]
                                    : T(0);
      }
      for (int index = threadIdx.x; index < tile_n * tile_k; index += blockDim.x) {
        int column = index / tile_k;
        int local_k = index - column * tile_k;
        int global_column = column_base + column;
        int global_k = k_base + local_k;
        float value = 0.0f;
        if (global_column < n && global_k < k) {
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
          value = (static_cast<float>(code) - zero) *
                  scales[global_column * group_count + group];
        }
        weight_tile[index] = value;
      }
      __syncthreads();
      int output_index = threadIdx.x;
      if (output_index < tile_m * tile_n) {
        int row = output_index / tile_n;
        int column = output_index - row * tile_n;
#pragma unroll
        for (int local_k = 0; local_k < tile_k; ++local_k) {
          accumulator += load_value<T>(activation_tile[row * tile_k + local_k]) *
                         weight_tile[column * tile_k + local_k];
        }
      }
      __syncthreads();
    }

    int output_index = threadIdx.x;
    if (output_index < tile_m * tile_n) {
      int row = output_index / tile_n;
      int column = output_index - row * tile_n;
      int global_row = row_base + row;
      int global_column = column_base + column;
      if (global_row < m && global_column < n) {
        float value = accumulator + (has_bias ? bias[global_column] : 0.0f);
        store_value<T>(&output[global_row * n + global_column], value);
      }
    }
    __syncthreads();
  }
}

template <typename T>
static int run_w4a16(
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
    void* stream,
    bool persistent) {
  if (!activation || !qweight || !scales || !output || m <= 0 || n <= 0 || k <= 0 ||
      padded_k < k || group_size <= 0) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
  if (m == 1) {
    w4a16_m1_gemv_kernel<T><<<n, 256, 0, cuda_stream>>>(
        static_cast<T const*>(activation),
        static_cast<unsigned char const*>(qweight),
        static_cast<float const*>(scales),
        static_cast<float const*>(zero_points),
        static_cast<float const*>(bias),
        static_cast<T*>(output),
        n,
        k,
        padded_k,
        group_size,
        signed_nibble,
        has_zero_points,
        has_bias);
  } else if (m <= 8) {
    constexpr int small_m_threads = 128;
    int blocks = (n + small_m_threads - 1) / small_m_threads;
    w4a16_small_m_kernel<T><<<blocks, small_m_threads, 0, cuda_stream>>>(
        static_cast<T const*>(activation),
        static_cast<unsigned char const*>(qweight),
        static_cast<float const*>(scales),
        static_cast<float const*>(zero_points),
        static_cast<float const*>(bias),
        static_cast<T*>(output),
        m,
        n,
        k,
        padded_k,
        group_size,
        signed_nibble,
        has_zero_points,
        has_bias);
  } else {
    constexpr int tile_m = 8;
    constexpr int tile_n = 16;
    constexpr int tile_k = 32;
    constexpr int threads = tile_m * tile_n;
    dim3 blocks((n + tile_n - 1) / tile_n, (m + tile_m - 1) / tile_m, 1);
    int shared_bytes = tile_m * tile_k * static_cast<int>(sizeof(T)) +
                       tile_n * tile_k * static_cast<int>(sizeof(float));
    if (persistent) {
      int multiprocessors = 0;
      int device = 0;
      cudaError_t attribute_status = cudaGetDevice(&device);
      if (attribute_status == cudaSuccess) {
        attribute_status = cudaDeviceGetAttribute(
            &multiprocessors, cudaDevAttrMultiProcessorCount, device);
      }
      if (attribute_status != cudaSuccess || multiprocessors <= 0) {
        return static_cast<int>(attribute_status);
      }
      int total_tiles = (n + tile_n - 1) / tile_n * (m + tile_m - 1) / tile_m;
      int persistent_blocks = total_tiles < multiprocessors ? total_tiles : multiprocessors;
      blocks = dim3(persistent_blocks, 1, 1);
    }
    w4a16_tiled_kernel<T><<<blocks, threads, shared_bytes, cuda_stream>>>(
        static_cast<T const*>(activation),
        static_cast<unsigned char const*>(qweight),
        static_cast<float const*>(scales),
        static_cast<float const*>(zero_points),
        static_cast<float const*>(bias),
        static_cast<T*>(output),
        m,
        n,
        k,
        padded_k,
        group_size,
        signed_nibble,
        has_zero_points,
        has_bias,
        persistent ? 1 : 0);
  }
  return static_cast<int>(cudaGetLastError());
}

template <typename T>
static cudaError_t query_w4a16_resources(
    int variant,
    int block_threads,
    int dynamic_shared_bytes,
    int* registers_per_thread,
    int* static_shared_bytes,
    int* max_active_blocks) {
  cudaFuncAttributes attributes{};
  cudaError_t status = cudaErrorInvalidValue;
  if (variant == 0) {
    status = cudaFuncGetAttributes(&attributes, w4a16_m1_gemv_kernel<T>);
  } else if (variant == 1) {
    status = cudaFuncGetAttributes(&attributes, w4a16_small_m_kernel<T>);
  } else if (variant == 2) {
    status = cudaFuncGetAttributes(&attributes, w4a16_tiled_kernel<T>);
  }
  if (status != cudaSuccess) {
    return status;
  }
  if (variant == 0) {
    status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        max_active_blocks,
        w4a16_m1_gemv_kernel<T>,
        block_threads,
        static_cast<size_t>(dynamic_shared_bytes));
  } else if (variant == 1) {
    status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        max_active_blocks,
        w4a16_small_m_kernel<T>,
        block_threads,
        static_cast<size_t>(dynamic_shared_bytes));
  } else {
    status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        max_active_blocks,
        w4a16_tiled_kernel<T>,
        block_threads,
        static_cast<size_t>(dynamic_shared_bytes));
  }
  if (status != cudaSuccess) {
    return status;
  }
  *registers_per_thread = attributes.numRegs;
  *static_shared_bytes = static_cast<int>(attributes.sharedSizeBytes);
  return cudaSuccess;
}

extern "C" int xqt_w4a16_sm89_fp16_run(
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
  return run_w4a16<half>(activation, qweight, scales, zero_points, bias, output, m, n, k,
                         padded_k, group_size, signed_nibble, has_zero_points, has_bias,
                         stream, false);
}

extern "C" int xqt_w4a16_sm89_bf16_run(
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
  return run_w4a16<__nv_bfloat16>(activation, qweight, scales, zero_points, bias, output, m, n,
                                  k, padded_k, group_size, signed_nibble, has_zero_points,
                                  has_bias, stream, false);
}

extern "C" int xqt_w4a16_sm89_fp16_run_persistent(
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
  return run_w4a16<half>(activation, qweight, scales, zero_points, bias, output, m, n, k,
                         padded_k, group_size, signed_nibble, has_zero_points, has_bias,
                         stream, true);
}

extern "C" int xqt_w4a16_sm89_bf16_run_persistent(
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
  return run_w4a16<__nv_bfloat16>(activation, qweight, scales, zero_points, bias, output, m, n,
                                  k, padded_k, group_size, signed_nibble, has_zero_points,
                                  has_bias, stream, true);
}

extern "C" const char* xqt_w4a16_sm89_version() {
  return "sm89-w4a16-dequant-fallback-v4 m1-gemv small-m-8 tile-8x16x32 groupwise-nibble";
}

// Query compile-time resource attributes and device-specific occupancy for
// one shape variant.  This is intentionally separate from the GEMM launch ABI
// so profiling tools and benchmark reports can inspect a loaded artifact
// without enqueueing a fake operation.
extern "C" int xqt_w4a16_sm89_resource_query(
    int variant,
    int block_threads,
    int dynamic_shared_bytes,
    int* registers_per_thread,
    int* static_shared_bytes,
    int* max_active_blocks) {
  if (variant < 0 || variant > 2 || block_threads <= 0 || dynamic_shared_bytes < 0 ||
      !registers_per_thread || !static_shared_bytes || !max_active_blocks) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  return static_cast<int>(query_w4a16_resources<half>(
      variant,
      block_threads,
      dynamic_shared_bytes,
      registers_per_thread,
      static_shared_bytes,
      max_active_blocks));
}
