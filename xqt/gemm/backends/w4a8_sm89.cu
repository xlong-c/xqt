// SM89 native W4A8 GEMM.
//
// The packed weight remains canonical [N, ceil(padded_K / 2)] uint8 storage.
// Each warp decodes the signed INT4 nibble for the current K tile directly into
// an INT8/FP8 MMA operand.  Group scales are applied after each K group, inside
// the main loop; this is deliberately different from a dequantize-then-dense
// fallback.  The artifact has separate INT8 and FP8 entry points and no
// architecture dispatch for SM90/SM100.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

#include "cutlass/arch/arch.h"
#include "cutlass/arch/mma.h"
#include "cutlass/float8.h"
#include "cutlass/gemm/gemm.h"

namespace {

constexpr int kRows = 16;
constexpr int kCols = 8;
constexpr int kKTile = 32;
// The row xor reaches byte offsets 32..63 for row classes 2 and 3.  Keep the
// same fragment layout as the SM89 INT8 MMA artifact used elsewhere in XQT.
constexpr int kALd = 64;
constexpr int kBLd = 32;

__device__ __forceinline__ int a_offset(int row, int col) {
  return row * kALd + (col ^ ((row & 3) << 4));
}

__device__ __forceinline__ int decode_signed_nibble(
    uint8_t packed, int high_nibble) {
  int code = high_nibble ? static_cast<int>((packed >> 4) & 0x0f)
                         : static_cast<int>(packed & 0x0f);
  return code >= 8 ? code - 16 : code;
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

template <typename Output>
__device__ __forceinline__ Output cast_output(float value);

template <>
__device__ __forceinline__ half cast_output<half>(float value) {
  return __float2half_rn(value);
}

template <>
__device__ __forceinline__ __nv_bfloat16 cast_output<__nv_bfloat16>(float value) {
  return __float2bfloat16_rn(value);
}

template <>
__device__ __forceinline__ float cast_output<float>(float value) {
  return value;
}

template <typename Output>
__device__ __forceinline__ void run_int8_tile(
    const int8_t* __restrict__ activation,
    const uint8_t* __restrict__ qweight,
    const float* __restrict__ weight_scales,
    const float* __restrict__ activation_scales,
    const float* __restrict__ bias,
    Output* __restrict__ output,
    int m,
    int n,
    int padded_k,
    int group_size,
    int activation_mode,
    int has_bias) {
  __shared__ __align__(16) int8_t shared_a[kRows * kALd];
  __shared__ __align__(16) int8_t shared_b[kCols * kBLd];
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int row_base = static_cast<int>(blockIdx.y) * kRows;
  const int column_base = static_cast<int>(blockIdx.x) * kCols;
  const int packed_stride = padded_k / 2;
  const int groups = padded_k / group_size;
  const int k_tiles_per_group = group_size / kKTile;
  const int quad = lane >> 2;
  const int lane_in_quad = lane & 3;
  float totals[4] = {0.0f, 0.0f, 0.0f, 0.0f};

  for (int group = 0; group < groups; ++group) {
    int accumulators[4] = {0, 0, 0, 0};
    const int group_k_base = group * group_size;
    for (int tile = 0; tile < k_tiles_per_group; ++tile) {
      const int k_base = group_k_base + tile * kKTile;
      for (int index = lane; index < kRows * kKTile; index += 32) {
        const int row = index / kKTile;
        const int local_k = index - row * kKTile;
        const int global_row = row_base + row;
        const int global_k = k_base + local_k;
        int8_t value = 0;
        if (global_row < m && global_k < padded_k) {
          value = activation[global_row * padded_k + global_k];
        }
        shared_a[a_offset(row, local_k)] = value;
      }
      for (int index = lane; index < kCols * kKTile; index += 32) {
        const int column = index / kKTile;
        const int local_k = index - column * kKTile;
        const int global_column = column_base + column;
        const int global_k = k_base + local_k;
        int8_t value = 0;
        if (global_column < n && global_k < padded_k) {
          const uint8_t packed =
              qweight[global_column * packed_stride + global_k / 2];
          value = static_cast<int8_t>(decode_signed_nibble(packed, global_k & 1));
        }
        shared_b[column * kBLd + local_k] = value;
      }
      __syncthreads();
      uint32_t a0, a1, a2, a3, b0, b1;
      load_a_fragment(a0, a1, a2, a3, shared_a, lane);
      load_b_fragment(b0, b1, shared_b, lane);
      mma_s8s8s32_m16n8k32(
          accumulators[0], accumulators[1], accumulators[2], accumulators[3],
          a0, a1, a2, a3, b0, b1);
      __syncthreads();
    }
    const float activation_scale0 =
        activation_mode == 0
            ? activation_scales[0]
            : activation_scales[row_base + quad];
    const float activation_scale1 =
        activation_mode == 0
            ? activation_scales[0]
            : activation_scales[row_base + quad + 8];
    const float weight_scale0 =
        weight_scales[(column_base + lane_in_quad * 2) * groups + group];
    const float weight_scale1 =
        weight_scales[(column_base + lane_in_quad * 2 + 1) * groups + group];
    totals[0] += static_cast<float>(accumulators[0]) * activation_scale0 * weight_scale0;
    totals[1] += static_cast<float>(accumulators[1]) * activation_scale0 * weight_scale1;
    totals[2] += static_cast<float>(accumulators[2]) * activation_scale1 * weight_scale0;
    totals[3] += static_cast<float>(accumulators[3]) * activation_scale1 * weight_scale1;
  }

  for (int row = 0; row < 2; ++row) {
    for (int col = 0; col < 2; ++col) {
      const int accumulator_index = row * 2 + col;
      const int global_row = row_base + quad + row * 8;
      const int global_column = column_base + lane_in_quad * 2 + col;
      if (global_row >= m || global_column >= n) {
        continue;
      }
      float value = totals[accumulator_index];
      if (has_bias) {
        value += bias[global_column];
      }
      output[global_row * n + global_column] = cast_output<Output>(value);
    }
  }
}

template <typename Output>
__global__ void w4a8_int8_mma_kernel(
    const int8_t* activation,
    const uint8_t* qweight,
    const float* weight_scales,
    const float* activation_scales,
    const float* bias,
    Output* output,
    int m,
    int n,
    int padded_k,
    int group_size,
    int activation_mode,
    int has_bias) {
  run_int8_tile<Output>(
      activation, qweight, weight_scales, activation_scales, bias, output,
      m, n, padded_k, group_size, activation_mode, has_bias);
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

template <typename ElementF8, typename Output>
__device__ __forceinline__ void run_fp8_tile(
    const ElementF8* __restrict__ activation,
    const uint8_t* __restrict__ qweight,
    const float* __restrict__ weight_scales,
    const float* __restrict__ activation_scales,
    const float* __restrict__ bias,
    Output* __restrict__ output,
    int m,
    int n,
    int padded_k,
    int group_size,
    int activation_mode,
    int has_bias) {
  __shared__ __align__(16) ElementF8 shared_b[kCols * kBLd];
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int row_base = static_cast<int>(blockIdx.y) * kRows;
  const int column_base = static_cast<int>(blockIdx.x) * kCols;
  const int packed_stride = padded_k / 2;
  const int groups = padded_k / group_size;
  const int k_tiles_per_group = group_size / kKTile;
  const int quad = lane >> 2;
  const int lane_in_quad = lane & 3;
  float totals[4] = {0.0f, 0.0f, 0.0f, 0.0f};

  for (int group = 0; group < groups; ++group) {
    typename Fp8WarpMma<ElementF8>::FragmentC accumulators;
    accumulators.clear();
    const int group_k_base = group * group_size;
    for (int tile = 0; tile < k_tiles_per_group; ++tile) {
      const int k_base = group_k_base + tile * kKTile;
      for (int index = lane; index < kCols * kKTile; index += 32) {
        const int column = index / kKTile;
        const int local_k = index - column * kKTile;
        const int global_column = column_base + column;
        const int global_k = k_base + local_k;
        ElementF8 value = ElementF8::from_float(0.0f);
        if (global_column < n && global_k < padded_k) {
          const uint8_t packed =
              qweight[global_column * packed_stride + global_k / 2];
          const int code = decode_signed_nibble(packed, global_k & 1);
          value = ElementF8::from_float(static_cast<float>(code));
        }
        shared_b[column * kBLd + local_k] = value;
      }
      __syncthreads();

      const int k_lane = k_base + lane_in_quad * 4;
      const ElementF8* a_row0 = activation + (row_base + quad) * padded_k;
      const ElementF8* a_row1 = activation + (row_base + quad + 8) * padded_k;
      const ElementF8* b_row = shared_b + quad * kBLd;
      typename Fp8WarpMma<ElementF8>::FragmentA fragment_a;
      uint32_t* a_words = reinterpret_cast<uint32_t*>(&fragment_a);
      a_words[0] = *reinterpret_cast<const uint32_t*>(a_row0 + k_lane);
      a_words[1] = *reinterpret_cast<const uint32_t*>(a_row1 + k_lane);
      a_words[2] = *reinterpret_cast<const uint32_t*>(a_row0 + k_lane + 16);
      a_words[3] = *reinterpret_cast<const uint32_t*>(a_row1 + k_lane + 16);
      typename Fp8WarpMma<ElementF8>::FragmentB fragment_b;
      uint32_t* b_words = reinterpret_cast<uint32_t*>(&fragment_b);
      b_words[0] = *reinterpret_cast<const uint32_t*>(b_row + lane_in_quad * 4);
      b_words[1] = *reinterpret_cast<const uint32_t*>(b_row + lane_in_quad * 4 + 16);
      Fp8WarpMma<ElementF8> mma;
      mma(accumulators, fragment_a, fragment_b, accumulators);
      __syncthreads();
    }
    const float activation_scale0 =
        activation_mode == 0
            ? activation_scales[0]
            : activation_mode == 1
                  ? activation_scales[row_base + quad]
                  : activation_scales[(row_base + quad) * groups + group];
    const float activation_scale1 =
        activation_mode == 0
            ? activation_scales[0]
            : activation_mode == 1
                  ? activation_scales[row_base + quad + 8]
                  : activation_scales[(row_base + quad + 8) * groups + group];
    const float weight_scale0 =
        weight_scales[(column_base + lane_in_quad * 2) * groups + group];
    const float weight_scale1 =
        weight_scales[(column_base + lane_in_quad * 2 + 1) * groups + group];
    totals[0] += accumulators[0] * activation_scale0 * weight_scale0;
    totals[1] += accumulators[1] * activation_scale0 * weight_scale1;
    totals[2] += accumulators[2] * activation_scale1 * weight_scale0;
    totals[3] += accumulators[3] * activation_scale1 * weight_scale1;
  }

  for (int row = 0; row < 2; ++row) {
    for (int col = 0; col < 2; ++col) {
      const int accumulator_index = row * 2 + col;
      const int global_row = row_base + quad + row * 8;
      const int global_column = column_base + lane_in_quad * 2 + col;
      if (global_row >= m || global_column >= n) {
        continue;
      }
      float value = totals[accumulator_index];
      if (has_bias) {
        value += bias[global_column];
      }
      output[global_row * n + global_column] = cast_output<Output>(value);
    }
  }
}

template <typename ElementF8, typename Output>
__global__ void w4a8_fp8_mma_kernel(
    const ElementF8* activation,
    const uint8_t* qweight,
    const float* weight_scales,
    const float* activation_scales,
    const float* bias,
    Output* output,
    int m,
    int n,
    int padded_k,
    int group_size,
    int activation_mode,
    int has_bias) {
  run_fp8_tile<ElementF8, Output>(
      activation, qweight, weight_scales, activation_scales, bias, output,
      m, n, padded_k, group_size, activation_mode, has_bias);
}

template <typename Output>
static int launch_int8(
    const void* activation,
    const void* qweight,
    const float* weight_scales,
    const float* activation_scales,
    const float* bias,
    void* output,
    int m,
    int n,
    int padded_k,
    int group_size,
    int activation_mode,
    int has_bias,
    void* stream_ptr) {
  if (!activation || !qweight || !weight_scales || !activation_scales || !output ||
      m <= 0 || n <= 0 || padded_k <= 0 || padded_k % 32 != 0 ||
      group_size <= 0 || group_size % 32 != 0 || padded_k % group_size != 0 ||
      activation_mode < 0 || activation_mode > 1 || (has_bias && !bias) ||
      m % 16 != 0 || n % 8 != 0) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  dim3 grid(static_cast<unsigned>(n / kCols), static_cast<unsigned>(m / kRows), 1);
  w4a8_int8_mma_kernel<Output><<<grid, 32, 0, stream>>>(
      static_cast<const int8_t*>(activation),
      static_cast<const uint8_t*>(qweight), weight_scales, activation_scales,
      bias, static_cast<Output*>(output), m, n, padded_k, group_size,
      activation_mode, has_bias);
  return static_cast<int>(cudaGetLastError());
}

template <typename ElementF8, typename Output>
static int launch_fp8(
    const void* activation,
    const void* qweight,
    const float* weight_scales,
    const float* activation_scales,
    const float* bias,
    void* output,
    int m,
    int n,
    int padded_k,
    int group_size,
    int activation_mode,
    int has_bias,
    void* stream_ptr) {
  if (!activation || !qweight || !weight_scales || !activation_scales || !output ||
      m <= 0 || n <= 0 || padded_k <= 0 || padded_k % 32 != 0 ||
      group_size <= 0 || group_size % 32 != 0 || padded_k % group_size != 0 ||
      activation_mode < 0 || activation_mode > 2 || (has_bias && !bias) ||
      m % 16 != 0 || n % 8 != 0) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  dim3 grid(static_cast<unsigned>(n / kCols), static_cast<unsigned>(m / kRows), 1);
  w4a8_fp8_mma_kernel<ElementF8, Output><<<grid, 32, 0, stream>>>(
      static_cast<const ElementF8*>(activation),
      static_cast<const uint8_t*>(qweight), weight_scales, activation_scales,
      bias, static_cast<Output*>(output), m, n, padded_k, group_size,
      activation_mode, has_bias);
  return static_cast<int>(cudaGetLastError());
}

}  // namespace

extern "C" int xqt_w4a8_int8_sm89_fp16_run(
    const void* activation, const void* qweight, const float* weight_scales,
    const float* activation_scales, const float* bias, void* output, int m,
    int n, int padded_k, int group_size, int activation_mode, int has_bias,
    void* stream) {
  return launch_int8<half>(activation, qweight, weight_scales, activation_scales,
                           bias, output, m, n, padded_k, group_size,
                           activation_mode, has_bias, stream);
}

extern "C" int xqt_w4a8_int8_sm89_bf16_run(
    const void* activation, const void* qweight, const float* weight_scales,
    const float* activation_scales, const float* bias, void* output, int m,
    int n, int padded_k, int group_size, int activation_mode, int has_bias,
    void* stream) {
  return launch_int8<__nv_bfloat16>(
      activation, qweight, weight_scales, activation_scales, bias, output, m,
      n, padded_k, group_size, activation_mode, has_bias, stream);
}

extern "C" int xqt_w4a8_int8_sm89_fp32_run(
    const void* activation, const void* qweight, const float* weight_scales,
    const float* activation_scales, const float* bias, void* output, int m,
    int n, int padded_k, int group_size, int activation_mode, int has_bias,
    void* stream) {
  return launch_int8<float>(activation, qweight, weight_scales, activation_scales,
                            bias, output, m, n, padded_k, group_size,
                            activation_mode, has_bias, stream);
}

#define XQT_W4A8_FP8_EXPORT(FORMAT, TYPE, NAME)                                      \
  extern "C" int NAME(                                                               \
      const void* activation, const void* qweight, const float* weight_scales,       \
      const float* activation_scales, const float* bias, void* output, int m,         \
      int n, int padded_k, int group_size, int activation_mode, int has_bias,         \
      void* stream) {                                                                 \
    return launch_fp8<TYPE, FORMAT>(                                                  \
        activation, qweight, weight_scales, activation_scales, bias, output, m, n,   \
        padded_k, group_size, activation_mode, has_bias, stream);                     \
  }

XQT_W4A8_FP8_EXPORT(half, cutlass::float_e4m3_t, xqt_w4a8_fp8_e4m3_sm89_fp16_run)
XQT_W4A8_FP8_EXPORT(__nv_bfloat16, cutlass::float_e4m3_t, xqt_w4a8_fp8_e4m3_sm89_bf16_run)
XQT_W4A8_FP8_EXPORT(float, cutlass::float_e4m3_t, xqt_w4a8_fp8_e4m3_sm89_fp32_run)
XQT_W4A8_FP8_EXPORT(half, cutlass::float_e5m2_t, xqt_w4a8_fp8_e5m2_sm89_fp16_run)
XQT_W4A8_FP8_EXPORT(__nv_bfloat16, cutlass::float_e5m2_t, xqt_w4a8_fp8_e5m2_sm89_bf16_run)
XQT_W4A8_FP8_EXPORT(float, cutlass::float_e5m2_t, xqt_w4a8_fp8_e5m2_sm89_fp32_run)

#undef XQT_W4A8_FP8_EXPORT

extern "C" const char* xqt_w4a8_sm89_version() {
  return "sm89-w4a8-native-v1 int8=mma.m16n8k32 fp8=mma.m16n8k32 group-scale-in-mainloop";
}
