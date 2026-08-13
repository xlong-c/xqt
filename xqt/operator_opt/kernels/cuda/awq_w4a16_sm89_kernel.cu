/*
 * The decode strategy follows the public AWQ weight-only GEMV design used by
 * FasterTransformer/TensorRT-LLM derivatives: four output channels are
 * interleaved, each CTA reduces K for eight output channels, and INT4 values
 * are converted in registers before half2/bfloat162 products. The low-bit
 * conversion sequence is adapted from NVIDIA FasterTransformer code released
 * under Apache-2.0. The surrounding packing, launch ABI and XQT integration
 * are implemented for this repository.
 */

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <type_traits>

namespace {

constexpr int kPackFactor = 8;
constexpr int kWarpSize = 32;
constexpr int kMemoryAccessBits = 128;
constexpr int kBlockSize = 256;
constexpr int kOutputsPerBlock = 2;
constexpr int kOutputInterleave = 4;
constexpr int kGroupSize = 64;

template <typename T>
struct Pair;

template <>
struct Pair<half> {
    using type = half2;
};

template <>
struct Pair<__nv_bfloat16> {
    using type = __nv_bfloat162;
};

template <typename T>
__device__ __forceinline__ typename Pair<T>::type broadcast_pair(T value);

template <>
__device__ __forceinline__ half2 broadcast_pair(half value) {
    return __half2half2(value);
}

template <>
__device__ __forceinline__ __nv_bfloat162 broadcast_pair(__nv_bfloat16 value) {
    return __bfloat162bfloat162(value);
}

template <typename PairT>
__device__ __forceinline__ float2 multiply_to_float2(PairT left, PairT right);

template <>
__device__ __forceinline__ float2 multiply_to_float2(half2 left, half2 right) {
    return __half22float2(__hmul2(left, right));
}

template <>
__device__ __forceinline__ float2 multiply_to_float2(
    __nv_bfloat162 left,
    __nv_bfloat162 right) {
    return __bfloat1622float2(__hmul2(left, right));
}

template <typename PairT>
__device__ __forceinline__ PairT fused_multiply_add(
    PairT left,
    PairT right,
    PairT addend);

template <>
__device__ __forceinline__ half2 fused_multiply_add(
    half2 left,
    half2 right,
    half2 addend) {
    return __hfma2(left, right, addend);
}

template <>
__device__ __forceinline__ __nv_bfloat162 fused_multiply_add(
    __nv_bfloat162 left,
    __nv_bfloat162 right,
    __nv_bfloat162 addend) {
    return __hfma2(left, right, addend);
}

template <typename T>
__device__ __forceinline__ T from_float(float value);

template <>
__device__ __forceinline__ half from_float(float value) {
    return __float2half_rn(value);
}

template <>
__device__ __forceinline__ __nv_bfloat16 from_float(float value) {
    return __float2bfloat16_rn(value);
}

template <typename T>
__device__ __forceinline__ T add_in_output_dtype(T left, T right);

template <>
__device__ __forceinline__ half add_in_output_dtype(half left, half right) {
    return __hadd(left, right);
}

template <>
__device__ __forceinline__ __nv_bfloat16 add_in_output_dtype(
    __nv_bfloat16 left,
    __nv_bfloat16 right) {
    return __hadd(left, right);
}

__device__ __forceinline__ void dequantize_u4_to_half(
    half2 const& source,
    uint4* result) {
    uint32_t* output = reinterpret_cast<uint32_t*>(result);
    uint32_t const packed = reinterpret_cast<uint32_t const&>(source);
    constexpr uint32_t lut = (0xf0 & 0xcc) | 0xaa;
    constexpr uint32_t bottom_mask = 0x000f000f;
    constexpr uint32_t top_mask = 0x00f000f0;
    constexpr uint32_t magic = 0x64006400;
    uint32_t const top = packed >> 8;
    asm volatile(
        "lop3.b32 %0, %1, %2, %3, %4;\n"
        : "=r"(output[0])
        : "r"(packed), "n"(bottom_mask), "n"(magic), "n"(lut));
    asm volatile(
        "lop3.b32 %0, %1, %2, %3, %4;\n"
        : "=r"(output[1])
        : "r"(packed), "n"(top_mask), "n"(magic), "n"(lut));
    asm volatile(
        "lop3.b32 %0, %1, %2, %3, %4;\n"
        : "=r"(output[2])
        : "r"(top), "n"(bottom_mask), "n"(magic), "n"(lut));
    asm volatile(
        "lop3.b32 %0, %1, %2, %3, %4;\n"
        : "=r"(output[3])
        : "r"(top), "n"(top_mask), "n"(magic), "n"(lut));
    constexpr uint32_t top_magic = 0x64006400;
    constexpr uint32_t one_sixteenth = 0x2c002c00;
    constexpr uint32_t negative_64 = 0xd400d400;
    asm volatile(
        "sub.f16x2 %0, %1, %2;\n"
        : "=r"(output[0])
        : "r"(output[0]), "r"(top_magic));
    asm volatile(
        "fma.rn.f16x2 %0, %1, %2, %3;\n"
        : "=r"(output[1])
        : "r"(output[1]), "r"(one_sixteenth), "r"(negative_64));
    asm volatile(
        "sub.f16x2 %0, %1, %2;\n"
        : "=r"(output[2])
        : "r"(output[2]), "r"(top_magic));
    asm volatile(
        "fma.rn.f16x2 %0, %1, %2, %3;\n"
        : "=r"(output[3])
        : "r"(output[3]), "r"(one_sixteenth), "r"(negative_64));
}

__device__ __forceinline__ void dequantize_u4_to_bfloat16(
    __nv_bfloat162 const& source,
    uint4* result) {
    uint32_t* output = reinterpret_cast<uint32_t*>(result);
    uint32_t const packed = reinterpret_cast<uint32_t const&>(source);
    constexpr uint32_t lut = (0xf0 & 0xcc) | 0xaa;
    constexpr uint32_t mask = 0x000f000f;
    constexpr uint32_t magic = 0x43004300;
    asm volatile(
        "lop3.b32 %0, %1, %2, %3, %4;\n"
        : "=r"(output[0])
        : "r"(packed), "n"(mask), "n"(magic), "n"(lut));
    asm volatile(
        "lop3.b32 %0, %1, %2, %3, %4;\n"
        : "=r"(output[1])
        : "r"(packed >> 4), "n"(mask), "n"(magic), "n"(lut));
    asm volatile(
        "lop3.b32 %0, %1, %2, %3, %4;\n"
        : "=r"(output[2])
        : "r"(packed >> 8), "n"(mask), "n"(magic), "n"(lut));
    asm volatile(
        "lop3.b32 %0, %1, %2, %3, %4;\n"
        : "=r"(output[3])
        : "r"(packed >> 12), "n"(mask), "n"(magic), "n"(lut));
    constexpr uint32_t bias = 0xc300c300;
    constexpr uint32_t one = 0x3f803f80;
    asm volatile(
        "fma.rn.bf16x2 %0, %1, %2, %3;\n"
        : "=r"(output[0])
        : "r"(output[0]), "r"(one), "r"(bias));
    asm volatile(
        "fma.rn.bf16x2 %0, %1, %2, %3;\n"
        : "=r"(output[1])
        : "r"(output[1]), "r"(one), "r"(bias));
    asm volatile(
        "fma.rn.bf16x2 %0, %1, %2, %3;\n"
        : "=r"(output[2])
        : "r"(output[2]), "r"(one), "r"(bias));
    asm volatile(
        "fma.rn.bf16x2 %0, %1, %2, %3;\n"
        : "=r"(output[3])
        : "r"(output[3]), "r"(one), "r"(bias));
}

template <typename T>
__device__ __forceinline__ void dequantize_u4(
    typename Pair<T>::type const& source,
    uint4* result);

template <>
__device__ __forceinline__ void dequantize_u4<half>(
    half2 const& source,
    uint4* result) {
    dequantize_u4_to_half(source, result);
}

template <>
__device__ __forceinline__ void dequantize_u4<__nv_bfloat16>(
    __nv_bfloat162 const& source,
    uint4* result) {
    dequantize_u4_to_bfloat16(source, result);
}

template <int Count>
__device__ __forceinline__ void warp_reduce(
    float (&partial)[Count],
    float (*shared)[Count * kOutputInterleave]) {
#pragma unroll
    for (int index = 0; index < Count; ++index) {
        partial[index] += __shfl_xor_sync(0xffffffff, partial[index], 16);
        partial[index] += __shfl_xor_sync(0xffffffff, partial[index], 8);
        partial[index] += __shfl_xor_sync(0xffffffff, partial[index], 1);
    }
    __syncthreads();
    int const warp = threadIdx.x / kWarpSize;
    int const lane = threadIdx.x % kWarpSize;
    if (lane == 0 || lane == 2 || lane == 4 || lane == 6) {
#pragma unroll
        for (int index = 0; index < Count; ++index) {
            shared[warp][index * kOutputInterleave + lane / 2] = partial[index];
        }
    }
    __syncthreads();
}

template <typename T, int Batch, bool HasBias>
__global__ void awq_decode_kernel(
    const T* inputs,
    const uint32_t* weight,
    const T* scales,
    const T* scaled_zeros,
    const T* bias,
    T* outputs,
    int input_channels,
    int output_channels) {
    using PairT = typename Pair<T>::type;
    constexpr int elements_per_thread = kMemoryAccessBits / 4;
    constexpr int threads_per_k_tile = kGroupSize / elements_per_thread;
    constexpr int output_count = kOutputsPerBlock * Batch;

    alignas(16) T local_inputs[elements_per_thread];
    alignas(16) uint32_t local_qweights[kMemoryAccessBits / 32];
    alignas(16) T half_weight_buffer[elements_per_thread];
    alignas(16) T dequantized_weight[elements_per_thread * kOutputsPerBlock];
    alignas(16) T local_scale[kOutputsPerBlock];
    alignas(16) T local_scaled_zero[kOutputsPerBlock];
    float partial[output_count]{};
    __shared__ float reduction[kBlockSize / kWarpSize * 2]
                              [output_count * kOutputInterleave];

    int const block_output_offset =
        blockIdx.x * kOutputsPerBlock * kOutputInterleave;
    int const thread_output_offset =
        (threadIdx.x / threads_per_k_tile) % kOutputInterleave;
    int const input_k_offset =
        threadIdx.x / (threads_per_k_tile * kOutputInterleave) * kGroupSize +
        (threadIdx.x % threads_per_k_tile) * elements_per_thread;
    int const group_offset = input_k_offset / kGroupSize;
    const uint32_t* block_weight =
        weight + block_output_offset * input_channels / kPackFactor;
    const T* scale_ptr =
        scales + block_output_offset + thread_output_offset +
        group_offset * output_channels;
    const T* zero_ptr =
        scaled_zeros + block_output_offset + thread_output_offset +
        group_offset * output_channels;
    const T* input_ptr = inputs + input_k_offset;
    int const input_forward =
        kBlockSize * elements_per_thread / kOutputInterleave;
    int const scale_forward = input_forward / kGroupSize * output_channels;

    for (
        int k_index = threadIdx.x * elements_per_thread;
        k_index < input_channels * kOutputInterleave;
        k_index += kBlockSize * elements_per_thread) {
#pragma unroll
        for (int output = 0; output < kOutputsPerBlock; ++output) {
            *reinterpret_cast<float4*>(local_qweights) =
                *reinterpret_cast<const float4*>(
                    block_weight +
                    (output * kOutputInterleave * input_channels + k_index) /
                        kPackFactor);
            local_scale[output] = *(scale_ptr + output * kOutputInterleave);
            local_scaled_zero[output] = *(zero_ptr + output * kOutputInterleave);
#pragma unroll
            for (int word = 0; word < kMemoryAccessBits / 32; ++word) {
                dequantize_u4<T>(
                    *reinterpret_cast<PairT*>(local_qweights + word),
                    reinterpret_cast<uint4*>(
                        half_weight_buffer + word * kPackFactor));
            }
#pragma unroll
            for (int continuous = 0; continuous < 4; ++continuous) {
#pragma unroll
                for (int strided = 0; strided < 4; ++strided) {
                    PairT value = *reinterpret_cast<PairT*>(
                        half_weight_buffer +
                        (continuous + strided * 4) * 2);
                    value = fused_multiply_add(
                        value,
                        broadcast_pair(local_scale[output]),
                        broadcast_pair(local_scaled_zero[output]));
                    dequantized_weight[
                        ((continuous * 4 + strided) * 2 + 0) *
                            kOutputsPerBlock +
                        output] = value.x;
                    dequantized_weight[
                        ((continuous * 4 + strided) * 2 + 1) *
                            kOutputsPerBlock +
                        output] = value.y;
                }
            }
        }
#pragma unroll
        for (int batch = 0; batch < Batch; ++batch) {
            const T* row_input = input_ptr + batch * input_channels;
#pragma unroll
            for (int index = 0; index < elements_per_thread / 8; ++index) {
                *reinterpret_cast<float4*>(local_inputs + index * 8) =
                    *reinterpret_cast<const float4*>(row_input + index * 8);
            }
#pragma unroll
            for (int output_pair = 0;
                 output_pair < kOutputsPerBlock / 2;
                 ++output_pair) {
#pragma unroll
                for (int element = 0; element < elements_per_thread; ++element) {
                    float2 product = multiply_to_float2(
                        *reinterpret_cast<PairT*>(
                            dequantized_weight +
                            element * kOutputsPerBlock + output_pair * 2),
                        broadcast_pair(local_inputs[element]));
                    int const partial_index =
                        batch * kOutputsPerBlock + output_pair * 2;
                    partial[partial_index] += product.x;
                    partial[partial_index + 1] += product.y;
                }
            }
        }
        input_ptr += input_forward;
        scale_ptr += scale_forward;
        zero_ptr += scale_forward;
    }

    warp_reduce(partial, reduction);
    for (
        int index = threadIdx.x;
        index < output_count * kOutputInterleave;
        index += kBlockSize) {
        int const batch = index / (kOutputsPerBlock * kOutputInterleave);
        int const output = index % (kOutputsPerBlock * kOutputInterleave);
        float value = 0.0f;
#pragma unroll
        for (int warp = 0; warp < kBlockSize / kWarpSize; ++warp) {
            value += reduction[warp][index];
        }
        T result = from_float<T>(value);
        if constexpr (HasBias) {
            // Match Nunchaku AWQW4A16Linear exactly: GEMV writes the output
            // dtype first, then ``output.add_(bias)`` performs one more
            // output-dtype rounding. Adding bias to the FP32 accumulator would
            // remove that intermediate rounding and change module numerics.
            result = add_in_output_dtype(
                result,
                bias[block_output_offset + output]);
        }
        outputs[batch * output_channels + block_output_offset + output] = result;
    }
}

__device__ __forceinline__ uint32_t canonical_code(
    const uint8_t* qweight,
    int packed_columns,
    int output,
    int k_index) {
    uint8_t const packed = qweight[output * packed_columns + k_index / 2];
    return (k_index & 1) == 0 ? packed & 0x0f : (packed >> 4) & 0x0f;
}

__global__ void pack_kernel(
    const uint8_t* canonical,
    uint32_t* interleaved,
    int output_channels,
    int input_channels) {
    int64_t const output_columns = input_channels / 2;
    int64_t const count =
        static_cast<int64_t>(output_channels / 4) * output_columns;
    for (
        int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
        index < count;
        index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        int const output_group = static_cast<int>(index / output_columns);
        int const column = static_cast<int>(index % output_columns);
        int const k_block = column / 32;
        int const within_block = column % 32;
        int const output_lane = within_block / 8;
        int const within_output = within_block % 8;
        int const k_half = within_output / 4;
        int const word = within_output % 4;
        int const output = output_group * 4 + output_lane;
        int const k_base = k_block * 64 + k_half * 32;
        int const even = k_base + word * 2;
        uint32_t packed = 0;
        packed |= canonical_code(
                      canonical,
                      input_channels / 2,
                      output,
                      even)
                  << 0;
        packed |= canonical_code(
                      canonical,
                      input_channels / 2,
                      output,
                      even + 8)
                  << 4;
        packed |= canonical_code(
                      canonical,
                      input_channels / 2,
                      output,
                      even + 16)
                  << 8;
        packed |= canonical_code(
                      canonical,
                      input_channels / 2,
                      output,
                      even + 24)
                  << 12;
        packed |= canonical_code(
                      canonical,
                      input_channels / 2,
                      output,
                      even + 1)
                  << 16;
        packed |= canonical_code(
                      canonical,
                      input_channels / 2,
                      output,
                      even + 9)
                  << 20;
        packed |= canonical_code(
                      canonical,
                      input_channels / 2,
                      output,
                      even + 17)
                  << 24;
        packed |= canonical_code(
                      canonical,
                      input_channels / 2,
                      output,
                      even + 25)
                  << 28;
        interleaved[index] = packed;
    }
}

template <typename T, bool HasBias>
int launch_decode(
    const void* input,
    const void* qweight,
    const void* scales,
    const void* scaled_zeros,
    const void* bias,
    void* output,
    int m,
    int n,
    int k,
    cudaStream_t stream) {
    dim3 const blocks(n / (kOutputsPerBlock * kOutputInterleave));
    dim3 const threads(kBlockSize);
    switch (m) {
        case 1:
            awq_decode_kernel<T, 1, HasBias><<<blocks, threads, 0, stream>>>(
                static_cast<const T*>(input),
                static_cast<const uint32_t*>(qweight),
                static_cast<const T*>(scales),
                static_cast<const T*>(scaled_zeros),
                static_cast<const T*>(bias),
                static_cast<T*>(output),
                k,
                n);
            break;
        case 2:
            awq_decode_kernel<T, 2, HasBias><<<blocks, threads, 0, stream>>>(
                static_cast<const T*>(input),
                static_cast<const uint32_t*>(qweight),
                static_cast<const T*>(scales),
                static_cast<const T*>(scaled_zeros),
                static_cast<const T*>(bias),
                static_cast<T*>(output),
                k,
                n);
            break;
        case 3:
            awq_decode_kernel<T, 3, HasBias><<<blocks, threads, 0, stream>>>(
                static_cast<const T*>(input),
                static_cast<const uint32_t*>(qweight),
                static_cast<const T*>(scales),
                static_cast<const T*>(scaled_zeros),
                static_cast<const T*>(bias),
                static_cast<T*>(output),
                k,
                n);
            break;
        case 4:
            awq_decode_kernel<T, 4, HasBias><<<blocks, threads, 0, stream>>>(
                static_cast<const T*>(input),
                static_cast<const uint32_t*>(qweight),
                static_cast<const T*>(scales),
                static_cast<const T*>(scaled_zeros),
                static_cast<const T*>(bias),
                static_cast<T*>(output),
                k,
                n);
            break;
        case 5:
            awq_decode_kernel<T, 5, HasBias><<<blocks, threads, 0, stream>>>(
                static_cast<const T*>(input),
                static_cast<const uint32_t*>(qweight),
                static_cast<const T*>(scales),
                static_cast<const T*>(scaled_zeros),
                static_cast<const T*>(bias),
                static_cast<T*>(output),
                k,
                n);
            break;
        case 6:
            awq_decode_kernel<T, 6, HasBias><<<blocks, threads, 0, stream>>>(
                static_cast<const T*>(input),
                static_cast<const uint32_t*>(qweight),
                static_cast<const T*>(scales),
                static_cast<const T*>(scaled_zeros),
                static_cast<const T*>(bias),
                static_cast<T*>(output),
                k,
                n);
            break;
        case 7:
            awq_decode_kernel<T, 7, HasBias><<<blocks, threads, 0, stream>>>(
                static_cast<const T*>(input),
                static_cast<const uint32_t*>(qweight),
                static_cast<const T*>(scales),
                static_cast<const T*>(scaled_zeros),
                static_cast<const T*>(bias),
                static_cast<T*>(output),
                k,
                n);
            break;
        case 8:
            awq_decode_kernel<T, 8, HasBias><<<blocks, threads, 0, stream>>>(
                static_cast<const T*>(input),
                static_cast<const uint32_t*>(qweight),
                static_cast<const T*>(scales),
                static_cast<const T*>(scaled_zeros),
                static_cast<const T*>(bias),
                static_cast<T*>(output),
                k,
                n);
            break;
        default:
            return static_cast<int>(cudaErrorInvalidValue);
    }
    return static_cast<int>(cudaGetLastError());
}

}  // namespace

extern "C" int xqt_awq_w4a16_sm89_pack(
    const void* canonical,
    void* interleaved,
    int n,
    int k,
    cudaStream_t stream) {
    int64_t const count = static_cast<int64_t>(n / 4) * (k / 2);
    int const threads = 256;
    int const blocks = static_cast<int>((count + threads - 1) / threads);
    pack_kernel<<<blocks, threads, 0, stream>>>(
        static_cast<const uint8_t*>(canonical),
        static_cast<uint32_t*>(interleaved),
        n,
        k);
    return static_cast<int>(cudaGetLastError());
}

extern "C" int xqt_awq_w4a16_sm89_decode(
    const void* input,
    const void* qweight,
    const void* scales,
    const void* scaled_zeros,
    void* output,
    int m,
    int n,
    int k,
    int scalar_kind,
    cudaStream_t stream) {
    if (scalar_kind == 0) {
        return launch_decode<half, false>(
            input, qweight, scales, scaled_zeros, nullptr, output, m, n, k, stream);
    }
    if (scalar_kind == 1) {
        return launch_decode<__nv_bfloat16, false>(
            input, qweight, scales, scaled_zeros, nullptr, output, m, n, k, stream);
    }
    return static_cast<int>(cudaErrorInvalidValue);
}

extern "C" int xqt_awq_w4a16_sm89_decode_bias(
    const void* input,
    const void* qweight,
    const void* scales,
    const void* scaled_zeros,
    const void* bias,
    void* output,
    int m,
    int n,
    int k,
    int scalar_kind,
    cudaStream_t stream) {
    if (scalar_kind == 0) {
        return launch_decode<half, true>(
            input, qweight, scales, scaled_zeros, bias, output, m, n, k, stream);
    }
    if (scalar_kind == 1) {
        return launch_decode<__nv_bfloat16, true>(
            input, qweight, scales, scaled_zeros, bias, output, m, n, k, stream);
    }
    return static_cast<int>(cudaErrorInvalidValue);
}

extern "C" const char* xqt_awq_w4a16_sm89_version() {
    return "xqt-awq-w4a16-sm89-v2";
}
