#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <type_traits>

#include "gemm_w8a8.cuh"

namespace {

using namespace nunchaku::kernels;
using GEMM = GEMM_W8A8;
using Config = GEMMConfig_W8A8;
using Base = GEMMBase<Config>;

struct QuantizeWeightKernel {
    static constexpr int MIN_ARCH =
        std::is_same_v<Base::half_t, __nv_bfloat16> ? 800 : 750;

    __device__ void operator()(
        const Base::half_t* input,
        const Base::half_t* scales,
        Base::packed_wgt_t* output,
        int k) {
        const int lane_id = static_cast<int>(threadIdx.x) % Base::WARP_SIZE;
        const int block_n = static_cast<int>(blockIdx.x);
        const int block_k = static_cast<int>(blockIdx.y);
        const int row_base = block_n * Base::BLOCK_N;
        const int col_base = block_k * Base::WARP_K;

        __shared__ alignas(128) uint8_t scratch[Base::INSN_M * Base::INSN_K];

        for (int tile_id = 0; tile_id < Base::WARP_N_TILES; ++tile_id) {
            Base::packed_wgt_t quantized;
            GEMM::template quantize_w8a8_warp<false>(
                input + (row_base + tile_id * Base::INSN_N) * k + col_base,
                scales + row_base + tile_id * Base::INSN_N,
                k,
                quantized,
                scratch);
            std::swap(quantized.y, quantized.z);
            const int output_index =
                ((block_n * (k / Base::WARP_K) + block_k) *
                     Base::WARP_N_TILES +
                 tile_id) *
                    Base::WARP_SIZE +
                lane_id;
            store(&output[output_index], quantized);
        }
    }
};

struct QuantizeRotatedActKernel {
    static constexpr int MIN_ARCH =
        std::is_same_v<Base::half_t, __nv_bfloat16> ? 800 : 750;
    static constexpr int ROT_SIZE = 256;
    static constexpr int VALUES_PER_LANE = ROT_SIZE / Base::WARP_SIZE;
    static constexpr size_t QUANT_SCRATCH_BYTES =
        Base::INSN_M * Base::INSN_K;

    struct Arguments {
        const Base::half_t* input;
        Base::packed_act_t* output;
        Base::packed_ascale_t* scales;
        int actual_m;
        int logical_k;
        int rotated_k;
        int padded_m;
        int padded_k;
        int rot_size;
    };

    template <bool StoreOutput>
    __device__ __forceinline__ static float rotate_256_row(
        const Arguments& args,
        int global_row,
        int col_base,
        int lane_id,
        Base::half_t* output) {
        constexpr unsigned FULL_MASK = 0xffffffffU;
        float values[VALUES_PER_LANE];

#pragma unroll
        for (int index = 0; index < VALUES_PER_LANE; ++index) {
            const int local_col = lane_id * VALUES_PER_LANE + index;
            const int global_col = col_base + local_col;
            values[index] =
                global_row < args.actual_m && global_col < args.logical_k
                    ? static_cast<float>(
                          args.input[global_row * args.logical_k + global_col])
                    : 0.0F;
        }

#pragma unroll
        for (int group = 0; group < VALUES_PER_LANE; group += 4) {
            const float a = values[group];
            const float b = values[group + 1];
            const float c = values[group + 2];
            const float d = values[group + 3];
            const float sum = a + b + c + d;
            values[group] = sum - 2.0F * d;
            values[group + 1] = sum - 2.0F * c;
            values[group + 2] = sum - 2.0F * b;
            values[group + 3] = sum - 2.0F * a;
        }

#pragma unroll
        for (int d0 = 0; d0 < 4; ++d0) {
            const float low = values[d0];
            const float high = values[d0 + 4];
            const float local_sum = low + high;
            const float sum =
                local_sum + __shfl_xor_sync(FULL_MASK, local_sum, 1);
            values[d0] =
                sum - 2.0F * __shfl_xor_sync(FULL_MASK, high, 1);
            values[d0 + 4] =
                sum - 2.0F * __shfl_xor_sync(FULL_MASK, low, 1);
        }

#pragma unroll
        for (int index = 0; index < VALUES_PER_LANE; ++index) {
            const float value = values[index];
            float sum = value + __shfl_xor_sync(FULL_MASK, value, 2);
            sum += __shfl_xor_sync(FULL_MASK, sum, 4);
            values[index] =
                sum - 2.0F * __shfl_xor_sync(FULL_MASK, value, 6);
        }

#pragma unroll
        for (int index = 0; index < VALUES_PER_LANE; ++index) {
            const float value = values[index];
            float sum = value + __shfl_xor_sync(FULL_MASK, value, 8);
            sum += __shfl_xor_sync(FULL_MASK, sum, 16);
            values[index] =
                sum - 2.0F * __shfl_xor_sync(FULL_MASK, value, 24);
        }

        float local_maximum = 0.0F;
#pragma unroll
        for (int index = 0; index < VALUES_PER_LANE; ++index) {
            const int local_col = lane_id * VALUES_PER_LANE + index;
            const int global_col = col_base + local_col;
            const float value = global_col < args.rotated_k
                ? values[index] * (1.0F / 16.0F)
                : 0.0F;
            const Base::half_t converted = cuda_cast<Base::half_t>(value);
            local_maximum =
                fmaxf(local_maximum, fabsf(static_cast<float>(converted)));
            if constexpr (StoreOutput) {
                output[local_col] = converted;
            }
        }
#pragma unroll
        for (int mask = Base::WARP_SIZE / 2; mask > 0; mask /= 2) {
            local_maximum = fmaxf(
                local_maximum,
                __shfl_xor_sync(FULL_MASK, local_maximum, mask));
        }
        return local_maximum;
    }

    __device__ __forceinline__ static float direct_row_maximum(
        const Arguments& args,
        int global_row,
        int lane_id) {
        float maximum = 0.0F;
        if (global_row < args.actual_m) {
            for (int col = lane_id; col < args.rotated_k;
                 col += Base::WARP_SIZE) {
                const Base::half_t value = col < args.logical_k
                    ? args.input[global_row * args.logical_k + col]
                    : cuda_cast<Base::half_t>(0.0F);
                maximum = fmaxf(maximum, fabsf(static_cast<float>(value)));
            }
        }
#pragma unroll
        for (int mask = Base::WARP_SIZE / 2; mask > 0; mask /= 2) {
            maximum = fmaxf(
                maximum,
                __shfl_xor_sync(0xffffffffU, maximum, mask));
        }
        return maximum;
    }

    __device__ void operator()(Arguments args) {
        const int thread_id = static_cast<int>(threadIdx.x);
        const int warp_id = thread_id / Base::WARP_SIZE;
        const int lane_id = thread_id % Base::WARP_SIZE;
        const int row_base = static_cast<int>(blockIdx.x) * Base::WARP_M;
        const int block_m = static_cast<int>(blockIdx.x) / Base::NUM_WARPS;
        const int block_warp = static_cast<int>(blockIdx.x) % Base::NUM_WARPS;

        __shared__ alignas(128) Base::half_t rotated[Base::WARP_M][ROT_SIZE];
        __shared__ alignas(128) float row_maximum[Base::WARP_M];
        __shared__ alignas(128) Base::half_t row_scale[Base::WARP_M];
        __shared__ alignas(128) uint8_t
            quant_scratch[Base::NUM_WARPS][QUANT_SCRATCH_BYTES];

        for (int row = warp_id; row < Base::WARP_M;
             row += Base::NUM_WARPS) {
            const int global_row = row_base + row;
            float maximum = 0.0F;
            if (args.rot_size == ROT_SIZE) {
                for (int col_base = 0; col_base < args.rotated_k;
                     col_base += ROT_SIZE) {
                    maximum = fmaxf(
                        maximum,
                        rotate_256_row<false>(
                            args,
                            global_row,
                            col_base,
                            lane_id,
                            nullptr));
                }
            } else {
                maximum = direct_row_maximum(args, global_row, lane_id);
            }
            if (lane_id == 0) {
                row_maximum[row] = maximum;
            }
        }
        __syncthreads();

        for (int row = thread_id; row < Base::WARP_M;
             row += static_cast<int>(blockDim.x)) {
            const float maximum = row_maximum[row];
            row_scale[row] = cuda_cast<Base::half_t>(
                maximum > 0.0F ? maximum / 127.0F : 1.0F);
        }
        __syncthreads();

        for (int col_base = 0; col_base < args.padded_k;
             col_base += ROT_SIZE) {
            if (args.rot_size == ROT_SIZE) {
                for (int row = warp_id; row < Base::WARP_M;
                     row += Base::NUM_WARPS) {
                    rotate_256_row<true>(
                        args,
                        row_base + row,
                        col_base,
                        lane_id,
                        rotated[row]);
                }
            } else {
                const int elements = Base::WARP_M * ROT_SIZE;
                for (int index = thread_id; index < elements;
                     index += static_cast<int>(blockDim.x)) {
                    const int row = index / ROT_SIZE;
                    const int local_col = index % ROT_SIZE;
                    const int global_row = row_base + row;
                    const int global_col = col_base + local_col;
                    rotated[row][local_col] =
                        global_row < args.actual_m &&
                            global_col < args.logical_k
                        ? args.input[
                              global_row * args.logical_k + global_col]
                        : cuda_cast<Base::half_t>(0.0F);
                }
            }
            __syncthreads();

#pragma unroll
            for (int tile_m = 0; tile_m < Base::WARP_M_TILES; ++tile_m) {
                Base::packed_act_t quantized;
                GEMM::template quantize_w8a8_warp<true>(
                    &rotated[tile_m * Base::INSN_M][warp_id * Base::WARP_K],
                    row_scale + tile_m * Base::INSN_M,
                    ROT_SIZE,
                    quantized,
                    quant_scratch[warp_id]);
                const int global_k = col_base / Base::WARP_K + warp_id;
                const int output_index =
                    (((block_m * (args.padded_k / Base::WARP_K) + global_k) *
                           Base::NUM_WARPS +
                       block_warp) *
                          Base::WARP_M_TILES +
                      tile_m) *
                        Base::WARP_SIZE +
                    lane_id;
                store(&args.output[output_index], quantized);
            }
            __syncthreads();
        }

        if (warp_id == 0) {
            Base::pack_ascales(
                row_scale,
                &args.scales[
                    (block_m * Base::NUM_WARPS + block_warp) *
                    Base::ASCALES_NUM_PACKS * Base::ASCALES_VALID_LANES]);
        }
    }
};

int launch_quantize_weight(
    const void* input,
    const void* scales,
    void* output,
    int n,
    int k,
    cudaStream_t stream) {
    auto function = invoke_kernel<
        QuantizeWeightKernel,
        const Base::half_t*,
        const Base::half_t*,
        Base::packed_wgt_t*,
        int>;
    function<<<
        dim3(n / Base::BLOCK_N, k / Base::WARP_K),
        Base::WARP_SIZE,
        0,
        stream>>>(
        static_cast<const Base::half_t*>(input),
        static_cast<const Base::half_t*>(scales),
        static_cast<Base::packed_wgt_t*>(output),
        k);
    return static_cast<int>(cudaGetLastError());
}

int launch_quantize_rotated_act(
    const void* input,
    void* output,
    void* scales,
    int actual_m,
    int logical_k,
    int rotated_k,
    int padded_m,
    int padded_k,
    int rot_size,
    cudaStream_t stream) {
    using Kernel = QuantizeRotatedActKernel;
    auto function = invoke_kernel<Kernel, typename Kernel::Arguments>;
    function<<<
        dim3(padded_m / Base::WARP_M),
        Base::WARP_SIZE * Base::NUM_WARPS,
        0,
        stream>>>(typename Kernel::Arguments{
        .input = static_cast<const Base::half_t*>(input),
        .output = static_cast<Base::packed_act_t*>(output),
        .scales = static_cast<Base::packed_ascale_t*>(scales),
        .actual_m = actual_m,
        .logical_k = logical_k,
        .rotated_k = rotated_k,
        .padded_m = padded_m,
        .padded_k = padded_k,
        .rot_size = rot_size,
    });
    return static_cast<int>(cudaGetLastError());
}

int launch_gemm(
    const void* activation,
    const void* weight,
    void* output,
    const void* activation_scales,
    const void* weight_scales,
    const void* bias,
    int actual_m,
    int actual_n,
    int padded_m,
    int padded_n,
    int padded_k,
    cudaStream_t stream) {
    using Bias = typename Base::template EpilogueBias<true, false>;
    using Default = typename Base::EpilogueDefault;
    using Nop = typename Base::EpilogueNop;
    using Epilogue = typename Base::template EpilogueCombination<
        Bias,
        Default,
        Nop>;
    const typename Epilogue::Arguments epilogue_args{
        typename Bias::Arguments{
            .bias = static_cast<const Base::packed_wscale_t*>(bias),
            .scale = nullptr,
        },
        typename Default::Arguments{
            .out = static_cast<Base::half_t*>(output),
            .actualM = actual_m,
            .actualN = actual_n,
        },
        typename Nop::Arguments{},
    };
    auto function = invoke_kernel<
        typename GEMM::template gemm_w8a8_kernel<Epilogue>,
        const Base::packed_act_t*,
        const Base::packed_wgt_t*,
        const Base::packed_ascale_t*,
        const Base::packed_wscale_t*,
        int,
        int,
        int,
        typename Epilogue::Arguments,
        bool,
        bool>;
    dim3 grid(padded_m / Base::BLOCK_M, padded_n / Base::BLOCK_N);
    const bool swap_blocks = padded_m > padded_n * 2;
    if (swap_blocks) {
        std::swap(grid.x, grid.y);
    }
    function<<<grid, Base::WARP_SIZE * Base::NUM_WARPS, 0, stream>>>(
        static_cast<const Base::packed_act_t*>(activation),
        static_cast<const Base::packed_wgt_t*>(weight),
        static_cast<const Base::packed_ascale_t*>(activation_scales),
        static_cast<const Base::packed_wscale_t*>(weight_scales),
        padded_m,
        padded_n,
        padded_k,
        epilogue_args,
        swap_blocks,
        false);
    return static_cast<int>(cudaGetLastError());
}

}  // namespace

extern "C" int xqt_convrot_w8a8_quantize_weight(
    const void* input,
    const void* scales,
    void* output,
    int n,
    int k,
    cudaStream_t stream) {
    return launch_quantize_weight(input, scales, output, n, k, stream);
}

extern "C" int xqt_convrot_w8a8_quantize_rotated_act(
    const void* input,
    void* output,
    void* scales,
    int actual_m,
    int logical_k,
    int rotated_k,
    int padded_m,
    int padded_k,
    int rot_size,
    cudaStream_t stream) {
    return launch_quantize_rotated_act(
        input,
        output,
        scales,
        actual_m,
        logical_k,
        rotated_k,
        padded_m,
        padded_k,
        rot_size,
        stream);
}

extern "C" int xqt_convrot_w8a8_gemm(
    const void* activation,
    const void* weight,
    void* output,
    const void* activation_scales,
    const void* weight_scales,
    const void* bias,
    int actual_m,
    int actual_n,
    int padded_m,
    int padded_n,
    int padded_k,
    cudaStream_t stream) {
    return launch_gemm(
        activation,
        weight,
        output,
        activation_scales,
        weight_scales,
        bias,
        actual_m,
        actual_n,
        padded_m,
        padded_n,
        padded_k,
        stream);
}

extern "C" const char* xqt_convrot_w8a8_version() {
#if defined(XQT_W8A8_FP16)
    return "nunchaku_convrot_w8a8_sm89_fp16_v1";
#else
    return "nunchaku_convrot_w8a8_sm89_bf16_v1";
#endif
}
