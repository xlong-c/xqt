#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cstdint>

#include "gemm_w4a4.cuh"
#include "attention.cuh"
#include "epilogues.cuh"
#include "lora.cuh"
#include "svdq_w4a4_sm89_norm_quantize.cuh"

namespace {

using namespace nunchaku::kernels;

enum class ScalarKind : int {
    FP16 = 0,
    BF16 = 1,
};

template <typename Config>
struct QuantizeActPaddedKernel {
    using GEMM = GEMM_W4A4<Config>;
    using Base = GEMMBase<Config>;
    using half_t = typename Base::half_t;
    using fpsum_warp = typename Base::fpsum_warp;
    using packed_act_t = typename Base::packed_act_t;
    using packed_ascale_t = typename Base::packed_ascale_t;
    using packed_wscale_t = typename Base::packed_wscale_t;
    using BlockInfo = typename Base::BlockInfo;
    using LoadAct = typename Base::template load_act_to_fpsum<false>;
    using EpilogueQuantize = typename GEMM::template EpilogueQuantize<false, false, false>;

    static constexpr int MIN_ARCH = 750;
    static constexpr size_t SHMEM_PER_WARP =
        ceilDiv<size_t>(LoadAct::SHMEM_SIZE, 128) * 128;
    static constexpr size_t SHMEM_SIZE = SHMEM_PER_WARP * Base::NUM_WARPS;

    struct Arguments {
        const half_t* input;
        const packed_wscale_t* smooth;
        packed_act_t* output;
        packed_ascale_t* scales;
        int padded_m;
        int padded_k;
        int actual_m;
        int actual_k;
    };

    __device__ void operator()(Arguments args) {
        const BlockInfo block_info{
            .bm = static_cast<int>(blockIdx.x),
            .bn = static_cast<int>(blockIdx.y),
            .numBlocksM = static_cast<int>(gridDim.x),
            .numBlocksN = static_cast<int>(gridDim.y),
        };
        const int warp_id = static_cast<int>(threadIdx.x) / Base::WARP_SIZE;
        const int m_offset = block_info.bm * Base::BLOCK_M + warp_id * Base::WARP_M;
        const int k_offset = block_info.bn * Base::BLOCK_N;

        extern __shared__ uint8_t shared[];
        fpsum_warp values;
        LoadAct{}(
            args.input + m_offset * args.actual_k + k_offset,
            args.actual_k,
            args.actual_m - m_offset,
            args.actual_k - k_offset,
            values,
            shared + warp_id * SHMEM_PER_WARP);
        EpilogueQuantize{}(
            block_info,
            values,
            args.padded_m,
            args.padded_k,
            0,
            typename EpilogueQuantize::Arguments{
                .qout = args.output,
                .oscales = args.scales,
                .shift_value = 0,
                .smooth_factor = args.smooth,
            });
    }
};

template <typename Config>
struct QuantizeRotatedActKernel {
    using GEMM = GEMM_W4A4<Config>;
    using Base = GEMMBase<Config>;
    using half_t = typename Base::half_t;
    using packed_act_t = typename Base::packed_act_t;
    using packed_ascale_t = typename Base::packed_ascale_t;

    static constexpr int MIN_ARCH = 750;
    static constexpr int ROWS = Base::WARP_M;
    static constexpr int TILE_K = Base::WARP_K;
    static constexpr int MAX_ROT_SIZE = 256;
    static constexpr size_t QUANT_SCRATCH_PER_WARP = Base::INSN_M * Base::INSN_K / 2;

    struct Arguments {
        const half_t* input;
        packed_act_t* output;
        packed_ascale_t* scales;
        int actual_m;
        int logical_k;
        int rotated_k;
        int padded_k;
        int rot_size;
    };

    __host__ __device__ static constexpr size_t align_up(size_t value, size_t alignment) {
        return ((value + alignment - 1) / alignment) * alignment;
    }

    static size_t shared_bytes(int tile_span, int warps, int rot_size) {
        size_t offset = 0;
        if (rot_size != MAX_ROT_SIZE) {
            offset += static_cast<size_t>(ROWS) * (tile_span + 1) * sizeof(float);
            offset = align_up(offset, 16);
        }
        offset += static_cast<size_t>(ROWS) * tile_span * sizeof(half_t);
        offset = align_up(offset, 16);
        offset += static_cast<size_t>(warps) * ROWS * sizeof(half_t);
        offset = align_up(offset, 128);
        offset += static_cast<size_t>(warps) * QUANT_SCRATCH_PER_WARP;
        return offset;
    }

    __device__ __forceinline__ static void rotate_256_row(
        const Arguments& args,
        int global_row,
        int col_base,
        int lane_id,
        half_t* output) {
        constexpr unsigned FULL_MASK = 0xffffffffU;
        constexpr int VALUES_PER_LANE = MAX_ROT_SIZE / Base::WARP_SIZE;
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

        // The register index stores d0 and the low bit of d1. Lane bits store
        // the high bit of d1 followed by d2 and d3 for a 4^4 transform.
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

#pragma unroll
        for (int index = 0; index < VALUES_PER_LANE; ++index) {
            const int local_col = lane_id * VALUES_PER_LANE + index;
            const int global_col = col_base + local_col;
            const float value = global_col < args.rotated_k
                ? values[index] * (1.0F / 16.0F)
                : 0.0F;
            output[local_col] = cuda_cast<half_t>(value);
        }
    }

    __device__ void operator()(Arguments args) {
        const int tile_span = args.rot_size > TILE_K ? args.rot_size : TILE_K;
        const int warps = tile_span / TILE_K;
        const int thread_id = static_cast<int>(threadIdx.x);
        const int warp_id = thread_id / Base::WARP_SIZE;
        const int lane_id = thread_id % Base::WARP_SIZE;
        const int row_base = static_cast<int>(blockIdx.x) * ROWS;
        const int col_base = static_cast<int>(blockIdx.y) * tile_span;
        const int float_stride = tile_span + 1;

        extern __shared__ __align__(128) uint8_t shared[];
        size_t offset = 0;
        float* values = nullptr;
        if (args.rot_size != MAX_ROT_SIZE) {
            values = reinterpret_cast<float*>(shared + offset);
            offset += static_cast<size_t>(ROWS) * float_stride * sizeof(float);
            offset = align_up(offset, 16);
        }
        half_t* rotated = reinterpret_cast<half_t*>(shared + offset);
        offset += static_cast<size_t>(ROWS) * tile_span * sizeof(half_t);
        offset = align_up(offset, 16);
        half_t* scale_storage = reinterpret_cast<half_t*>(shared + offset);
        offset += static_cast<size_t>(warps) * ROWS * sizeof(half_t);
        offset = align_up(offset, 128);
        uint8_t* quant_scratch = shared + offset;

        const int element_count = ROWS * tile_span;
        if (args.rot_size == MAX_ROT_SIZE) {
            for (int row = warp_id; row < ROWS; row += warps) {
                rotate_256_row(
                    args,
                    row_base + row,
                    col_base,
                    lane_id,
                    rotated + row * tile_span);
            }
        } else {
            for (int index = thread_id; index < element_count;
                 index += static_cast<int>(blockDim.x)) {
                const int row = index / tile_span;
                const int local_col = index % tile_span;
                const int global_row = row_base + row;
                const int global_col = col_base + local_col;
                float value = 0.0F;
                if (global_row < args.actual_m && global_col < args.logical_k) {
                    value = static_cast<float>(
                        args.input[global_row * args.logical_k + global_col]);
                }
                values[row * float_stride + local_col] = value;
            }
            __syncthreads();

            const int quartets_per_row = tile_span / 4;
            const int quartet_count = ROWS * quartets_per_row;
            for (int stride = 1; stride < args.rot_size; stride *= 4) {
                for (int index = thread_id; index < quartet_count;
                     index += static_cast<int>(blockDim.x)) {
                    const int row = index / quartets_per_row;
                    const int quartet = index % quartets_per_row;
                    const int low = quartet % stride;
                    const int high = quartet / stride;
                    const int base = high * 4 * stride + low;
                    float* row_values = values + row * float_stride;
                    const float a = row_values[base];
                    const float b = row_values[base + stride];
                    const float c = row_values[base + 2 * stride];
                    const float d = row_values[base + 3 * stride];
                    row_values[base] = a + b + c - d;
                    row_values[base + stride] = a + b - c + d;
                    row_values[base + 2 * stride] = a - b + c + d;
                    row_values[base + 3 * stride] = -a + b + c + d;
                }
                __syncthreads();
            }

            const float normalization = rsqrtf(static_cast<float>(args.rot_size));
            for (int index = thread_id; index < element_count;
                 index += static_cast<int>(blockDim.x)) {
                const int row = index / tile_span;
                const int col = index % tile_span;
                float value = values[row * float_stride + col] * normalization;
                if (col_base + col >= args.rotated_k) {
                    value = 0.0F;
                }
                rotated[index] = cuda_cast<half_t>(value);
            }
        }
        __syncthreads();

        const int global_warp_row = static_cast<int>(blockIdx.x);
        const int block_m = global_warp_row / Base::NUM_WARPS;
        const int block_warp = global_warp_row % Base::NUM_WARPS;
        const int block_k = static_cast<int>(blockIdx.y) * warps + warp_id;
        const int k_tiles = args.padded_k / TILE_K;
        half_t* warp_scales = scale_storage + warp_id * ROWS;
        uint8_t* warp_scratch = quant_scratch + warp_id * QUANT_SCRATCH_PER_WARP;

        for (int tile_id = 0; tile_id < Base::WARP_M_TILES; ++tile_id) {
            packed_act_t quantized;
            GEMM::template quantize_w4a4_warp<true>(
                rotated + tile_id * Base::INSN_M * tile_span + warp_id * TILE_K,
                tile_span,
                quantized,
                warp_scales + tile_id * Base::INSN_M,
                warp_scratch);
            const int output_index =
                (((block_m * k_tiles + block_k) * Base::NUM_WARPS + block_warp) *
                     Base::WARP_M_TILES +
                 tile_id) *
                    Base::WARP_SIZE +
                lane_id;
            store(&args.output[output_index], quantized);
        }
        Base::pack_ascales(
            warp_scales,
            &args.scales[
                ((block_m * k_tiles + block_k) * Base::NUM_WARPS + block_warp) *
                Base::ASCALES_NUM_PACKS * Base::ASCALES_VALID_LANES]);
    }
};

template <typename Config>
int launch_quantize_weight(
    const void* input,
    void* output,
    void* scales,
    int n,
    int k,
    cudaStream_t stream) {
    using GEMM = GEMM_W4A4<Config>;
    using Kernel = typename GEMM::quantize_w4a4_wgt_kernel;
    auto function = invoke_kernel<
        Kernel,
        const typename GEMM::half_t*,
        typename GEMM::packed_wgt_t*,
        typename GEMM::packed_wscale_t*,
        int>;
    function<<<dim3(n / GEMM::WARP_N, k / GEMM::WARP_K), GEMM::WARP_SIZE, 0, stream>>>(
        static_cast<const typename GEMM::half_t*>(input),
        static_cast<typename GEMM::packed_wgt_t*>(output),
        static_cast<typename GEMM::packed_wscale_t*>(scales),
        k);
    return static_cast<int>(cudaGetLastError());
}

template <typename Config>
int launch_quantize_act(
    const void* input,
    void* output,
    void* scales,
    const void* smooth,
    int actual_m,
    int actual_k,
    int padded_m,
    int padded_k,
    cudaStream_t stream) {
    using Kernel = QuantizeActPaddedKernel<Config>;
    auto function = invoke_kernel<Kernel, typename Kernel::Arguments>;
    cudaError_t status = cudaFuncSetAttribute(
        function,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(Kernel::SHMEM_SIZE));
    if (status != cudaSuccess) {
        return static_cast<int>(status);
    }
    function<<<
        dim3(padded_m / Kernel::GEMM::BLOCK_M, padded_k / Kernel::GEMM::BLOCK_N),
        Kernel::GEMM::WARP_SIZE * Kernel::GEMM::NUM_WARPS,
        Kernel::SHMEM_SIZE,
        stream>>>(typename Kernel::Arguments{
        .input = static_cast<const typename Kernel::half_t*>(input),
        .smooth = static_cast<const typename Kernel::packed_wscale_t*>(smooth),
        .output = static_cast<typename Kernel::packed_act_t*>(output),
        .scales = static_cast<typename Kernel::packed_ascale_t*>(scales),
        .padded_m = padded_m,
        .padded_k = padded_k,
        .actual_m = actual_m,
        .actual_k = actual_k,
    });
    return static_cast<int>(cudaGetLastError());
}

template <typename Config>
int launch_quantize_rotated_act(
    const void* input,
    void* output,
    void* scales,
    int actual_m,
    int logical_k,
    int rotated_k,
    int padded_k,
    int rot_size,
    cudaStream_t stream) {
    using Kernel = QuantizeRotatedActKernel<Config>;
    const int tile_span = rot_size > Kernel::TILE_K ? rot_size : Kernel::TILE_K;
    const int warps = tile_span / Kernel::TILE_K;
    const size_t shared_bytes = Kernel::shared_bytes(tile_span, warps, rot_size);
    auto function = invoke_kernel<Kernel, typename Kernel::Arguments>;
    cudaError_t status = cudaFuncSetAttribute(
        function,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(shared_bytes));
    if (status != cudaSuccess) {
        return static_cast<int>(status);
    }
    function<<<
        dim3(ceilDiv(actual_m, Kernel::ROWS), padded_k / tile_span),
        warps * Kernel::GEMM::WARP_SIZE,
        shared_bytes,
        stream>>>(typename Kernel::Arguments{
        .input = static_cast<const typename Kernel::half_t*>(input),
        .output = static_cast<typename Kernel::packed_act_t*>(output),
        .scales = static_cast<typename Kernel::packed_ascale_t*>(scales),
        .actual_m = actual_m,
        .logical_k = logical_k,
        .rotated_k = rotated_k,
        .padded_k = padded_k,
        .rot_size = rot_size,
    });
    return static_cast<int>(cudaGetLastError());
}

template <typename Config>
int launch_quantize_act_lora(
    const void* input,
    void* output,
    void* scales,
    const void* lora_down,
    void* lora_act,
    const void* smooth,
    int actual_m,
    int actual_k,
    int padded_m,
    int padded_k,
    int rank,
    cudaStream_t stream) {
    using GEMM = GEMM_W4A4<Config>;
    using Kernel = typename GEMM::template quantize_w4a4_fuse_lora_kernel<false, false>;
    auto function = invoke_kernel<Kernel, typename Kernel::Arguments>;
    cudaError_t status = cudaFuncSetAttribute(
        function,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(Kernel::SHMEM_SIZE));
    if (status != cudaSuccess) {
        return static_cast<int>(status);
    }
    function<<<
        dim3(padded_m / GEMM::BLOCK_M, padded_k / GEMM::BLOCK_N),
        GEMM::WARP_SIZE * GEMM::NUM_WARPS,
        Kernel::SHMEM_SIZE,
        stream>>>(typename Kernel::Arguments{
        .input = static_cast<const typename GEMM::half_t*>(input),
        .smooth_factor = static_cast<const typename GEMM::packed_wscale_t*>(smooth),
        .output = static_cast<typename GEMM::packed_act_t*>(output),
        .oscales = static_cast<typename GEMM::packed_ascale_t*>(scales),
        .lora_wgt_down = static_cast<const typename GEMM::packed_fpsum_t*>(lora_down),
        .lora_act = static_cast<float*>(lora_act),
        .lora_rank = rank,
        .M = padded_m,
        .N = padded_k,
        .actualM = actual_m,
        .actualN = actual_k,
        .alwaysfalse = false,
    });
    return static_cast<int>(cudaGetLastError());
}

template <typename Config, bool UseLora, bool ActUnsigned = false>
int launch_gemm(
    const void* act,
    const void* weight,
    void* output,
    const void* activation_scales,
    const void* weight_scales,
    const void* lora_act,
    const void* lora_up,
    const void* bias,
    int actual_m,
    int actual_n,
    int padded_m,
    int padded_n,
    int padded_k,
    int rank,
    float lora_scale,
    cudaStream_t stream) {
    using GEMM = GEMM_W4A4<Config>;
    using Base = GEMMBase<Config>;
    using Bias = typename Base::template EpilogueBias<true, false>;
    using Default = typename Base::EpilogueDefault;
    using Nop = typename Base::EpilogueNop;

    const typename Bias::Arguments bias_args{
        .bias = static_cast<const typename Base::packed_wscale_t*>(bias),
        .scale = nullptr,
    };
    const typename Default::Arguments output_args{
        .out = static_cast<typename Base::half_t*>(output),
        .actualM = actual_m,
        .actualN = actual_n,
    };

    auto launch = [&]<typename Epilogue>(const typename Epilogue::Arguments& arguments) {
        auto function = invoke_kernel<
            typename GEMM::template gemm_w4a4_kernel<Epilogue, ActUnsigned>,
            const typename Base::packed_act_t*,
            const typename Base::packed_wgt_t*,
            const typename Base::packed_ascale_t*,
            const typename Base::packed_wscale_t*,
            int,
            int,
            int,
            typename Epilogue::Arguments,
            bool,
            bool>;
        dim3 grid(padded_m / GEMM::BLOCK_M, padded_n / GEMM::BLOCK_N);
        bool swap_blocks = padded_m > padded_n * 2;
        if (swap_blocks) {
            std::swap(grid.x, grid.y);
        }
        function<<<grid, GEMM::WARP_SIZE * GEMM::NUM_WARPS, 0, stream>>>(
            static_cast<const typename Base::packed_act_t*>(act),
            static_cast<const typename Base::packed_wgt_t*>(weight),
            static_cast<const typename Base::packed_ascale_t*>(activation_scales),
            static_cast<const typename Base::packed_wscale_t*>(weight_scales),
            padded_m,
            padded_n,
            padded_k,
            arguments,
            swap_blocks,
            false);
    };

    if constexpr (UseLora) {
        using LoraImpl = Lora<Config>;
        using LoraUp = typename LoraImpl::EpilogueLoraUp;
        using LoraChain = typename Base::template EpilogueCombination<LoraUp, Nop, Default, Nop>;
        using Epilogue = typename Base::template EpilogueCombination<Bias, LoraChain, Nop>;
        typename LoraImpl::scale_t scales{};
        const int scale_count = std::min(rank / LoraImpl::WARP_R, static_cast<int>(scales.size()));
        for (int index = 0; index < scale_count; ++index) {
            scales[static_cast<size_t>(index)] = lora_scale;
        }
        const typename LoraUp::Arguments lora_args{
            .lora_act = static_cast<const float*>(lora_act),
            .lora_wgt_up = static_cast<const typename Base::packed_fpsum_t*>(lora_up),
            .rank = rank,
            .scales = scales,
            .alwaysfalse = false,
        };
        launch.template operator()<Epilogue>(typename Epilogue::Arguments{
            bias_args,
            typename LoraChain::Arguments{lora_args, typename Nop::Arguments{}, output_args, typename Nop::Arguments{}},
            typename Nop::Arguments{},
        });
    } else {
        using Epilogue = typename Base::template EpilogueCombination<Bias, Default, Nop>;
        launch.template operator()<Epilogue>(typename Epilogue::Arguments{
            bias_args,
            output_args,
            typename Nop::Arguments{},
        });
    }
    return static_cast<int>(cudaGetLastError());
}

template <typename Config>
int launch_gemm_lora_qkv_rmsnorm_rope(
    const void* act,
    const void* weight,
    void* output,
    const void* activation_scales,
    const void* weight_scales,
    const void* lora_act,
    const void* lora_up,
    const void* bias,
    const void* norm_q,
    const void* norm_k,
    const void* rotary_emb,
    int actual_m,
    int actual_n,
    int padded_m,
    int padded_n,
    int padded_k,
    int rank,
    float lora_scale,
    float eps,
    cudaStream_t stream) {
    using GEMM = GEMM_W4A4<Config>;
    using Base = GEMMBase<Config>;
    using Bias = typename Base::template EpilogueBias<true, false>;
    using Default = typename Base::EpilogueDefault;
    using Nop = typename Base::EpilogueNop;
    using LoraImpl = Lora<Config>;
    using LoraUp = typename LoraImpl::EpilogueLoraUp;
    using ExtraEpilogues = Epilogues<Config>;
    using Rope = typename ExtraEpilogues::EpilogueRMSNormRope;
    using QKVRopeChain = typename Base::template EpilogueCombination<
        LoraUp,
        Nop,
        Rope,
        Default,
        Nop>;
    using Epilogue = typename Base::template EpilogueCombination<
        Bias,
        QKVRopeChain,
        Nop>;

    static_assert(Rope::HEAD_DIM == 128);

    typename LoraImpl::scale_t scales{};
    const int scale_count = std::min(
        rank / LoraImpl::WARP_R,
        static_cast<int>(scales.size()));
    for (int index = 0; index < scale_count; ++index) {
        scales[static_cast<size_t>(index)] = lora_scale;
    }

    const typename Bias::Arguments bias_args{
        .bias = static_cast<const typename Base::packed_wscale_t*>(bias),
        .scale = nullptr,
    };
    const typename LoraUp::Arguments lora_args{
        .lora_act = static_cast<const float*>(lora_act),
        .lora_wgt_up = static_cast<const typename Base::packed_fpsum_t*>(lora_up),
        .rank = rank,
        .scales = scales,
        .alwaysfalse = false,
    };
    const typename Rope::Arguments rope_args{
        .rotary_emb = static_cast<const typename Rope::packed_rotemb_t*>(rotary_emb),
        .rmsnorm_weight_q = static_cast<const typename Base::half_t*>(norm_q),
        .rmsnorm_weight_k = static_cast<const typename Base::half_t*>(norm_k),
        .epsilon = eps,
    };
    const typename Default::Arguments output_args{
        .out = static_cast<typename Base::half_t*>(output),
        .actualM = actual_m,
        .actualN = actual_n,
    };

    auto function = invoke_kernel<
        typename GEMM::template gemm_w4a4_kernel<Epilogue, false>,
        const typename Base::packed_act_t*,
        const typename Base::packed_wgt_t*,
        const typename Base::packed_ascale_t*,
        const typename Base::packed_wscale_t*,
        int,
        int,
        int,
        typename Epilogue::Arguments,
        bool,
        bool>;
    dim3 grid(padded_m / GEMM::BLOCK_M, padded_n / GEMM::BLOCK_N);
    bool swap_blocks = padded_m > padded_n * 2;
    if (swap_blocks) {
        std::swap(grid.x, grid.y);
    }
    function<<<grid, GEMM::WARP_SIZE * GEMM::NUM_WARPS, 0, stream>>>(
        static_cast<const typename Base::packed_act_t*>(act),
        static_cast<const typename Base::packed_wgt_t*>(weight),
        static_cast<const typename Base::packed_ascale_t*>(activation_scales),
        static_cast<const typename Base::packed_wscale_t*>(weight_scales),
        padded_m,
        padded_n,
        padded_k,
        typename Epilogue::Arguments{
            bias_args,
            typename QKVRopeChain::Arguments{
                lora_args,
                typename Nop::Arguments{},
                rope_args,
                output_args,
                typename Nop::Arguments{},
            },
            typename Nop::Arguments{},
        },
        swap_blocks,
        false);
    return static_cast<int>(cudaGetLastError());
}

template <typename Config>
int launch_gemm_lora_qkv_rmsnorm_rope_packed(
    const void* act,
    const void* weight,
    const void* activation_scales,
    const void* weight_scales,
    const void* lora_act,
    const void* lora_up,
    const void* bias,
    const void* norm_q,
    const void* norm_k,
    const void* rotary_emb,
    void* out_q,
    void* out_k,
    void* out_v,
    int stride_head_q,
    int stride_head_k,
    int stride_head_v,
    int actual_m,
    int padded_m,
    int padded_n,
    int padded_k,
    int rank,
    float lora_scale,
    float eps,
    cudaStream_t stream) {
    using GEMM = GEMM_W4A4<Config>;
    using Base = GEMMBase<Config>;
    using Bias = typename Base::template EpilogueBias<true, false>;
    using Nop = typename Base::EpilogueNop;
    using LoraImpl = Lora<Config>;
    using LoraUp = typename LoraImpl::EpilogueLoraUp;
    using ExtraEpilogues = Epilogues<Config>;
    using Rope = typename ExtraEpilogues::EpilogueRMSNormRope;
    using PackQKV = typename ExtraEpilogues::EpiloguePackQKV;
    using QKVRopePackChain = typename Base::template EpilogueCombination<
        LoraUp,
        Nop,
        Rope,
        PackQKV,
        Nop>;
    using Epilogue = typename Base::template EpilogueCombination<
        Bias,
        QKVRopePackChain,
        Nop>;

    static_assert(Rope::HEAD_DIM == 128);
    static_assert(PackQKV::HEAD_DIM == 128);

    typename LoraImpl::scale_t scales{};
    const int scale_count = std::min(
        rank / LoraImpl::WARP_R,
        static_cast<int>(scales.size()));
    for (int index = 0; index < scale_count; ++index) {
        scales[static_cast<size_t>(index)] = lora_scale;
    }

    const typename Bias::Arguments bias_args{
        .bias = static_cast<const typename Base::packed_wscale_t*>(bias),
        .scale = nullptr,
    };
    const typename LoraUp::Arguments lora_args{
        .lora_act = static_cast<const float*>(lora_act),
        .lora_wgt_up = static_cast<const typename Base::packed_fpsum_t*>(lora_up),
        .rank = rank,
        .scales = scales,
        .alwaysfalse = false,
    };
    const typename Rope::Arguments rope_args{
        .rotary_emb = static_cast<const typename Rope::packed_rotemb_t*>(rotary_emb),
        .rmsnorm_weight_q = static_cast<const typename Base::half_t*>(norm_q),
        .rmsnorm_weight_k = static_cast<const typename Base::half_t*>(norm_k),
        .epsilon = eps,
    };
    const typename PackQKV::Arguments pack_args{
        .out_q = static_cast<typename PackQKV::packed_qkv_t*>(out_q),
        .out_k = static_cast<typename PackQKV::packed_qkv_t*>(out_k),
        .out_v = static_cast<typename PackQKV::packed_qkv_t*>(out_v),
        .actualM = actual_m,
        .strideHead_q = stride_head_q,
        .strideHead_k = stride_head_k,
        .strideHead_v = stride_head_v,
    };

    auto function = invoke_kernel<
        typename GEMM::template gemm_w4a4_kernel<Epilogue, false>,
        const typename Base::packed_act_t*,
        const typename Base::packed_wgt_t*,
        const typename Base::packed_ascale_t*,
        const typename Base::packed_wscale_t*,
        int,
        int,
        int,
        typename Epilogue::Arguments,
        bool,
        bool>;
    dim3 grid(padded_m / GEMM::BLOCK_M, padded_n / GEMM::BLOCK_N);
    bool swap_blocks = padded_m > padded_n * 2;
    if (swap_blocks) {
        std::swap(grid.x, grid.y);
    }
    function<<<grid, GEMM::WARP_SIZE * GEMM::NUM_WARPS, 0, stream>>>(
        static_cast<const typename Base::packed_act_t*>(act),
        static_cast<const typename Base::packed_wgt_t*>(weight),
        static_cast<const typename Base::packed_ascale_t*>(activation_scales),
        static_cast<const typename Base::packed_wscale_t*>(weight_scales),
        padded_m,
        padded_n,
        padded_k,
        typename Epilogue::Arguments{
            bias_args,
            typename QKVRopePackChain::Arguments{
                lora_args,
                typename Nop::Arguments{},
                rope_args,
                pack_args,
                typename Nop::Arguments{},
            },
            typename Nop::Arguments{},
        },
        swap_blocks,
        false);
    return static_cast<int>(cudaGetLastError());
}

template <bool BF16Output>
int launch_attention_fp16(
    const void* query,
    const void* key,
    const void* value,
    void* output,
    int batch,
    int heads,
    int query_tokens,
    int key_value_tokens,
    float scale,
    cudaStream_t stream) {
    using AttentionImpl = Attention<AttentionFP16Config<BF16Output>>;
    using GEMM = typename AttentionImpl::GEMM;
    using Epilogue = typename GEMM::EpilogueDefault;

    const typename Epilogue::Arguments output_args{
        .out = static_cast<typename GEMM::half_t*>(output),
        .actualM = batch * query_tokens,
        .actualN = heads * AttentionImpl::HEAD_DIM,
    };
    auto function = invoke_kernel<
        typename AttentionImpl::template attention_fp16_kernel<Epilogue>,
        const typename AttentionImpl::packed_q_t*,
        const typename AttentionImpl::packed_k_t*,
        const typename AttentionImpl::packed_v_t*,
        float,
        int,
        int,
        typename Epilogue::Arguments,
        bool>;
    const dim3 grid(
        query_tokens / AttentionImpl::BLOCK_M,
        heads,
        batch);
    function<<<grid, GEMM::WARP_SIZE * GEMM::NUM_WARPS, 0, stream>>>(
        static_cast<const typename AttentionImpl::packed_q_t*>(query),
        static_cast<const typename AttentionImpl::packed_k_t*>(key),
        static_cast<const typename AttentionImpl::packed_v_t*>(value),
        scale * 1.4426950408889634074F,
        query_tokens,
        key_value_tokens,
        output_args,
        false);
    return static_cast<int>(cudaGetLastError());
}

template <typename Config>
int launch_gemm_lora_gelu_quantize_lora(
    const void* act,
    const void* weight,
    void* quantized_output,
    const void* activation_scales,
    const void* weight_scales,
    void* output_scales,
    const void* lora_act_in,
    const void* lora_up,
    const void* lora_down,
    void* lora_act_out,
    const void* bias,
    const void* smooth,
    int actual_m,
    int actual_n,
    int padded_m,
    int padded_n,
    int padded_k,
    int rank_up,
    int rank_down,
    float lora_scale,
    cudaStream_t stream) {
    using GEMM = GEMM_W4A4<Config>;
    using Base = GEMMBase<Config>;
    using Bias = typename Base::template EpilogueBias<true, false>;
    using Gelu = typename Epilogues<Config>::EpilogueGelu;
    using LoraImpl = Lora<Config>;
    using LoraUp = typename LoraImpl::EpilogueLoraUp;
    using LoraDown = typename LoraImpl::EpilogueLoraDown;
    using Quantize = typename GEMM::template EpilogueQuantize<false, true, false>;
    using Nop = typename Base::EpilogueNop;
    using Chain = typename Base::template EpilogueCombination<
        LoraUp,
        Gelu,
        LoraDown,
        Quantize,
        Nop>;
    using Epilogue = typename Base::template EpilogueCombination<Bias, Chain, Nop>;

    typename LoraImpl::scale_t scales{};
    const int scale_count = std::min(
        rank_up / LoraImpl::WARP_R,
        static_cast<int>(scales.size()));
    for (int index = 0; index < scale_count; ++index) {
        scales[static_cast<size_t>(index)] = lora_scale;
    }

    const typename Bias::Arguments bias_args{
        .bias = static_cast<const typename Base::packed_wscale_t*>(bias),
        .scale = nullptr,
    };
    const typename LoraUp::Arguments lora_up_args{
        .lora_act = static_cast<const float*>(lora_act_in),
        .lora_wgt_up = static_cast<const typename Base::packed_fpsum_t*>(lora_up),
        .rank = rank_up,
        .scales = scales,
        .alwaysfalse = false,
    };
    const typename LoraDown::Arguments lora_down_args{
        .lora_wgt_down = static_cast<const typename Base::packed_fpsum_t*>(lora_down),
        .lora_act = static_cast<float*>(lora_act_out),
        .rank = rank_down,
        .alwaysfalse = false,
    };
    const typename Quantize::Arguments quantize_args{
        .qout = static_cast<typename Base::packed_act_t*>(quantized_output),
        .oscales = static_cast<typename Quantize::oscales_t*>(output_scales),
        .shift_value = 0.171875F,
        .smooth_factor = static_cast<const typename Base::packed_wscale_t*>(smooth),
    };

    auto function = invoke_kernel<
        typename GEMM::template gemm_w4a4_kernel<Epilogue, false>,
        const typename Base::packed_act_t*,
        const typename Base::packed_wgt_t*,
        const typename Base::packed_ascale_t*,
        const typename Base::packed_wscale_t*,
        int,
        int,
        int,
        typename Epilogue::Arguments,
        bool,
        bool>;
    dim3 grid(padded_m / GEMM::BLOCK_M, padded_n / GEMM::BLOCK_N);
    bool swap_blocks = padded_m > padded_n * 2;
    if (swap_blocks) {
        std::swap(grid.x, grid.y);
    }
    function<<<grid, GEMM::WARP_SIZE * GEMM::NUM_WARPS, 0, stream>>>(
        static_cast<const typename Base::packed_act_t*>(act),
        static_cast<const typename Base::packed_wgt_t*>(weight),
        static_cast<const typename Base::packed_ascale_t*>(activation_scales),
        static_cast<const typename Base::packed_wscale_t*>(weight_scales),
        padded_m,
        padded_n,
        padded_k,
        typename Epilogue::Arguments{
            bias_args,
            typename Chain::Arguments{
                lora_up_args,
                typename Gelu::Arguments{},
                lora_down_args,
                quantize_args,
                typename Nop::Arguments{},
            },
            typename Nop::Arguments{},
        },
        swap_blocks,
        false);
    return static_cast<int>(cudaGetLastError());
}

template <typename Function>
int dispatch_scalar(int scalar_kind, Function&& function) {
    switch (static_cast<ScalarKind>(scalar_kind)) {
        case ScalarKind::FP16:
            return function.template operator()<GEMMConfig_W4A4_FP16>();
        case ScalarKind::BF16:
            return function.template operator()<GEMMConfig_W4A4_BF16>();
    }
    return static_cast<int>(cudaErrorInvalidValue);
}

}  // namespace

extern "C" int xqt_svdq_w4a4_quantize_weight(
    const void* input,
    void* output,
    void* scales,
    int n,
    int k,
    int scalar_kind,
    cudaStream_t stream) {
    return dispatch_scalar(scalar_kind, [&]<typename Config>() {
        return launch_quantize_weight<Config>(input, output, scales, n, k, stream);
    });
}

extern "C" int xqt_svdq_w4a4_quantize_act(
    const void* input,
    void* output,
    void* scales,
    const void* smooth,
    int actual_m,
    int actual_k,
    int padded_m,
    int padded_k,
    int scalar_kind,
    cudaStream_t stream) {
    return dispatch_scalar(scalar_kind, [&]<typename Config>() {
        return launch_quantize_act<Config>(
            input,
            output,
            scales,
            smooth,
            actual_m,
            actual_k,
            padded_m,
            padded_k,
            stream);
    });
}

extern "C" int xqt_svdq_w4a4_quantize_act_lora(
    const void* input,
    void* output,
    void* scales,
    const void* lora_down,
    void* lora_act,
    const void* smooth,
    int actual_m,
    int actual_k,
    int padded_m,
    int padded_k,
    int rank,
    int scalar_kind,
    cudaStream_t stream) {
    return dispatch_scalar(scalar_kind, [&]<typename Config>() {
        return launch_quantize_act_lora<Config>(
            input,
            output,
            scales,
            lora_down,
            lora_act,
            smooth,
            actual_m,
            actual_k,
            padded_m,
            padded_k,
            rank,
            stream);
    });
}

extern "C" int xqt_svdq_w4a4_norm_quantize_act_lora(
    const void* input,
    const void* norm_weight,
    const void* row_scales,
    void* output,
    void* activation_scales,
    const void* lora_down,
    void* lora_activation,
    const void* smooth,
    int actual_m,
    int actual_k,
    int padded_m,
    int padded_k,
    int rank,
    int scalar_kind,
    cudaStream_t stream) {
    return dispatch_scalar(scalar_kind, [&]<typename Config>() {
        return launch_xqt_norm_quantize_act_lora<Config>(
            input,
            norm_weight,
            row_scales,
            output,
            activation_scales,
            lora_down,
            lora_activation,
            smooth,
            actual_m,
            actual_k,
            padded_m,
            padded_k,
            rank,
            stream);
    });
}

extern "C" int xqt_svdq_w4a4_quantize_rotated_act(
    const void* input,
    void* output,
    void* scales,
    int actual_m,
    int logical_k,
    int rotated_k,
    int padded_k,
    int rot_size,
    int scalar_kind,
    cudaStream_t stream) {
    return dispatch_scalar(scalar_kind, [&]<typename Config>() {
        return launch_quantize_rotated_act<Config>(
            input,
            output,
            scales,
            actual_m,
            logical_k,
            rotated_k,
            padded_k,
            rot_size,
            stream);
    });
}

extern "C" int xqt_svdq_w4a4_gemm(
    const void* act,
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
    int scalar_kind,
    cudaStream_t stream) {
    return dispatch_scalar(scalar_kind, [&]<typename Config>() {
        return launch_gemm<Config, false>(
            act,
            weight,
            output,
            activation_scales,
            weight_scales,
            nullptr,
            nullptr,
            bias,
            actual_m,
            actual_n,
            padded_m,
            padded_n,
            padded_k,
            0,
            0.0F,
            stream);
    });
}

extern "C" int xqt_svdq_w4a4_gemm_lora(
    const void* act,
    const void* weight,
    void* output,
    const void* activation_scales,
    const void* weight_scales,
    const void* lora_act,
    const void* lora_up,
    const void* bias,
    int actual_m,
    int actual_n,
    int padded_m,
    int padded_n,
    int padded_k,
    int rank,
    float lora_scale,
    int scalar_kind,
    cudaStream_t stream) {
    return dispatch_scalar(scalar_kind, [&]<typename Config>() {
        return launch_gemm<Config, true>(
            act,
            weight,
            output,
            activation_scales,
            weight_scales,
            lora_act,
            lora_up,
            bias,
            actual_m,
            actual_n,
            padded_m,
            padded_n,
            padded_k,
            rank,
            lora_scale,
            stream);
    });
}

extern "C" int xqt_svdq_w4a4_gemm_lora_unsigned(
    const void* act,
    const void* weight,
    void* output,
    const void* activation_scales,
    const void* weight_scales,
    const void* lora_act,
    const void* lora_up,
    const void* bias,
    int actual_m,
    int actual_n,
    int padded_m,
    int padded_n,
    int padded_k,
    int rank,
    float lora_scale,
    int scalar_kind,
    cudaStream_t stream) {
    return dispatch_scalar(scalar_kind, [&]<typename Config>() {
        return launch_gemm<Config, true, true>(
            act,
            weight,
            output,
            activation_scales,
            weight_scales,
            lora_act,
            lora_up,
            bias,
            actual_m,
            actual_n,
            padded_m,
            padded_n,
            padded_k,
            rank,
            lora_scale,
            stream);
    });
}

extern "C" int xqt_svdq_w4a4_gemm_lora_qkv_rmsnorm_rope(
    const void* act,
    const void* weight,
    void* output,
    const void* activation_scales,
    const void* weight_scales,
    const void* lora_act,
    const void* lora_up,
    const void* bias,
    const void* norm_q,
    const void* norm_k,
    const void* rotary_emb,
    int actual_m,
    int actual_n,
    int padded_m,
    int padded_n,
    int padded_k,
    int rank,
    float lora_scale,
    float eps,
    int scalar_kind,
    cudaStream_t stream) {
    return dispatch_scalar(scalar_kind, [&]<typename Config>() {
        return launch_gemm_lora_qkv_rmsnorm_rope<Config>(
            act,
            weight,
            output,
            activation_scales,
            weight_scales,
            lora_act,
            lora_up,
            bias,
            norm_q,
            norm_k,
            rotary_emb,
            actual_m,
            actual_n,
            padded_m,
            padded_n,
            padded_k,
            rank,
            lora_scale,
            eps,
            stream);
    });
}

extern "C" int xqt_svdq_w4a4_gemm_lora_qkv_rmsnorm_rope_packed(
    const void* act,
    const void* weight,
    const void* activation_scales,
    const void* weight_scales,
    const void* lora_act,
    const void* lora_up,
    const void* bias,
    const void* norm_q,
    const void* norm_k,
    const void* rotary_emb,
    void* out_q,
    void* out_k,
    void* out_v,
    int stride_head_q,
    int stride_head_k,
    int stride_head_v,
    int actual_m,
    int padded_m,
    int padded_n,
    int padded_k,
    int rank,
    float lora_scale,
    float eps,
    int scalar_kind,
    cudaStream_t stream) {
    return dispatch_scalar(scalar_kind, [&]<typename Config>() {
        return launch_gemm_lora_qkv_rmsnorm_rope_packed<Config>(
            act,
            weight,
            activation_scales,
            weight_scales,
            lora_act,
            lora_up,
            bias,
            norm_q,
            norm_k,
            rotary_emb,
            out_q,
            out_k,
            out_v,
            stride_head_q,
            stride_head_k,
            stride_head_v,
            actual_m,
            padded_m,
            padded_n,
            padded_k,
            rank,
            lora_scale,
            eps,
            stream);
    });
}

extern "C" int xqt_svdq_w4a4_attention_fp16(
    const void* query,
    const void* key,
    const void* value,
    void* output,
    int batch,
    int heads,
    int query_tokens,
    int key_value_tokens,
    float scale,
    int output_scalar_kind,
    cudaStream_t stream) {
    if (output_scalar_kind == static_cast<int>(ScalarKind::FP16)) {
        return launch_attention_fp16<false>(
            query,
            key,
            value,
            output,
            batch,
            heads,
            query_tokens,
            key_value_tokens,
            scale,
            stream);
    }
    if (output_scalar_kind == static_cast<int>(ScalarKind::BF16)) {
        return launch_attention_fp16<true>(
            query,
            key,
            value,
            output,
            batch,
            heads,
            query_tokens,
            key_value_tokens,
            scale,
            stream);
    }
    return static_cast<int>(cudaErrorInvalidValue);
}

extern "C" int xqt_svdq_w4a4_gemm_lora_gelu_quantize_lora(
    const void* act,
    const void* weight,
    void* quantized_output,
    const void* activation_scales,
    const void* weight_scales,
    void* output_scales,
    const void* lora_act_in,
    const void* lora_up,
    const void* lora_down,
    void* lora_act_out,
    const void* bias,
    const void* smooth,
    int actual_m,
    int actual_n,
    int padded_m,
    int padded_n,
    int padded_k,
    int rank_up,
    int rank_down,
    float lora_scale,
    int scalar_kind,
    cudaStream_t stream) {
    return dispatch_scalar(scalar_kind, [&]<typename Config>() {
        return launch_gemm_lora_gelu_quantize_lora<Config>(
            act,
            weight,
            quantized_output,
            activation_scales,
            weight_scales,
            output_scales,
            lora_act_in,
            lora_up,
            lora_down,
            lora_act_out,
            bias,
            smooth,
            actual_m,
            actual_n,
            padded_m,
            padded_n,
            padded_k,
            rank_up,
            rank_down,
            lora_scale,
            stream);
    });
}

extern "C" const char* xqt_svdq_w4a4_version() {
    return "xqt_w4a4_sm89_packed_qkv_attention_v6";
}
