#pragma once

#include <array>
#include <cstdint>

#include "gemm_w4a4.cuh"
#include "lora.cuh"

namespace nunchaku::kernels {

// Loads one activation tile, applies the explicit RMSNorm cast semantics,
// and exposes the normalized half/BF16 fragment to both LoRA-down and W4A4
// quantization. The row RMS reduction remains a separate kernel because its
// result spans every K tile in the row.
template <typename Config>
struct XQTNormQuantizeActLoraKernel {
    using GEMM = GEMM_W4A4<Config>;
    using Base = GEMMBase<Config>;
    using half_t = typename Base::half_t;
    using fpsum_warp = typename Base::fpsum_warp;
    using packed_act_t = typename Base::packed_act_t;
    using packed_ascale_t = typename Base::packed_ascale_t;
    using packed_fpsum_t = typename Base::packed_fpsum_t;
    using packed_wscale_t = typename Base::packed_wscale_t;
    using BlockInfo = typename Base::BlockInfo;
    using EpilogueQuantize =
        typename GEMM::template EpilogueQuantize<false, false, false>;
    using EpilogueLoraDown = typename Lora<Config>::EpilogueLoraDown;

    struct LoadNormalizedAct {
        using matrix_t = half_t[Base::INSN_M][Base::WARP_N + 8];
        static constexpr size_t SHMEM_SIZE = sizeof(matrix_t);

        __device__ __forceinline__ void operator()(
            const half_t* input,
            const half_t* norm_weight,
            const float* row_scales,
            int stride,
            int max_rows,
            int max_cols,
            int global_row_base,
            fpsum_warp& output,
            void* shared) const {
            const int lane_id = static_cast<int>(threadIdx.x) % Base::WARP_SIZE;
            matrix_t& matrix = *reinterpret_cast<matrix_t*>(shared);

            constexpr int kPackSize = Base::WARP_N / Base::WARP_SIZE;
            using packed_input = std::array<half_t, kPackSize>;
            packed_input norm;
            norm.fill(half_t(0));
            const bool column_predicate = lane_id * kPackSize < max_cols;
            if (column_predicate) {
                norm = load(reinterpret_cast<const packed_input*>(
                    norm_weight + lane_id * kPackSize));
            }

#pragma unroll
            for (int m = 0; m < Base::WARP_M_TILES; ++m) {
#pragma unroll
                for (int row = 0; row < Base::INSN_M; ++row) {
                    const int local_row = m * Base::INSN_M + row;
                    const bool row_predicate = local_row < max_rows;
                    packed_input values;
                    values.fill(half_t(0));
                    if (row_predicate && column_predicate) {
                        values = load(reinterpret_cast<const packed_input*>(
                            input + local_row * stride + lane_id * kPackSize));
                    }

                    float row_scale = 0.0F;
                    if (lane_id == 0 && row_predicate) {
                        row_scale = row_scales[global_row_base + local_row];
                    }
                    row_scale = __shfl_sync(0xffffffffU, row_scale, 0);
#pragma unroll
                    for (int index = 0; index < kPackSize; ++index) {
                        const float normalized =
                            (static_cast<float>(values[index]) * row_scale) *
                            static_cast<float>(norm[index]);
                        values[index] = static_cast<half_t>(normalized);
                    }
                    store<true>(
                        reinterpret_cast<packed_input*>(
                            &matrix[row][lane_id * kPackSize]),
                        values);
                }
                __syncwarp();

#pragma unroll
                for (int n = 0; n < Base::WARP_N_TILES; ++n) {
                    const int row = lane_id % 16;
                    const int column = n * Base::INSN_N + lane_id / 16 * 8;
                    uint4 fragment;
                    ldmatrix(&matrix[row][column], fragment);
                    *reinterpret_cast<uint4*>(
                        &output[m * Base::WARP_N_TILES + n]) = fragment;
                }
                __syncwarp();
            }
        }
    };

    static constexpr int MIN_ARCH =
        std::is_same_v<half_t, __nv_bfloat16> ? 800 : 750;
    static constexpr size_t SHMEM_PER_WARP =
        ceilDiv<size_t>(LoadNormalizedAct::SHMEM_SIZE, 128) * 128;
    static constexpr size_t SHMEM_SIZE = SHMEM_PER_WARP * Base::NUM_WARPS;

    struct Arguments {
        const half_t* input;
        const half_t* norm_weight;
        const float* row_scales;
        const packed_wscale_t* smooth_factor;
        packed_act_t* output;
        packed_ascale_t* activation_scales;
        const packed_fpsum_t* lora_weight_down;
        float* lora_activation;
        int lora_rank;
        int padded_m;
        int padded_k;
        int actual_m;
        int actual_k;
        bool always_false;
    };

    __device__ __forceinline__ void operator()(Arguments args) const {
        const BlockInfo block_info{
            .bm = static_cast<int>(blockIdx.x),
            .bn = static_cast<int>(blockIdx.y),
            .numBlocksM = static_cast<int>(gridDim.x),
            .numBlocksN = static_cast<int>(gridDim.y),
        };
        const int warp_id = static_cast<int>(threadIdx.x) / Base::WARP_SIZE;
        const int row_offset =
            block_info.bm * Base::BLOCK_M + warp_id * Base::WARP_M;
        const int column_offset = block_info.bn * Base::BLOCK_N;

        extern __shared__ uint8_t shared[];
        fpsum_warp values;
        LoadNormalizedAct{}(
            args.input + row_offset * args.actual_k + column_offset,
            args.norm_weight + column_offset,
            args.row_scales,
            args.actual_k,
            args.actual_m - row_offset,
            args.actual_k - column_offset,
            row_offset,
            values,
            shared + warp_id * SHMEM_PER_WARP);

        EpilogueLoraDown{}(
            block_info,
            values,
            args.padded_m,
            args.padded_k,
            0,
            typename EpilogueLoraDown::Arguments{
                .lora_wgt_down = args.lora_weight_down,
                .lora_act = args.lora_activation,
                .rank = args.lora_rank,
                .alwaysfalse = args.always_false,
            });
        EpilogueQuantize{}(
            block_info,
            values,
            args.padded_m,
            args.padded_k,
            0,
            typename EpilogueQuantize::Arguments{
                .qout = args.output,
                .oscales = args.activation_scales,
                .shift_value = 0,
                .smooth_factor = args.smooth_factor,
            });
    }
};

template <typename Config>
int launch_xqt_norm_quantize_act_lora(
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
    cudaStream_t stream) {
    using Kernel = XQTNormQuantizeActLoraKernel<Config>;
    using Base = GEMMBase<Config>;
    auto function = invoke_kernel<Kernel, typename Kernel::Arguments>;
    const cudaError_t attribute_status = cudaFuncSetAttribute(
        function,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(Kernel::SHMEM_SIZE));
    if (attribute_status != cudaSuccess) {
        return static_cast<int>(attribute_status);
    }
    function<<<
        dim3(padded_m / Base::BLOCK_M, padded_k / Base::BLOCK_N),
        Base::WARP_SIZE * Base::NUM_WARPS,
        Kernel::SHMEM_SIZE,
        stream>>>(typename Kernel::Arguments{
        .input = static_cast<const typename Base::half_t*>(input),
        .norm_weight = static_cast<const typename Base::half_t*>(norm_weight),
        .row_scales = static_cast<const float*>(row_scales),
        .smooth_factor =
            static_cast<const typename Base::packed_wscale_t*>(smooth),
        .output = static_cast<typename Base::packed_act_t*>(output),
        .activation_scales =
            static_cast<typename Base::packed_ascale_t*>(activation_scales),
        .lora_weight_down =
            static_cast<const typename Base::packed_fpsum_t*>(lora_down),
        .lora_activation = static_cast<float*>(lora_activation),
        .lora_rank = rank,
        .padded_m = padded_m,
        .padded_k = padded_k,
        .actual_m = actual_m,
        .actual_k = actual_k,
        .always_false = false,
    });
    return static_cast<int>(cudaGetLastError());
}

}  // namespace nunchaku::kernels
