#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cstdint>

#include "gemm_w8a8.cuh"
#include "lora.cuh"

namespace {

using namespace nunchaku::kernels;
using GEMM = GEMM_W8A8;
using Config = GEMMConfig_W8A8;
using Base = GEMMBase<Config>;
using LoraImpl = Lora<Config>;

struct QuantizeWeightKernel {
    static constexpr int MIN_ARCH = 800;

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

template <int RankTiles>
struct QuantizeActLoraKernel {
    using LoadAct = typename Base::template load_act_to_fpsum<false>;
    using Unpack = typename Base::unpack_fpsum;
    using LoraDown = typename LoraImpl::EpilogueLoraDown;
    using LoraActWarp = typename LoraImpl::lora_act_warp;
    using LoraWeightWarp = typename LoraImpl::lora_wgt_warp;

    static constexpr int MIN_ARCH = 800;
    static constexpr size_t TILE_BYTES =
        static_cast<size_t>(Base::WARP_M) * Base::BLOCK_N * sizeof(Base::half_t);
    static constexpr size_t SCRATCH_BYTES =
        ceilDiv<size_t>(Unpack::SHMEM_SIZE, 128) * 128;
    static constexpr size_t PER_WARP_BYTES = TILE_BYTES + SCRATCH_BYTES;
    static constexpr size_t SHMEM_SIZE = PER_WARP_BYTES;
    static constexpr size_t LORA_ACT_PER_WARP =
        LoraImpl::LORA_M_TILES * LoraImpl::LORA_R_TILES * 8 *
        Base::WARP_SIZE;
    static constexpr int ACCUMULATED_RANK_TILES = RankTiles > 0 ? RankTiles : 0;

    struct Arguments {
        const Base::half_t* input;
        Base::packed_act_t* output;
        Base::packed_ascale_t* scales;
        const Base::packed_fpsum_t* lora_down;
        float* lora_activation;
        int actual_m;
        int actual_k;
        int padded_m;
        int padded_k;
        int rank;
    };

    __device__ __forceinline__ static void unpack_to_shared(
        Base::fpsum_warp values,
        Base::half_t* output,
        void* scratch,
        float* maximum) {
        using matrix_t = Base::half_t[8][Base::WARP_N + 8];
        constexpr int PACK_SIZE = Base::WARP_N / Base::WARP_SIZE;
        using pack_t = std::array<Base::half_t, PACK_SIZE>;
        matrix_t& matrix = *reinterpret_cast<matrix_t*>(scratch);
        const int lane_id = static_cast<int>(threadIdx.x) % Base::WARP_SIZE;

#pragma unroll
        for (int tile_m = 0; tile_m < Base::WARP_M_TILES; ++tile_m) {
#pragma unroll
            for (int tile_n = 0; tile_n < Base::WARP_N_TILES; ++tile_n) {
                Base::packed_fpsum_t& fragment =
                    values[tile_m * Base::WARP_N_TILES + tile_n];
                const int row = lane_id / 4;
                const int col = lane_id % 4 * 2 + tile_n * Base::INSN_N;
                *reinterpret_cast<Base::half2_t*>(&matrix[row][col]) =
                    fragment.data[0];
                *reinterpret_cast<Base::half2_t*>(&matrix[row][col + 8]) =
                    fragment.data[2];
            }
            __syncwarp();

#pragma unroll
            for (int row = 0; row < 8; ++row) {
                pack_t pack = *reinterpret_cast<pack_t*>(
                    &matrix[row][lane_id * PACK_SIZE]);
                if (maximum != nullptr) {
                    float local_max = 0.0F;
#pragma unroll
                    for (int index = 0; index < PACK_SIZE; ++index) {
                        local_max = fmaxf(
                            local_max,
                            fabsf(static_cast<float>(pack[index])));
                    }
#pragma unroll
                    for (int mask = Base::WARP_SIZE / 2; mask > 0; mask /= 2) {
                        local_max = fmaxf(
                            local_max,
                            __shfl_xor_sync(0xffffffffU, local_max, mask));
                    }
                    if (lane_id == 0) {
                        maximum[tile_m * Base::INSN_M + row] = fmaxf(
                            maximum[tile_m * Base::INSN_M + row],
                            local_max);
                    }
                }
                store<true>(
                    reinterpret_cast<pack_t*>(
                        &output[(tile_m * Base::INSN_M + row) * Base::BLOCK_N +
                                lane_id * PACK_SIZE]),
                    pack);
            }
            __syncwarp();

#pragma unroll
            for (int tile_n = 0; tile_n < Base::WARP_N_TILES; ++tile_n) {
                Base::packed_fpsum_t& fragment =
                    values[tile_m * Base::WARP_N_TILES + tile_n];
                const int row = lane_id / 4;
                const int col = lane_id % 4 * 2 + tile_n * Base::INSN_N;
                *reinterpret_cast<Base::half2_t*>(&matrix[row][col]) =
                    fragment.data[1];
                *reinterpret_cast<Base::half2_t*>(&matrix[row][col + 8]) =
                    fragment.data[3];
            }
            __syncwarp();

#pragma unroll
            for (int row = 0; row < 8; ++row) {
                pack_t pack = *reinterpret_cast<pack_t*>(
                    &matrix[row][lane_id * PACK_SIZE]);
                const int output_row = tile_m * Base::INSN_M + 8 + row;
                if (maximum != nullptr) {
                    float local_max = 0.0F;
#pragma unroll
                    for (int index = 0; index < PACK_SIZE; ++index) {
                        local_max = fmaxf(
                            local_max,
                            fabsf(static_cast<float>(pack[index])));
                    }
#pragma unroll
                    for (int mask = Base::WARP_SIZE / 2; mask > 0; mask /= 2) {
                        local_max = fmaxf(
                            local_max,
                            __shfl_xor_sync(0xffffffffU, local_max, mask));
                    }
                    if (lane_id == 0) {
                        maximum[output_row] =
                            fmaxf(maximum[output_row], local_max);
                    }
                }
                store<true>(
                    reinterpret_cast<pack_t*>(
                        &output[output_row * Base::BLOCK_N +
                                lane_id * PACK_SIZE]),
                    pack);
            }
            __syncwarp();
        }
    }

    __device__ __forceinline__ static void accumulate_lora_down(
        Base::fpsum_warp values,
        const Base::packed_fpsum_t* weight,
        int rank,
        std::array<LoraActWarp, ACCUMULATED_RANK_TILES>& accumulators) {
        if constexpr (RankTiles > 0) {
#pragma unroll
            for (int rank_tile = 0; rank_tile < RankTiles; ++rank_tile) {
                LoraWeightWarp lora_weight;
                LoraImpl::load_lora_wgt(
                    weight,
                    rank_tile,
                    rank,
                    lora_weight,
                    true);
#pragma unroll
                for (int tile_m = 0; tile_m < LoraImpl::LORA_M_TILES;
                     ++tile_m) {
#pragma unroll
                    for (int tile_n = 0; tile_n < LoraImpl::LORA_N_TILES;
                         ++tile_n) {
                        accumulators[rank_tile][tile_m] = Base::mma_f16xf16_f32(
                            values[tile_m * Base::WARP_N_TILES + tile_n],
                            lora_weight[tile_n],
                            accumulators[rank_tile][tile_m]);
                    }
                }
            }
        }
    }

    __device__ __forceinline__ static void store_lora_down(
        const Arguments& args,
        int block_m,
        int block_warp,
        int lane_id,
        const std::array<LoraActWarp, ACCUMULATED_RANK_TILES>& accumulators) {
        if constexpr (RankTiles > 0) {
            const size_t block_stride =
                static_cast<size_t>(args.rank / LoraImpl::WARP_R) *
                Base::NUM_WARPS * LORA_ACT_PER_WARP;
            float* block_base = args.lora_activation + block_m * block_stride;
#pragma unroll
            for (int rank_tile = 0; rank_tile < RankTiles; ++rank_tile) {
                float* warp_base = block_base +
                    (rank_tile * Base::NUM_WARPS + block_warp) *
                        LORA_ACT_PER_WARP +
                    lane_id;
#pragma unroll
                for (int tile_m = 0; tile_m < LoraImpl::LORA_M_TILES;
                     ++tile_m) {
#pragma unroll
                    for (int element = 0; element < 8; ++element) {
                        warp_base[(tile_m * 8 + element) * Base::WARP_SIZE] =
                            accumulators[rank_tile][tile_m].data[element];
                    }
                }
            }
        }
    }

    __device__ void operator()(Arguments args) {
        const int lane_id = static_cast<int>(threadIdx.x) % Base::WARP_SIZE;
        const int global_warp_row = static_cast<int>(blockIdx.x);
        const int block_m = global_warp_row / Base::NUM_WARPS;
        const int block_warp = global_warp_row % Base::NUM_WARPS;
        const int row_base = global_warp_row * Base::WARP_M;
        const int k_blocks = args.padded_k / Base::BLOCK_N;

        __shared__ float row_max[Base::WARP_M];
        __shared__ Base::half_t row_scale[Base::WARP_M];
        extern __shared__ __align__(128) uint8_t shared[];
        Base::half_t* tile = reinterpret_cast<Base::half_t*>(shared);
        void* scratch = shared + TILE_BYTES;

        for (int index = lane_id; index < Base::WARP_M;
             index += Base::WARP_SIZE) {
            row_max[index] = 0.0F;
        }
        __syncwarp();

        std::array<LoraActWarp, ACCUMULATED_RANK_TILES> lora_accumulators;
        if constexpr (RankTiles > 0) {
#pragma unroll
            for (int rank_tile = 0; rank_tile < RankTiles; ++rank_tile) {
                lora_accumulators[rank_tile].fill(
                    Base::packed_f32psum_t::zeros());
            }
        }

        for (int block_k = 0; block_k < k_blocks; ++block_k) {
            const int col_base = block_k * Base::BLOCK_N;
            Base::fpsum_warp values;
            LoadAct{}(
                args.input + row_base * args.actual_k + col_base,
                args.actual_k,
                args.actual_m - row_base,
                args.actual_k - col_base,
                values,
                tile);

            const typename Base::BlockInfo block_info{
                .bm = block_m,
                .bn = block_k,
                .numBlocksM = args.padded_m / Base::BLOCK_M,
                .numBlocksN = k_blocks,
            };
            if constexpr (RankTiles > 0) {
                const Base::packed_fpsum_t* weight = args.lora_down +
                    block_k * (Base::BLOCK_N / Base::INSN_N) *
                        (args.rank / LoraImpl::WARP_R) * Base::WARP_SIZE;
                accumulate_lora_down(
                    values,
                    weight,
                    args.rank,
                    lora_accumulators);
            } else if constexpr (RankTiles < 0) {
                LoraDown{}(
                    block_info,
                    values,
                    args.padded_m,
                    args.padded_k,
                    0,
                    typename LoraDown::Arguments{
                        .lora_wgt_down = args.lora_down,
                        .lora_act = args.lora_activation +
                            block_warp * LORA_ACT_PER_WARP,
                        .rank = args.rank,
                        .alwaysfalse = false,
                    });
            }

            unpack_to_shared(
                values,
                tile,
                scratch,
                row_max);
        }

        store_lora_down(
            args,
            block_m,
            block_warp,
            lane_id,
            lora_accumulators);

        for (int row = lane_id; row < Base::WARP_M; row += Base::WARP_SIZE) {
            const float maximum = row_max[row];
            row_scale[row] = static_cast<Base::half_t>(
                maximum > 0.0F ? maximum / 127.0F : 1.0F);
        }
        __syncwarp();

        for (int block_k = 0; block_k < k_blocks; ++block_k) {
            const int col_base = block_k * Base::BLOCK_N;
            Base::fpsum_warp values;
            LoadAct{}(
                args.input + row_base * args.actual_k + col_base,
                args.actual_k,
                args.actual_m - row_base,
                args.actual_k - col_base,
                values,
                tile);
            unpack_to_shared(values, tile, scratch, nullptr);

            for (int tile_m = 0; tile_m < Base::WARP_M_TILES; ++tile_m) {
                for (int tile_k = 0;
                     tile_k < Base::BLOCK_N / Base::WARP_K;
                     ++tile_k) {
                    Base::packed_act_t quantized;
                    GEMM::template quantize_w8a8_warp<true>(
                        tile + tile_m * Base::INSN_M * Base::BLOCK_N +
                            tile_k * Base::WARP_K,
                        row_scale + tile_m * Base::INSN_M,
                        Base::BLOCK_N,
                        quantized,
                        scratch);
                    const int global_k =
                        block_k * (Base::BLOCK_N / Base::WARP_K) + tile_k;
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
            }
        }

        Base::pack_ascales(
            row_scale,
            &args.scales[
                (block_m * Base::NUM_WARPS + block_warp) *
                Base::ASCALES_NUM_PACKS * Base::ASCALES_VALID_LANES]);
    }
};

template <bool FuseGlu>
struct QuantizeActPlainKernel {
    static constexpr int MIN_ARCH = 800;

    static constexpr size_t smem_size() {
        if constexpr (!FuseGlu) {
            return 0;
        }
        return Base::INSN_M * (Base::WARP_N / 2) * sizeof(Base::half_t);
    }

    __device__ void operator()(
        const Base::half_t* input,
        Base::packed_act_t* output,
        Base::packed_ascale_t* oscales,
        int K,
        bool alwaysfalse) {
        const int lane_id = static_cast<int>(threadIdx.x) % Base::WARP_SIZE;
        const int warp_id = static_cast<int>(threadIdx.x) / Base::WARP_SIZE;
        const int num_warps = static_cast<int>(blockDim.x) / Base::WARP_SIZE;
        const int global_warp_row = static_cast<int>(blockIdx.x);
        const int block_m = global_warp_row / (Base::BLOCK_M / Base::WARP_M);
        const int gemm_warp_id = global_warp_row % (Base::BLOCK_M / Base::WARP_M);

        __shared__ alignas(128) Base::half_t oscale_shmem[Base::WARP_M];
        __shared__ alignas(128) uint8_t tmp_shmem[Base::NUM_WARPS][512];

        const int K2 = FuseGlu ? K / 2 : K;
        extern __shared__ uint8_t smem[];
        Base::half_t* shmem = reinterpret_cast<Base::half_t*>(smem);

        for (int tile_m = 0; tile_m < Base::WARP_M_TILES; tile_m++) {
            for (int index = warp_id; index < Base::INSN_M; index += num_warps) {
                const int row_local = tile_m * Base::INSN_M + index;
                const int row_global = global_warp_row * Base::WARP_M + row_local;
                const Base::half_t max_value =
                    GEMM::template findmax_warp<FuseGlu>(
                        input + row_global * K,
                        shmem + index * K2,
                        K,
                        alwaysfalse);
                oscale_shmem[row_local] = max_value / Base::half_t(127);
            }
            __syncthreads();

            for (int block_k = warp_id; block_k < K2 / Base::WARP_K; block_k += num_warps) {
                const int row_local = tile_m * Base::INSN_M;
                const int row_global = global_warp_row * Base::WARP_M + row_local;
                const int col = block_k * Base::WARP_K;
                Base::packed_act_t quantized;
                GEMM::template quantize_w8a8_warp<FuseGlu>(
                    FuseGlu ? shmem + col : input + row_global * K + col,
                    oscale_shmem + row_local,
                    FuseGlu ? K2 : K,
                    quantized,
                    &tmp_shmem[warp_id]);
                const int output_index =
                    (((block_m * (K2 / Base::WARP_K) + block_k) *
                          Base::NUM_WARPS +
                      gemm_warp_id) *
                         Base::WARP_M_TILES +
                     tile_m) *
                        Base::WARP_SIZE +
                    lane_id;
                store(&output[output_index], quantized);
            }
            __syncthreads();
        }

        GEMM::pack_ascales(
            oscale_shmem,
            &oscales[
                (block_m * Base::NUM_WARPS + gemm_warp_id) *
                Base::ASCALES_NUM_PACKS * Base::ASCALES_VALID_LANES]);
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
    function<<<dim3(n / Base::BLOCK_N, k / Base::WARP_K), Base::WARP_SIZE, 0, stream>>>(
        static_cast<const Base::half_t*>(input),
        static_cast<const Base::half_t*>(scales),
        static_cast<Base::packed_wgt_t*>(output),
        k);
    return static_cast<int>(cudaGetLastError());
}

template <int RankTiles>
int launch_quantize_act_lora_specialized(
    const void* input,
    void* output,
    void* scales,
    const void* lora_down,
    void* lora_activation,
    int actual_m,
    int actual_k,
    int padded_m,
    int padded_k,
    int rank,
    cudaStream_t stream) {
    using Kernel = QuantizeActLoraKernel<RankTiles>;
    auto function = invoke_kernel<Kernel, typename Kernel::Arguments>;
    const cudaError_t attribute_status = cudaFuncSetAttribute(
        function,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(Kernel::SHMEM_SIZE));
    if (attribute_status != cudaSuccess) {
        return static_cast<int>(attribute_status);
    }
    function<<<
        dim3(padded_m / Base::WARP_M),
        Base::WARP_SIZE,
        Kernel::SHMEM_SIZE,
        stream>>>(typename Kernel::Arguments{
        .input = static_cast<const Base::half_t*>(input),
        .output = static_cast<Base::packed_act_t*>(output),
        .scales = static_cast<Base::packed_ascale_t*>(scales),
        .lora_down = static_cast<const Base::packed_fpsum_t*>(lora_down),
        .lora_activation = static_cast<float*>(lora_activation),
        .actual_m = actual_m,
        .actual_k = actual_k,
        .padded_m = padded_m,
        .padded_k = padded_k,
        .rank = rank,
    });
    return static_cast<int>(cudaGetLastError());
}

int launch_quantize_act_fast(
    const void* input,
    void* output,
    void* scales,
    int padded_m,
    int padded_k,
    cudaStream_t stream) {
    using Kernel = QuantizeActPlainKernel<false>;
    auto function = invoke_kernel<
        Kernel,
        const Base::half_t*,
        Base::packed_act_t*,
        Base::packed_ascale_t*,
        int,
        bool>;
    function<<<
        dim3(padded_m / Base::WARP_M),
        Base::WARP_SIZE * Base::NUM_WARPS,
        Kernel::smem_size(),
        stream>>>(
        static_cast<const Base::half_t*>(input),
        static_cast<Base::packed_act_t*>(output),
        static_cast<Base::packed_ascale_t*>(scales),
        padded_k,
        false);
    return static_cast<int>(cudaGetLastError());
}

int launch_quantize_act_lora(
    const void* input,
    void* output,
    void* scales,
    const void* lora_down,
    void* lora_activation,
    int actual_m,
    int actual_k,
    int padded_m,
    int padded_k,
    int rank,
    cudaStream_t stream) {
    switch (rank) {
        case 0:
            return launch_quantize_act_lora_specialized<0>(
                input, output, scales, lora_down, lora_activation, actual_m,
                actual_k, padded_m, padded_k, rank, stream);
        case 16:
            return launch_quantize_act_lora_specialized<1>(
                input, output, scales, lora_down, lora_activation, actual_m,
                actual_k, padded_m, padded_k, rank, stream);
        case 32:
            return launch_quantize_act_lora_specialized<2>(
                input, output, scales, lora_down, lora_activation, actual_m,
                actual_k, padded_m, padded_k, rank, stream);
        case 48:
            return launch_quantize_act_lora_specialized<3>(
                input, output, scales, lora_down, lora_activation, actual_m,
                actual_k, padded_m, padded_k, rank, stream);
        case 64:
            return launch_quantize_act_lora_specialized<4>(
                input, output, scales, lora_down, lora_activation, actual_m,
                actual_k, padded_m, padded_k, rank, stream);
        default:
            return launch_quantize_act_lora_specialized<-1>(
                input, output, scales, lora_down, lora_activation, actual_m,
                actual_k, padded_m, padded_k, rank, stream);
    }
}

int launch_gemm_lora(
    const void* activation,
    const void* weight,
    void* output,
    const void* activation_scales,
    const void* weight_scales,
    const void* lora_activation,
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
    using Bias = typename Base::template EpilogueBias<true, false>;
    using Default = typename Base::EpilogueDefault;
    using Nop = typename Base::EpilogueNop;
    using LoraUp = typename LoraImpl::EpilogueLoraUp;
    using LoraChain = typename Base::template EpilogueCombination<
        LoraUp,
        Nop,
        Default,
        Nop>;
    using Epilogue = typename Base::template EpilogueCombination<
        Bias,
        LoraChain,
        Nop>;

    typename LoraImpl::scale_t lora_scales{};
    const int scale_count =
        std::min(rank / LoraImpl::WARP_R, static_cast<int>(lora_scales.size()));
    for (int index = 0; index < scale_count; ++index) {
        lora_scales[static_cast<size_t>(index)] = lora_scale;
    }

    const typename Bias::Arguments bias_args{
        .bias = static_cast<const Base::packed_wscale_t*>(bias),
        .scale = nullptr,
    };
    const typename LoraUp::Arguments lora_args{
        .lora_act = static_cast<const float*>(lora_activation),
        .lora_wgt_up = static_cast<const Base::packed_fpsum_t*>(lora_up),
        .rank = rank,
        .scales = lora_scales,
        .alwaysfalse = false,
    };
    const typename Default::Arguments output_args{
        .out = static_cast<Base::half_t*>(output),
        .actualM = actual_m,
        .actualN = actual_n,
    };
    const typename Epilogue::Arguments epilogue_args{
        bias_args,
        typename LoraChain::Arguments{
            lora_args,
            typename Nop::Arguments{},
            output_args,
            typename Nop::Arguments{},
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

extern "C" int xqt_svdq_w8a8_quantize_weight(
    const void* input,
    const void* scales,
    void* output,
    int n,
    int k,
    cudaStream_t stream) {
    return launch_quantize_weight(input, scales, output, n, k, stream);
}

extern "C" int xqt_svdq_w8a8_quantize_act_lora(
    const void* input,
    void* output,
    void* scales,
    const void* lora_down,
    void* lora_activation,
    int actual_m,
    int actual_k,
    int padded_m,
    int padded_k,
    int rank,
    cudaStream_t stream) {
    return launch_quantize_act_lora(
        input,
        output,
        scales,
        lora_down,
        lora_activation,
        actual_m,
        actual_k,
        padded_m,
        padded_k,
        rank,
        stream);
}

extern "C" int xqt_svdq_w8a8_quantize_act(
    const void* input,
    void* output,
    void* scales,
    int actual_m,
    int actual_k,
    int padded_m,
    int padded_k,
    cudaStream_t stream) {
    return launch_quantize_act_lora(
        input,
        output,
        scales,
        nullptr,
        nullptr,
        actual_m,
        actual_k,
        padded_m,
        padded_k,
        0,
        stream);
}

extern "C" int xqt_svdq_w8a8_quantize_act_fast(
    const void* input,
    void* output,
    void* scales,
    int padded_m,
    int padded_k,
    cudaStream_t stream) {
    return launch_quantize_act_fast(
        input,
        output,
        scales,
        padded_m,
        padded_k,
        stream);
}

extern "C" int xqt_svdq_w8a8_gemm_lora(
    const void* activation,
    const void* weight,
    void* output,
    const void* activation_scales,
    const void* weight_scales,
    const void* lora_activation,
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
    return launch_gemm_lora(
        activation,
        weight,
        output,
        activation_scales,
        weight_scales,
        lora_activation,
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
}

extern "C" int xqt_svdq_w8a8_gemm(
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

extern "C" const char* xqt_svdq_w8a8_version() {
    return "nunchaku_svdq_w8a8_sm89_v1";
}
