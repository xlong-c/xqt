// Small-BLOCK_N (64) W4A4 GEMM variant for Ada sm_89.
//
// Motivation: the upstream Nunchaku GEMMConfig_W4A4 fixes BLOCK_M=256 and
// BLOCK_N=128.  For short-prefill shapes (e.g. M<=256, N=1024) the GEMM grid
// is only (1, N/128) = 8 CTAs with WARP_N_TILES=8 per warp, so both SM
// utilization and per-warp latency are poor.  This variant keeps BLOCK_M,
// NUM_WARPS and all K-side geometry identical (so the packed activation and
// packed LoRA layouts are shared with the base extension) and only halves
// BLOCK_N/WARP_N to 64: twice as many CTAs, half the per-warp MMA work and
// half the accumulator register pressure.  Weights must be repacked with the
// matching WARP_N=64 fragment layout via xqt_svdq_w4a4_smalln_quantize_weight.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cstdint>

#include "gemm_w4a4.cuh"
#include "lora.cuh"

namespace {

using namespace nunchaku::kernels;

template <bool bf16>
class GEMMConfig_W4A4_BN64 {
public:
    // Identical to GEMMConfig_W4A4 except BLOCK_N/WARP_N = 64.
    static constexpr int BLOCK_M   = 256;
    static constexpr int BLOCK_N   = 64;
    static constexpr int WARP_SIZE = 32;
    static constexpr int NUM_WARPS = 8;

    static constexpr int INSN_M = 16;
    static constexpr int INSN_N = 16;
    static constexpr int INSN_K = 64;

    static constexpr bool FASTER_I2F = false;

    using half_t  = typename std::conditional_t<bf16, __nv_bfloat16, half>;
    using half2_t = typename std::conditional_t<bf16, __nv_bfloat162, half2>;
};

using GEMMConfig_W4A4_FP16_BN64 = GEMMConfig_W4A4_BN64<false>;
using GEMMConfig_W4A4_BF16_BN64 = GEMMConfig_W4A4_BN64<true>;

enum class ScalarKind : int {
    FP16 = 0,
    BF16 = 1,
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

template <typename Config, bool UseLora>
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

template <typename Function>
int dispatch_scalar(int scalar_kind, Function&& function) {
    switch (static_cast<ScalarKind>(scalar_kind)) {
        case ScalarKind::FP16:
            return function.template operator()<GEMMConfig_W4A4_FP16_BN64>();
        case ScalarKind::BF16:
            return function.template operator()<GEMMConfig_W4A4_BF16_BN64>();
    }
    return static_cast<int>(cudaErrorInvalidValue);
}

}  // namespace

extern "C" int xqt_svdq_w4a4_smalln_quantize_weight(
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

extern "C" int xqt_svdq_w4a4_smalln_quantize_act_lora(
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
    // The activation quantization plus LoRA-down stage is BLOCK_N-independent,
    // so it runs the upstream GEMMConfig_W4A4 instantiation: its packed
    // activation and lora_act layouts are shared with the BN64 GEMM.
    switch (static_cast<ScalarKind>(scalar_kind)) {
        case ScalarKind::FP16:
            return launch_quantize_act_lora<GEMMConfig_W4A4_FP16>(
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
        case ScalarKind::BF16:
            return launch_quantize_act_lora<GEMMConfig_W4A4_BF16>(
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
    }
    return static_cast<int>(cudaErrorInvalidValue);
}

extern "C" int xqt_svdq_w4a4_smalln_gemm(
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

extern "C" int xqt_svdq_w4a4_smalln_gemm_lora(
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

extern "C" const char* xqt_svdq_w4a4_smalln_version() {
    return "xqt_w4a4_sm89_smalln_bn64_v2";
}
