// SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// The warp-FHT structure and CUTLASS epilogue contract are derived from
// Comfy-Kitchen and ComfyUI-AnimaTurbo, both Apache-2.0 licensed. This version
// is reduced to XQT's sm_89 FP16/BF16 rowwise W4A4 inference contract.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/default_gemm_universal_with_visitor.h>
#include <cutlass/epilogue/threadblock/fusion/visitors.hpp>

#include <algorithm>
#include <cfloat>
#include <cstdint>
#include <map>
#include <mutex>
#include <tuple>
#include <type_traits>

namespace {

constexpr int kWarpSize = 32;
constexpr int kConvRotGroup = 256;

template <typename T>
__device__ __forceinline__ float to_float(T value);

template <>
__device__ __forceinline__ float to_float(__half value) {
    return __half2float(value);
}

template <>
__device__ __forceinline__ float to_float(__nv_bfloat16 value) {
    return __bfloat162float(value);
}

template <typename T>
__device__ __forceinline__ T from_float(float value);

template <>
__device__ __forceinline__ __half from_float(float value) {
    return __float2half_rn(value);
}

template <>
__device__ __forceinline__ __nv_bfloat16 from_float(float value) {
    return __float2bfloat16_rn(value);
}

template <typename T>
__device__ __forceinline__ float finite_max();

template <>
__device__ __forceinline__ float finite_max<__half>() {
    return 65504.0F;
}

template <>
__device__ __forceinline__ float finite_max<__nv_bfloat16>() {
    return 3.38953139e38F;
}

template <typename T>
struct PairedType;

template <>
struct PairedType<__half> {
    using type = __half2;
};

template <>
struct PairedType<__nv_bfloat16> {
    using type = __nv_bfloat162;
};

__device__ __forceinline__ __half low_half(__half2 value) {
    return __low2half(value);
}

__device__ __forceinline__ __half high_half(__half2 value) {
    return __high2half(value);
}

__device__ __forceinline__ __nv_bfloat16 low_half(__nv_bfloat162 value) {
    return __low2bfloat16(value);
}

__device__ __forceinline__ __nv_bfloat16 high_half(__nv_bfloat162 value) {
    return __high2bfloat16(value);
}

__device__ __forceinline__ __half2 make_pair(__half low, __half high) {
    return __halves2half2(low, high);
}

__device__ __forceinline__ __nv_bfloat162 make_pair(
    __nv_bfloat16 low,
    __nv_bfloat16 high) {
    return __halves2bfloat162(low, high);
}

template <typename PackedT>
__device__ __forceinline__ PackedT h4_combine(
    int digit,
    PackedT own,
    PackedT partner1,
    PackedT partner2,
    PackedT partner3,
    PackedT half_value) {
    const PackedT pair01 = __hadd2(own, partner1);
    const PackedT diff_even = __hsub2(partner2, partner3);
    const PackedT diff_odd = __hsub2(partner3, partner2);
    const PackedT even = __hmul2(__hadd2(pair01, diff_even), half_value);
    const PackedT odd = __hmul2(__hsub2(pair01, diff_odd), half_value);
    return (digit & 1) != 0 ? odd : even;
}

template <typename PackedT, int PairsPerWarp>
__device__ __forceinline__ void regular_hadamard_256(
    PackedT values[PairsPerWarp][8],
    int lane,
    PackedT half_value) {
#pragma unroll
    for (int group = 0; group < PairsPerWarp; ++group) {
#pragma unroll
        for (int slot = 0; slot < 8; ++slot) {
            const int digit = lane & 3;
            const PackedT b1 = __shfl_xor_sync(0xffffffffU, values[group][slot], 1);
            const PackedT b2 = __shfl_xor_sync(0xffffffffU, values[group][slot], 2);
            const PackedT b3 = __shfl_xor_sync(0xffffffffU, values[group][slot], 3);
            values[group][slot] = h4_combine(
                digit,
                values[group][slot],
                b1,
                b2,
                b3,
                half_value);
        }
    }

#pragma unroll
    for (int group = 0; group < PairsPerWarp; ++group) {
#pragma unroll
        for (int slot = 0; slot < 8; ++slot) {
            const int digit = (lane >> 2) & 3;
            const PackedT b1 = __shfl_xor_sync(0xffffffffU, values[group][slot], 4);
            const PackedT b2 = __shfl_xor_sync(0xffffffffU, values[group][slot], 8);
            const PackedT b3 = __shfl_xor_sync(0xffffffffU, values[group][slot], 12);
            values[group][slot] = h4_combine(
                digit,
                values[group][slot],
                b1,
                b2,
                b3,
                half_value);
        }
    }

#pragma unroll
    for (int group = 0; group < PairsPerWarp; ++group) {
#pragma unroll
        for (int slot = 0; slot < 8; slot += 2) {
            const PackedT low = values[group][slot];
            const PackedT high = values[group][slot + 1];
            const int digit = (lane >> 4) & 1;
            const PackedT low_partner = __shfl_xor_sync(0xffffffffU, low, 16);
            const PackedT high_partner = __shfl_xor_sync(0xffffffffU, high, 16);
            values[group][slot] = h4_combine(
                digit,
                low,
                low_partner,
                high,
                high_partner,
                half_value);
            values[group][slot + 1] = h4_combine(
                digit | 2,
                high,
                high_partner,
                low,
                low_partner,
                half_value);
        }
    }

#pragma unroll
    for (int group = 0; group < PairsPerWarp; ++group) {
#pragma unroll
        for (int base = 0; base < 2; ++base) {
            const PackedT value0 = values[group][base];
            const PackedT value2 = values[group][base + 2];
            const PackedT value4 = values[group][base + 4];
            const PackedT value6 = values[group][base + 6];
            values[group][base] = h4_combine(
                0, value0, value2, value4, value6, half_value);
            values[group][base + 2] = h4_combine(
                1, value2, value0, value6, value4, half_value);
            values[group][base + 4] = h4_combine(
                2, value4, value6, value0, value2, half_value);
            values[group][base + 6] = h4_combine(
                3, value6, value4, value2, value0, half_value);
        }
    }
}

__device__ __forceinline__ int8_t pack_int4(int low, int high) {
    const uint32_t packed =
        (static_cast<uint32_t>(low) & 0x0FU) |
        ((static_cast<uint32_t>(high) & 0x0FU) << 4);
    return static_cast<int8_t>(packed);
}

template <typename InputType, int PairsPerWarp, int RowsPerBlock>
__global__ void quantize_convrot_rowwise_kernel(
    const InputType* __restrict__ input,
    int8_t* __restrict__ output,
    float* __restrict__ scales,
    int rows,
    int k) {
    using PackedT = typename PairedType<InputType>::type;
    constexpr int GroupsPerWarp = 2 * PairsPerWarp;
    extern __shared__ float row_warp_max[];

    const int warps_per_row = k / (GroupsPerWarp * kConvRotGroup);
    const int lane = static_cast<int>(threadIdx.x) & (kWarpSize - 1);
    const int warp_id = static_cast<int>(threadIdx.x) / kWarpSize;
    const int local_row = warp_id / warps_per_row;
    const int warp_in_row = warp_id % warps_per_row;
    const int row = static_cast<int>(blockIdx.x) * RowsPerBlock + local_row;
    const bool active = row < rows;
    const int64_t row_offset = static_cast<int64_t>(row) * k;
    const int64_t packed_row_offset = static_cast<int64_t>(row) * (k / 2);
    const int group_base = warp_in_row * GroupsPerWarp;

    const InputType zero = from_float<InputType>(0.0F);
    const InputType half = from_float<InputType>(0.5F);
    const PackedT half2 = make_pair(half, half);
    PackedT values[PairsPerWarp][8];

#pragma unroll
    for (int pair = 0; pair < PairsPerWarp; ++pair) {
        const int low_col = (group_base + 2 * pair) * kConvRotGroup;
        const int high_col = low_col + kConvRotGroup;
#pragma unroll
        for (int slot = 0; slot < 8; ++slot) {
            const int offset = lane + kWarpSize * slot;
            const InputType low = active ? input[row_offset + low_col + offset] : zero;
            const InputType high = active ? input[row_offset + high_col + offset] : zero;
            values[pair][slot] = make_pair(low, high);
        }
    }

    regular_hadamard_256<PackedT, PairsPerWarp>(values, lane, half2);

    float local_max = 0.0F;
#pragma unroll
    for (int pair = 0; pair < PairsPerWarp; ++pair) {
#pragma unroll
        for (int slot = 0; slot < 8; ++slot) {
            local_max = fmaxf(local_max, fabsf(to_float(low_half(values[pair][slot]))));
            local_max = fmaxf(local_max, fabsf(to_float(high_half(values[pair][slot]))));
        }
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_max = fmaxf(
            local_max,
            __shfl_xor_sync(0xffffffffU, local_max, offset));
    }

    float absmax = local_max;
    if (warps_per_row > 1) {
        row_warp_max[local_row * warps_per_row + warp_in_row] = local_max;
        __syncthreads();
        absmax = 0.0F;
        for (int warp = 0; warp < warps_per_row; ++warp) {
            absmax = fmaxf(absmax, row_warp_max[local_row * warps_per_row + warp]);
        }
    }

    const float scale = fmaxf(
        fminf(absmax, finite_max<InputType>()) * (1.0F / 7.0F),
        1.0e-10F);
    if (active && lane == 0 && warp_in_row == 0) {
        scales[row] = scale;
    }
    const float inverse_scale = 1.0F / scale;

    if (!active) {
        return;
    }
#pragma unroll
    for (int pair = 0; pair < PairsPerWarp; ++pair) {
        const int low_group = group_base + 2 * pair;
        const int high_group = low_group + 1;
        const int64_t low_base = packed_row_offset + low_group * (kConvRotGroup / 2);
        const int64_t high_base = packed_row_offset + high_group * (kConvRotGroup / 2);
#pragma unroll
        for (int slot = 0; slot < 8; ++slot) {
            int low = __float2int_rn(to_float(low_half(values[pair][slot])) * inverse_scale);
            low = min(7, max(-7, low));
            const int low_partner = __shfl_xor_sync(0xffffffffU, low, 1);
            if ((lane & 1) == 0) {
                output[low_base + (lane >> 1) + 16 * slot] = pack_int4(low, low_partner);
            }

            int high = __float2int_rn(to_float(high_half(values[pair][slot])) * inverse_scale);
            high = min(7, max(-7, high));
            const int high_partner = __shfl_xor_sync(0xffffffffU, high, 1);
            if ((lane & 1) == 0) {
                output[high_base + (lane >> 1) + 16 * slot] = pack_int4(high, high_partner);
            }
        }
    }
}

template <typename InputType, int PairsPerWarp>
void launch_quantize_rows(
    int rows_per_block,
    int blocks,
    int threads,
    size_t shared_bytes,
    const InputType* input,
    int8_t* output,
    float* scales,
    int rows,
    int k,
    cudaStream_t stream) {
    if (rows_per_block == 4) {
        quantize_convrot_rowwise_kernel<InputType, PairsPerWarp, 4>
            <<<blocks, threads, shared_bytes, stream>>>(input, output, scales, rows, k);
    } else if (rows_per_block == 2) {
        quantize_convrot_rowwise_kernel<InputType, PairsPerWarp, 2>
            <<<blocks, threads, shared_bytes, stream>>>(input, output, scales, rows, k);
    } else {
        quantize_convrot_rowwise_kernel<InputType, PairsPerWarp, 1>
            <<<blocks, threads, shared_bytes, stream>>>(input, output, scales, rows, k);
    }
}

template <typename ElementOutput, int TBM, int TBN, int TBK, int WM, int WN, int WK, int Stages>
struct FusedInt4Gemm {
    using ElementA = cutlass::int4b_t;
    using ElementB = cutlass::int4b_t;
    using ElementC = ElementOutput;
    using ElementAccumulator = int32_t;
    using ElementCompute = float;
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutC = cutlass::layout::RowMajor;
    static constexpr int AlignA = 32;
    static constexpr int AlignB = 32;
    static constexpr int AlignC = 128 / cutlass::sizeof_bits<ElementC>::value;
    using ThreadblockShape = cutlass::gemm::GemmShape<TBM, TBN, TBK>;
    using WarpShape = cutlass::gemm::GemmShape<WM, WN, WK>;
    using InstructionShape = cutlass::gemm::GemmShape<16, 8, 64>;
    static constexpr int EvtStages = 1;

    using ThreadMap = cutlass::epilogue::threadblock::OutputTileThreadLayout<
        ThreadblockShape,
        WarpShape,
        ElementC,
        AlignC,
        EvtStages>;
    using Accumulator = cutlass::epilogue::threadblock::VisitorAccFetch;
    using ActivationScale = cutlass::epilogue::threadblock::VisitorColBroadcast<
        ThreadMap,
        ElementCompute,
        cute::Stride<cute::_1, cute::_0, int32_t>>;
    using WeightScale = cutlass::epilogue::threadblock::VisitorRowBroadcast<
        ThreadMap,
        ElementCompute,
        cute::Stride<cute::_0, cute::_1, int32_t>>;
    using Bias = cutlass::epilogue::threadblock::VisitorRowBroadcast<
        ThreadMap,
        ElementCompute,
        cute::Stride<cute::_0, cute::_1, int32_t>>;
    using Multiply0 = cutlass::epilogue::threadblock::VisitorCompute<
        cutlass::multiplies,
        ElementCompute,
        ElementCompute,
        cutlass::FloatRoundStyle::round_to_nearest>;
    using Evt0 = cutlass::epilogue::threadblock::Sm80EVT<
        Multiply0,
        Accumulator,
        ActivationScale>;
    using Multiply1 = cutlass::epilogue::threadblock::VisitorCompute<
        cutlass::multiplies,
        ElementCompute,
        ElementCompute,
        cutlass::FloatRoundStyle::round_to_nearest>;
    using Evt1 = cutlass::epilogue::threadblock::Sm80EVT<
        Multiply1,
        Evt0,
        WeightScale>;
    using AddBias = cutlass::epilogue::threadblock::VisitorCompute<
        cutlass::plus,
        ElementOutput,
        ElementCompute,
        cutlass::FloatRoundStyle::round_to_nearest>;
    using Evt2 = cutlass::epilogue::threadblock::Sm80EVT<AddBias, Evt1, Bias>;
    using Store = cutlass::epilogue::threadblock::VisitorAuxStore<
        ThreadMap,
        ElementOutput,
        cutlass::FloatRoundStyle::round_to_nearest,
        cute::Stride<int64_t, cute::_1, int64_t>>;
    using EvtStore = cutlass::epilogue::threadblock::Sm80EVT<Store, Evt2>;

    using Kernel = typename cutlass::gemm::kernel::DefaultGemmWithVisitor<
        ElementA,
        LayoutA,
        cutlass::ComplexTransform::kNone,
        AlignA,
        ElementB,
        LayoutB,
        cutlass::ComplexTransform::kNone,
        AlignB,
        ElementC,
        LayoutC,
        AlignC,
        ElementAccumulator,
        ElementCompute,
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm89,
        ThreadblockShape,
        WarpShape,
        InstructionShape,
        EvtStore,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        Stages,
        cutlass::arch::OpMultiplyAddSaturate,
        EvtStages>::GemmKernel;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;

    static bool run(
        const int8_t* activation,
        const int8_t* weight,
        const float* activation_scales,
        const float* weight_scales,
        const float* bias,
        ElementOutput* output,
        int m,
        int n,
        int k,
        cudaStream_t stream) {
        const cutlass::gemm::GemmCoord problem(m, n, k);
        typename EvtStore::Arguments callback{
            {{{{}, {const_cast<float*>(activation_scales), 0.0F, {cute::_1{}, cute::_0{}, m}}, {}},
               {const_cast<float*>(weight_scales), 0.0F, {cute::_0{}, cute::_1{}, n}}, {}},
              {const_cast<float*>(bias), 0.0F, {cute::_0{}, cute::_1{}, n}}, {}},
             {output, {n, cute::_1{}, static_cast<int64_t>(m) * n}}};
        typename Gemm::Arguments arguments(
            cutlass::gemm::GemmUniversalMode::kGemm,
            problem,
            1,
            callback,
            reinterpret_cast<cutlass::int4b_t*>(const_cast<int8_t*>(activation)),
            reinterpret_cast<cutlass::int4b_t*>(const_cast<int8_t*>(weight)),
            nullptr,
            nullptr,
            static_cast<int64_t>(m) * k,
            static_cast<int64_t>(n) * k,
            0,
            0,
            k,
            k,
            0,
            0);
        Gemm gemm;
        if (gemm.can_implement(arguments) != cutlass::Status::kSuccess) {
            return false;
        }
        if (Gemm::get_workspace_size(arguments) != 0) {
            return false;
        }
        if (gemm.initialize(arguments, nullptr, stream) != cutlass::Status::kSuccess) {
            return false;
        }
        return gemm(stream) == cutlass::Status::kSuccess;
    }
};

template <typename ElementOutput>
bool dispatch_gemm(
    const int8_t* activation,
    const int8_t* weight,
    const float* activation_scales,
    const float* weight_scales,
    const float* bias,
    ElementOutput* output,
    int m,
    int n,
    int k,
    cudaStream_t stream) {
    using Runner = bool (*)(
        const int8_t*,
        const int8_t*,
        const float*,
        const float*,
        const float*,
        ElementOutput*,
        int,
        int,
        int,
        cudaStream_t);
    static const Runner runners[] = {
        &FusedInt4Gemm<ElementOutput, 128, 256, 128, 64, 64, 128, 3>::run,
        &FusedInt4Gemm<ElementOutput, 128, 128, 256, 64, 64, 256, 3>::run,
    };
    constexpr int RunnerCount = static_cast<int>(sizeof(runners) / sizeof(runners[0]));
    using Key = std::tuple<int, int, int>;
    static std::mutex mutex;
    static std::map<Key, int> cache;
    static thread_local Key last_key{-1, -1, -1};
    static thread_local int last_runner = -2;
    const Key key{m, n, k};
    if (key == last_key && last_runner >= 0) {
        return runners[last_runner](
            activation,
            weight,
            activation_scales,
            weight_scales,
            bias,
            output,
            m,
            n,
            k,
            stream);
    }

    int selected = -2;
    {
        std::lock_guard<std::mutex> lock(mutex);
        const auto found = cache.find(key);
        if (found != cache.end()) {
            selected = found->second;
        }
    }
    if (selected == -2) {
        float best_ms = FLT_MAX;
        selected = -1;
        cudaEvent_t start;
        cudaEvent_t end;
        if (cudaEventCreate(&start) != cudaSuccess || cudaEventCreate(&end) != cudaSuccess) {
            return false;
        }
        for (int index = 0; index < RunnerCount; ++index) {
            if (!runners[index](
                    activation,
                    weight,
                    activation_scales,
                    weight_scales,
                    bias,
                    output,
                    m,
                    n,
                    k,
                    stream)) {
                cudaGetLastError();
                continue;
            }
            const cudaError_t sync_status = cudaStreamSynchronize(stream);
            if (sync_status != cudaSuccess) {
                cudaEventDestroy(start);
                cudaEventDestroy(end);
                return false;
            }
            bool warmup_ok = true;
            for (int warmup = 0; warmup < 8; ++warmup) {
                if (!runners[index](
                    activation,
                    weight,
                    activation_scales,
                    weight_scales,
                    bias,
                    output,
                    m,
                    n,
                    k,
                    stream)) {
                    warmup_ok = false;
                    break;
                }
            }
            if (!warmup_ok) {
                cudaGetLastError();
                continue;
            }
            const cudaError_t warmup_sync_status = cudaStreamSynchronize(stream);
            if (warmup_sync_status != cudaSuccess) {
                cudaEventDestroy(start);
                cudaEventDestroy(end);
                return false;
            }
            if (cudaEventRecord(start, stream) != cudaSuccess) {
                cudaGetLastError();
                continue;
            }
            bool measure_ok = true;
            for (int repeat = 0; repeat < 32; ++repeat) {
                if (!runners[index](
                    activation,
                    weight,
                    activation_scales,
                    weight_scales,
                    bias,
                    output,
                    m,
                    n,
                    k,
                    stream)) {
                    measure_ok = false;
                    break;
                }
            }
            if (!measure_ok) {
                cudaGetLastError();
                continue;
            }
            if (cudaEventRecord(end, stream) != cudaSuccess ||
                cudaEventSynchronize(end) != cudaSuccess) {
                cudaGetLastError();
                continue;
            }
            float elapsed_ms = 0.0F;
            cudaEventElapsedTime(&elapsed_ms, start, end);
            if (elapsed_ms < best_ms) {
                best_ms = elapsed_ms;
                selected = index;
            }
        }
        cudaEventDestroy(start);
        cudaEventDestroy(end);
        std::lock_guard<std::mutex> lock(mutex);
        cache[key] = selected;
    }
    last_key = key;
    last_runner = selected;
    if (selected < 0) {
        return false;
    }
    return runners[selected](
        activation,
        weight,
        activation_scales,
        weight_scales,
        bias,
        output,
        m,
        n,
        k,
        stream);
}

}  // namespace

extern "C" int xqt_convrot_w4a4_rowwise_quantize(
    const void* input,
    void* output,
    void* scales,
    int rows,
    int k,
    int scalar_kind,
    cudaStream_t stream) {
    if (rows <= 0 || k < 1024 || k > 32768 || (k != 1024 && k % 2048 != 0)) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    const int pairs_per_warp = k == 1024 ? 2 : 4;
    const int groups_per_warp = 2 * pairs_per_warp;
    const int warps_per_row = k / (groups_per_warp * kConvRotGroup);
    int rows_per_block = 1;
    if (k != 1024 && warps_per_row == 1) {
        rows_per_block = 4;
    } else if (k != 1024 && warps_per_row <= 3) {
        rows_per_block = 2;
    }
    const int threads = rows_per_block * warps_per_row * kWarpSize;
    const int blocks = (rows + rows_per_block - 1) / rows_per_block;
    const size_t shared_bytes = warps_per_row == 1
        ? 0
        : static_cast<size_t>(rows_per_block) * warps_per_row * sizeof(float);
    if (scalar_kind == 0) {
        if (pairs_per_warp == 2) {
            launch_quantize_rows<__half, 2>(
                rows_per_block,
                blocks,
                threads,
                shared_bytes,
                static_cast<const __half*>(input),
                static_cast<int8_t*>(output),
                static_cast<float*>(scales),
                rows,
                k,
                stream);
        } else {
            launch_quantize_rows<__half, 4>(
                rows_per_block,
                blocks,
                threads,
                shared_bytes,
                static_cast<const __half*>(input),
                static_cast<int8_t*>(output),
                static_cast<float*>(scales),
                rows,
                k,
                stream);
        }
    } else if (scalar_kind == 1) {
        if (pairs_per_warp == 2) {
            launch_quantize_rows<__nv_bfloat16, 2>(
                rows_per_block,
                blocks,
                threads,
                shared_bytes,
                static_cast<const __nv_bfloat16*>(input),
                static_cast<int8_t*>(output),
                static_cast<float*>(scales),
                rows,
                k,
                stream);
        } else {
            launch_quantize_rows<__nv_bfloat16, 4>(
                rows_per_block,
                blocks,
                threads,
                shared_bytes,
                static_cast<const __nv_bfloat16*>(input),
                static_cast<int8_t*>(output),
                static_cast<float*>(scales),
                rows,
                k,
                stream);
        }
    } else {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    return static_cast<int>(cudaGetLastError());
}

extern "C" int xqt_convrot_w4a4_rowwise_gemm(
    const void* activation,
    const void* weight,
    const void* activation_scales,
    const void* weight_scales,
    const void* bias,
    void* output,
    int m,
    int n,
    int k,
    int scalar_kind,
    cudaStream_t stream) {
    bool success = false;
    if (scalar_kind == 0) {
        success = dispatch_gemm(
            static_cast<const int8_t*>(activation),
            static_cast<const int8_t*>(weight),
            static_cast<const float*>(activation_scales),
            static_cast<const float*>(weight_scales),
            static_cast<const float*>(bias),
            static_cast<cutlass::half_t*>(output),
            m,
            n,
            k,
            stream);
    } else if (scalar_kind == 1) {
        success = dispatch_gemm(
            static_cast<const int8_t*>(activation),
            static_cast<const int8_t*>(weight),
            static_cast<const float*>(activation_scales),
            static_cast<const float*>(weight_scales),
            static_cast<const float*>(bias),
            static_cast<cutlass::bfloat16_t*>(output),
            m,
            n,
            k,
            stream);
    }
    return static_cast<int>(success ? cudaSuccess : cudaErrorNotSupported);
}

extern "C" const char* xqt_convrot_w4a4_rowwise_version() {
    return "convrot_w4a4_rowwise_sm89_v1";
}
