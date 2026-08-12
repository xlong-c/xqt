// RMSNorm-fused SVDQuant W4A4 helper kernels for Ada sm_89.
//
// Motivation: a DiT block runs RMSNorm before every linear.  The standalone
// norm kernel costs a full activation read+write round trip per linear
// (measured +6.9% e2e at M256, +12% at M1024 on the DiT-like block bench).
// The norm factors exactly through the W4A4 quantization: with
// r[row] = rsqrt(mean(x[row]^2) + eps) > 0 and the per-channel norm weight
// folded into smooth_factor and the LoRA-down columns at pack time, the int4
// codes are unchanged and only the group absmax scales and the LoRA-down
// partials must be multiplied by r[row].  These two tiny custom kernels keep
// the upstream Nunchaku quantize/GEMM templates untouched.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <cstdint>

namespace {

// GEMMConfig_W4A4 activation-scale layout constants (gemm_base.cuh):
//   BLOCK_M=256, NUM_WARPS=8, WARP_M=32, ASCALES_PACK_SIZE=2,
//   ASCALES_NUM_PACKS=1, ASCALES_VALID_LANES=16.
// A packed_ascale_t (half2) at storage lane holds rows
//   lane%8 + (lane/8)*16 (element x) and +8 (element y) of the warp tile.
constexpr int kQuantGroup = 64;
constexpr int kBlockM = 256;
constexpr int kNumWarps = 8;
constexpr int kWarpM = 32;
constexpr int kAscalesValidLanes = 16;

enum class ScalarKind : int {
    FP16 = 0,
    BF16 = 1,
};

template <typename scalar_t>
__global__ void row_rms_kernel(
    const scalar_t* input,
    float* row_scales,
    int rows,
    int actual_k,
    float eps) {
    const int row = blockIdx.x;
    float sum = 0.0F;
    if (row < rows) {
        const scalar_t* base = input + static_cast<int64_t>(row) * actual_k;
        if (actual_k % 8 == 0) {
            // Vectorized main path: eight halves per 16-byte load.
            const uint4* vec = reinterpret_cast<const uint4*>(base);
            const int vec_count = actual_k / 8;
            for (int k = threadIdx.x; k < vec_count; k += blockDim.x) {
                const uint4 packed = __ldg(vec + k);
                const scalar_t* values = reinterpret_cast<const scalar_t*>(&packed);
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const float value = static_cast<float>(values[j]);
                    sum += value * value;
                }
            }
        } else {
            for (int k = threadIdx.x; k < actual_k; k += blockDim.x) {
                const float value = static_cast<float>(base[k]);
                sum += value * value;
            }
        }
    }
    __shared__ float shared[32];
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffffU, sum, offset);
    }
    if ((threadIdx.x & 31) == 0) {
        shared[threadIdx.x >> 5] = sum;
    }
    __syncthreads();
    if (threadIdx.x < 32) {
        const int warp_count = static_cast<int>(blockDim.x) >> 5;
        sum = threadIdx.x < warp_count ? shared[threadIdx.x] : 0.0F;
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xffffffffU, sum, offset);
        }
        if (threadIdx.x == 0) {
            row_scales[row] = row < rows ? rsqrtf(sum / static_cast<float>(actual_k) + eps) : 0.0F;
        }
    }
}

// One thread per (row, group) scales exactly the component belonging to that
// row inside the interleaved packed_ascale_t unit; one thread per physical
// lora_act element applies r[row] using the MMA fragment layout measured
// against quantize_act_lora (LORA_M_TILES=2, LORA_R_TILES=1, WARP_R=16):
//   per-bm segment = rank * 256 floats;
//   segment offset = rtile*4096 + warp*512 + mtile*256 + j8*32 + lane
//   row  = bm*256 + warp*32 + mtile*16 + ((j8>>1)&1)*8 + lane/4
//   rcol = rtile*16 + (j8>>2)*8 + (lane%4)*2 + (j8&1)
template <typename scalar_t>
__global__ void scale_scales_lora_kernel(
    scalar_t* ascales,
    float* lora_act,
    const float* row_scales,
    int padded_m,
    int groups,
    int rank) {
    const int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t ascales_total = static_cast<int64_t>(padded_m) * groups;
    if (tid < ascales_total) {
        const int row = static_cast<int>(tid / groups);
        const int group = static_cast<int>(tid % groups);
        const float scale = row_scales[row];
        const int block_m = row / kBlockM;
        const int warp = (row % kBlockM) / kWarpM;
        const int k = row % kWarpM;
        const int lane = (k & 7) | ((k & 16) >> 1);
        const int element = (k >> 3) & 1;
        const int64_t unit =
            ((static_cast<int64_t>(block_m) * groups + group) * kNumWarps + warp) *
                kAscalesValidLanes +
            lane;
        scalar_t* slot = ascales + unit * 2 + element;
        *slot = static_cast<scalar_t>(static_cast<float>(*slot) * scale);
        return;
    }
    const int64_t lora_tid = tid - ascales_total;
    const int64_t lora_total = static_cast<int64_t>(padded_m) * rank;
    if (lora_tid >= lora_total) {
        return;
    }
    const int bm = static_cast<int>(lora_tid / (static_cast<int64_t>(rank) * kBlockM));
    const int rem = static_cast<int>(lora_tid % (static_cast<int64_t>(rank) * kBlockM));
    const int warp = (rem % 4096) / 512;
    const int mtile = ((rem % 4096) % 512) / 256;
    const int j8 = ((rem % 4096) % 256) / 32;
    const int lane = ((rem % 4096) % 256) % 32;
    const int row =
        bm * kBlockM + warp * kWarpM + mtile * 16 + ((j8 >> 1) & 1) * 8 + (lane >> 2);
    lora_act[lora_tid] *= row_scales[row];
}

template <typename scalar_t>
int launch_row_rms(
    const void* input,
    float* row_scales,
    int rows,
    int actual_k,
    int padded_m,
    float eps,
    cudaStream_t stream) {
    row_rms_kernel<scalar_t><<<padded_m, 256, 0, stream>>>(
        static_cast<const scalar_t*>(input),
        row_scales,
        rows,
        actual_k,
        eps);
    return static_cast<int>(cudaGetLastError());
}

template <typename scalar_t>
int launch_scale_scales_lora(
    void* ascales,
    float* lora_act,
    const float* row_scales,
    int padded_m,
    int padded_k,
    int rank,
    cudaStream_t stream) {
    const int groups = padded_k / kQuantGroup;
    const int64_t total =
        static_cast<int64_t>(padded_m) * groups + static_cast<int64_t>(padded_m) * rank;
    const int threads = 256;
    const int64_t blocks = (total + threads - 1) / threads;
    scale_scales_lora_kernel<scalar_t><<<static_cast<unsigned int>(blocks), threads, 0, stream>>>(
        static_cast<scalar_t*>(ascales),
        lora_act,
        row_scales,
        padded_m,
        groups,
        rank);
    return static_cast<int>(cudaGetLastError());
}

}  // namespace

extern "C" int xqt_svdq_w4a4_norm_row_rms(
    const void* input,
    void* row_scales,
    int rows,
    int actual_k,
    int padded_m,
    float eps,
    int scalar_kind,
    cudaStream_t stream) {
    switch (static_cast<ScalarKind>(scalar_kind)) {
        case ScalarKind::FP16:
            return launch_row_rms<half>(input, static_cast<float*>(row_scales), rows, actual_k, padded_m, eps, stream);
        case ScalarKind::BF16:
            return launch_row_rms<__nv_bfloat16>(
                input, static_cast<float*>(row_scales), rows, actual_k, padded_m, eps, stream);
    }
    return static_cast<int>(cudaErrorInvalidValue);
}

extern "C" int xqt_svdq_w4a4_norm_scale_scales(
    void* ascales,
    void* lora_act,
    const void* row_scales,
    int padded_m,
    int padded_k,
    int rank,
    int scalar_kind,
    cudaStream_t stream) {
    switch (static_cast<ScalarKind>(scalar_kind)) {
        case ScalarKind::FP16:
            return launch_scale_scales_lora<half>(
                ascales, static_cast<float*>(lora_act), static_cast<const float*>(row_scales),
                padded_m, padded_k, rank, stream);
        case ScalarKind::BF16:
            return launch_scale_scales_lora<__nv_bfloat16>(
                ascales, static_cast<float*>(lora_act), static_cast<const float*>(row_scales),
                padded_m, padded_k, rank, stream);
    }
    return static_cast<int>(cudaErrorInvalidValue);
}
