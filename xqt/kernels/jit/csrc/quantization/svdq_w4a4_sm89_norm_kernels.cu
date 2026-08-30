// RMSNorm row-reduction helper for the SVDQuant W4A4 activation kernel.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <cstdint>

namespace {

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
