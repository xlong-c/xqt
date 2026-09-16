#pragma once
// 精简版 common.h — 仅提供 kernel 编译所需的最小依赖

#include <cstddef>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime_api.h>

namespace nunchaku::kernels {

template<typename T>
constexpr T ceilDiv(T a, T b) {
    return (a + b - 1) / b;
}

// === CUDA 错误检查 (仅 host 端) ===
inline cudaError_t checkCUDA(cudaError_t retValue) {
    if (retValue != cudaSuccess) {
        const char *msg = cudaGetErrorString(retValue);
        fprintf(stderr, "CUDA error: %s\n", msg);
        abort();
    }
    return retValue;
}

// === 获取当前 CUDA 设备属性 ===
inline cudaDeviceProp* getCurrentDeviceProperties() {
    static cudaDeviceProp prop;
    static bool initialized = false;
    if (!initialized) {
        int dev;
        cudaGetDevice(&dev);
        cudaGetDeviceProperties(&prop, dev);
        initialized = true;
    }
    return &prop;
}

}  // namespace nunchaku::kernels
