#pragma once
// 精简版 Tensor.h — 仅提供 dispatch_utils.h 需要的 ScalarType 枚举
// 实际 torch 桥接在 nunchaku_bridge.cu 中

#include <cassert>
#include <cstddef>
#include <vector>

struct Tensor {
    enum ScalarType {
        FP32 = 0,
        FP16 = 1,
        BF16 = 2,
        INT8  = 3,
        INT32 = 4,
        INT64 = 5,
        INVALID_SCALAR_TYPE = -1,
    };

    ScalarType dtype() const { return dtype_; }
    bool valid() const { return data_ptr_ != nullptr; }
    int numel() const { return numel_; }

    template<typename T>
    T* data_ptr() const { return reinterpret_cast<T*>(data_ptr_); }

    // 占位: 实际不调用
    const int* shape = nullptr;
    int ndims() const { return 0; }

    // 内部
    void* data_ptr_ = nullptr;
    ScalarType dtype_ = INVALID_SCALAR_TYPE;
    int numel_ = 0;
    int shape_[8] = {0};
};
