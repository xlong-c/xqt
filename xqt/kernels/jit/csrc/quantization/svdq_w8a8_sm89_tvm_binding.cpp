#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/optional.h>
#include <cuda_runtime_api.h>

#include <cstdint>
#include <string>

extern "C" int xqt_svdq_w8a8_quantize_weight(
    const void*, const void*, void*, int, int, cudaStream_t);
extern "C" int xqt_svdq_w8a8_quantize_act_lora(
    const void*, void*, void*, const void*, void*, int, int, int, int, int, cudaStream_t);
extern "C" int xqt_svdq_w8a8_quantize_act(
    const void*, void*, void*, int, int, int, int, cudaStream_t);
extern "C" int xqt_svdq_w8a8_quantize_act_fast(
    const void*, void*, void*, int, int, cudaStream_t);
extern "C" int xqt_svdq_w8a8_gemm_lora(
    const void*, const void*, void*, const void*, const void*, const void*, const void*, const void*, int, int, int, int,
    int, int, float, cudaStream_t);
extern "C" int xqt_svdq_w8a8_gemm(
    const void*, const void*, void*, const void*, const void*, const void*, int, int, int, int, int, cudaStream_t);
extern "C" const char* xqt_svdq_w8a8_version();

namespace {

static const DLDataType kBf16Type{kDLBfloat, 16, 1};
static const DLDataType kFp16Type{kDLFloat, 16, 1};
static const DLDataType kInt8Type{kDLInt, 8, 1};
static const DLDataType kFloat32Type{kDLFloat, 32, 1};
static const DLDataType kInt32Type{kDLInt, 32, 1};
static const DLDataType kUint8Type{kDLUInt, 8, 1};

inline bool match_dtype(const DLDataType& a, const DLDataType& b) {
    return a.code == b.code && a.bits == b.bits && a.lanes == b.lanes;
}

void require_cuda_contiguous(const tvm::ffi::TensorView& tensor, const char* name) {
    TVM_FFI_ICHECK(tensor.device().device_type == kDLCUDA) << name << " must be CUDA";
    TVM_FFI_ICHECK(tensor.is_contiguous()) << name << " must be contiguous";
}

void require_same_device(
    const tvm::ffi::TensorView& reference,
    const tvm::ffi::TensorView& tensor,
    const char* name) {
    TVM_FFI_ICHECK(
        reference.device().device_type == tensor.device().device_type &&
        reference.device().device_id == tensor.device().device_id)
        << name << " must share the CUDA device";
}

cudaStream_t get_current_stream(const tvm::ffi::TensorView& tensor) {
    void* stream_handle = TVMFFIEnvGetStream(
        static_cast<int32_t>(tensor.device().device_type),
        static_cast<int32_t>(tensor.device().device_id));
    return static_cast<cudaStream_t>(stream_handle);
}

void check_status(int status, const char* operation) {
    TVM_FFI_ICHECK(status == static_cast<int>(cudaSuccess))
        << operation << " failed: " << cudaGetErrorString(static_cast<cudaError_t>(status));
}

void quantize_weight(
    tvm::ffi::TensorView input,
    tvm::ffi::TensorView scales,
    tvm::ffi::TensorView output) {
    require_cuda_contiguous(input, "input");
    require_cuda_contiguous(scales, "scales");
    require_cuda_contiguous(output, "output");
    TVM_FFI_ICHECK(match_dtype(input.dtype(), kBf16Type)) << "input must be bfloat16";
    TVM_FFI_ICHECK(match_dtype(scales.dtype(), kBf16Type)) << "scales must be bfloat16";
    TVM_FFI_ICHECK(input.ndim() == 2) << "input must be [N, K]";
    TVM_FFI_ICHECK(scales.ndim() == 1 && scales.size(0) == input.size(0)) << "scales must be [N]";
    TVM_FFI_ICHECK(match_dtype(output.dtype(), kInt8Type)) << "output must be int8";
    TVM_FFI_ICHECK(output.size(0) == input.size(0) && output.size(1) == input.size(1)) << "output must match input shape";
    require_same_device(input, scales, "scales");
    require_same_device(input, output, "output");
    const int n = static_cast<int>(input.size(0));
    const int k = static_cast<int>(input.size(1));
    TVM_FFI_ICHECK(n % 128 == 0 && k % 32 == 0) << "weight shape must align to N=128 and K=32";
    cudaSetDevice(input.device().device_id);
    check_status(
        xqt_svdq_w8a8_quantize_weight(
            input.data_ptr(), scales.data_ptr(), output.data_ptr(), n, k, get_current_stream(input)),
        "W8A8 weight quantization");
}

void quantize_act_lora(
    tvm::ffi::TensorView input,
    tvm::ffi::TensorView output,
    tvm::ffi::TensorView scales,
    tvm::ffi::TensorView lora_down,
    tvm::ffi::TensorView lora_activation) {
    require_cuda_contiguous(input, "input");
    require_cuda_contiguous(output, "output");
    require_cuda_contiguous(scales, "scales");
    require_cuda_contiguous(lora_down, "lora_down");
    require_cuda_contiguous(lora_activation, "lora_activation");
    
    require_same_device(input, output, "output");
    require_same_device(input, scales, "scales");
    require_same_device(input, lora_down, "lora_down");
    require_same_device(input, lora_activation, "lora_activation");

    TVM_FFI_ICHECK(match_dtype(input.dtype(), kBf16Type)) << "input must be bfloat16";
    TVM_FFI_ICHECK(match_dtype(scales.dtype(), kBf16Type)) << "scales must be bfloat16";
    TVM_FFI_ICHECK(match_dtype(lora_down.dtype(), kBf16Type)) << "lora_down must be bfloat16";
    TVM_FFI_ICHECK(input.ndim() == 2) << "input must be [M, K]";
    TVM_FFI_ICHECK(output.ndim() == 2 && match_dtype(output.dtype(), kInt8Type)) << "output must be 2D int8";
    
    const int actual_m = static_cast<int>(input.size(0));
    const int actual_k = static_cast<int>(input.size(1));
    const int padded_m = static_cast<int>(output.size(0));
    const int padded_k = static_cast<int>(output.size(1));
    
    TVM_FFI_ICHECK(padded_m % 256 == 0 && padded_m >= actual_m && padded_m - actual_m < 256)
        << "invalid M padding";
    TVM_FFI_ICHECK(padded_k % 128 == 0 && padded_k >= actual_k && padded_k - actual_k < 128)
        << "invalid K padding";
    TVM_FFI_ICHECK(scales.numel() == padded_m) << "activation scales must contain M_pad values";
    TVM_FFI_ICHECK(lora_down.ndim() == 2 && lora_down.size(0) == padded_k) << "lora_down must be [K_pad, rank]";
    
    const int rank = static_cast<int>(lora_down.size(1));
    TVM_FFI_ICHECK(rank > 0 && rank % 16 == 0 && rank <= 1024) << "rank must align to 16 and be <= 1024";
    TVM_FFI_ICHECK(match_dtype(lora_activation.dtype(), kFloat32Type)) << "lora_activation must be float32";
    TVM_FFI_ICHECK(lora_activation.ndim() == 2 && lora_activation.size(0) == padded_m && lora_activation.size(1) == rank)
        << "lora_activation must be [M_pad, rank]";
        
    cudaSetDevice(input.device().device_id);
    const cudaStream_t stream = get_current_stream(input);
    check_status(
        static_cast<int>(cudaMemsetAsync(
            lora_activation.data_ptr(),
            0,
            static_cast<size_t>(lora_activation.numel()) * sizeof(float),
            stream)),
        "W8A8 LoRA activation reset");
    check_status(
        xqt_svdq_w8a8_quantize_act_lora(
            input.data_ptr(),
            output.data_ptr(),
            scales.data_ptr(),
            lora_down.data_ptr(),
            lora_activation.data_ptr(),
            actual_m,
            actual_k,
            padded_m,
            padded_k,
            rank,
            stream),
        "W8A8 activation quantization plus LoRA down");
}

void quantize_act(
    tvm::ffi::TensorView input,
    tvm::ffi::TensorView output,
    tvm::ffi::TensorView scales) {
    require_cuda_contiguous(input, "input");
    require_cuda_contiguous(output, "output");
    require_cuda_contiguous(scales, "scales");
    
    require_same_device(input, output, "output");
    require_same_device(input, scales, "scales");

    TVM_FFI_ICHECK(match_dtype(input.dtype(), kBf16Type)) << "input must be bfloat16";
    TVM_FFI_ICHECK(match_dtype(scales.dtype(), kBf16Type)) << "scales must be bfloat16";
    TVM_FFI_ICHECK(input.ndim() == 2) << "input must be [M, K]";
    TVM_FFI_ICHECK(output.ndim() == 2 && match_dtype(output.dtype(), kInt8Type)) << "output must be 2D int8";
    
    const int actual_m = static_cast<int>(input.size(0));
    const int actual_k = static_cast<int>(input.size(1));
    const int padded_m = static_cast<int>(output.size(0));
    const int padded_k = static_cast<int>(output.size(1));
    
    TVM_FFI_ICHECK(padded_m % 256 == 0 && padded_m >= actual_m && padded_m - actual_m < 256)
        << "invalid M padding";
    TVM_FFI_ICHECK(padded_k % 128 == 0 && padded_k >= actual_k && padded_k - actual_k < 128)
        << "invalid K padding";
    TVM_FFI_ICHECK(scales.numel() == padded_m) << "activation scales must contain M_pad values";
    
    cudaSetDevice(input.device().device_id);
    const cudaStream_t stream = get_current_stream(input);
    if (actual_m == padded_m && actual_k == padded_k) {
        check_status(
            xqt_svdq_w8a8_quantize_act_fast(
                input.data_ptr(),
                output.data_ptr(),
                scales.data_ptr(),
                padded_m,
                padded_k,
                stream),
            "fast W8A8 activation quantization");
        return;
    }
    check_status(
        xqt_svdq_w8a8_quantize_act(
            input.data_ptr(),
            output.data_ptr(),
            scales.data_ptr(),
            actual_m,
            actual_k,
            padded_m,
            padded_k,
            stream),
        "W8A8 activation quantization");
}

void gemm_lora(
    tvm::ffi::TensorView activation,
    tvm::ffi::TensorView weight,
    tvm::ffi::TensorView output,
    tvm::ffi::TensorView activation_scales,
    tvm::ffi::TensorView weight_scales,
    tvm::ffi::TensorView lora_activation,
    tvm::ffi::TensorView lora_up,
    tvm::ffi::TensorView bias,
    double lora_scale) {
    require_cuda_contiguous(activation, "activation");
    require_cuda_contiguous(weight, "weight");
    require_cuda_contiguous(output, "output");
    require_cuda_contiguous(activation_scales, "activation_scales");
    require_cuda_contiguous(weight_scales, "weight_scales");
    require_cuda_contiguous(lora_activation, "lora_activation");
    require_cuda_contiguous(lora_up, "lora_up");
    require_cuda_contiguous(bias, "bias");

    require_same_device(output, activation, "activation");
    require_same_device(output, weight, "weight");
    require_same_device(output, activation_scales, "activation_scales");
    require_same_device(output, weight_scales, "weight_scales");
    require_same_device(output, lora_activation, "lora_activation");
    require_same_device(output, lora_up, "lora_up");
    require_same_device(output, bias, "bias");

    TVM_FFI_ICHECK(activation.ndim() == 2 && match_dtype(activation.dtype(), kInt8Type)) << "activation must be 2D int8";
    TVM_FFI_ICHECK(weight.ndim() == 2 && match_dtype(weight.dtype(), kInt8Type)) << "weight must be 2D int8";
    
    TVM_FFI_ICHECK(match_dtype(output.dtype(), kBf16Type)) << "output must be bfloat16";
    TVM_FFI_ICHECK(match_dtype(activation_scales.dtype(), kBf16Type)) << "activation_scales must be bfloat16";
    TVM_FFI_ICHECK(match_dtype(weight_scales.dtype(), kBf16Type)) << "weight_scales must be bfloat16";
    TVM_FFI_ICHECK(match_dtype(lora_up.dtype(), kBf16Type)) << "lora_up must be bfloat16";
    TVM_FFI_ICHECK(match_dtype(bias.dtype(), kBf16Type)) << "bias must be bfloat16";
    TVM_FFI_ICHECK(match_dtype(lora_activation.dtype(), kFloat32Type)) << "lora_activation must be float32";

    const int padded_m = static_cast<int>(activation.size(0));
    const int padded_n = static_cast<int>(weight.size(0));
    const int padded_k = static_cast<int>(activation.size(1));
    
    TVM_FFI_ICHECK(weight.size(1) == padded_k) << "activation and weight K mismatch";
    TVM_FFI_ICHECK(padded_m % 256 == 0 && padded_n % 128 == 0 && padded_k % 128 == 0) << "invalid padded GEMM shape";
    TVM_FFI_ICHECK(output.size(0) <= padded_m && padded_m - output.size(0) < 256) << "invalid output M";
    TVM_FFI_ICHECK(output.size(1) <= padded_n && padded_n - output.size(1) < 128) << "invalid output N";
    TVM_FFI_ICHECK(output.size(1) % 4 == 0) << "output N must be a multiple of 4 for vectorized epilogue stores";
    
    TVM_FFI_ICHECK(activation_scales.numel() == padded_m) << "activation scale storage mismatch";
    TVM_FFI_ICHECK(weight_scales.numel() == padded_n) << "weight scale storage mismatch";
    TVM_FFI_ICHECK(bias.numel() == padded_n) << "bias storage mismatch";
    
    const int rank = static_cast<int>(lora_up.size(1));
    TVM_FFI_ICHECK(rank > 0 && rank % 16 == 0 && rank <= 1024) << "rank must align to 16 and be <= 1024";
    TVM_FFI_ICHECK(lora_up.ndim() == 2 && lora_up.size(0) == padded_n && lora_up.size(1) == rank) << "lora_up must be [N_pad, rank]";
    TVM_FFI_ICHECK(lora_activation.ndim() == 2 && lora_activation.size(0) == padded_m && lora_activation.size(1) == rank) << "lora_activation must be [M_pad, rank]";

    cudaSetDevice(output.device().device_id);
    check_status(
        xqt_svdq_w8a8_gemm_lora(
            activation.data_ptr(),
            weight.data_ptr(),
            output.data_ptr(),
            activation_scales.data_ptr(),
            weight_scales.data_ptr(),
            lora_activation.data_ptr(),
            lora_up.data_ptr(),
            bias.data_ptr(),
            static_cast<int>(output.size(0)),
            static_cast<int>(output.size(1)),
            padded_m,
            padded_n,
            padded_k,
            rank,
            static_cast<float>(lora_scale),
            get_current_stream(output)),
        "W8A8 GEMM plus LoRA up");
}

void gemm(
    tvm::ffi::TensorView activation,
    tvm::ffi::TensorView weight,
    tvm::ffi::TensorView output,
    tvm::ffi::TensorView activation_scales,
    tvm::ffi::TensorView weight_scales,
    tvm::ffi::TensorView bias) {
    require_cuda_contiguous(activation, "activation");
    require_cuda_contiguous(weight, "weight");
    require_cuda_contiguous(output, "output");
    require_cuda_contiguous(activation_scales, "activation_scales");
    require_cuda_contiguous(weight_scales, "weight_scales");
    require_cuda_contiguous(bias, "bias");

    require_same_device(output, activation, "activation");
    require_same_device(output, weight, "weight");
    require_same_device(output, activation_scales, "activation_scales");
    require_same_device(output, weight_scales, "weight_scales");
    require_same_device(output, bias, "bias");

    TVM_FFI_ICHECK(activation.ndim() == 2 && match_dtype(activation.dtype(), kInt8Type)) << "activation must be 2D int8";
    TVM_FFI_ICHECK(weight.ndim() == 2 && match_dtype(weight.dtype(), kInt8Type)) << "weight must be 2D int8";
    
    TVM_FFI_ICHECK(match_dtype(output.dtype(), kBf16Type)) << "output must be bfloat16";
    TVM_FFI_ICHECK(match_dtype(activation_scales.dtype(), kBf16Type)) << "activation_scales must be bfloat16";
    TVM_FFI_ICHECK(match_dtype(weight_scales.dtype(), kBf16Type)) << "weight_scales must be bfloat16";
    TVM_FFI_ICHECK(match_dtype(bias.dtype(), kBf16Type)) << "bias must be bfloat16";

    const int padded_m = static_cast<int>(activation.size(0));
    const int padded_n = static_cast<int>(weight.size(0));
    const int padded_k = static_cast<int>(activation.size(1));
    
    TVM_FFI_ICHECK(weight.size(1) == padded_k) << "activation and weight K mismatch";
    TVM_FFI_ICHECK(padded_m % 256 == 0 && padded_n % 128 == 0 && padded_k % 128 == 0) << "invalid padded GEMM shape";
    TVM_FFI_ICHECK(output.size(0) <= padded_m && padded_m - output.size(0) < 256) << "invalid output M";
    TVM_FFI_ICHECK(output.size(1) <= padded_n && padded_n - output.size(1) < 128) << "invalid output N";
    TVM_FFI_ICHECK(output.size(1) % 4 == 0) << "output N must be a multiple of 4 for vectorized epilogue stores";
    
    TVM_FFI_ICHECK(activation_scales.numel() == padded_m) << "activation scale storage mismatch";
    TVM_FFI_ICHECK(weight_scales.numel() == padded_n) << "weight scale storage mismatch";
    TVM_FFI_ICHECK(bias.numel() == padded_n) << "bias storage mismatch";

    cudaSetDevice(output.device().device_id);
    check_status(
        xqt_svdq_w8a8_gemm(
            activation.data_ptr(),
            weight.data_ptr(),
            output.data_ptr(),
            activation_scales.data_ptr(),
            weight_scales.data_ptr(),
            bias.data_ptr(),
            static_cast<int>(output.size(0)),
            static_cast<int>(output.size(1)),
            padded_m,
            padded_n,
            padded_k,
            get_current_stream(output)),
        "W8A8 GEMM");
}

std::string version() {
    return std::string(xqt_svdq_w8a8_version());
}

}  // namespace

TVM_FFI_DLL_EXPORT_TYPED_FUNC(quantize_weight, quantize_weight);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(quantize_act_lora, quantize_act_lora);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(quantize_act, quantize_act);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(gemm_lora, gemm_lora);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(gemm, gemm);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(version, version);
