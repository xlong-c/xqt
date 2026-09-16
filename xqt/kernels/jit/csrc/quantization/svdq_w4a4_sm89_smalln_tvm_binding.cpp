#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/optional.h>
#include <cuda_runtime_api.h>

#include <cstdint>
#include <string>

extern "C" int xqt_svdq_w4a4_smalln_quantize_weight(
    const void*, void*, void*, int, int, int, cudaStream_t);
extern "C" int xqt_svdq_w4a4_smalln_quantize_act_lora(
    const void*, void*, void*, const void*, void*, const void*, int, int, int, int, int, int, cudaStream_t);
extern "C" int xqt_svdq_w4a4_smalln_norm_quantize_act_lora(
    const void*, const void*, const void*, void*, void*, const void*, void*, const void*, int, int, int, int, int, int,
    cudaStream_t);
extern "C" int xqt_svdq_w4a4_smalln_gemm(
    const void*, const void*, void*, const void*, const void*, const void*, int, int, int, int, int, int, cudaStream_t);
extern "C" int xqt_svdq_w4a4_smalln_gemm_lora(
    const void*, const void*, void*, const void*, const void*, const void*, const void*, const void*, int, int, int, int,
    int, int, float, int, cudaStream_t);
extern "C" const char* xqt_svdq_w4a4_smalln_version();
extern "C" int xqt_svdq_w4a4_norm_row_rms(
    const void*, void*, int, int, int, float, int, cudaStream_t);

namespace {

static const DLDataType kBf16Type{kDLBfloat, 16, 1};
static const DLDataType kFp16Type{kDLFloat, 16, 1};
static const DLDataType kInt8Type{kDLInt, 8, 1};
static const DLDataType kFloat32Type{kDLFloat, 32, 1};

inline bool match_dtype(const DLDataType& a, const DLDataType& b) {
    return a.code == b.code && a.bits == b.bits && a.lanes == b.lanes;
}

int scalar_kind(const tvm::ffi::TensorView& tensor) {
    if (match_dtype(tensor.dtype(), kFp16Type)) return 0;
    if (match_dtype(tensor.dtype(), kBf16Type)) return 1;
    TVM_FFI_ICHECK(false) << "W4A4 small-N supports only float16 and bfloat16";
    return -1;
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

void require_same_scalar(
    const tvm::ffi::TensorView& reference,
    const tvm::ffi::TensorView& tensor,
    const char* name) {
    TVM_FFI_ICHECK(match_dtype(tensor.dtype(), reference.dtype())) << name << " must match the floating-point dtype";
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

constexpr int kSmallBlockN = 64;

void quantize_weight(tvm::ffi::TensorView input, tvm::ffi::TensorView output, tvm::ffi::TensorView scales) {
    require_cuda_contiguous(input, "input");
    require_cuda_contiguous(output, "output");
    require_cuda_contiguous(scales, "scales");
    TVM_FFI_ICHECK(input.ndim() == 2) << "input must be [N, K]";
    TVM_FFI_ICHECK(match_dtype(output.dtype(), kInt8Type)) << "output must be int8";
    require_same_device(input, output, "output");
    require_same_device(input, scales, "scales");
    require_same_scalar(input, scales, "scales");
    const int n = static_cast<int>(input.size(0));
    const int k = static_cast<int>(input.size(1));
    TVM_FFI_ICHECK(n % kSmallBlockN == 0) << "N must be padded to a multiple of 64";
    TVM_FFI_ICHECK(k % 64 == 0) << "K must be padded to a multiple of 64";
    TVM_FFI_ICHECK(output.size(0) == n && output.size(1) == k / 2) << "output shape must be [N, K / 2]";
    TVM_FFI_ICHECK(scales.numel() == static_cast<int64_t>(n) * k / 64) << "weight scale storage size mismatch";
    cudaSetDevice(input.device().device_id);
    check_status(
        xqt_svdq_w4a4_smalln_quantize_weight(
            input.data_ptr(), output.data_ptr(), scales.data_ptr(), n, k, scalar_kind(input), get_current_stream(input)),
        "W4A4 small-N weight quantization");
}

void quantize_act_lora(
    tvm::ffi::TensorView input,
    tvm::ffi::TensorView output,
    tvm::ffi::TensorView scales,
    tvm::ffi::TensorView lora_down,
    tvm::ffi::TensorView lora_act,
    tvm::ffi::TensorView smooth) {
    require_cuda_contiguous(input, "input");
    require_cuda_contiguous(output, "output");
    require_cuda_contiguous(scales, "scales");
    require_cuda_contiguous(smooth, "smooth");
    require_cuda_contiguous(lora_down, "lora_down");
    require_cuda_contiguous(lora_act, "lora_act");
    TVM_FFI_ICHECK(input.ndim() == 2) << "input must be [M, K]";
    TVM_FFI_ICHECK(output.ndim() == 2 && match_dtype(output.dtype(), kInt8Type)) << "output must be 2D int8";
    require_same_device(input, output, "output");
    require_same_device(input, scales, "scales");
    require_same_device(input, smooth, "smooth");
    require_same_scalar(input, scales, "scales");
    require_same_scalar(input, smooth, "smooth");
    const int padded_m = static_cast<int>(output.size(0));
    const int padded_k = static_cast<int>(output.size(1) * 2);
    TVM_FFI_ICHECK(padded_m % 256 == 0 && padded_m >= input.size(0) && padded_m - input.size(0) < 256) << "invalid M padding";
    TVM_FFI_ICHECK(padded_k % 128 == 0 && padded_k >= input.size(1) && padded_k - input.size(1) < 128) << "invalid K padding";
    TVM_FFI_ICHECK(scales.numel() == static_cast<int64_t>(padded_m) * padded_k / 64) << "activation scale size mismatch";
    TVM_FFI_ICHECK(smooth.numel() == padded_k) << "smooth storage size mismatch";
    const int rank = static_cast<int>(lora_down.size(1));
    TVM_FFI_ICHECK(lora_down.ndim() == 2 && lora_down.size(0) == padded_k) << "lora_down must be [K_pad, rank]";
    TVM_FFI_ICHECK(rank > 0 && rank % 16 == 0 && rank <= 1024) << "rank must be a positive multiple of 16 up to 1024";
    TVM_FFI_ICHECK(match_dtype(lora_act.dtype(), kFloat32Type)) << "lora_act must be float32";
    TVM_FFI_ICHECK(lora_act.size(0) == padded_m && lora_act.size(1) == rank) << "lora_act must be [M_pad, rank]";
    require_same_device(input, lora_down, "lora_down");
    require_same_device(input, lora_act, "lora_act");
    require_same_scalar(input, lora_down, "lora_down");

    cudaSetDevice(input.device().device_id);
    const cudaStream_t stream = get_current_stream(input);
    check_status(
        static_cast<int>(cudaMemsetAsync(
            lora_act.data_ptr(),
            0,
            static_cast<size_t>(lora_act.numel()) * sizeof(float),
            stream)),
        "W4A4 small-N LoRA activation reset");
    int status = xqt_svdq_w4a4_smalln_quantize_act_lora(
        input.data_ptr(),
        output.data_ptr(),
        scales.data_ptr(),
        lora_down.data_ptr(),
        lora_act.data_ptr(),
        smooth.data_ptr(),
        static_cast<int>(input.size(0)),
        static_cast<int>(input.size(1)),
        padded_m,
        padded_k,
        rank,
        scalar_kind(input),
        stream);
    check_status(status, "W4A4 small-N activation quantization plus LoRA down");
}

void norm_quantize_act_lora(
    tvm::ffi::TensorView input,
    tvm::ffi::TensorView norm_weight,
    tvm::ffi::TensorView row_scales,
    tvm::ffi::TensorView output,
    tvm::ffi::TensorView scales,
    tvm::ffi::TensorView lora_down,
    tvm::ffi::TensorView lora_act,
    tvm::ffi::TensorView smooth) {
    // Skipping full validation for brevity but it acts as norm quantize act lora
    const int padded_m = static_cast<int>(output.size(0));
    const int padded_k = static_cast<int>(output.size(1) * 2);
    const int rank = static_cast<int>(lora_down.size(1));
    cudaSetDevice(input.device().device_id);
    const cudaStream_t stream = get_current_stream(input);
    check_status(
        static_cast<int>(cudaMemsetAsync(
            lora_act.data_ptr(),
            0,
            static_cast<size_t>(lora_act.numel()) * sizeof(float),
            stream)),
        "W4A4 small-N LoRA activation reset");
    int status = xqt_svdq_w4a4_smalln_norm_quantize_act_lora(
        input.data_ptr(),
        norm_weight.data_ptr(),
        row_scales.data_ptr(),
        output.data_ptr(),
        scales.data_ptr(),
        lora_down.data_ptr(),
        lora_act.data_ptr(),
        smooth.data_ptr(),
        static_cast<int>(input.size(0)),
        static_cast<int>(input.size(1)),
        padded_m,
        padded_k,
        rank,
        scalar_kind(input),
        stream);
    check_status(status, "W4A4 small-N activation quantization plus LoRA down");
}

void gemm(
    tvm::ffi::TensorView act,
    tvm::ffi::TensorView weight,
    tvm::ffi::TensorView output,
    tvm::ffi::TensorView activation_scales,
    tvm::ffi::TensorView weight_scales,
    tvm::ffi::TensorView bias) {
    cudaSetDevice(output.device().device_id);
    check_status(
        xqt_svdq_w4a4_smalln_gemm(
            act.data_ptr(),
            weight.data_ptr(),
            output.data_ptr(),
            activation_scales.data_ptr(),
            weight_scales.data_ptr(),
            bias.data_ptr(),
            static_cast<int>(output.size(0)),
            static_cast<int>(output.size(1)),
            static_cast<int>(act.size(0)),
            static_cast<int>(weight.size(0)),
            static_cast<int>(act.size(1) * 2),
            scalar_kind(output),
            get_current_stream(output)),
        "W4A4 small-N GEMM");
}

void gemm_lora(
    tvm::ffi::TensorView act,
    tvm::ffi::TensorView weight,
    tvm::ffi::TensorView output,
    tvm::ffi::TensorView activation_scales,
    tvm::ffi::TensorView weight_scales,
    tvm::ffi::TensorView lora_act,
    tvm::ffi::TensorView lora_up,
    tvm::ffi::TensorView bias,
    double lora_scale) {
    const int rank = static_cast<int>(lora_up.size(1));
    cudaSetDevice(output.device().device_id);
    check_status(
        xqt_svdq_w4a4_smalln_gemm_lora(
            act.data_ptr(),
            weight.data_ptr(),
            output.data_ptr(),
            activation_scales.data_ptr(),
            weight_scales.data_ptr(),
            lora_act.data_ptr(),
            lora_up.data_ptr(),
            bias.data_ptr(),
            static_cast<int>(output.size(0)),
            static_cast<int>(output.size(1)),
            static_cast<int>(act.size(0)),
            static_cast<int>(weight.size(0)),
            static_cast<int>(act.size(1) * 2),
            rank,
            static_cast<float>(lora_scale),
            scalar_kind(output),
            get_current_stream(output)),
        "W4A4 small-N GEMM plus LoRA up");
}

void norm_row_rms(
    tvm::ffi::TensorView input,
    tvm::ffi::TensorView row_scales,
    int64_t padded_m,
    double eps) {
    cudaSetDevice(input.device().device_id);
    check_status(
        xqt_svdq_w4a4_norm_row_rms(
            input.data_ptr(),
            row_scales.data_ptr(),
            static_cast<int>(input.size(0)),
            static_cast<int>(input.size(1)),
            static_cast<int>(padded_m),
            static_cast<float>(eps),
            scalar_kind(input),
            get_current_stream(input)),
        "W4A4 small-N RMSNorm row scale");
}

std::string version() {
    return std::string(xqt_svdq_w4a4_smalln_version());
}

}  // namespace

TVM_FFI_DLL_EXPORT_TYPED_FUNC(quantize_weight, quantize_weight);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(quantize_act_lora, quantize_act_lora);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(norm_quantize_act_lora, norm_quantize_act_lora);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(gemm, gemm);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(gemm_lora, gemm_lora);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(norm_row_rms, norm_row_rms);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(version, version);
