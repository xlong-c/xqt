#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/optional.h>
#include <cuda_runtime_api.h>

#include <cstdint>
#include <string>

extern "C" int xqt_convrot_w4a4_rowwise_quantize(
    const void*, void*, void*, int, int, int, cudaStream_t);
extern "C" int xqt_convrot_w4a4_rowwise_gemm(
    const void*, const void*, const void*, const void*, const void*, void*, int, int, int, int, cudaStream_t);
extern "C" const char* xqt_convrot_w4a4_rowwise_version();

namespace {

static const DLDataType kBf16Type{kDLBfloat, 16, 1};
static const DLDataType kFp16Type{kDLFloat, 16, 1};
static const DLDataType kInt8Type{kDLInt, 8, 1};
static const DLDataType kFloat32Type{kDLFloat, 32, 1};

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

int scalar_kind(const tvm::ffi::TensorView& tensor) {
    if (match_dtype(tensor.dtype(), kFp16Type)) return 0;
    if (match_dtype(tensor.dtype(), kBf16Type)) return 1;
    TVM_FFI_ICHECK(false) << "W4A4 rowwise requires float16 or bfloat16";
    return -1;
}

void quantize(
    tvm::ffi::TensorView input,
    tvm::ffi::TensorView act,
    tvm::ffi::TensorView activation_scales) {
    require_cuda_contiguous(input, "input");
    require_cuda_contiguous(act, "act");
    require_cuda_contiguous(activation_scales, "activation_scales");
    require_same_device(input, act, "act");
    require_same_device(input, activation_scales, "activation_scales");

    TVM_FFI_ICHECK(input.ndim() >= 2) << "input must be at least 2D";
    const int input_features = static_cast<int>(input.size(input.ndim() - 1));
    const int rows = static_cast<int>(input.numel() / input_features);

    TVM_FFI_ICHECK(rows > 0) << "rows must be positive";
    TVM_FFI_ICHECK(
        input_features >= 1024 && input_features <= 32768 &&
        (input_features == 1024 || input_features % 2048 == 0))
        << "input_features must be 1024 or a multiple of 2048 through 32768";

    TVM_FFI_ICHECK(match_dtype(act.dtype(), kInt8Type) &&
                   act.numel() == rows * (input_features / 2))
        << "act must be int8 [M, K / 2]";
    TVM_FFI_ICHECK(match_dtype(activation_scales.dtype(), kFloat32Type) &&
                   activation_scales.numel() == rows)
        << "activation_scales must be float32 [M]";

    const int kind = scalar_kind(input);
    cudaSetDevice(input.device().device_id);
    const cudaStream_t stream = get_current_stream(input);

    check_status(
        xqt_convrot_w4a4_rowwise_quantize(
            input.data_ptr(),
            act.data_ptr(),
            activation_scales.data_ptr(),
            rows,
            input_features,
            kind,
            stream),
        "rowwise ConvRot rotation plus INT4 quantization");
}

void gemm(
    tvm::ffi::TensorView act,
    tvm::ffi::TensorView weight,
    tvm::ffi::TensorView activation_scales,
    tvm::ffi::TensorView weight_scales,
    tvm::ffi::TensorView bias,
    tvm::ffi::TensorView output,
    int64_t output_features) {
    require_cuda_contiguous(act, "act");
    require_cuda_contiguous(weight, "weight");
    require_cuda_contiguous(activation_scales, "activation_scales");
    require_cuda_contiguous(weight_scales, "weight_scales");
    require_cuda_contiguous(bias, "bias");
    require_cuda_contiguous(output, "output");
    require_same_device(output, act, "act");
    require_same_device(output, weight, "weight");
    require_same_device(output, activation_scales, "activation_scales");
    require_same_device(output, weight_scales, "weight_scales");
    require_same_device(output, bias, "bias");

    const int output_feat = static_cast<int>(output_features);
    TVM_FFI_ICHECK(output_feat > 0 && output_feat % 8 == 0) << "invalid output_features";

    TVM_FFI_ICHECK(output.ndim() >= 2) << "output must be at least 2D";
    const int rows = static_cast<int>(output.numel() / output_feat);
    TVM_FFI_ICHECK(output.size(output.ndim() - 1) == output_feat)
        << "output trailing dimension must match output_features";
    const int input_features = static_cast<int>(act.numel() / rows) * 2;

    TVM_FFI_ICHECK(rows > 0) << "rows must be positive";
    TVM_FFI_ICHECK(
        input_features >= 1024 && input_features <= 32768 &&
        (input_features == 1024 || input_features % 2048 == 0))
        << "input_features must be 1024 or a multiple of 2048 through 32768";

    TVM_FFI_ICHECK(match_dtype(act.dtype(), kInt8Type)) << "act must be int8 [M, K / 2]";
    TVM_FFI_ICHECK(weight.ndim() == 2 && match_dtype(weight.dtype(), kInt8Type) &&
                   weight.size(0) == output_feat && weight.size(1) == input_features / 2)
        << "weight must be int8 [N, K / 2]";
    TVM_FFI_ICHECK(match_dtype(activation_scales.dtype(), kFloat32Type) &&
                   activation_scales.numel() == rows)
        << "activation_scales must be float32 [M]";
    TVM_FFI_ICHECK(match_dtype(weight_scales.dtype(), kFloat32Type) &&
                   weight_scales.numel() == output_feat)
        << "weight_scales must be float32 [N]";
    TVM_FFI_ICHECK(match_dtype(bias.dtype(), kFloat32Type) &&
                   bias.numel() == output_feat)
        << "bias must be float32 [N]";

    const int kind = scalar_kind(output);
    cudaSetDevice(output.device().device_id);
    const cudaStream_t stream = get_current_stream(output);

    check_status(
        xqt_convrot_w4a4_rowwise_gemm(
            act.data_ptr(),
            weight.data_ptr(),
            activation_scales.data_ptr(),
            weight_scales.data_ptr(),
            bias.data_ptr(),
            output.data_ptr(),
            rows,
            output_feat,
            input_features,
            kind,
            stream),
        "rowwise ConvRot CUTLASS W4A4 GEMM");
}

void linear(
    tvm::ffi::TensorView input,
    tvm::ffi::TensorView act,
    tvm::ffi::TensorView activation_scales,
    tvm::ffi::TensorView weight,
    tvm::ffi::TensorView weight_scales,
    tvm::ffi::TensorView bias,
    tvm::ffi::TensorView output,
    int64_t output_features) {
    quantize(input, act, activation_scales);
    gemm(act, weight, activation_scales, weight_scales, bias, output, output_features);
}

std::string version() {
    return std::string(xqt_convrot_w4a4_rowwise_version());
}

}  // namespace

TVM_FFI_DLL_EXPORT_TYPED_FUNC(quantize, quantize);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(gemm, gemm);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(linear, linear);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(version, version);
