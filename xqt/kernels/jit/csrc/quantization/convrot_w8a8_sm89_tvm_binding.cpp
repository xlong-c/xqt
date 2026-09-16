#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/optional.h>
#include <cuda_runtime_api.h>

#include <cstdint>
#include <string>

extern "C" int xqt_convrot_w8a8_quantize_weight(
    const void*, const void*, void*, int, int, cudaStream_t);
extern "C" int xqt_convrot_w8a8_quantize_rotated_act(
    const void*, void*, void*, int, int, int, int, int, int, const void*, float, cudaStream_t);
extern "C" int xqt_convrot_w8a8_gemm(
    const void*, const void*, void*, const void*, const void*, const void*, int,
    int, int, int, int, cudaStream_t);
extern "C" int xqt_convrot_w8a8_small_quantize(
    const void*, void*, void*, int, int, int, int, int, const void*, float, cudaStream_t);
extern "C" int xqt_convrot_w8a8_small_gemm(
    const void*, const void*, const void*, const void*, const void*, void*, int,
    int, int, cudaStream_t);
extern "C" int xqt_convrot_w8a8_fused_swiglu(
    const void*, void*, int, int, cudaStream_t);
extern "C" const char* xqt_convrot_w8a8_version();

namespace {

#if defined(XQT_W8A8_FP16)
static const DLDataType kComputeType{kDLFloat, 16, 1};
static constexpr const char* kComputeName = "float16";
#else
static const DLDataType kComputeType{kDLBfloat, 16, 1};
static constexpr const char* kComputeName = "bfloat16";
#endif

static const DLDataType kInt8Type{kDLInt, 8, 1};
static const DLDataType kFloat32Type{kDLFloat, 32, 1};

inline bool match_dtype(const DLDataType& a, const DLDataType& b) {
    return a.code == b.code && a.bits == b.bits && a.lanes == b.lanes;
}

void require_cuda_contiguous(const tvm::ffi::TensorView& tensor, const char* name) {
    TVM_FFI_ICHECK(tensor.device().device_type == kDLCUDA) << name << " must be CUDA";
    TVM_FFI_ICHECK(tensor.is_contiguous()) << name << " must be contiguous";
}

void require_compute_type(const tvm::ffi::TensorView& tensor, const char* name) {
    TVM_FFI_ICHECK(match_dtype(tensor.dtype(), kComputeType))
        << name << " must be " << kComputeName;
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
    require_compute_type(input, "input");
    require_compute_type(scales, "scales");
    require_same_device(input, scales, "scales");
    require_same_device(input, output, "output");
    TVM_FFI_ICHECK(input.ndim() == 2) << "input must be [N, K]";
    TVM_FFI_ICHECK(scales.ndim() == 1 && scales.size(0) == input.size(0))
        << "scales must be [N]";
    TVM_FFI_ICHECK(match_dtype(output.dtype(), kInt8Type)) << "output must be int8";
    TVM_FFI_ICHECK(output.size(0) == input.size(0) && output.size(1) == input.size(1))
        << "output must match input shape";
    const int n = static_cast<int>(input.size(0));
    const int k = static_cast<int>(input.size(1));
    TVM_FFI_ICHECK(n % 128 == 0 && k % 32 == 0) << "weight shape is misaligned";
    cudaSetDevice(input.device().device_id);
    check_status(
        xqt_convrot_w8a8_quantize_weight(
            input.data_ptr(),
            scales.data_ptr(),
            output.data_ptr(),
            n,
            k,
            get_current_stream(input)),
        "ConvRot W8A8 weight packing");
}

void quantize_rotated_act(
    tvm::ffi::TensorView input,
    tvm::ffi::TensorView output,
    tvm::ffi::TensorView scales,
    int64_t rotated_k,
    int64_t rot_size,
    tvm::ffi::Optional<tvm::ffi::TensorView> norm_weight = std::nullopt,
    double eps = 1e-6) {
    require_cuda_contiguous(input, "input");
    require_cuda_contiguous(output, "output");
    require_cuda_contiguous(scales, "scales");
    require_same_device(input, output, "output");
    require_same_device(input, scales, "scales");
    require_compute_type(input, "input");
    require_compute_type(scales, "scales");
    TVM_FFI_ICHECK(input.ndim() == 2) << "input must be [M, logical_K]";
    TVM_FFI_ICHECK(output.ndim() == 2 && match_dtype(output.dtype(), kInt8Type))
        << "output must be 2D int8";
    const int actual_m = static_cast<int>(input.size(0));
    const int logical_k = static_cast<int>(input.size(1));
    const int padded_m = static_cast<int>(output.size(0));
    const int padded_k = static_cast<int>(output.size(1));
    const int rotation_extent = static_cast<int>(rotated_k);
    const int rotation_size = static_cast<int>(rot_size);
    TVM_FFI_ICHECK(
        padded_m % 256 == 0 && padded_m >= actual_m && padded_m - actual_m < 256)
        << "invalid M padding";
    TVM_FFI_ICHECK(padded_k % 256 == 0) << "padded K must be a multiple of 256";
    TVM_FFI_ICHECK(rotation_extent >= logical_k && rotation_extent <= padded_k)
        << "rotated_k must cover logical K and fit padded K";
    TVM_FFI_ICHECK(rotation_size == 1 || rotation_size == 256)
        << "rot_size must be 1 or 256";
    TVM_FFI_ICHECK(rotation_size == 1 || rotation_extent % rotation_size == 0)
        << "rotated_k must be divisible by rot_size";
    TVM_FFI_ICHECK(scales.numel() == padded_m) << "activation scale size mismatch";
    const void* norm_weight_ptr = nullptr;
    if (norm_weight.has_value()) {
        auto nw = norm_weight.value();
        require_cuda_contiguous(nw, "norm_weight");
        require_compute_type(nw, "norm_weight");
        require_same_device(input, nw, "norm_weight");
        TVM_FFI_ICHECK(nw.ndim() == 1 && nw.size(0) == logical_k)
            << "norm_weight must be 1D [logical_k]";
        norm_weight_ptr = nw.data_ptr();
    }
    cudaSetDevice(input.device().device_id);
    check_status(
        xqt_convrot_w8a8_quantize_rotated_act(
            input.data_ptr(),
            output.data_ptr(),
            scales.data_ptr(),
            actual_m,
            logical_k,
            rotation_extent,
            padded_m,
            padded_k,
            rotation_size,
            norm_weight_ptr,
            static_cast<float>(eps),
            get_current_stream(input)),
        "ConvRot Hadamard rotation plus dynamic W8 activation packing");
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
    TVM_FFI_ICHECK(activation.ndim() == 2 && match_dtype(activation.dtype(), kInt8Type))
        << "activation must be 2D int8";
    TVM_FFI_ICHECK(weight.ndim() == 2 && match_dtype(weight.dtype(), kInt8Type))
        << "weight must be 2D int8";
    require_compute_type(output, "output");
    require_compute_type(activation_scales, "activation_scales");
    require_compute_type(weight_scales, "weight_scales");
    require_compute_type(bias, "bias");
    const int padded_m = static_cast<int>(activation.size(0));
    const int padded_n = static_cast<int>(weight.size(0));
    const int padded_k = static_cast<int>(activation.size(1));
    TVM_FFI_ICHECK(weight.size(1) == padded_k) << "activation and weight K mismatch";
    TVM_FFI_ICHECK(padded_m % 256 == 0 && padded_n % 128 == 0 && padded_k % 256 == 0)
        << "invalid padded GEMM shape";
    TVM_FFI_ICHECK(output.size(0) <= padded_m && padded_m - output.size(0) < 256)
        << "invalid output M";
    TVM_FFI_ICHECK(output.size(1) <= padded_n && padded_n - output.size(1) < 128)
        << "invalid output N";
    TVM_FFI_ICHECK(output.size(1) % 4 == 0)
        << "output N must be a multiple of 4 for vectorized epilogue stores";
    TVM_FFI_ICHECK(activation_scales.numel() == padded_m)
        << "activation scale storage mismatch";
    TVM_FFI_ICHECK(weight_scales.numel() == padded_n) << "weight scale storage mismatch";
    TVM_FFI_ICHECK(bias.numel() == padded_n) << "bias storage mismatch";
    cudaSetDevice(output.device().device_id);
    check_status(
        xqt_convrot_w8a8_gemm(
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
        "ConvRot W8A8 GEMM plus bias");
}

void small_quantize(
    tvm::ffi::TensorView input,
    tvm::ffi::TensorView output,
    tvm::ffi::TensorView scales,
    int64_t rotated_k,
    int64_t rot_size,
    tvm::ffi::Optional<tvm::ffi::TensorView> norm_weight = std::nullopt,
    double eps = 1e-6) {
    require_cuda_contiguous(input, "input");
    require_cuda_contiguous(output, "output");
    require_cuda_contiguous(scales, "scales");
    require_same_device(input, output, "output");
    require_same_device(input, scales, "scales");
    require_compute_type(input, "input");
    require_compute_type(scales, "scales");
    TVM_FFI_ICHECK(input.ndim() == 2) << "input must be [M, logical_K]";
    TVM_FFI_ICHECK(output.ndim() == 2 && match_dtype(output.dtype(), kInt8Type))
        << "output must be 2D int8";
    const int actual_m = static_cast<int>(input.size(0));
    const int logical_k = static_cast<int>(input.size(1));
    const int padded_k = static_cast<int>(output.size(1));
    const int rotation_extent = static_cast<int>(rotated_k);
    const int rotation_size = static_cast<int>(rot_size);
    TVM_FFI_ICHECK(actual_m > 0 && actual_m <= 128)
        << "small-M path requires 0 < M <= 128";
    TVM_FFI_ICHECK(output.size(0) >= actual_m && output.size(0) % 16 == 0)
        << "invalid small-M padding";
    TVM_FFI_ICHECK(
        rotation_size == 1 ||
        (rotation_size >= 2 && rotation_size <= 256 &&
         rotation_size % 4 == 0 &&
         ((rotation_size / 4) & (rotation_size / 4 - 1)) == 0))
        << "rot_size must be one or a power of four up to 256";
    TVM_FFI_ICHECK(rotation_extent >= logical_k && rotation_extent <= padded_k)
        << "rotated_k must cover logical K and fit padded K";
    TVM_FFI_ICHECK(rotation_size == 1 || rotation_extent % rotation_size == 0)
        << "rotated_k must be divisible by rot_size";
    TVM_FFI_ICHECK(scales.numel() >= actual_m)
        << "activation scale storage mismatch";
    const void* norm_weight_ptr = nullptr;
    if (norm_weight.has_value()) {
        auto nw = norm_weight.value();
        require_cuda_contiguous(nw, "norm_weight");
        require_compute_type(nw, "norm_weight");
        require_same_device(input, nw, "norm_weight");
        TVM_FFI_ICHECK(nw.ndim() == 1 && nw.size(0) == logical_k)
            << "norm_weight must be 1D [logical_k]";
        norm_weight_ptr = nw.data_ptr();
    }
    cudaSetDevice(input.device().device_id);
    check_status(
        xqt_convrot_w8a8_small_quantize(
            input.data_ptr(),
            output.data_ptr(),
            scales.data_ptr(),
            actual_m,
            logical_k,
            rotation_extent,
            padded_k,
            rotation_size,
            norm_weight_ptr,
            static_cast<float>(eps),
            get_current_stream(input)),
        "small-M ConvRot activation rotation and dynamic quantization");
}

void small_gemm(
    tvm::ffi::TensorView activation,
    tvm::ffi::TensorView activation_scales,
    tvm::ffi::TensorView qweight_t,
    tvm::ffi::TensorView weight_scales,
    tvm::ffi::TensorView bias,
    tvm::ffi::TensorView output) {
    require_cuda_contiguous(activation, "activation");
    require_cuda_contiguous(activation_scales, "activation_scales");
    require_cuda_contiguous(qweight_t, "qweight_t");
    require_cuda_contiguous(weight_scales, "weight_scales");
    require_cuda_contiguous(bias, "bias");
    require_cuda_contiguous(output, "output");
    require_same_device(activation, activation_scales, "activation_scales");
    require_same_device(activation, qweight_t, "qweight_t");
    require_same_device(activation, weight_scales, "weight_scales");
    require_same_device(activation, bias, "bias");
    require_same_device(activation, output, "output");
    require_compute_type(activation_scales, "activation_scales");
    require_compute_type(output, "output");
    TVM_FFI_ICHECK(activation.ndim() == 2 && match_dtype(activation.dtype(), kInt8Type))
        << "activation must be 2D int8";
    TVM_FFI_ICHECK(qweight_t.ndim() == 2 && match_dtype(qweight_t.dtype(), kInt8Type))
        << "qweight_t must be 2D int8 [K, N]";
    TVM_FFI_ICHECK(match_dtype(weight_scales.dtype(), kFloat32Type))
        << "weight_scales must be float32";
    TVM_FFI_ICHECK(match_dtype(bias.dtype(), kFloat32Type)) << "bias must be float32";
    const int actual_m = static_cast<int>(output.size(0));
    const int n = static_cast<int>(qweight_t.size(1));
    const int padded_k = static_cast<int>(activation.size(1));
    TVM_FFI_ICHECK(actual_m > 0 && actual_m <= 128)
        << "small-M path requires 0 < M <= 128";
    TVM_FFI_ICHECK(qweight_t.size(0) == padded_k)
        << "qweight_t and activation K mismatch";
    TVM_FFI_ICHECK(n > 0 && n % 4 == 0) << "output N must be a multiple of 4";
    TVM_FFI_ICHECK(weight_scales.numel() == n) << "weight scale size mismatch";
    TVM_FFI_ICHECK(bias.numel() == n) << "bias size mismatch";
    TVM_FFI_ICHECK(activation_scales.numel() >= actual_m)
        << "activation scale storage mismatch";
    cudaSetDevice(activation.device().device_id);
    check_status(
        xqt_convrot_w8a8_small_gemm(
            activation.data_ptr(),
            activation_scales.data_ptr(),
            qweight_t.data_ptr(),
            weight_scales.data_ptr(),
            bias.data_ptr(),
            output.data_ptr(),
            actual_m,
            n,
            padded_k,
            get_current_stream(activation)),
        "small-M ConvRot dense int8 GEMM");
}

void fused_swiglu(
    tvm::ffi::TensorView gate_up,
    tvm::ffi::TensorView output) {
    require_cuda_contiguous(gate_up, "gate_up");
    require_cuda_contiguous(output, "output");
    require_compute_type(gate_up, "gate_up");
    require_compute_type(output, "output");
    require_same_device(gate_up, output, "output");
    TVM_FFI_ICHECK(gate_up.ndim() == 2) << "gate_up must be 2D [M, 2*D]";
    TVM_FFI_ICHECK(output.ndim() == 2) << "output must be 2D [M, D]";
    const int m = static_cast<int>(gate_up.size(0));
    const int total_d = static_cast<int>(gate_up.size(1));
    TVM_FFI_ICHECK(total_d % 2 == 0) << "gate_up dim 1 must be even";
    const int d = total_d / 2;
    TVM_FFI_ICHECK(output.size(0) == m && output.size(1) == d) << "output shape mismatch";
    cudaSetDevice(gate_up.device().device_id);
    check_status(
        xqt_convrot_w8a8_fused_swiglu(
            gate_up.data_ptr(),
            output.data_ptr(),
            m,
            d,
            get_current_stream(gate_up)),
        "ConvRot fused SwiGLU");
}

std::string version() {
    return std::string(xqt_convrot_w8a8_version());
}

}  // namespace

TVM_FFI_DLL_EXPORT_TYPED_FUNC(quantize_weight, quantize_weight);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(quantize_rotated_act, quantize_rotated_act);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(gemm, gemm);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(small_quantize, small_quantize);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(small_gemm, small_gemm);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(fused_swiglu, fused_swiglu);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(version, version);
