#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/optional.h>
#include <cuda_runtime_api.h>

#include <cstdint>
#include <string>

extern "C" int xqt_awq_w4a16_sm89_pack(
    const void*, void*, int, int, cudaStream_t);
extern "C" int xqt_awq_w4a16_sm89_decode(
    const void*, const void*, const void*, const void*, void*, int, int, int, int, cudaStream_t);
extern "C" int xqt_awq_w4a16_sm89_decode_bias(
    const void*, const void*, const void*, const void*, const void*, void*, int, int, int, int, cudaStream_t);
extern "C" int xqt_awq_w4a8_sm89_decode(
    const void*, const void*, const void*, const float*, float, void*, int, int, int, cudaStream_t);
extern "C" int xqt_awq_w4a8_sm89_decode_bias(
    const void*, const void*, const void*, const float*, float, const void*, void*, int, int, int, cudaStream_t);
extern "C" const char* xqt_awq_w4a16_sm89_version();

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

int scalar_kind(const tvm::ffi::TensorView& tensor) {
    if (match_dtype(tensor.dtype(), kFp16Type)) return 0;
    if (match_dtype(tensor.dtype(), kBf16Type)) return 1;
    TVM_FFI_ICHECK(false) << "AWQ decode requires float16 or bfloat16";
    return -1;
}

void require_decode_contract(
    const tvm::ffi::TensorView& input,
    const tvm::ffi::TensorView& qweight,
    const tvm::ffi::TensorView& scales,
    const tvm::ffi::TensorView& scaled_zeros,
    const tvm::ffi::TensorView& output) {
    require_cuda_contiguous(input, "input");
    require_cuda_contiguous(qweight, "qweight");
    require_cuda_contiguous(scales, "scales");
    require_cuda_contiguous(scaled_zeros, "scaled_zeros");
    require_cuda_contiguous(output, "output");
    require_same_device(input, qweight, "qweight");
    require_same_device(input, scales, "scales");
    require_same_device(input, scaled_zeros, "scaled_zeros");
    require_same_device(input, output, "output");

    TVM_FFI_ICHECK(input.ndim() == 2) << "input must have shape [M,K]";
    TVM_FFI_ICHECK(
        match_dtype(input.dtype(), kFp16Type) ||
        match_dtype(input.dtype(), kBf16Type))
        << "input must be float16 or bfloat16";
    TVM_FFI_ICHECK(match_dtype(qweight.dtype(), kInt32Type)) << "qweight must be int32";
    TVM_FFI_ICHECK(qweight.ndim() == 2) << "qweight must have shape [N/4,K/2]";
    TVM_FFI_ICHECK(match_dtype(scales.dtype(), input.dtype())) << "scales dtype mismatch";
    TVM_FFI_ICHECK(match_dtype(scaled_zeros.dtype(), input.dtype())) << "scaled_zeros dtype mismatch";
    TVM_FFI_ICHECK(match_dtype(output.dtype(), input.dtype())) << "output dtype mismatch";

    const int64_t m = input.size(0);
    const int64_t k = input.size(1);
    const int64_t n = qweight.size(0) * 4;
    TVM_FFI_ICHECK(m >= 1 && m <= 8) << "AWQ decode requires 1 <= M <= 8";
    TVM_FFI_ICHECK(n > 0 && n % 8 == 0) << "AWQ decode requires N % 8 == 0";
    TVM_FFI_ICHECK(k > 0 && k % 64 == 0) << "AWQ decode requires K % 64 == 0";
    TVM_FFI_ICHECK(qweight.size(1) == k / 2) << "qweight K storage mismatch";
    TVM_FFI_ICHECK(scales.ndim() == 2) << "scales must have shape [K/64,N]";
    TVM_FFI_ICHECK(
        scales.size(0) == k / 64 && scales.size(1) == n)
        << "scales shape mismatch";
    TVM_FFI_ICHECK(
        scaled_zeros.ndim() == 2 && scaled_zeros.size(0) == scales.size(0) && scaled_zeros.size(1) == scales.size(1))
        << "scaled_zeros shape mismatch";
    TVM_FFI_ICHECK(
        output.ndim() == 2 && output.size(0) == m && output.size(1) == n)
        << "output shape mismatch";
}

void pack(tvm::ffi::TensorView canonical_qweight, tvm::ffi::TensorView output) {
    require_cuda_contiguous(canonical_qweight, "canonical_qweight");
    require_cuda_contiguous(output, "output");
    require_same_device(canonical_qweight, output, "output");
    
    TVM_FFI_ICHECK(match_dtype(canonical_qweight.dtype(), kUint8Type)) << "canonical_qweight must be uint8";
    TVM_FFI_ICHECK(canonical_qweight.ndim() == 2) << "canonical_qweight must have shape [N,K/2]";
    TVM_FFI_ICHECK(match_dtype(output.dtype(), kInt32Type)) << "output must be int32";
    TVM_FFI_ICHECK(output.ndim() == 2) << "output must have shape [N/4,K/2]";
    
    const int64_t n = canonical_qweight.size(0);
    const int64_t k = canonical_qweight.size(1) * 2;
    TVM_FFI_ICHECK(n > 0 && n % 8 == 0) << "AWQ prepack requires N % 8 == 0";
    TVM_FFI_ICHECK(k > 0 && k % 64 == 0) << "AWQ prepack requires K % 64 == 0";
    TVM_FFI_ICHECK(output.size(0) == n / 4 && output.size(1) == k / 2) << "output shape mismatch";

    cudaSetDevice(canonical_qweight.device().device_id);
    check_status(
        xqt_awq_w4a16_sm89_pack(
            canonical_qweight.data_ptr(),
            output.data_ptr(),
            static_cast<int>(n),
            static_cast<int>(k),
            get_current_stream(canonical_qweight)),
        "AWQ W4A16 prepack");
}

void decode_out(
    tvm::ffi::TensorView input,
    tvm::ffi::TensorView qweight,
    tvm::ffi::TensorView scales,
    tvm::ffi::TensorView scaled_zeros,
    tvm::ffi::TensorView output) {
    require_decode_contract(input, qweight, scales, scaled_zeros, output);
    cudaSetDevice(input.device().device_id);
    check_status(
        xqt_awq_w4a16_sm89_decode(
            input.data_ptr(),
            qweight.data_ptr(),
            scales.data_ptr(),
            scaled_zeros.data_ptr(),
            output.data_ptr(),
            static_cast<int>(input.size(0)),
            static_cast<int>(qweight.size(0) * 4),
            static_cast<int>(input.size(1)),
            scalar_kind(input),
            get_current_stream(input)),
        "AWQ W4A16 decode");
}

void require_bias(const tvm::ffi::TensorView& input, const tvm::ffi::TensorView& bias, int64_t n) {
    require_cuda_contiguous(bias, "bias");
    require_same_device(input, bias, "bias");
    TVM_FFI_ICHECK(match_dtype(bias.dtype(), input.dtype())) << "bias dtype mismatch";
    TVM_FFI_ICHECK(bias.ndim() == 1 && bias.size(0) == n) << "bias must have shape [N]";
}

void decode_bias_out(
    tvm::ffi::TensorView input,
    tvm::ffi::TensorView qweight,
    tvm::ffi::TensorView scales,
    tvm::ffi::TensorView scaled_zeros,
    tvm::ffi::TensorView bias,
    tvm::ffi::TensorView output) {
    require_decode_contract(input, qweight, scales, scaled_zeros, output);
    require_bias(input, bias, qweight.size(0) * 4);
    cudaSetDevice(input.device().device_id);
    check_status(
        xqt_awq_w4a16_sm89_decode_bias(
            input.data_ptr(),
            qweight.data_ptr(),
            scales.data_ptr(),
            scaled_zeros.data_ptr(),
            bias.data_ptr(),
            output.data_ptr(),
            static_cast<int>(input.size(0)),
            static_cast<int>(qweight.size(0) * 4),
            static_cast<int>(input.size(1)),
            scalar_kind(input),
            get_current_stream(input)),
        "AWQ W4A16 decode bias");
}

void decode_w4a8_out(
    tvm::ffi::TensorView input,
    tvm::ffi::TensorView qweight,
    tvm::ffi::TensorView scales,
    tvm::ffi::TensorView scale_a,
    tvm::ffi::TensorView output) {
    require_cuda_contiguous(input, "input");
    require_cuda_contiguous(qweight, "qweight");
    require_cuda_contiguous(scales, "scales");
    require_cuda_contiguous(scale_a, "scale_a");
    require_cuda_contiguous(output, "output");
    require_same_device(input, qweight, "qweight");
    require_same_device(input, scales, "scales");
    require_same_device(input, scale_a, "scale_a");
    require_same_device(input, output, "output");

    TVM_FFI_ICHECK(input.ndim() == 2) << "input must have shape [M,K]";
    TVM_FFI_ICHECK(match_dtype(input.dtype(), kInt8Type)) << "input must be int8";
    TVM_FFI_ICHECK(match_dtype(qweight.dtype(), kInt32Type)) << "qweight must be int32";
    TVM_FFI_ICHECK(qweight.ndim() == 2) << "qweight must have shape [N/4,K/2]";
    TVM_FFI_ICHECK(match_dtype(scale_a.dtype(), kFloat32Type)) << "scale_a must be float32";
    TVM_FFI_ICHECK(
        match_dtype(output.dtype(), kFp16Type) ||
        match_dtype(output.dtype(), kBf16Type))
        << "output must be float16 or bfloat16";
    TVM_FFI_ICHECK(match_dtype(scales.dtype(), output.dtype())) << "scales dtype mismatch";

    const int64_t n = qweight.size(0) * 4;
    const int64_t k = input.size(1);
    cudaSetDevice(input.device().device_id);
    check_status(
        xqt_awq_w4a8_sm89_decode(
            input.data_ptr(),
            qweight.data_ptr(),
            scales.data_ptr(),
            static_cast<const float*>(scale_a.data_ptr()),
            1.0f,
            output.data_ptr(),
            static_cast<int>(n),
            static_cast<int>(k),
            scalar_kind(output),
            get_current_stream(input)),
        "AWQ W4A8 decode");
}

void decode_w4a8_scalar_out(
    tvm::ffi::TensorView input,
    tvm::ffi::TensorView qweight,
    tvm::ffi::TensorView scales,
    double scale_a,
    tvm::ffi::TensorView output) {
    require_cuda_contiguous(input, "input");
    require_cuda_contiguous(qweight, "qweight");
    require_cuda_contiguous(scales, "scales");
    require_cuda_contiguous(output, "output");
    require_same_device(input, qweight, "qweight");
    require_same_device(input, scales, "scales");
    require_same_device(input, output, "output");

    TVM_FFI_ICHECK(input.ndim() == 2) << "input must have shape [M,K]";
    TVM_FFI_ICHECK(match_dtype(input.dtype(), kInt8Type)) << "input must be int8";
    TVM_FFI_ICHECK(match_dtype(qweight.dtype(), kInt32Type)) << "qweight must be int32";
    TVM_FFI_ICHECK(qweight.ndim() == 2) << "qweight must have shape [N/4,K/2]";
    TVM_FFI_ICHECK(
        match_dtype(output.dtype(), kFp16Type) ||
        match_dtype(output.dtype(), kBf16Type))
        << "output must be float16 or bfloat16";
    TVM_FFI_ICHECK(match_dtype(scales.dtype(), output.dtype())) << "scales dtype mismatch";

    const int64_t n = qweight.size(0) * 4;
    const int64_t k = input.size(1);
    cudaSetDevice(input.device().device_id);
    check_status(
        xqt_awq_w4a8_sm89_decode(
            input.data_ptr(),
            qweight.data_ptr(),
            scales.data_ptr(),
            nullptr,
            static_cast<float>(scale_a),
            output.data_ptr(),
            static_cast<int>(n),
            static_cast<int>(k),
            scalar_kind(output),
            get_current_stream(input)),
        "AWQ W4A8 decode");
}

void decode_w4a8_bias_out(
    tvm::ffi::TensorView input,
    tvm::ffi::TensorView qweight,
    tvm::ffi::TensorView scales,
    tvm::ffi::TensorView scale_a,
    tvm::ffi::TensorView residual,
    tvm::ffi::TensorView output) {
    require_cuda_contiguous(input, "input");
    require_cuda_contiguous(qweight, "qweight");
    require_cuda_contiguous(scales, "scales");
    require_cuda_contiguous(scale_a, "scale_a");
    require_cuda_contiguous(residual, "residual");
    require_cuda_contiguous(output, "output");
    require_same_device(input, qweight, "qweight");
    require_same_device(input, scales, "scales");
    require_same_device(input, scale_a, "scale_a");
    require_same_device(input, residual, "residual");
    require_same_device(input, output, "output");

    TVM_FFI_ICHECK(input.ndim() == 2) << "input must have shape [M,K]";
    TVM_FFI_ICHECK(match_dtype(input.dtype(), kInt8Type)) << "input must be int8";
    TVM_FFI_ICHECK(match_dtype(qweight.dtype(), kInt32Type)) << "qweight must be int32";
    TVM_FFI_ICHECK(qweight.ndim() == 2) << "qweight must have shape [N/4,K/2]";
    TVM_FFI_ICHECK(match_dtype(scale_a.dtype(), kFloat32Type)) << "scale_a must be float32";
    TVM_FFI_ICHECK(
        match_dtype(output.dtype(), kFp16Type) ||
        match_dtype(output.dtype(), kBf16Type))
        << "output must be float16 or bfloat16";
    TVM_FFI_ICHECK(match_dtype(scales.dtype(), output.dtype())) << "scales dtype mismatch";
    TVM_FFI_ICHECK(match_dtype(residual.dtype(), output.dtype())) << "residual dtype mismatch";

    const int64_t n = qweight.size(0) * 4;
    const int64_t k = input.size(1);
    cudaSetDevice(input.device().device_id);
    check_status(
        xqt_awq_w4a8_sm89_decode_bias(
            input.data_ptr(),
            qweight.data_ptr(),
            scales.data_ptr(),
            static_cast<const float*>(scale_a.data_ptr()),
            1.0f,
            residual.data_ptr(),
            output.data_ptr(),
            static_cast<int>(n),
            static_cast<int>(k),
            scalar_kind(output),
            get_current_stream(input)),
        "AWQ W4A8 decode bias");
}

void decode_w4a8_scalar_bias_out(
    tvm::ffi::TensorView input,
    tvm::ffi::TensorView qweight,
    tvm::ffi::TensorView scales,
    double scale_a,
    tvm::ffi::TensorView residual,
    tvm::ffi::TensorView output) {
    require_cuda_contiguous(input, "input");
    require_cuda_contiguous(qweight, "qweight");
    require_cuda_contiguous(scales, "scales");
    require_cuda_contiguous(residual, "residual");
    require_cuda_contiguous(output, "output");
    require_same_device(input, qweight, "qweight");
    require_same_device(input, scales, "scales");
    require_same_device(input, residual, "residual");
    require_same_device(input, output, "output");

    TVM_FFI_ICHECK(input.ndim() == 2) << "input must have shape [M,K]";
    TVM_FFI_ICHECK(match_dtype(input.dtype(), kInt8Type)) << "input must be int8";
    TVM_FFI_ICHECK(match_dtype(qweight.dtype(), kInt32Type)) << "qweight must be int32";
    TVM_FFI_ICHECK(qweight.ndim() == 2) << "qweight must have shape [N/4,K/2]";
    TVM_FFI_ICHECK(
        match_dtype(output.dtype(), kFp16Type) ||
        match_dtype(output.dtype(), kBf16Type))
        << "output must be float16 or bfloat16";
    TVM_FFI_ICHECK(match_dtype(scales.dtype(), output.dtype())) << "scales dtype mismatch";
    TVM_FFI_ICHECK(match_dtype(residual.dtype(), output.dtype())) << "residual dtype mismatch";

    const int64_t n = qweight.size(0) * 4;
    const int64_t k = input.size(1);
    cudaSetDevice(input.device().device_id);
    check_status(
        xqt_awq_w4a8_sm89_decode_bias(
            input.data_ptr(),
            qweight.data_ptr(),
            scales.data_ptr(),
            nullptr,
            static_cast<float>(scale_a),
            residual.data_ptr(),
            output.data_ptr(),
            static_cast<int>(n),
            static_cast<int>(k),
            scalar_kind(output),
            get_current_stream(input)),
        "AWQ W4A8 decode bias");
}

std::string version() {
    return std::string(xqt_awq_w4a16_sm89_version());
}

}  // namespace

TVM_FFI_DLL_EXPORT_TYPED_FUNC(pack, pack);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(decode_out, decode_out);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(decode_bias_out, decode_bias_out);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(decode_w4a8_out, decode_w4a8_out);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(decode_w4a8_scalar_out, decode_w4a8_scalar_out);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(decode_w4a8_bias_out, decode_w4a8_bias_out);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(decode_w4a8_scalar_bias_out, decode_w4a8_scalar_bias_out);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(version, version);
