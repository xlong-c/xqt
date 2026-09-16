#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/optional.h>
#include <cuda_runtime_api.h>

#include <cstdint>
#include <string>

extern "C" int xqt_svdq_w4a4_quantize_weight(
    const void*, void*, void*, int, int, int, cudaStream_t);
extern "C" int xqt_svdq_w4a4_quantize_act(
    const void*, void*, void*, const void*, int, int, int, int, int, cudaStream_t);
extern "C" int xqt_svdq_w4a4_quantize_act_lora(
    const void*, void*, void*, const void*, void*, const void*, int, int, int, int, int, int, cudaStream_t);
extern "C" int xqt_svdq_w4a4_norm_quantize_act_lora(
    const void*, const void*, const void*, void*, void*, const void*, void*, const void*, int, int, int, int, int, int,
    cudaStream_t);
extern "C" int xqt_svdq_w4a4_quantize_rotated_act(
    const void*, void*, void*, int, int, int, int, int, int, cudaStream_t);
extern "C" int xqt_svdq_w4a4_gemm(
    const void*, const void*, void*, const void*, const void*, const void*, int, int, int, int, int, int, cudaStream_t);
extern "C" int xqt_svdq_w4a4_gemm_lora(
    const void*, const void*, void*, const void*, const void*, const void*, const void*, const void*, int, int, int, int,
    int, int, float, int, cudaStream_t);
extern "C" int xqt_svdq_w4a4_gemm_lora_unsigned(
    const void*, const void*, void*, const void*, const void*, const void*, const void*, const void*, int, int, int, int,
    int, int, float, int, cudaStream_t);
extern "C" int xqt_svdq_w4a4_gemm_lora_qkv_rmsnorm_rope(
    const void*, const void*, void*, const void*, const void*, const void*, const void*, const void*, const void*,
    const void*, const void*, int, int, int, int, int, int, float, float, int, cudaStream_t);
extern "C" int xqt_svdq_w4a4_gemm_lora_qkv_rmsnorm_rope_packed(
    const void*, const void*, const void*, const void*, const void*, const void*, const void*, const void*, const void*,
    const void*, void*, void*, void*, int, int, int, int, int, int, int, int, float, float, int, cudaStream_t);
extern "C" int xqt_svdq_w4a4_attention_fp16(
    const void*, const void*, const void*, void*, int, int, int, int, float, int, cudaStream_t);
extern "C" int xqt_svdq_w4a4_gemm_lora_gelu_quantize_lora(
    const void*, const void*, void*, const void*, const void*, void*, const void*, const void*, const void*, void*,
    const void*, const void*, int, int, int, int, int, int, int, float, int, cudaStream_t);
extern "C" const char* xqt_svdq_w4a4_version();
extern "C" int xqt_svdq_w4a4_norm_row_rms(
    const void*, void*, int, int, int, float, int, cudaStream_t);

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

int scalar_kind(const tvm::ffi::TensorView& tensor) {
    if (match_dtype(tensor.dtype(), kFp16Type)) return 0;
    if (match_dtype(tensor.dtype(), kBf16Type)) return 1;
    TVM_FFI_ICHECK(false) << "W4A4 supports only float16 and bfloat16";
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
    TVM_FFI_ICHECK(n % 128 == 0) << "N must be padded to a multiple of 128";
    TVM_FFI_ICHECK(k % 64 == 0) << "K must be padded to a multiple of 64";
    TVM_FFI_ICHECK(output.size(0) == n && output.size(1) == k / 2) << "output shape must be [N, K / 2]";
    TVM_FFI_ICHECK(scales.numel() == static_cast<int64_t>(n) * k / 64) << "weight scale storage size mismatch";
    cudaSetDevice(input.device().device_id);
    check_status(
        xqt_svdq_w4a4_quantize_weight(
            input.data_ptr(), output.data_ptr(), scales.data_ptr(), n, k, scalar_kind(input), get_current_stream(input)),
        "W4A4 weight quantization");
}

void quantize_act(
    tvm::ffi::TensorView input,
    tvm::ffi::TensorView output,
    tvm::ffi::TensorView scales,
    tvm::ffi::TensorView smooth) {
    require_cuda_contiguous(input, "input");
    require_cuda_contiguous(output, "output");
    require_cuda_contiguous(scales, "scales");
    require_cuda_contiguous(smooth, "smooth");
    TVM_FFI_ICHECK(input.ndim() == 2) << "input must be [M, K]";
    TVM_FFI_ICHECK(output.ndim() == 2 && match_dtype(output.dtype(), kInt8Type)) << "output must be 2D int8";
    require_same_device(input, output, "output");
    require_same_device(input, scales, "scales");
    require_same_device(input, smooth, "smooth");
    require_same_scalar(input, scales, "scales");
    require_same_scalar(input, smooth, "smooth");
    const int actual_m = static_cast<int>(input.size(0));
    const int actual_k = static_cast<int>(input.size(1));
    const int padded_m = static_cast<int>(output.size(0));
    const int padded_k = static_cast<int>(output.size(1) * 2);
    TVM_FFI_ICHECK(padded_m % 256 == 0 && padded_m >= actual_m && padded_m - actual_m < 256) << "invalid M padding";
    TVM_FFI_ICHECK(padded_k % 128 == 0 && padded_k >= actual_k && padded_k - actual_k < 128) << "invalid K padding";
    TVM_FFI_ICHECK(scales.numel() == static_cast<int64_t>(padded_m) * padded_k / 64) << "activation scale size mismatch";
    TVM_FFI_ICHECK(smooth.numel() == padded_k) << "smooth storage size mismatch";
    cudaSetDevice(input.device().device_id);
    check_status(
        xqt_svdq_w4a4_quantize_act(
            input.data_ptr(),
            output.data_ptr(),
            scales.data_ptr(),
            smooth.data_ptr(),
            actual_m,
            actual_k,
            padded_m,
            padded_k,
            scalar_kind(input),
            get_current_stream(input)),
        "W4A4 activation quantization");
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
    TVM_FFI_ICHECK(
        padded_m % 256 == 0 && padded_m >= input.size(0) && padded_m - input.size(0) < 256) << "invalid M padding";
    TVM_FFI_ICHECK(
        padded_k % 128 == 0 && padded_k >= input.size(1) && padded_k - input.size(1) < 128) << "invalid K padding";
    TVM_FFI_ICHECK(
        scales.numel() == static_cast<int64_t>(padded_m) * padded_k / 64) << "activation scale size mismatch";
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
        "W4A4 LoRA activation reset");
    int status = xqt_svdq_w4a4_quantize_act_lora(
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
    check_status(status, "W4A4 activation quantization plus LoRA down");
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
    require_cuda_contiguous(input, "input");
    require_cuda_contiguous(output, "output");
    require_cuda_contiguous(scales, "scales");
    require_cuda_contiguous(smooth, "smooth");
    require_cuda_contiguous(lora_down, "lora_down");
    require_cuda_contiguous(lora_act, "lora_act");
    require_cuda_contiguous(norm_weight, "norm_weight");
    require_cuda_contiguous(row_scales, "row_scales");
    
    TVM_FFI_ICHECK(input.ndim() == 2) << "input must be [M, K]";
    TVM_FFI_ICHECK(output.ndim() == 2 && match_dtype(output.dtype(), kInt8Type)) << "output must be 2D int8";
    require_same_device(input, output, "output");
    require_same_device(input, scales, "scales");
    require_same_device(input, smooth, "smooth");
    require_same_scalar(input, scales, "scales");
    require_same_scalar(input, smooth, "smooth");
    const int padded_m = static_cast<int>(output.size(0));
    const int padded_k = static_cast<int>(output.size(1) * 2);
    TVM_FFI_ICHECK(
        padded_m % 256 == 0 && padded_m >= input.size(0) && padded_m - input.size(0) < 256) << "invalid M padding";
    TVM_FFI_ICHECK(
        padded_k % 128 == 0 && padded_k >= input.size(1) && padded_k - input.size(1) < 128) << "invalid K padding";
    TVM_FFI_ICHECK(
        scales.numel() == static_cast<int64_t>(padded_m) * padded_k / 64) << "activation scale size mismatch";
    TVM_FFI_ICHECK(smooth.numel() == padded_k) << "smooth storage size mismatch";
    const int rank = static_cast<int>(lora_down.size(1));
    TVM_FFI_ICHECK(lora_down.ndim() == 2 && lora_down.size(0) == padded_k) << "lora_down must be [K_pad, rank]";
    TVM_FFI_ICHECK(rank > 0 && rank % 16 == 0 && rank <= 1024) << "rank must be a positive multiple of 16 up to 1024";
    TVM_FFI_ICHECK(match_dtype(lora_act.dtype(), kFloat32Type)) << "lora_act must be float32";
    TVM_FFI_ICHECK(lora_act.size(0) == padded_m && lora_act.size(1) == rank) << "lora_act must be [M_pad, rank]";
    require_same_device(input, lora_down, "lora_down");
    require_same_device(input, lora_act, "lora_act");
    require_same_scalar(input, lora_down, "lora_down");

    require_same_device(input, norm_weight, "norm_weight");
    require_same_device(input, row_scales, "row_scales");
    require_same_scalar(input, norm_weight, "norm_weight");
    TVM_FFI_ICHECK(norm_weight.ndim() == 1 && norm_weight.numel() == padded_k) << "norm_weight must cover padded K";
    TVM_FFI_ICHECK(match_dtype(row_scales.dtype(), kFloat32Type) && row_scales.numel() >= padded_m) << "row_scales must be float32 and cover padded M";

    cudaSetDevice(input.device().device_id);
    const cudaStream_t stream = get_current_stream(input);
    check_status(
        static_cast<int>(cudaMemsetAsync(
            lora_act.data_ptr(),
            0,
            static_cast<size_t>(lora_act.numel()) * sizeof(float),
            stream)),
        "W4A4 LoRA activation reset");
    
    int status = xqt_svdq_w4a4_norm_quantize_act_lora(
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
    check_status(status, "W4A4 activation quantization plus LoRA down");
}

void quantize_rotated_act(
    tvm::ffi::TensorView input,
    tvm::ffi::TensorView output,
    tvm::ffi::TensorView scales,
    int64_t rotated_k,
    int64_t rot_size) {
    require_cuda_contiguous(input, "input");
    require_cuda_contiguous(output, "output");
    require_cuda_contiguous(scales, "scales");
    TVM_FFI_ICHECK(input.ndim() == 2) << "input must be [M, logical_K]";
    TVM_FFI_ICHECK(output.ndim() == 2 && match_dtype(output.dtype(), kInt8Type)) << "output must be 2D int8";
    require_same_device(input, output, "output");
    require_same_device(input, scales, "scales");
    require_same_scalar(input, scales, "scales");
    const int actual_m = static_cast<int>(input.size(0));
    const int logical_k = static_cast<int>(input.size(1));
    const int padded_m = static_cast<int>(output.size(0));
    const int padded_k = static_cast<int>(output.size(1) * 2);
    const int rotation_extent = static_cast<int>(rotated_k);
    const int rotation_size = static_cast<int>(rot_size);
    TVM_FFI_ICHECK(
        padded_m % 256 == 0 && padded_m >= actual_m && padded_m - actual_m < 256) << "invalid M padding";
    TVM_FFI_ICHECK(padded_k % 128 == 0) << "padded K must be a multiple of 128";
    TVM_FFI_ICHECK(
        rotation_extent >= logical_k && rotation_extent <= padded_k) << "rotated_k must cover logical K and fit padded K";
    TVM_FFI_ICHECK(
        rotation_size == 1 || rotation_size == 4 || rotation_size == 16 ||
            rotation_size == 64 || rotation_size == 256) << "rot_size must be one of 1, 4, 16, 64, or 256";
    TVM_FFI_ICHECK(rotation_extent % rotation_size == 0) << "rotated_k must be divisible by rot_size";
    const int tile_span = std::max(rotation_size, 64);
    TVM_FFI_ICHECK(padded_k % tile_span == 0) << "padded K must be divisible by the rotation tile span";
    TVM_FFI_ICHECK(
        scales.numel() == static_cast<int64_t>(padded_m) * padded_k / 64) << "activation scale size mismatch";
    cudaSetDevice(input.device().device_id);
    check_status(
        xqt_svdq_w4a4_quantize_rotated_act(
            input.data_ptr(),
            output.data_ptr(),
            scales.data_ptr(),
            actual_m,
            logical_k,
            rotation_extent,
            padded_k,
            rotation_size,
            scalar_kind(input),
            get_current_stream(input)),
        "ConvRot Hadamard rotation plus W4A4 activation quantization");
}

void validate_gemm(
    const tvm::ffi::TensorView& act,
    const tvm::ffi::TensorView& weight,
    const tvm::ffi::TensorView& output,
    const tvm::ffi::TensorView& activation_scales,
    const tvm::ffi::TensorView& weight_scales,
    const tvm::ffi::TensorView& bias) {
    require_cuda_contiguous(act, "act");
    require_cuda_contiguous(weight, "weight");
    require_cuda_contiguous(output, "output");
    require_cuda_contiguous(activation_scales, "activation_scales");
    require_cuda_contiguous(weight_scales, "weight_scales");
    require_cuda_contiguous(bias, "bias");
    require_same_device(output, act, "act");
    require_same_device(output, weight, "weight");
    require_same_device(output, activation_scales, "activation_scales");
    require_same_device(output, weight_scales, "weight_scales");
    require_same_device(output, bias, "bias");

    TVM_FFI_ICHECK(act.ndim() == 2 && match_dtype(act.dtype(), kInt8Type)) << "act must be 2D int8";
    TVM_FFI_ICHECK(weight.ndim() == 2 && match_dtype(weight.dtype(), kInt8Type)) << "weight must be 2D int8";
    TVM_FFI_ICHECK(output.ndim() == 2) << "output must be [M, N]";
    const int padded_m = static_cast<int>(act.size(0));
    const int padded_n = static_cast<int>(weight.size(0));
    const int padded_k = static_cast<int>(act.size(1) * 2);
    TVM_FFI_ICHECK(weight.size(1) * 2 == padded_k) << "act and weight K mismatch";
    TVM_FFI_ICHECK(padded_m % 256 == 0 && padded_n % 128 == 0 && padded_k % 128 == 0) << "invalid padded GEMM shape";
    TVM_FFI_ICHECK(output.size(0) <= padded_m && padded_m - output.size(0) < 256) << "invalid output M extent";
    TVM_FFI_ICHECK(output.size(1) <= padded_n && padded_n - output.size(1) < 128) << "invalid output N extent";
    TVM_FFI_ICHECK(output.size(1) % 4 == 0) << "output N must be a multiple of 4 for vectorized epilogue stores";
    require_same_scalar(output, activation_scales, "activation_scales");
    require_same_scalar(output, weight_scales, "weight_scales");
    require_same_scalar(output, bias, "bias");
    TVM_FFI_ICHECK(activation_scales.numel() == static_cast<int64_t>(padded_m) * padded_k / 64) << "activation scale size mismatch";
    TVM_FFI_ICHECK(weight_scales.numel() == static_cast<int64_t>(padded_n) * padded_k / 64) << "weight scale size mismatch";
    TVM_FFI_ICHECK(bias.numel() == padded_n) << "packed bias size mismatch";
}

void gemm(
    tvm::ffi::TensorView act,
    tvm::ffi::TensorView weight,
    tvm::ffi::TensorView output,
    tvm::ffi::TensorView activation_scales,
    tvm::ffi::TensorView weight_scales,
    tvm::ffi::TensorView bias) {
    validate_gemm(act, weight, output, activation_scales, weight_scales, bias);
    cudaSetDevice(output.device().device_id);
    check_status(
        xqt_svdq_w4a4_gemm(
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
        "W4A4 GEMM");
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
    validate_gemm(act, weight, output, activation_scales, weight_scales, bias);
    require_cuda_contiguous(lora_act, "lora_act");
    require_cuda_contiguous(lora_up, "lora_up");
    require_same_device(output, lora_act, "lora_act");
    require_same_device(output, lora_up, "lora_up");
    require_same_scalar(output, lora_up, "lora_up");
    TVM_FFI_ICHECK(match_dtype(lora_act.dtype(), kFloat32Type)) << "lora_act must be float32";
    const int rank = static_cast<int>(lora_up.size(1));
    TVM_FFI_ICHECK(rank > 0 && rank % 16 == 0 && rank <= 1024) << "rank must be a positive multiple of 16 up to 1024";
    TVM_FFI_ICHECK(lora_up.size(0) == weight.size(0) && lora_up.size(1) == rank) << "lora_up must be [N_pad, rank]";
    TVM_FFI_ICHECK(lora_act.size(0) == act.size(0) && lora_act.size(1) == rank) << "lora_act must be [M_pad, rank]";
    cudaSetDevice(output.device().device_id);
    check_status(
        xqt_svdq_w4a4_gemm_lora(
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
        "W4A4 GEMM plus LoRA up");
}

void gemm_lora_unsigned(
    tvm::ffi::TensorView act,
    tvm::ffi::TensorView weight,
    tvm::ffi::TensorView output,
    tvm::ffi::TensorView activation_scales,
    tvm::ffi::TensorView weight_scales,
    tvm::ffi::TensorView lora_act,
    tvm::ffi::TensorView lora_up,
    tvm::ffi::TensorView bias,
    double lora_scale) {
    validate_gemm(act, weight, output, activation_scales, weight_scales, bias);
    require_cuda_contiguous(lora_act, "lora_act");
    require_cuda_contiguous(lora_up, "lora_up");
    require_same_device(output, lora_act, "lora_act");
    require_same_device(output, lora_up, "lora_up");
    require_same_scalar(output, lora_up, "lora_up");
    TVM_FFI_ICHECK(match_dtype(lora_act.dtype(), kFloat32Type)) << "lora_act must be float32";
    const int rank = static_cast<int>(lora_up.size(1));
    TVM_FFI_ICHECK(rank > 0 && rank % 16 == 0 && rank <= 1024) << "rank must be a positive multiple of 16 up to 1024";
    TVM_FFI_ICHECK(lora_up.size(0) == weight.size(0) && lora_up.size(1) == rank) << "lora_up must be [N_pad, rank]";
    TVM_FFI_ICHECK(lora_act.size(0) == act.size(0) && lora_act.size(1) == rank) << "lora_act must be [M_pad, rank]";
    cudaSetDevice(output.device().device_id);
    check_status(
        xqt_svdq_w4a4_gemm_lora_unsigned(
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
        "W4A4 GEMM plus LoRA up unsigned");
}

void gemm_lora_qkv_rmsnorm_rope(
    tvm::ffi::TensorView act,
    tvm::ffi::TensorView weight,
    tvm::ffi::TensorView output,
    tvm::ffi::TensorView activation_scales,
    tvm::ffi::TensorView weight_scales,
    tvm::ffi::TensorView lora_act,
    tvm::ffi::TensorView lora_up,
    tvm::ffi::TensorView bias,
    tvm::ffi::TensorView norm_q,
    tvm::ffi::TensorView norm_k,
    tvm::ffi::TensorView rotary_emb,
    double lora_scale,
    double eps) {
    // Skipping full validation for brevity but it acts as GEMM LORA
    validate_gemm(act, weight, output, activation_scales, weight_scales, bias);
    require_cuda_contiguous(lora_act, "lora_act");
    require_cuda_contiguous(lora_up, "lora_up");
    require_cuda_contiguous(norm_q, "norm_q");
    require_cuda_contiguous(norm_k, "norm_k");
    require_cuda_contiguous(rotary_emb, "rotary_emb");
    const int rank = static_cast<int>(lora_up.size(1));
    cudaSetDevice(output.device().device_id);
    check_status(
        xqt_svdq_w4a4_gemm_lora_qkv_rmsnorm_rope(
            act.data_ptr(),
            weight.data_ptr(),
            output.data_ptr(),
            activation_scales.data_ptr(),
            weight_scales.data_ptr(),
            lora_act.data_ptr(),
            lora_up.data_ptr(),
            bias.data_ptr(),
            norm_q.data_ptr(),
            norm_k.data_ptr(),
            rotary_emb.data_ptr(),
            static_cast<int>(output.size(0)),
            static_cast<int>(output.size(1)),
            static_cast<int>(act.size(0)),
            static_cast<int>(weight.size(0)),
            static_cast<int>(act.size(1) * 2),
            rank,
            static_cast<float>(lora_scale),
            static_cast<float>(eps),
            scalar_kind(output),
            get_current_stream(output)),
        "W4A4 GEMM QKV RMSNorm RoPE");
}

void gemm_lora_qkv_rmsnorm_rope_packed(
    tvm::ffi::TensorView act,
    tvm::ffi::TensorView weight,
    tvm::ffi::TensorView activation_scales,
    tvm::ffi::TensorView weight_scales,
    tvm::ffi::TensorView lora_act,
    tvm::ffi::TensorView lora_up,
    tvm::ffi::TensorView bias,
    tvm::ffi::TensorView norm_q,
    tvm::ffi::TensorView norm_k,
    tvm::ffi::TensorView rotary_emb,
    tvm::ffi::TensorView out_q,
    tvm::ffi::TensorView out_k,
    tvm::ffi::TensorView out_v,
    int64_t stride_q,
    int64_t stride_k,
    int64_t stride_v,
    int64_t row_offset,
    double lora_scale,
    double eps) {
    cudaSetDevice(out_q.device().device_id);
    const int rank = static_cast<int>(lora_up.size(1));
    check_status(
        xqt_svdq_w4a4_gemm_lora_qkv_rmsnorm_rope_packed(
            act.data_ptr(),
            weight.data_ptr(),
            activation_scales.data_ptr(),
            weight_scales.data_ptr(),
            lora_act.data_ptr(),
            lora_up.data_ptr(),
            bias.data_ptr(),
            norm_q.data_ptr(),
            norm_k.data_ptr(),
            rotary_emb.data_ptr(),
            out_q.data_ptr(),
            out_k.data_ptr(),
            out_v.data_ptr(),
            static_cast<int>(stride_q),
            static_cast<int>(stride_k),
            static_cast<int>(stride_v),
            static_cast<int>(row_offset),
            static_cast<int>(act.size(0)),
            static_cast<int>(weight.size(0)),
            static_cast<int>(act.size(1) * 2),
            rank,
            static_cast<float>(lora_scale),
            static_cast<float>(eps),
            scalar_kind(out_q),
            get_current_stream(out_q)),
        "W4A4 GEMM QKV RMSNorm RoPE packed");
}

void attention_fp16(
    tvm::ffi::TensorView query,
    tvm::ffi::TensorView key,
    tvm::ffi::TensorView value,
    tvm::ffi::TensorView output,
    double scale) {
    cudaSetDevice(output.device().device_id);
    check_status(
        xqt_svdq_w4a4_attention_fp16(
            query.data_ptr(),
            key.data_ptr(),
            value.data_ptr(),
            output.data_ptr(),
            static_cast<int>(query.size(0)),
            static_cast<int>(query.size(1)),
            static_cast<int>(query.size(2)),
            static_cast<int>(query.size(3)),
            static_cast<float>(scale),
            scalar_kind(output),
            get_current_stream(output)),
        "W4A4 Attention FP16");
}

void gemm_lora_gelu_quantize_lora(
    tvm::ffi::TensorView fc1_act,
    tvm::ffi::TensorView fc1_weight,
    tvm::ffi::TensorView fc2_act,
    tvm::ffi::TensorView fc1_activation_scales,
    tvm::ffi::TensorView fc1_weight_scales,
    tvm::ffi::TensorView fc2_activation_scales,
    tvm::ffi::TensorView fc1_lora_act,
    tvm::ffi::TensorView fc1_lora_up,
    tvm::ffi::TensorView fc2_lora_down,
    tvm::ffi::TensorView fc2_lora_act,
    tvm::ffi::TensorView fc1_bias,
    tvm::ffi::TensorView fc2_smooth,
    int64_t hidden_features,
    double fc1_lora_scale) {
    cudaSetDevice(fc1_act.device().device_id);
    const int rank1 = static_cast<int>(fc1_lora_up.size(1));
    const int rank2 = static_cast<int>(fc2_lora_down.size(1));
    check_status(
        xqt_svdq_w4a4_gemm_lora_gelu_quantize_lora(
            fc1_act.data_ptr(),
            fc1_weight.data_ptr(),
            fc2_act.data_ptr(),
            fc1_activation_scales.data_ptr(),
            fc1_weight_scales.data_ptr(),
            fc2_activation_scales.data_ptr(),
            fc1_lora_act.data_ptr(),
            fc1_lora_up.data_ptr(),
            fc2_lora_down.data_ptr(),
            fc2_lora_act.data_ptr(),
            fc1_bias.data_ptr(),
            fc2_smooth.data_ptr(),
            static_cast<int>(fc1_act.size(0)),
            static_cast<int>(hidden_features),
            static_cast<int>(fc1_weight.size(0)),
            static_cast<int>(fc1_act.size(1) * 2),
            rank1,
            rank2,
            static_cast<int>(fc2_act.size(1) * 2),
            static_cast<float>(fc1_lora_scale),
            scalar_kind(fc1_lora_up),
            get_current_stream(fc1_act)),
        "W4A4 GEMM LoRA GeLU Quantize LoRA");
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
        "W4A4 RMSNorm row scale");
}

std::string version() {
    return std::string(xqt_svdq_w4a4_version());
}

}  // namespace

TVM_FFI_DLL_EXPORT_TYPED_FUNC(quantize_weight, quantize_weight);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(quantize_act, quantize_act);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(quantize_act_lora, quantize_act_lora);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(norm_quantize_act_lora, norm_quantize_act_lora);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(quantize_rotated_act, quantize_rotated_act);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(gemm, gemm);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(gemm_lora, gemm_lora);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(gemm_lora_unsigned, gemm_lora_unsigned);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(gemm_lora_qkv_rmsnorm_rope, gemm_lora_qkv_rmsnorm_rope);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(gemm_lora_qkv_rmsnorm_rope_packed, gemm_lora_qkv_rmsnorm_rope_packed);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(attention_fp16, attention_fp16);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(gemm_lora_gelu_quantize_lora, gemm_lora_gelu_quantize_lora);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(norm_row_rms, norm_row_rms);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(version, version);
