#include <torch/extension.h>

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>

namespace py = pybind11;

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

int scalar_kind(const torch::Tensor& tensor) {
    if (tensor.scalar_type() == torch::kFloat16) {
        return 0;
    }
    if (tensor.scalar_type() == torch::kBFloat16) {
        return 1;
    }
    TORCH_CHECK(false, "native W4A4 supports only float16 and bfloat16 tensors");
    return -1;
}

void require_cuda_contiguous(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be CUDA");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void require_same_device(const torch::Tensor& reference, const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.device() == reference.device(), name, " must be on the same CUDA device");
}

void require_same_scalar(const torch::Tensor& reference, const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.scalar_type() == reference.scalar_type(), name, " must match the floating-point dtype");
}

cudaStream_t current_stream(const torch::Tensor& tensor) {
    return c10::cuda::getCurrentCUDAStream(tensor.get_device()).stream();
}

void check_status(int status, const char* operation) {
    TORCH_CHECK(
        status == static_cast<int>(cudaSuccess),
        operation,
        " failed: ",
        cudaGetErrorString(static_cast<cudaError_t>(status)));
}

void quantize_weight(torch::Tensor input, torch::Tensor output, torch::Tensor scales) {
    require_cuda_contiguous(input, "input");
    require_cuda_contiguous(output, "output");
    require_cuda_contiguous(scales, "scales");
    TORCH_CHECK(input.dim() == 2, "input must be [N, K]");
    TORCH_CHECK(output.scalar_type() == torch::kInt8, "output must be int8");
    require_same_device(input, output, "output");
    require_same_device(input, scales, "scales");
    require_same_scalar(input, scales, "scales");
    const int n = static_cast<int>(input.size(0));
    const int k = static_cast<int>(input.size(1));
    TORCH_CHECK(n % 128 == 0, "N must be padded to a multiple of 128");
    TORCH_CHECK(k % 64 == 0, "K must be padded to a multiple of 64");
    TORCH_CHECK(output.sizes() == torch::IntArrayRef({n, k / 2}), "output shape must be [N, K / 2]");
    TORCH_CHECK(scales.numel() == static_cast<int64_t>(n) * k / 64, "weight scale storage size mismatch");
    c10::cuda::CUDAGuard guard(input.device());
    check_status(
        xqt_svdq_w4a4_quantize_weight(
            input.data_ptr(), output.data_ptr(), scales.data_ptr(), n, k, scalar_kind(input), current_stream(input)),
        "W4A4 weight quantization");
}

void quantize_act(
    torch::Tensor input,
    torch::Tensor output,
    torch::Tensor scales,
    torch::Tensor smooth) {
    require_cuda_contiguous(input, "input");
    require_cuda_contiguous(output, "output");
    require_cuda_contiguous(scales, "scales");
    require_cuda_contiguous(smooth, "smooth");
    TORCH_CHECK(input.dim() == 2, "input must be [M, K]");
    TORCH_CHECK(output.dim() == 2 && output.scalar_type() == torch::kInt8, "output must be 2D int8");
    require_same_device(input, output, "output");
    require_same_device(input, scales, "scales");
    require_same_device(input, smooth, "smooth");
    require_same_scalar(input, scales, "scales");
    require_same_scalar(input, smooth, "smooth");
    const int actual_m = static_cast<int>(input.size(0));
    const int actual_k = static_cast<int>(input.size(1));
    const int padded_m = static_cast<int>(output.size(0));
    const int padded_k = static_cast<int>(output.size(1) * 2);
    TORCH_CHECK(padded_m % 256 == 0 && padded_m >= actual_m && padded_m - actual_m < 256, "invalid M padding");
    TORCH_CHECK(padded_k % 128 == 0 && padded_k >= actual_k && padded_k - actual_k < 128, "invalid K padding");
    TORCH_CHECK(scales.numel() == static_cast<int64_t>(padded_m) * padded_k / 64, "activation scale size mismatch");
    TORCH_CHECK(smooth.numel() == padded_k, "smooth storage size mismatch");
    c10::cuda::CUDAGuard guard(input.device());
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
            current_stream(input)),
        "W4A4 activation quantization");
}

void quantize_act_lora_impl(
    torch::Tensor input,
    torch::Tensor output,
    torch::Tensor scales,
    torch::Tensor lora_down,
    torch::Tensor lora_act,
    torch::Tensor smooth,
    const torch::Tensor* norm_weight,
    const torch::Tensor* row_scales) {
    require_cuda_contiguous(input, "input");
    require_cuda_contiguous(output, "output");
    require_cuda_contiguous(scales, "scales");
    require_cuda_contiguous(smooth, "smooth");
    require_cuda_contiguous(lora_down, "lora_down");
    require_cuda_contiguous(lora_act, "lora_act");
    TORCH_CHECK(input.dim() == 2, "input must be [M, K]");
    TORCH_CHECK(output.dim() == 2 && output.scalar_type() == torch::kInt8, "output must be 2D int8");
    require_same_device(input, output, "output");
    require_same_device(input, scales, "scales");
    require_same_device(input, smooth, "smooth");
    require_same_scalar(input, scales, "scales");
    require_same_scalar(input, smooth, "smooth");
    const int padded_m = static_cast<int>(output.size(0));
    const int padded_k = static_cast<int>(output.size(1) * 2);
    TORCH_CHECK(
        padded_m % 256 == 0 && padded_m >= input.size(0) && padded_m - input.size(0) < 256,
        "invalid M padding");
    TORCH_CHECK(
        padded_k % 128 == 0 && padded_k >= input.size(1) && padded_k - input.size(1) < 128,
        "invalid K padding");
    TORCH_CHECK(
        scales.numel() == static_cast<int64_t>(padded_m) * padded_k / 64,
        "activation scale size mismatch");
    TORCH_CHECK(smooth.numel() == padded_k, "smooth storage size mismatch");
    const int rank = static_cast<int>(lora_down.size(1));
    TORCH_CHECK(lora_down.dim() == 2 && lora_down.size(0) == padded_k, "lora_down must be [K_pad, rank]");
    TORCH_CHECK(rank > 0 && rank % 16 == 0 && rank <= 1024, "rank must be a positive multiple of 16 up to 1024");
    TORCH_CHECK(lora_act.scalar_type() == torch::kFloat32, "lora_act must be float32");
    TORCH_CHECK(lora_act.sizes() == torch::IntArrayRef({padded_m, rank}), "lora_act must be [M_pad, rank]");
    require_same_device(input, lora_down, "lora_down");
    require_same_device(input, lora_act, "lora_act");
    require_same_scalar(input, lora_down, "lora_down");
    if (norm_weight != nullptr || row_scales != nullptr) {
        TORCH_CHECK(norm_weight != nullptr && row_scales != nullptr, "norm state must be complete");
        require_cuda_contiguous(*norm_weight, "norm_weight");
        require_cuda_contiguous(*row_scales, "row_scales");
        require_same_device(input, *norm_weight, "norm_weight");
        require_same_device(input, *row_scales, "row_scales");
        require_same_scalar(input, *norm_weight, "norm_weight");
        TORCH_CHECK(
            norm_weight->dim() == 1 && norm_weight->numel() == padded_k,
            "norm_weight must cover padded K");
        TORCH_CHECK(
            row_scales->scalar_type() == torch::kFloat32 && row_scales->numel() >= padded_m,
            "row_scales must be float32 and cover padded M");
    }
    c10::cuda::CUDAGuard guard(input.device());
    const cudaStream_t stream = current_stream(input);
    check_status(
        static_cast<int>(cudaMemsetAsync(
            lora_act.data_ptr(),
            0,
            static_cast<size_t>(lora_act.numel()) * sizeof(float),
            stream)),
        "W4A4 LoRA activation reset");
    const int status = norm_weight == nullptr
        ? xqt_svdq_w4a4_quantize_act_lora(
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
              stream)
        : xqt_svdq_w4a4_norm_quantize_act_lora(
              input.data_ptr(),
              norm_weight->data_ptr(),
              row_scales->data_ptr(),
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

void quantize_act_lora(
    torch::Tensor input,
    torch::Tensor output,
    torch::Tensor scales,
    torch::Tensor lora_down,
    torch::Tensor lora_act,
    torch::Tensor smooth) {
    quantize_act_lora_impl(
        std::move(input),
        std::move(output),
        std::move(scales),
        std::move(lora_down),
        std::move(lora_act),
        std::move(smooth),
        nullptr,
        nullptr);
}

void quantize_rotated_act(
    torch::Tensor input,
    torch::Tensor output,
    torch::Tensor scales,
    int64_t rotated_k,
    int64_t rot_size) {
    require_cuda_contiguous(input, "input");
    require_cuda_contiguous(output, "output");
    require_cuda_contiguous(scales, "scales");
    TORCH_CHECK(input.dim() == 2, "input must be [M, logical_K]");
    TORCH_CHECK(output.dim() == 2 && output.scalar_type() == torch::kInt8, "output must be 2D int8");
    require_same_device(input, output, "output");
    require_same_device(input, scales, "scales");
    require_same_scalar(input, scales, "scales");
    const int actual_m = static_cast<int>(input.size(0));
    const int logical_k = static_cast<int>(input.size(1));
    const int padded_m = static_cast<int>(output.size(0));
    const int padded_k = static_cast<int>(output.size(1) * 2);
    const int rotation_extent = static_cast<int>(rotated_k);
    const int rotation_size = static_cast<int>(rot_size);
    TORCH_CHECK(
        padded_m % 256 == 0 && padded_m >= actual_m && padded_m - actual_m < 256,
        "invalid M padding");
    TORCH_CHECK(padded_k % 128 == 0, "padded K must be a multiple of 128");
    TORCH_CHECK(
        rotation_extent >= logical_k && rotation_extent <= padded_k,
        "rotated_k must cover logical K and fit padded K");
    TORCH_CHECK(
        rotation_size == 1 || rotation_size == 4 || rotation_size == 16 ||
            rotation_size == 64 || rotation_size == 256,
        "rot_size must be one of 1, 4, 16, 64, or 256");
    TORCH_CHECK(rotation_extent % rotation_size == 0, "rotated_k must be divisible by rot_size");
    const int tile_span = std::max(rotation_size, 64);
    TORCH_CHECK(padded_k % tile_span == 0, "padded K must be divisible by the rotation tile span");
    TORCH_CHECK(
        scales.numel() == static_cast<int64_t>(padded_m) * padded_k / 64,
        "activation scale size mismatch");
    c10::cuda::CUDAGuard guard(input.device());
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
            current_stream(input)),
        "ConvRot Hadamard rotation plus W4A4 activation quantization");
}

void validate_gemm(
    const torch::Tensor& act,
    const torch::Tensor& weight,
    const torch::Tensor& output,
    const torch::Tensor& activation_scales,
    const torch::Tensor& weight_scales,
    const torch::Tensor& bias) {
    for (const auto& item : {
             std::pair<const torch::Tensor*, const char*>{&act, "act"},
             {&weight, "weight"},
             {&output, "output"},
             {&activation_scales, "activation_scales"},
             {&weight_scales, "weight_scales"},
             {&bias, "bias"},
         }) {
        require_cuda_contiguous(*item.first, item.second);
        require_same_device(output, *item.first, item.second);
    }
    TORCH_CHECK(act.dim() == 2 && act.scalar_type() == torch::kInt8, "act must be 2D int8");
    TORCH_CHECK(weight.dim() == 2 && weight.scalar_type() == torch::kInt8, "weight must be 2D int8");
    TORCH_CHECK(output.dim() == 2, "output must be [M, N]");
    const int padded_m = static_cast<int>(act.size(0));
    const int padded_n = static_cast<int>(weight.size(0));
    const int padded_k = static_cast<int>(act.size(1) * 2);
    TORCH_CHECK(weight.size(1) * 2 == padded_k, "act and weight K mismatch");
    TORCH_CHECK(padded_m % 256 == 0 && padded_n % 128 == 0 && padded_k % 128 == 0, "invalid padded GEMM shape");
    TORCH_CHECK(output.size(0) <= padded_m && padded_m - output.size(0) < 256, "invalid output M extent");
    TORCH_CHECK(output.size(1) <= padded_n && padded_n - output.size(1) < 128, "invalid output N extent");
    TORCH_CHECK(output.size(1) % 4 == 0, "output N must be a multiple of 4 for vectorized epilogue stores");
    require_same_scalar(output, activation_scales, "activation_scales");
    require_same_scalar(output, weight_scales, "weight_scales");
    require_same_scalar(output, bias, "bias");
    TORCH_CHECK(activation_scales.numel() == static_cast<int64_t>(padded_m) * padded_k / 64, "activation scale size mismatch");
    TORCH_CHECK(weight_scales.numel() == static_cast<int64_t>(padded_n) * padded_k / 64, "weight scale size mismatch");
    TORCH_CHECK(bias.numel() == padded_n, "packed bias size mismatch");
}

void gemm(
    torch::Tensor act,
    torch::Tensor weight,
    torch::Tensor output,
    torch::Tensor activation_scales,
    torch::Tensor weight_scales,
    torch::Tensor bias) {
    validate_gemm(act, weight, output, activation_scales, weight_scales, bias);
    c10::cuda::CUDAGuard guard(output.device());
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
            current_stream(output)),
        "W4A4 GEMM");
}

void gemm_lora(
    torch::Tensor act,
    torch::Tensor weight,
    torch::Tensor output,
    torch::Tensor activation_scales,
    torch::Tensor weight_scales,
    torch::Tensor lora_act,
    torch::Tensor lora_up,
    torch::Tensor bias,
    double lora_scale) {
    validate_gemm(act, weight, output, activation_scales, weight_scales, bias);
    require_cuda_contiguous(lora_act, "lora_act");
    require_cuda_contiguous(lora_up, "lora_up");
    require_same_device(output, lora_act, "lora_act");
    require_same_device(output, lora_up, "lora_up");
    require_same_scalar(output, lora_up, "lora_up");
    TORCH_CHECK(lora_act.scalar_type() == torch::kFloat32, "lora_act must be float32");
    const int rank = static_cast<int>(lora_up.size(1));
    TORCH_CHECK(rank > 0 && rank % 16 == 0 && rank <= 1024, "rank must be a positive multiple of 16 up to 1024");
    TORCH_CHECK(lora_up.sizes() == torch::IntArrayRef({weight.size(0), rank}), "lora_up must be [N_pad, rank]");
    TORCH_CHECK(lora_act.sizes() == torch::IntArrayRef({act.size(0), rank}), "lora_act must be [M_pad, rank]");
    c10::cuda::CUDAGuard guard(output.device());
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
            current_stream(output)),
        "W4A4 GEMM plus LoRA up");
}

torch::Tensor allocate_output(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    int64_t output_features) {
    TORCH_CHECK(input.dim() == 2, "input must be [M, K]");
    TORCH_CHECK(
        output_features > 0 && output_features <= weight.size(0),
        "output_features must fit the padded weight extent");
    TORCH_CHECK(output_features % 4 == 0, "output_features must be a multiple of 4");
    return torch::empty({input.size(0), output_features}, input.options());
}

torch::Tensor linear(
    torch::Tensor input,
    torch::Tensor act,
    torch::Tensor activation_scales,
    torch::Tensor smooth,
    torch::Tensor weight,
    torch::Tensor weight_scales,
    torch::Tensor bias,
    int64_t output_features) {
    auto output = allocate_output(input, weight, output_features);
    quantize_act(input, act, activation_scales, smooth);
    gemm(act, weight, output, activation_scales, weight_scales, bias);
    return output;
}

torch::Tensor convrot_linear(
    torch::Tensor input,
    torch::Tensor act,
    torch::Tensor activation_scales,
    torch::Tensor weight,
    torch::Tensor weight_scales,
    torch::Tensor bias,
    int64_t rotated_k,
    int64_t rot_size,
    int64_t output_features) {
    auto output = allocate_output(input, weight, output_features);
    quantize_rotated_act(input, act, activation_scales, rotated_k, rot_size);
    gemm(act, weight, output, activation_scales, weight_scales, bias);
    return output;
}

class BoundConvRotW4A4Linear {
public:
    BoundConvRotW4A4Linear(
        torch::Tensor act,
        torch::Tensor activation_scales,
        torch::Tensor weight,
        torch::Tensor weight_scales,
        torch::Tensor bias,
        int64_t rows,
        int64_t logical_input_features,
        int64_t rotated_input_features,
        int64_t output_features,
        int64_t rot_size)
        : act_(std::move(act)),
          activation_scales_(std::move(activation_scales)),
          weight_(std::move(weight)),
          weight_scales_(std::move(weight_scales)),
          bias_(std::move(bias)),
          rows_(static_cast<int>(rows)),
          logical_k_(static_cast<int>(logical_input_features)),
          rotated_k_(static_cast<int>(rotated_input_features)),
          output_features_(static_cast<int>(output_features)),
          rot_size_(static_cast<int>(rot_size)),
          padded_m_(static_cast<int>(act_.size(0))),
          padded_n_(static_cast<int>(weight_.size(0))),
          padded_k_(static_cast<int>(act_.size(1) * 2)),
          scalar_kind_(scalar_kind(weight_scales_)) {
        for (const auto& item : {
                 std::pair<const torch::Tensor*, const char*>{&act_, "act"},
                 {&activation_scales_, "activation_scales"},
                 {&weight_, "weight"},
                 {&weight_scales_, "weight_scales"},
                 {&bias_, "bias"},
             }) {
            require_cuda_contiguous(*item.first, item.second);
            require_same_device(weight_, *item.first, item.second);
        }
        TORCH_CHECK(rows_ > 0, "rows must be positive");
        TORCH_CHECK(logical_k_ > 0 && logical_k_ <= rotated_k_, "invalid logical input features");
        TORCH_CHECK(rotated_k_ <= padded_k_, "rotated input features must fit padded K");
        TORCH_CHECK(
            output_features_ > 0 && output_features_ <= padded_n_ && output_features_ % 4 == 0,
            "invalid output_features");
        TORCH_CHECK(
            rot_size_ == 1 || rot_size_ == 4 || rot_size_ == 16 || rot_size_ == 64 ||
                rot_size_ == 256,
            "rot_size must be one of 1, 4, 16, 64, or 256");
        TORCH_CHECK(rotated_k_ % rot_size_ == 0, "rotated input features must be divisible by rot_size");
        TORCH_CHECK(act_.dim() == 2 && act_.scalar_type() == torch::kInt8, "act must be 2D int8");
        TORCH_CHECK(weight_.dim() == 2 && weight_.scalar_type() == torch::kInt8, "weight must be 2D int8");
        TORCH_CHECK(weight_.size(1) * 2 == padded_k_, "act and weight K mismatch");
        TORCH_CHECK(
            padded_m_ % 256 == 0 && padded_m_ >= rows_ && padded_m_ - rows_ < 256,
            "invalid padded M extent");
        TORCH_CHECK(padded_n_ % 128 == 0 && padded_k_ % 128 == 0, "invalid padded GEMM shape");
        TORCH_CHECK(
            padded_k_ % std::max(rot_size_, 64) == 0,
            "padded K must be divisible by the rotation tile span");
        require_same_scalar(weight_scales_, activation_scales_, "activation_scales");
        require_same_scalar(weight_scales_, bias_, "bias");
        TORCH_CHECK(
            activation_scales_.numel() == static_cast<int64_t>(padded_m_) * padded_k_ / 64,
            "activation scale size mismatch");
        TORCH_CHECK(
            weight_scales_.numel() == static_cast<int64_t>(padded_n_) * padded_k_ / 64,
            "weight scale size mismatch");
        TORCH_CHECK(bias_.numel() == padded_n_, "packed bias size mismatch");
    }

    torch::Tensor run(torch::Tensor input) const {
        auto contiguous_input = input.contiguous();
        require_cuda_contiguous(contiguous_input, "input");
        require_same_device(weight_, contiguous_input, "input");
        require_same_scalar(weight_scales_, contiguous_input, "input");
        TORCH_CHECK(
            contiguous_input.dim() == 2 && contiguous_input.size(0) == rows_ &&
                contiguous_input.size(1) == logical_k_,
            "input shape does not match the bound ConvRot runner");
        c10::cuda::CUDAGuard guard(contiguous_input.device());
        auto output = torch::empty({rows_, output_features_}, contiguous_input.options());
        const cudaStream_t stream = current_stream(contiguous_input);
        check_status(
            xqt_svdq_w4a4_quantize_rotated_act(
                contiguous_input.data_ptr(),
                act_.data_ptr(),
                activation_scales_.data_ptr(),
                rows_,
                logical_k_,
                rotated_k_,
                padded_k_,
                rot_size_,
                scalar_kind_,
                stream),
            "ConvRot Hadamard rotation plus W4A4 activation quantization");
        check_status(
            xqt_svdq_w4a4_gemm(
                act_.data_ptr(),
                weight_.data_ptr(),
                output.data_ptr(),
                activation_scales_.data_ptr(),
                weight_scales_.data_ptr(),
                bias_.data_ptr(),
                rows_,
                output_features_,
                padded_m_,
                padded_n_,
                padded_k_,
                scalar_kind_,
                stream),
            "W4A4 GEMM");
        return output;
    }

private:
    torch::Tensor act_;
    torch::Tensor activation_scales_;
    torch::Tensor weight_;
    torch::Tensor weight_scales_;
    torch::Tensor bias_;
    int rows_;
    int logical_k_;
    int rotated_k_;
    int output_features_;
    int rot_size_;
    int padded_m_;
    int padded_n_;
    int padded_k_;
    int scalar_kind_;
};

torch::Tensor svdq_linear(
    torch::Tensor input,
    torch::Tensor act,
    torch::Tensor activation_scales,
    torch::Tensor lora_down,
    torch::Tensor lora_act,
    torch::Tensor smooth,
    torch::Tensor weight,
    torch::Tensor weight_scales,
    torch::Tensor lora_up,
    torch::Tensor bias,
    int64_t output_features,
    double lora_scale) {
    auto output = allocate_output(input, weight, output_features);
    quantize_act_lora(
        input,
        act,
        activation_scales,
        lora_down,
        lora_act,
        smooth);
    gemm_lora(
        act,
        weight,
        output,
        activation_scales,
        weight_scales,
        lora_act,
        lora_up,
        bias,
        lora_scale);
    return output;
}

// RMSNorm-fused variant: input is the PRE-norm activation. The row scale and
// per-channel norm weight are applied to the shared half/BF16 fragment before
// both activation quantization and LoRA-down.
torch::Tensor svdq_linear_norm(
    torch::Tensor input,
    torch::Tensor norm_weight,
    torch::Tensor row_scales,
    torch::Tensor act,
    torch::Tensor activation_scales,
    torch::Tensor lora_down,
    torch::Tensor lora_act,
    torch::Tensor smooth,
    torch::Tensor weight,
    torch::Tensor weight_scales,
    torch::Tensor lora_up,
    torch::Tensor bias,
    int64_t output_features,
    double lora_scale,
    double eps) {
    auto output = allocate_output(input, weight, output_features);
    require_cuda_contiguous(row_scales, "row_scales");
    require_same_device(input, row_scales, "row_scales");
    TORCH_CHECK(row_scales.scalar_type() == torch::kFloat32, "row_scales must be float32");
    const int padded_m = static_cast<int>(act.size(0));
    TORCH_CHECK(row_scales.numel() >= padded_m, "row_scales must cover the padded M extent");
    {
        c10::cuda::CUDAGuard guard(input.device());
        check_status(
            xqt_svdq_w4a4_norm_row_rms(
                input.data_ptr(),
                row_scales.data_ptr(),
                static_cast<int>(input.size(0)),
                static_cast<int>(input.size(1)),
                padded_m,
                static_cast<float>(eps),
                scalar_kind(input),
                current_stream(input)),
            "W4A4 RMSNorm row scale");
    }
    quantize_act_lora_impl(
        input,
        act,
        activation_scales,
        lora_down,
        lora_act,
        smooth,
        &norm_weight,
        &row_scales);
    gemm_lora(
        act,
        weight,
        output,
        activation_scales,
        weight_scales,
        lora_act,
        lora_up,
        bias,
        lora_scale);
    return output;
}

class BoundSVDQLinear {
public:
    BoundSVDQLinear(
        torch::Tensor act,
        torch::Tensor activation_scales,
        torch::Tensor lora_down,
        torch::Tensor lora_act,
        torch::Tensor smooth,
        torch::Tensor weight,
        torch::Tensor weight_scales,
        torch::Tensor lora_up,
        torch::Tensor bias,
        int64_t rows,
        int64_t input_features,
        int64_t output_features,
        double lora_scale)
        : act_(std::move(act)),
          activation_scales_(std::move(activation_scales)),
          lora_down_(std::move(lora_down)),
          lora_act_(std::move(lora_act)),
          smooth_(std::move(smooth)),
          weight_(std::move(weight)),
          weight_scales_(std::move(weight_scales)),
          lora_up_(std::move(lora_up)),
          bias_(std::move(bias)),
          rows_(static_cast<int>(rows)),
          input_features_(static_cast<int>(input_features)),
          output_features_(static_cast<int>(output_features)),
          padded_m_(static_cast<int>(act_.size(0))),
          padded_n_(static_cast<int>(weight_.size(0))),
          padded_k_(static_cast<int>(act_.size(1) * 2)),
          rank_(static_cast<int>(lora_down_.size(1))),
          scalar_kind_(scalar_kind(weight_scales_)),
          lora_scale_(static_cast<float>(lora_scale)) {
        for (const auto& item : {
                 std::pair<const torch::Tensor*, const char*>{&act_, "act"},
                 {&activation_scales_, "activation_scales"},
                 {&lora_down_, "lora_down"},
                 {&lora_act_, "lora_act"},
                 {&smooth_, "smooth"},
                 {&weight_, "weight"},
                 {&weight_scales_, "weight_scales"},
                 {&lora_up_, "lora_up"},
                 {&bias_, "bias"},
             }) {
            require_cuda_contiguous(*item.first, item.second);
            require_same_device(weight_, *item.first, item.second);
        }
        TORCH_CHECK(rows_ > 0, "rows must be positive");
        TORCH_CHECK(input_features_ > 0 && input_features_ <= padded_k_, "invalid input_features");
        TORCH_CHECK(
            output_features_ > 0 && output_features_ <= padded_n_ && output_features_ % 4 == 0,
            "invalid output_features");
        TORCH_CHECK(act_.dim() == 2 && act_.scalar_type() == torch::kInt8, "act must be 2D int8");
        TORCH_CHECK(weight_.dim() == 2 && weight_.scalar_type() == torch::kInt8, "weight must be 2D int8");
        TORCH_CHECK(weight_.size(1) * 2 == padded_k_, "act and weight K mismatch");
        TORCH_CHECK(
            padded_m_ % 256 == 0 && padded_m_ >= rows_ && padded_m_ - rows_ < 256,
            "invalid padded M extent");
        TORCH_CHECK(padded_n_ % 128 == 0 && padded_k_ % 128 == 0, "invalid padded GEMM shape");
        require_same_scalar(weight_scales_, activation_scales_, "activation_scales");
        require_same_scalar(weight_scales_, lora_down_, "lora_down");
        require_same_scalar(weight_scales_, smooth_, "smooth");
        require_same_scalar(weight_scales_, lora_up_, "lora_up");
        require_same_scalar(weight_scales_, bias_, "bias");
        TORCH_CHECK(
            activation_scales_.numel() == static_cast<int64_t>(padded_m_) * padded_k_ / 64,
            "activation scale size mismatch");
        TORCH_CHECK(
            weight_scales_.numel() == static_cast<int64_t>(padded_n_) * padded_k_ / 64,
            "weight scale size mismatch");
        TORCH_CHECK(smooth_.numel() == padded_k_, "smooth storage size mismatch");
        TORCH_CHECK(bias_.numel() == padded_n_, "packed bias size mismatch");
        TORCH_CHECK(rank_ > 0 && rank_ % 16 == 0 && rank_ <= 1024, "invalid padded rank");
        TORCH_CHECK(
            lora_down_.sizes() == torch::IntArrayRef({padded_k_, rank_}),
            "lora_down must be [K_pad, rank]");
        TORCH_CHECK(lora_act_.scalar_type() == torch::kFloat32, "lora_act must be float32");
        TORCH_CHECK(
            lora_act_.sizes() == torch::IntArrayRef({padded_m_, rank_}),
            "lora_act must be [M_pad, rank]");
        TORCH_CHECK(
            lora_up_.sizes() == torch::IntArrayRef({padded_n_, rank_}),
            "lora_up must be [N_pad, rank]");
    }

    torch::Tensor run(torch::Tensor input) const {
        auto contiguous_input = input.contiguous();
        require_cuda_contiguous(contiguous_input, "input");
        require_same_device(weight_, contiguous_input, "input");
        require_same_scalar(weight_scales_, contiguous_input, "input");
        TORCH_CHECK(
            contiguous_input.dim() == 2 && contiguous_input.size(0) == rows_ &&
                contiguous_input.size(1) == input_features_,
            "input shape does not match the bound SVDQuant runner");
        c10::cuda::CUDAGuard guard(contiguous_input.device());
        auto output = torch::empty({rows_, output_features_}, contiguous_input.options());
        const cudaStream_t stream = current_stream(contiguous_input);
        check_status(
            static_cast<int>(cudaMemsetAsync(
                lora_act_.data_ptr(),
                0,
                static_cast<size_t>(lora_act_.numel()) * sizeof(float),
                stream)),
            "W4A4 LoRA activation reset");
        check_status(
            xqt_svdq_w4a4_quantize_act_lora(
                contiguous_input.data_ptr(),
                act_.data_ptr(),
                activation_scales_.data_ptr(),
                lora_down_.data_ptr(),
                lora_act_.data_ptr(),
                smooth_.data_ptr(),
                rows_,
                input_features_,
                padded_m_,
                padded_k_,
                rank_,
                scalar_kind_,
                stream),
            "W4A4 activation quantization plus LoRA down");
        check_status(
            xqt_svdq_w4a4_gemm_lora(
                act_.data_ptr(),
                weight_.data_ptr(),
                output.data_ptr(),
                activation_scales_.data_ptr(),
                weight_scales_.data_ptr(),
                lora_act_.data_ptr(),
                lora_up_.data_ptr(),
                bias_.data_ptr(),
                rows_,
                output_features_,
                padded_m_,
                padded_n_,
                padded_k_,
                rank_,
                lora_scale_,
                scalar_kind_,
                stream),
            "W4A4 GEMM plus LoRA up");
        return output;
    }

private:
    torch::Tensor act_;
    torch::Tensor activation_scales_;
    torch::Tensor lora_down_;
    torch::Tensor lora_act_;
    torch::Tensor smooth_;
    torch::Tensor weight_;
    torch::Tensor weight_scales_;
    torch::Tensor lora_up_;
    torch::Tensor bias_;
    int rows_;
    int input_features_;
    int output_features_;
    int padded_m_;
    int padded_n_;
    int padded_k_;
    int rank_;
    int scalar_kind_;
    float lora_scale_;
};

class BoundSVDQQKVRMSNormRope {
public:
    BoundSVDQQKVRMSNormRope(
        torch::Tensor act,
        torch::Tensor activation_scales,
        torch::Tensor lora_down,
        torch::Tensor lora_act,
        torch::Tensor smooth,
        torch::Tensor weight,
        torch::Tensor weight_scales,
        torch::Tensor lora_up,
        torch::Tensor bias,
        torch::Tensor norm_q,
        torch::Tensor norm_k,
        int64_t rows,
        int64_t input_features,
        int64_t output_features,
        double lora_scale,
        double eps)
        : act_(std::move(act)),
          activation_scales_(std::move(activation_scales)),
          lora_down_(std::move(lora_down)),
          lora_act_(std::move(lora_act)),
          smooth_(std::move(smooth)),
          weight_(std::move(weight)),
          weight_scales_(std::move(weight_scales)),
          lora_up_(std::move(lora_up)),
          bias_(std::move(bias)),
          norm_q_(std::move(norm_q)),
          norm_k_(std::move(norm_k)),
          rows_(static_cast<int>(rows)),
          input_features_(static_cast<int>(input_features)),
          output_features_(static_cast<int>(output_features)),
          padded_m_(static_cast<int>(act_.size(0))),
          padded_n_(static_cast<int>(weight_.size(0))),
          padded_k_(static_cast<int>(act_.size(1) * 2)),
          rank_(static_cast<int>(lora_down_.size(1))),
          scalar_kind_(scalar_kind(weight_scales_)),
          lora_scale_(static_cast<float>(lora_scale)),
          eps_(static_cast<float>(eps)) {
        for (const auto& item : {
                 std::pair<const torch::Tensor*, const char*>{&act_, "act"},
                 {&activation_scales_, "activation_scales"},
                 {&lora_down_, "lora_down"},
                 {&lora_act_, "lora_act"},
                 {&smooth_, "smooth"},
                 {&weight_, "weight"},
                 {&weight_scales_, "weight_scales"},
                 {&lora_up_, "lora_up"},
                 {&bias_, "bias"},
                 {&norm_q_, "norm_q"},
                 {&norm_k_, "norm_k"},
             }) {
            require_cuda_contiguous(*item.first, item.second);
            require_same_device(weight_, *item.first, item.second);
        }
        TORCH_CHECK(rows_ > 0, "rows must be positive");
        TORCH_CHECK(input_features_ > 0 && input_features_ <= padded_k_, "invalid input_features");
        TORCH_CHECK(
            output_features_ == padded_n_ && output_features_ % (3 * 128) == 0,
            "QKV output_features must equal padded N and be divisible by 384");
        TORCH_CHECK(act_.dim() == 2 && act_.scalar_type() == torch::kInt8, "act must be 2D int8");
        TORCH_CHECK(weight_.dim() == 2 && weight_.scalar_type() == torch::kInt8, "weight must be 2D int8");
        TORCH_CHECK(weight_.size(1) * 2 == padded_k_, "act and weight K mismatch");
        TORCH_CHECK(
            padded_m_ % 256 == 0 && padded_m_ >= rows_ && padded_m_ - rows_ < 256,
            "invalid padded M extent");
        TORCH_CHECK(padded_n_ % 128 == 0 && padded_k_ % 128 == 0, "invalid padded GEMM shape");
        require_same_scalar(weight_scales_, activation_scales_, "activation_scales");
        require_same_scalar(weight_scales_, lora_down_, "lora_down");
        require_same_scalar(weight_scales_, smooth_, "smooth");
        require_same_scalar(weight_scales_, lora_up_, "lora_up");
        require_same_scalar(weight_scales_, bias_, "bias");
        require_same_scalar(weight_scales_, norm_q_, "norm_q");
        require_same_scalar(weight_scales_, norm_k_, "norm_k");
        TORCH_CHECK(
            activation_scales_.numel() == static_cast<int64_t>(padded_m_) * padded_k_ / 64,
            "activation scale size mismatch");
        TORCH_CHECK(
            weight_scales_.numel() == static_cast<int64_t>(padded_n_) * padded_k_ / 64,
            "weight scale size mismatch");
        TORCH_CHECK(smooth_.numel() == padded_k_, "smooth storage size mismatch");
        TORCH_CHECK(bias_.numel() == padded_n_, "packed bias size mismatch");
        TORCH_CHECK(norm_q_.dim() == 1 && norm_q_.numel() == 128, "norm_q must have 128 elements");
        TORCH_CHECK(norm_k_.dim() == 1 && norm_k_.numel() == 128, "norm_k must have 128 elements");
        TORCH_CHECK(rank_ > 0 && rank_ % 16 == 0 && rank_ <= 1024, "invalid padded rank");
        TORCH_CHECK(
            lora_down_.sizes() == torch::IntArrayRef({padded_k_, rank_}),
            "lora_down must be [K_pad, rank]");
        TORCH_CHECK(lora_act_.scalar_type() == torch::kFloat32, "lora_act must be float32");
        TORCH_CHECK(
            lora_act_.sizes() == torch::IntArrayRef({padded_m_, rank_}),
            "lora_act must be [M_pad, rank]");
        TORCH_CHECK(
            lora_up_.sizes() == torch::IntArrayRef({padded_n_, rank_}),
            "lora_up must be [N_pad, rank]");
    }

    torch::Tensor run(torch::Tensor input, torch::Tensor rotary_emb) const {
        auto contiguous_input = input.contiguous();
        auto contiguous_rotary = rotary_emb.contiguous();
        require_cuda_contiguous(contiguous_input, "input");
        require_cuda_contiguous(contiguous_rotary, "rotary_emb");
        require_same_device(weight_, contiguous_input, "input");
        require_same_device(weight_, contiguous_rotary, "rotary_emb");
        require_same_scalar(weight_scales_, contiguous_input, "input");
        TORCH_CHECK(
            contiguous_input.dim() == 2 && contiguous_input.size(0) == rows_ &&
                contiguous_input.size(1) == input_features_,
            "input shape does not match the bound SVDQuant QKV runner");
        TORCH_CHECK(contiguous_rotary.scalar_type() == torch::kFloat32, "rotary_emb must be float32");
        TORCH_CHECK(
            contiguous_rotary.dim() >= 2 && contiguous_rotary.size(-1) == 128 &&
                contiguous_rotary.numel() == static_cast<int64_t>(padded_m_) * 128,
            "rotary_emb must be packed with shape [..., M_pad, 128]");
        c10::cuda::CUDAGuard guard(contiguous_input.device());
        auto output = torch::empty({rows_, output_features_}, contiguous_input.options());
        const cudaStream_t stream = current_stream(contiguous_input);
        check_status(
            static_cast<int>(cudaMemsetAsync(
                lora_act_.data_ptr(),
                0,
                static_cast<size_t>(lora_act_.numel()) * sizeof(float),
                stream)),
            "W4A4 QKV LoRA activation reset");
        check_status(
            xqt_svdq_w4a4_quantize_act_lora(
                contiguous_input.data_ptr(),
                act_.data_ptr(),
                activation_scales_.data_ptr(),
                lora_down_.data_ptr(),
                lora_act_.data_ptr(),
                smooth_.data_ptr(),
                rows_,
                input_features_,
                padded_m_,
                padded_k_,
                rank_,
                scalar_kind_,
                stream),
            "W4A4 QKV activation quantization plus LoRA down");
        check_status(
            xqt_svdq_w4a4_gemm_lora_qkv_rmsnorm_rope(
                act_.data_ptr(),
                weight_.data_ptr(),
                output.data_ptr(),
                activation_scales_.data_ptr(),
                weight_scales_.data_ptr(),
                lora_act_.data_ptr(),
                lora_up_.data_ptr(),
                bias_.data_ptr(),
                norm_q_.data_ptr(),
                norm_k_.data_ptr(),
                contiguous_rotary.data_ptr(),
                rows_,
                output_features_,
                padded_m_,
                padded_n_,
                padded_k_,
                rank_,
                lora_scale_,
                eps_,
                scalar_kind_,
                stream),
            "W4A4 QKV GEMM plus LoRA, RMSNorm, and RoPE");
        return output;
    }

    void run_packed(
        torch::Tensor input,
        torch::Tensor rotary_emb,
        torch::Tensor out_q,
        torch::Tensor out_k,
        torch::Tensor out_v,
        int64_t row_offset) const {
        auto contiguous_input = input.contiguous();
        auto contiguous_rotary = rotary_emb.contiguous();
        require_cuda_contiguous(contiguous_input, "input");
        require_cuda_contiguous(contiguous_rotary, "rotary_emb");
        require_cuda_contiguous(out_q, "out_q");
        require_cuda_contiguous(out_k, "out_k");
        require_cuda_contiguous(out_v, "out_v");
        require_same_device(weight_, contiguous_input, "input");
        require_same_device(weight_, contiguous_rotary, "rotary_emb");
        require_same_device(weight_, out_q, "out_q");
        require_same_device(weight_, out_k, "out_k");
        require_same_device(weight_, out_v, "out_v");
        require_same_scalar(weight_scales_, contiguous_input, "input");
        TORCH_CHECK(contiguous_rotary.scalar_type() == torch::kFloat32, "rotary_emb must be float32");
        TORCH_CHECK(out_q.scalar_type() == torch::kFloat16, "out_q must use float16 storage");
        TORCH_CHECK(out_k.scalar_type() == torch::kFloat16, "out_k must use float16 storage");
        TORCH_CHECK(out_v.scalar_type() == torch::kFloat16, "out_v must use float16 storage");
        TORCH_CHECK(out_q.dim() == 4 && out_k.dim() == 4 && out_v.dim() == 4, "packed QKV outputs must be 4D");
        TORCH_CHECK(out_q.sizes() == out_k.sizes() && out_q.sizes() == out_v.sizes(), "packed QKV output shapes must match");
        const int heads = padded_n_ / (3 * 128);
        TORCH_CHECK(out_q.size(0) == 1 && out_q.size(1) == heads && out_q.size(3) == 128, "invalid packed QKV output shape");
        const int storage_rows = static_cast<int>(out_q.size(2));
        TORCH_CHECK(row_offset >= 0 && row_offset + padded_m_ <= storage_rows, "packed QKV row offset is out of range");
        TORCH_CHECK(contiguous_input.dim() == 2 && contiguous_input.size(0) == rows_ && contiguous_input.size(1) == input_features_, "input shape does not match the bound SVDQuant QKV runner");
        TORCH_CHECK(contiguous_rotary.numel() == static_cast<int64_t>(padded_m_) * 128, "rotary_emb must cover the padded branch extent");
        const int packed_row_offset = static_cast<int>(row_offset) * 16;
        const int stride_head = static_cast<int>(out_q.stride(1) * out_q.element_size() / sizeof(uint4));
        c10::cuda::CUDAGuard guard(contiguous_input.device());
        const cudaStream_t stream = current_stream(contiguous_input);
        check_status(
            static_cast<int>(cudaMemsetAsync(
                lora_act_.data_ptr(),
                0,
                static_cast<size_t>(lora_act_.numel()) * sizeof(float),
                stream)),
            "W4A4 packed QKV LoRA activation reset");
        check_status(
            xqt_svdq_w4a4_quantize_act_lora(
                contiguous_input.data_ptr(),
                act_.data_ptr(),
                activation_scales_.data_ptr(),
                lora_down_.data_ptr(),
                lora_act_.data_ptr(),
                smooth_.data_ptr(),
                rows_,
                input_features_,
                padded_m_,
                padded_k_,
                rank_,
                scalar_kind_,
                stream),
            "W4A4 packed QKV activation quantization plus LoRA down");
        check_status(
            xqt_svdq_w4a4_gemm_lora_qkv_rmsnorm_rope_packed(
                act_.data_ptr(),
                weight_.data_ptr(),
                activation_scales_.data_ptr(),
                weight_scales_.data_ptr(),
                lora_act_.data_ptr(),
                lora_up_.data_ptr(),
                bias_.data_ptr(),
                norm_q_.data_ptr(),
                norm_k_.data_ptr(),
                contiguous_rotary.data_ptr(),
                static_cast<void*>(static_cast<uint4*>(out_q.data_ptr()) + packed_row_offset),
                static_cast<void*>(static_cast<uint4*>(out_k.data_ptr()) + packed_row_offset),
                static_cast<void*>(static_cast<uint4*>(out_v.data_ptr()) + packed_row_offset),
                stride_head,
                stride_head,
                stride_head,
                rows_,
                padded_m_,
                padded_n_,
                padded_k_,
                rank_,
                lora_scale_,
                eps_,
                scalar_kind_,
                stream),
            "W4A4 QKV GEMM plus LoRA, RMSNorm, RoPE, and packed output");
    }

private:
    torch::Tensor act_;
    torch::Tensor activation_scales_;
    torch::Tensor lora_down_;
    torch::Tensor lora_act_;
    torch::Tensor smooth_;
    torch::Tensor weight_;
    torch::Tensor weight_scales_;
    torch::Tensor lora_up_;
    torch::Tensor bias_;
    torch::Tensor norm_q_;
    torch::Tensor norm_k_;
    int rows_;
    int input_features_;
    int output_features_;
    int padded_m_;
    int padded_n_;
    int padded_k_;
    int rank_;
    int scalar_kind_;
    float lora_scale_;
    float eps_;
};

class BoundSVDQGeluMLP {
public:
    BoundSVDQGeluMLP(
        torch::Tensor fc1_act,
        torch::Tensor fc1_activation_scales,
        torch::Tensor fc1_lora_down,
        torch::Tensor fc1_lora_act,
        torch::Tensor fc1_smooth,
        torch::Tensor fc1_weight,
        torch::Tensor fc1_weight_scales,
        torch::Tensor fc1_lora_up,
        torch::Tensor fc1_bias,
        torch::Tensor fc2_act,
        torch::Tensor fc2_activation_scales,
        torch::Tensor fc2_lora_down,
        torch::Tensor fc2_lora_act,
        torch::Tensor fc2_smooth,
        torch::Tensor fc2_weight,
        torch::Tensor fc2_weight_scales,
        torch::Tensor fc2_lora_up,
        torch::Tensor fc2_bias,
        int64_t rows,
        int64_t input_features,
        int64_t hidden_features,
        int64_t output_features,
        double fc1_lora_scale,
        double fc2_lora_scale)
        : fc1_act_(std::move(fc1_act)),
          fc1_activation_scales_(std::move(fc1_activation_scales)),
          fc1_lora_down_(std::move(fc1_lora_down)),
          fc1_lora_act_(std::move(fc1_lora_act)),
          fc1_smooth_(std::move(fc1_smooth)),
          fc1_weight_(std::move(fc1_weight)),
          fc1_weight_scales_(std::move(fc1_weight_scales)),
          fc1_lora_up_(std::move(fc1_lora_up)),
          fc1_bias_(std::move(fc1_bias)),
          fc2_act_(std::move(fc2_act)),
          fc2_activation_scales_(std::move(fc2_activation_scales)),
          fc2_lora_down_(std::move(fc2_lora_down)),
          fc2_lora_act_(std::move(fc2_lora_act)),
          fc2_smooth_(std::move(fc2_smooth)),
          fc2_weight_(std::move(fc2_weight)),
          fc2_weight_scales_(std::move(fc2_weight_scales)),
          fc2_lora_up_(std::move(fc2_lora_up)),
          fc2_bias_(std::move(fc2_bias)),
          rows_(static_cast<int>(rows)),
          input_features_(static_cast<int>(input_features)),
          hidden_features_(static_cast<int>(hidden_features)),
          output_features_(static_cast<int>(output_features)),
          padded_m_(static_cast<int>(fc1_act_.size(0))),
          fc1_padded_n_(static_cast<int>(fc1_weight_.size(0))),
          fc1_padded_k_(static_cast<int>(fc1_act_.size(1) * 2)),
          fc2_padded_n_(static_cast<int>(fc2_weight_.size(0))),
          fc2_padded_k_(static_cast<int>(fc2_act_.size(1) * 2)),
          fc1_rank_(static_cast<int>(fc1_lora_down_.size(1))),
          fc2_rank_(static_cast<int>(fc2_lora_down_.size(1))),
          scalar_kind_(scalar_kind(fc1_weight_scales_)),
          fc1_lora_scale_(static_cast<float>(fc1_lora_scale)),
          fc2_lora_scale_(static_cast<float>(fc2_lora_scale)) {
        for (const auto& item : {
                 std::pair<const torch::Tensor*, const char*>{&fc1_act_, "fc1_act"},
                 {&fc1_activation_scales_, "fc1_activation_scales"},
                 {&fc1_lora_down_, "fc1_lora_down"},
                 {&fc1_lora_act_, "fc1_lora_act"},
                 {&fc1_smooth_, "fc1_smooth"},
                 {&fc1_weight_, "fc1_weight"},
                 {&fc1_weight_scales_, "fc1_weight_scales"},
                 {&fc1_lora_up_, "fc1_lora_up"},
                 {&fc1_bias_, "fc1_bias"},
                 {&fc2_act_, "fc2_act"},
                 {&fc2_activation_scales_, "fc2_activation_scales"},
                 {&fc2_lora_down_, "fc2_lora_down"},
                 {&fc2_lora_act_, "fc2_lora_act"},
                 {&fc2_smooth_, "fc2_smooth"},
                 {&fc2_weight_, "fc2_weight"},
                 {&fc2_weight_scales_, "fc2_weight_scales"},
                 {&fc2_lora_up_, "fc2_lora_up"},
                 {&fc2_bias_, "fc2_bias"},
             }) {
            require_cuda_contiguous(*item.first, item.second);
            require_same_device(fc1_weight_, *item.first, item.second);
        }
        TORCH_CHECK(rows_ > 0, "rows must be positive");
        TORCH_CHECK(
            input_features_ > 0 && input_features_ <= fc1_padded_k_,
            "invalid input_features");
        TORCH_CHECK(
            hidden_features_ > 0 && hidden_features_ <= fc1_padded_n_ &&
                hidden_features_ <= fc2_padded_k_ && hidden_features_ % 4 == 0,
            "invalid hidden_features");
        TORCH_CHECK(
            output_features_ > 0 && output_features_ <= fc2_padded_n_ &&
                output_features_ % 4 == 0,
            "invalid output_features");
        TORCH_CHECK(
            fc1_padded_n_ == fc2_padded_k_,
            "fc1 padded output extent must match fc2 padded input extent");
        TORCH_CHECK(
            padded_m_ % 256 == 0 && padded_m_ >= rows_ && padded_m_ - rows_ < 256,
            "invalid padded M extent");
        TORCH_CHECK(
            fc1_padded_n_ % 128 == 0 && fc1_padded_k_ % 128 == 0 &&
                fc2_padded_n_ % 128 == 0 && fc2_padded_k_ % 128 == 0,
            "invalid padded MLP GEMM shape");
        TORCH_CHECK(
            fc1_act_.dim() == 2 && fc1_act_.scalar_type() == torch::kInt8 &&
                fc2_act_.dim() == 2 && fc2_act_.scalar_type() == torch::kInt8,
            "MLP activation workspaces must be 2D int8");
        TORCH_CHECK(
            fc1_weight_.dim() == 2 && fc1_weight_.scalar_type() == torch::kInt8 &&
                fc2_weight_.dim() == 2 && fc2_weight_.scalar_type() == torch::kInt8,
            "MLP weights must be 2D int8");
        TORCH_CHECK(
            fc1_weight_.size(1) * 2 == fc1_padded_k_ &&
                fc2_weight_.size(1) * 2 == fc2_padded_k_,
            "MLP activation and weight K extents do not match");
        for (const auto& item : {
                 std::pair<const torch::Tensor*, const char*>{&fc1_activation_scales_, "fc1_activation_scales"},
                 {&fc1_lora_down_, "fc1_lora_down"},
                 {&fc1_smooth_, "fc1_smooth"},
                 {&fc1_lora_up_, "fc1_lora_up"},
                 {&fc1_bias_, "fc1_bias"},
                 {&fc2_activation_scales_, "fc2_activation_scales"},
                 {&fc2_lora_down_, "fc2_lora_down"},
                 {&fc2_smooth_, "fc2_smooth"},
                 {&fc2_weight_scales_, "fc2_weight_scales"},
                 {&fc2_lora_up_, "fc2_lora_up"},
                 {&fc2_bias_, "fc2_bias"},
             }) {
            require_same_scalar(fc1_weight_scales_, *item.first, item.second);
        }
        TORCH_CHECK(
            fc1_activation_scales_.numel() ==
                static_cast<int64_t>(padded_m_) * fc1_padded_k_ / 64,
            "fc1 activation scale size mismatch");
        TORCH_CHECK(
            fc1_weight_scales_.numel() ==
                static_cast<int64_t>(fc1_padded_n_) * fc1_padded_k_ / 64,
            "fc1 weight scale size mismatch");
        TORCH_CHECK(
            fc2_activation_scales_.numel() ==
                static_cast<int64_t>(padded_m_) * fc2_padded_k_ / 64,
            "fc2 activation scale size mismatch");
        TORCH_CHECK(
            fc2_weight_scales_.numel() ==
                static_cast<int64_t>(fc2_padded_n_) * fc2_padded_k_ / 64,
            "fc2 weight scale size mismatch");
        TORCH_CHECK(fc1_smooth_.numel() == fc1_padded_k_, "fc1 smooth storage size mismatch");
        TORCH_CHECK(fc2_smooth_.numel() == fc2_padded_k_, "fc2 smooth storage size mismatch");
        TORCH_CHECK(fc1_bias_.numel() == fc1_padded_n_, "fc1 bias size mismatch");
        TORCH_CHECK(fc2_bias_.numel() == fc2_padded_n_, "fc2 bias size mismatch");
        TORCH_CHECK(
            fc1_rank_ > 0 && fc1_rank_ % 16 == 0 && fc1_rank_ <= 1024,
            "invalid fc1 padded rank");
        TORCH_CHECK(
            fc2_rank_ > 0 && fc2_rank_ % 16 == 0 && fc2_rank_ <= 1024,
            "invalid fc2 padded rank");
        TORCH_CHECK(
            fc1_lora_down_.sizes() == torch::IntArrayRef({fc1_padded_k_, fc1_rank_}),
            "fc1_lora_down must be [K_pad, rank]");
        TORCH_CHECK(
            fc1_lora_up_.sizes() == torch::IntArrayRef({fc1_padded_n_, fc1_rank_}),
            "fc1_lora_up must be [N_pad, rank]");
        TORCH_CHECK(
            fc2_lora_down_.sizes() == torch::IntArrayRef({fc2_padded_k_, fc2_rank_}),
            "fc2_lora_down must be [K_pad, rank]");
        TORCH_CHECK(
            fc2_lora_up_.sizes() == torch::IntArrayRef({fc2_padded_n_, fc2_rank_}),
            "fc2_lora_up must be [N_pad, rank]");
        TORCH_CHECK(
            fc1_lora_act_.scalar_type() == torch::kFloat32 &&
                fc1_lora_act_.sizes() == torch::IntArrayRef({padded_m_, fc1_rank_}),
            "fc1_lora_act must be float32 [M_pad, rank]");
        TORCH_CHECK(
            fc2_lora_act_.scalar_type() == torch::kFloat32 &&
                fc2_lora_act_.sizes() == torch::IntArrayRef({padded_m_, fc2_rank_}),
            "fc2_lora_act must be float32 [M_pad, rank]");
    }

    torch::Tensor run(torch::Tensor input) const {
        auto contiguous_input = input.contiguous();
        require_cuda_contiguous(contiguous_input, "input");
        require_same_device(fc1_weight_, contiguous_input, "input");
        require_same_scalar(fc1_weight_scales_, contiguous_input, "input");
        TORCH_CHECK(
            contiguous_input.dim() == 2 && contiguous_input.size(0) == rows_ &&
                contiguous_input.size(1) == input_features_,
            "input shape does not match the bound SVDQuant GELU MLP runner");
        c10::cuda::CUDAGuard guard(contiguous_input.device());
        auto output = torch::empty({rows_, output_features_}, contiguous_input.options());
        const cudaStream_t stream = current_stream(contiguous_input);
        check_status(
            static_cast<int>(cudaMemsetAsync(
                fc1_lora_act_.data_ptr(),
                0,
                static_cast<size_t>(fc1_lora_act_.numel()) * sizeof(float),
                stream)),
            "W4A4 GELU MLP fc1 LoRA activation reset");
        check_status(
            xqt_svdq_w4a4_quantize_act_lora(
                contiguous_input.data_ptr(),
                fc1_act_.data_ptr(),
                fc1_activation_scales_.data_ptr(),
                fc1_lora_down_.data_ptr(),
                fc1_lora_act_.data_ptr(),
                fc1_smooth_.data_ptr(),
                rows_,
                input_features_,
                padded_m_,
                fc1_padded_k_,
                fc1_rank_,
                scalar_kind_,
                stream),
            "W4A4 GELU MLP fc1 activation quantization plus LoRA down");
        check_status(
            static_cast<int>(cudaMemsetAsync(
                fc2_lora_act_.data_ptr(),
                0,
                static_cast<size_t>(fc2_lora_act_.numel()) * sizeof(float),
                stream)),
            "W4A4 GELU MLP fc2 LoRA activation reset");
        check_status(
            xqt_svdq_w4a4_gemm_lora_gelu_quantize_lora(
                fc1_act_.data_ptr(),
                fc1_weight_.data_ptr(),
                fc2_act_.data_ptr(),
                fc1_activation_scales_.data_ptr(),
                fc1_weight_scales_.data_ptr(),
                fc2_activation_scales_.data_ptr(),
                fc1_lora_act_.data_ptr(),
                fc1_lora_up_.data_ptr(),
                fc2_lora_down_.data_ptr(),
                fc2_lora_act_.data_ptr(),
                fc1_bias_.data_ptr(),
                fc2_smooth_.data_ptr(),
                rows_,
                hidden_features_,
                padded_m_,
                fc1_padded_n_,
                fc1_padded_k_,
                fc1_rank_,
                fc2_rank_,
                fc1_lora_scale_,
                scalar_kind_,
                stream),
            "W4A4 GELU MLP fc1 GEMM plus fused GELU and fc2 quantization");
        check_status(
            xqt_svdq_w4a4_gemm_lora_unsigned(
                fc2_act_.data_ptr(),
                fc2_weight_.data_ptr(),
                output.data_ptr(),
                fc2_activation_scales_.data_ptr(),
                fc2_weight_scales_.data_ptr(),
                fc2_lora_act_.data_ptr(),
                fc2_lora_up_.data_ptr(),
                fc2_bias_.data_ptr(),
                rows_,
                output_features_,
                padded_m_,
                fc2_padded_n_,
                fc2_padded_k_,
                fc2_rank_,
                fc2_lora_scale_,
                scalar_kind_,
                stream),
            "W4A4 GELU MLP unsigned fc2 GEMM plus LoRA up");
        return output;
    }

private:
    torch::Tensor fc1_act_;
    torch::Tensor fc1_activation_scales_;
    torch::Tensor fc1_lora_down_;
    torch::Tensor fc1_lora_act_;
    torch::Tensor fc1_smooth_;
    torch::Tensor fc1_weight_;
    torch::Tensor fc1_weight_scales_;
    torch::Tensor fc1_lora_up_;
    torch::Tensor fc1_bias_;
    torch::Tensor fc2_act_;
    torch::Tensor fc2_activation_scales_;
    torch::Tensor fc2_lora_down_;
    torch::Tensor fc2_lora_act_;
    torch::Tensor fc2_smooth_;
    torch::Tensor fc2_weight_;
    torch::Tensor fc2_weight_scales_;
    torch::Tensor fc2_lora_up_;
    torch::Tensor fc2_bias_;
    int rows_;
    int input_features_;
    int hidden_features_;
    int output_features_;
    int padded_m_;
    int fc1_padded_n_;
    int fc1_padded_k_;
    int fc2_padded_n_;
    int fc2_padded_k_;
    int fc1_rank_;
    int fc2_rank_;
    int scalar_kind_;
    float fc1_lora_scale_;
    float fc2_lora_scale_;
};

class BoundSVDQNormLinear {
public:
    BoundSVDQNormLinear(
        torch::Tensor norm_weight,
        torch::Tensor row_scales,
        torch::Tensor act,
        torch::Tensor activation_scales,
        torch::Tensor lora_down,
        torch::Tensor lora_act,
        torch::Tensor smooth,
        torch::Tensor weight,
        torch::Tensor weight_scales,
        torch::Tensor lora_up,
        torch::Tensor bias,
        int64_t rows,
        int64_t input_features,
        int64_t output_features,
        double lora_scale,
        double eps)
        : norm_weight_(std::move(norm_weight)),
          row_scales_(std::move(row_scales)),
          act_(std::move(act)),
          activation_scales_(std::move(activation_scales)),
          lora_down_(std::move(lora_down)),
          lora_act_(std::move(lora_act)),
          smooth_(std::move(smooth)),
          weight_(std::move(weight)),
          weight_scales_(std::move(weight_scales)),
          lora_up_(std::move(lora_up)),
          bias_(std::move(bias)),
          rows_(static_cast<int>(rows)),
          input_features_(static_cast<int>(input_features)),
          output_features_(static_cast<int>(output_features)),
          padded_m_(static_cast<int>(act_.size(0))),
          padded_n_(static_cast<int>(weight_.size(0))),
          padded_k_(static_cast<int>(act_.size(1) * 2)),
          rank_(static_cast<int>(lora_down_.size(1))),
          scalar_kind_(scalar_kind(weight_scales_)),
          lora_scale_(static_cast<float>(lora_scale)),
          eps_(static_cast<float>(eps)) {
        for (const auto& item : {
                 std::pair<const torch::Tensor*, const char*>{&norm_weight_, "norm_weight"},
                 {&row_scales_, "row_scales"},
                 {&act_, "act"},
                 {&activation_scales_, "activation_scales"},
                 {&lora_down_, "lora_down"},
                 {&lora_act_, "lora_act"},
                 {&smooth_, "smooth"},
                 {&weight_, "weight"},
                 {&weight_scales_, "weight_scales"},
                 {&lora_up_, "lora_up"},
                 {&bias_, "bias"},
             }) {
            require_cuda_contiguous(*item.first, item.second);
            require_same_device(weight_, *item.first, item.second);
        }
        TORCH_CHECK(rows_ > 0, "rows must be positive");
        TORCH_CHECK(input_features_ > 0 && input_features_ <= padded_k_, "invalid input_features");
        TORCH_CHECK(
            output_features_ > 0 && output_features_ <= padded_n_ && output_features_ % 4 == 0,
            "invalid output_features");
        TORCH_CHECK(act_.dim() == 2 && act_.scalar_type() == torch::kInt8, "act must be 2D int8");
        TORCH_CHECK(weight_.dim() == 2 && weight_.scalar_type() == torch::kInt8, "weight must be 2D int8");
        TORCH_CHECK(weight_.size(1) * 2 == padded_k_, "act and weight K mismatch");
        TORCH_CHECK(
            padded_m_ % 256 == 0 && padded_m_ >= rows_ && padded_m_ - rows_ < 256,
            "invalid padded M extent");
        TORCH_CHECK(padded_n_ % 128 == 0 && padded_k_ % 128 == 0, "invalid padded GEMM shape");
        require_same_scalar(weight_scales_, activation_scales_, "activation_scales");
        require_same_scalar(weight_scales_, lora_down_, "lora_down");
        require_same_scalar(weight_scales_, smooth_, "smooth");
        require_same_scalar(weight_scales_, lora_up_, "lora_up");
        require_same_scalar(weight_scales_, bias_, "bias");
        TORCH_CHECK(
            activation_scales_.numel() == static_cast<int64_t>(padded_m_) * padded_k_ / 64,
            "activation scale size mismatch");
        TORCH_CHECK(
            weight_scales_.numel() == static_cast<int64_t>(padded_n_) * padded_k_ / 64,
            "weight scale size mismatch");
        TORCH_CHECK(smooth_.numel() == padded_k_, "smooth storage size mismatch");
        TORCH_CHECK(bias_.numel() == padded_n_, "packed bias size mismatch");
        TORCH_CHECK(rank_ > 0 && rank_ % 16 == 0 && rank_ <= 1024, "invalid padded rank");
        TORCH_CHECK(
            lora_down_.sizes() == torch::IntArrayRef({padded_k_, rank_}),
            "lora_down must be [K_pad, rank]");
        TORCH_CHECK(lora_act_.scalar_type() == torch::kFloat32, "lora_act must be float32");
        TORCH_CHECK(
            lora_act_.sizes() == torch::IntArrayRef({padded_m_, rank_}),
            "lora_act must be [M_pad, rank]");
        TORCH_CHECK(
            lora_up_.sizes() == torch::IntArrayRef({padded_n_, rank_}),
            "lora_up must be [N_pad, rank]");
        TORCH_CHECK(
            row_scales_.scalar_type() == torch::kFloat32 && row_scales_.numel() >= padded_m_,
            "row_scales must be float32 and cover the padded M extent");
        require_same_scalar(weight_scales_, norm_weight_, "norm_weight");
        TORCH_CHECK(
            norm_weight_.dim() == 1 && norm_weight_.numel() == padded_k_,
            "norm_weight must cover the padded K extent");
    }

    torch::Tensor run(torch::Tensor input) const {
        auto contiguous_input = input.contiguous();
        require_cuda_contiguous(contiguous_input, "input");
        require_same_device(weight_, contiguous_input, "input");
        require_same_scalar(weight_scales_, contiguous_input, "input");
        TORCH_CHECK(
            contiguous_input.dim() == 2 && contiguous_input.size(0) == rows_ &&
                contiguous_input.size(1) == input_features_,
            "input shape does not match the bound SVDQuant norm runner");
        c10::cuda::CUDAGuard guard(contiguous_input.device());
        auto output = torch::empty({rows_, output_features_}, contiguous_input.options());
        const cudaStream_t stream = current_stream(contiguous_input);
        check_status(
            xqt_svdq_w4a4_norm_row_rms(
                contiguous_input.data_ptr(),
                row_scales_.data_ptr(),
                rows_,
                input_features_,
                padded_m_,
                eps_,
                scalar_kind_,
                stream),
            "W4A4 RMSNorm row scale");
        check_status(
            static_cast<int>(cudaMemsetAsync(
                lora_act_.data_ptr(),
                0,
                static_cast<size_t>(lora_act_.numel()) * sizeof(float),
                stream)),
            "W4A4 LoRA activation reset");
        check_status(
            xqt_svdq_w4a4_norm_quantize_act_lora(
                contiguous_input.data_ptr(),
                norm_weight_.data_ptr(),
                row_scales_.data_ptr(),
                act_.data_ptr(),
                activation_scales_.data_ptr(),
                lora_down_.data_ptr(),
                lora_act_.data_ptr(),
                smooth_.data_ptr(),
                rows_,
                input_features_,
                padded_m_,
                padded_k_,
                rank_,
                scalar_kind_,
                stream),
            "W4A4 activation quantization plus LoRA down");
        check_status(
            xqt_svdq_w4a4_gemm_lora(
                act_.data_ptr(),
                weight_.data_ptr(),
                output.data_ptr(),
                activation_scales_.data_ptr(),
                weight_scales_.data_ptr(),
                lora_act_.data_ptr(),
                lora_up_.data_ptr(),
                bias_.data_ptr(),
                rows_,
                output_features_,
                padded_m_,
                padded_n_,
                padded_k_,
                rank_,
                lora_scale_,
                scalar_kind_,
                stream),
            "W4A4 GEMM plus LoRA up");
        return output;
    }

private:
    torch::Tensor norm_weight_;
    torch::Tensor row_scales_;
    torch::Tensor act_;
    torch::Tensor activation_scales_;
    torch::Tensor lora_down_;
    torch::Tensor lora_act_;
    torch::Tensor smooth_;
    torch::Tensor weight_;
    torch::Tensor weight_scales_;
    torch::Tensor lora_up_;
    torch::Tensor bias_;
    int rows_;
    int input_features_;
    int output_features_;
    int padded_m_;
    int padded_n_;
    int padded_k_;
    int rank_;
    int scalar_kind_;
    float lora_scale_;
    float eps_;
};

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    py::class_<BoundConvRotW4A4Linear>(module, "BoundConvRotW4A4Linear")
        .def("__call__", &BoundConvRotW4A4Linear::run);
    py::class_<BoundSVDQLinear>(module, "BoundSVDQLinear")
        .def("__call__", &BoundSVDQLinear::run);
    py::class_<BoundSVDQQKVRMSNormRope>(module, "BoundSVDQQKVRMSNormRope")
        .def("__call__", &BoundSVDQQKVRMSNormRope::run)
        .def("run_packed", &BoundSVDQQKVRMSNormRope::run_packed);
    py::class_<BoundSVDQGeluMLP>(module, "BoundSVDQGeluMLP")
        .def("__call__", &BoundSVDQGeluMLP::run);
    py::class_<BoundSVDQNormLinear>(module, "BoundSVDQNormLinear")
        .def("__call__", &BoundSVDQNormLinear::run);
    module.def("quantize_weight", &quantize_weight);
    module.def("quantize_act", &quantize_act);
    module.def("quantize_act_lora", &quantize_act_lora);
    module.def(
        "quantize_rotated_act",
        &quantize_rotated_act,
        py::arg("input"),
        py::arg("output"),
        py::arg("scales"),
        py::arg("rotated_k"),
        py::arg("rot_size"));
    module.def("gemm", &gemm);
    module.def(
        "attention_fp16",
        [](torch::Tensor query,
           torch::Tensor key,
           torch::Tensor value,
           torch::Tensor output,
           double scale) {
            require_cuda_contiguous(query, "query");
            require_cuda_contiguous(key, "key");
            require_cuda_contiguous(value, "value");
            require_cuda_contiguous(output, "output");
            TORCH_CHECK(query.dim() == 4 && key.dim() == 4 && value.dim() == 4, "attention inputs must be [B,H,S,D]");
            TORCH_CHECK(query.sizes() == key.sizes() && query.sizes() == value.sizes(), "attention Q/K/V shapes must match");
            TORCH_CHECK(query.size(0) > 0 && query.size(1) > 0 && query.size(3) == 128, "unsupported attention shape");
            TORCH_CHECK(query.scalar_type() == torch::kFloat16, "attention Q/K/V storage must be float16");
            TORCH_CHECK(output.dim() == 3 && output.size(0) == query.size(0) && output.size(1) == query.size(2) && output.size(2) == query.size(1) * 128, "invalid attention output shape");
            require_same_device(query, key, "key");
            require_same_device(query, value, "value");
            require_same_device(query, output, "output");
            c10::cuda::CUDAGuard guard(query.device());
            check_status(
                xqt_svdq_w4a4_attention_fp16(
                    query.data_ptr(),
                    key.data_ptr(),
                    value.data_ptr(),
                    output.data_ptr(),
                    static_cast<int>(query.size(0)),
                    static_cast<int>(query.size(1)),
                    static_cast<int>(query.size(2)),
                    static_cast<int>(key.size(2)),
                    static_cast<float>(scale),
                    scalar_kind(output),
                    current_stream(query)),
                "SM89 FP16 attention");
            return output;
        },
        py::arg("query"),
        py::arg("key"),
        py::arg("value"),
        py::arg("output"),
        py::arg("scale"));
    module.def("gemm_lora", &gemm_lora, py::arg("act"), py::arg("weight"), py::arg("output"),
               py::arg("activation_scales"), py::arg("weight_scales"), py::arg("lora_act"),
               py::arg("lora_up"), py::arg("bias"), py::arg("lora_scale") = 1.0);
    module.def(
        "linear",
        &linear,
        py::arg("input"),
        py::arg("act"),
        py::arg("activation_scales"),
        py::arg("smooth"),
        py::arg("weight"),
        py::arg("weight_scales"),
        py::arg("bias"),
        py::arg("output_features"));
    module.def(
        "convrot_linear",
        &convrot_linear,
        py::arg("input"),
        py::arg("act"),
        py::arg("activation_scales"),
        py::arg("weight"),
        py::arg("weight_scales"),
        py::arg("bias"),
        py::arg("rotated_k"),
        py::arg("rot_size"),
        py::arg("output_features"));
    module.def(
        "svdq_linear",
        &svdq_linear,
        py::arg("input"),
        py::arg("act"),
        py::arg("activation_scales"),
        py::arg("lora_down"),
        py::arg("lora_act"),
        py::arg("smooth"),
        py::arg("weight"),
        py::arg("weight_scales"),
        py::arg("lora_up"),
        py::arg("bias"),
        py::arg("output_features"),
        py::arg("lora_scale") = 1.0);
    module.def(
        "svdq_linear_norm",
        &svdq_linear_norm,
        py::arg("input"),
        py::arg("norm_weight"),
        py::arg("row_scales"),
        py::arg("act"),
        py::arg("activation_scales"),
        py::arg("lora_down"),
        py::arg("lora_act"),
        py::arg("smooth"),
        py::arg("weight"),
        py::arg("weight_scales"),
        py::arg("lora_up"),
        py::arg("bias"),
        py::arg("output_features"),
        py::arg("lora_scale") = 1.0,
        py::arg("eps") = 1e-6);
    module.def(
        "bind_svdq_linear_norm",
        [](torch::Tensor norm_weight,
           torch::Tensor row_scales,
           torch::Tensor act,
           torch::Tensor activation_scales,
           torch::Tensor lora_down,
           torch::Tensor lora_act,
           torch::Tensor smooth,
           torch::Tensor weight,
           torch::Tensor weight_scales,
           torch::Tensor lora_up,
           torch::Tensor bias,
           int64_t rows,
           int64_t input_features,
           int64_t output_features,
           double lora_scale,
           double eps) {
            return std::make_unique<BoundSVDQNormLinear>(
                std::move(norm_weight),
                std::move(row_scales),
                std::move(act),
                std::move(activation_scales),
                std::move(lora_down),
                std::move(lora_act),
                std::move(smooth),
                std::move(weight),
                std::move(weight_scales),
                std::move(lora_up),
                std::move(bias),
                rows,
                input_features,
                output_features,
                lora_scale,
                eps);
        },
        py::arg("norm_weight"),
        py::arg("row_scales"),
        py::arg("act"),
        py::arg("activation_scales"),
        py::arg("lora_down"),
        py::arg("lora_act"),
        py::arg("smooth"),
        py::arg("weight"),
        py::arg("weight_scales"),
        py::arg("lora_up"),
        py::arg("bias"),
        py::arg("rows"),
        py::arg("input_features"),
        py::arg("output_features"),
        py::arg("lora_scale") = 1.0,
        py::arg("eps") = 1e-6);
    module.def(
        "bind_convrot_linear",
        [](torch::Tensor act,
           torch::Tensor activation_scales,
           torch::Tensor weight,
           torch::Tensor weight_scales,
           torch::Tensor bias,
           int64_t rows,
           int64_t logical_input_features,
           int64_t rotated_input_features,
           int64_t output_features,
           int64_t rot_size) {
            return std::make_unique<BoundConvRotW4A4Linear>(
                std::move(act),
                std::move(activation_scales),
                std::move(weight),
                std::move(weight_scales),
                std::move(bias),
                rows,
                logical_input_features,
                rotated_input_features,
                output_features,
                rot_size);
        },
        py::arg("act"),
        py::arg("activation_scales"),
        py::arg("weight"),
        py::arg("weight_scales"),
        py::arg("bias"),
        py::arg("rows"),
        py::arg("logical_input_features"),
        py::arg("rotated_input_features"),
        py::arg("output_features"),
        py::arg("rot_size"));
    module.def(
        "bind_svdq_linear",
        [](torch::Tensor act,
           torch::Tensor activation_scales,
           torch::Tensor lora_down,
           torch::Tensor lora_act,
           torch::Tensor smooth,
           torch::Tensor weight,
           torch::Tensor weight_scales,
           torch::Tensor lora_up,
           torch::Tensor bias,
           int64_t rows,
           int64_t input_features,
           int64_t output_features,
           double lora_scale) {
            return std::make_unique<BoundSVDQLinear>(
                std::move(act),
                std::move(activation_scales),
                std::move(lora_down),
                std::move(lora_act),
                std::move(smooth),
                std::move(weight),
                std::move(weight_scales),
                std::move(lora_up),
                std::move(bias),
                rows,
                input_features,
                output_features,
                lora_scale);
        },
        py::arg("act"),
        py::arg("activation_scales"),
        py::arg("lora_down"),
        py::arg("lora_act"),
        py::arg("smooth"),
        py::arg("weight"),
        py::arg("weight_scales"),
        py::arg("lora_up"),
        py::arg("bias"),
        py::arg("rows"),
        py::arg("input_features"),
        py::arg("output_features"),
        py::arg("lora_scale") = 1.0);
    module.def(
        "bind_svdq_qkv_rmsnorm_rope",
        [](torch::Tensor act,
           torch::Tensor activation_scales,
           torch::Tensor lora_down,
           torch::Tensor lora_act,
           torch::Tensor smooth,
           torch::Tensor weight,
           torch::Tensor weight_scales,
           torch::Tensor lora_up,
           torch::Tensor bias,
           torch::Tensor norm_q,
           torch::Tensor norm_k,
           int64_t rows,
           int64_t input_features,
           int64_t output_features,
           double lora_scale,
           double eps) {
            return std::make_unique<BoundSVDQQKVRMSNormRope>(
                std::move(act),
                std::move(activation_scales),
                std::move(lora_down),
                std::move(lora_act),
                std::move(smooth),
                std::move(weight),
                std::move(weight_scales),
                std::move(lora_up),
                std::move(bias),
                std::move(norm_q),
                std::move(norm_k),
                rows,
                input_features,
                output_features,
                lora_scale,
                eps);
        },
        py::arg("act"),
        py::arg("activation_scales"),
        py::arg("lora_down"),
        py::arg("lora_act"),
        py::arg("smooth"),
        py::arg("weight"),
        py::arg("weight_scales"),
        py::arg("lora_up"),
        py::arg("bias"),
        py::arg("norm_q"),
        py::arg("norm_k"),
        py::arg("rows"),
        py::arg("input_features"),
        py::arg("output_features"),
        py::arg("lora_scale") = 1.0,
        py::arg("eps") = 1e-6);
    module.def(
        "bind_svdq_gelu_mlp",
        [](torch::Tensor fc1_act,
           torch::Tensor fc1_activation_scales,
           torch::Tensor fc1_lora_down,
           torch::Tensor fc1_lora_act,
           torch::Tensor fc1_smooth,
           torch::Tensor fc1_weight,
           torch::Tensor fc1_weight_scales,
           torch::Tensor fc1_lora_up,
           torch::Tensor fc1_bias,
           torch::Tensor fc2_act,
           torch::Tensor fc2_activation_scales,
           torch::Tensor fc2_lora_down,
           torch::Tensor fc2_lora_act,
           torch::Tensor fc2_smooth,
           torch::Tensor fc2_weight,
           torch::Tensor fc2_weight_scales,
           torch::Tensor fc2_lora_up,
           torch::Tensor fc2_bias,
           int64_t rows,
           int64_t input_features,
           int64_t hidden_features,
           int64_t output_features,
           double fc1_lora_scale,
           double fc2_lora_scale) {
            return std::make_unique<BoundSVDQGeluMLP>(
                std::move(fc1_act),
                std::move(fc1_activation_scales),
                std::move(fc1_lora_down),
                std::move(fc1_lora_act),
                std::move(fc1_smooth),
                std::move(fc1_weight),
                std::move(fc1_weight_scales),
                std::move(fc1_lora_up),
                std::move(fc1_bias),
                std::move(fc2_act),
                std::move(fc2_activation_scales),
                std::move(fc2_lora_down),
                std::move(fc2_lora_act),
                std::move(fc2_smooth),
                std::move(fc2_weight),
                std::move(fc2_weight_scales),
                std::move(fc2_lora_up),
                std::move(fc2_bias),
                rows,
                input_features,
                hidden_features,
                output_features,
                fc1_lora_scale,
                fc2_lora_scale);
        },
        py::arg("fc1_act"),
        py::arg("fc1_activation_scales"),
        py::arg("fc1_lora_down"),
        py::arg("fc1_lora_act"),
        py::arg("fc1_smooth"),
        py::arg("fc1_weight"),
        py::arg("fc1_weight_scales"),
        py::arg("fc1_lora_up"),
        py::arg("fc1_bias"),
        py::arg("fc2_act"),
        py::arg("fc2_activation_scales"),
        py::arg("fc2_lora_down"),
        py::arg("fc2_lora_act"),
        py::arg("fc2_smooth"),
        py::arg("fc2_weight"),
        py::arg("fc2_weight_scales"),
        py::arg("fc2_lora_up"),
        py::arg("fc2_bias"),
        py::arg("rows"),
        py::arg("input_features"),
        py::arg("hidden_features"),
        py::arg("output_features"),
        py::arg("fc1_lora_scale") = 1.0,
        py::arg("fc2_lora_scale") = 1.0);
    module.def("version", []() { return std::string(xqt_svdq_w4a4_version()); });
}
