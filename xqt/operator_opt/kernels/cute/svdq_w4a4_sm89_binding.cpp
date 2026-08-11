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
extern "C" int xqt_svdq_w4a4_quantize_rotated_act(
    const void*, void*, void*, int, int, int, int, int, int, cudaStream_t);
extern "C" int xqt_svdq_w4a4_gemm(
    const void*, const void*, void*, const void*, const void*, const void*, int, int, int, int, int, int, cudaStream_t);
extern "C" int xqt_svdq_w4a4_gemm_lora(
    const void*, const void*, void*, const void*, const void*, const void*, const void*, const void*, int, int, int, int,
    int, int, float, int, cudaStream_t);
extern "C" const char* xqt_svdq_w4a4_version();

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

void quantize_act_lora(
    torch::Tensor input,
    torch::Tensor output,
    torch::Tensor scales,
    torch::Tensor lora_down,
    torch::Tensor lora_act,
    torch::Tensor smooth) {
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
    c10::cuda::CUDAGuard guard(input.device());
    const cudaStream_t stream = current_stream(input);
    check_status(
        static_cast<int>(cudaMemsetAsync(
            lora_act.data_ptr(),
            0,
            static_cast<size_t>(lora_act.numel()) * sizeof(float),
            stream)),
        "W4A4 LoRA activation reset");
    check_status(
        xqt_svdq_w4a4_quantize_act_lora(
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
            stream),
        "W4A4 activation quantization plus LoRA down");
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

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    py::class_<BoundConvRotW4A4Linear>(module, "BoundConvRotW4A4Linear")
        .def("__call__", &BoundConvRotW4A4Linear::run);
    py::class_<BoundSVDQLinear>(module, "BoundSVDQLinear")
        .def("__call__", &BoundSVDQLinear::run);
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
    module.def("version", []() { return std::string(xqt_svdq_w4a4_version()); });
}
