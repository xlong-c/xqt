#include <torch/extension.h>

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
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

void require_cuda_contiguous(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be CUDA");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void require_bf16(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.scalar_type() == torch::kBFloat16, name, " must be bfloat16");
}

void require_same_device(
    const torch::Tensor& reference,
    const torch::Tensor& tensor,
    const char* name) {
    TORCH_CHECK(tensor.device() == reference.device(), name, " must share the CUDA device");
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

void quantize_weight(
    torch::Tensor input,
    torch::Tensor scales,
    torch::Tensor output) {
    require_cuda_contiguous(input, "input");
    require_cuda_contiguous(scales, "scales");
    require_cuda_contiguous(output, "output");
    require_bf16(input, "input");
    require_bf16(scales, "scales");
    TORCH_CHECK(input.dim() == 2, "input must be [N, K]");
    TORCH_CHECK(scales.dim() == 1 && scales.size(0) == input.size(0), "scales must be [N]");
    TORCH_CHECK(output.scalar_type() == torch::kInt8, "output must be int8");
    TORCH_CHECK(output.sizes() == input.sizes(), "output must match input shape");
    require_same_device(input, scales, "scales");
    require_same_device(input, output, "output");
    const int n = static_cast<int>(input.size(0));
    const int k = static_cast<int>(input.size(1));
    TORCH_CHECK(n % 128 == 0 && k % 32 == 0, "weight shape must align to N=128 and K=32");
    c10::cuda::CUDAGuard guard(input.device());
    check_status(
        xqt_svdq_w8a8_quantize_weight(
            input.data_ptr(), scales.data_ptr(), output.data_ptr(), n, k, current_stream(input)),
        "W8A8 weight quantization");
}

void quantize_act_lora(
    torch::Tensor input,
    torch::Tensor output,
    torch::Tensor scales,
    torch::Tensor lora_down,
    torch::Tensor lora_activation) {
    for (const auto& item : {
             std::pair<const torch::Tensor*, const char*>{&input, "input"},
             {&output, "output"},
             {&scales, "scales"},
             {&lora_down, "lora_down"},
             {&lora_activation, "lora_activation"},
         }) {
        require_cuda_contiguous(*item.first, item.second);
        require_same_device(input, *item.first, item.second);
    }
    require_bf16(input, "input");
    require_bf16(scales, "scales");
    require_bf16(lora_down, "lora_down");
    TORCH_CHECK(input.dim() == 2, "input must be [M, K]");
    TORCH_CHECK(output.dim() == 2 && output.scalar_type() == torch::kInt8, "output must be 2D int8");
    const int actual_m = static_cast<int>(input.size(0));
    const int actual_k = static_cast<int>(input.size(1));
    const int padded_m = static_cast<int>(output.size(0));
    const int padded_k = static_cast<int>(output.size(1));
    TORCH_CHECK(
        padded_m % 256 == 0 && padded_m >= actual_m && padded_m - actual_m < 256,
        "invalid M padding");
    TORCH_CHECK(
        padded_k % 128 == 0 && padded_k >= actual_k && padded_k - actual_k < 128,
        "invalid K padding");
    TORCH_CHECK(scales.numel() == padded_m, "activation scales must contain M_pad values");
    TORCH_CHECK(lora_down.dim() == 2 && lora_down.size(0) == padded_k, "lora_down must be [K_pad, rank]");
    const int rank = static_cast<int>(lora_down.size(1));
    TORCH_CHECK(rank > 0 && rank % 16 == 0 && rank <= 1024, "rank must align to 16 and be <= 1024");
    TORCH_CHECK(lora_activation.scalar_type() == torch::kFloat32, "lora_activation must be float32");
    TORCH_CHECK(
        lora_activation.sizes() == torch::IntArrayRef({padded_m, rank}),
        "lora_activation must be [M_pad, rank]");
    c10::cuda::CUDAGuard guard(input.device());
    const cudaStream_t stream = current_stream(input);
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
    torch::Tensor input,
    torch::Tensor output,
    torch::Tensor scales) {
    for (const auto& item : {
             std::pair<const torch::Tensor*, const char*>{&input, "input"},
             {&output, "output"},
             {&scales, "scales"},
         }) {
        require_cuda_contiguous(*item.first, item.second);
        require_same_device(input, *item.first, item.second);
    }
    require_bf16(input, "input");
    require_bf16(scales, "scales");
    TORCH_CHECK(input.dim() == 2, "input must be [M, K]");
    TORCH_CHECK(output.dim() == 2 && output.scalar_type() == torch::kInt8, "output must be 2D int8");
    const int actual_m = static_cast<int>(input.size(0));
    const int actual_k = static_cast<int>(input.size(1));
    const int padded_m = static_cast<int>(output.size(0));
    const int padded_k = static_cast<int>(output.size(1));
    TORCH_CHECK(
        padded_m % 256 == 0 && padded_m >= actual_m && padded_m - actual_m < 256,
        "invalid M padding");
    TORCH_CHECK(
        padded_k % 128 == 0 && padded_k >= actual_k && padded_k - actual_k < 128,
        "invalid K padding");
    TORCH_CHECK(scales.numel() == padded_m, "activation scales must contain M_pad values");
    c10::cuda::CUDAGuard guard(input.device());
    if (actual_m == padded_m && actual_k == padded_k) {
        check_status(
            xqt_svdq_w8a8_quantize_act_fast(
                input.data_ptr(),
                output.data_ptr(),
                scales.data_ptr(),
                padded_m,
                padded_k,
                current_stream(input)),
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
            current_stream(input)),
        "W8A8 activation quantization");
}

void gemm_lora(
    torch::Tensor activation,
    torch::Tensor weight,
    torch::Tensor output,
    torch::Tensor activation_scales,
    torch::Tensor weight_scales,
    torch::Tensor lora_activation,
    torch::Tensor lora_up,
    torch::Tensor bias,
    double lora_scale) {
    for (const auto& item : {
             std::pair<const torch::Tensor*, const char*>{&activation, "activation"},
             {&weight, "weight"},
             {&output, "output"},
             {&activation_scales, "activation_scales"},
             {&weight_scales, "weight_scales"},
             {&lora_activation, "lora_activation"},
             {&lora_up, "lora_up"},
             {&bias, "bias"},
         }) {
        require_cuda_contiguous(*item.first, item.second);
        require_same_device(output, *item.first, item.second);
    }
    TORCH_CHECK(activation.dim() == 2 && activation.scalar_type() == torch::kInt8, "activation must be 2D int8");
    TORCH_CHECK(weight.dim() == 2 && weight.scalar_type() == torch::kInt8, "weight must be 2D int8");
    require_bf16(output, "output");
    require_bf16(activation_scales, "activation_scales");
    require_bf16(weight_scales, "weight_scales");
    require_bf16(lora_up, "lora_up");
    require_bf16(bias, "bias");
    TORCH_CHECK(lora_activation.scalar_type() == torch::kFloat32, "lora_activation must be float32");
    const int padded_m = static_cast<int>(activation.size(0));
    const int padded_n = static_cast<int>(weight.size(0));
    const int padded_k = static_cast<int>(activation.size(1));
    TORCH_CHECK(weight.size(1) == padded_k, "activation and weight K mismatch");
    TORCH_CHECK(padded_m % 256 == 0 && padded_n % 128 == 0 && padded_k % 128 == 0, "invalid padded GEMM shape");
    TORCH_CHECK(output.size(0) <= padded_m && padded_m - output.size(0) < 256, "invalid output M");
    TORCH_CHECK(output.size(1) <= padded_n && padded_n - output.size(1) < 128, "invalid output N");
    TORCH_CHECK(output.size(1) % 4 == 0, "output N must be a multiple of 4 for vectorized epilogue stores");
    TORCH_CHECK(activation_scales.numel() == padded_m, "activation scale storage mismatch");
    TORCH_CHECK(weight_scales.numel() == padded_n, "weight scale storage mismatch");
    TORCH_CHECK(bias.numel() == padded_n, "bias storage mismatch");
    const int rank = static_cast<int>(lora_up.size(1));
    TORCH_CHECK(rank > 0 && rank % 16 == 0 && rank <= 1024, "rank must align to 16 and be <= 1024");
    TORCH_CHECK(lora_up.sizes() == torch::IntArrayRef({padded_n, rank}), "lora_up must be [N_pad, rank]");
    TORCH_CHECK(
        lora_activation.sizes() == torch::IntArrayRef({padded_m, rank}),
        "lora_activation must be [M_pad, rank]");
    c10::cuda::CUDAGuard guard(output.device());
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
            current_stream(output)),
        "W8A8 GEMM plus LoRA up");
}

void gemm(
    torch::Tensor activation,
    torch::Tensor weight,
    torch::Tensor output,
    torch::Tensor activation_scales,
    torch::Tensor weight_scales,
    torch::Tensor bias) {
    for (const auto& item : {
             std::pair<const torch::Tensor*, const char*>{&activation, "activation"},
             {&weight, "weight"},
             {&output, "output"},
             {&activation_scales, "activation_scales"},
             {&weight_scales, "weight_scales"},
             {&bias, "bias"},
         }) {
        require_cuda_contiguous(*item.first, item.second);
        require_same_device(output, *item.first, item.second);
    }
    TORCH_CHECK(activation.dim() == 2 && activation.scalar_type() == torch::kInt8, "activation must be 2D int8");
    TORCH_CHECK(weight.dim() == 2 && weight.scalar_type() == torch::kInt8, "weight must be 2D int8");
    require_bf16(output, "output");
    require_bf16(activation_scales, "activation_scales");
    require_bf16(weight_scales, "weight_scales");
    require_bf16(bias, "bias");
    const int padded_m = static_cast<int>(activation.size(0));
    const int padded_n = static_cast<int>(weight.size(0));
    const int padded_k = static_cast<int>(activation.size(1));
    TORCH_CHECK(weight.size(1) == padded_k, "activation and weight K mismatch");
    TORCH_CHECK(padded_m % 256 == 0 && padded_n % 128 == 0 && padded_k % 128 == 0, "invalid padded GEMM shape");
    TORCH_CHECK(output.size(0) <= padded_m && padded_m - output.size(0) < 256, "invalid output M");
    TORCH_CHECK(output.size(1) <= padded_n && padded_n - output.size(1) < 128, "invalid output N");
    TORCH_CHECK(output.size(1) % 4 == 0, "output N must be a multiple of 4 for vectorized epilogue stores");
    TORCH_CHECK(activation_scales.numel() == padded_m, "activation scale storage mismatch");
    TORCH_CHECK(weight_scales.numel() == padded_n, "weight scale storage mismatch");
    TORCH_CHECK(bias.numel() == padded_n, "bias storage mismatch");
    c10::cuda::CUDAGuard guard(output.device());
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
            current_stream(output)),
        "W8A8 GEMM");
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

torch::Tensor svdq_linear(
    torch::Tensor input,
    torch::Tensor activation,
    torch::Tensor activation_scales,
    torch::Tensor lora_down,
    torch::Tensor lora_activation,
    torch::Tensor weight,
    torch::Tensor weight_scales,
    torch::Tensor lora_up,
    torch::Tensor bias,
    int64_t output_features,
    double lora_scale) {
    auto output = allocate_output(input, weight, output_features);
    quantize_act_lora(
        input,
        activation,
        activation_scales,
        lora_down,
        lora_activation);
    gemm_lora(
        activation,
        weight,
        output,
        activation_scales,
        weight_scales,
        lora_activation,
        lora_up,
        bias,
        lora_scale);
    return output;
}

torch::Tensor linear(
    torch::Tensor input,
    torch::Tensor activation,
    torch::Tensor activation_scales,
    torch::Tensor weight,
    torch::Tensor weight_scales,
    torch::Tensor bias,
    int64_t output_features) {
    auto output = allocate_output(input, weight, output_features);
    quantize_act(input, activation, activation_scales);
    gemm(
        activation,
        weight,
        output,
        activation_scales,
        weight_scales,
        bias);
    return output;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("quantize_weight", &quantize_weight);
    module.def("quantize_act", &quantize_act);
    module.def("quantize_act_lora", &quantize_act_lora);
    module.def("gemm", &gemm);
    module.def(
        "gemm_lora",
        &gemm_lora,
        pybind11::arg("activation"),
        pybind11::arg("weight"),
        pybind11::arg("output"),
        pybind11::arg("activation_scales"),
        pybind11::arg("weight_scales"),
        pybind11::arg("lora_activation"),
        pybind11::arg("lora_up"),
        pybind11::arg("bias"),
        pybind11::arg("lora_scale") = 1.0);
    module.def(
        "svdq_linear",
        &svdq_linear,
        pybind11::arg("input"),
        pybind11::arg("activation"),
        pybind11::arg("activation_scales"),
        pybind11::arg("lora_down"),
        pybind11::arg("lora_activation"),
        pybind11::arg("weight"),
        pybind11::arg("weight_scales"),
        pybind11::arg("lora_up"),
        pybind11::arg("bias"),
        pybind11::arg("output_features"),
        pybind11::arg("lora_scale") = 1.0);
    module.def(
        "linear",
        &linear,
        pybind11::arg("input"),
        pybind11::arg("activation"),
        pybind11::arg("activation_scales"),
        pybind11::arg("weight"),
        pybind11::arg("weight_scales"),
        pybind11::arg("bias"),
        pybind11::arg("output_features"));
    module.def("version", []() { return std::string(xqt_svdq_w8a8_version()); });
}
