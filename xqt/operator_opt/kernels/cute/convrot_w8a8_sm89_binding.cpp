#include <torch/extension.h>

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime_api.h>

#include <cstdint>
#include <string>
#include <utility>

extern "C" int xqt_convrot_w8a8_quantize_weight(
    const void*, const void*, void*, int, int, cudaStream_t);
extern "C" int xqt_convrot_w8a8_quantize_rotated_act(
    const void*, void*, void*, int, int, int, int, int, int, cudaStream_t);
extern "C" int xqt_convrot_w8a8_gemm(
    const void*, const void*, void*, const void*, const void*, const void*, int,
    int, int, int, int, cudaStream_t);
extern "C" const char* xqt_convrot_w8a8_version();

namespace {

#if defined(XQT_W8A8_FP16)
constexpr torch::ScalarType kComputeType = torch::kFloat16;
constexpr const char* kComputeName = "float16";
#else
constexpr torch::ScalarType kComputeType = torch::kBFloat16;
constexpr const char* kComputeName = "bfloat16";
#endif

void require_cuda_contiguous(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be CUDA");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void require_compute_type(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(
        tensor.scalar_type() == kComputeType,
        name,
        " must be ",
        kComputeName);
}

void require_same_device(
    const torch::Tensor& reference,
    const torch::Tensor& tensor,
    const char* name) {
    TORCH_CHECK(
        tensor.device() == reference.device(),
        name,
        " must share the CUDA device");
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
    require_compute_type(input, "input");
    require_compute_type(scales, "scales");
    require_same_device(input, scales, "scales");
    require_same_device(input, output, "output");
    TORCH_CHECK(input.dim() == 2, "input must be [N, K]");
    TORCH_CHECK(
        scales.dim() == 1 && scales.size(0) == input.size(0),
        "scales must be [N]");
    TORCH_CHECK(output.scalar_type() == torch::kInt8, "output must be int8");
    TORCH_CHECK(output.sizes() == input.sizes(), "output must match input shape");
    const int n = static_cast<int>(input.size(0));
    const int k = static_cast<int>(input.size(1));
    TORCH_CHECK(n % 128 == 0 && k % 32 == 0, "weight shape is misaligned");
    c10::cuda::CUDAGuard guard(input.device());
    check_status(
        xqt_convrot_w8a8_quantize_weight(
            input.data_ptr(),
            scales.data_ptr(),
            output.data_ptr(),
            n,
            k,
            current_stream(input)),
        "ConvRot W8A8 weight packing");
}

void quantize_rotated_act(
    torch::Tensor input,
    torch::Tensor output,
    torch::Tensor scales,
    int64_t rotated_k,
    int64_t rot_size) {
    for (const auto& item : {
             std::pair<const torch::Tensor*, const char*>{&input, "input"},
             {&output, "output"},
             {&scales, "scales"},
         }) {
        require_cuda_contiguous(*item.first, item.second);
        require_same_device(input, *item.first, item.second);
    }
    require_compute_type(input, "input");
    require_compute_type(scales, "scales");
    TORCH_CHECK(input.dim() == 2, "input must be [M, logical_K]");
    TORCH_CHECK(
        output.dim() == 2 && output.scalar_type() == torch::kInt8,
        "output must be 2D int8");
    const int actual_m = static_cast<int>(input.size(0));
    const int logical_k = static_cast<int>(input.size(1));
    const int padded_m = static_cast<int>(output.size(0));
    const int padded_k = static_cast<int>(output.size(1));
    const int rotation_extent = static_cast<int>(rotated_k);
    const int rotation_size = static_cast<int>(rot_size);
    TORCH_CHECK(
        padded_m % 256 == 0 && padded_m >= actual_m && padded_m - actual_m < 256,
        "invalid M padding");
    TORCH_CHECK(padded_k % 256 == 0, "padded K must be a multiple of 256");
    TORCH_CHECK(
        rotation_extent >= logical_k && rotation_extent <= padded_k,
        "rotated_k must cover logical K and fit padded K");
    TORCH_CHECK(
        rotation_size == 1 || rotation_size == 256,
        "rot_size must be 1 or 256");
    TORCH_CHECK(
        rotation_size == 1 || rotation_extent % rotation_size == 0,
        "rotated_k must be divisible by rot_size");
    TORCH_CHECK(scales.numel() == padded_m, "activation scale size mismatch");
    c10::cuda::CUDAGuard guard(input.device());
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
            current_stream(input)),
        "ConvRot Hadamard rotation plus dynamic W8 activation packing");
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
    TORCH_CHECK(
        activation.dim() == 2 && activation.scalar_type() == torch::kInt8,
        "activation must be 2D int8");
    TORCH_CHECK(
        weight.dim() == 2 && weight.scalar_type() == torch::kInt8,
        "weight must be 2D int8");
    require_compute_type(output, "output");
    require_compute_type(activation_scales, "activation_scales");
    require_compute_type(weight_scales, "weight_scales");
    require_compute_type(bias, "bias");
    const int padded_m = static_cast<int>(activation.size(0));
    const int padded_n = static_cast<int>(weight.size(0));
    const int padded_k = static_cast<int>(activation.size(1));
    TORCH_CHECK(weight.size(1) == padded_k, "activation and weight K mismatch");
    TORCH_CHECK(
        padded_m % 256 == 0 && padded_n % 128 == 0 && padded_k % 256 == 0,
        "invalid padded GEMM shape");
    TORCH_CHECK(
        output.size(0) <= padded_m && padded_m - output.size(0) < 256,
        "invalid output M");
    TORCH_CHECK(
        output.size(1) <= padded_n && padded_n - output.size(1) < 128,
        "invalid output N");
    TORCH_CHECK(
        output.size(1) % 4 == 0,
        "output N must be a multiple of 4 for vectorized epilogue stores");
    TORCH_CHECK(
        activation_scales.numel() == padded_m,
        "activation scale storage mismatch");
    TORCH_CHECK(weight_scales.numel() == padded_n, "weight scale storage mismatch");
    TORCH_CHECK(bias.numel() == padded_n, "bias storage mismatch");
    c10::cuda::CUDAGuard guard(output.device());
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
            current_stream(output)),
        "ConvRot W8A8 GEMM plus bias");
}

torch::Tensor linear(
    torch::Tensor input,
    torch::Tensor activation,
    torch::Tensor activation_scales,
    torch::Tensor weight,
    torch::Tensor weight_scales,
    torch::Tensor bias,
    int64_t rotated_k,
    int64_t rot_size,
    int64_t output_features) {
    TORCH_CHECK(input.dim() == 2, "input must be [M, logical_K]");
    TORCH_CHECK(
        output_features > 0 && output_features <= weight.size(0),
        "output_features must fit the padded weight extent");
    TORCH_CHECK(
        output_features % 4 == 0,
        "output_features must be a multiple of 4");
    auto output = torch::empty(
        {input.size(0), output_features},
        input.options());
    quantize_rotated_act(
        input,
        activation,
        activation_scales,
        rotated_k,
        rot_size);
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
    module.def("quantize_rotated_act", &quantize_rotated_act);
    module.def("gemm", &gemm);
    module.def(
        "linear",
        &linear,
        pybind11::arg("input"),
        pybind11::arg("activation"),
        pybind11::arg("activation_scales"),
        pybind11::arg("weight"),
        pybind11::arg("weight_scales"),
        pybind11::arg("bias"),
        pybind11::arg("rotated_k"),
        pybind11::arg("rot_size"),
        pybind11::arg("output_features"));
    module.def("version", []() { return std::string(xqt_convrot_w8a8_version()); });
}
