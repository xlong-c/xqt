#include <torch/extension.h>

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime_api.h>

#include <cstdint>
#include <string>

namespace py = pybind11;

extern "C" int xqt_awq_w4a16_sm89_pack(
    const void*, void*, int, int, cudaStream_t);
extern "C" int xqt_awq_w4a16_sm89_decode(
    const void*, const void*, const void*, const void*, void*, int, int, int,
    int, cudaStream_t);
extern "C" int xqt_awq_w4a16_sm89_decode_bias(
    const void*, const void*, const void*, const void*, const void*, void*, int,
    int, int, int, cudaStream_t);
extern "C" int xqt_awq_w4a8_sm89_decode(
    const void*, const void*, const void*, const float*, float, void*, int, int, int, cudaStream_t);
extern "C" int xqt_awq_w4a8_sm89_decode_bias(
    const void*, const void*, const void*, const float*, float, const void*, void*, int, int, int, cudaStream_t);
extern "C" const char* xqt_awq_w4a16_sm89_version();

namespace {

void require_cuda_contiguous(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be CUDA");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void require_same_device(
    const torch::Tensor& reference,
    const torch::Tensor& tensor,
    const char* name) {
    TORCH_CHECK(
        tensor.device() == reference.device(),
        name,
        " must share the input CUDA device");
}

void require_decode_contract(
    const torch::Tensor& input,
    const torch::Tensor& qweight,
    const torch::Tensor& scales,
    const torch::Tensor& scaled_zeros,
    const torch::Tensor& output) {
    require_cuda_contiguous(input, "input");
    require_cuda_contiguous(qweight, "qweight");
    require_cuda_contiguous(scales, "scales");
    require_cuda_contiguous(scaled_zeros, "scaled_zeros");
    require_cuda_contiguous(output, "output");
    require_same_device(input, qweight, "qweight");
    require_same_device(input, scales, "scales");
    require_same_device(input, scaled_zeros, "scaled_zeros");
    require_same_device(input, output, "output");

    TORCH_CHECK(input.dim() == 2, "input must have shape [M,K]");
    TORCH_CHECK(
        input.scalar_type() == torch::kFloat16 ||
            input.scalar_type() == torch::kBFloat16,
        "input must be float16 or bfloat16");
    TORCH_CHECK(qweight.scalar_type() == torch::kInt32, "qweight must be int32");
    TORCH_CHECK(qweight.dim() == 2, "qweight must have shape [N/4,K/2]");
    TORCH_CHECK(scales.scalar_type() == input.scalar_type(), "scales dtype mismatch");
    TORCH_CHECK(
        scaled_zeros.scalar_type() == input.scalar_type(),
        "scaled_zeros dtype mismatch");
    TORCH_CHECK(output.scalar_type() == input.scalar_type(), "output dtype mismatch");

    const int64_t m = input.size(0);
    const int64_t k = input.size(1);
    const int64_t n = qweight.size(0) * 4;
    TORCH_CHECK(m >= 1 && m <= 8, "AWQ decode requires 1 <= M <= 8");
    TORCH_CHECK(n > 0 && n % 8 == 0, "AWQ decode requires N % 8 == 0");
    TORCH_CHECK(k > 0 && k % 64 == 0, "AWQ decode requires K % 64 == 0");
    TORCH_CHECK(qweight.size(1) == k / 2, "qweight K storage mismatch");
    TORCH_CHECK(scales.dim() == 2, "scales must have shape [K/64,N]");
    TORCH_CHECK(
        scales.sizes() == torch::IntArrayRef({k / 64, n}),
        "scales shape mismatch");
    TORCH_CHECK(
        scaled_zeros.sizes() == scales.sizes(),
        "scaled_zeros shape mismatch");
    TORCH_CHECK(
        output.sizes() == torch::IntArrayRef({m, n}),
        "output shape mismatch");
}

void check_status(int status, const char* operation) {
    TORCH_CHECK(
        status == static_cast<int>(cudaSuccess),
        operation,
        " failed: ",
        cudaGetErrorString(static_cast<cudaError_t>(status)));
}

cudaStream_t current_stream(const torch::Tensor& tensor) {
    return c10::cuda::getCurrentCUDAStream(tensor.get_device()).stream();
}

torch::Tensor pack(torch::Tensor canonical_qweight) {
    require_cuda_contiguous(canonical_qweight, "canonical_qweight");
    TORCH_CHECK(
        canonical_qweight.scalar_type() == torch::kUInt8,
        "canonical_qweight must be uint8");
    TORCH_CHECK(
        canonical_qweight.dim() == 2,
        "canonical_qweight must have shape [N,K/2]");
    const int64_t n = canonical_qweight.size(0);
    const int64_t k = canonical_qweight.size(1) * 2;
    TORCH_CHECK(n > 0 && n % 8 == 0, "AWQ prepack requires N % 8 == 0");
    TORCH_CHECK(k > 0 && k % 64 == 0, "AWQ prepack requires K % 64 == 0");

    c10::cuda::CUDAGuard guard(canonical_qweight.device());
    torch::Tensor output = torch::empty(
        {n / 4, k / 2},
        canonical_qweight.options().dtype(torch::kInt32));
    check_status(
        xqt_awq_w4a16_sm89_pack(
            canonical_qweight.data_ptr(),
            output.data_ptr(),
            static_cast<int>(n),
            static_cast<int>(k),
            current_stream(canonical_qweight)),
        "AWQ W4A16 prepack");
    return output;
}

void decode_out(
    torch::Tensor input,
    torch::Tensor qweight,
    torch::Tensor scales,
    torch::Tensor scaled_zeros,
    torch::Tensor output) {
    require_decode_contract(input, qweight, scales, scaled_zeros, output);
    c10::cuda::CUDAGuard guard(input.device());
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
            input.scalar_type() == torch::kFloat16 ? 0 : 1,
            current_stream(input)),
        "AWQ W4A16 decode");
}

void require_bias(const torch::Tensor& input, const torch::Tensor& bias, int64_t n) {
    require_cuda_contiguous(bias, "bias");
    require_same_device(input, bias, "bias");
    TORCH_CHECK(bias.scalar_type() == input.scalar_type(), "bias dtype mismatch");
    TORCH_CHECK(bias.dim() == 1 && bias.numel() == n, "bias must have shape [N]");
}

void decode_bias_out(
    torch::Tensor input,
    torch::Tensor qweight,
    torch::Tensor scales,
    torch::Tensor scaled_zeros,
    torch::Tensor bias,
    torch::Tensor output) {
    require_decode_contract(input, qweight, scales, scaled_zeros, output);
    require_bias(input, bias, qweight.size(0) * 4);
    c10::cuda::CUDAGuard guard(input.device());
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
            input.scalar_type() == torch::kFloat16 ? 0 : 1,
            current_stream(input)),
        "AWQ W4A16 decode bias");
}

torch::Tensor decode_bias(
    torch::Tensor input,
    torch::Tensor qweight,
    torch::Tensor scales,
    torch::Tensor scaled_zeros,
    torch::Tensor bias) {
    TORCH_CHECK(qweight.dim() == 2, "qweight must have shape [N/4,K/2]");
    torch::Tensor output = torch::empty(
        {input.size(0), qweight.size(0) * 4},
        input.options());
    decode_bias_out(input, qweight, scales, scaled_zeros, bias, output);
    return output;
}

torch::Tensor decode(
    torch::Tensor input,
    torch::Tensor qweight,
    torch::Tensor scales,
    torch::Tensor scaled_zeros) {
    TORCH_CHECK(qweight.dim() == 2, "qweight must have shape [N/4,K/2]");
    torch::Tensor output = torch::empty(
        {input.size(0), qweight.size(0) * 4},
        input.options());
    decode_out(input, qweight, scales, scaled_zeros, output);
    return output;
}

void decode_w4a8_out(
    const torch::Tensor& inputs,
    const torch::Tensor& qweight,
    const torch::Tensor& scales,
    const torch::Tensor& scale_a,
    torch::Tensor& output) {
    require_cuda_contiguous(inputs, "inputs");
    require_cuda_contiguous(qweight, "qweight");
    require_cuda_contiguous(scales, "scales");
    require_cuda_contiguous(scale_a, "scale_a");
    require_cuda_contiguous(output, "output");
    require_same_device(inputs, qweight, "qweight");
    require_same_device(inputs, scales, "scales");
    require_same_device(inputs, scale_a, "scale_a");
    require_same_device(inputs, output, "output");

    TORCH_CHECK(inputs.dim() == 2, "inputs must have shape [M,K]");
    TORCH_CHECK(inputs.scalar_type() == torch::kInt8, "inputs must be int8");
    TORCH_CHECK(qweight.scalar_type() == torch::kInt32, "qweight must be int32");
    TORCH_CHECK(qweight.dim() == 2, "qweight must have shape [N/4,K/2]");
    TORCH_CHECK(scale_a.scalar_type() == torch::kFloat32, "scale_a must be float32");
    TORCH_CHECK(
        output.scalar_type() == torch::kFloat16 ||
        output.scalar_type() == torch::kBFloat16,
        "output must be float16 or bfloat16");

    const int64_t n = qweight.size(0) * 4;
    const int64_t k = inputs.size(1);
    c10::cuda::CUDAGuard guard(inputs.device());
    check_status(
        xqt_awq_w4a8_sm89_decode(
            inputs.data_ptr(),
            qweight.data_ptr(),
            scales.data_ptr(),
            static_cast<const float*>(scale_a.data_ptr()),
            1.0f,
            output.data_ptr(),
            static_cast<int>(n),
            static_cast<int>(k),
            output.scalar_type() == torch::kFloat16 ? 0 : 1,
            current_stream(inputs)),
        "AWQ W4A8 decode");
}

void decode_w4a8_scalar_out(
    const torch::Tensor& inputs,
    const torch::Tensor& qweight,
    const torch::Tensor& scales,
    double scale_a,
    torch::Tensor& output) {
    require_cuda_contiguous(inputs, "inputs");
    require_cuda_contiguous(qweight, "qweight");
    require_cuda_contiguous(scales, "scales");
    require_cuda_contiguous(output, "output");
    require_same_device(inputs, qweight, "qweight");
    require_same_device(inputs, scales, "scales");
    require_same_device(inputs, output, "output");

    TORCH_CHECK(inputs.dim() == 2, "inputs must have shape [M,K]");
    TORCH_CHECK(inputs.scalar_type() == torch::kInt8, "inputs must be int8");
    TORCH_CHECK(qweight.scalar_type() == torch::kInt32, "qweight must be int32");
    TORCH_CHECK(qweight.dim() == 2, "qweight must have shape [N/4,K/2]");
    TORCH_CHECK(
        output.scalar_type() == torch::kFloat16 ||
        output.scalar_type() == torch::kBFloat16,
        "output must be float16 or bfloat16");

    const int64_t n = qweight.size(0) * 4;
    const int64_t k = inputs.size(1);
    c10::cuda::CUDAGuard guard(inputs.device());
    check_status(
        xqt_awq_w4a8_sm89_decode(
            inputs.data_ptr(),
            qweight.data_ptr(),
            scales.data_ptr(),
            nullptr,
            static_cast<float>(scale_a),
            output.data_ptr(),
            static_cast<int>(n),
            static_cast<int>(k),
            output.scalar_type() == torch::kFloat16 ? 0 : 1,
            current_stream(inputs)),
        "AWQ W4A8 decode");
}

void decode_w4a8_bias_out(
    const torch::Tensor& inputs,
    const torch::Tensor& qweight,
    const torch::Tensor& scales,
    const torch::Tensor& scale_a,
    const torch::Tensor& residual,
    torch::Tensor& output) {
    require_cuda_contiguous(inputs, "inputs");
    require_cuda_contiguous(qweight, "qweight");
    require_cuda_contiguous(scales, "scales");
    require_cuda_contiguous(scale_a, "scale_a");
    require_cuda_contiguous(residual, "residual");
    require_cuda_contiguous(output, "output");
    require_same_device(inputs, qweight, "qweight");
    require_same_device(inputs, scales, "scales");
    require_same_device(inputs, scale_a, "scale_a");
    require_same_device(inputs, residual, "residual");
    require_same_device(inputs, output, "output");

    TORCH_CHECK(inputs.dim() == 2, "inputs must have shape [M,K]");
    TORCH_CHECK(inputs.scalar_type() == torch::kInt8, "inputs must be int8");
    TORCH_CHECK(qweight.scalar_type() == torch::kInt32, "qweight must be int32");
    TORCH_CHECK(qweight.dim() == 2, "qweight must have shape [N/4,K/2]");
    TORCH_CHECK(scale_a.scalar_type() == torch::kFloat32, "scale_a must be float32");
    TORCH_CHECK(
        output.scalar_type() == torch::kFloat16 ||
        output.scalar_type() == torch::kBFloat16,
        "output must be float16 or bfloat16");

    const int64_t n = qweight.size(0) * 4;
    const int64_t k = inputs.size(1);
    c10::cuda::CUDAGuard guard(inputs.device());
    check_status(
        xqt_awq_w4a8_sm89_decode_bias(
            inputs.data_ptr(),
            qweight.data_ptr(),
            scales.data_ptr(),
            static_cast<const float*>(scale_a.data_ptr()),
            1.0f,
            residual.data_ptr(),
            output.data_ptr(),
            static_cast<int>(n),
            static_cast<int>(k),
            output.scalar_type() == torch::kFloat16 ? 0 : 1,
            current_stream(inputs)),
        "AWQ W4A8 decode bias");
}

void decode_w4a8_scalar_bias_out(
    const torch::Tensor& inputs,
    const torch::Tensor& qweight,
    const torch::Tensor& scales,
    double scale_a,
    const torch::Tensor& residual,
    torch::Tensor& output) {
    require_cuda_contiguous(inputs, "inputs");
    require_cuda_contiguous(qweight, "qweight");
    require_cuda_contiguous(scales, "scales");
    require_cuda_contiguous(residual, "residual");
    require_cuda_contiguous(output, "output");
    require_same_device(inputs, qweight, "qweight");
    require_same_device(inputs, scales, "scales");
    require_same_device(inputs, residual, "residual");
    require_same_device(inputs, output, "output");

    TORCH_CHECK(inputs.dim() == 2, "inputs must have shape [M,K]");
    TORCH_CHECK(inputs.scalar_type() == torch::kInt8, "inputs must be int8");
    TORCH_CHECK(qweight.scalar_type() == torch::kInt32, "qweight must be int32");
    TORCH_CHECK(qweight.dim() == 2, "qweight must have shape [N/4,K/2]");
    TORCH_CHECK(
        output.scalar_type() == torch::kFloat16 ||
        output.scalar_type() == torch::kBFloat16,
        "output must be float16 or bfloat16");

    const int64_t n = qweight.size(0) * 4;
    const int64_t k = inputs.size(1);
    c10::cuda::CUDAGuard guard(inputs.device());
    check_status(
        xqt_awq_w4a8_sm89_decode_bias(
            inputs.data_ptr(),
            qweight.data_ptr(),
            scales.data_ptr(),
            nullptr,
            static_cast<float>(scale_a),
            residual.data_ptr(),
            output.data_ptr(),
            static_cast<int>(n),
            static_cast<int>(k),
            output.scalar_type() == torch::kFloat16 ? 0 : 1,
            current_stream(inputs)),
        "AWQ W4A8 decode bias");
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("pack", &pack, "Pack canonical AWQ qweight for SM89 decode");
    module.def("decode", &decode, "Run SM89 AWQ W4A16 decode");
    module.def("decode_out", &decode_out, "Run SM89 AWQ W4A16 decode into output");
    module.def("decode_bias", &decode_bias, "Run SM89 AWQ W4A16 decode with bias");
    module.def(
        "decode_bias_out",
        &decode_bias_out,
        "Run SM89 AWQ W4A16 decode with bias into output");
    module.def("decode_w4a8_out", &decode_w4a8_out, "Run SM89 AWQ W4A8 decode into output");
    module.def("decode_w4a8_scalar_out", &decode_w4a8_scalar_out, "Run SM89 AWQ W4A8 decode into output with scalar scale_a");
    module.def(
        "decode_w4a8_bias_out",
        &decode_w4a8_bias_out,
        "Run SM89 AWQ W4A8 decode with residual bias into output");
    module.def(
        "decode_w4a8_scalar_bias_out",
        &decode_w4a8_scalar_bias_out,
        "Run SM89 AWQ W4A8 decode with residual bias into output with scalar scale_a");
    module.def(
        "version",
        []() { return std::string(xqt_awq_w4a16_sm89_version()); });
}

