#include <torch/extension.h>

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime_api.h>

#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace py = pybind11;

extern "C" int xqt_convrot_w4a4_rowwise_quantize(
    const void*, void*, void*, int, int, int, cudaStream_t);
extern "C" int xqt_convrot_w4a4_rowwise_gemm(
    const void*, const void*, const void*, const void*, const void*, void*, int, int, int, int, cudaStream_t);
extern "C" const char* xqt_convrot_w4a4_rowwise_version();

namespace {

int scalar_kind(const torch::Tensor& tensor) {
    if (tensor.scalar_type() == torch::kFloat16) {
        return 0;
    }
    if (tensor.scalar_type() == torch::kBFloat16) {
        return 1;
    }
    TORCH_CHECK(false, "rowwise ConvRot W4A4 supports only float16 and bfloat16");
    return -1;
}

void require_cuda_contiguous(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be CUDA");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void require_same_device(
    const torch::Tensor& reference,
    const torch::Tensor& tensor,
    const char* name) {
    TORCH_CHECK(tensor.device() == reference.device(), name, " must be on the same CUDA device");
}

cudaStream_t current_stream(const torch::Tensor& tensor) {
    return c10::cuda::getCurrentCUDAStream(tensor.get_device()).stream();
}

int64_t tensor_version(const torch::Tensor& tensor) {
    return tensor.defined() ? tensor._version() : -1;
}

torch::Tensor tensor_or_undefined(const py::object& value) {
    if (value.is_none()) {
        return torch::Tensor();
    }
    return value.cast<torch::Tensor>();
}

void check_status(int status, const char* operation) {
    TORCH_CHECK(
        status == static_cast<int>(cudaSuccess),
        operation,
        " failed: ",
        cudaGetErrorString(static_cast<cudaError_t>(status)));
}

void validate_static_state(
    const torch::Tensor& act,
    const torch::Tensor& activation_scales,
    const torch::Tensor& weight,
    const torch::Tensor& weight_scales,
    const torch::Tensor& bias,
    int rows,
    int input_features,
    int output_features) {
    for (const auto& item : {
             std::pair<const torch::Tensor*, const char*>{&act, "act"},
             {&activation_scales, "activation_scales"},
             {&weight, "weight"},
             {&weight_scales, "weight_scales"},
             {&bias, "bias"},
         }) {
        require_cuda_contiguous(*item.first, item.second);
        require_same_device(weight, *item.first, item.second);
    }
    TORCH_CHECK(rows > 0, "rows must be positive");
    TORCH_CHECK(
        input_features >= 1024 && input_features <= 32768 &&
            (input_features == 1024 || input_features % 2048 == 0),
        "input_features must be 1024 or a multiple of 2048 through 32768");
    TORCH_CHECK(output_features > 0 && output_features % 8 == 0, "invalid output_features");
    TORCH_CHECK(
        act.dim() == 2 && act.scalar_type() == torch::kInt8 &&
            act.sizes() == torch::IntArrayRef({rows, input_features / 2}),
        "act must be int8 [M, K / 2]");
    TORCH_CHECK(
        activation_scales.scalar_type() == torch::kFloat32 && activation_scales.numel() == rows,
        "activation_scales must be float32 [M]");
    TORCH_CHECK(
        weight.dim() == 2 && weight.scalar_type() == torch::kInt8 &&
            weight.sizes() == torch::IntArrayRef({output_features, input_features / 2}),
        "weight must be int8 [N, K / 2]");
    TORCH_CHECK(
        weight_scales.scalar_type() == torch::kFloat32 && weight_scales.numel() == output_features,
        "weight_scales must be float32 [N]");
    TORCH_CHECK(
        bias.scalar_type() == torch::kFloat32 && bias.numel() == output_features,
        "bias must be float32 [N]");
}

torch::Tensor run_linear(
    torch::Tensor input,
    torch::Tensor act,
    torch::Tensor activation_scales,
    torch::Tensor weight,
    torch::Tensor weight_scales,
    torch::Tensor bias,
    int output_features) {
    auto contiguous_input = input.contiguous();
    require_cuda_contiguous(contiguous_input, "input");
    TORCH_CHECK(contiguous_input.dim() == 2, "input must be [M, K]");
    const int rows = static_cast<int>(contiguous_input.size(0));
    const int input_features = static_cast<int>(contiguous_input.size(1));
    validate_static_state(
        act,
        activation_scales,
        weight,
        weight_scales,
        bias,
        rows,
        input_features,
        output_features);
    require_same_device(weight, contiguous_input, "input");
    c10::cuda::CUDAGuard guard(contiguous_input.device());
    auto output = torch::empty({rows, output_features}, contiguous_input.options());
    const int kind = scalar_kind(contiguous_input);
    const cudaStream_t stream = current_stream(contiguous_input);
    check_status(
        xqt_convrot_w4a4_rowwise_quantize(
            contiguous_input.data_ptr(),
            act.data_ptr(),
            activation_scales.data_ptr(),
            rows,
            input_features,
            kind,
            stream),
        "rowwise ConvRot rotation plus INT4 quantization");
    check_status(
        xqt_convrot_w4a4_rowwise_gemm(
            act.data_ptr(),
            weight.data_ptr(),
            activation_scales.data_ptr(),
            weight_scales.data_ptr(),
            bias.data_ptr(),
            output.data_ptr(),
            rows,
            output_features,
            input_features,
            kind,
            stream),
        "rowwise ConvRot CUTLASS W4A4 GEMM");
    return output;
}

class BoundConvRotW4A4RowwiseLinear {
public:
    BoundConvRotW4A4RowwiseLinear(
        torch::Tensor act,
        torch::Tensor activation_scales,
        torch::Tensor weight,
        torch::Tensor weight_scales,
        torch::Tensor bias,
        int64_t rows,
        int64_t input_features,
        int64_t output_features)
        : act_(std::move(act)),
          activation_scales_(std::move(activation_scales)),
          weight_(std::move(weight)),
          weight_scales_(std::move(weight_scales)),
          bias_(std::move(bias)),
          rows_(static_cast<int>(rows)),
          input_features_(static_cast<int>(input_features)),
          output_features_(static_cast<int>(output_features)) {
        validate_static_state(
            act_,
            activation_scales_,
            weight_,
            weight_scales_,
            bias_,
            rows_,
            input_features_,
            output_features_);
    }

    torch::Tensor run(torch::Tensor input) const {
        auto contiguous_input = input.contiguous();
        require_cuda_contiguous(contiguous_input, "input");
        require_same_device(weight_, contiguous_input, "input");
        TORCH_CHECK(
            contiguous_input.dim() == 2 && contiguous_input.size(0) == rows_ &&
                contiguous_input.size(1) == input_features_,
            "input shape does not match the bound rowwise ConvRot runner");
        c10::cuda::CUDAGuard guard(contiguous_input.device());
        auto output = torch::empty({rows_, output_features_}, contiguous_input.options());
        const int kind = scalar_kind(contiguous_input);
        const cudaStream_t stream = current_stream(contiguous_input);
        check_status(
            xqt_convrot_w4a4_rowwise_quantize(
                contiguous_input.data_ptr(),
                act_.data_ptr(),
                activation_scales_.data_ptr(),
                rows_,
                input_features_,
                kind,
                stream),
            "rowwise ConvRot rotation plus INT4 quantization");
        check_status(
            xqt_convrot_w4a4_rowwise_gemm(
                act_.data_ptr(),
                weight_.data_ptr(),
                activation_scales_.data_ptr(),
                weight_scales_.data_ptr(),
                bias_.data_ptr(),
                output.data_ptr(),
                rows_,
                output_features_,
                input_features_,
                kind,
                stream),
            "rowwise ConvRot CUTLASS W4A4 GEMM");
        return output;
    }

private:
    torch::Tensor act_;
    torch::Tensor activation_scales_;
    torch::Tensor weight_;
    torch::Tensor weight_scales_;
    torch::Tensor bias_;
    int rows_;
    int input_features_;
    int output_features_;
};

class DynamicConvRotW4A4RowwiseLinear {
public:
    DynamicConvRotW4A4RowwiseLinear(
        torch::Tensor weight,
        torch::Tensor weight_scales,
        torch::Tensor bias,
        torch::Tensor source_weight,
        torch::Tensor source_weight_scales,
        torch::Tensor source_bias,
        int64_t expected_scalar_kind,
        int64_t input_features,
        int64_t output_features)
        : weight_(std::move(weight)),
          weight_scales_(std::move(weight_scales)),
          bias_(std::move(bias)),
          source_weight_(std::move(source_weight)),
          source_weight_scales_(std::move(source_weight_scales)),
          source_bias_(std::move(source_bias)),
          source_weight_version_(tensor_version(source_weight_)),
          source_weight_scales_version_(tensor_version(source_weight_scales_)),
          source_bias_version_(tensor_version(source_bias_)),
          expected_scalar_kind_(static_cast<int>(expected_scalar_kind)),
          input_features_(static_cast<int>(input_features)),
          output_features_(static_cast<int>(output_features)) {
        require_cuda_contiguous(weight_, "weight");
        require_cuda_contiguous(weight_scales_, "weight_scales");
        require_cuda_contiguous(bias_, "bias");
        require_same_device(weight_, weight_scales_, "weight_scales");
        require_same_device(weight_, bias_, "bias");
        TORCH_CHECK(
            input_features_ >= 1024 && input_features_ <= 32768 &&
                (input_features_ == 1024 || input_features_ % 2048 == 0),
            "input_features must be 1024 or a multiple of 2048 through 32768");
        TORCH_CHECK(output_features_ > 0 && output_features_ % 8 == 0, "invalid output_features");
        TORCH_CHECK(
            weight_.dim() == 2 && weight_.scalar_type() == torch::kInt8 &&
                weight_.sizes() == torch::IntArrayRef({output_features_, input_features_ / 2}),
            "weight must be int8 [N, K / 2]");
        TORCH_CHECK(
            weight_scales_.scalar_type() == torch::kFloat32 &&
                weight_scales_.numel() == output_features_,
            "weight_scales must be float32 [N]");
        TORCH_CHECK(
            bias_.scalar_type() == torch::kFloat32 && bias_.numel() == output_features_,
            "bias must be float32 [N]");
    }

    torch::Tensor run(torch::Tensor input) const {
        TORCH_CHECK(
            tensor_version(source_weight_) == source_weight_version_ &&
                tensor_version(source_weight_scales_) == source_weight_scales_version_ &&
                tensor_version(source_bias_) == source_bias_version_,
            "XQT_ROWWISE_W4A4_STALE_STATE");
        auto contiguous_input = input.contiguous();
        require_cuda_contiguous(contiguous_input, "input");
        require_same_device(weight_, contiguous_input, "input");
        TORCH_CHECK(
            contiguous_input.dim() >= 1 && contiguous_input.size(-1) == input_features_,
            "input trailing dimension does not match the dynamic rowwise ConvRot runner");
        const int64_t rows_64 = contiguous_input.numel() / input_features_;
        TORCH_CHECK(rows_64 > 0 && rows_64 <= INT32_MAX, "input row extent is unsupported");
        const int rows = static_cast<int>(rows_64);
        TORCH_CHECK(rows > 0, "input rows must be positive");
        c10::cuda::CUDAGuard guard(contiguous_input.device());
        const int kind = scalar_kind(contiguous_input);
        TORCH_CHECK(
            expected_scalar_kind_ < 0 || kind == expected_scalar_kind_,
            "XQT_ROWWISE_W4A4_INPUT_DTYPE_CHANGED");
        const cudaStream_t stream = current_stream(contiguous_input);
        const Workspace workspace = workspace_for(rows, stream);
        std::vector<int64_t> output_shape(contiguous_input.sizes().begin(), contiguous_input.sizes().end());
        output_shape.back() = output_features_;
        auto output = torch::empty(output_shape, contiguous_input.options());
        check_status(
            xqt_convrot_w4a4_rowwise_quantize(
                contiguous_input.data_ptr(),
                workspace.activation.data_ptr(),
                workspace.scales.data_ptr(),
                rows,
                input_features_,
                kind,
                stream),
            "rowwise ConvRot rotation plus INT4 quantization");
        check_status(
            xqt_convrot_w4a4_rowwise_gemm(
                workspace.activation.data_ptr(),
                weight_.data_ptr(),
                workspace.scales.data_ptr(),
                weight_scales_.data_ptr(),
                bias_.data_ptr(),
                output.data_ptr(),
                rows,
                output_features_,
                input_features_,
                kind,
                stream),
            "rowwise ConvRot CUTLASS W4A4 GEMM");
        return output;
    }

    int64_t workspace_count() const {
        std::lock_guard<std::mutex> lock(workspace_mutex_);
        return static_cast<int64_t>(workspaces_.size());
    }

private:
    struct Workspace {
        torch::Tensor activation;
        torch::Tensor scales;
    };

    using WorkspaceKey = std::tuple<int, std::uintptr_t>;

    Workspace workspace_for(int rows, cudaStream_t stream) const {
        const WorkspaceKey key{rows, reinterpret_cast<std::uintptr_t>(stream)};
        std::lock_guard<std::mutex> lock(workspace_mutex_);
        const auto found = workspaces_.find(key);
        if (found != workspaces_.end()) {
            return found->second;
        }
        Workspace workspace{
            torch::empty(
                {rows, input_features_ / 2},
                weight_.options().dtype(torch::kInt8)),
            torch::empty({rows}, weight_.options().dtype(torch::kFloat32)),
        };
        validate_static_state(
            workspace.activation,
            workspace.scales,
            weight_,
            weight_scales_,
            bias_,
            rows,
            input_features_,
            output_features_);
        workspaces_.emplace(key, workspace);
        return workspace;
    }

    torch::Tensor weight_;
    torch::Tensor weight_scales_;
    torch::Tensor bias_;
    torch::Tensor source_weight_;
    torch::Tensor source_weight_scales_;
    torch::Tensor source_bias_;
    int64_t source_weight_version_;
    int64_t source_weight_scales_version_;
    int64_t source_bias_version_;
    int expected_scalar_kind_;
    int input_features_;
    int output_features_;
    mutable std::mutex workspace_mutex_;
    mutable std::map<WorkspaceKey, Workspace> workspaces_;
};

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    py::class_<BoundConvRotW4A4RowwiseLinear>(module, "BoundConvRotW4A4RowwiseLinear")
        .def("__call__", &BoundConvRotW4A4RowwiseLinear::run);
    py::class_<DynamicConvRotW4A4RowwiseLinear>(module, "DynamicConvRotW4A4RowwiseLinear")
        .def("__call__", &DynamicConvRotW4A4RowwiseLinear::run)
        .def("workspace_count", &DynamicConvRotW4A4RowwiseLinear::workspace_count);
    module.def(
        "linear",
        &run_linear,
        py::arg("input"),
        py::arg("act"),
        py::arg("activation_scales"),
        py::arg("weight"),
        py::arg("weight_scales"),
        py::arg("bias"),
        py::arg("output_features"));
    module.def(
        "bind_linear",
        [](torch::Tensor act,
           torch::Tensor activation_scales,
           torch::Tensor weight,
           torch::Tensor weight_scales,
           torch::Tensor bias,
           int64_t rows,
           int64_t input_features,
           int64_t output_features) {
            return std::make_unique<BoundConvRotW4A4RowwiseLinear>(
                std::move(act),
                std::move(activation_scales),
                std::move(weight),
                std::move(weight_scales),
                std::move(bias),
                rows,
                input_features,
                output_features);
        },
        py::arg("act"),
        py::arg("activation_scales"),
        py::arg("weight"),
        py::arg("weight_scales"),
        py::arg("bias"),
        py::arg("rows"),
        py::arg("input_features"),
        py::arg("output_features"));
    module.def(
        "bind_dynamic_linear",
        [](torch::Tensor weight,
           torch::Tensor weight_scales,
           torch::Tensor bias,
           int64_t input_features,
           int64_t output_features,
           py::object source_weight,
           py::object source_weight_scales,
           py::object source_bias,
           int64_t expected_scalar_kind) {
            return std::make_unique<DynamicConvRotW4A4RowwiseLinear>(
                std::move(weight),
                std::move(weight_scales),
                std::move(bias),
                tensor_or_undefined(source_weight),
                tensor_or_undefined(source_weight_scales),
                tensor_or_undefined(source_bias),
                expected_scalar_kind,
                input_features,
                output_features);
        },
        py::arg("weight"),
        py::arg("weight_scales"),
        py::arg("bias"),
        py::arg("input_features"),
        py::arg("output_features"),
        py::arg("source_weight") = py::none(),
        py::arg("source_weight_scales") = py::none(),
        py::arg("source_bias") = py::none(),
        py::arg("expected_scalar_kind") = -1);
    module.def("version", []() { return std::string(xqt_convrot_w4a4_rowwise_version()); });
}
