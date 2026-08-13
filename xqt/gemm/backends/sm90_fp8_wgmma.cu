#include <cuda_runtime_api.h>

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cute/tensor.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/packed_stride.hpp"

using namespace cute;

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

namespace {

using DenseLayoutA = cutlass::layout::RowMajor;
using DenseLayoutB = cutlass::layout::ColumnMajor;
using DenseLayoutC = cutlass::layout::RowMajor;
using DenseAccumulator = float;
using DenseTile = Shape<_128, _128, _64>;
using DenseCluster = Shape<_2, _1, _1>;

template <typename Element>
struct DenseKernel {
  using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90,
      cutlass::arch::OpClassTensorOp,
      DenseTile,
      DenseCluster,
      cutlass::epilogue::collective::EpilogueTileAuto,
      DenseAccumulator,
      DenseAccumulator,
      Element,
      DenseLayoutC,
      8,
      Element,
      DenseLayoutC,
      8,
      cutlass::epilogue::collective::EpilogueScheduleAuto
    >::CollectiveOp;
  using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90,
      cutlass::arch::OpClassTensorOp,
      Element,
      DenseLayoutA,
      8,
      Element,
      DenseLayoutB,
      8,
      DenseAccumulator,
      DenseTile,
      DenseCluster,
      cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename Epilogue::SharedStorage))>,
      cutlass::gemm::collective::KernelScheduleAuto
    >::CollectiveOp;
  using Kernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>,
      Mainloop,
      Epilogue>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
};

using DenseFp16Gemm = typename DenseKernel<cutlass::half_t>::Gemm;
using DenseBf16Gemm = typename DenseKernel<cutlass::bfloat16_t>::Gemm;

template <typename Element, typename ElementOutput = float>
struct Fp8Kernel {
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using Tile = Shape<_128, _128, _128>;
  using Cluster = Shape<_1, _2, _1>;
  using ScaleConfig = decltype(
      cutlass::detail::sm90_trivial_blockwise_scale_config(Tile{}));
  using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
  using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());
  using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90,
      cutlass::arch::OpClassTensorOp,
      Tile,
      Cluster,
      cutlass::epilogue::collective::EpilogueTileAuto,
      float,
      float,
      ElementOutput,
      LayoutC,
      128 / cutlass::sizeof_bits<ElementOutput>::value,
      ElementOutput,
      LayoutC,
      128 / cutlass::sizeof_bits<ElementOutput>::value,
      cutlass::epilogue::TmaWarpSpecializedCooperative
    >::CollectiveOp;
  using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90,
      cutlass::arch::OpClassTensorOp,
      Element,
      cute::tuple<LayoutA, LayoutSFA>,
      16,
      Element,
      cute::tuple<LayoutB, LayoutSFB>,
      16,
      float,
      Tile,
      Cluster,
      cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename Epilogue::SharedStorage))>,
      cutlass::gemm::KernelTmaWarpSpecializedCooperativeFP8Blockwise
    >::CollectiveOp;
  using Kernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>,
      Mainloop,
      Epilogue>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
};

using Fp8E4M3Gemm = typename Fp8Kernel<cutlass::float_e4m3_t>::Gemm;
using Fp8E5M2Gemm = typename Fp8Kernel<cutlass::float_e5m2_t>::Gemm;
using Fp8E4M3Fp16Gemm =
    typename Fp8Kernel<cutlass::float_e4m3_t, cutlass::half_t>::Gemm;
using Fp8E4M3Bf16Gemm =
    typename Fp8Kernel<cutlass::float_e4m3_t, cutlass::bfloat16_t>::Gemm;
using Fp8E5M2Fp16Gemm =
    typename Fp8Kernel<cutlass::float_e5m2_t, cutlass::half_t>::Gemm;
using Fp8E5M2Bf16Gemm =
    typename Fp8Kernel<cutlass::float_e5m2_t, cutlass::bfloat16_t>::Gemm;

using Fp8ScaleConfig = decltype(
    cutlass::detail::sm90_trivial_blockwise_scale_config(
        Shape<_128, _128, _128>{}));

using Fp8GroupwiseScaleConfig = cutlass::detail::Sm90BlockwiseScaleConfig<
    1,
    128,
    128,
    cute::GMMA::Major::MN,
    cute::GMMA::Major::K>;

template <typename Element, typename ElementOutput>
struct Fp8GroupwisePingpongKernel {
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using Tile = Shape<_128, _128, _128>;
  using Cluster = Shape<_1, _2, _1>;
  using ScaleConfig = Fp8GroupwiseScaleConfig;
  using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
  using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());
  using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90,
      cutlass::arch::OpClassTensorOp,
      Tile,
      Cluster,
      cutlass::epilogue::collective::EpilogueTileAuto,
      float,
      float,
      ElementOutput,
      LayoutC,
      128 / cutlass::sizeof_bits<ElementOutput>::value,
      ElementOutput,
      LayoutC,
      128 / cutlass::sizeof_bits<ElementOutput>::value,
      cutlass::epilogue::TmaWarpSpecializedCooperative
    >::CollectiveOp;
  using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90,
      cutlass::arch::OpClassTensorOp,
      Element,
      cute::tuple<LayoutA, LayoutSFA>,
      16,
      Element,
      cute::tuple<LayoutB, LayoutSFB>,
      16,
      float,
      Tile,
      Cluster,
      cutlass::gemm::collective::StageCountAutoCarveout<
          static_cast<int>(sizeof(typename Epilogue::SharedStorage))>,
      cutlass::gemm::KernelTmaWarpSpecializedPingpongFP8Blockwise
    >::CollectiveOp;
  using Kernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>,
      Mainloop,
      Epilogue>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
};

using Fp8GroupwisePingpongE4M3Fp16Gemm =
    typename Fp8GroupwisePingpongKernel<
        cutlass::float_e4m3_t,
        cutlass::half_t>::Gemm;
using Fp8GroupwisePingpongE4M3Bf16Gemm =
    typename Fp8GroupwisePingpongKernel<
        cutlass::float_e4m3_t,
        cutlass::bfloat16_t>::Gemm;
using Fp8GroupwisePingpongE5M2Fp16Gemm =
    typename Fp8GroupwisePingpongKernel<
        cutlass::float_e5m2_t,
        cutlass::half_t>::Gemm;
using Fp8GroupwisePingpongE5M2Bf16Gemm =
    typename Fp8GroupwisePingpongKernel<
        cutlass::float_e5m2_t,
        cutlass::bfloat16_t>::Gemm;

template <typename Gemm>
int run_dense_gemm(
    int m,
    int n,
    int k,
    typename Gemm::ElementA const* a,
    typename Gemm::ElementB const* b,
    typename Gemm::ElementC const* c,
    typename Gemm::ElementD* d,
    void* stream_ptr) {
  if (m <= 0 || n <= 0 || k <= 0 || a == nullptr || b == nullptr ||
      c == nullptr || d == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;
  auto stride_a = cutlass::make_cute_packed_stride(
      StrideA{}, cute::make_shape(m, k, 1));
  auto stride_b = cutlass::make_cute_packed_stride(
      StrideB{}, cute::make_shape(n, k, 1));
  auto stride_c = cutlass::make_cute_packed_stride(
      StrideC{}, cute::make_shape(m, n, 1));
  auto stride_d = cutlass::make_cute_packed_stride(
      StrideD{}, cute::make_shape(m, n, 1));
  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {m, n, k, 1},
      {a, stride_a, b, stride_b},
      {{}, c, stride_c, d, stride_d}};
  Gemm gemm;
  if (gemm.can_implement(arguments) != cutlass::Status::kSuccess) {
    return 1;
  }
  cutlass::device_memory::allocation<uint8_t> workspace(
      Gemm::get_workspace_size(arguments));
  if (gemm.initialize(arguments, workspace.get(), stream) != cutlass::Status::kSuccess) {
    return 2;
  }
  return gemm.run(stream) == cutlass::Status::kSuccess ? 0 : 3;
}

template <typename Gemm, typename ScaleConfig>
int run_fp8_gemm(
    int m,
    int n,
    int k,
    typename Gemm::ElementA const* a,
    typename Gemm::ElementB const* b,
    float const* scale_a,
    float const* scale_b,
    typename Gemm::ElementC const* c,
    typename Gemm::ElementD* d,
    void* stream_ptr) {
  if (m <= 0 || n <= 0 || k <= 0 || a == nullptr || b == nullptr ||
      scale_a == nullptr || scale_b == nullptr || c == nullptr || d == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;
  auto stride_a = cutlass::make_cute_packed_stride(
      StrideA{}, cute::make_shape(m, k, 1));
  auto stride_b = cutlass::make_cute_packed_stride(
      StrideB{}, cute::make_shape(n, k, 1));
  auto stride_c = cutlass::make_cute_packed_stride(
      StrideC{}, cute::make_shape(m, n, 1));
  auto stride_d = cutlass::make_cute_packed_stride(
      StrideD{}, cute::make_shape(m, n, 1));
  auto layout_sfa =
      ScaleConfig::tile_atom_to_shape_SFA(cute::make_shape(m, n, k, 1));
  auto layout_sfb =
      ScaleConfig::tile_atom_to_shape_SFB(cute::make_shape(m, n, k, 1));
  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {m, n, k, 1},
      {a, stride_a, b, stride_b, scale_a, layout_sfa, scale_b, layout_sfb},
      {{}, c, stride_c, d, stride_d}};
  Gemm gemm;
  if (gemm.can_implement(arguments) != cutlass::Status::kSuccess) {
    return 1;
  }
  cutlass::device_memory::allocation<uint8_t> workspace(
      Gemm::get_workspace_size(arguments));
  if (gemm.initialize(arguments, workspace.get(), stream) != cutlass::Status::kSuccess) {
    return 2;
  }
  return gemm.run(stream) == cutlass::Status::kSuccess ? 0 : 3;
}

}  // namespace

extern "C" __attribute__((visibility("default"))) int
xqt_sm90_dense_wgmma_compile_probe() {
  return static_cast<int>(sizeof(DenseFp16Gemm) + sizeof(DenseBf16Gemm));
}

extern "C" __attribute__((visibility("default"))) int
xqt_sm90_dense_fp16_wgmma_run(
    int m,
    int n,
    int k,
    void const* a,
    void const* b,
    void const* c,
    void* d,
    void* stream) {
  return run_dense_gemm<DenseFp16Gemm>(
      m, n, k, static_cast<cutlass::half_t const*>(a),
      static_cast<cutlass::half_t const*>(b),
      static_cast<cutlass::half_t const*>(c),
      static_cast<cutlass::half_t*>(d), stream);
}

extern "C" __attribute__((visibility("default"))) int
xqt_sm90_dense_bf16_wgmma_run(
    int m,
    int n,
    int k,
    void const* a,
    void const* b,
    void const* c,
    void* d,
    void* stream) {
  return run_dense_gemm<DenseBf16Gemm>(
      m, n, k, static_cast<cutlass::bfloat16_t const*>(a),
      static_cast<cutlass::bfloat16_t const*>(b),
      static_cast<cutlass::bfloat16_t const*>(c),
      static_cast<cutlass::bfloat16_t*>(d), stream);
}

extern "C" __attribute__((visibility("default"))) int
xqt_sm90_fp8_e4m3_wgmma_compile_probe() {
  return static_cast<int>(sizeof(Fp8E4M3Gemm));
}

extern "C" __attribute__((visibility("default"))) int
xqt_sm90_fp8_e4m3_fp16_wgmma_run(
    int m,
    int n,
    int k,
    void const* a,
    void const* b,
    float const* scale_a,
    float const* scale_b,
    void const* c,
    void* d,
    void* stream) {
  return run_fp8_gemm<Fp8E4M3Fp16Gemm, Fp8ScaleConfig>(
      m, n, k, static_cast<cutlass::float_e4m3_t const*>(a),
      static_cast<cutlass::float_e4m3_t const*>(b), scale_a, scale_b,
      static_cast<cutlass::half_t const*>(c), static_cast<cutlass::half_t*>(d),
      stream);
}

extern "C" __attribute__((visibility("default"))) int
xqt_sm90_fp8_e4m3_bf16_wgmma_run(
    int m,
    int n,
    int k,
    void const* a,
    void const* b,
    float const* scale_a,
    float const* scale_b,
    void const* c,
    void* d,
    void* stream) {
  return run_fp8_gemm<Fp8E4M3Bf16Gemm, Fp8ScaleConfig>(
      m, n, k, static_cast<cutlass::float_e4m3_t const*>(a),
      static_cast<cutlass::float_e4m3_t const*>(b), scale_a, scale_b,
      static_cast<cutlass::bfloat16_t const*>(c),
      static_cast<cutlass::bfloat16_t*>(d), stream);
}

extern "C" __attribute__((visibility("default"))) int
xqt_sm90_fp8_e5m2_wgmma_compile_probe() {
  return static_cast<int>(sizeof(Fp8E5M2Gemm));
}

extern "C" __attribute__((visibility("default"))) int
xqt_sm90_fp8_e5m2_fp16_wgmma_run(
    int m,
    int n,
    int k,
    void const* a,
    void const* b,
    float const* scale_a,
    float const* scale_b,
    void const* c,
    void* d,
    void* stream) {
  return run_fp8_gemm<Fp8E5M2Fp16Gemm, Fp8ScaleConfig>(
      m, n, k, static_cast<cutlass::float_e5m2_t const*>(a),
      static_cast<cutlass::float_e5m2_t const*>(b), scale_a, scale_b,
      static_cast<cutlass::half_t const*>(c), static_cast<cutlass::half_t*>(d),
      stream);
}

extern "C" __attribute__((visibility("default"))) int
xqt_sm90_fp8_e5m2_bf16_wgmma_run(
    int m,
    int n,
    int k,
    void const* a,
    void const* b,
    float const* scale_a,
    float const* scale_b,
    void const* c,
    void* d,
    void* stream) {
  return run_fp8_gemm<Fp8E5M2Bf16Gemm, Fp8ScaleConfig>(
      m, n, k, static_cast<cutlass::float_e5m2_t const*>(a),
      static_cast<cutlass::float_e5m2_t const*>(b), scale_a, scale_b,
      static_cast<cutlass::bfloat16_t const*>(c),
      static_cast<cutlass::bfloat16_t*>(d), stream);
}

extern "C" __attribute__((visibility("default"))) int
xqt_sm90_fp8_e4m3_groupwise_pingpong_compile_probe() {
  return static_cast<int>(sizeof(Fp8GroupwisePingpongE4M3Bf16Gemm));
}

extern "C" __attribute__((visibility("default"))) int
xqt_sm90_fp8_e4m3_groupwise_pingpong_fp16_run(
    int m,
    int n,
    int k,
    void const* a,
    void const* b,
    float const* scale_a,
    float const* scale_b,
    void const* c,
    void* d,
    void* stream) {
  return run_fp8_gemm<Fp8GroupwisePingpongE4M3Fp16Gemm, Fp8GroupwiseScaleConfig>(
      m, n, k, static_cast<cutlass::float_e4m3_t const*>(a),
      static_cast<cutlass::float_e4m3_t const*>(b), scale_a, scale_b,
      static_cast<cutlass::half_t const*>(c),
      static_cast<cutlass::half_t*>(d), stream);
}

extern "C" __attribute__((visibility("default"))) int
xqt_sm90_fp8_e4m3_groupwise_pingpong_bf16_run(
    int m,
    int n,
    int k,
    void const* a,
    void const* b,
    float const* scale_a,
    float const* scale_b,
    void const* c,
    void* d,
    void* stream) {
  return run_fp8_gemm<Fp8GroupwisePingpongE4M3Bf16Gemm, Fp8GroupwiseScaleConfig>(
      m, n, k, static_cast<cutlass::float_e4m3_t const*>(a),
      static_cast<cutlass::float_e4m3_t const*>(b), scale_a, scale_b,
      static_cast<cutlass::bfloat16_t const*>(c),
      static_cast<cutlass::bfloat16_t*>(d), stream);
}

extern "C" __attribute__((visibility("default"))) int
xqt_sm90_fp8_e5m2_groupwise_pingpong_compile_probe() {
  return static_cast<int>(sizeof(Fp8GroupwisePingpongE5M2Bf16Gemm));
}

extern "C" __attribute__((visibility("default"))) int
xqt_sm90_fp8_e5m2_groupwise_pingpong_fp16_run(
    int m,
    int n,
    int k,
    void const* a,
    void const* b,
    float const* scale_a,
    float const* scale_b,
    void const* c,
    void* d,
    void* stream) {
  return run_fp8_gemm<Fp8GroupwisePingpongE5M2Fp16Gemm, Fp8GroupwiseScaleConfig>(
      m, n, k, static_cast<cutlass::float_e5m2_t const*>(a),
      static_cast<cutlass::float_e5m2_t const*>(b), scale_a, scale_b,
      static_cast<cutlass::half_t const*>(c),
      static_cast<cutlass::half_t*>(d), stream);
}

extern "C" __attribute__((visibility("default"))) int
xqt_sm90_fp8_e5m2_groupwise_pingpong_bf16_run(
    int m,
    int n,
    int k,
    void const* a,
    void const* b,
    float const* scale_a,
    float const* scale_b,
    void const* c,
    void* d,
    void* stream) {
  return run_fp8_gemm<Fp8GroupwisePingpongE5M2Bf16Gemm, Fp8GroupwiseScaleConfig>(
      m, n, k, static_cast<cutlass::float_e5m2_t const*>(a),
      static_cast<cutlass::float_e5m2_t const*>(b), scale_a, scale_b,
      static_cast<cutlass::bfloat16_t const*>(c),
      static_cast<cutlass::bfloat16_t*>(d), stream);
}

#else

extern "C" __attribute__((visibility("default"))) int
xqt_sm90_dense_wgmma_compile_probe() {
  return 0;
}

extern "C" __attribute__((visibility("default"))) int
xqt_sm90_dense_fp16_wgmma_run(
    int, int, int, void const*, void const*, void const*, void*, void*) {
  return 0;
}

extern "C" __attribute__((visibility("default"))) int
xqt_sm90_dense_bf16_wgmma_run(
    int, int, int, void const*, void const*, void const*, void*, void*) {
  return 0;
}

extern "C" __attribute__((visibility("default"))) int
xqt_sm90_fp8_e4m3_wgmma_compile_probe() {
  return 0;
}

extern "C" __attribute__((visibility("default"))) int
xqt_sm90_fp8_e4m3_fp16_wgmma_run(
    int, int, int, void const*, void const*, float const*, float const*,
    void const*, void*, void*) {
  return 0;
}

extern "C" __attribute__((visibility("default"))) int
xqt_sm90_fp8_e4m3_bf16_wgmma_run(
    int, int, int, void const*, void const*, float const*, float const*,
    void const*, void*, void*) {
  return 0;
}

extern "C" __attribute__((visibility("default"))) int
xqt_sm90_fp8_e5m2_wgmma_compile_probe() {
  return 0;
}

extern "C" __attribute__((visibility("default"))) int
xqt_sm90_fp8_e5m2_fp16_wgmma_run(
    int, int, int, void const*, void const*, float const*, float const*,
    void const*, void*, void*) {
  return 0;
}

extern "C" __attribute__((visibility("default"))) int
xqt_sm90_fp8_e5m2_bf16_wgmma_run(
    int, int, int, void const*, void const*, float const*, float const*,
    void const*, void*, void*) {
  return 0;
}

extern "C" __attribute__((visibility("default"))) int
xqt_sm90_fp8_e4m3_groupwise_pingpong_compile_probe() {
  return 0;
}

extern "C" __attribute__((visibility("default"))) int
xqt_sm90_fp8_e4m3_groupwise_pingpong_fp16_run(
    int, int, int, void const*, void const*, float const*, float const*,
    void const*, void*, void*) {
  return 0;
}

extern "C" __attribute__((visibility("default"))) int
xqt_sm90_fp8_e4m3_groupwise_pingpong_bf16_run(
    int, int, int, void const*, void const*, float const*, float const*,
    void const*, void*, void*) {
  return 0;
}

extern "C" __attribute__((visibility("default"))) int
xqt_sm90_fp8_e5m2_groupwise_pingpong_compile_probe() {
  return 0;
}

extern "C" __attribute__((visibility("default"))) int
xqt_sm90_fp8_e5m2_groupwise_pingpong_fp16_run(
    int, int, int, void const*, void const*, float const*, float const*,
    void const*, void*, void*) {
  return 0;
}

extern "C" __attribute__((visibility("default"))) int
xqt_sm90_fp8_e5m2_groupwise_pingpong_bf16_run(
    int, int, int, void const*, void const*, float const*, float const*,
    void const*, void*, void*) {
  return 0;
}

#endif
