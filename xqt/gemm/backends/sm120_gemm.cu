#include <cuda_runtime_api.h>

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cute/tensor.hpp"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/packed_stride.hpp"

using namespace cute;

#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)

namespace {

template <typename Element>
struct Fp8Kernel {
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using Tile = Shape<_128, _128, _128>;
  using Cluster = Shape<_1, _1, _1>;
  using ScaleConfig = decltype(
      cutlass::detail::sm120_trivial_blockwise_scale_config(Tile{}));
  using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
  using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());
  using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm120,
      cutlass::arch::OpClassTensorOp,
      Tile,
      Cluster,
      cutlass::epilogue::collective::EpilogueTileAuto,
      float,
      float,
      cutlass::bfloat16_t,
      LayoutC,
      8,
      cutlass::bfloat16_t,
      LayoutC,
      8,
      cutlass::epilogue::collective::EpilogueScheduleAuto
    >::CollectiveOp;
  using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm120,
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
      cutlass::gemm::collective::KernelScheduleAuto
    >::CollectiveOp;
  using Kernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>,
      Mainloop,
      Epilogue,
      void>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
};

using Fp8E4M3Gemm = typename Fp8Kernel<cutlass::float_e4m3_t>::Gemm;
using Fp8E5M2Gemm = typename Fp8Kernel<cutlass::float_e5m2_t>::Gemm;

template <typename Element, typename TileShape, typename Schedule>
struct Fp8GroupwiseKernel {
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using Tile = TileShape;
  using Cluster = Shape<_1, _1, _1>;
  using ScaleConfig = cutlass::detail::Sm120BlockwiseScaleConfig<1, 128, 128>;
  using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
  using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());
  using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm120,
      cutlass::arch::OpClassTensorOp,
      Tile,
      Cluster,
      cutlass::epilogue::collective::EpilogueTileAuto,
      float,
      float,
      cutlass::bfloat16_t,
      LayoutC,
      8,
      cutlass::bfloat16_t,
      LayoutC,
      8,
      cutlass::epilogue::collective::EpilogueScheduleAuto
    >::CollectiveOp;
  using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm120,
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
      Schedule
    >::CollectiveOp;
  using Kernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>,
      Mainloop,
      Epilogue,
      void>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
};

using Fp8GroupwiseE4M3Gemm =
    typename Fp8GroupwiseKernel<
        cutlass::float_e4m3_t,
        Shape<_128, _128, _128>,
        cutlass::gemm::KernelScheduleSm120Blockwise>::Gemm;
using Fp8GroupwiseE5M2Gemm =
    typename Fp8GroupwiseKernel<
        cutlass::float_e5m2_t,
        Shape<_128, _128, _128>,
        cutlass::gemm::KernelScheduleSm120Blockwise>::Gemm;
using Fp8GroupwisePingpongE4M3Gemm =
    typename Fp8GroupwiseKernel<
        cutlass::float_e4m3_t,
        Shape<_64, _128, _128>,
        cutlass::gemm::KernelTmaWarpSpecializedBlockwisePingpongSm120>::Gemm;
using Fp8GroupwisePingpongE5M2Gemm =
    typename Fp8GroupwiseKernel<
        cutlass::float_e5m2_t,
        Shape<_64, _128, _128>,
        cutlass::gemm::KernelTmaWarpSpecializedBlockwisePingpongSm120>::Gemm;

using Nvfp4Element = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using Nvfp4LayoutA = cutlass::layout::RowMajor;
using Nvfp4LayoutB = cutlass::layout::ColumnMajor;
using Nvfp4LayoutC = cutlass::layout::RowMajor;

template <typename TileShape, typename Schedule>
struct Nvfp4Kernel {
  using Tile = TileShape;
  using Cluster = Shape<_1, _1, _1>;
  using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm120,
      cutlass::arch::OpClassBlockScaledTensorOp,
      Tile,
      Cluster,
      cutlass::epilogue::collective::EpilogueTileAuto,
      float,
      float,
      cutlass::bfloat16_t,
      Nvfp4LayoutC,
      8,
      cutlass::bfloat16_t,
      Nvfp4LayoutC,
      8,
      cutlass::epilogue::collective::EpilogueScheduleAuto
    >::CollectiveOp;
  using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm120,
      cutlass::arch::OpClassBlockScaledTensorOp,
      Nvfp4Element,
      Nvfp4LayoutA,
      32,
      Nvfp4Element,
      Nvfp4LayoutB,
      32,
      float,
      Tile,
      Cluster,
      cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename Epilogue::SharedStorage))>,
      Schedule
    >::CollectiveOp;
  using Kernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>,
      Mainloop,
      Epilogue,
      void>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
};

using Nvfp4Gemm = typename Nvfp4Kernel<
    Shape<_128, _128, _128>,
    cutlass::gemm::KernelTmaWarpSpecializedNvf4Sm120>::Gemm;
using Nvfp4K256Gemm = typename Nvfp4Kernel<
    Shape<_128, _128, _256>,
    cutlass::gemm::KernelTmaWarpSpecializedNvf4Sm120>::Gemm;

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

template <typename Gemm>
int run_nvfp4_gemm(
    int m,
    int n,
    int k,
    typename Gemm::ElementA const* a,
    typename Gemm::ElementB const* b,
    cutlass::float_ue4m3_t const* scale_a,
    cutlass::float_ue4m3_t const* scale_b,
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
  using ScaleConfig =
      typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
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
xqt_sm120_fp8_e4m3_blockwise_compile_probe() {
  return static_cast<int>(sizeof(Fp8E4M3Gemm));
}

extern "C" __attribute__((visibility("default"))) int
xqt_sm120_fp8_e4m3_blockwise_bf16_run(
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
  using ScaleConfig = decltype(
      cutlass::detail::sm120_trivial_blockwise_scale_config(
          typename Fp8E4M3Gemm::TileShape{}));
  return run_fp8_gemm<Fp8E4M3Gemm, ScaleConfig>(
      m, n, k, static_cast<cutlass::float_e4m3_t const*>(a),
      static_cast<cutlass::float_e4m3_t const*>(b), scale_a, scale_b,
      static_cast<cutlass::bfloat16_t const*>(c),
      static_cast<cutlass::bfloat16_t*>(d), stream);
}

extern "C" __attribute__((visibility("default"))) int
xqt_sm120_fp8_e5m2_blockwise_compile_probe() {
  return static_cast<int>(sizeof(Fp8E5M2Gemm));
}

extern "C" __attribute__((visibility("default"))) int
xqt_sm120_fp8_e5m2_blockwise_bf16_run(
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
  using ScaleConfig = decltype(
      cutlass::detail::sm120_trivial_blockwise_scale_config(
          typename Fp8E5M2Gemm::TileShape{}));
  return run_fp8_gemm<Fp8E5M2Gemm, ScaleConfig>(
      m, n, k, static_cast<cutlass::float_e5m2_t const*>(a),
      static_cast<cutlass::float_e5m2_t const*>(b), scale_a, scale_b,
      static_cast<cutlass::bfloat16_t const*>(c),
      static_cast<cutlass::bfloat16_t*>(d), stream);
}

extern "C" __attribute__((visibility("default"))) int
xqt_sm120_fp8_e4m3_groupwise_compile_probe() {
  return static_cast<int>(sizeof(Fp8GroupwiseE4M3Gemm));
}

extern "C" __attribute__((visibility("default"))) int
xqt_sm120_fp8_e4m3_groupwise_bf16_run(
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
  using ScaleConfig = cutlass::detail::Sm120BlockwiseScaleConfig<1, 128, 128>;
  return run_fp8_gemm<Fp8GroupwiseE4M3Gemm, ScaleConfig>(
      m, n, k, static_cast<cutlass::float_e4m3_t const*>(a),
      static_cast<cutlass::float_e4m3_t const*>(b), scale_a, scale_b,
      static_cast<cutlass::bfloat16_t const*>(c),
      static_cast<cutlass::bfloat16_t*>(d), stream);
}

extern "C" __attribute__((visibility("default"))) int
xqt_sm120_fp8_e5m2_groupwise_compile_probe() {
  return static_cast<int>(sizeof(Fp8GroupwiseE5M2Gemm));
}

extern "C" __attribute__((visibility("default"))) int
xqt_sm120_fp8_e5m2_groupwise_bf16_run(
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
  using ScaleConfig = cutlass::detail::Sm120BlockwiseScaleConfig<1, 128, 128>;
  return run_fp8_gemm<Fp8GroupwiseE5M2Gemm, ScaleConfig>(
      m, n, k, static_cast<cutlass::float_e5m2_t const*>(a),
      static_cast<cutlass::float_e5m2_t const*>(b), scale_a, scale_b,
      static_cast<cutlass::bfloat16_t const*>(c),
      static_cast<cutlass::bfloat16_t*>(d), stream);
}

extern "C" __attribute__((visibility("default"))) int
xqt_sm120_fp8_e4m3_groupwise_pingpong_compile_probe() {
  return static_cast<int>(sizeof(Fp8GroupwisePingpongE4M3Gemm));
}

extern "C" __attribute__((visibility("default"))) int
xqt_sm120_fp8_e4m3_groupwise_pingpong_bf16_run(
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
  using ScaleConfig = cutlass::detail::Sm120BlockwiseScaleConfig<1, 128, 128>;
  return run_fp8_gemm<Fp8GroupwisePingpongE4M3Gemm, ScaleConfig>(
      m, n, k, static_cast<cutlass::float_e4m3_t const*>(a),
      static_cast<cutlass::float_e4m3_t const*>(b), scale_a, scale_b,
      static_cast<cutlass::bfloat16_t const*>(c),
      static_cast<cutlass::bfloat16_t*>(d), stream);
}

extern "C" __attribute__((visibility("default"))) int
xqt_sm120_fp8_e5m2_groupwise_pingpong_compile_probe() {
  return static_cast<int>(sizeof(Fp8GroupwisePingpongE5M2Gemm));
}

extern "C" __attribute__((visibility("default"))) int
xqt_sm120_fp8_e5m2_groupwise_pingpong_bf16_run(
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
  using ScaleConfig = cutlass::detail::Sm120BlockwiseScaleConfig<1, 128, 128>;
  return run_fp8_gemm<Fp8GroupwisePingpongE5M2Gemm, ScaleConfig>(
      m, n, k, static_cast<cutlass::float_e5m2_t const*>(a),
      static_cast<cutlass::float_e5m2_t const*>(b), scale_a, scale_b,
      static_cast<cutlass::bfloat16_t const*>(c),
      static_cast<cutlass::bfloat16_t*>(d), stream);
}

extern "C" __attribute__((visibility("default"))) int
xqt_sm120_nvfp4_compile_probe() {
  return static_cast<int>(sizeof(Nvfp4Gemm));
}

extern "C" __attribute__((visibility("default"))) int
xqt_sm120_nvfp4_bf16_run(
    int m,
    int n,
    int k,
    void const* a,
    void const* b,
    void const* scale_a,
    void const* scale_b,
    void const* c,
    void* d,
    void* stream) {
  return run_nvfp4_gemm<Nvfp4Gemm>(
      m, n, k, static_cast<Nvfp4Gemm::ElementA const*>(a),
      static_cast<Nvfp4Gemm::ElementB const*>(b),
      static_cast<cutlass::float_ue4m3_t const*>(scale_a),
      static_cast<cutlass::float_ue4m3_t const*>(scale_b),
      static_cast<Nvfp4Gemm::ElementC const*>(c),
      static_cast<Nvfp4Gemm::ElementD*>(d), stream);
}

extern "C" __attribute__((visibility("default"))) int
xqt_sm120_nvfp4_k256_compile_probe() {
  return static_cast<int>(sizeof(Nvfp4K256Gemm));
}

extern "C" __attribute__((visibility("default"))) int
xqt_sm120_nvfp4_k256_bf16_run(
    int m,
    int n,
    int k,
    void const* a,
    void const* b,
    void const* scale_a,
    void const* scale_b,
    void const* c,
    void* d,
    void* stream) {
  return run_nvfp4_gemm<Nvfp4K256Gemm>(
      m, n, k, static_cast<Nvfp4K256Gemm::ElementA const*>(a),
      static_cast<Nvfp4K256Gemm::ElementB const*>(b),
      static_cast<cutlass::float_ue4m3_t const*>(scale_a),
      static_cast<cutlass::float_ue4m3_t const*>(scale_b),
      static_cast<Nvfp4K256Gemm::ElementC const*>(c),
      static_cast<Nvfp4K256Gemm::ElementD*>(d), stream);
}

#else

extern "C" __attribute__((visibility("default"))) int
xqt_sm120_fp8_e4m3_blockwise_compile_probe() {
  return 0;
}

extern "C" __attribute__((visibility("default"))) int
xqt_sm120_fp8_e4m3_blockwise_bf16_run(
    int, int, int, void const*, void const*, float const*, float const*,
    void const*, void*, void*) {
  return 0;
}

extern "C" __attribute__((visibility("default"))) int
xqt_sm120_fp8_e5m2_blockwise_compile_probe() {
  return 0;
}

extern "C" __attribute__((visibility("default"))) int
xqt_sm120_fp8_e5m2_blockwise_bf16_run(
    int, int, int, void const*, void const*, float const*, float const*,
    void const*, void*, void*) {
  return 0;
}

extern "C" __attribute__((visibility("default"))) int
xqt_sm120_fp8_e4m3_groupwise_compile_probe() {
  return 0;
}

extern "C" __attribute__((visibility("default"))) int
xqt_sm120_fp8_e4m3_groupwise_bf16_run(
    int, int, int, void const*, void const*, float const*, float const*,
    void const*, void*, void*) {
  return 0;
}

extern "C" __attribute__((visibility("default"))) int
xqt_sm120_fp8_e5m2_groupwise_compile_probe() {
  return 0;
}

extern "C" __attribute__((visibility("default"))) int
xqt_sm120_fp8_e5m2_groupwise_bf16_run(
    int, int, int, void const*, void const*, float const*, float const*,
    void const*, void*, void*) {
  return 0;
}

extern "C" __attribute__((visibility("default"))) int
xqt_sm120_fp8_e4m3_groupwise_pingpong_compile_probe() {
  return 0;
}

extern "C" __attribute__((visibility("default"))) int
xqt_sm120_fp8_e4m3_groupwise_pingpong_bf16_run(
    int, int, int, void const*, void const*, float const*, float const*,
    void const*, void*, void*) {
  return 0;
}

extern "C" __attribute__((visibility("default"))) int
xqt_sm120_fp8_e5m2_groupwise_pingpong_compile_probe() {
  return 0;
}

extern "C" __attribute__((visibility("default"))) int
xqt_sm120_fp8_e5m2_groupwise_pingpong_bf16_run(
    int, int, int, void const*, void const*, float const*, float const*,
    void const*, void*, void*) {
  return 0;
}

extern "C" __attribute__((visibility("default"))) int
xqt_sm120_nvfp4_compile_probe() {
  return 0;
}

extern "C" __attribute__((visibility("default"))) int
xqt_sm120_nvfp4_bf16_run(
    int, int, int, void const*, void const*, void const*, void const*,
    void const*, void*, void*) {
  return 0;
}

extern "C" __attribute__((visibility("default"))) int
xqt_sm120_nvfp4_k256_compile_probe() {
  return 0;
}

extern "C" __attribute__((visibility("default"))) int
xqt_sm120_nvfp4_k256_bf16_run(
    int, int, int, void const*, void const*, void const*, void const*,
    void const*, void*, void*) {
  return 0;
}

#endif
