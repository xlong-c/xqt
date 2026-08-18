#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cute/tensor.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"

using namespace cute;

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

template <typename Element>
struct Sm90Fp8PingpongProbe {
  using Tile = Shape<_128, _128, _128>;
  using Cluster = Shape<_1, _2, _1>;
  using ScaleConfig = cutlass::detail::Sm90BlockwiseScaleConfig<
      1, 128, 128, cute::GMMA::Major::MN, cute::GMMA::Major::K>;
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
      cutlass::bfloat16_t,
      cutlass::layout::RowMajor,
      8,
      cutlass::bfloat16_t,
      cutlass::layout::RowMajor,
      8,
      cutlass::epilogue::TmaWarpSpecializedCooperative
    >::CollectiveOp;
  using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90,
      cutlass::arch::OpClassTensorOp,
      Element,
      cute::tuple<cutlass::layout::RowMajor, LayoutSFA>,
      16,
      Element,
      cute::tuple<cutlass::layout::ColumnMajor, LayoutSFB>,
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

using Sm90Fp8PingpongE4M3 =
    typename Sm90Fp8PingpongProbe<cutlass::float_e4m3_t>::Gemm;
using Sm90Fp8PingpongE5M2 =
    typename Sm90Fp8PingpongProbe<cutlass::float_e5m2_t>::Gemm;

extern "C" int xqt_compile_probe_sm90_fp8_pingpong() {
  return static_cast<int>(
      sizeof(Sm90Fp8PingpongE4M3) + sizeof(Sm90Fp8PingpongE5M2));
}

#endif
