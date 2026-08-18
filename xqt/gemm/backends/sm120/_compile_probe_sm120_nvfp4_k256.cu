#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cute/tensor.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"

using namespace cute;

#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)

struct Sm120Nvfp4K256Probe {
  using Tile = Shape<_128, _128, _256>;
  using Cluster = Shape<_1, _1, _1>;
  using Nvfp4Element = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
  using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm120,
      cutlass::arch::OpClassBlockScaledTensorOp,
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
      cutlass::epilogue::collective::EpilogueScheduleAuto
    >::CollectiveOp;
  using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm120,
      cutlass::arch::OpClassBlockScaledTensorOp,
      Nvfp4Element,
      cutlass::layout::RowMajor,
      32,
      Nvfp4Element,
      cutlass::layout::ColumnMajor,
      32,
      float,
      Tile,
      Cluster,
      cutlass::gemm::collective::StageCountAutoCarveout<
          static_cast<int>(sizeof(typename Epilogue::SharedStorage))>,
      cutlass::gemm::KernelTmaWarpSpecializedNvf4Sm120
    >::CollectiveOp;
  using Kernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>,
      Mainloop,
      Epilogue,
      void>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
};

extern "C" int xqt_compile_probe_sm120_nvfp4_k256() {
  return static_cast<int>(sizeof(Sm120Nvfp4K256Probe::Gemm));
}

#endif
