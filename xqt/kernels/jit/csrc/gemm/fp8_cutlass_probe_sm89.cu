// SM89 FP8 capability probe.
//
// This artifact deliberately does not expose a production GEMM entry.  It
// instantiates CUTLASS's SM89 m16n8k32 FP8 MMA for E4M3 and E5M2 so the build
// and SASS gates can distinguish a real instruction path from a dequantized
// dense fallback.

#include <cuda_runtime.h>

#include "cutlass/arch/mma.h"

template <typename ElementA, typename ElementB>
__global__ void fp8_mma_probe_kernel(float* output) {
  using Mma = cutlass::arch::Mma<
      cutlass::gemm::GemmShape<16, 8, 32>,
      32,
      ElementA,
      cutlass::layout::RowMajor,
      ElementB,
      cutlass::layout::ColumnMajor,
      float,
      cutlass::layout::RowMajor,
      cutlass::arch::OpMultiplyAdd>;
  typename Mma::FragmentA fragment_a;
  typename Mma::FragmentB fragment_b;
  typename Mma::FragmentC fragment_c;
  fragment_c.clear();
#pragma unroll
  for (int index = 0; index < int(Mma::FragmentA::kElements); ++index) {
    fragment_a[index] = ElementA(0.0f);
  }
#pragma unroll
  for (int index = 0; index < int(Mma::FragmentB::kElements); ++index) {
    fragment_b[index] = ElementB(0.0f);
  }
  Mma mma;
  mma(fragment_c, fragment_a, fragment_b, fragment_c);
  if ((threadIdx.x & 31) == 0) {
    output[0] = fragment_c[0];
  }
}

extern "C" int xqt_fp8_cutlass_probe_sm89_run(
    void* output,
    int format,
    void* stream) {
  if (!output || (format != 0 && format != 1)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  if (format == 0) {
    fp8_mma_probe_kernel<cutlass::float_e4m3_t, cutlass::float_e4m3_t>
        <<<1, 32, 0, static_cast<cudaStream_t>(stream)>>>(
            static_cast<float*>(output));
  } else {
    fp8_mma_probe_kernel<cutlass::float_e5m2_t, cutlass::float_e5m2_t>
        <<<1, 32, 0, static_cast<cudaStream_t>(stream)>>>(
            static_cast<float*>(output));
  }
  return static_cast<int>(cudaGetLastError());
}

extern "C" const char* xqt_fp8_cutlass_probe_sm89_version() {
  return "sm89-fp8-cutlass-mma-probe-v1 m16n8k32 e4m3/e5m2";
}
