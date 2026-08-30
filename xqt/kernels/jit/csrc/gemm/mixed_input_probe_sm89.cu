// CUTLASS SM89 mixed-input capability probe.
//
// This probe deliberately targets the supported one-step upcast pair
// (int8 x int4 -> int32). It is evidence for the CUTLASS MMA primitive and
// canonical nibble decoding only; it is not a W4A16 production kernel.

#include <cuda_runtime.h>

#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/device/gemm.h"

using ProbeGemm = cutlass::gemm::device::Gemm<
    int8_t,
    cutlass::layout::RowMajor,
    cutlass::int4b_t,
    cutlass::layout::ColumnMajor,
    int32_t,
    cutlass::layout::RowMajor,
    int32_t,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 128>,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<16, 8, 32>,
    cutlass::epilogue::thread::LinearCombination<int32_t, 8, int32_t, int32_t>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    2,
    16,
    16,
    false,
    cutlass::arch::OpMultiplyAddMixedInputUpcast>;

__global__ void decode_low_high_kernel(
    unsigned char const* packed,
    int8_t* output,
    int elements) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= elements) {
    return;
  }
  unsigned char value = packed[index / 2];
  int code = (index & 1) == 0 ? static_cast<int>(value & 0x0f)
                             : static_cast<int>((value >> 4) & 0x0f);
  output[index] = static_cast<int8_t>(code >= 8 ? code - 16 : code);
}

extern "C" int xqt_sm89_mixed_input_probe(
    void const* packed,
    void* decoded,
    int elements,
    void* stream) {
  if (packed == nullptr || decoded == nullptr || elements <= 0) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  int blocks = (elements + 255) / 256;
  decode_low_high_kernel<<<blocks, 256, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<unsigned char const*>(packed), static_cast<int8_t*>(decoded), elements);
  return static_cast<int>(cudaGetLastError());
}

extern "C" int xqt_sm89_mixed_input_probe_type_size() {
  return static_cast<int>(sizeof(ProbeGemm));
}

extern "C" const char* xqt_sm89_mixed_input_probe_version() {
  return "sm89-cutlass-mixed-input-int8xint4-v1 canonical-low-high";
}
