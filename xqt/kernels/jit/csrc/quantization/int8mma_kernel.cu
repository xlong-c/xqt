// int8mma Ada sm_89: math B[K,N] or prepacked B[N,K]
// C = half((A@B)_i32 * sa * sw)
// Prepacked B[N,K]: offline transpose so G2S is coalesced along K and B fragments are uint32 loads

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cstdint>
#include <memory>
#include <mutex>
#include <type_traits>
#include <unordered_map>

#include "cutlass/arch/arch.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/default_gemm_universal_with_visitor.h"
#include "cutlass/gemm/device/gemm_universal_with_broadcast.h"
#include "cutlass/epilogue/threadblock/fusion/visitors.hpp"

#ifndef INT8MMA_STAGES
#define INT8MMA_STAGES 3
#endif

// Ada sm_89 tuning: 64x128x64 trades one M tile for 2x lower accumulator
// pressure and allows two 3-stage CTAs to reside on one SM.
static constexpr int BM = 64;
static constexpr int BN = 128;
static constexpr int BK = 64;
static constexpr int STAGES = INT8MMA_STAGES;
static constexpr int WARPS = 8;
static constexpr int THREADS = WARPS * 32;
static constexpr int WARP_M = 32;
static constexpr int WARP_N = 32;
static constexpr int WARP_COLS = BN / WARP_N;
static constexpr int MMA_M = 16;
static constexpr int MMA_N = 8;
static constexpr int MMA_K = 32;
static constexpr int WARP_TM = WARP_M / MMA_M;
static constexpr int WARP_TN = WARP_N / MMA_N;
static constexpr int WARP_TK = BK / MMA_K;
static constexpr int A_LD = 64;
// B smem for prepack path: transposed [BN][B_LD_K], K contiguous
static constexpr int B_LD_K = 64;
// B smem for math path: [BK][B_LD_N]
static constexpr int B_LD_N = 144;
static constexpr int BYTES_A = BM * A_LD;
static constexpr int BYTES_B_PRE = BN * B_LD_K;
static constexpr int BYTES_B_MATH = BK * B_LD_N;
static constexpr int BYTES_STAGE_PRE = BYTES_A + BYTES_B_PRE;
static constexpr int BYTES_STAGE_MATH = BYTES_A + BYTES_B_MATH;
static constexpr int BYTES_TOTAL_PRE = BYTES_STAGE_PRE * STAGES;
static constexpr int BYTES_TOTAL_MATH = BYTES_STAGE_MATH * STAGES;

__device__ __forceinline__ void cp_async_cg_16(void* dst_smem, const void* src_gmem) {
  unsigned smem_addr = static_cast<unsigned>(__cvta_generic_to_shared(dst_smem));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(smem_addr), "l"(src_gmem));
}
__device__ __forceinline__ void cp_async_commit_group() {
  asm volatile("cp.async.commit_group;\n" ::);
}
template <int N>
__device__ __forceinline__ void cp_async_wait_group() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}
__device__ __forceinline__ void mma_s8s8s32_m16n8k32(int& d0, int& d1, int& d2, int& d3, uint32_t a0,
                                                     uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                                     uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ int smem_offset_a(int row, int col) {
  return row * A_LD + (col ^ ((row & 3) << 4));
}
__device__ __forceinline__ int smem_offset_b_math(int row, int col) {
  return row * B_LD_N + (col ^ ((row & 7) << 4));
}
__device__ __forceinline__ int smem_offset_b_pre(int n, int k) {
  return n * B_LD_K + k;
}

__device__ __forceinline__ void g2s_A(int8_t* smem, const int8_t* A, int M, int K, int row0, int col0,
                                      int tid) {
  constexpr int VEC = 16;
  constexpr int NVEC = (BM * BK) / VEC;
#pragma unroll
  for (int i = tid; i < NVEC; i += THREADS) {
    int e = i * VEC;
    int r = e / BK;
    int c = e % BK;
    int gr = row0 + r;
    int gc = col0 + c;
    int off = smem_offset_a(r, c);
    const int8_t* gptr = A + static_cast<int64_t>(gr) * K + gc;
    bool in_full = (gr < M) && (gc + VEC <= K);
    bool aligned = (reinterpret_cast<uintptr_t>(gptr) % 16u) == 0u;
    if (in_full && aligned) {
      cp_async_cg_16(smem + off, gptr);
    } else {
      int8_t tmp[VEC] = {};
      if (gr < M) {
#pragma unroll
        for (int v = 0; v < VEC; ++v) {
          if (gc + v < K) tmp[v] = gptr[v];
        }
      }
      *reinterpret_cast<uint4*>(smem + off) = *reinterpret_cast<const uint4*>(tmp);
    }
  }
}

// Math layout B[K,N]
__device__ __forceinline__ void g2s_B_math(int8_t* smem, const int8_t* B, int N, int K, int col0,
                                           int row0, int tid) {
  constexpr int VEC = 16;
  constexpr int NVEC = (BK * BN) / VEC;
#pragma unroll
  for (int i = tid; i < NVEC; i += THREADS) {
    int e = i * VEC;
    int r = e / BN;
    int c = e % BN;
    int gr = row0 + r;
    int gc = col0 + c;
    int off = smem_offset_b_math(r, c);
    const int8_t* gptr = B + static_cast<int64_t>(gr) * N + gc;
    bool in_full = (gr < K) && (gc + VEC <= N);
    bool aligned = (reinterpret_cast<uintptr_t>(gptr) % 16u) == 0u;
    if (in_full && aligned) {
      cp_async_cg_16(smem + off, gptr);
    } else {
      int8_t tmp[VEC] = {};
      if (gr < K) {
#pragma unroll
        for (int v = 0; v < VEC; ++v) {
          if (gc + v < N) tmp[v] = gptr[v];
        }
      }
      *reinterpret_cast<uint4*>(smem + off) = *reinterpret_cast<const uint4*>(tmp);
    }
  }
}

// Prepacked B[N,K]: coalesced vector load along K into smem [n][k]
__device__ __forceinline__ void g2s_B_pre(int8_t* smem, const int8_t* B_nk, int N, int K, int col0,
                                          int row0, int tid) {
  constexpr int VEC = 16;
  constexpr int NVEC = (BN * BK) / VEC;
#pragma unroll
  for (int i = tid; i < NVEC; i += THREADS) {
    int e = i * VEC;
    int n = e / BK;
    int k = e % BK;
    int gn = col0 + n;
    int gk = row0 + k;
    int off = smem_offset_b_pre(n, k);
    const int8_t* gptr = B_nk + static_cast<int64_t>(gn) * K + gk;
    bool in_full = (gn < N) && (gk + VEC <= K);
    bool aligned = (reinterpret_cast<uintptr_t>(gptr) % 16u) == 0u;
    if (in_full && aligned) {
      cp_async_cg_16(smem + off, gptr);
    } else {
      int8_t tmp[VEC] = {};
      if (gn < N) {
#pragma unroll
        for (int v = 0; v < VEC; ++v) {
          if (gk + v < K) tmp[v] = gptr[v];
        }
      }
      *reinterpret_cast<uint4*>(smem + off) = *reinterpret_cast<const uint4*>(tmp);
    }
  }
}

__device__ __forceinline__ void load_a_frag(uint32_t& a0, uint32_t& a1, uint32_t& a2, uint32_t& a3,
                                            const int8_t* As, int row_base, int k_base, int lane) {
  int group = lane >> 2;
  int thr = lane & 3;
  int c = thr * 4;
  auto at = [&](int rr, int cc) -> uint32_t {
    return *reinterpret_cast<const uint32_t*>(As + smem_offset_a(row_base + rr, k_base + cc));
  };
  a0 = at(group, c);
  a1 = at(group + 8, c);
  a2 = at(group, c + 16);
  a3 = at(group + 8, c + 16);
}

__device__ __forceinline__ void load_b_frag_math(uint32_t& b0, uint32_t& b1, const int8_t* Bs,
                                                 int k_base, int n_base, int lane) {
  int group = lane >> 2;
  int thr = lane & 3;
  auto pack4 = [&](int k, int n) -> uint32_t {
    uint32_t v = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      uint8_t byte = static_cast<uint8_t>(Bs[smem_offset_b_math(k_base + k + i, n_base + n)]);
      v |= static_cast<uint32_t>(byte) << (8 * i);
    }
    return v;
  };
  b0 = pack4(thr * 4, group);
  b1 = pack4(thr * 4 + 16, group);
}

__device__ __forceinline__ void load_b_frag_pre(uint32_t& b0, uint32_t& b1, const int8_t* Bs,
                                                int k_base, int n_base, int lane) {
  int group = lane >> 2;
  int thr = lane & 3;
  int n = n_base + group;
  b0 = *reinterpret_cast<const uint32_t*>(Bs + smem_offset_b_pre(n, k_base + thr * 4));
  b1 = *reinterpret_cast<const uint32_t*>(Bs + smem_offset_b_pre(n, k_base + thr * 4 + 16));
}

template <bool PrepackedB>
__global__ void __launch_bounds__(THREADS, 2) int8mma_kernel_t(
    const int8_t* __restrict__ A, const int8_t* __restrict__ B, half* __restrict__ C, float sa,
    const float* __restrict__ sw, int M, int N, int K) {
  int bx = blockIdx.x;
  int by = blockIdx.y;
  if ((by & 1) != 0) bx = gridDim.x - 1 - bx;
  const int tile_m = by * BM;
  const int tile_n = bx * BN;
  if (tile_m >= M || tile_n >= N) return;

  extern __shared__ __align__(16) int8_t smem_base[];
  constexpr int STAGE = PrepackedB ? BYTES_STAGE_PRE : BYTES_STAGE_MATH;
  constexpr int BYTES_A_LOC = BYTES_A;
  int8_t* stage_A[STAGES];
  int8_t* stage_B[STAGES];
#pragma unroll
  for (int s = 0; s < STAGES; ++s) {
    stage_A[s] = smem_base + s * STAGE;
    stage_B[s] = stage_A[s] + BYTES_A_LOC;
  }

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int warp_m = (warp / WARP_COLS) * WARP_M;
  const int warp_n = (warp % WARP_COLS) * WARP_N;

  int acc[WARP_TM][WARP_TN][4];
#pragma unroll
  for (int i = 0; i < WARP_TM; ++i)
#pragma unroll
    for (int j = 0; j < WARP_TN; ++j)
      acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0;

  __shared__ float smem_sw[BN];
  for (int i = tid; i < BN; i += THREADS) {
    int col = tile_n + i;
    smem_sw[i] = (col < N) ? (sa * sw[col]) : 0.f;
  }

  const int k_tiles = (K + BK - 1) / BK;
  int write_stage = 0;
#pragma unroll
  for (int s = 0; s < STAGES - 1; ++s) {
    if (s < k_tiles) {
      g2s_A(stage_A[write_stage], A, M, K, tile_m, s * BK, tid);
      if constexpr (PrepackedB) {
        g2s_B_pre(stage_B[write_stage], B, N, K, tile_n, s * BK, tid);
      } else {
        g2s_B_math(stage_B[write_stage], B, N, K, tile_n, s * BK, tid);
      }
    }
    cp_async_commit_group();
    write_stage = (write_stage + 1) % STAGES;
  }

  int read_stage = 0;
  for (int kt = 0; kt < k_tiles; ++kt) {
    int kt_prefetch = kt + (STAGES - 1);
    if (kt_prefetch < k_tiles) {
      g2s_A(stage_A[write_stage], A, M, K, tile_m, kt_prefetch * BK, tid);
      if constexpr (PrepackedB) {
        g2s_B_pre(stage_B[write_stage], B, N, K, tile_n, kt_prefetch * BK, tid);
      } else {
        g2s_B_math(stage_B[write_stage], B, N, K, tile_n, kt_prefetch * BK, tid);
      }
    }
    cp_async_commit_group();
    write_stage = (write_stage + 1) % STAGES;
    cp_async_wait_group<STAGES - 2>();
    __syncthreads();

    const int8_t* As = stage_A[read_stage];
    const int8_t* Bs = stage_B[read_stage];
#pragma unroll
    for (int kk = 0; kk < WARP_TK; ++kk) {
      uint32_t a_frag[WARP_TM][4];
#pragma unroll
      for (int mi = 0; mi < WARP_TM; ++mi)
        load_a_frag(a_frag[mi][0], a_frag[mi][1], a_frag[mi][2], a_frag[mi][3], As,
                    warp_m + mi * MMA_M, kk * MMA_K, lane);
      uint32_t b_frag[WARP_TN][2];
#pragma unroll
      for (int ni = 0; ni < WARP_TN; ++ni) {
        if constexpr (PrepackedB) {
          load_b_frag_pre(b_frag[ni][0], b_frag[ni][1], Bs, kk * MMA_K, warp_n + ni * MMA_N, lane);
        } else {
          load_b_frag_math(b_frag[ni][0], b_frag[ni][1], Bs, kk * MMA_K, warp_n + ni * MMA_N, lane);
        }
      }
#pragma unroll
      for (int mi = 0; mi < WARP_TM; ++mi)
#pragma unroll
        for (int ni = 0; ni < WARP_TN; ++ni)
          mma_s8s8s32_m16n8k32(acc[mi][ni][0], acc[mi][ni][1], acc[mi][ni][2], acc[mi][ni][3],
                               a_frag[mi][0], a_frag[mi][1], a_frag[mi][2], a_frag[mi][3],
                               b_frag[ni][0], b_frag[ni][1]);
    }
    __syncthreads();
    read_stage = (read_stage + 1) % STAGES;
  }
  cp_async_wait_group<0>();

  int group = lane >> 2;
  int thr = lane & 3;
#pragma unroll
  for (int mi = 0; mi < WARP_TM; ++mi) {
#pragma unroll
    for (int ni = 0; ni < WARP_TN; ++ni) {
      int row = tile_m + warp_m + mi * MMA_M + group;
      int col_local = warp_n + ni * MMA_N + thr * 2;
      int col = tile_n + col_local;
      float s0 = smem_sw[col_local];
      float s1 = smem_sw[col_local + 1];
      half2 h0 = __floats2half2_rn(static_cast<float>(acc[mi][ni][0]) * s0,
                                   static_cast<float>(acc[mi][ni][1]) * s1);
      half2 h1 = __floats2half2_rn(static_cast<float>(acc[mi][ni][2]) * s0,
                                   static_cast<float>(acc[mi][ni][3]) * s1);
      auto store_pair = [&](int r, half2 h) {
        if (r >= M) return;
        half* ptr = &C[static_cast<int64_t>(r) * N + col];
        bool can_vec = (col + 1 < N) && ((col & 1) == 0) &&
                       ((reinterpret_cast<uintptr_t>(ptr) & 3u) == 0u);
        if (can_vec) {
          *reinterpret_cast<half2*>(ptr) = h;
        } else {
          if (col < N) ptr[0] = __low2half(h);
          if (col + 1 < N) ptr[1] = __high2half(h);
        }
      };
      store_pair(row, h0);
      store_pair(row + 8, h1);
    }
  }
}

extern "C" int int8mma_smem_bytes() { return BYTES_TOTAL_MATH; }
extern "C" int int8mma_smem_bytes_prepacked() { return BYTES_TOTAL_PRE; }

extern "C" const char* int8mma_version() {
  return "int8mma-ada-sm89-v13 cutlass-visitor-w8a8-convrot stages=3 tile=64x128x64 warps=8";
}

// CUTLASS owns the SM80+ IMMA mainloop, including ldmatrix fragment loads and
// shared-memory swizzles. This candidate consumes the already cached B[N,K]
// transpose as a column-major KxN matrix and applies a cached [N, 2] float
// vector containing (activation_scale * weight_scale[n], bias[n]).
struct alignas(8) CutlassScaleBias {
  float scale;
  float bias;

  // cutlass::Array clears fragments with T(0), while CUDA's float2 does not
  // provide an int constructor. The layout remains ABI-compatible with one
  // contiguous torch.float32[N, 2] tensor.
  CUTLASS_HOST_DEVICE constexpr CutlassScaleBias(float value = 0.0f)
      : scale(value), bias(value) {}
};

template <typename OutputElement>
class CutlassPerChannelScaleEpilogueT {
 public:
  using ElementOutput = OutputElement;
  using ElementD = ElementOutput;
  using ElementC = ElementOutput;
  using ElementT = ElementOutput;
  using ElementVector = CutlassScaleBias;
  using ElementAccumulator = int32_t;
  using ElementCompute = CutlassScaleBias;
  static constexpr int kElementsPerAccess = 8;
  static constexpr int kCount = kElementsPerAccess;
  static constexpr bool kIsSingleSource = true;
  static constexpr bool kStoreZ = true;
  static constexpr bool kStoreT = false;
  static constexpr bool kIsHeavy = false;

  using FragmentAccumulator = cutlass::Array<ElementAccumulator, kElementsPerAccess>;
  using FragmentCompute = cutlass::Array<ElementCompute, kElementsPerAccess>;
  using FragmentZ = cutlass::Array<ElementOutput, kElementsPerAccess>;
  using FragmentT = cutlass::Array<ElementT, kElementsPerAccess>;

  struct Params {};

  CUTLASS_HOST_DEVICE CutlassPerChannelScaleEpilogueT(Params const&) {}
  CUTLASS_HOST_DEVICE bool is_source_needed() const { return false; }
  CUTLASS_HOST_DEVICE void set_k_partition(int, int) {}

  CUTLASS_HOST_DEVICE void operator()(
      FragmentZ& out, FragmentT&, const FragmentAccumulator& accum,
      const FragmentCompute& output_scale) const {
    cutlass::Array<float, kElementsPerAccess> values;
#pragma unroll
    for (int i = 0; i < kElementsPerAccess; ++i) {
      values[i] = static_cast<float>(accum[i]) * output_scale[i].scale + output_scale[i].bias;
    }
    out = cutlass::NumericArrayConverter<ElementOutput, float, kElementsPerAccess>()(values);
  }

  CUTLASS_HOST_DEVICE void operator()(
      FragmentZ& out, FragmentT& tensor, const FragmentAccumulator& accum,
      const cutlass::Array<ElementOutput, kElementsPerAccess>&,
      const FragmentCompute& output_scale) const {
    (*this)(out, tensor, accum, output_scale);
  }
};

using CutlassPerChannelScaleEpilogue =
    CutlassPerChannelScaleEpilogueT<cutlass::half_t>;
using CutlassPerChannelScaleEpilogueBf16 =
    CutlassPerChannelScaleEpilogueT<cutlass::bfloat16_t>;

using CutlassInt8Gemm128x256 = cutlass::gemm::device::GemmUniversalWithBroadcast<
    int8_t, cutlass::layout::RowMajor,
    int8_t, cutlass::layout::ColumnMajor,
    cutlass::half_t, cutlass::layout::RowMajor,
    int32_t, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 256, 64>,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<16, 8, 32>,
    CutlassPerChannelScaleEpilogue,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    3, 16, 16, cutlass::arch::OpMultiplyAddSaturate>;

using CutlassInt8GemmBf16_128x256 =
    cutlass::gemm::device::GemmUniversalWithBroadcast<
        int8_t,
        cutlass::layout::RowMajor,
        int8_t,
        cutlass::layout::ColumnMajor,
        cutlass::bfloat16_t,
        cutlass::layout::RowMajor,
        int32_t,
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<128, 256, 64>,
        cutlass::gemm::GemmShape<64, 64, 64>,
        cutlass::gemm::GemmShape<16, 8, 32>,
        CutlassPerChannelScaleEpilogueBf16,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        3,
        16,
        16,
        cutlass::arch::OpMultiplyAddSaturate>;

using CutlassInt8Gemm64x128 = cutlass::gemm::device::GemmUniversalWithBroadcast<
    int8_t, cutlass::layout::RowMajor,
    int8_t, cutlass::layout::ColumnMajor,
    cutlass::half_t, cutlass::layout::RowMajor,
    int32_t, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 128, 64>,
    cutlass::gemm::GemmShape<32, 32, 64>,
    cutlass::gemm::GemmShape<16, 8, 32>,
    CutlassPerChannelScaleEpilogue,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    3, 16, 16, cutlass::arch::OpMultiplyAddSaturate>;

using CutlassInt8GemmBf16_64x128 =
    cutlass::gemm::device::GemmUniversalWithBroadcast<
        int8_t,
        cutlass::layout::RowMajor,
        int8_t,
        cutlass::layout::ColumnMajor,
        cutlass::bfloat16_t,
        cutlass::layout::RowMajor,
        int32_t,
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<64, 128, 64>,
        cutlass::gemm::GemmShape<32, 32, 64>,
        cutlass::gemm::GemmShape<16, 8, 32>,
        CutlassPerChannelScaleEpilogueBf16,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        3,
        16,
        16,
        cutlass::arch::OpMultiplyAddSaturate>;

template <
    typename ElementOutput,
    int TBM,
    int TBN,
    int TBK,
    int WM,
    int WN,
    int WK,
    int Stages>
struct FusedInt8Gemm {
  using ElementA = int8_t;
  using ElementB = int8_t;
  using ElementC = ElementOutput;
  using ElementAccumulator = int32_t;
  using ElementCompute = float;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;
  static constexpr int AlignA = 16;
  static constexpr int AlignB = 16;
  static constexpr int AlignC =
      128 / cutlass::sizeof_bits<ElementC>::value;
  static constexpr int EvtStages = 1;

  using ThreadblockShape = cutlass::gemm::GemmShape<TBM, TBN, TBK>;
  using WarpShape = cutlass::gemm::GemmShape<WM, WN, WK>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;
  using ThreadMap = cutlass::epilogue::threadblock::OutputTileThreadLayout<
      ThreadblockShape,
      WarpShape,
      ElementC,
      AlignC,
      EvtStages>;
  using Accumulator = cutlass::epilogue::threadblock::VisitorAccFetch;
  using ActivationScale =
      cutlass::epilogue::threadblock::VisitorColBroadcast<
          ThreadMap,
          ElementCompute,
          cute::Stride<cute::_1, cute::_0, int32_t>>;
  using WeightScale = cutlass::epilogue::threadblock::VisitorRowBroadcast<
      ThreadMap,
      ElementCompute,
      cute::Stride<cute::_0, cute::_1, int32_t>>;
  using Bias = cutlass::epilogue::threadblock::VisitorRowBroadcast<
      ThreadMap,
      ElementCompute,
      cute::Stride<cute::_0, cute::_1, int32_t>>;
  using MultiplyActivation = cutlass::epilogue::threadblock::VisitorCompute<
      cutlass::multiplies,
      ElementCompute,
      ElementCompute,
      cutlass::FloatRoundStyle::round_to_nearest>;
  using EvtActivation = cutlass::epilogue::threadblock::Sm80EVT<
      MultiplyActivation,
      Accumulator,
      ActivationScale>;
  using MultiplyWeight = cutlass::epilogue::threadblock::VisitorCompute<
      cutlass::multiplies,
      ElementCompute,
      ElementCompute,
      cutlass::FloatRoundStyle::round_to_nearest>;
  using EvtWeight = cutlass::epilogue::threadblock::Sm80EVT<
      MultiplyWeight,
      EvtActivation,
      WeightScale>;
  using AddBias = cutlass::epilogue::threadblock::VisitorCompute<
      cutlass::plus,
      ElementOutput,
      ElementCompute,
      cutlass::FloatRoundStyle::round_to_nearest>;
  using EvtBias = cutlass::epilogue::threadblock::Sm80EVT<
      AddBias,
      EvtWeight,
      Bias>;
  using Store = cutlass::epilogue::threadblock::VisitorAuxStore<
      ThreadMap,
      ElementOutput,
      cutlass::FloatRoundStyle::round_to_nearest,
      cute::Stride<int64_t, cute::_1, int64_t>>;
  using EvtStore = cutlass::epilogue::threadblock::Sm80EVT<Store, EvtBias>;
  using Kernel = typename cutlass::gemm::kernel::DefaultGemmWithVisitor<
      ElementA,
      LayoutA,
      cutlass::ComplexTransform::kNone,
      AlignA,
      ElementB,
      LayoutB,
      cutlass::ComplexTransform::kNone,
      AlignB,
      ElementC,
      LayoutC,
      AlignC,
      ElementAccumulator,
      ElementCompute,
      cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm89,
      ThreadblockShape,
      WarpShape,
      InstructionShape,
      EvtStore,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
      Stages,
      cutlass::arch::OpMultiplyAddSaturate,
      EvtStages>::GemmKernel;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;

  static bool run(
      const int8_t* activation,
      const int8_t* weight,
      const float* activation_scales,
      const float* weight_scales,
      const float* bias,
      ElementOutput* output,
      int actual_m,
      int actual_n,
      int problem_m,
      int problem_n,
      int problem_k,
      cudaStream_t stream) {
    const cutlass::gemm::GemmCoord problem(problem_m, problem_n, problem_k);
    typename EvtStore::Arguments epilogue_args{
        {{{{}, {const_cast<float*>(activation_scales), 0.0F,
                {cute::_1{}, cute::_0{}, problem_m}},
           {}},
          {const_cast<float*>(weight_scales), 0.0F,
           {cute::_0{}, cute::_1{}, problem_n}},
          {}},
         {const_cast<float*>(bias), 0.0F,
          {cute::_0{}, cute::_1{}, problem_n}},
         {}},
        {output,
         {actual_n, cute::_1{}, static_cast<int64_t>(problem_m) * actual_n}}};
    typename Gemm::Arguments arguments(
        cutlass::gemm::GemmUniversalMode::kGemm,
        problem,
        1,
        epilogue_args,
        activation,
        weight,
        nullptr,
        nullptr,
        static_cast<int64_t>(problem_m) * problem_k,
        static_cast<int64_t>(problem_n) * problem_k,
        0,
        0,
        problem_k,
        problem_k,
        0,
        0);
    struct State {
      Gemm gemm;
      bool initialized = false;
    };
    static thread_local std::unordered_map<
        uint64_t,
        std::unique_ptr<State>>
        cache;
    const uint64_t key =
        (static_cast<uint64_t>(static_cast<uint32_t>(problem_m)) << 42) |
        (static_cast<uint64_t>(static_cast<uint32_t>(problem_n)) << 21) |
        static_cast<uint64_t>(static_cast<uint32_t>(problem_k));
    auto& state = cache[key];
    if (!state) {
      state = std::make_unique<State>();
    }

    cutlass::Status status = cutlass::Status::kSuccess;
    if (!state->initialized) {
      status = state->gemm.can_implement(arguments);
      if (status != cutlass::Status::kSuccess) {
        return false;
      }
      if (Gemm::get_workspace_size(arguments) != 0) {
        return false;
      }
      status = state->gemm.initialize(arguments, nullptr, stream);
      if (status != cutlass::Status::kSuccess) {
        return false;
      }
      state->initialized = true;
    } else {
      status = state->gemm.update(arguments);
      if (status != cutlass::Status::kSuccess) {
        return false;
      }
    }
    return state->gemm(stream) == cutlass::Status::kSuccess;
  }
};

template <
    typename ElementOutput,
    int TBM,
    int TBN,
    int TBK,
    int WM,
    int WN,
    int WK,
    int Stages>
int run_cutlass_visitor_w8a8(
    const void* activation,
    const void* weight,
    void* output,
    const void* activation_scales,
    const void* weight_scales,
    const void* bias,
    int actual_m,
    int actual_n,
    int problem_m,
    int problem_n,
    int problem_k,
    cudaStream_t stream) {
  if (!activation || !weight || !output || !activation_scales ||
      !weight_scales || !bias || actual_m <= 0 || actual_n <= 0 ||
      problem_m < actual_m || problem_n < actual_n || problem_k <= 0) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const bool ok = FusedInt8Gemm<
      ElementOutput,
      TBM,
      TBN,
      TBK,
      WM,
      WN,
      WK,
      Stages>::run(
      static_cast<const int8_t*>(activation),
      static_cast<const int8_t*>(weight),
      static_cast<const float*>(activation_scales),
      static_cast<const float*>(weight_scales),
      static_cast<const float*>(bias),
      static_cast<ElementOutput*>(output),
      actual_m,
      actual_n,
      problem_m,
      problem_n,
      problem_k,
      stream);
  return static_cast<int>(ok ? cudaSuccess : cudaErrorNotSupported);
}

using CutlassInt8GemmI32_64x128 = cutlass::gemm::device::Gemm<
    int8_t, cutlass::layout::RowMajor,
    int8_t, cutlass::layout::ColumnMajor,
    int32_t, cutlass::layout::RowMajor,
    int32_t, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 128, 64>,
    cutlass::gemm::GemmShape<32, 32, 64>,
    cutlass::gemm::GemmShape<16, 8, 32>,
    cutlass::epilogue::thread::LinearCombination<int32_t, 8, int32_t, int32_t>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    3, 16, 16>;

using CutlassInt8GemmI32_128x256 = cutlass::gemm::device::Gemm<
    int8_t, cutlass::layout::RowMajor,
    int8_t, cutlass::layout::ColumnMajor,
    int32_t, cutlass::layout::RowMajor,
    int32_t, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 256, 64>,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<16, 8, 32>,
    cutlass::epilogue::thread::LinearCombination<int32_t, 8, int32_t, int32_t>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    3, 16, 16>;

template <typename Gemm>
struct CutlassI32GemmState {
  Gemm gemm;
  bool initialized = false;
};

inline uint64_t cutlass_i32_shape_key(int M, int N, int K) {
  return (static_cast<uint64_t>(static_cast<uint32_t>(M)) << 42) |
         (static_cast<uint64_t>(static_cast<uint32_t>(N)) << 21) |
         static_cast<uint64_t>(static_cast<uint32_t>(K));
}

template <typename Gemm>
int run_cutlass_prepacked_b(
    const void* a, const void* b_nk, void* c, const void* output_scale, int M, int N, int K) {
  const auto* a_ptr = static_cast<const int8_t*>(a);
  const auto* b_ptr = static_cast<const int8_t*>(b_nk);
  auto* c_ptr = static_cast<cutlass::half_t*>(c);
  Gemm gemm;
  typename Gemm::Arguments args(
      cutlass::gemm::GemmUniversalMode::kGemm,
      {M, N, K}, 1,
      CutlassPerChannelScaleEpilogue::Params{},
      a_ptr, b_ptr, c_ptr, c_ptr,
      const_cast<void*>(output_scale), nullptr,
      0, 0, 0, 0, 0, 0,
      K, K, N, N, 0, 0);
  cutlass::Status status = gemm.can_implement(args);
  if (status != cutlass::Status::kSuccess) return static_cast<int>(cudaErrorNotSupported);
  status = gemm.initialize(args);
  if (status != cutlass::Status::kSuccess) return static_cast<int>(cudaErrorInvalidValue);
  status = gemm();
  return status == cutlass::Status::kSuccess ? static_cast<int>(cudaSuccess)
                                              : static_cast<int>(cudaErrorLaunchFailure);
}

template <typename Gemm, typename OutputElement, typename Epilogue>
int run_cutlass_prepacked_b_stream(
    const void* a,
    const void* b_nk,
    void* c,
    const void* output_scale,
    int M,
    int N,
    int K,
    cudaStream_t stream) {
  const auto* a_ptr = static_cast<const int8_t*>(a);
  const auto* b_ptr = static_cast<const int8_t*>(b_nk);
  auto* c_ptr = static_cast<OutputElement*>(c);
  typename Gemm::Arguments args(
      cutlass::gemm::GemmUniversalMode::kGemm,
      {M, N, K}, 1,
      typename Epilogue::Params{},
      a_ptr, b_ptr, c_ptr, c_ptr,
      const_cast<void*>(output_scale), nullptr,
      0, 0, 0, 0, 0, 0,
      K, K, N, N, 0, 0);

  static std::mutex cache_mutex;
  static std::unordered_map<
      uint64_t,
      std::unique_ptr<CutlassI32GemmState<Gemm>>>
      cache;
  std::lock_guard<std::mutex> lock(cache_mutex);
  auto& state = cache[cutlass_i32_shape_key(M, N, K)];
  if (!state) {
    state = std::make_unique<CutlassI32GemmState<Gemm>>();
  }

  cutlass::Status status = cutlass::Status::kSuccess;
  if (!state->initialized) {
    status = state->gemm.can_implement(args);
    if (status != cutlass::Status::kSuccess) {
      return static_cast<int>(cudaErrorNotSupported);
    }
    status = state->gemm.initialize(args, nullptr, stream);
    if (status != cutlass::Status::kSuccess) {
      return static_cast<int>(cudaErrorInvalidValue);
    }
    state->initialized = true;
  } else {
    status = state->gemm.update(args);
    if (status != cutlass::Status::kSuccess) {
      return static_cast<int>(cudaErrorInvalidValue);
    }
  }
  status = state->gemm(stream);
  return status == cutlass::Status::kSuccess
             ? static_cast<int>(cudaSuccess)
             : static_cast<int>(cudaErrorLaunchFailure);
}

extern "C" int int8mma_run_cutlass_prepacked_b(
    const void* a, const void* b_nk, void* c, const void* output_scale, int M, int N, int K) {
  if (!a || !b_nk || !c || !output_scale || M <= 0 || N <= 0 || K <= 0) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  if ((N % CutlassPerChannelScaleEpilogue::kElementsPerAccess) != 0 || (K % 32) != 0) {
    return static_cast<int>(cudaErrorNotSupported);
  }
  return run_cutlass_prepacked_b<CutlassInt8Gemm128x256>(a, b_nk, c, output_scale, M, N, K);
}

extern "C" int int8mma_run_cutlass_64x128_prepacked_b(
    const void* a, const void* b_nk, void* c, const void* output_scale, int M, int N, int K) {
  if (!a || !b_nk || !c || !output_scale || M <= 0 || N <= 0 || K <= 0) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  if ((N % CutlassPerChannelScaleEpilogue::kElementsPerAccess) != 0 || (K % 32) != 0) {
    return static_cast<int>(cudaErrorNotSupported);
  }
  return run_cutlass_prepacked_b<CutlassInt8Gemm64x128>(a, b_nk, c, output_scale, M, N, K);
}

extern "C" int int8mma_run_cutlass_scale_64x128_prepacked_b_stream(
    const void* a,
    const void* b_nk,
    void* c,
    const void* output_scale,
    int M,
    int N,
    int K,
    void* stream_ptr) {
  if (!a || !b_nk || !c || !output_scale || M <= 0 || N <= 0 || K <= 0) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  if ((N % CutlassPerChannelScaleEpilogue::kElementsPerAccess) != 0 ||
      (K % 32) != 0) {
    return static_cast<int>(cudaErrorNotSupported);
  }
  return run_cutlass_prepacked_b_stream<
      CutlassInt8Gemm64x128,
      cutlass::half_t,
      CutlassPerChannelScaleEpilogue>(
      a,
      b_nk,
      c,
      output_scale,
      M,
      N,
      K,
      reinterpret_cast<cudaStream_t>(stream_ptr));
}

extern "C" int int8mma_run_cutlass_scale_128x256_prepacked_b_stream(
    const void* a,
    const void* b_nk,
    void* c,
    const void* output_scale,
    int M,
    int N,
    int K,
    void* stream_ptr) {
  if (!a || !b_nk || !c || !output_scale || M <= 0 || N <= 0 || K <= 0) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  if ((N % CutlassPerChannelScaleEpilogue::kElementsPerAccess) != 0 ||
      (K % 32) != 0) {
    return static_cast<int>(cudaErrorNotSupported);
  }
  return run_cutlass_prepacked_b_stream<
      CutlassInt8Gemm128x256,
      cutlass::half_t,
      CutlassPerChannelScaleEpilogue>(
      a,
      b_nk,
      c,
      output_scale,
      M,
      N,
      K,
      reinterpret_cast<cudaStream_t>(stream_ptr));
}

extern "C" int int8mma_run_cutlass_scale_bf16_64x128_prepacked_b_stream(
    const void* a,
    const void* b_nk,
    void* c,
    const void* output_scale,
    int M,
    int N,
    int K,
    void* stream_ptr) {
  if (!a || !b_nk || !c || !output_scale || M <= 0 || N <= 0 || K <= 0) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  if ((N % CutlassPerChannelScaleEpilogueBf16::kElementsPerAccess) != 0 ||
      (K % 32) != 0) {
    return static_cast<int>(cudaErrorNotSupported);
  }
  return run_cutlass_prepacked_b_stream<
      CutlassInt8GemmBf16_64x128,
      cutlass::bfloat16_t,
      CutlassPerChannelScaleEpilogueBf16>(
      a,
      b_nk,
      c,
      output_scale,
      M,
      N,
      K,
      reinterpret_cast<cudaStream_t>(stream_ptr));
}

extern "C" int int8mma_run_cutlass_scale_bf16_128x256_prepacked_b_stream(
    const void* a,
    const void* b_nk,
    void* c,
    const void* output_scale,
    int M,
    int N,
    int K,
    void* stream_ptr) {
  if (!a || !b_nk || !c || !output_scale || M <= 0 || N <= 0 || K <= 0) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  if ((N % CutlassPerChannelScaleEpilogueBf16::kElementsPerAccess) != 0 ||
      (K % 32) != 0) {
    return static_cast<int>(cudaErrorNotSupported);
  }
  return run_cutlass_prepacked_b_stream<
      CutlassInt8GemmBf16_128x256,
      cutlass::bfloat16_t,
      CutlassPerChannelScaleEpilogueBf16>(
      a,
      b_nk,
      c,
      output_scale,
      M,
      N,
      K,
      reinterpret_cast<cudaStream_t>(stream_ptr));
}

extern "C" int int8mma_run_cutlass_visitor_bf16_64x128_prepacked_b_stream(
    const void* a,
    const void* b_nk,
    void* output,
    const void* activation_scale,
    const void* weight_scale,
    const void* bias,
    int actual_m,
    int actual_n,
    int problem_m,
    int problem_k,
    void* stream_ptr) {
  if (actual_n % 8 != 0 || problem_k % 32 != 0) {
    return static_cast<int>(cudaErrorNotSupported);
  }
  return run_cutlass_visitor_w8a8<
      cutlass::bfloat16_t,
      64,
      128,
      64,
      32,
      32,
      64,
      3>(
      a,
      b_nk,
      output,
      activation_scale,
      weight_scale,
      bias,
      actual_m,
      actual_n,
      problem_m,
      actual_n,
      problem_k,
      reinterpret_cast<cudaStream_t>(stream_ptr));
}

extern "C" int int8mma_run_cutlass_visitor_bf16_128x256_prepacked_b_stream(
    const void* a,
    const void* b_nk,
    void* output,
    const void* activation_scale,
    const void* weight_scale,
    const void* bias,
    int actual_m,
    int actual_n,
    int problem_m,
    int problem_k,
    void* stream_ptr) {
  if (actual_n % 8 != 0 || problem_k % 32 != 0) {
    return static_cast<int>(cudaErrorNotSupported);
  }
  return run_cutlass_visitor_w8a8<
      cutlass::bfloat16_t,
      128,
      256,
      64,
      64,
      64,
      64,
      3>(
      a,
      b_nk,
      output,
      activation_scale,
      weight_scale,
      bias,
      actual_m,
      actual_n,
      problem_m,
      actual_n,
      problem_k,
      reinterpret_cast<cudaStream_t>(stream_ptr));
}

extern "C" int int8mma_run_cutlass_visitor_half_64x128_prepacked_b_stream(
    const void* a,
    const void* b_nk,
    void* output,
    const void* activation_scale,
    const void* weight_scale,
    const void* bias,
    int actual_m,
    int actual_n,
    int problem_m,
    int problem_k,
    void* stream_ptr) {
  if (actual_n % 8 != 0 || problem_k % 32 != 0) {
    return static_cast<int>(cudaErrorNotSupported);
  }
  return run_cutlass_visitor_w8a8<
      cutlass::half_t,
      64,
      128,
      64,
      32,
      32,
      64,
      3>(
      a,
      b_nk,
      output,
      activation_scale,
      weight_scale,
      bias,
      actual_m,
      actual_n,
      problem_m,
      actual_n,
      problem_k,
      reinterpret_cast<cudaStream_t>(stream_ptr));
}

extern "C" int int8mma_run_cutlass_visitor_half_128x256_prepacked_b_stream(
    const void* a,
    const void* b_nk,
    void* output,
    const void* activation_scale,
    const void* weight_scale,
    const void* bias,
    int actual_m,
    int actual_n,
    int problem_m,
    int problem_k,
    void* stream_ptr) {
  if (actual_n % 8 != 0 || problem_k % 32 != 0) {
    return static_cast<int>(cudaErrorNotSupported);
  }
  return run_cutlass_visitor_w8a8<
      cutlass::half_t,
      128,
      256,
      64,
      64,
      64,
      64,
      3>(
      a,
      b_nk,
      output,
      activation_scale,
      weight_scale,
      bias,
      actual_m,
      actual_n,
      problem_m,
      actual_n,
      problem_k,
      reinterpret_cast<cudaStream_t>(stream_ptr));
}

template <typename Gemm>
int run_cutlass_i32_prepacked_b(
    const void* a,
    const void* b_nk,
    void* c,
    int M,
    int N,
    int K,
    cudaStream_t stream) {
  const auto* a_ptr = static_cast<const int8_t*>(a);
  const auto* b_ptr = static_cast<const int8_t*>(b_nk);
  auto* c_ptr = static_cast<int32_t*>(c);
  typename Gemm::Arguments args(
      {M, N, K},
      {a_ptr, K},
      {b_ptr, K},
      {c_ptr, N},
      {c_ptr, N},
      typename Gemm::EpilogueOutputOp::Params(1, 0),
      1);

  static std::mutex cache_mutex;
  static std::unordered_map<
      uint64_t,
      std::unique_ptr<CutlassI32GemmState<Gemm>>>
      cache;
  std::lock_guard<std::mutex> lock(cache_mutex);
  auto& state = cache[cutlass_i32_shape_key(M, N, K)];
  if (!state) {
    state = std::make_unique<CutlassI32GemmState<Gemm>>();
  }

  cutlass::Status status = cutlass::Status::kSuccess;
  if (!state->initialized) {
    status = state->gemm.can_implement(args);
    if (status != cutlass::Status::kSuccess) {
      return static_cast<int>(cudaErrorNotSupported);
    }
    status = state->gemm.initialize(args, nullptr, stream);
    if (status != cutlass::Status::kSuccess) {
      return static_cast<int>(cudaErrorInvalidValue);
    }
    state->initialized = true;
  } else {
    status = state->gemm.update(args);
    if (status != cutlass::Status::kSuccess) {
      return static_cast<int>(cudaErrorInvalidValue);
    }
  }
  status = state->gemm(stream);
  return status == cutlass::Status::kSuccess
             ? static_cast<int>(cudaSuccess)
             : static_cast<int>(cudaErrorLaunchFailure);
}

extern "C" int int8mma_run_cutlass_i32_64x128_prepacked_b(
    const void* a,
    const void* b_nk,
    void* c,
    int M,
    int N,
    int K,
    void* stream_ptr) {
  if (!a || !b_nk || !c || M <= 0 || N <= 0 || K <= 0) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  if (N % 8 != 0 || K % 32 != 0) {
    return static_cast<int>(cudaErrorNotSupported);
  }
  return run_cutlass_i32_prepacked_b<CutlassInt8GemmI32_64x128>(
      a,
      b_nk,
      c,
      M,
      N,
      K,
      reinterpret_cast<cudaStream_t>(stream_ptr));
}

extern "C" int int8mma_run_cutlass_i32_128x256_prepacked_b(
    const void* a,
    const void* b_nk,
    void* c,
    int M,
    int N,
    int K,
    void* stream_ptr) {
  if (!a || !b_nk || !c || M <= 0 || N <= 0 || K <= 0) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  if (N % 8 != 0 || K % 32 != 0) {
    return static_cast<int>(cudaErrorNotSupported);
  }
  return run_cutlass_i32_prepacked_b<CutlassInt8GemmI32_128x256>(
      a,
      b_nk,
      c,
      M,
      N,
      K,
      reinterpret_cast<cudaStream_t>(stream_ptr));
}

template <typename Output>
__device__ __forceinline__ Output convrot_convert_output(float value);

template <>
__device__ __forceinline__ half convrot_convert_output<half>(float value) {
  return __float2half_rn(value);
}

template <>
__device__ __forceinline__ __nv_bfloat16 convrot_convert_output<__nv_bfloat16>(
    float value) {
  return __float2bfloat16(value);
}

template <typename Input>
__device__ __forceinline__ float convrot_to_float(Input value);

template <>
__device__ __forceinline__ float convrot_to_float<half>(half value) {
  return __half2float(value);
}

template <>
__device__ __forceinline__ float convrot_to_float<__nv_bfloat16>(
    __nv_bfloat16 value) {
  return __bfloat162float(value);
}

template <typename Input, bool StoreOutput>
__device__ __forceinline__ float convrot_rotate_256_group(
    const Input* input,
    int global_row,
    int logical_k,
    int rotated_k,
    int padded_k,
    int col_base,
    int lane_id,
    int8_t* output,
    float scale) {
  constexpr unsigned FULL_MASK = 0xffffffffU;
  constexpr int VALUES_PER_LANE = 8;
  float values[VALUES_PER_LANE];

#pragma unroll
  for (int index = 0; index < VALUES_PER_LANE; ++index) {
    const int local_col = lane_id * VALUES_PER_LANE + index;
    const int global_col = col_base + local_col;
    values[index] =
        global_col < logical_k
            ? convrot_to_float<Input>(
                  input[static_cast<int64_t>(global_row) * logical_k + global_col])
            : 0.0F;
  }

#pragma unroll
  for (int group = 0; group < VALUES_PER_LANE; group += 4) {
    const float a = values[group];
    const float b = values[group + 1];
    const float c = values[group + 2];
    const float d = values[group + 3];
    const float sum = a + b + c + d;
    values[group] = sum - 2.0F * d;
    values[group + 1] = sum - 2.0F * c;
    values[group + 2] = sum - 2.0F * b;
    values[group + 3] = sum - 2.0F * a;
  }

#pragma unroll
  for (int d0 = 0; d0 < 4; ++d0) {
    const float low = values[d0];
    const float high = values[d0 + 4];
    const float local_sum = low + high;
    const float sum =
        local_sum + __shfl_xor_sync(FULL_MASK, local_sum, 1);
    values[d0] =
        sum - 2.0F * __shfl_xor_sync(FULL_MASK, high, 1);
    values[d0 + 4] =
        sum - 2.0F * __shfl_xor_sync(FULL_MASK, low, 1);
  }

#pragma unroll
  for (int index = 0; index < VALUES_PER_LANE; ++index) {
    const float value = values[index];
    float sum = value + __shfl_xor_sync(FULL_MASK, value, 2);
    sum += __shfl_xor_sync(FULL_MASK, sum, 4);
    values[index] =
        sum - 2.0F * __shfl_xor_sync(FULL_MASK, value, 6);
  }

#pragma unroll
  for (int index = 0; index < VALUES_PER_LANE; ++index) {
    const float value = values[index];
    float sum = value + __shfl_xor_sync(FULL_MASK, value, 8);
    sum += __shfl_xor_sync(FULL_MASK, sum, 16);
    values[index] =
        sum - 2.0F * __shfl_xor_sync(FULL_MASK, value, 24);
  }

  float local_maximum = 0.0F;
#pragma unroll
  for (int index = 0; index < VALUES_PER_LANE; ++index) {
    const int local_col = lane_id * VALUES_PER_LANE + index;
    const int global_col = col_base + local_col;
    const float rotated =
        global_col < rotated_k ? values[index] * (1.0F / 16.0F) : 0.0F;
    const Input rounded = convrot_convert_output<Input>(rotated);
    const float rounded_value = convrot_to_float<Input>(rounded);
    local_maximum = fmaxf(local_maximum, fabsf(rounded_value));
    if constexpr (StoreOutput) {
      const int quantized =
          scale > 0.0F
              ? __float2int_rn(rounded_value / scale)
              : 0;
      int clamped = quantized < -127 ? -127 : quantized;
      clamped = clamped > 127 ? 127 : clamped;
      if (global_col < padded_k) {
        output[static_cast<int64_t>(global_row) * padded_k + global_col] =
            static_cast<int8_t>(clamped);
      }
    }
  }

#pragma unroll
  for (int mask = 16; mask > 0; mask /= 2) {
    local_maximum = fmaxf(
        local_maximum,
        __shfl_xor_sync(FULL_MASK, local_maximum, mask));
  }
  return local_maximum;
}

template <typename Input>
__global__ void convrot_quantize_rows_kernel(
    const Input* __restrict__ input,
    int8_t* __restrict__ output,
    float* __restrict__ scales,
    int actual_m,
    int logical_k,
    int rotated_k,
    int padded_m,
    int padded_k) {
  const int warp_id = static_cast<int>(threadIdx.x) / 32;
  const int lane_id = static_cast<int>(threadIdx.x) % 32;
  const int global_row =
      static_cast<int>(blockIdx.x) * (static_cast<int>(blockDim.x) / 32) +
      warp_id;
  if (global_row >= padded_m) {
    return;
  }

  float maximum = 0.0F;
  for (int col_base = 0; col_base < rotated_k; col_base += 256) {
    maximum = fmaxf(
        maximum,
        convrot_rotate_256_group<Input, false>(
            input,
            global_row < actual_m ? global_row : 0,
            global_row < actual_m ? logical_k : 0,
            rotated_k,
            padded_k,
            col_base,
            lane_id,
            nullptr,
            1.0F));
  }
  const float scale = maximum > 0.0F ? maximum / 127.0F : 1.0F;
  if (lane_id == 0) {
    scales[global_row] = scale;
  }

  for (int col_base = 0; col_base < padded_k; col_base += 256) {
    convrot_rotate_256_group<Input, true>(
        input,
        global_row < actual_m ? global_row : 0,
        global_row < actual_m ? logical_k : 0,
        rotated_k,
        padded_k,
        col_base,
        lane_id,
        output,
        scale);
  }
}

extern "C" int int8mma_convrot_quantize_rows(
    const void* input,
    void* output,
    void* scales,
    int actual_m,
    int logical_k,
    int rotated_k,
    int padded_m,
    int padded_k,
    int input_kind,
    void* stream_ptr) {
  if (!input || !output || !scales || actual_m <= 0 || logical_k <= 0 ||
      rotated_k <= 0 || padded_m < actual_m || padded_k < rotated_k ||
      logical_k > rotated_k || rotated_k % 256 != 0 ||
      padded_k % 256 != 0) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  constexpr int WARPS = 4;
  const dim3 block(WARPS * 32);
  const dim3 grid((padded_m + WARPS - 1) / WARPS);
  const auto stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  if (input_kind == 0) {
    convrot_quantize_rows_kernel<__nv_bfloat16>
        <<<grid, block, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(input),
            static_cast<int8_t*>(output),
            static_cast<float*>(scales),
            actual_m,
            logical_k,
            rotated_k,
            padded_m,
            padded_k);
  } else if (input_kind == 1) {
    convrot_quantize_rows_kernel<half>
        <<<grid, block, 0, stream>>>(
            static_cast<const half*>(input),
            static_cast<int8_t*>(output),
            static_cast<float*>(scales),
            actual_m,
            logical_k,
            rotated_k,
            padded_m,
            padded_k);
  } else {
    return static_cast<int>(cudaErrorNotSupported);
  }
  return static_cast<int>(cudaGetLastError());
}

template <typename ElementOutput>
int run_cutlass_visitor_convrot_w8a8(
    const void* input,
    void* quantized_activation,
    void* activation_scales,
    const void* prepacked_b,
    void* output,
    const void* weight_scales,
    const void* bias,
    int actual_m,
    int logical_k,
    int rotated_k,
    int padded_m,
    int padded_k,
    int n,
    cudaStream_t stream) {
  constexpr int WARPS = 4;
  const dim3 block(WARPS * 32);
  const dim3 grid((padded_m + WARPS - 1) / WARPS);
  if constexpr (std::is_same<ElementOutput, cutlass::bfloat16_t>::value) {
    convrot_quantize_rows_kernel<__nv_bfloat16>
        <<<grid, block, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(input),
            static_cast<int8_t*>(quantized_activation),
            static_cast<float*>(activation_scales),
            actual_m,
            logical_k,
            rotated_k,
            padded_m,
            padded_k);
  } else {
    convrot_quantize_rows_kernel<half>
        <<<grid, block, 0, stream>>>(
            static_cast<const half*>(input),
            static_cast<int8_t*>(quantized_activation),
            static_cast<float*>(activation_scales),
            actual_m,
            logical_k,
            rotated_k,
            padded_m,
            padded_k);
  }
  const cudaError_t quantize_error = cudaGetLastError();
  if (quantize_error != cudaSuccess) {
    return static_cast<int>(quantize_error);
  }
  // M=256 benefits from the lower-register tile; preserve the legacy tile for
  // other explicit CUTLASS shapes until they have independent measurements.
  if (actual_m == 256 && n >= 512) {
    return run_cutlass_visitor_w8a8<
        ElementOutput,
        64,
        128,
        64,
        32,
        32,
        64,
        3>(
        quantized_activation,
        prepacked_b,
        output,
        activation_scales,
        weight_scales,
        bias,
        actual_m,
        n,
        padded_m,
        n,
        padded_k,
        stream);
  }
  return run_cutlass_visitor_w8a8<
      ElementOutput,
      128,
      256,
      64,
      64,
      64,
      64,
      3>(
      quantized_activation,
      prepacked_b,
      output,
      activation_scales,
      weight_scales,
      bias,
      actual_m,
      n,
      padded_m,
      n,
      padded_k,
      stream);
}

extern "C" int int8mma_run_cutlass_visitor_convrot_bf16_prepacked_b_stream(
    const void* input,
    void* quantized_activation,
    void* activation_scales,
    const void* prepacked_b,
    void* output,
    const void* weight_scales,
    const void* bias,
    int actual_m,
    int logical_k,
    int rotated_k,
    int padded_m,
    int padded_k,
    int n,
    void* stream_ptr) {
  if (!input || !quantized_activation || !activation_scales || !prepacked_b ||
      !output || !weight_scales || !bias || actual_m <= 0 || logical_k <= 0 ||
      rotated_k <= 0 || padded_m < actual_m || padded_k < rotated_k || n <= 0) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  return run_cutlass_visitor_convrot_w8a8<cutlass::bfloat16_t>(
      input,
      quantized_activation,
      activation_scales,
      prepacked_b,
      output,
      weight_scales,
      bias,
      actual_m,
      logical_k,
      rotated_k,
      padded_m,
      padded_k,
      n,
      reinterpret_cast<cudaStream_t>(stream_ptr));
}

extern "C" int int8mma_run_cutlass_visitor_convrot_half_prepacked_b_stream(
    const void* input,
    void* quantized_activation,
    void* activation_scales,
    const void* prepacked_b,
    void* output,
    const void* weight_scales,
    const void* bias,
    int actual_m,
    int logical_k,
    int rotated_k,
    int padded_m,
    int padded_k,
    int n,
    void* stream_ptr) {
  if (!input || !quantized_activation || !activation_scales || !prepacked_b ||
      !output || !weight_scales || !bias || actual_m <= 0 || logical_k <= 0 ||
      rotated_k <= 0 || padded_m < actual_m || padded_k < rotated_k || n <= 0) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  return run_cutlass_visitor_convrot_w8a8<cutlass::half_t>(
      input,
      quantized_activation,
      activation_scales,
      prepacked_b,
      output,
      weight_scales,
      bias,
      actual_m,
      logical_k,
      rotated_k,
      padded_m,
      padded_k,
      n,
      reinterpret_cast<cudaStream_t>(stream_ptr));
}

namespace {

cublasLtHandle_t g_cublaslt_handle = nullptr;
cublasStatus_t g_cublaslt_status = CUBLAS_STATUS_NOT_INITIALIZED;
std::once_flag g_cublaslt_once;

cublasStatus_t cublaslt_handle(cublasLtHandle_t* handle) {
  std::call_once(g_cublaslt_once, []() {
    g_cublaslt_status = cublasLtCreate(&g_cublaslt_handle);
  });
  if (g_cublaslt_status == CUBLAS_STATUS_SUCCESS) {
    *handle = g_cublaslt_handle;
  }
  return g_cublaslt_status;
}

cublasStatus_t set_row_major(cublasLtMatrixLayout_t layout) {
  const cublasLtOrder_t order = CUBLASLT_ORDER_ROW;
  return cublasLtMatrixLayoutSetAttribute(
      layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order));
}

}  // namespace

// CuBLASLt supports the W8A8 Tensor Core product with an int32 output. The
// scale/bias fp16 epilogue is launched separately below because this CUDA
// version does not expose per-column float scaling for an int32 IMMA output.
extern "C" int int8mma_run_cublaslt_i32(
    const void* a, const void* b, void* d, int M, int N, int K,
    void* workspace, size_t workspace_size, void* stream_ptr) {
  if (!a || !b || !d || M <= 0 || N <= 0 || K <= 0) {
    return static_cast<int>(CUBLAS_STATUS_INVALID_VALUE);
  }

  cublasLtHandle_t handle = nullptr;
  cublasLtMatmulDesc_t operation = nullptr;
  cublasLtMatrixLayout_t a_layout = nullptr;
  cublasLtMatrixLayout_t b_layout = nullptr;
  cublasLtMatrixLayout_t c_layout = nullptr;
  cublasLtMatrixLayout_t d_layout = nullptr;
  cublasLtMatmulPreference_t preference = nullptr;
  cublasStatus_t status = cublaslt_handle(&handle);
  if (status != CUBLAS_STATUS_SUCCESS) return static_cast<int>(status);

  do {
    status = cublasLtMatmulDescCreate(&operation, CUBLAS_COMPUTE_32I, CUDA_R_32I);
    if (status != CUBLAS_STATUS_SUCCESS) break;

    status = cublasLtMatrixLayoutCreate(&a_layout, CUDA_R_8I, M, K, K);
    if (status != CUBLAS_STATUS_SUCCESS) break;
    status = set_row_major(a_layout);
    if (status != CUBLAS_STATUS_SUCCESS) break;
    status = cublasLtMatrixLayoutCreate(&b_layout, CUDA_R_8I, K, N, N);
    if (status != CUBLAS_STATUS_SUCCESS) break;
    status = set_row_major(b_layout);
    if (status != CUBLAS_STATUS_SUCCESS) break;
    status = cublasLtMatrixLayoutCreate(&c_layout, CUDA_R_32I, M, N, N);
    if (status != CUBLAS_STATUS_SUCCESS) break;
    status = set_row_major(c_layout);
    if (status != CUBLAS_STATUS_SUCCESS) break;
    status = cublasLtMatrixLayoutCreate(&d_layout, CUDA_R_32I, M, N, N);
    if (status != CUBLAS_STATUS_SUCCESS) break;
    status = set_row_major(d_layout);
    if (status != CUBLAS_STATUS_SUCCESS) break;

    status = cublasLtMatmulPreferenceCreate(&preference);
    if (status != CUBLAS_STATUS_SUCCESS) break;
    status = cublasLtMatmulPreferenceSetAttribute(
        preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &workspace_size, sizeof(workspace_size));
    if (status != CUBLAS_STATUS_SUCCESS) break;

    cublasLtMatmulHeuristicResult_t heuristic{};
    int returned = 0;
    status = cublasLtMatmulAlgoGetHeuristic(
        handle, operation, a_layout, b_layout, c_layout, d_layout, preference,
        1, &heuristic, &returned);
    if (status != CUBLAS_STATUS_SUCCESS || returned == 0 ||
        heuristic.state != CUBLAS_STATUS_SUCCESS) {
      status = CUBLAS_STATUS_NOT_SUPPORTED;
      break;
    }

    const int32_t alpha = 1;
    const int32_t beta = 0;
    status = cublasLtMatmul(
        handle, operation, &alpha, a, a_layout, b, b_layout, &beta, d, c_layout,
        d, d_layout, &heuristic.algo, workspace, workspace_size,
        reinterpret_cast<cudaStream_t>(stream_ptr));
  } while (false);

  if (preference) cublasLtMatmulPreferenceDestroy(preference);
  if (d_layout) cublasLtMatrixLayoutDestroy(d_layout);
  if (c_layout) cublasLtMatrixLayoutDestroy(c_layout);
  if (b_layout) cublasLtMatrixLayoutDestroy(b_layout);
  if (a_layout) cublasLtMatrixLayoutDestroy(a_layout);
  if (operation) cublasLtMatmulDescDestroy(operation);
  return static_cast<int>(status);
}

__global__ void int8mma_dequant_i32_kernel(
    const int32_t* __restrict__ accum, half* __restrict__ output,
    const float* __restrict__ activation_scale,
    const float* __restrict__ weight_scale, const half* __restrict__ bias,
    int64_t total, int N) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= total) return;
  const int col = static_cast<int>(index % N);
  float value = static_cast<float>(accum[index]) * activation_scale[0] * weight_scale[col];
  if (bias != nullptr) value += __half2float(bias[col]);
  output[index] = __float2half_rn(value);
}

template <typename Output>
__global__ void int8mma_dequant_i32_rowwise_kernel(
    const int32_t* __restrict__ accum,
    Output* __restrict__ output,
    const float* __restrict__ activation_scale,
    const float* __restrict__ weight_scale,
    const float* __restrict__ bias,
    int64_t total,
    int N) {
  const int64_t index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= total) {
    return;
  }
  const int row = static_cast<int>(index / N);
  const int col = static_cast<int>(index % N);
  float value =
      static_cast<float>(accum[index]) * activation_scale[row] *
      weight_scale[col];
  if (bias != nullptr) {
    value += bias[col];
  }
  output[index] = convrot_convert_output<Output>(value);
}

template <typename Input, typename Output>
__global__ void int8mma_apply_rowwise_scale_bias_kernel(
    const Input* __restrict__ scaled,
    Output* __restrict__ output,
    const float* __restrict__ activation_scale,
    const float* __restrict__ bias,
    int64_t total,
    int N) {
  const int64_t index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= total) {
    return;
  }
  const int row = static_cast<int>(index / N);
  const int col = static_cast<int>(index % N);
  const float value =
      convrot_to_float<Input>(scaled[index]) * activation_scale[row] +
      (bias == nullptr ? 0.0f : bias[col]);
  output[index] = convrot_convert_output<Output>(value);
}

__global__ void int8mma_apply_bf16_rowwise_scale_bias_vec2_kernel(
    const __nv_bfloat162* __restrict__ scaled,
    __nv_bfloat162* __restrict__ output,
    const float* __restrict__ activation_scale,
    const float* __restrict__ bias,
    int64_t pair_count,
    int pairs_per_row) {
  const int64_t pair_index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (pair_index >= pair_count) {
    return;
  }
  const int row = static_cast<int>(pair_index / pairs_per_row);
  const int pair_col = static_cast<int>(pair_index % pairs_per_row);
  const float scale = activation_scale[row];
  float2 value = __bfloat1622float2(scaled[pair_index]);
  if (bias != nullptr) {
    const float2 bias_pair = reinterpret_cast<const float2*>(bias)[pair_col];
    value.x = value.x * scale + bias_pair.x;
    value.y = value.y * scale + bias_pair.y;
  } else {
    value.x *= scale;
    value.y *= scale;
  }
  output[pair_index] = __float22bfloat162_rn(value);
}

extern "C" int int8mma_dequant_i32(
    const void* accum, void* output, const void* activation_scale, const void* weight_scale,
    const void* bias, int M, int N) {
  if (!accum || !output || !activation_scale || !weight_scale || M <= 0 || N <= 0) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const int64_t total = static_cast<int64_t>(M) * N;
  constexpr int threads = 256;
  const dim3 block(threads);
  const dim3 grid(static_cast<unsigned int>((total + threads - 1) / threads));
  int8mma_dequant_i32_kernel<<<grid, block>>>(
      static_cast<const int32_t*>(accum), static_cast<half*>(output),
      static_cast<const float*>(activation_scale),
      static_cast<const float*>(weight_scale), static_cast<const half*>(bias), total, N);
  return static_cast<int>(cudaGetLastError());
}

extern "C" int int8mma_dequant_i32_rowwise(
    const void* accum,
    void* output,
    const void* activation_scale,
    const void* weight_scale,
    const void* bias,
    int M,
    int N,
    int output_kind,
    void* stream_ptr) {
  if (!accum || !output || !activation_scale || !weight_scale ||
      M <= 0 || N <= 0) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const int64_t total = static_cast<int64_t>(M) * N;
  constexpr int threads = 256;
  const dim3 block(threads);
  const dim3 grid(static_cast<unsigned int>((total + threads - 1) / threads));
  const auto stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  if (output_kind == 0) {
    int8mma_dequant_i32_rowwise_kernel<__nv_bfloat16>
        <<<grid, block, 0, stream>>>(
            static_cast<const int32_t*>(accum),
            static_cast<__nv_bfloat16*>(output),
            static_cast<const float*>(activation_scale),
            static_cast<const float*>(weight_scale),
            static_cast<const float*>(bias),
            total,
            N);
  } else if (output_kind == 1) {
    int8mma_dequant_i32_rowwise_kernel<half>
        <<<grid, block, 0, stream>>>(
            static_cast<const int32_t*>(accum),
            static_cast<half*>(output),
            static_cast<const float*>(activation_scale),
            static_cast<const float*>(weight_scale),
            static_cast<const float*>(bias),
            total,
            N);
  } else {
    return static_cast<int>(cudaErrorNotSupported);
  }
  return static_cast<int>(cudaGetLastError());
}

extern "C" int int8mma_apply_half_rowwise_scale_bias(
    const void* scaled,
    void* output,
    const void* activation_scale,
    const void* bias,
    int M,
    int N,
    int output_kind,
    void* stream_ptr) {
  if (!scaled || !output || !activation_scale || M <= 0 || N <= 0) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const int64_t total = static_cast<int64_t>(M) * N;
  constexpr int threads = 256;
  const dim3 block(threads);
  const dim3 grid(static_cast<unsigned int>((total + threads - 1) / threads));
  const auto stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  if (output_kind == 0) {
    int8mma_apply_rowwise_scale_bias_kernel<half, __nv_bfloat16>
        <<<grid, block, 0, stream>>>(
            static_cast<const half*>(scaled),
            static_cast<__nv_bfloat16*>(output),
            static_cast<const float*>(activation_scale),
            static_cast<const float*>(bias),
            total,
            N);
  } else if (output_kind == 1) {
    int8mma_apply_rowwise_scale_bias_kernel<half, half>
        <<<grid, block, 0, stream>>>(
            static_cast<const half*>(scaled),
            static_cast<half*>(output),
            static_cast<const float*>(activation_scale),
            static_cast<const float*>(bias),
            total,
            N);
  } else {
    return static_cast<int>(cudaErrorNotSupported);
  }
  return static_cast<int>(cudaGetLastError());
}

extern "C" int int8mma_apply_bf16_rowwise_scale_bias(
    const void* scaled,
    void* output,
    const void* activation_scale,
    const void* bias,
    int M,
    int N,
    int output_kind,
    void* stream_ptr) {
  if (!scaled || !output || !activation_scale || M <= 0 || N <= 0) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const int64_t total = static_cast<int64_t>(M) * N;
  constexpr int threads = 256;
  const auto stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  if (output_kind == 0) {
    if ((N & 1) == 0) {
      const int pairs_per_row = N / 2;
      const int64_t pair_count = static_cast<int64_t>(M) * pairs_per_row;
      const dim3 block(threads);
      const dim3 grid(
          static_cast<unsigned int>((pair_count + threads - 1) / threads));
      int8mma_apply_bf16_rowwise_scale_bias_vec2_kernel<<<grid, block, 0, stream>>>(
          static_cast<const __nv_bfloat162*>(scaled),
          static_cast<__nv_bfloat162*>(output),
          static_cast<const float*>(activation_scale),
          static_cast<const float*>(bias),
          pair_count,
          pairs_per_row);
      return static_cast<int>(cudaGetLastError());
    }
    const dim3 block(threads);
    const dim3 grid(static_cast<unsigned int>((total + threads - 1) / threads));
    int8mma_apply_rowwise_scale_bias_kernel<__nv_bfloat16, __nv_bfloat16>
        <<<grid, block, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(scaled),
            static_cast<__nv_bfloat16*>(output),
            static_cast<const float*>(activation_scale),
            static_cast<const float*>(bias),
            total,
            N);
  } else if (output_kind == 1) {
    const dim3 block(threads);
    const dim3 grid(static_cast<unsigned int>((total + threads - 1) / threads));
    int8mma_apply_rowwise_scale_bias_kernel<__nv_bfloat16, half>
        <<<grid, block, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(scaled),
            static_cast<half*>(output),
            static_cast<const float*>(activation_scale),
            static_cast<const float*>(bias),
            total,
            N);
  } else {
    return static_cast<int>(cudaErrorNotSupported);
  }
  return static_cast<int>(cudaGetLastError());
}

static int launch_impl(void const* a, void const* b, void* c, float sa, float const* sw, int M,
                       int N, int K, bool prepacked) {
  if (!a || !b || !c || !sw || M <= 0 || N <= 0 || K <= 0)
    return static_cast<int>(cudaErrorInvalidValue);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
  dim3 block(THREADS);
  int smem = prepacked ? BYTES_TOTAL_PRE : BYTES_TOTAL_MATH;
  if (prepacked) {
    auto err = cudaFuncSetAttribute(int8mma_kernel_t<true>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    if (err != cudaSuccess) return static_cast<int>(err);
    int8mma_kernel_t<true><<<grid, block, smem>>>(
        static_cast<const int8_t*>(a), static_cast<const int8_t*>(b), static_cast<half*>(c), sa, sw,
        M, N, K);
  } else {
    auto err = cudaFuncSetAttribute(int8mma_kernel_t<false>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    if (err != cudaSuccess) return static_cast<int>(err);
    int8mma_kernel_t<false><<<grid, block, smem>>>(
        static_cast<const int8_t*>(a), static_cast<const int8_t*>(b), static_cast<half*>(c), sa, sw,
        M, N, K);
  }
  return static_cast<int>(cudaGetLastError());
}

extern "C" int int8mma_run(void const* a, void const* b, void* c, float sa, float const* sw, int M,
                           int N, int K) {
  return launch_impl(a, b, c, sa, sw, M, N, K, false);
}

extern "C" int int8mma_run_prepacked_b(void const* a, void const* b_nk, void* c, float sa,
                                       float const* sw, int M, int N, int K) {
  return launch_impl(a, b_nk, c, sa, sw, M, N, K, true);
}


// ---------------------------------------------------------------------------
// Fused static activation quant + prepacked-B INT8 MMA
// X is half [M,K]; quantize into A smem as int8 with scale sa during G2S.
// ---------------------------------------------------------------------------
__device__ __forceinline__ int8_t quant_half_to_s8(half v, float inv_sa) {
  float f = __half2float(v) * inv_sa;
  int q = __float2int_rn(f);
  q = q < -127 ? -127 : (q > 127 ? 127 : q);
  return static_cast<int8_t>(q);
}

__device__ __forceinline__ void g2s_A_quant_half(int8_t* smem, const half* X, int M, int K,
                                                 int row0, int col0, int tid, float inv_sa) {
  constexpr int VEC = 8;  // half2 x4 = 16B load when aligned
  constexpr int NVEC = (BM * BK) / VEC;
#pragma unroll
  for (int i = tid; i < NVEC; i += THREADS) {
    int e = i * VEC;
    int r = e / BK;
    int c = e % BK;
    int gr = row0 + r;
    int gc = col0 + c;
    int off = smem_offset_a(r, c);
    int8_t tmp[VEC] = {};
    if (gr < M) {
#pragma unroll
      for (int v = 0; v < VEC; ++v) {
        if (gc + v < K) {
          tmp[v] = quant_half_to_s8(X[static_cast<int64_t>(gr) * K + gc + v], inv_sa);
        }
      }
    }
    // store VEC bytes; for VEC=8 write as uint2
    *reinterpret_cast<uint2*>(smem + off) = *reinterpret_cast<const uint2*>(tmp);
  }
}

__global__ void __launch_bounds__(THREADS, 2) int8mma_fused_static_prepacked_kernel(
    const half* __restrict__ X, const int8_t* __restrict__ B_nk, half* __restrict__ C, float sa,
    const float* __restrict__ sw, int M, int N, int K) {
  int bx = blockIdx.x;
  int by = blockIdx.y;
  if ((by & 1) != 0) bx = gridDim.x - 1 - bx;
  const int tile_m = by * BM;
  const int tile_n = bx * BN;
  if (tile_m >= M || tile_n >= N) return;

  extern __shared__ __align__(16) int8_t smem_base[];
  int8_t* stage_A[STAGES];
  int8_t* stage_B[STAGES];
#pragma unroll
  for (int s = 0; s < STAGES; ++s) {
    stage_A[s] = smem_base + s * BYTES_STAGE_PRE;
    stage_B[s] = stage_A[s] + BYTES_A;
  }

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int warp_m = (warp / WARP_COLS) * WARP_M;
  const int warp_n = (warp % WARP_COLS) * WARP_N;
  const float inv_sa = 1.0f / sa;

  int acc[WARP_TM][WARP_TN][4];
#pragma unroll
  for (int i = 0; i < WARP_TM; ++i)
#pragma unroll
    for (int j = 0; j < WARP_TN; ++j)
      acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0;

  __shared__ float smem_sw[BN];
  for (int i = tid; i < BN; i += THREADS) {
    int col = tile_n + i;
    smem_sw[i] = (col < N) ? (sa * sw[col]) : 0.f;
  }

  const int k_tiles = (K + BK - 1) / BK;
  int write_stage = 0;
#pragma unroll
  for (int s = 0; s < STAGES - 1; ++s) {
    if (s < k_tiles) {
      g2s_A_quant_half(stage_A[write_stage], X, M, K, tile_m, s * BK, tid, inv_sa);
      g2s_B_pre(stage_B[write_stage], B_nk, N, K, tile_n, s * BK, tid);
    }
    cp_async_commit_group();
    write_stage = (write_stage + 1) % STAGES;
  }

  int read_stage = 0;
  for (int kt = 0; kt < k_tiles; ++kt) {
    int kt_prefetch = kt + (STAGES - 1);
    if (kt_prefetch < k_tiles) {
      g2s_A_quant_half(stage_A[write_stage], X, M, K, tile_m, kt_prefetch * BK, tid, inv_sa);
      g2s_B_pre(stage_B[write_stage], B_nk, N, K, tile_n, kt_prefetch * BK, tid);
    }
    cp_async_commit_group();
    write_stage = (write_stage + 1) % STAGES;
    // A quant path is sync stores, not all cp.async; still wait B copies
    cp_async_wait_group<STAGES - 2>();
    __syncthreads();

    const int8_t* As = stage_A[read_stage];
    const int8_t* Bs = stage_B[read_stage];
#pragma unroll
    for (int kk = 0; kk < WARP_TK; ++kk) {
      uint32_t a_frag[WARP_TM][4];
#pragma unroll
      for (int mi = 0; mi < WARP_TM; ++mi)
        load_a_frag(a_frag[mi][0], a_frag[mi][1], a_frag[mi][2], a_frag[mi][3], As,
                    warp_m + mi * MMA_M, kk * MMA_K, lane);
      uint32_t b_frag[WARP_TN][2];
#pragma unroll
      for (int ni = 0; ni < WARP_TN; ++ni)
        load_b_frag_pre(b_frag[ni][0], b_frag[ni][1], Bs, kk * MMA_K, warp_n + ni * MMA_N, lane);
#pragma unroll
      for (int mi = 0; mi < WARP_TM; ++mi)
#pragma unroll
        for (int ni = 0; ni < WARP_TN; ++ni)
          mma_s8s8s32_m16n8k32(acc[mi][ni][0], acc[mi][ni][1], acc[mi][ni][2], acc[mi][ni][3],
                               a_frag[mi][0], a_frag[mi][1], a_frag[mi][2], a_frag[mi][3],
                               b_frag[ni][0], b_frag[ni][1]);
    }
    __syncthreads();
    read_stage = (read_stage + 1) % STAGES;
  }
  cp_async_wait_group<0>();

  int group = lane >> 2;
  int thr = lane & 3;
#pragma unroll
  for (int mi = 0; mi < WARP_TM; ++mi) {
#pragma unroll
    for (int ni = 0; ni < WARP_TN; ++ni) {
      int row = tile_m + warp_m + mi * MMA_M + group;
      int col_local = warp_n + ni * MMA_N + thr * 2;
      int col = tile_n + col_local;
      float s0 = smem_sw[col_local];
      float s1 = smem_sw[col_local + 1];
      half2 h0 = __floats2half2_rn(static_cast<float>(acc[mi][ni][0]) * s0,
                                   static_cast<float>(acc[mi][ni][1]) * s1);
      half2 h1 = __floats2half2_rn(static_cast<float>(acc[mi][ni][2]) * s0,
                                   static_cast<float>(acc[mi][ni][3]) * s1);
      auto store_pair = [&](int r, half2 h) {
        if (r >= M) return;
        half* ptr = &C[static_cast<int64_t>(r) * N + col];
        bool can_vec = (col + 1 < N) && ((col & 1) == 0) &&
                       ((reinterpret_cast<uintptr_t>(ptr) & 3u) == 0u);
        if (can_vec) {
          *reinterpret_cast<half2*>(ptr) = h;
        } else {
          if (col < N) ptr[0] = __low2half(h);
          if (col + 1 < N) ptr[1] = __high2half(h);
        }
      };
      store_pair(row, h0);
      store_pair(row + 8, h1);
    }
  }
}

extern "C" int int8mma_run_fused_static_prepacked_b(void const* x_half, void const* b_nk, void* c,
                                                    float sa, float const* sw, int M, int N, int K) {
  if (!x_half || !b_nk || !c || !sw || M <= 0 || N <= 0 || K <= 0 || sa <= 0.f)
    return static_cast<int>(cudaErrorInvalidValue);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
  dim3 block(THREADS);
  int smem = BYTES_TOTAL_PRE;
  auto err = cudaFuncSetAttribute(int8mma_fused_static_prepacked_kernel,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  if (err != cudaSuccess) return static_cast<int>(err);
  int8mma_fused_static_prepacked_kernel<<<grid, block, smem>>>(
      static_cast<const half*>(x_half), static_cast<const int8_t*>(b_nk), static_cast<half*>(c), sa,
      sw, M, N, K);
  return static_cast<int>(cudaGetLastError());
}


// Fast static quant: half [M,K] -> int8 [M,K]
__global__ void quantize_static_half_kernel(const half* __restrict__ X, int8_t* __restrict__ Q,
                                            float inv_sa, int64_t n_elem) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  int64_t stride = (int64_t)blockDim.x * gridDim.x;
  for (; i < n_elem; i += stride) {
    float f = __half2float(X[i]) * inv_sa;
    int q = __float2int_rn(f);
    q = q < -127 ? -127 : (q > 127 ? 127 : q);
    Q[i] = static_cast<int8_t>(q);
  }
}

extern "C" int int8mma_quantize_static_half(void const* x_half, void* q_int8, float sa, int M, int K) {
  if (!x_half || !q_int8 || M <= 0 || K <= 0 || sa <= 0.f)
    return static_cast<int>(cudaErrorInvalidValue);
  int64_t n = (int64_t)M * K;
  float inv = 1.0f / sa;
  int threads = 256;
  int blocks = (int)min((n + threads - 1) / threads, (int64_t)2048);
  quantize_static_half_kernel<<<blocks, threads>>>(
      static_cast<const half*>(x_half), static_cast<int8_t*>(q_int8), inv, n);
  return static_cast<int>(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// True M=1 INT8 GEMV (no MMA tile pad).
// A: int8 [1,K] or fused half [1,K]; B math layout: int8 [K,N]; C: half [1,N]
// ---------------------------------------------------------------------------

// One thread per output column; shared A; DP4A over K using prepacked B[N,K].
static constexpr int GEMV_THREADS = 256;

__device__ __forceinline__ int dp4a_s8(int a, int b, int c) {
#if __CUDA_ARCH__ >= 610
  return __dp4a(a, b, c);
#else
  char4 aa = *reinterpret_cast<char4*>(&a);
  char4 bb = *reinterpret_cast<char4*>(&b);
  return c + int(aa.x) * int(bb.x) + int(aa.y) * int(bb.y) + int(aa.z) * int(bb.z) +
         int(aa.w) * int(bb.w);
#endif
}

__global__ void __launch_bounds__(GEMV_THREADS)
int8_gemv_m1_math_kernel(const int8_t* __restrict__ A, const int8_t* __restrict__ B,
                         half* __restrict__ C, const float* __restrict__ sa,
                         const float* __restrict__ sw, int N, int K) {
  const int col = blockIdx.x * GEMV_THREADS + threadIdx.x;
  extern __shared__ __align__(16) int8_t a_smem[];
  const int k_pad = (K + 15) & ~15;
  for (int i = threadIdx.x; i < k_pad; i += GEMV_THREADS) {
    a_smem[i] = (i < K) ? A[i] : 0;
  }
  __syncthreads();
  if (col >= N) return;
  const float scale_a = sa[0];
  int acc = 0;
  const int8_t* b_col = B + col;
  for (int k0 = 0; k0 < K; k0 += 4) {
    int a_pack = *reinterpret_cast<const int*>(a_smem + k0);
    int b_pack = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      int kk = k0 + i;
      int8_t bv = (kk < K) ? b_col[static_cast<int64_t>(kk) * N] : 0;
      b_pack |= (static_cast<uint32_t>(static_cast<uint8_t>(bv)) << (8 * i));
    }
    acc = dp4a_s8(a_pack, b_pack, acc);
  }
  C[col] = __float2half(static_cast<float>(acc) * scale_a * sw[col]);
}

__global__ void __launch_bounds__(GEMV_THREADS)
int8_gemv_m1_fused_static_kernel(const half* __restrict__ X, const int8_t* __restrict__ B,
                                 half* __restrict__ C, const float* __restrict__ sa,
                                 const float* __restrict__ sw, int N, int K) {
  const int col = blockIdx.x * GEMV_THREADS + threadIdx.x;
  extern __shared__ __align__(16) int8_t a_smem[];
  const int k_pad = (K + 15) & ~15;
  const float scale_a = sa[0];
  const float inv_sa = 1.0f / scale_a;
  for (int i = threadIdx.x; i < k_pad; i += GEMV_THREADS) {
    if (i < K) {
      float f = __half2float(X[i]) * inv_sa;
      int q = __float2int_rn(f);
      q = q < -127 ? -127 : (q > 127 ? 127 : q);
      a_smem[i] = static_cast<int8_t>(q);
    } else {
      a_smem[i] = 0;
    }
  }
  __syncthreads();
  if (col >= N) return;
  int acc = 0;
  const int8_t* b_col = B + col;
  for (int k0 = 0; k0 < K; k0 += 4) {
    int a_pack = *reinterpret_cast<const int*>(a_smem + k0);
    int b_pack = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      int kk = k0 + i;
      int8_t bv = (kk < K) ? b_col[static_cast<int64_t>(kk) * N] : 0;
      b_pack |= (static_cast<uint32_t>(static_cast<uint8_t>(bv)) << (8 * i));
    }
    acc = dp4a_s8(a_pack, b_pack, acc);
  }
  C[col] = __float2half(static_cast<float>(acc) * scale_a * sw[col]);
}

__global__ void __launch_bounds__(GEMV_THREADS)
int8_gemv_m1_prepacked_kernel(const int8_t* __restrict__ A, const int8_t* __restrict__ B_nk,
                              half* __restrict__ C, const float* __restrict__ sa,
                              const float* __restrict__ sw, int N, int K) {
  const int col = blockIdx.x * GEMV_THREADS + threadIdx.x;
  extern __shared__ __align__(16) int8_t a_smem[];
  const int k_pad = (K + 15) & ~15;
  for (int i = threadIdx.x; i < k_pad; i += GEMV_THREADS) {
    a_smem[i] = (i < K) ? A[i] : 0;
  }
  __syncthreads();
  if (col >= N) return;
  const float scale_a = sa[0];
  int acc = 0;
  const int8_t* brow = B_nk + static_cast<int64_t>(col) * K;
  int k0 = 0;
  for (; k0 + 16 <= K; k0 += 16) {
    const int* ap = reinterpret_cast<const int*>(a_smem + k0);
    const int* bp = reinterpret_cast<const int*>(brow + k0);
#pragma unroll
    for (int t = 0; t < 4; ++t) {
      acc = dp4a_s8(ap[t], bp[t], acc);
    }
  }
  for (; k0 < K; k0 += 4) {
    int a_pack = *reinterpret_cast<const int*>(a_smem + k0);
    int b_pack = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      int kk = k0 + i;
      int8_t bv = (kk < K) ? brow[kk] : 0;
      b_pack |= (static_cast<uint32_t>(static_cast<uint8_t>(bv)) << (8 * i));
    }
    acc = dp4a_s8(a_pack, b_pack, acc);
  }
  C[col] = __float2half(static_cast<float>(acc) * scale_a * sw[col]);
}

__global__ void __launch_bounds__(GEMV_THREADS)
int8_gemv_m1_fused_static_prepacked_kernel(const half* __restrict__ X,
                                           const int8_t* __restrict__ B_nk, half* __restrict__ C,
                                           const float* __restrict__ sa,
                                           const float* __restrict__ sw, int N, int K) {
  const int col = blockIdx.x * GEMV_THREADS + threadIdx.x;
  extern __shared__ __align__(16) int8_t a_smem[];
  const int k_pad = (K + 15) & ~15;
  const float scale_a = sa[0];
  const float inv_sa = 1.0f / scale_a;
  for (int i = threadIdx.x; i < k_pad; i += GEMV_THREADS) {
    if (i < K) {
      float f = __half2float(X[i]) * inv_sa;
      int q = __float2int_rn(f);
      q = q < -127 ? -127 : (q > 127 ? 127 : q);
      a_smem[i] = static_cast<int8_t>(q);
    } else {
      a_smem[i] = 0;
    }
  }
  __syncthreads();
  if (col >= N) return;
  int acc = 0;
  const int8_t* brow = B_nk + static_cast<int64_t>(col) * K;
  int k0 = 0;
  for (; k0 + 16 <= K; k0 += 16) {
    const int* ap = reinterpret_cast<const int*>(a_smem + k0);
    const int* bp = reinterpret_cast<const int*>(brow + k0);
#pragma unroll
    for (int t = 0; t < 4; ++t) {
      acc = dp4a_s8(ap[t], bp[t], acc);
    }
  }
  for (; k0 < K; k0 += 4) {
    int a_pack = *reinterpret_cast<const int*>(a_smem + k0);
    int b_pack = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      int kk = k0 + i;
      int8_t bv = (kk < K) ? brow[kk] : 0;
      b_pack |= (static_cast<uint32_t>(static_cast<uint8_t>(bv)) << (8 * i));
    }
    acc = dp4a_s8(a_pack, b_pack, acc);
  }
  C[col] = __float2half(static_cast<float>(acc) * scale_a * sw[col]);
}

extern "C" int int8_gemv_m1_run(void const* a_int8, void const* b_kn, void* c_half,
                                float const* sa, float const* sw, int N, int K) {
  if (!a_int8 || !b_kn || !c_half || !sa || !sw || N <= 0 || K <= 0)
    return static_cast<int>(cudaErrorInvalidValue);
  int k_pad = (K + 15) & ~15;
  dim3 grid((N + GEMV_THREADS - 1) / GEMV_THREADS);
  int8_gemv_m1_math_kernel<<<grid, GEMV_THREADS, k_pad>>>(
      static_cast<const int8_t*>(a_int8), static_cast<const int8_t*>(b_kn),
      static_cast<half*>(c_half), sa, sw, N, K);
  return static_cast<int>(cudaGetLastError());
}

extern "C" int int8_gemv_m1_run_prepacked_b(void const* a_int8, void const* b_nk, void* c_half,
                                            float const* sa, float const* sw, int N, int K) {
  if (!a_int8 || !b_nk || !c_half || !sa || !sw || N <= 0 || K <= 0)
    return static_cast<int>(cudaErrorInvalidValue);
  int k_pad = (K + 15) & ~15;
  dim3 grid((N + GEMV_THREADS - 1) / GEMV_THREADS);
  int8_gemv_m1_prepacked_kernel<<<grid, GEMV_THREADS, k_pad>>>(
      static_cast<const int8_t*>(a_int8), static_cast<const int8_t*>(b_nk),
      static_cast<half*>(c_half), sa, sw, N, K);
  return static_cast<int>(cudaGetLastError());
}

extern "C" int int8_gemv_m1_run_fused_static(void const* x_half, void const* b_kn, void* c_half,
                                             float const* sa, float const* sw, int N, int K) {
  if (!x_half || !b_kn || !c_half || !sa || !sw || N <= 0 || K <= 0)
    return static_cast<int>(cudaErrorInvalidValue);
  int k_pad = (K + 15) & ~15;
  dim3 grid((N + GEMV_THREADS - 1) / GEMV_THREADS);
  int8_gemv_m1_fused_static_kernel<<<grid, GEMV_THREADS, k_pad>>>(
      static_cast<const half*>(x_half), static_cast<const int8_t*>(b_kn),
      static_cast<half*>(c_half), sa, sw, N, K);
  return static_cast<int>(cudaGetLastError());
}

extern "C" int int8_gemv_m1_run_fused_static_prepacked_b(void const* x_half, void const* b_nk,
                                                         void* c_half, float const* sa,
                                                         float const* sw, int N, int K) {
  if (!x_half || !b_nk || !c_half || !sa || !sw || N <= 0 || K <= 0)
    return static_cast<int>(cudaErrorInvalidValue);
  int k_pad = (K + 15) & ~15;
  dim3 grid((N + GEMV_THREADS - 1) / GEMV_THREADS);
  int8_gemv_m1_fused_static_prepacked_kernel<<<grid, GEMV_THREADS, k_pad>>>(
      static_cast<const half*>(x_half), static_cast<const int8_t*>(b_nk),
      static_cast<half*>(c_half), sa, sw, N, K);
  return static_cast<int>(cudaGetLastError());
}
