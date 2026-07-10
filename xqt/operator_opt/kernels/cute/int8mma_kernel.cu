// int8mma Ada sm_89: math B[K,N] or prepacked B[N,K]
// C = half((A@B)_i32 * sa * sw)
// Prepacked B[N,K]: offline transpose so G2S is coalesced along K and B fragments are uint32 loads

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

#ifndef INT8MMA_STAGES
#define INT8MMA_STAGES 3
#endif

static constexpr int BM = 128;
static constexpr int BN = 128;
static constexpr int BK = 64;
static constexpr int STAGES = INT8MMA_STAGES;
static constexpr int WARPS = 4;
static constexpr int THREADS = WARPS * 32;
static constexpr int WARP_M = 64;
static constexpr int WARP_N = 64;
static constexpr int MMA_M = 16;
static constexpr int MMA_N = 8;
static constexpr int MMA_K = 32;
static constexpr int WARP_TM = WARP_M / MMA_M;
static constexpr int WARP_TN = WARP_N / MMA_N;
static constexpr int WARP_TK = BK / MMA_K;
static constexpr int A_LD = 128;
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
  return row * A_LD + (col ^ ((row & 7) << 4));
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
  const int warp_m = (warp >> 1) * WARP_M;
  const int warp_n = (warp & 1) * WARP_N;

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
  return "int8mma-ada-sm89-v8 math+prepack+fused_static_act stages=3 tile=128x128x64";
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
  const int warp_m = (warp >> 1) * WARP_M;
  const int warp_n = (warp & 1) * WARP_N;
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
