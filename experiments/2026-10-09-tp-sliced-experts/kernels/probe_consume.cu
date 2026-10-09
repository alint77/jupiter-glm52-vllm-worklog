// Consumer-only throughput of the routed unit (128 rows x 512 K, INT4 + bf16
// group-32 scales, 8 token columns) from smem, no loads: how fast can NW
// consumer warps chew units, with or without a per-unit CTA barrier?
//   probe_consume [units per CTA]
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdint>

constexpr int W_BYTES = 128 * 512 / 2, S_BYTES = 16 * 128 * 2;
constexpr int XROW_STRIDE = 1024 + 64;
constexpr int STAGE_BYTES = (W_BYTES + S_BYTES + 8 * XROW_STRIDE + 1023) / 1024 * 1024;
constexpr int STAGES = 4;

__device__ __forceinline__ uint4 lds_v4(uint32_t a) {
  uint4 v;
  asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(a));
  return v;
}
__device__ __forceinline__ uint4 lds_v4_nv(uint32_t a) {
  uint4 v;
  asm("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
      : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(a));
  return v;
}
template <bool VOL>
__device__ __forceinline__ void mma_f16(float* d, const uint32_t* a, uint32_t b0, uint32_t b1,
                                        const float* c) {
  if (VOL)
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "f"(c[0]), "f"(c[1]),
          "f"(c[2]), "f"(c[3]));
  else
    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "f"(c[0]), "f"(c[1]),
          "f"(c[2]), "f"(c[3]));
}
__device__ __forceinline__ uint32_t lop3_and_or(uint32_t a, uint32_t mask, uint32_t magic) {
  uint32_t d;
  asm("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(d) : "r"(a), "r"(mask), "r"(magic));
  return d;
}
__device__ __forceinline__ void decode_int4_fast(uint32_t w, uint32_t* a) {
  const uint32_t magic = 0x64006400u;
  const uint32_t w8 = w >> 8;
  const uint32_t lo0 = lop3_and_or(w, 0x000F000Fu, magic);
  const uint32_t lo1 = lop3_and_or(w8, 0x000F000Fu, magic);
  const uint32_t hi0 = lop3_and_or(w, 0x00F000F0u, magic);
  const uint32_t hi1 = lop3_and_or(w8, 0x00F000F0u, magic);
  const __half2 unit = __halves2half2(__ushort_as_half(0x0400), __ushort_as_half(0x0400));
  const __half2 bias = __halves2half2(__ushort_as_half(0xAC08), __ushort_as_half(0xAC08));
  const __half2 unit16 = __halves2half2(__ushort_as_half(0x0040), __ushort_as_half(0x0040));
  const __half2 bias16 = __halves2half2(__ushort_as_half(0x9C80), __ushort_as_half(0x9C80));
  __half2 v;
  v = __hfma2(*reinterpret_cast<const __half2*>(&lo0), unit, bias);
  a[0] = *reinterpret_cast<const uint32_t*>(&v);
  v = __hfma2(*reinterpret_cast<const __half2*>(&lo1), unit, bias);
  a[1] = *reinterpret_cast<const uint32_t*>(&v);
  v = __hfma2(*reinterpret_cast<const __half2*>(&hi0), unit16, bias16);
  a[2] = *reinterpret_cast<const uint32_t*>(&v);
  v = __hfma2(*reinterpret_cast<const __half2*>(&hi1), unit16, bias16);
  a[3] = *reinterpret_cast<const uint32_t*>(&v);
}

// MODE 0: v15 as is (volatile asm). 1: non-volatile asm. 2: non-volatile, all
// loads of the warp's K slice issued up front.
template <int J, int MODE>
__device__ __forceinline__ void consume(uint32_t wb, uint32_t sb, uint32_t xb, float (*acc)[4]) {
  constexpr int KROW = 1024, SROW = 256;
  constexpr bool VOL = MODE == 0;
  uint4 W0[J], W1[J], SW[J], XV[J];
  if (MODE == 2) {
#pragma unroll
    for (int j = 0; j < J; ++j) {
      W0[j] = lds_v4_nv(wb + (2 * j) * KROW);
      W1[j] = lds_v4_nv(wb + (2 * j + 1) * KROW);
      SW[j] = lds_v4_nv(sb + j * SROW);
      XV[j] = lds_v4_nv(xb + j * 64);
    }
  }
#pragma unroll
  for (int j = 0; j < J; ++j) {
    uint4 w0, w1, sw, xv;
    if (MODE == 2) {
      w0 = W0[j], w1 = W1[j], sw = SW[j], xv = XV[j];
    } else if (MODE == 1) {
      w0 = lds_v4_nv(wb + (2 * j) * KROW), w1 = lds_v4_nv(wb + (2 * j + 1) * KROW);
      sw = lds_v4_nv(sb + j * SROW), xv = lds_v4_nv(xb + j * 64);
    } else {
      w0 = lds_v4(wb + (2 * j) * KROW), w1 = lds_v4(wb + (2 * j + 1) * KROW);
      sw = lds_v4(sb + j * SROW), xv = lds_v4(xb + j * 64);
    }
    const uint32_t w0s[4] = {w0.x, w0.y, w0.z, w0.w}, w1s[4] = {w1.x, w1.y, w1.z, w1.w};
    const uint32_t sws[4] = {sw.x, sw.y, sw.z, sw.w};
#pragma unroll
    for (int mb = 0; mb < 4; ++mb) {
      const float zero[4] = {0.f, 0.f, 0.f, 0.f};
      float d[4];
      uint32_t a[4];
      decode_int4_fast(w0s[mb], a);
      mma_f16<VOL>(d, a, xv.x, xv.y, zero);
      decode_int4_fast(w1s[mb], a);
      mma_f16<VOL>(d, a, xv.z, xv.w, d);
      const float s0 = __uint_as_float(sws[mb] << 16), s1 = __uint_as_float(sws[mb] & 0xFFFF0000u);
      acc[mb][0] = fmaf(s0, d[0], acc[mb][0]);
      acc[mb][1] = fmaf(s0, d[1], acc[mb][1]);
      acc[mb][2] = fmaf(s1, d[2], acc[mb][2]);
      acc[mb][3] = fmaf(s1, d[3], acc[mb][3]);
    }
  }
}

// NW consumer warps split a unit into 2 row halves x NW/2 K slices of 8/(NW/2)*... k16 rows.
template <int NW, int MODE, int SYNC>
__global__ void __launch_bounds__(NW * 32, 1) probe(int units, float* out) {
  extern __shared__ __align__(1024) unsigned char smem[];
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, g = lane / 4, tq = lane % 4;
  for (int i = threadIdx.x; i < STAGES * STAGE_BYTES / 4; i += blockDim.x)
    reinterpret_cast<uint32_t*>(smem)[i] = (i * 2654435761u) & 0x3F7F3F7Fu;
  __syncthreads();
  constexpr int KS = NW / 2, J = 16 / KS;  // K slices; k32 steps per warp
  const int h1 = warp / KS, k1 = warp % KS;
  const uint32_t base = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
  const uint32_t wo = (k1 * 2 * J) * 1024 + h1 * 512 + lane * 16;
  const uint32_t so = W_BYTES + (k1 * J) * 256 + h1 * 128 + 16 * g;
  const uint32_t xo = W_BYTES + S_BYTES + g * XROW_STRIDE + (k1 * 4 * J + tq) * 16;
  float acc[4][4] = {};
  for (int u = 0; u < units; ++u) {
    const uint32_t st = base + (u % STAGES) * STAGE_BYTES;
    consume<J, MODE>(st + wo, st + so, st + xo, acc);
    if (SYNC) asm volatile("bar.sync 1, %0;" ::"n"(NW * 32) : "memory");
  }
  float s = 0.f;
#pragma unroll
  for (int mb = 0; mb < 4; ++mb)
#pragma unroll
    for (int i = 0; i < 4; ++i) s += acc[mb][i];
  out[blockIdx.x * blockDim.x + threadIdx.x] = s;
}

template <int NW, int MODE, int SYNC>
void run(const char* name, int units, float* out) {
  auto k = probe<NW, MODE, SYNC>;
  const int smem = STAGES * STAGE_BYTES;
  cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  cudaFuncAttributes fa;
  cudaFuncGetAttributes(&fa, k);
  cudaEvent_t a, b;
  cudaEventCreate(&a);
  cudaEventCreate(&b);
  for (int w = 0; w < 3; ++w) k<<<132, NW * 32, smem>>>(units, out);
  cudaEventRecord(a);
  const int reps = 10;
  for (int r = 0; r < reps; ++r) k<<<132, NW * 32, smem>>>(units, out);
  cudaEventRecord(b);
  cudaEventSynchronize(b);
  float ms;
  cudaEventElapsedTime(&ms, a, b);
  const double us_unit = ms * 1e3 / reps / units;
  // weights + scales per unit, per SM; 132 SMs
  const double gbs = (W_BYTES + S_BYTES) / (us_unit * 1e-6) * 132 / 1e9;
  printf("%-28s regs %3d  %.3f us/unit  -> %6.0f GB/s of weights over 132 SMs  %s\n", name,
         fa.numRegs, us_unit, gbs, cudaGetErrorString(cudaGetLastError()));
}

int main(int argc, char** argv) {
  const int units = argc > 1 ? atoi(argv[1]) : 400;
  float* out;
  cudaMalloc(&out, 132 * 1024 * sizeof(float));
  run<8, 0, 0>("nw8 volatile", units, out);
  run<8, 0, 1>("nw8 volatile sync", units, out);
  run<8, 1, 0>("nw8 nonvol", units, out);
  run<8, 1, 1>("nw8 nonvol sync", units, out);
  run<8, 2, 1>("nw8 preload sync", units, out);
  run<16, 0, 1>("nw16 volatile sync", units, out);
  run<16, 1, 1>("nw16 nonvol sync", units, out);
  run<16, 2, 1>("nw16 preload sync", units, out);
  run<16, 2, 0>("nw16 preload", units, out);
  return 0;
}
