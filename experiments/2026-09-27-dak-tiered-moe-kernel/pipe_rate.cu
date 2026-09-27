// Issue rate of the dequant building blocks on sm_90: cycles per warp
// instruction per SM sub-partition, 4 warps per SMSP, 8 independent chains each.
//   pipe_rate
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>

template <int OP>
__global__ void rate(uint32_t* out, int iters, long long* cyc) {
  uint32_t r[8];
  for (int i = 0; i < 8; ++i) r[i] = threadIdx.x * 2654435761u + i;
  __syncthreads();
  long long t0 = clock64();
  for (int n = 0; n < iters; ++n) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      if (OP == 0) {  // F2FP.F16.E4M3.UNPACK_B
        asm volatile("{ .reg .b16 lo, hi; mov.b32 {lo, hi}, %0; cvt.rn.f16x2.e4m3x2 %0, lo; }" : "+r"(r[i]));
      } else if (OP == 1) {  // LOP3
        asm volatile("lop3.b32 %0, %0, 0x9c9c9c9c, %0, 0xf8;" : "+r"(r[i]));
      } else if (OP == 2) {  // IMAD.SHL
        asm volatile("mul.lo.u32 %0, %0, 4;" : "+r"(r[i]));
      } else if (OP == 3) {  // IMAD.HI (right shift on the FMA pipe)
        asm volatile("mul.hi.u32 %0, %0, 0x40000000;" : "+r"(r[i]));
      } else if (OP == 4) {  // SHF
        asm volatile("shr.b32 %0, %0, 2;" : "+r"(r[i]));
      } else if (OP == 5) {  // PRMT
        asm volatile("prmt.b32 %0, %0, 0, 0x3120;" : "+r"(r[i]));
      } else if (OP == 6) {  // HMUL2
        asm volatile("mul.rn.f16x2 %0, %0, %0;" : "+r"(r[i]));
      }
    }
  }
  long long t1 = clock64();
  uint32_t s = 0;
  for (int i = 0; i < 8; ++i) s ^= r[i];
  out[blockIdx.x * blockDim.x + threadIdx.x] = s;
  if (threadIdx.x == 0) cyc[blockIdx.x] = t1 - t0;
}

int main() {
  const int iters = 4096, threads = 512;   // 16 warps / SM = 4 per SMSP
  uint32_t* out; long long* cyc;
  cudaMalloc(&out, 132 * threads * 4); cudaMalloc(&cyc, 132 * 8);
  const char* names[] = {"F2FP e4m3x2->f16x2", "LOP3", "IMAD.SHL", "IMAD.HI", "SHF", "PRMT", "HMUL2"};
  auto run = [&](auto kern, int op) {
    kern<<<132, threads>>>(out, iters, cyc);
    kern<<<132, threads>>>(out, iters, cyc);
    cudaDeviceSynchronize();
    long long c; cudaMemcpy(&c, cyc, 8, cudaMemcpyDeviceToHost);
    // per SMSP: 4 warps x 8 chains x iters instructions
    printf("%-20s %.2f cycles per warp-instruction per SMSP\n", names[op], double(c) / (4.0 * 8 * iters));
  };
  run(rate<0>, 0); run(rate<1>, 1); run(rate<2>, 2); run(rate<3>, 3); run(rate<4>, 4); run(rate<5>, 5); run(rate<6>, 6);
  return 0;
}
