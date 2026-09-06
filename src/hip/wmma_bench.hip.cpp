// HIP WMMA micro-benchmark: validates the native gfx11 (RDNA3) wave32
// 16x16x16 fp16 fragment layout via the __builtin_amdgcn_wmma_*_w32 builtin,
// then measures the WMMA pipe throughput ceiling. De-risks the pure-HIP EP
// idea: no RADV coopmat store bug, wave32 is native.
//
// Semantics (hipfire.dev/learn/rdna-wmma): D[i][j] = sum_k A[i][k] * B[j][k]
// (the B operand is the TRANSPOSED B). wave32 fragments:
//   A: half16_t per lane (16 fp16)
//   B: half16_t per lane (16 fp16)
//   C/D: float8_t (8 fp32 per lane)
// Candidate lane mapping (verified empirically below): lane L holds
// A[row=L%16][*], B[row=L%16][*], D[col=L%16][row=(L/16)*8+e].
//
// Build: hipcc --offload-arch=gfx1100 wmma_bench.hip.cpp -o wmma_bench

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>

#define CHECK(x)                                                               \
  do {                                                                         \
    hipError_t e = (x);                                                        \
    if (e != hipSuccess) {                                                     \
      fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(e),         \
              __FILE__, __LINE__);                                             \
      return 1;                                                                \
    }                                                                          \
  } while (0)

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float __attribute__((ext_vector_type(8))) float8_t;

// Correctness kernel: one wave32 (32 threads). Each lane loads its A and B
// fragments from global and does one WMMA; the D fragment is stored back.
__global__ void wmma_correct(const _Float16* A, const _Float16* B, float* D)
{
  const int l = threadIdx.x;  // lane 0..31 (one wave32)
  half16_t a, b;
  for (int e = 0; e < 16; ++e) {
    a[e] = A[(l % 16) * 16 + e];  // row l%16, all 16 k
    b[e] = B[(l % 16) * 16 + e];  // row l%16, all 16 k (B operand is B^T)
  }
  float8_t d = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  d = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, d);
  // verified layout: lane l holds D[row = 2e + (l/16)][col = l%16]
  for (int e = 0; e < 8; ++e)
    D[(2 * e + (l / 16)) * 16 + (l % 16)] = d[e];
}

// Throughput kernel: each wave issues many quad WMMAs on register-resident
// operands (no memory traffic in the loop).
__global__ void wmma_throughput(const _Float16* A, const _Float16* B, float* D,
                                int quads)
{
  const int l = threadIdx.x;
  const int wg = blockIdx.x;
  half16_t a0, a1, a2, a3, b0, b1, b2, b3;
  for (int e = 0; e < 16; ++e) {
    _Float16 av = A[(l % 16) * 16 + e];
    _Float16 bv = B[(l % 16) * 16 + e];
    a0[e] = a1[e] = a2[e] = a3[e] = av;
    b0[e] = b1[e] = b2[e] = b3[e] = bv;
  }
  float8_t m[8];
#pragma unroll
  for (int j = 0; j < 8; ++j)
    for (int e = 0; e < 8; ++e) m[j][e] = 0.f;
  for (int q = 0; q < quads; ++q) {
    m[0] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a0, b0, m[0]);
    m[1] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a1, b1, m[1]);
    m[2] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a2, b2, m[2]);
    m[3] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a3, b3, m[3]);
    m[4] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a0, b1, m[4]);
    m[5] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a1, b2, m[5]);
    m[6] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a2, b3, m[6]);
    m[7] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a3, b0, m[7]);
  }
  float s = 0.f;
#pragma unroll
  for (int j = 0; j < 8; ++j)
    for (int e = 0; e < 8; ++e) s += m[j][e];
  if (s == 12345.678f) D[wg * 32 + l] = s;
}

int main()
{
  CHECK(hipSetDevice(0));
  hipDeviceProp_t prop;
  CHECK(hipGetDeviceProperties(&prop, 0));
  printf("device: %s, gcnArch: %s\n", prop.name, prop.gcnArchName);

  // ---------- correctness ----------
  std::vector<_Float16> A(16 * 16), B(16 * 16);
  std::vector<float> D(16 * 16), Dref(16 * 16);
  for (int m = 0; m < 16; ++m)
    for (int k = 0; k < 16; ++k) A[m * 16 + k] = (_Float16)(0.1f * m + 0.01f * k);
  for (int n = 0; n < 16; ++n)
    for (int k = 0; k < 16; ++k) B[n * 16 + k] = (_Float16)(0.05f * n - 0.02f * k);
  // D = A @ B^T : D[m][n] = sum_k A[m][k] * B[n][k]
  for (int m = 0; m < 16; ++m)
    for (int n = 0; n < 16; ++n) {
      float s = 0.f;
      for (int k = 0; k < 16; ++k)
        s += (float)A[m * 16 + k] * (float)B[n * 16 + k];
      Dref[m * 16 + n] = s;
    }
  _Float16 *dA, *dB;
  float* dD;
  CHECK(hipMalloc(&dA, 16 * 16 * sizeof(_Float16)));
  CHECK(hipMalloc(&dB, 16 * 16 * sizeof(_Float16)));
  CHECK(hipMalloc(&dD, 16 * 16 * sizeof(float)));
  CHECK(hipMemcpy(dA, A.data(), 16 * 16 * sizeof(_Float16), hipMemcpyHostToDevice));
  CHECK(hipMemcpy(dB, B.data(), 16 * 16 * sizeof(_Float16), hipMemcpyHostToDevice));
  wmma_correct<<<1, 32>>>(dA, dB, dD);
  CHECK(hipGetLastError());
  CHECK(hipDeviceSynchronize());
  CHECK(hipMemcpy(D.data(), dD, 16 * 16 * sizeof(float), hipMemcpyDeviceToHost));
  float maxerr = 0.f;
  int worst = -1;
  for (int i = 0; i < 256; ++i) {
    float e = fabsf(D[i] - Dref[i]);
    if (e > maxerr) { maxerr = e; worst = i; }
  }
  printf("correctness: max err = %.5f at (%d,%d) [expect ~0 if layout matches]\n",
         maxerr, worst / 16, worst % 16);
  // DEBUG: print lane 0, 1, 16 fragments + ref expectations
  printf("lane0 d: ");
  for (int e = 0; e < 8; ++e) printf("%8.2f ", D[0 * 8 + e]);
  printf("\nlane1 d: ");
  for (int e = 0; e < 8; ++e) printf("%8.2f ", D[1 * 8 + e]);
  printf("\nlane16 d: ");
  for (int e = 0; e < 8; ++e) printf("%8.2f ", D[16 * 8 + e]);
  printf("\nref[0][0..7]  : ");
  for (int n = 0; n < 8; ++n) printf("%8.2f ", Dref[n]);
  printf("\nref[1][0..7]  : ");
  for (int n = 0; n < 8; ++n) printf("%8.2f ", Dref[16 + n]);
  printf("\nref[0..7][0]  : ");
  for (int m = 0; m < 8; ++m) printf("%8.2f ", Dref[m * 16]);
  printf("\nref[0..7][1]  : ");
  for (int m = 0; m < 8; ++m) printf("%8.2f ", Dref[m * 16 + 1]);
  printf("\n");
  if (maxerr > 1e-2f) {
    printf("  mismatch - D[0..3][0..3]:\n");
    for (int m = 0; m < 4; ++m) {
      printf("    ");
      for (int n = 0; n < 4; ++n) printf("%7.2f ", D[m * 16 + n]);
      printf("\n");
    }
  }

  // ---------- throughput ----------
  const int nw = 96 * 64;   // 16 waves/SIMD  // one wave32 per block, 16 per SIMD-slot x 4 SIMD x 96 CUs
  const int quads = 4096;
  float* dD2;
  CHECK(hipMalloc(&dD2, nw * 32 * sizeof(float)));
  hipEvent_t t0, t1;
  CHECK(hipEventCreate(&t0));
  CHECK(hipEventCreate(&t1));
  wmma_throughput<<<nw, 32>>>(dA, dB, dD2, quads);
  CHECK(hipDeviceSynchronize());
  CHECK(hipEventRecord(t0));
  const int iters = 10;
  for (int it = 0; it < iters; ++it)
    wmma_throughput<<<nw, 32>>>(dA, dB, dD2, quads);
  CHECK(hipEventRecord(t1));
  CHECK(hipEventSynchronize(t1));
  float ms = 0.f;
  CHECK(hipEventElapsedTime(&ms, t0, t1));
  ms /= iters;
  double flops = (double)nw * quads * 8.0 * 2.0 * 16.0 * 16.0 * 16.0;
  printf("throughput: %d waves x %d quads = %.0f GFLOP in %.3f ms -> %.1f TFLOPS\n",
         nw, quads, flops / 1e9, ms, flops / (ms * 1e-3) / 1e12);
  CHECK(hipFree(dA));
  CHECK(hipFree(dB));
  CHECK(hipFree(dD));
  CHECK(hipFree(dD2));
  return 0;
}
