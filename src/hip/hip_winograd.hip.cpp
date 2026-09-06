// HIP Winograd F(2x2,3x3) WMMA conv kernel (gfx1100, native wave32).
//
// Port of the Vulkan EP's winograd_wmma.comp to HIP using the native
// __builtin_amdgcn_wmma_f32_16x16x16_f16_w32 path. Structure:
//   - WG = 128 threads = 4 wave32s; KB=1; each wave owns one k-block (16 ko)
//   - 16 nt tiles per WG (2 tile rows), 6x18x16c input strip in LDS
//   - V transform -> v_lds (normal LDS stores; no coopMatStore -> no RADV bug)
//   - WMMA: A = U rows (half16_t, lane L = row L%16), B = V columns from LDS
//     (lane L = nt L%16, 16 c values), D = float8_t in registers
//   - fold + writeback straight from the register accumulators
//
// Fragment layouts (verified by the wmma_bench micro test):
//   A: lane L holds A[row=L%16][k=0..15]          (16 fp16)
//   B: lane L holds B[row=L%16][k=0..15]          (16 fp16, semantics D=A@B^T)
//   D: lane L holds D[row=2e+(L/16)][col=L%16]     (8 fp32)

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include <cmath>

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

struct ConvParams {
  uint32_t H, W, C, M, M_pad, C_pad, tiles_w, do_silu, do_add;
};

__device__ __forceinline__ void wmma_mul(half16_t a, half16_t b, float8_t& d)
{
  d = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, d);
}

// B^T d B input transform (same math as the Vulkan kernel).
// Pass 1 (B^T d): per column j, over the 4 rows: t[i][j] = sum_k B^T[i][k] d[k][j]
// Pass 2 (t B):   per row i, over the 4 cols: v[i][j] = sum_k t[i][k] B[k][j]
__device__ __forceinline__ void winograd_in4(const _Float16* d, _Float16* v)
{
  float t[16];
  for (int j = 0; j < 4; ++j) {
    float d0 = (float)d[0 * 4 + j];
    float d1 = (float)d[1 * 4 + j];
    float d2 = (float)d[2 * 4 + j];
    float d3 = (float)d[3 * 4 + j];
    t[0 * 4 + j] = d0 - d2;
    t[1 * 4 + j] = d1 + d2;
    t[2 * 4 + j] = d2 - d1;
    t[3 * 4 + j] = d1 - d3;
  }
  for (int i = 0; i < 4; ++i) {
    v[i * 4 + 0] = (_Float16)(t[i * 4 + 0] - t[i * 4 + 2]);
    v[i * 4 + 1] = (_Float16)(t[i * 4 + 1] + t[i * 4 + 2]);
    v[i * 4 + 2] = (_Float16)(t[i * 4 + 2] - t[i * 4 + 1]);
    v[i * 4 + 3] = (_Float16)(t[i * 4 + 1] - t[i * 4 + 3]);
  }
}

// FOLD4: output transform row-sums, accumulated into the 4 output pixels.
#define FOLD4(Y00, Y01, Y10, Y11, w, m0, m1, m2, m3)                           \
  {                                                                            \
    float8_t s0 = m0 + m1 + m2;                                                \
    float8_t s1 = m1 - m2 - m3;                                                \
    if (w == 0) { Y00 = Y00 + s0; Y01 = Y01 + s1; }                            \
    else if (w == 1) { Y00 = Y00 + s0; Y01 = Y01 + s1; Y10 = Y10 + s0; Y11 = Y11 + s1; } \
    else if (w == 2) { Y00 = Y00 + s0; Y01 = Y01 + s1; Y10 = Y10 - s0; Y11 = Y11 - s1; } \
    else { Y10 = Y10 - s0; Y11 = Y11 - s1; }                                   \
  }

// 128 threads = 4 wave32s. Each wave = one k-block (16 ko). Grid = nt_wg.
__global__ __launch_bounds__(128) void winograd_conv(
    const _Float16* __restrict__ in, const _Float16* __restrict__ u,
    _Float16* __restrict__ out, const _Float16* __restrict__ add,
    ConvParams p)
{
  const uint wave = threadIdx.x / 32;
  const uint lane = threadIdx.x % 32;
  const uint kgroups = p.M_pad / 16;
  const uint kgroup = min(wave, kgroups - 1);
  const uint wg = blockIdx.x;
  const uint k_base = kgroup * 16;

  const uint tiles_w8 = p.tiles_w / 8;
  const uint row_pair = wg / tiles_w8;
  const uint tw_group = wg % tiles_w8;
  const uint th_a = row_pair * 2;
  const uint th_b = th_a + 1;
  const uint tw0 = tw_group * 8;

  const uint C_blocks = p.C_pad / 16;
  const uint Cblocks = C_blocks;

  __shared__ _Float16 strip4[6 * 18 * 16];
  __shared__ _Float16 v_lds[16 * 16 * 24];

  const uint bt = 16;

  // ---- strip load: 108 cells x 16 channels, split across 4 waves ----
  {
    const uint p0 = wave * 27;
    for (uint pc = p0 + lane; pc < p0 + 27; pc += 32) {
      const int h = int(2u * th_a) - 1 + int(pc / 18u);
      const int w = int(2u * tw0) - 1 + int(pc % 18u);
      for (uint q = 0; q < 4; ++q) {
        const uint c_b = q * 4;
        if (c_b + 3 < p.C && h >= 0 && h < int(p.H) && w >= 0 && w < int(p.W)) {
          const uint gbase = (uint(h) * p.W + uint(w)) * p.C + c_b;
          strip4[pc * 16 + q * 4 + 0] = in[gbase + 0];
          strip4[pc * 16 + q * 4 + 1] = in[gbase + 1];
          strip4[pc * 16 + q * 4 + 2] = in[gbase + 2];
          strip4[pc * 16 + q * 4 + 3] = in[gbase + 3];
        } else {
          const bool spatial_in = (h >= 0 && h < int(p.H) && w >= 0 && w < int(p.W));
          for (uint cc = 0; cc < 4; ++cc) {
            _Float16 val = 0.0f;
            if (spatial_in && c_b + cc < p.C)
              val = in[(uint(h) * p.W + uint(w)) * p.C + c_b + cc];
            strip4[pc * 16 + q * 4 + cc] = val;
          }
        }
      }
    }
  }

  // ---- y accumulators (fp32, in registers; the fold) ----
  float8_t y00 = {}, y01 = {}, y10 = {}, y11 = {};

  for (uint cb = 0; cb < Cblocks; ++cb) {
    const uint c_base = cb * 16;
    const uint cb_off = cb * 1024;

    __syncthreads();  // strip visible

    // ---- V transform: 64 idx split across 4 waves ----
    {
      const uint i0 = wave * 16;
      for (uint idx = i0 + lane; idx < i0 + 16; idx += 32) {
        const uint nt_local = idx / 4;
        const uint c4 = (idx % 4) * 4;
        const uint row_off = (nt_local >= 8) ? 2 : 0;
        const uint tw_local = nt_local & 7;
        for (uint cc2 = 0; cc2 < 4; ++cc2) {
          _Float16 d4[16];
          for (uint i = 0; i < 4; ++i)
            for (uint j = 0; j < 4; ++j) {
              const uint s4 = ((row_off + i) * 18 + (2 * tw_local + j)) * 16 + c4 + cc2;
              d4[i * 4 + j] = strip4[s4];
            }
          _Float16 v4[16];
          winograd_in4(d4, v4);
          for (uint wp = 0; wp < 16; ++wp)
            v_lds[wp * 384 + (c4 + cc2) * 24 + nt_local] = v4[wp];
        }
      }
    }
    __syncthreads();  // v_lds ready

    // ---- WMMA: 4 wp groups x 4 WMMAs each, A from U, B from v_lds ----
    for (uint wpi = 0; wpi < 4; ++wpi) {
      // A fragments: lane L holds U[ko=L%16][c=0..15] for (kblock, wp, cb)
      // U layout (host): [kblock][wp_i][cb][wp_local][ko][c], ko-major stride 16
      const uint ubase = (kgroup * 4 + wpi) * (C_blocks * 1024) + cb_off + wpi * 0;
      half16_t a0, a1, a2, a3;
      {
        const uint krow = lane % 16;
        const uint ub = (kgroup * 4 + wpi) * (C_blocks * 1024) + cb_off;
        for (int e = 0; e < 16; ++e) {
          a0[e] = u[ub + (wpi * 4 + 0) * 256 + krow * 16 + e];
          a1[e] = u[ub + (wpi * 4 + 1) * 256 + krow * 16 + e];
          a2[e] = u[ub + (wpi * 4 + 2) * 256 + krow * 16 + e];
          a3[e] = u[ub + (wpi * 4 + 3) * 256 + krow * 16 + e];
        }
      }
      // B fragments: lane L holds V[wp][c=0..15][nt=L%16]
      half16_t b0, b1, b2, b3;
      const uint nt = lane % 16;
      for (int e = 0; e < 16; ++e) {
        b0[e] = v_lds[(wpi * 4 + 0) * 384 + e * 24 + nt];
        b1[e] = v_lds[(wpi * 4 + 1) * 384 + e * 24 + nt];
        b2[e] = v_lds[(wpi * 4 + 2) * 384 + e * 24 + nt];
        b3[e] = v_lds[(wpi * 4 + 3) * 384 + e * 24 + nt];
      }
      float8_t m0 = {}, m1 = {}, m2 = {}, m3 = {};
      wmma_mul(a0, b0, m0);
      wmma_mul(a1, b1, m1);
      wmma_mul(a2, b2, m2);
      wmma_mul(a3, b3, m3);
      FOLD4(y00, y01, y10, y11, wpi, m0, m1, m2, m3);
    }
  }

  // ---- writeback: D fragment col = lane%16 (the nt), rows = 2e+(lane/16) ----
  const uint nt = lane % 16;
  const uint th = (nt < 8) ? th_a : th_b;
  const uint tw = tw0 + (nt & 7);
  const uint oh0 = 2 * th;
  const uint ow0 = 2 * tw;
  const uint o0 = (oh0 * p.W + ow0) * p.M;
  const uint o1 = (oh0 * p.W + ow0 + 1) * p.M;
  const uint o2 = ((oh0 + 1) * p.W + ow0) * p.M;
  const uint o3 = ((oh0 + 1) * p.W + ow0 + 1) * p.M;
  const uint koff = k_base + (lane / 16);
  const bool h0ok = oh0 < p.H && ow0 < p.W;
  const bool h1ok = oh0 < p.H && ow0 + 1 < p.W;
  const bool h2ok = oh0 + 1 < p.H && ow0 < p.W;
  const bool h3ok = oh0 + 1 < p.H && ow0 + 1 < p.W;
  for (uint e = 0; e < 8; ++e) {
    const uint ko = koff + 2 * e;
    if (ko >= p.M) break;
    const _Float16 bias = u[p.M_pad * C_blocks * 256 + ko];
    float v0 = y00[e] + (float)bias;
    float v1 = y01[e] + (float)bias;
    float v2 = y10[e] + (float)bias;
    float v3 = y11[e] + (float)bias;
    if (p.do_silu) {
      v0 = v0 / (1.0f + expf(-v0));
      v1 = v1 / (1.0f + expf(-v1));
      v2 = v2 / (1.0f + expf(-v2));
      v3 = v3 / (1.0f + expf(-v3));
    }
    if (p.do_add) {
      v0 += (float)add[o0 + ko];
      v1 += (float)add[o1 + ko];
      v2 += (float)add[o2 + ko];
      v3 += (float)add[o3 + ko];
    }
    if (h0ok) out[o0 + ko] = (_Float16)v0;
    if (h1ok) out[o1 + ko] = (_Float16)v1;
    if (h2ok) out[o2 + ko] = (_Float16)v2;
    if (h3ok) out[o3 + ko] = (_Float16)v3;
  }
}

// ===================== host test harness =====================
static void winograd_in4_host(const _Float16* d, _Float16* v)
{
  float t[16];
  for (int j = 0; j < 4; ++j) {
    float d0 = (float)d[0 * 4 + j], d1 = (float)d[1 * 4 + j];
    float d2 = (float)d[2 * 4 + j], d3 = (float)d[3 * 4 + j];
    t[0 * 4 + j] = d0 - d2;
    t[1 * 4 + j] = d1 + d2;
    t[2 * 4 + j] = d2 - d1;
    t[3 * 4 + j] = d1 - d3;
  }
  for (int i = 0; i < 4; ++i) {
    v[i * 4 + 0] = (_Float16)(t[i * 4 + 0] - t[i * 4 + 2]);
    v[i * 4 + 1] = (_Float16)(t[i * 4 + 1] + t[i * 4 + 2]);
    v[i * 4 + 2] = (_Float16)(t[i * 4 + 2] - t[i * 4 + 1]);
    v[i * 4 + 3] = (_Float16)(t[i * 4 + 1] - t[i * 4 + 3]);
  }
}

static void winograd_u_transform(const std::vector<_Float16>& w, int M, int C,
                                 const std::vector<_Float16>& bias,
                                 std::vector<_Float16>& u, uint32_t& u_size)
{
  const uint32_t M_pad = (M + 31) / 32 * 32;
  const uint32_t C_pad = (C + 15) / 16 * 16;
  const uint32_t C_blocks = C_pad / 16;
  const uint32_t kb_stride = M_pad / 16;
  const float G[4][3] = {{1, 0, 0}, {0.5f, 0.5f, 0.5f}, {0.5f, -0.5f, 0.5f}, {0, 0, 1}};
  u_size = kb_stride * 4 * C_blocks * 4 * 256 + M_pad;
  u.assign(u_size, 0);
  for (uint32_t ko = 0; ko < M; ++ko)
    for (uint32_t c = 0; c < C; ++c) {
      float g[9];
      for (int t = 0; t < 9; ++t)
        g[t] = (float)w[(size_t)ko * C * 9 + c * 9 + t];
      float Gg[4][3];
      for (int r = 0; r < 4; ++r)
        for (int cc = 0; cc < 3; ++cc) {
          Gg[r][cc] = 0.f;
          for (int k = 0; k < 3; ++k) Gg[r][cc] += G[r][k] * g[k * 3 + cc];
        }
      for (int r = 0; r < 4; ++r)
        for (int cc = 0; cc < 4; ++cc) {
          float uv = 0.f;
          for (int k = 0; k < 3; ++k) uv += Gg[r][k] * G[cc][k];
          uint32_t wp = r * 4 + cc;
          u[(ko / 16) * (4 * C_blocks * 4 * 256) + (wp / 4) * (C_blocks * 4 * 256) +
            (c / 16) * 1024 + (wp % 4) * 256 + (ko % 16) * 16 + (c % 16)] = (_Float16)uv;
        }
    }
  for (uint32_t ko = 0; ko < M; ++ko) u[kb_stride * 4 * C_blocks * 4 * 256 + ko] = bias[ko];
}

static void cpu_conv(const std::vector<_Float16>& in, int H, int W, int C, int M,
                     const std::vector<_Float16>& w, const std::vector<_Float16>& b,
                     std::vector<_Float16>& out)
{
  out.assign((size_t)H * W * M, 0);
  for (int oh = 0; oh < H; ++oh)
    for (int ow = 0; ow < W; ++ow)
      for (int m = 0; m < M; ++m) {
        float acc = (float)b[m];
        for (int c = 0; c < C; ++c)
          for (int r = 0; r < 3; ++r)
            for (int cc = 0; cc < 3; ++cc) {
              int ih = oh - 1 + r, iw = ow - 1 + cc;
              if (ih < 0 || ih >= H || iw < 0 || iw >= W) continue;
              acc += (float)in[((size_t)ih * W + iw) * C + c] *
                     (float)w[((size_t)m * C + c) * 9 + r * 3 + cc];
            }
        out[((size_t)oh * W + ow) * M + m] = (_Float16)acc;
      }
}

// Debug: dump the computed V (v_lds) for WG0 into the output buffer
__global__ __launch_bounds__(128) void v_dump(const _Float16* __restrict__ in,
                                              const _Float16* __restrict__ u,
                                              _Float16* __restrict__ out, ConvParams p)
{
  const uint wave = threadIdx.x / 32;
  const uint lane = threadIdx.x % 32;
  const uint wg = blockIdx.x;
  const uint tiles_w8 = p.tiles_w / 8;
  const uint row_pair = wg / tiles_w8;
  const uint tw_group = wg % tiles_w8;
  const uint th_a = row_pair * 2;
  const uint tw0 = tw_group * 8;
  __shared__ _Float16 strip4[6 * 18 * 16];
  __shared__ _Float16 v_lds[16 * 16 * 24];
  {
    const uint p0 = wave * 27;
    for (uint pc = p0 + lane; pc < p0 + 27; pc += 32) {
      const int h = int(2u * th_a) - 1 + int(pc / 18u);
      const int w = int(2u * tw0) - 1 + int(pc % 18u);
      for (uint q = 0; q < 4; ++q) {
        const uint c_b = q * 4;
        if (c_b + 3 < p.C && h >= 0 && h < int(p.H) && w >= 0 && w < int(p.W)) {
          const uint gbase = (uint(h) * p.W + uint(w)) * p.C + c_b;
          strip4[pc * 16 + q * 4 + 0] = in[gbase + 0];
          strip4[pc * 16 + q * 4 + 1] = in[gbase + 1];
          strip4[pc * 16 + q * 4 + 2] = in[gbase + 2];
          strip4[pc * 16 + q * 4 + 3] = in[gbase + 3];
        } else {
          for (uint cc = 0; cc < 4; ++cc) strip4[pc * 16 + q * 4 + cc] = 0.0f;
        }
      }
    }
  }
  __syncthreads();
  {
    const uint i0 = wave * 16;
    for (uint idx = i0 + lane; idx < i0 + 16; idx += 32) {
      const uint nt_local = idx / 4;
      const uint c4 = (idx % 4) * 4;
      const uint row_off = (nt_local >= 8) ? 2 : 0;
      const uint tw_local = nt_local & 7;
      for (uint cc2 = 0; cc2 < 4; ++cc2) {
        _Float16 d4[16];
        for (uint i = 0; i < 4; ++i)
          for (uint j = 0; j < 4; ++j) {
            const uint s4 = ((row_off + i) * 18 + (2 * tw_local + j)) * 16 + c4 + cc2;
            d4[i * 4 + j] = strip4[s4];
          }
        _Float16 v4[16];
        winograd_in4(d4, v4);
        for (uint wp = 0; wp < 16; ++wp)
          v_lds[wp * 384 + (c4 + cc2) * 24 + nt_local] = v4[wp];
      }
    }
  }
  __syncthreads();
  if (wg == 0) {
    if (lane < 16) out[lane] = v_lds[lane];
    if (lane == 16) {
      // A fragment for (kgroup=0, wpi=0, wp_local=0): U[ko=0][c=0..15]
      const uint C_blocks = p.C_pad / 16;
      const uint ub = (0 * 4 + 0) * (C_blocks * 1024);
      for (int e = 0; e < 16; ++e) out[32 + e] = u[ub + 0 + 0 + e];
    }
    if (lane == 18) {
      for (int e = 0; e < 16; ++e) out[96 + e] = v_lds[e * 24];
    }
    if (lane == 17) {
      // wpi=0 WMMA (wp0): m0 raw for lane 0 (ko even rows, nt=0)
      half16_t a0, b0;
      const uint C_blocks = p.C_pad / 16;
      const uint ub = (0 * 4 + 0) * (C_blocks * 1024);
      const uint krow = 0;
      for (int e = 0; e < 16; ++e) a0[e] = u[ub + 0 + krow * 16 + e];
      const uint nt = 0;
      for (int e = 0; e < 16; ++e) b0[e] = v_lds[(0) * 384 + e * 24 + nt];
      float8_t m0 = {};
      m0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a0, b0, m0);
      for (int e = 0; e < 8; ++e) out[64 + e] = m0[e];
    }
  }
}

int main()
{
  const int H = 16, W = 16, C = 16, M = 16;
  std::vector<_Float16> in((size_t)H * W * C), w((size_t)M * C * 9), b(M);
  for (size_t i = 0; i < in.size(); ++i) in[i] = (_Float16)(((i * 2654435761u) >> 13 & 0xFFFF) / 65536.0f - 0.5f);
  for (size_t i = 0; i < w.size(); ++i) w[i] = (_Float16)(((i * 40503u) >> 11 & 0xFF) / 128.0f - 0.5f);
  for (int m = 0; m < M; ++m) b[m] = (_Float16)(0.01f * (m % 7) - 0.02f);

  std::vector<_Float16> u;
  uint32_t u_size;
  winograd_u_transform(w, M, C, b, u, u_size);

  std::vector<_Float16> out((size_t)H * W * M), ref;
  cpu_conv(in, H, W, C, M, w, b, ref);

  _Float16 *d_in, *d_u, *d_out;
  CHECK(hipMalloc(&d_in, in.size() * 2));
  CHECK(hipMalloc(&d_u, u_size * 2));
  CHECK(hipMalloc(&d_out, out.size() * 2));
  CHECK(hipMemcpy(d_in, in.data(), in.size() * 2, hipMemcpyHostToDevice));
  CHECK(hipMemcpy(d_u, u.data(), u_size * 2, hipMemcpyHostToDevice));

  ConvParams p;
  p.H = H; p.W = W; p.C = C; p.M = M;
  p.M_pad = (M + 31) / 32 * 32; p.C_pad = (C + 15) / 16 * 16;
  p.tiles_w = W / 2; p.do_silu = 0; p.do_add = 0;
  const uint32_t nt_total = (H / 2) * (W / 2);
  const uint32_t nt_wg = (nt_total + 15) / 16;

  if (getenv("VDUMP")) {
    v_dump<<<nt_wg, 128>>>(d_in, d_u, d_out, p);
    CHECK(hipDeviceSynchronize());
    std::vector<_Float16> vd(256);
    CHECK(hipMemcpy(vd.data(), d_out, 512, hipMemcpyDeviceToHost));
    // host-side V: compute V[0][c=0..3][nt=0..3] from the strip
    printf("kernel v_lds[0..15]: ");
    for (int i = 0; i < 16; ++i) printf("%8.3f ", (float)vd[i]);
    printf("\n");
    printf("host   v[wp0][c0][nt0..15]: ");
    for (uint n = 0; n < 16; ++n) {
      int h0 = (n >= 8) ? 2 : 0, w0 = 2 * (int)(n & 7);
      _Float16 d4[16];
      for (uint i = 0; i < 4; ++i)
        for (uint j = 0; j < 4; ++j) {
          int h = h0 - 1 + (int)i, w = w0 - 1 + (int)j;
          d4[i*4+j] = (h >= 0 && h < H && w >= 0 && w < W) ? in[((size_t)h*W+w)*C] : 0.0f;
        }
      _Float16 v4[16];
      winograd_in4_host(d4, v4);
      printf("%8.3f ", (float)v4[0]);
    }
    printf("\n");
    printf("kernel A (U ko0 wp0): ");
    for (int i = 0; i < 16; ++i) printf("%8.3f ", (float)vd[32+i]);
    printf("\nkernel m0 (U0 x V wp0 nt0): ");
    for (int i = 0; i < 8; ++i) printf("%8.3f ", (float)vd[64+i]);
    printf("\nkernel B (V wp0 c0..15 nt0): ");
    for (int i = 0; i < 16; ++i) printf("%8.3f ", (float)vd[96+i]);
    printf("\n");
  }
  winograd_conv<<<nt_wg, 128>>>(d_in, d_u, d_out, d_in, p);
  CHECK(hipGetLastError());
  CHECK(hipDeviceSynchronize());
  CHECK(hipMemcpy(out.data(), d_out, out.size() * 2, hipMemcpyDeviceToHost));

  float maxerr = 0.f;
  int worst = -1;
  for (size_t i = 0; i < out.size(); ++i) {
    float e = fabsf((float)out[i] - (float)ref[i]);
    if (e > maxerr) { maxerr = e; worst = (int)i; }
  }
  size_t bad = 0;
  for (size_t i = 0; i < out.size(); ++i)
    if (fabsf((float)out[i] - (float)ref[i]) > 0.01f) bad++;
  printf("HIDDIZEN conv: maxerr=%.5f at idx %d (h=%d w=%d m=%d), bad=%zu/%zu (%.2f%%)\n",
         maxerr, worst, worst / (W * M) / W, worst / M % W, worst % M, bad, out.size(),
         100.0 * bad / out.size());
  printf("out[0..7]:  ");
  for (int i = 0; i < 8; ++i) printf("%8.3f ", (float)out[i]);
  printf("\nref[0..7]:  ");
  for (int i = 0; i < 8; ++i) printf("%8.3f ", (float)ref[i]);
  printf("\n");
  CHECK(hipFree(d_in));
  CHECK(hipFree(d_u));
  CHECK(hipFree(d_out));
  return 0;
}
