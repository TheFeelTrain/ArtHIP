// HIP device kernels for the HIPExecutionProvider.
//
// winograd_conv: F(2x2,3x3) Winograd WMMA conv (gfx1100 native wave32).
//   WG = 128 threads = 4 wave32s; KB=1; each wave owns one k-block (16 ko).
//   16 nt tiles per WG (2 tile rows), 6x18x16c input strip in LDS.
//   WMMA: A = U rows (half16_t, lane L = row L%16), B = V columns from LDS,
//   D = float8_t accumulators (fp32 fold, fp16 only at writeback).
// direct_conv: plain 3x3 SAME conv for C==1 first convs (no WMMA).
// unary_kernel / binary_kernel / dts_kernel: pointwise ops + DepthToSpace.
#pragma once

#include <hip/hip_runtime.h>
#include <cstdint>

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float __attribute__((ext_vector_type(8))) float8_t;
typedef float __attribute__((ext_vector_type(4))) float4_t;

struct ConvParams {
  uint32_t H, W, C, M, M_pad, C_pad, tiles_w, do_silu, do_add;
};

// Wave32 KB=1 Winograd F(2x2,3x3) WMMA conv - wave32 conversion of the
// proven Vulkan EP kernel (winograd_wmma.comp, 1.02% @256). WG = 128 threads
// = 4 wave32s; each wave owns ONE k-block (16 ko) -> the whole M=64 sits in
// one WG (strip loaded once, used by all 4 k-blocks; no L2 strip sharing
// needed). Verified w32 fragments (wmma_bench, exact):
//   A/B: lane l holds row l%16, all 16 k        (a[e] = A[l%16][e])
//   D:   lane l, element e: D[row = 2e + (l/16)][col = l%16]  (8 fp32)
// Vectorized like the Vulkan EP: strip loaded as half4 (b64), V transform on
// half4 (4 channels/ALU op), writeback staged col-major into v_lds and read
// back as vec4 (8-byte LDS reads + 8-byte output stores).
typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef _Float16 __attribute__((ext_vector_type(4))) half4_t;
typedef float __attribute__((ext_vector_type(8))) float8_t;

__device__ __forceinline__ void wmma_mul(half16_t a, half16_t b, float8_t& d)
{
  d = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, d);
}

// Fast SiLU: sigmoid via a degree-4 least-squares 2^f polynomial (max err
// ~8e-6 on 2^f over [0,1) => ~1e-4 on SiLU). Costs ~4 FMA + floor + ldexp +
// 1 rcp instead of libm expf + div (both 1/4-rate transcendentals). No
// range branches: out-of-range t over/underflows ldexp to INF/0, which
// yields the correct limits (0 for x->-inf, x for x->+inf).
__device__ __forceinline__ float fast_silu(float x)
{
  float t = -x * 1.44269502f;
  float nf = floorf(t);
  float f = t - nf;
  float p = (((0.01367415f * f + 0.05167156f) * f + 0.24170720f) * f + 0.69293204f) * f + 1.00000723f;
  float e = ldexpf(p, (int)nf);
  return x / (1.0f + e);
}

// 4-channel Winograd input transform (half4 math, same as Vulkan shader).
__device__ __forceinline__ void winograd_in4_v4(const half4_t* d, half4_t* v)
{
  half4_t t0  = d[0] - d[2];   half4_t t1  = d[1] + d[2];   half4_t t2  = d[2] - d[1];   half4_t t3  = d[1] - d[3];
  half4_t t4  = d[4] - d[6];   half4_t t5  = d[5] + d[6];   half4_t t6  = d[6] - d[5];   half4_t t7  = d[5] - d[7];
  half4_t t8  = d[8] - d[10];  half4_t t9  = d[9] + d[10];  half4_t t10 = d[10] - d[9];  half4_t t11 = d[9] - d[11];
  half4_t t12 = d[12] - d[14]; half4_t t13 = d[13] + d[14]; half4_t t14 = d[14] - d[13]; half4_t t15 = d[13] - d[15];
  v[0]  = t0 - t8;   v[1]  = t1 - t9;   v[2]  = t2 - t10;  v[3]  = t3 - t11;
  v[4]  = t4 + t8;   v[5]  = t5 + t9;   v[6]  = t6 + t10;  v[7]  = t7 + t11;
  v[8]  = t8 - t4;   v[9]  = t9 - t5;   v[10] = t10 - t6;  v[11] = t11 - t7;
  v[12] = t4 - t12;  v[13] = t5 - t13;  v[14] = t6 - t14;  v[15] = t7 - t15;
}

// Load a lane's half16 B-fragment from the transposed V store: the 16 halfs
// are contiguous (c-major within the row), so this is 4x ds_load_b64 instead
// of 16 scalar u16 gathers. Caller guarantees 8-byte alignment (row stride 20
// halfs = 40 B, plus a 16-byte-aligned base).
__device__ __forceinline__ half16_t lds_v16(const _Float16* p)
{
  union { half16_t v; half4_t q[4]; } u;
  u.q[0] = *reinterpret_cast<const half4_t*>(p + 0);
  u.q[1] = *reinterpret_cast<const half4_t*>(p + 4);
  u.q[2] = *reinterpret_cast<const half4_t*>(p + 8);
  u.q[3] = *reinterpret_cast<const half4_t*>(p + 12);
  return u.v;
}

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

#define FOLD4(Y00, Y01, Y10, Y11, w, m0, m1, m2, m3)                           \
  {                                                                            \
    float8_t s0 = m0 + m1 + m2;                                                \
    float8_t s1 = m1 - m2 - m3;                                                \
    if (w == 0) { Y00 = Y00 + s0; Y01 = Y01 + s1; }                            \
    else if (w == 1) { Y00 = Y00 + s0; Y01 = Y01 + s1; Y10 = Y10 + s0; Y11 = Y11 + s1; } \
    else if (w == 2) { Y00 = Y00 + s0; Y01 = Y01 + s1; Y10 = Y10 - s0; Y11 = Y11 - s1; } \
    else { Y10 = Y10 - s0; Y11 = Y11 - s1; }                                   \
  }

__global__ void winograd_conv(
    const _Float16* __restrict__ in, const _Float16* __restrict__ u,
    _Float16* __restrict__ out, const _Float16* __restrict__ add,
    ConvParams p)
{
  const uint wave = threadIdx.x / 32;
  const uint lane = threadIdx.x % 32;
  const uint wg = blockIdx.x;

  const uint tiles_w8 = p.tiles_w / 8u;
  const uint kgroups = p.M_pad / 16u;
  const uint kgroup = min(wave, kgroups - 1u);
  const uint row_pair = wg / tiles_w8;
  const uint tw_group = wg % tiles_w8;
  const uint th_a = row_pair * 2u;
  const uint th_b = th_a + 1u;
  const uint tw0 = tw_group * 8u;
  const uint k_base = kgroup * 16u;
  const uint Cblocks = p.C_pad / 16u;
  const uint c_blocks = Cblocks;

  __shared__ _Float16 strip4[6 * 18 * 16];
  // Transposed V store: [wp][nt][c] with c-row stride 20 (c=0..15 used, 4 pad).
  // Row starts are 40 B apart => 8-byte aligned, so each lane's 16-c fragment
  // is 4x ds_load_b64 (aligned) instead of 16 scalar u16 gathers. 16-byte base
  // alignment is asserted so the pair-loads never straddle a bank-quad edge.
  __shared__ __attribute__((aligned(16))) _Float16 v_lds[16 * 16 * 20];

  float8_t y00 = {}, y01 = {}, y10 = {}, y11 = {};

  auto load_strip = [&](uint cb) {
    const uint c_b = cb * 16u;
    const uint p0 = wave * 27u;
    for (uint pc = p0 + lane; pc < p0 + 27u; pc += 32u) {
      const int h = int(2u * th_a) - 1 + int(pc / 18u);
      const int w = int(2u * tw0) - 1 + int(pc % 18u);
      const uint s4 = pc * 16u;
      if (c_b + 15u < p.C && h >= 0 && h < int(p.H) && w >= 0 && w < int(p.W)) {
        // 16 contiguous fp16 channels = 4 b64 half4 loads (gbase 16-aligned).
        const uint gbase = (uint(h) * p.W + uint(w)) * p.C + c_b;
        const half4_t* src = reinterpret_cast<const half4_t*>(in + gbase);
        half4_t* dst = reinterpret_cast<half4_t*>(strip4 + s4);
        dst[0] = src[0];
        dst[1] = src[1];
        dst[2] = src[2];
        dst[3] = src[3];
      } else {
        const bool spatial_in = (h >= 0 && h < int(p.H) && w >= 0 && w < int(p.W));
        for (uint q = 0u; q < 16u; ++q) {
          _Float16 val = 0.0f;
          if (spatial_in && c_b + q < p.C) val = in[(uint(h) * p.W + uint(w)) * p.C + c_b + q];
          strip4[s4 + q] = val;
        }
      }
    }
  };

  load_strip(0u);

    for (uint cb = 0u; cb < Cblocks; ++cb) {
    const uint cb_off = cb * 256u;

    __syncthreads();  // strip (cb) visible

    // ---- V transform (strip -> v_lds), 64 idx split across 4 waves ----
    {
      const uint i0 = wave * 16u;
      for (uint idx = i0 + lane; idx < i0 + 16u; idx += 32u) {
        const uint nt_local = idx / 4u;
        const uint c4 = (idx % 4u) * 4u;
        const uint row_off = (nt_local >= 8u) ? 2u : 0u;
        const uint tw_local = nt_local & 7u;
        // half4 loads + ALU; stores stay scalar (v_lds is c-major, stride 24).
        half4_t d4[16];
        for (uint i = 0u; i < 4u; ++i)
          for (uint j = 0u; j < 4u; ++j) {
            const uint s4 = ((row_off + i) * 18u + (2u * tw_local + j)) * 16u + c4;
            d4[i * 4u + j] = *reinterpret_cast<half4_t*>(strip4 + s4);
          }
        half4_t v4[16];
        winograd_in4_v4(d4, v4);
        // Store the 16 wp values as 16 x 8-byte LDS writes (c4..c4+3 are
        // contiguous in the transposed [wp][nt][c] layout) instead of 64
        // scalar u16 stores.
        for (uint wp = 0u; wp < 16u; ++wp)
          *reinterpret_cast<half4_t*>(v_lds + wp * 320u + nt_local * 20u + c4) = v4[wp];
      }
    }
    __syncthreads();  // v_lds ready

    // ---- WMMA: 4 wp groups, A from U (this wave's k-block), B from v_lds ----
    // A: lane l holds U[ko=l%16][c=0..15] for (kblock=kgroup, wp, cb)
    const uint ub = kgroup * (c_blocks * 4096u) + cb_off * 4u;
#pragma unroll 1
    for (uint wpi = 0u; wpi < 4u; ++wpi) {
      half16_t a0, a1, a2, a3;
      {
        // Each lane's fragment is 32 contiguous bytes in U: one vector copy
        // (lowers to 2x global_load_b128 instead of 16 scalar loads).
        const uint abase = ub + wpi * (c_blocks * 1024u) + (lane % 16u) * 16u;
        a0 = *reinterpret_cast<const half16_t*>(u + abase + 0u * 256u);
        a1 = *reinterpret_cast<const half16_t*>(u + abase + 1u * 256u);
        a2 = *reinterpret_cast<const half16_t*>(u + abase + 2u * 256u);
        a3 = *reinterpret_cast<const half16_t*>(u + abase + 3u * 256u);
      }
      // B fragments: lane l holds V[wp][c=0..15][nt=l%16]. Transposed store
      // makes those 16 c values one contiguous row -> 4 aligned b64 LDS reads
      // per matrix (conflict-free: nt*40 B rows map to distinct bank-quads).
      half16_t b0, b1, b2, b3;
      {
        const uint bnt = (lane % 16u) * 20u;
        b0 = lds_v16(v_lds + (wpi * 4u + 0u) * 320u + bnt);
        b1 = lds_v16(v_lds + (wpi * 4u + 1u) * 320u + bnt);
        b2 = lds_v16(v_lds + (wpi * 4u + 2u) * 320u + bnt);
        b3 = lds_v16(v_lds + (wpi * 4u + 3u) * 320u + bnt);
      }
      float8_t m0 = {}, m1 = {}, m2 = {}, m3 = {};
      wmma_mul(a0, b0, m0);
      wmma_mul(a1, b1, m1);
      wmma_mul(a2, b2, m2);
      wmma_mul(a3, b3, m3);
      FOLD4(y00, y01, y10, y11, wpi, m0, m1, m2, m3);
    }

    // Prefetch the next c-block's strip; its global-load latency overlaps the
    // WMMA drain (strip4 is free: the current strip was consumed above).
    if (cb + 1u < Cblocks) load_strip(cb + 1u);
  }

  __syncthreads();  // all waves done with v_lds

  // ---- Output transform + writeback (LDS-staged, vec4) ----
  // Stage the 4 y matrices ColumnMajor into this wave's v_lds region:
  // element (ko, nt) at nt*16 + ko, so each writeback vec4 (4 consecutive ko)
  // is one contiguous 8-byte LDS read. 4 matrices x 256 fp16 = 1KB per wave.
  {
    const uint nt = lane % 16u;
    const uint koff = k_base + lane / 16u;
    const uint yb = wave * 1024u;
    const uint base = yb + nt * 16u;
    for (uint e = 0u; e < 8u; ++e) {
      const uint ko = koff + 2u * e;
      if (ko >= p.M) break;
      const uint kol = ko - k_base;  // local row (the readback reads 0..15)
      v_lds[base + 0u * 256u + kol] = (_Float16)y00[e];
      v_lds[base + 1u * 256u + kol] = (_Float16)y01[e];
      v_lds[base + 2u * 256u + kol] = (_Float16)y10[e];
      v_lds[base + 3u * 256u + kol] = (_Float16)y11[e];
    }
  }
  __syncthreads();

  // 64 (nt, k4) combos per wave; each lane handles 2.
  const uint yb = wave * 1024u;
  for (uint idx = lane; idx < 64u; idx += 32u) {
    const uint nt_local = idx / 4u;
    const uint k4 = idx % 4u;
    const uint ko0 = k4 * 4u;
    const uint th = (nt_local < 8u) ? th_a : th_b;
    const uint tw = tw0 + (nt_local & 7u);
    const uint oh0 = 2u * th;
    const uint ow0 = 2u * tw;
    const uint o = (oh0 * p.W + ow0) * p.M;
    const uint k = k_base + ko0;
    if (k >= p.M) continue;
    const uint ntb = yb + nt_local * 16u + ko0;
    half4_t v00, v01, v10, v11;
    for (int q = 0; q < 4; ++q) {
      v00[q] = v_lds[ntb + 0u * 256u + q];
      v01[q] = v_lds[ntb + 1u * 256u + q];
      v10[q] = v_lds[ntb + 2u * 256u + q];
      v11[q] = v_lds[ntb + 3u * 256u + q];
    }
    // Bias: 4 consecutive fp16 at the end of U.
    half4_t bias;
    for (int q = 0; q < 4; ++q) bias[q] = u[p.M_pad * c_blocks * 256u + k + q];
    v00 += bias;
    v01 += bias;
    v10 += bias;
    v11 += bias;
    if (p.do_silu) {
      for (int q = 0; q < 4; ++q) {
        v00[q] = (_Float16)fast_silu((float)v00[q]);
        v01[q] = (_Float16)fast_silu((float)v01[q]);
        v10[q] = (_Float16)fast_silu((float)v10[q]);
        v11[q] = (_Float16)fast_silu((float)v11[q]);
      }
    }
    const bool wg_interior = (2u * th_b + 1u < p.H) && (2u * tw0 + 31u < p.W);
    const uint o1 = o + p.M;
    const uint o2 = o + p.M * p.W;
    const uint o3 = o + p.M * (p.W + 1u);
    if (wg_interior) {
      // Fast path: all 4 output pixels in bounds; only the k guard remains.
      if (p.do_add) {
        for (int q = 0; q < 4; ++q) {
          v00[q] = (_Float16)((float)v00[q] + (float)add[o + k + q]);
          v01[q] = (_Float16)((float)v01[q] + (float)add[o1 + k + q]);
          v10[q] = (_Float16)((float)v10[q] + (float)add[o2 + k + q]);
          v11[q] = (_Float16)((float)v11[q] + (float)add[o3 + k + q]);
        }
      }
      if (k + 3u < p.M) {
        *reinterpret_cast<half4_t*>(out + o + k) = v00;
        *reinterpret_cast<half4_t*>(out + o1 + k) = v01;
        *reinterpret_cast<half4_t*>(out + o2 + k) = v10;
        *reinterpret_cast<half4_t*>(out + o3 + k) = v11;
      } else {
        for (uint q = 0u; q < 4u; ++q) {
          if (k + q < p.M) {
            out[o + k + q] = v00[q];
            out[o1 + k + q] = v01[q];
            out[o2 + k + q] = v10[q];
            out[o3 + k + q] = v11[q];
          }
        }
      }
    } else {
      if (oh0 < p.H && ow0 < p.W) {
        if (p.do_add) for (int q = 0; q < 4; ++q) v00[q] = (_Float16)((float)v00[q] + (float)add[o + k + q]);
        if (k + 3u < p.M) *reinterpret_cast<half4_t*>(out + o + k) = v00;
        else for (uint q = 0u; q < 4u; ++q) if (k + q < p.M) out[o + k + q] = v00[q];
      }
      if (oh0 < p.H && ow0 + 1u < p.W) {
        if (p.do_add) for (int q = 0; q < 4; ++q) v01[q] = (_Float16)((float)v01[q] + (float)add[o1 + k + q]);
        if (k + 3u < p.M) *reinterpret_cast<half4_t*>(out + o1 + k) = v01;
        else for (uint q = 0u; q < 4u; ++q) if (k + q < p.M) out[o1 + k + q] = v01[q];
      }
      if (oh0 + 1u < p.H && ow0 < p.W) {
        if (p.do_add) for (int q = 0; q < 4; ++q) v10[q] = (_Float16)((float)v10[q] + (float)add[o2 + k + q]);
        if (k + 3u < p.M) *reinterpret_cast<half4_t*>(out + o2 + k) = v10;
        else for (uint q = 0u; q < 4u; ++q) if (k + q < p.M) out[o2 + k + q] = v10[q];
      }
      if (oh0 + 1u < p.H && ow0 + 1u < p.W) {
        if (p.do_add) for (int q = 0; q < 4; ++q) v11[q] = (_Float16)((float)v11[q] + (float)add[o3 + k + q]);
        if (k + 3u < p.M) *reinterpret_cast<half4_t*>(out + o3 + k) = v11;
        else for (uint q = 0u; q < 4u; ++q) if (k + q < p.M) out[o3 + k + q] = v11[q];
      }
    }
  }
}

// Direct 3x3 SAME conv for the model's FIRST conv (C == 1, M == 64).
// Weights re-laid out by the host as [t][c][co] fp16 (t = r*3+cc, c = input
// channel, co = output channel), bias appended at 9*C*M + co.
// Input/output are fp16 NHWC. WG = 8 px x all 64 co; lane = px*8 + co8.
__global__ __launch_bounds__(64) void direct_conv(
    const _Float16* __restrict__ in, const _Float16* __restrict__ u,
    _Float16* __restrict__ out, const _Float16* __restrict__ add,
    ConvParams p)
{
  const uint lane = threadIdx.x;
  const uint px = lane >> 3u;
  const uint co8 = lane & 7u;
  const uint co = co8 * 8u;
  const uint x0 = blockIdx.x * 8u;
  const uint y = blockIdx.y;

  float4_t y0 = {}, y1 = {};
  const uint bias_off = 9u * p.C * p.M + co;
  if (co + 7u < p.M) {
    half4_t b0 = *reinterpret_cast<const half4_t*>(u + bias_off);
    half4_t b1 = *reinterpret_cast<const half4_t*>(u + bias_off + 4u);
    for (int q = 0; q < 4; ++q) { y0[q] = (float)b0[q]; y1[q] = (float)b1[q]; }
  } else {
    for (int q = 0; q < 8; ++q) {
      if (co + q >= p.M) break;
      const float b = (float)u[bias_off + q];
      if (q < 4) y0[q] = b; else y1[q - 4] = b;
    }
  }
  for (uint t = 0; t < 9; ++t) {
    const int gh = int(y) - 1 + int(t / 3u);
    const int gw = int(x0) + int(px) - 1 + int(t % 3u);
    float x = 0.0f;
    if (gh >= 0 && gh < int(p.H) && gw >= 0 && gw < int(p.W)) {
      const uint g = uint(gh) * p.W + uint(gw);
      x = (float)in[g * p.C];
    }
    const uint wb = t * p.M + co;
    if (co + 7u < p.M) {
      half4_t w0 = *reinterpret_cast<const half4_t*>(u + wb);
      half4_t w1 = *reinterpret_cast<const half4_t*>(u + wb + 4u);
      for (int q = 0; q < 4; ++q) {
        y0[q] += (float)w0[q] * x;
        y1[q] += (float)w1[q] * x;
      }
    } else {
      for (int q = 0; q < 8; ++q) {
        if (co + q >= p.M) break;
        const float wv = (float)u[wb + q];
        if (q < 4) y0[q] += wv * x; else y1[q - 4] += wv * x;
      }
    }
  }
  if (p.do_silu) {
    for (int q = 0; q < 4; ++q) {
      y0[q] = fast_silu(y0[q]);
      y1[q] = fast_silu(y1[q]);
    }
  }
  if (x0 + px < p.W) {
    if (p.do_add) {
      const uint ab = (y * p.W + x0 + px) * p.M + co;
      if (co + 7u < p.M) {
        half4_t a0 = *reinterpret_cast<const half4_t*>(add + ab);
        half4_t a1 = *reinterpret_cast<const half4_t*>(add + ab + 4u);
        for (int q = 0; q < 4; ++q) { y0[q] += (float)a0[q]; y1[q] += (float)a1[q]; }
      } else {
        for (int q = 0; q < 4; ++q) {
          y0[q] += (float)add[ab + q];
          y1[q] += (float)add[ab + q + 4];
        }
      }
    }
    const uint ob = (y * p.W + x0 + px) * p.M + co;
    if (co + 7u < p.M) {
      half4_t o0 = {(_Float16)y0[0], (_Float16)y0[1], (_Float16)y0[2], (_Float16)y0[3]};
      half4_t o1 = {(_Float16)y1[0], (_Float16)y1[1], (_Float16)y1[2], (_Float16)y1[3]};
      *reinterpret_cast<half4_t*>(out + ob) = o0;
      *reinterpret_cast<half4_t*>(out + ob + 4u) = o1;
    } else {
      for (int q = 0; q < 8; ++q) {
        if (co + q < p.M) out[ob + q] = (_Float16)((q < 4) ? y0[q] : y1[q - 4]);
      }
    }
  }
}

// Pointwise unary: op 0 = sigmoid, 1 = clip(min,max).
struct UnaryParams {
  uint32_t numel;
  uint32_t op;
  float min_val;
  float max_val;
};

__global__ __launch_bounds__(256) void unary_kernel(
    const _Float16* __restrict__ in, _Float16* __restrict__ out, UnaryParams p)
{
  uint i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= p.numel) return;
  float x = (float)in[i];
  float y = (p.op == 0u) ? 1.0f / (1.0f + expf(-x)) : fminf(fmaxf(x, p.min_val), p.max_val);
  out[i] = (_Float16)y;
}

// Pointwise binary: op 0 = add, 1 = mul.
struct BinaryParams {
  uint32_t numel;
  uint32_t op;
};

__global__ __launch_bounds__(256) void binary_kernel(
    const _Float16* __restrict__ a, const _Float16* __restrict__ b,
    _Float16* __restrict__ out, BinaryParams p)
{
  uint i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= p.numel) return;
  float av = (float)a[i];
  float bv = (float)b[i];
  float y = (p.op == 1u) ? (av * bv) : (av + bv);
  out[i] = (_Float16)y;
}

// DepthToSpace DCR in NHWC: out[oh][ow][c] = in[h][w][c*B*B + r*B + q]
// with r = oh % B, q = ow % B, h = oh / B, w = ow / B. C = output channels.
struct DtsParams {
  uint32_t c, h, w, b;
  uint32_t do_clip;
  float min_val, max_val;
};

__global__ __launch_bounds__(256) void dts_kernel(
    const _Float16* __restrict__ in, _Float16* __restrict__ out, DtsParams p)
{
  uint idx = blockIdx.x * blockDim.x + threadIdx.x;
  uint total = p.h * p.w * p.b * p.b * p.c;
  if (idx >= total) return;
  uint c = idx % p.c;
  uint t = idx / p.c;
  uint ow = t % (p.w * p.b);
  uint oh = t / (p.w * p.b);
  uint r = oh % p.b;
  uint q = ow % p.b;
  uint h = oh / p.b;
  uint w = ow / p.b;
  uint cin = c * p.b * p.b + r * p.b + q;
  uint in_idx = (h * p.w + w) * (p.c * p.b * p.b) + cin;
  _Float16 v = in[in_idx];
  if (p.do_clip) v = (_Float16)fminf(fmaxf((float)v, p.min_val), p.max_val);
  out[idx] = v;
}

// 2D-grid DTS for the standalone plugin engine (the EP keeps dts_kernel above).
// ow/oh come straight from block/thread indices: zero div/mod in the common
// path (only h=oh/B, w=ow/B remain). C is looped (C==1 on the ArtCNN tail).
// _in_f32 is input-centric: one thread per INPUT pixel loads its whole
// B*B*C block as contiguous half4s (coalesced) and scatters float outputs -
// loads stall warps, scattered stores don't. It also fuses the fp16->fp32
// output cast (clip in fp32), so the tail needs no extra pass over 8M px.
// NOTE: dts_kernel_in_f32 assumes blocksize 2 (whole-block half4 load).
__global__ __launch_bounds__(256) void dts_kernel_2d(
    const _Float16* __restrict__ in, _Float16* __restrict__ out, DtsParams p)
{
  const uint ow = blockIdx.x * 32u + threadIdx.x;
  const uint oh = blockIdx.y * 8u + threadIdx.y;
  const uint Wout = p.w * p.b;
  const uint Hout = p.h * p.b;
  if (ow >= Wout || oh >= Hout) return;
  const uint h = oh / p.b;
  const uint w = ow / p.b;
  const uint r = oh - h * p.b;
  const uint q = ow - w * p.b;
  const uint cstride = p.c * p.b * p.b;
  const uint in_row = (h * p.w + w) * cstride;
  const uint out_row = (oh * Wout + ow) * p.c;
  for (uint c = 0u; c < p.c; ++c) {
    const uint in_idx = in_row + c * p.b * p.b + r * p.b + q;
    _Float16 v = in[in_idx];
    if (p.do_clip) v = (_Float16)fminf(fmaxf((float)v, p.min_val), p.max_val);
    out[out_row + c] = v;
  }
}

__global__ __launch_bounds__(256) void dts_kernel_in_f32(
    const _Float16* __restrict__ in, float* __restrict__ out, DtsParams p)
{
  const uint w = blockIdx.x * 32u + threadIdx.x;
  const uint h = blockIdx.y * 8u + threadIdx.y;
  if (w >= p.w || h >= p.h) return;
  const uint Wout = p.w * p.b;
  const uint bb = p.b * p.b;
  const uint in_base = (h * p.w + w) * (p.c * bb);
  for (uint c = 0u; c < p.c; ++c) {
    const half4_t* src = reinterpret_cast<const half4_t*>(in + in_base + c * bb);
    half4_t lv = src[0];
    for (uint k = 0u; k < bb; ++k) {
      const uint r = k / p.b;
      const uint q = k - r * p.b;
      float v = (float)lv[k];
      if (p.do_clip) v = fminf(fmaxf(v, p.min_val), p.max_val);
      out[((h * p.b + r) * Wout + (w * p.b + q)) * p.c + c] = v;
    }
  }
}
