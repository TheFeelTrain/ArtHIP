## ROADMAP — occupancy & latency (next working session)

Kernel is LATENCY-bound: 4.4 ms/conv vs 0.56 ms DRAM floor and 0.6 ms MMA
floor (68 GFLOP @ ~15 TFLOPS of ~113 peak). Compute util ~13%. Everything
below targets more resident waves / better hiding.

Measured facts (kern_probe.s, gfx1100):
- winograd_conv: VGPR=133, SGPR=68, spills=0, wavefront=32, WG=128 (4 wave32s).
  Waves/SIMD = floor(16384/(133*32)) = 3. OPEN Q: does a 128-thread WG's 4
  waves fit one SIMD at 133 regs (4*133*32=17024 > 16384)? If not, how does
  HW place them (cross-SIMD WG split within CU?) — verify achieved occupancy
  with rocprofv2 (SQ_AVG_WAVES_PER_SIMD) before/after each experiment.
- LDS 15.7 KB/WG (v_lds 12288 B c-major stride 24 + strip4 3456 B).
- Live-register inventory during WMMA: y00-y11 float8 x4 = 32,
  a0-a3 half16 x4 = 32, b0-b3 half16 x4 = 32, m0-m3 float8 x4 = 32
  => ~128 + addressing ~= 133. Fragments are the cuttable part.

Ideas, priority order (ONE change per build; validate vs ORT_VULKAN + bench):

R1. Fragment load-consume diet. Inner loop: per j in 0..3 -> load a_j (one
    half16_t copy), load b_j, wmma_mul(a_j,b_j,m_j), drop frags. Fragment
    liveness 96 -> ~24 regs; peak ~100 -> 5 waves/SIMD. Risk: serialized
    global A-load latency inside the wave (compensated by other waves' TLP).
    Watch: spills (scratch_size_bytes must stay 0), accuracy unchanged.
R2. Transposed v_lds [wp][nt][c] (pad 16, i.e. 512 B per wp-row block):
    - B fragment = ONE contiguous 32 B read per lane (conflict-free:
      lanes tile linearly over the row block),
    - V-transform stores become half4-wide (c4 group contiguous),
    - replaces 64 scalar ds_load_u16 + scalar stores per c-block.
    NOTE: first attempt on THIS raw-HIP kernel was CATASTROPHIC
    (5.8 fps, -65%; see TRIED table). Suspected LDS bank conflicts at 40 B
    row pitch were never confirmed by counter data — retry only WITH
    rocprofv2 LDS-conflict metrics, and consider smaller pads / swizzled
    XOR layouts. Do AFTER R1 (fewer things live during swap).
R3. Cross-cb A prefetch: issue cb+1 group-0 A loads just before current
    cb's WMMA drain (global ~400-800 cyc hides under WMMA). Costs ~32 live
    regs across the loop boundary — evaluate only after R1 lands (regs
    budget), else skip.
R4. launch_bounds sweep: (128,5)/(128,6)/(128,8) and no-launch-bounds.
    Fail signal: scratch_size_bytes > 0 (spills) or fps drop. Cheap tests.
R5. Strip loads widen: cells are 16 contiguous ch = 32 B; currently 4x b64
    stores/loads -> 2x b128. Minor, fold into any other edit.
R6. Direct-from-register writeback, SECOND ATTEMPT. Previously worked
    standalone (16.88 median, correct) then lost in the DTS-fusion tangle.
    D layout: lane L holds y[2e+L/16][L%16], ko=k_base+L/16+2e. Removes
    staging stores + barrier + scatter loop. Build as SEPARATE kernel
    (winograd_conv_dw) validated standalone vs staged before wiring.
R7. DTS fused into tail-conv writeback (after R6 works): saves DTS kernel
    (~0.37 ms) + intermediate traffic. Mapping now fully understood:
    DCR, channel ko = block pos (rh=ko>>1, rw=ko&1); corners y00/y01/y10/
    y11 = small pixels (oh0,ow0),(oh0,ow0+1),(oh0+1,ow0),(oh0+1,ow0+1);
    each lands at big[(2oh+rh)][(2ow+rw)], dims 2H x 2W single ch.
    Previous failure modes documented in TRIED table (E-series).
R8. LDS diet for 5th WG/CU: v_lds stride 24->20 (12->10 KB) etc. Only
    relevant alongside R1/R2 (VGPR must allow the extra WG too).
R9. Persistent/cooperative megakernel (fuse conv chain, grid.sync()):
    kills inter-kernel gaps + weight reloads. Large effort, low confidence
    — park unless R1/R2 show big wins.
R10. rocprofv2 pass FIRST next session: SQ_AVG_WAVES_PER_SIMD,
    SQ_INST_LDS_BANK_CONFLICT, dram throughput on one frame, PLUS a
    tmp/sample_gpu.sh pass DURING a HIP run (HIP busy%/clocks/power never
    sampled - the 89%/76% duty figures are VULKAN/MIGX-sampled). 30 min
    spent here de-risks every choice above.

## TRIED — results ledger

Format: change -> result (fps @1080p 500f interleaved unless noted).

KEPT (in shipping build):
+ A-fragment vector loads (half16_t bulk copy, 2x b128): neutral-to-+,
  fewer instructions. In build.
+ launch_bounds(128, 4): neutral (16.46/16.78/16.88). In build.
+ Clip folded into dts_kernel + engine parse fusion: 17.01 median
  (from 16.88). In build.
+ GPU cast_f32_to_f16 kernel replacing host convert loop: kept (part of
  fp32 fix; host loop was ~ms-scale).
+ Multi-engine num_streams round-robin: kept (no vspipe scaling — see
  num_streams analysis — but harmless and helps concurrent-request cases).

FAILED / REVERTED (do not blind-retry):
- Transposed v_lds [wp][nt][20] on THIS HIP kernel: 5.8 fps (-65%) +
  correct-but-unchecked values. Genuine HIP result (bench_ab.py already
  fixed at that point). Cause unconfirmed (bank model said mild); retry
  needs rocprof conflict counters. (The separate VULKAN transposed attempt
  was never benchmarked — it produced zeros and was abandoned.)
- Direct writeback + DTS tail fusion, first attempt: crash/garbage stack
  (consecutive-write scatter bug, missing blocksize carry, probe leftovers
  disabling cast + all launches). Fully reverted to staged writeback.
  Redo properly as R6/R7.
- KB1 wave64 variant (VULKAN): 9.34 fps (-35%): halves k-coverage per WG,
  doubles strip/V traffic per pixel.
- ROW geometry variant (VULKAN): 9.92 fps (-35%): same redundancy cause.
- Wave32 4-wave WG (VULKAN coopmat): 14.47 fps (-5%): RADV coopmat
  lowering unpacks fp16 fragments (8 VGPR/mat instead of 4) -> register
  bloat; raw-HIP equivalent is the SHIPPING kernel and doesn't suffer this.
- T-prefetch removal (VULKAN main kernel): 14.31 (-6%): prefetch earns
  its registers there.
- int8/int4 quantization: REJECTED pre-implementation — kernel is
  latency-bound at ~13% MMA utilization; 2x compute ceiling buys ~nothing;
  int8 Winograd numerically treacherous; revisit only if kernel becomes
  compute-bound (>60% WMMA util) after R1/R2.

## OPEN QUESTIONS

- Actual resident waves/SIMD for winograd_conv (rocprofv2) — 3 predicted
  from 133 VGPRs; verify, plus LDS bank-conflict counts for B gather.
- Why did vs_ab_check segfault intermittently at teardown in 4-backend
  same-process runs (migx+vulkan+hip loaded together)? HIP-only runs are
  stable. Suspect cross-runtime (ROCm+RADV) teardown ordering; not a
  plugin blocker but makes scripted validation flaky — prefer per-backend
  processes for automated checks.

# HIP NOTES - standalone HIP plugin + HIP execution provider

Working notes for `src/vapoursynth/hip/` (VapourSynth plugin) and `src/hip/`
(EP / winograd WMMA kernels). Vulkan-side history lives in
`../vulkan/OPTIMIZATION_NOTES.md`.

## Layout & Build

- Plugin: `src/vapoursynth/hip/{vs_hip.cpp, hip_engine.cc, hip_engine.h}`
  built by `src/vapoursynth/build_hip.sh`
  (`hipcc --offload-arch=gfx1100`, links system onnx + protobuf; no ORT).
- Kernels: `src/hip/hip_kernels.h` - native wave32 WMMA via
  `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32`; shared with `src/hip/`
  execution-provider code.
- Install: `cp src/vapoursynth/hip/build/libhip.so
  /usr/lib/python3.14/site-packages/vapoursynth/plugins/libhip.so`
- vsscale: `Backend.HIP` registered in
  `/usr/lib/python3.14/site-packages/vsscale/mlrt/backend/base.py`
  (`HIP = hip.HIP`; upstream hip.py existed but was unwired).
  NOTE: like the VULKAN entries, a vsjetpack update resets base.py -
  re-add the import + `HIP = hip.HIP` line after every update.
- Run: `MANGOHUD=0 VS_BACKEND=hip vspipe -p tests/vs_test.py --`
  (tests/vs_test.py already routes `hip` to Backend.HIP).

## Speed session (2026-09-06) - HIP 17.2-17.4 vs MIGX 15.7-15.8 (+10%)

Target: RTX 3080 TensorRT does 18-19 fps on this model (similar on-paper fp16
TFLOPS to the 7900 XTX), so headroom should exist. Method: rocprofv3 kernel
times (winograd 478ms/8 frames = 2.3ms true exec, not the 2.2ms event span),
HIP_CONV_SKIP phase toggles, VGPR probe via hsaco notes, ABA benchmarks.

SHIPPED (all in build, accuracy-gated vs MIGX on photo: max 0.0024, unchanged):
+ Tail-conv wave skip (kgroups<4 waves idle in WMMA/staging/writeback):
  neutral-to-+, kills duplicate-wave waste. KEPT.
+ -ffast-math (build_hip.sh): +2.4% (17.02 -> 17.45 ABA), identical accuracy
  (max 0.002441 both). KEPT.
+ fast_silu (degree-4 least-squares 2^f poly, 4 FMA + floor + ldexp + rcp,
  replaces 1/4-rate expf+div in conv epilogues): accuracy-gated (unchanged
  0.0024), ~neutral on fps (SiLU is writeback-fringe, not the bottleneck).
  KEPT for lower transcendental pressure.
+ DTS 1D-div/mod -> 2D-grid kernels: input-centric dts_kernel_in_f32 (one
  thread per INPUT px, coalesced half4 block load, scattered float stores)
  fuses the fp16->fp32 output cast, deleting the extra 8M-px cast pass.
  1.24ms -> 1.21ms event; GRAYH path uses dts_kernel_2d. KEPT.
+ Strip 4xhalf4 -> 1xhalf16_t vector copy: neutral, cleaner ISA. KEPT.

REJECTED (do not retry without new evidence):
- R1 fragment load-consume diet (per-j load+mma+fold, 105 VGPR 8 waves/SIMD):
  17.4 -> 15.6 fps (-11%). Memory-level parallelism (batched A/B loads
  feeding 4 back-to-back independent WMMAs) beats occupancy. The 4-acc
  pipeline hides the 32c WMMA latency; single-acc chains serialize it.
- R1b ping-pong (2 live accs, FOLD1): 15.5 fps (-14%). Same cause.
- Cross-wpi A prefetch buffer: 142 VGPR, -3%. The straight schedule is optimal.
- 4-row geometry (32 tiles/WG, shared 10x18 strip, 1.41x halo vs 1.69x):
  MISCOMPILED - global garbage (max 0.80 vs MIGX), identical bit-output
  across runs. Bisected: bb0-only isolation ALSO garbage, bb-local and
  global indices verified correct by inspection - points at a compiler/codegen
  issue with the restructured loop nest, not a plain indexing bug.
  REVERTED whole; kernel back at HEAD + fast_silu only.
- ds_read2-style B-gather widening: B already loads as 4x half16_t vector
  copies; further widening needs layout change (transposed v_lds was already
  -65% in TRIED table).

MEASURED FACTS (rocprofv3, 8 frames @1080p, sclk 2245MHz / 303W / 100% GPU):
- winograd: 208 dispatches, 478ms total = 2.30ms true exec each (26 convs x 8).
- direct_conv: 5.5ms total. dts_in_f32: 0.36ms total (~45us each).
- SKIP profile (steady-state event spans): full 2.17 / skip-WB 1.99 /
  skip-WMMA 0.83 / strip-only ~0.5 => strip+V ~0.5, WMMA ~1.2, WB ~0.2.
- VGPR (hsaco notes): winograd 134 (176 allocated = 6 waves/SIMD; the 128
  cliff needs <=128). LDS 15744 (4 WGs/CU). No spills anywhere.
- Frame math: 26 x 2.3 = 60ms compute, but 300 frames in ~17.3s = 57.7ms/frame
  wall. GPU 100% busy - the convs ARE the frame; ~10% over MIGX, ~8% under
  the 3080/TensorRT target.

NEXT (not tried, in priority order):
1. VGPR-bank fix (LLVM#204254: A/B/C on distinct banks or +2c/WMMA) - check
   disasm, near-free if it's allocation order. Then re-attack the 128 cliff
   ONLY via scheduling (not diets): shrink address/temporary VGPRs.
2. Fused tail writeback+DTS (R7-lite): tail conv writes DTS-swizzled output
   directly (saves the 45us DTS + one 8M intermediate). Needs care with the
   D-fragment corner mapping (previous E-series failures); build as a
   SEPARATE kernel validated standalone first.
3. Bigger levers if the kernel stalls: persistent megakernel (grid.sync,
   weights stay in L2 across 26 convs), or IMPLICIT-GEMM reformulation
   (CUTLASS-style, kills the Winograd transform + LDS traffic entirely).
4. Lock clocks for benchmarking (rocm-smi --setperflevel high); all numbers
   above at sustained 2245MHz but the card idles at 130MHz between runs.

## Output format (2026-09-06)

- The plugin used to output GRAYH (fp16) with fp16=true; it now matches the
  MIGX plugin: fp16 compute, fp32 clip IO (GRAYS in -> GRAYS out).
- `convert_float_to_float16(..., cast_input=clip_fp32, cast_output=clip_fp32)`
  (same rule as vsort's output_format logic); the engine consumes the
  boundary Casts (input_cast was already handled; output_cast sets
  `output_is_fp32_`).
- fp32 download is a `cast_f16_to_f32` kernel into a device fp32 buffer +
  fp32 D2H (NOT the old host HalfBitsToFloat loop, which is gone - 8M
  px/frame would be ms-scale on CPU). GRAYH/int clips still output GRAYH.
- Verified: GRAYS->GrayS, GRAYH->GrayH; photo max absdiff vs MIGX 0.0024
  (fp16 rounding class); 120f @1080p HIP 16.26 vs MIGX 15.30 fps (no
  regression - still ahead).

## Correctness (2026-08-25/26 session)

Same fp32-model bug the VULKAN plugin had - fixed the same way plus extras:
- `convert_float_to_float16(model, force=true, {}, cast_input=clip_is_fp32,
  cast_output=false)` at load time when fp16=true.
- Consolidated detect+consume pass for boundary Cast nodes in
  HipEngine::Build (detection rewrites input_name_/output_name_, so a later
  name-based consumed[] pass misses converter-inserted casts ->
  "unsupported op 'Cast'").
- Multi-engine num_streams: N HipEngines round-robin via atomic index,
  per-engine stream + internal mutex.
- Dtype-aware packing in hipGetFrame: raw fp32 passthrough when
  eng->InputIsFp32(), fp16 otherwise. GOTCHA: branch must key on
  InputIsFp32() x clip format, NOT clip format alone (GRAYS clip + already-
  fp16 model would overflow a raw-fp32 memcpy). Same fix applied to the
  VULKAN plugin.
- Device-side cast: `cast_f32_to_f16` kernel + pinned/device fp32 staging
  buffers replace the old per-pixel host convert.
  GOTCHA: `input_staging_f32_` must be allocated in EnsureBuilt - Run()
  memcpys into it unconditionally on the fp32 path (NULL = segfault).
- ACCURACY verified: vs MIGX max absdiff 0.0055; vs ORT_VULKAN/VULKAN
  0.0034 (fp16 rounding class). All four backends agree.

## Performance

FINAL BUILD (2026-08-26: staged writeback + A-fragment vector loads +
launch_bounds(128,4) + Clip-in-DTS fusion), 500 frames @1080p:

    HIP     17.18 / 17.16 / 17.06   (median 17.16)
    MIGX    ~16.6      VULKAN ~15.3

Earlier fully-correct build (pre-clip-fusion) interleaved x3 round, kept
for its paired per-run numbers: HIP 16.93/16.73/16.41,
MIGX 16.68/16.64/16.01, VULKAN 15.23/15.31/15.26.

HIP > MIGX in every paired run, and no desktop jank (unlike VULKAN).

BENCHMARK GOTCHA: early "hip 15.38" numbers were INVALID - tmp/bench_ab.py
routed every non-migx backend to VULKAN until fixed. Always verify
`Using 'HIP' backend` appears in stderr when benchmarking a backend.

### num_streams semantics
- vspipe DOES issue overlapping GetFrame calls (VSHIP_TRACE shows multiple
  frames in flight; both engines used).
- vsmigx scales +15% with num_streams because its s=1 leaves GPU gaps:
  MIGX s=1 15.03 -> s=2 17.33 under the identical harness.
- DUTY-CYCLE ATTRIBUTION (do not cross wires): GPU busy/clock/power was
  sampled via tmp/sample_gpu.sh ONLY during VULKAN-vs-MIGX runs -
  VULKAN ~89% busy / 2057 MHz / 287 W avg vs MIGX ~76% / 1684 MHz /
  248 W. HIP busy% has NEVER been sampled; do not quote 89% for HIP.
  Fold into the R10 rocprofv2/sysfs pass.
- Design reference if ever needed: TicketSemaphore + instances vector in
  reference/vs-mlrt/vsmigx/vs_migraphx.cpp.

### Debug/env knobs (all default off)
- `VSHIP_PROFILE=1` per-op HIP event timestamps in HipEngine::Run.
  Steady profile @1080p: winograd convs ~4.4 ms each x26 (flat, unlike
  VULKAN's tapering profile); DTS 0.37ms; Clip 0.72ms; direct conv 0.66ms.
  (Event sums exceed wall time - includes inter-op gaps/warmup bias; trust
  shape, not absolutes.)
- `VSHIP_TRACE=1` per-frame engine enter/leave lines (overlap analysis).
- `VSHIP_DEBUG=1` fp32-path diagnostics (memcpy/launch rc, input dump).

### Kernel tuning log (winograd_conv in src/hip/hip_kernels.h)
Roofline @1080p per 64->64 conv: ~68 GFLOP winograd => 4.4 ms measured =
~15 TFLOPS (gfx11 fp16 WMMA peak ~113 T) => 13% of peak; DRAM floor
~0.56 ms (506 MB in+out) => 8x above bandwidth floor. LATENCY-bound, not
compute- or bandwidth-bound. CAVEAT: the 4.4 ms is VSHIP_PROFILE EVENT
timing, which sums above wall time (queue/inter-op wait included); true
exec could be ~half => util maybe ~27%. Conclusion direction unchanged,
but re-derive from R10 counter data before citing 13% anywhere.
Consequence: INT8/INT4 quantization attacks
the wrong bottleneck (2x compute ceiling buys ~nothing at 13% MMA util;
int8 Winograd is numerically treacherous anyway). Revisit ONLY after
occupancy/latency work pushes the kernel toward compute-bound.

ISA findings (hipcc -S gfx1100 via tmp/kern_probe.hip.cpp -> kern_probe.s):
- A-fragments lowered to 56 scalar global_load_d16_b16 per c-block even
  though each lane's 16 halfs are 32 contiguous bytes.
- V-transform stores to v_lds + B-fragment loads: all scalar u16 LDS ops
  in the original c-major layout.

Tuning experiments (500f @1080p, interleaved x3):
1. Vectorized A loads (half16_t copy = 2x b128): CORRECT, kept.
2. __launch_bounds__(128, 4): neutral, kept as documentation/protection.
3. Clip folded into dts_kernel (DtsParams +do_clip/min/max; engine parse
   fuses a following Clip into the DepthToSpace op and rewrites its output
   to the model output tensor): CORRECT, 17.01 median. KEPT.
4. Transposed v_lds [wp][nt][20] for contiguous B loads: CATASTROPHIC
   (5.8 fps). REVERTED - do not retry without rocprof LDS conflict data.
5. Direct-from-register writeback + DTS fused into tail conv writeback:
   implemented but REVERTED after repeated correctness failures (corner/
   channel mapping subtleties in the D fragment layout; multiple distinct
   bugs stacked: consecutive-write scatter, missing blocksize carry,
   probe leftovers silently disabling the cast kernel and all launches).
   The staged writeback stays. If retried, build it as a SEPARATE kernel
   file validated standalone first - do not patch inline.

STANDING RESULT (2026-08-26 final): HIP 17.18/17.16/17.06 fps (median
17.16) vs MIGX ~16.6 vs VULKAN ~15.3. Correct output verified vs MIGX
(0.0055 max) and ORT_VULKAN/VULKAN (0.0034 max).

DEBUG GOTCHAS that cost time tonight (avoid repeating):
- `./build_X.sh | tail` masks compile errors -> stale .so installed.
  Check exit code, not output tail.
- python str.replace edits SILENTLY no-op on anchor mismatch - assert
  every replacement.
- After any crash round, diff the kernel against git HEAD to find
  leftover PROBE/disabled blocks before trusting a benchmark.
- tmp/bench_ab.py originally routed non-migx backends to VULKAN; verify
  'Using <X> backend' in stderr.
