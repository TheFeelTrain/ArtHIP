## CURRENT BEST — 2026-09-10: transposed v_lds SHIPPED (+19-20%)

vs-mlrt's libvsmigx.so was updated (2026-09-10) and MIGX now does 19.3-19.7 fps
on R8F64_Chroma @1080p (real jpbd source, `.scale(clip)` 1:1) vs HIP 18.6 — we
were LOSING. Baseline paired 100f x3 (tmp/bench_pair.sh chroma): hip 18.72/
18.68/18.58, migx 19.71/19.38/19.32. Base model: hip 18.43, migx 15.3 (MIGX
base is slower because R8F64 is a 2x model, so the luma path also writes a 4x
output + Catrom downscale; the two models are NOT equal-work — the fps "bug"
the user saw is that extra output stage, not a MIGX conv win).

FIX (KEPT): v_lds was c-major [wp][c][nt-stride20]; the WMMA B fragment needs a
column (16 c at fixed nt), which the compiler lowered to 64 ds_load_u16_d16 per
wpi. Transpose to [wp][nt][c-stride20] so each lane's 16 c are one contiguous,
8-byte-aligned row: B is now 4x ds_load_b64 per matrix (conflict-free — rows
40 B apart map nt=0..15 to 16 distinct even banks), 4x fewer LDS ops and 4x
fewer V-store ops. MUST keep `#pragma unroll 1` on the wpi loop: without it the
compiler fully unrolls and hoists all 16 A-fragments -> 192 VGPR + 160 scratch
(that is exactly the previously-rejected "transposed v_lds" signature; the
layout was never the problem, the unroll was). Pre-window hypothesis that the
2-way fp16-packed nt conflict was the stall was WRONG in detail but right in
direction: LDS *instruction* count was the stall, not bandwidth.

Stats: winograd_conv VGPR 134 -> 135, SGPR 66, scratch 0 (both). LDS unchanged
10 KB. Paired 100f x3 after change:
  chroma: hip 22.61/22.28/22.35  migx 19.52/19.43/19.40  (+15% over MIGX)
  base:   hip 22.06/22.00/21.92  migx 14.92/15.12/15.05  (+46% over MIGX)
Accuracy (tmp/acc_gen.py, per-backend processes, stride-aware):
  chroma 1920x1080 random: max 0.0117 (all planes), 0% px > 0.02 — identical
  class to the pre-change 0.0117/0.0112. base on real jpbd frame 0: luma max
  0.00097, 0% > 0.02; U/V bit-identical (Catrom path, model not involved).

## REVIEW.md follow-up (2026-09-12) — non-P1/P2 items, and a corruption hunt

Gate for everything here: `tests/correctness/run.py` 82/82 and the 1920px
EP+luma accuracy vs ORT CPU at its 0.00134 baseline. RX 7900 XTX, MANGOHUD=0,
paired/interleaved runs.

### Rare output corruption — CHARACTERIZED, NOT FIXED (top open item)

The pre-existing intermittent corruption is REAL and reproduces in BOTH paths
(standalone plugin and EP), so it lives in the shared kernels or their
environment, not in one host path. What was measured this session:

- Rate is environmental, not binary-dependent: the same .so gives 0/40 and
  10/25 corrupt samples in different windows. Sizes matter: ~0 % at 64x64,
  ~10 % at 512px, up to ~40 % at 1920px (EP + luma, one frame per process).
  Nothing tried re-opened a window: GPU heat or 150 s idle, 12 CPU burners,
  4-way concurrent GPU load, running the CPU reference first, 4-way mixed-size
  workload rotation, 20 fresh engines in ONE process (0/20 -> not per-engine).
- Where it lands: the EP at 1920 shows SMALL, spread perturbations
  (max 0.003-0.016 over ~0.5-valued pixels; clean runs cap at 0.00134) confined
  to a few pixels/rows; the plugin chroma probe at 64x64 can instead show
  garbage-level blocks (0.05-0.5). Both are the same defect at different depths.
- Op-level bisection (plugin, VSHIP_DUMP, chroma probe): in a corrupt process
  the intermediate dumps are bit-identical to a clean process up to a RANDOM
  conv op (observed first-diverging op 1, 3, 9, 10, 12, 12, 16, 18, 19, 25),
  that op's inputs (in0/in1/in2 dumps) are bit-identical, and its output is
  wrong; everything downstream is wrong accordingly. Run 2 in the same process
  was bit-identical to a clean run2 (so it is a first-RUN transient, not
  persistent bad data).
- RULED OUT, with the artifact that ruled it out:
  - uninitialized LDS: rebuilt with both shared arrays poisoned with fp16 NaN
    through `volatile` pointers (fill loops verified present in the ISA as
    `flat_store_b16 0x7e00`) -> output stayed finite and correct, 5/5. The
    kernel never reads LDS it did not write.
  - uninitialized global activations: `VSHIP_POISON=1` (engine + EP) fills every
    non-constant tensor with NaN each frame -> 2/30 corrupt, same as unpoisoned.
  - tail-wave skip: interleaved A/B, baseline 6/50 vs skip 5/50.
  - barrier structure: every LDS phase is separated by `__syncthreads()` with
    `s_waitcnt lgkmcnt(0)` emitted before each `s_barrier` (checked in asm);
    strip4/v_lds coverage is complete by construction (also proven by the NaN
    fill above).
  - one source of FALSE evidence: the engine stream is `hipStreamNonBlocking`,
    so a plain `hipMemcpy` dump on the null stream is NOT ordered after the
    kernels. The dump uses `hipMemcpyAsync(..., stream_)` + sync now; the first
    dump attempt produced bogus "non-determinism at op07" that this caused.
- Leading remaining hypothesis: a memory-VISIBILITY race (a kernel reading a
  value a copy or the previous kernel has written but not yet published), not a
  logic bug — the "everything downstream is consistent with one bad input
  value" signature and the NaN results both fit it. Untested because no window
  was open when the sync-each knob existed.

Tools for the next attempt (all env-gated, off by default):
- `tests/corruption_hunt.py [--cases luma1920] [--rounds N] [--tolerance T]`:
  fresh process per sample, hash-keyed ORT-CPU reference, saves corrupt frames
  to `tests/.corrupt/`, exits non-zero on any hit. This is the way to notice a
  window and to grade a fix.
- `VSHIP_DUMP=<dir>` (engine): dumps every device op's inputs/output for the
  first two runs of a frame -> diff a corrupt run against a clean one op by op.
- `VSHIP_POISON=1` (engine + EP): NaN-fill all non-constant activation tensors
  each frame, turning read-before-write into NaN.
- `VSHIP_SYNC_EACH=1` (EP): `hipStreamSynchronize` after every op — the direct
  test of the cross-kernel visibility hypothesis.
- `tmp/p2probe/find.py` (plugin chroma 64x64), `tmp/ab_corrupt.sh <soA> <soB> N`
  (interleaved corruption-rate A/B), `tmp/find_corrupt_ep.py` (1920px EP
  characterisation).

### Experiments run (results, all accuracy-gated)

- Tail-wave skip (REVIEW "concrete candidate", SHIPPED): `winograd_conv` now
  computes `k_wave = wave < kgroups` and the waves past the last k-group skip
  the WMMA loop, the staging and the writeback while still joining every
  `__syncthreads()`. Whole-frame effect: NEUTRAL — paired A/B chroma
  22.08/21.93/21.84 (baseline) vs 22.09/21.88/21.87, base 21.51/21.26/21.55 vs
  21.04/21.53/21.40, and the corruption A/B above. Only the M<16 tail conv
  (one of 27) has kgroups<4, so the ceiling was ~1-2 % and the kernel is
  latency-bound. Kept: correct by construction, passes 82/82, removes work.
- Host-IO reuse / direct pinned staging (REVIEW suggestion, REVERTED): moved the
  plugin's packing and writeback into engine callbacks that run under the engine
  mutex, removing a ~8.3 MB + ~33 MB per-frame host allocation+zero and the two
  engine memcpys. Result: NEUTRAL at num_streams=2 and **-2.5 % at
  num_streams=1** (5 rounds: 19.94 median baseline vs 19.55 hooked) because the
  host work moved inside the lock and stopped overlapping the GPU. The host
  path is NOT the bottleneck: `VSHIP_HOSTPROF=1` (same knob as VSHIP_DUMP
  instrumentation) measured in_alloc 0.64 ms + pack 0.42 ms + out_alloc 1.90 ms
  + write 2.73 ms of ~5.7 ms/frame of host work, all hidden behind the GPU.
  Reverted in full (engine + plugin back to `Run(memcpy, memcpy)`).
- Engine pool sweep (REVIEW suggestion): chroma 100f paired, hip 1/2/3/4
  engines = 19.86 / 21.66 / 21.25 / 21.10 fps. 2 is optimal (+9 % over 1); the
  plugin/vsscale default of 2 is already right, no pool change worth making.
- Shape-cache / weight-sharing (REVIEW suggestions): not worth code. They are
  inside the lock (per-frame `PropagateShapes` is ~1.75 us) or startup-only, and
  the host-path measurement above shows host time is fully hidden.
- F16C host conversion (REVIEW suggestion): NOT implemented, because the cost it
  targets is smaller than the review's probe suggested and is already hidden.
  The GRAYS->fp16 pack loop is the whole f32->f16 host cost here and
  `VSHIP_HOSTPROF` measures it at **0.42 ms per 1080p frame** (the review
  extrapolated ~1.13 ms of saving from a scalar helper); with host work
  overlapped by the GPU, even a free conversion buys no fps. Revisit only if
  some path becomes host-bound.
- Epilogue specialization (REVIEW suggestion): inspected, not implemented.
  `do_silu`/`do_add`/geometry are uniform branches OUTSIDE the WMMA loop, the
  kernel is at 135 VGPR with zero spills, and the historical register-diet
  variants (R1/R1b, -11 %/-14 %) show this kernel wants more memory-level
  parallelism, not fewer live registers. The small-M half of that suggestion is
  the shipped tail-wave guard above.
- rocprofv3 on the real 1080p chroma run: winograd_conv VGPR=136 (source asm
  says 135), SGPR=128, scratch 0, LDS_BLOCK_SIZE 13824 B, ~2.97 ms per dispatch
  in the profiled run (profiling inflates; the unprofiled frame rate implies
  less), 27 dispatches/frame. The notes' "4.4 ms/conv, 13 % MMA util" came from
  VSHIP_PROFILE EVENT spans, which include queue gaps — do not quote it as
  execution time (the review's warning was right).

### Numeric corrections to old text in this file

- LDS per workgroup is **13,696 B** by source (3,456 strip4 + 10,240 v_lds at
  stride 20); the hardware reports 13,824 B. Any "15,744 B / 10 KB LDS"
  statement predates the stride-24 -> 20 change and is stale.
- RX 7900 XTX L2 is **6 MiB** with **96 MiB Infinity Cache**, not 256 MB. A
  1080p 64-channel fp16 tensor is ~265 MB, so it still exceeds both and the
  "no reuse across convs" conclusion stands — but not for the reason recorded.
- MIGraphX is at 19.3-19.7 fps (updated libvsmigx) while HIP is 21.8-22.2 on
  the current build: +13-15 %, not the +25 % in older sections.

### Test tooling changes (REVIEW "Tests performed and their limits")

- `tests/multires.py` rewritten: each backend runs in its OWN process (the
  HIP+MIGX same-process teardown crash), the CPU reference cache is keyed by
  SHA-256 of the model file and the input tensor, accuracy is ASSERTED against
  the reference (fail on non-finite, maxdiff > 0.01, or >0.5 % 8-bit
  mismatches), and a session that cannot claim the graph fails instead of
  falling back (`session.disable_cpu_ep_fallback`). `--no-speed` is the fast
  accuracy gate; rounds alternate HIP/MIGX.
- `tests/corruption_hunt.py` (new): the intermittent-corruption hunter above.
- `tests/benchmark.py`: was silently SKIPPING the HIP EP (a plugin EP is absent
  from `get_available_providers()`) and swallowed per-provider exceptions. It now
  builds each session with the CPU fallback disabled (placement is proven, not
  inferred from a provider name), reports failures, and exits non-zero.
  HIP 2.28 ms/iter vs MIGX 2.73 at 256px on the 7900 XTX.

## P1 review fixes (2026-09-10, SHIPPED) — REVIEW.md items 1-9

All nine [P1] findings from REVIEW.md are fixed in the tree (the nine [P2]
findings were fixed in the follow-up session — see the P2 section below; the
performance items from the review are still open). The engine/EP now REJECT
what they cannot compute correctly instead of reading out of bounds or silently
under-writing. Verified by the correctness suite (`tests/correctness/`, run with
`python tests/correctness/run.py`): engine fixtures vs numpy references, EP vs
the ONNX reference, plugin via vspipe — 62 checks at the time of these fixes
(the suite has grown since; see the P2 section). The same checks fail against
pre-fix builds (10 of them), so they demonstrate the bugs rather than merely
passing. The plugin suite also runs all three shipped models, including the
chroma model (3 planes -> 2 chroma planes) whose U/V outputs are compared
against an independent ORT-CPU run: correct pairing ~0.006, a swapped or
shifted plane 1.0, so the chroma layout is gated.

- **#1 subsampled/channel mismatch (vs_hip.cpp).** Creation rejects
  subSamplingW/H != 0 (no resampling path exists) and requires
  `fmt.numPlanes == model input channels` (model C>3 also rejected); GetFrame
  re-checks per-plane W/H and the plane count for variable-format clips. The
  old `pi = (numPlanes>1)?p:0` luma-replication for single-plane clips is gone
  (it was only reachable when the counts already disagreed).
- **#2 weight validation (hip_engine.cc, hip_graph.cc).** Before ANY shape
  probe or the fixed 9-tap packing: weight rank==4, dims[2..3]==3, M/C>0,
  dtype float/float16, backing bytes >= numel*elem; bias rank==1 and
  length==M with the same dtype/byte checks. External-data initializers are
  rejected at materialization. EP gets the same checks (it already checked
  3x3 in the capability predicate, but Compile is the real boundary).
- **#7 typed fp16 initializers (hip_engine.cc).** FLOAT16 values stored in
  `int32_data` are 16-bit BIT PATTERNS; they are now unpacked as such instead
  of being numerically converted and re-typed (identity kernel with typed
  storage was 15295.0 off, now bit-exact). dtype is preserved from the proto.
  This also fixes ArtHIP's own converter output (`convert_float_to_float16`
  emits `int32_data` for float16).
- **#8 constant/broadcast binary ops (both).** After Add fusion, every
  Mul/Add/Sigmoid/Clip/fused-residual operand must be an activation produced
  in the graph, and binary operands must have identical shapes. Initializer
  constants and graph inputs now fail at Build/Compile instead of binding a
  null device buffer (the `x + 0.25` ASan/UBSan crash). Broadcasting is
  rejected, not implemented.
- **#9 DTS blocksize (hip_engine.cc, hip_kernels.h).** `dts_kernel_in_f32`
  is used only when blocksize==2 (was: any final fp32 DTS); blocksize 1/3 fall
  back to `dts_kernel_2d` + the cast. blocksize>=1 and `C % B^2 == 0` are
  validated in shape propagation. b=3 was 1.37 wrong, now 0.0005.
- **#3 dispatch geometry (hip_engine.cc, hip_graph.cc, hip_kernels.h).** N
  must be 1; H,W >= 2. Direct conv: `ceil(W/8)` 2D grid (was `W/8`, zero
  blocks below W=8). Winograd: `ceil` tile counts, dispatch
  `tiles_w8 * row_pairs * kgroups_pairs` (space-filling, was
  `ceil(floor(H/2)*floor(W/2)/16)` which skipped the last band, e.g. 1920x1082)
  and the kernel derives `tiles_w8 = ceil(tiles_w/8)` with a zero guard (was
  a divide-by-zero for W<16). M>64 is rejected in both paths (direct writes
  <=64 channels; winograd's 4 waves cover exactly 64) instead of under-writing.
  A 12-size H/W sweep (odd, non-multiple-of-16, 1082 rows, 1920 wide) is now
  within 0.0005/0.0022; pre-fix it was 3.0/3.6.
- **#5 error propagation (both).** Every allocation, transfer, launch, event
  and `hipStreamSynchronize` is checked; failures report the HIP error string
  and unwind partial device state (staging pointers, buffer indices) so a
  later frame on the same engine still works. `hipLaunchKernelGGL` discards
  its status, so `hipGetLastError()` is read after each launch — with a
  last-error consume at Run() entry so a previously REPORTED failure is not
  misattributed to this frame's launch. The EP returned OK on a forced
  synchronization error before; it now fails the node.
- **#6 device selection (both).** HIP's current device is thread-local, so
  `HipEngine::Run`/`EnsureBuilt`/`DestroyDeviceState` and the EP's
  `HipGraph` paths re-select the owning device. `HipContext::Initialize` takes
  the configured `device_id`, validates the ordinal, and creates the stream on
  it; `info_.device_id` is now actually wired (it was hard-coded to 0). A
  failed context init makes GetCapability return empty and Compile fail, so
  the EP never advertises nodes it cannot run.
- **#4 EP fp16->fp32 output cast (hip_graph.cc).** The half source is read
  from the pinned staging COPY, not from `output_data` (which the wider float
  writes were overwriting). The review's `[0.25,0.5,0.75,1]` case now converts
  correctly (0.00011 vs reference; pre-fix 512.0 off).

Gate after the fixes (all on the RX 7900 XTX): luma 320x180 vs MIGX maxdiff
0.001884 (0% >0.02), chroma 0.000/0.0095/0.0083 (0% >0.02), EP multires 256
HIP-vs-CPU 0.00098 = MIGX-vs-CPU 0.00098. Paired 100f: base hip 21.6-21.9 vs
migx 14.7, chroma hip 22.0 vs migx 19.1 — unchanged from the pre-fix build, so
the added validation costs nothing measurable.

## P2 review fixes (2026-09-10, SHIPPED) — REVIEW.md items 10-18

All nine [P2] findings are fixed in the tree. The correctness suite is now 82
checks (48 engine / 8 EP / 26 plugin) and passes; the strict xfail that tracked
the EP multi-channel layout (P2-11) was removed because that check now passes.
Every fix has a regression check named after its finding, and 17 of the 18 new
P2 checks FAIL against a stashed pre-P2 build (the 18th, "frame-sized tilesize
is accepted", passes on both — it guards against over-strict rejection).

- **#10 DTS DCR + fp32 tail layout.** Every DTS kernel used CRD order
  (`c*B*B + r*B + q`); ONNX DCR requires `(r*B + q)*Cout + c`. They coincide for
  Cout==1, which is why the luma tail hid the defect. All three kernels now use
  DCR. `dts_kernel_in_f32` (input-centric fp32 fast path) is now restricted to
  b==2 AND Cout==1 — only then is the whole-block half4 load contiguous under
  DCR — and writes NCHW, so the caller can skip the output transpose. Fixtures
  with Cout=2, b=2/3 and fp16/fp32 outputs are exact.
- **#11 fp16 multi-channel layout (engine + plugin + EP).** The engine's public
  boundary is now NCHW plane-major for BOTH IO dtypes (it was NHWC for fp16,
  NCHW for fp32), matching its documented contract. fp16 IO with C>1 is
  transposed on-device by a new `transpose_f16` in `hip_kernels.h` (shared with
  the EP); C==1 skips it, so the shipped luma path is untouched. The plugin packs
  plane-major for every dtype/path; the EP uploads/downloads through the same
  kernel, so its multi-channel support is real instead of silently mis-laid-out.
- **#12 integer/half clip -> fp32 model.** Packing branches on the engine dtype
  as well as the clip dtype: an int or half clip feeding an fp32-input model is
  widened to float (it wrote 2 bytes/element into a 4-byte buffer). GRAY8 128
  reaches the model as 0.50195 (was ~3.15e-5).
- **#13 fusion vs other consumers (both engines).** SiLU, residual-Add and
  DTS->Clip fusion require the renamed tensor to have no consumer besides the
  nodes being folded and to not be a graph output; the Add fusion also requires
  distinct operands. `raw=conv(x); z=raw*sigmoid(raw); y=raw+z` now runs (it
  used to lose `raw`), and a residual whose conv output feeds another op is no
  longer renamed away. The shipped models still fuse everything: 28 device ops
  each (27 conv + tail), unchanged.
- **#14 Conv pads / Clip bounds (both engines).** ONNX pads are
  [top,left,bottom,right] and the kernel only implements symmetric 1-padding, so
  all four entries must be 1 (`[1,0,1,0]` was compiled as full padding with the
  wrong output shape). Clip bounds default to the element type's extrema
  (+/-inf), not [0,1], in both the attribute and input forms; a non-initializer
  (dynamic) bound is rejected. The EP capability predicate rejects the same
  cases so they fall back instead of failing Compile.
- **#15 options (vs_hip.cpp).** Omitted `fp16` keeps its documented `true`
  default (mapGetIntSaturated's error flag, not its 0 return). `overlap` /
  `tilesize` are read: the no-tiling default (absent, zero overlap, frame-sized
  tilesize) is accepted, anything else is rejected explicitly — there is no
  tiling path.
- **#16 flexible num_planes.** Comes from the propagated final output shape
  instead of the tail Conv's M: the luma model (4 tail channels -> DTS) reports
  1 plane, matching the frame it produces.
- **#17 EP partition boundaries.** External inputs are deduplicated; an output
  is exported whenever it has a consumer outside the run or is a graph output
  (consumers counted per edge). A run is claimed only with exactly one
  activation input and one output; everything else falls back rather than being
  run with input 0 / output 0 only. Unsupported Conv pads, non-DCR DepthToSpace
  and dynamic Clip bounds now fail the capability predicate, not Compile.
- **#18 profiling events.** A separate final-transfer event (2*ops+1) is
  recorded after the output conversion/download, so the last op's span no longer
  absorbs it and the transfer has its own printed span. Events are created once
  per built plan (freed in DestroyDeviceState) and the trace/frame counters are
  per-engine instead of function-static (they raced across engines, each holding
  a different mutex).

Gate after the P2 fixes (RX 7900 XTX): photo `tests/test_1920.png` GRAYS -> 2x,
HIP vs MIGX max absdiff **0.002441** (= the documented 0.0024, unchanged); 200
blank 1080p frames hip **22.3 fps** vs migx 8.4; EP `multires.py 256`
HIP-vs-CPU 0.00098 (= MIGX-vs-CPU) at 1.78 ms/iter vs MIGX 2.57; suite 82/82.

**NOT from the review — intermittent chroma corruption (OPEN, pre-existing).**
While validating #11 the chroma probe (3 planes -> 2, fp32 clip, 64x64) produced
a corrupted frame: 3-5% of pixels off by 0.02-0.11, scattered over the whole
plane, different garbage each process but bit-identical for repeated frames
*within* one process. Rate is low and variable (~1/29 with the P2 build, ~1/37
with a stashed pre-P2 build, 0/80 in another P2 run), so it is NOT a P2
regression — the same binary produces good or bad output at random. That
signature (per-process variation, in-process repeatability) points at reading
state that was never written (LDS or device memory), not at a missing barrier in
the winograd kernel (audited: strip and v_lds coverage is complete; the only
unwritten bytes are the 4-half v_lds row padding and unwritten small-M tail
channels, neither of which any store consumes). Reproducer:
`tmp/p2probe/find.py` (loops until a bad chroma frame appears; ~30 runs).
Next step: bisect by dumping an intermediate tensor (VSHIP_DEBUG device-op
indices + a tensor readback) on a bad run, or run the same input repeatedly
inside one process with `rocprofv2` LDS counters.

Determinism measured the same session (direct plugin, `core.hip.Model`, one
frame per process, SHA-256 of the plane):
- luma, 1920x1080 photo -> 3840x2160: **12/12 bit-identical** (6 runs at
  num_streams=1 and 6 at num_streams=2).
- luma, 1920x1080 BLANK -> fp16: 2/6 and 2/8 runs differ, but only by 3.05e-5
  (= 1 fp16 ULP, subnormal) at ~1400 pixels where the output is ~1e-5. Harmless
  rounding at the very bottom of the range, not corruption.
- Going through vsscale's `ArtCNN.R8F64` (which TILES the frame and calls
  `core.hip.Model` per tile) adds up to ~0.03 differences at a few tile-seam
  clusters, HIP-vs-HIP, while the same plugin called directly is identical. So
  that layer — not ArtHIP — is what makes the vsscale-based "photo vs MIGX"
  gate noisy; the documented 0.002441 shows up when the wrapper happens to agree
  with itself. Prefer direct `core.hip.Model` comparisons for accuracy gates
  until that is explained.

## VapourSynth API4 port (2026-09-10, SHIPPED)

vs_hip.cpp moved from API3 to API4 (`VapourSynthPluginInit2` / `configPlugin` /
`createVideoFilter` / `VapourSynth4.h`); build_hip.sh adds `-DVS_USE_LATEST_API`
(declares API 4.2, like the other plugins). Ported against
reference/vs-mlrt-api4/vsmigx. Mechanical changes: prop*->map*, VSFrameRef->
VSFrame, VSFormat->VSVideoFormat (now embedded by value in VSVideoInfo),
registerFormat/getFormatPreset -> queryVideoFormat, createFilter(w/ init
callback) -> createVideoFilter(w/ out vi), getFrame's instanceData is `void*`,
arg spec `clips:vnode[]`. API4 has no filter-init callback, so the model load,
shape probe and output-vi resolution moved from hipInit into hipCreate; the
flexible path now writes "clip"+"num_planes" straight into `out`
(createVideoFilter appends "clip"), dropping the temp-map copy.
Verification (A/B same session, tmp/bench_port_ab.sh chroma 150f x3): api3
21.77/21.44/21.57 vs api4 21.49/21.57/21.48 — perf-neutral. chroma/base/dehalo
outputs BIT-IDENTICAL to the API3 build (chroma 320px: max 0.0095/0.0083 vs
MIGX; base real jpbd luma 0.00097; dehalo RGBS 0.0034/0.0044/0.0039, 0% > 0.02).
All input paths re-checked: GRAY8/GRAY16/GRAYH/GRAYS in, GRAYH/GRAYS out,
num_streams 1-2, Version/DeviceProperties, 0-clip + missing-path errors clean.
fp16=0 status (CORRECTED 2026-09-10 in the P2 session; the old "output is ~0,
do not use fp16=0" claim was stale — REVIEW.md P2-12 had already flagged it, and
`WeightFloat()` reads fp32 weight storage fine). Measured on the shipped models
with fp16=0 vs fp16=1: luma GRAY8 -> Gray16 vs GrayH, means identical; chroma
YUV444PS -> GrayS both, means within 5e-5; dehalo YUV444PS -> RGBS both, means
within 2e-4. The concrete fp16=0 defect was narrower: an INTEGER or HALF clip
feeding an fp32-input model wrote 2 bytes/element into a 4-byte buffer (GRAY8
128 arrived as ~3e-5). Fixed by P2-12 (packing now branches on the engine's
dtype too); `test_p2_12_integer_clip_into_fp32_model` gates it.

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
- LDS 13.7 KB/WG (v_lds 10240 B [wp][nt][20] + strip4 3456 B); the 15.7 KB
  figure predates the stride-24 -> 20 V change. Hardware reports 13,824 B.
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
    ==> DONE 2026-09-10 (stride 20, 4x b64/lane): the 5.8 fps was the wpi
    loop unrolling, not the layout. `#pragma unroll 1` -> +19-20%. Top of file.
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
  ==> RETRIED AND SHIPPED 2026-09-10; real cause was wpi-loop unroll, see top.
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

- RARE nondeterminism / intermittent corruption — INVESTIGATED 2026-09-12, see
  "REVIEW.md follow-up (2026-09-12) -> Rare output corruption" at the top. It
  reproduces in the EP as well as the plugin, is a first-run-per-process
  transient at a random conv op, is NOT uninitialized LDS or uninitialized
  global memory, and correlates with nothing controllable (heat, idle, CPU/GPU
  load, engine count). Hunt with `tests/corruption_hunt.py`; the leading
  hypothesis is a memory-visibility race, testable with `VSHIP_SYNC_EACH=1`.
- Actual resident waves/SIMD for winograd_conv (rocprofv2) — 3 predicted
  from 133 VGPRs; verify, plus LDS bank-conflict counts for B gather.
- Why did vs_ab_check segfault intermittently at teardown in 4-backend
  same-process runs (migx+vulkan+hip loaded together)? HIP-only runs are
  stable. STILL OPEN 2026-09-06: plain `python -c` HIP-then-MIGX
  ArtCNN.R8F64 same-process run segfaults in teardown AFTER printing both
  correct accuracy lines (luma maxdiff 0.00571 printed fine); per-backend
  processes are the workaround. Suspect cross-runtime (ROCm+RADV) teardown
  ordering; not a plugin blocker but makes scripted validation flaky — prefer
  per-backend processes for automated checks.

# HIP NOTES - standalone HIP plugin + HIP execution provider

Working notes for `src/vapoursynth/` (VapourSynth plugin) and
`src/onnxruntime-hip/` (EP); the winograd WMMA kernels they share live in
`src/common/`. Vulkan-side history lives in
`src/vulkan/OPTIMIZATION_NOTES.md`.

## Layout & Build

- Plugin: `src/vapoursynth/{vs_hip.cpp, hip_engine.cc, hip_engine.h}`
  built by `src/vapoursynth/build_hip.sh`
  (`hipcc --offload-arch=gfx1100`, links system onnx + protobuf; no ORT).
  VapourSynth API4 (`VapourSynth4.h`, `VapourSynthPluginInit2`,
  `-DVS_USE_LATEST_API` = API 4.2).
- Kernels: `src/common/hip_kernels.h` - native wave32 WMMA via
  `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32`; shared by the plugin and the
  execution-provider code.
- Provider (ORT EP): `src/onnxruntime-hip/build.sh` (ORT v1.29.0 sources in git-ignored
  `tmp/ort129`; vtable defines `-DENABLE_TRAINING -DORT_USE_NCCL
  -DENABLE_STRIDED_TENSORS` + `-mf16c -mavx2` to match Arch onnxruntime-rocm
  1.29.0, like vulkanonnx `build_provider_system.sh`; links
  libonnxruntime_providers_shared). `tests/multires.py` fail-fasts if the
  .so is missing / registration fails / the session falls back from HIP.
- Install: `cp src/vapoursynth/build/libhip.so
  /usr/lib/python3.14/site-packages/vapoursynth/plugins/libhip.so`
- vsscale: `Backend.HIP` registered in
  `/usr/lib/python3.14/site-packages/vsscale/mlrt/backend/base.py`
  (`HIP = hip.HIP`; upstream hip.py existed but was unwired).
  NOTE: like the VULKAN entries, a vsjetpack update resets base.py -
  re-add the import + `HIP = hip.HIP` line after every update.
- Run: `MANGOHUD=0 VS_BACKEND=hip vspipe -p tests/vs_test.py --`
  (tests/vs_test.py already routes `hip` to Backend.HIP).

## Speed session (2026-09-06) - HIP 19.4 vs MIGX 15.5 (+25%, cooled pair)

Paired 500f after cooldown (2026-09-06): HIP 19.44 vs MIGX 15.49 (+25%).
Photo accuracy vs MIGX unchanged
(max 0.0057, 0% pixels >0.02). At/above the RTX 3080 TensorRT target band.
BENCH HYGIENE: back-to-back runs heat-soak the card (MIGX 13.5, HIP 16.6 when
hot); always cooldown + pair before quoting numbers.

Target: RTX 3080 TensorRT does 18-19 fps on this model (similar on-paper fp16
TFLOPS to the 7900 XTX), so headroom should exist. Method: rocprofv3 kernel
times (winograd 478ms/8 frames = 2.3ms true exec, not the 2.2ms event span),
HIP_CONV_SKIP phase toggles, VGPR probe via hsaco notes, ABA benchmarks.

SHIPPED (all in build, accuracy-gated vs MIGX on photo: max 0.0024, unchanged):
+ v_lds stride 24->20 (2026-09-06): 17.0 -> 19.1-19.3 @120f (+12%), 19.56
+ @500f. Bit-exact vs stride-24 (photo maxdiff 0.00571 both, same mean).
+ VGPR unchanged (134), no spills. WHY: stride-24's c-row step (12 banks)
+ revisits 8 bank sets twice per 16-deep gather (8 conflicts/lane); stride-20
+ (step 10 banks) visits all 16 distinct banks, zero c-direction conflicts.
+ Pure win: 12KB->10KB LDS (still 4 WGs/CU), zero new live state. KEPT.
+ (Stride 18 tried: BLACK — nt=16..17 escape the row and collide with the
+ next wp block. 20 is the minimum safe stride for 16-wide nt; do not retry
+ narrower without re-deriving the writeback staging overlap.)
+ Drop __launch_bounds__(128,4) (2026-09-06): no-bounds 16.9-17.1 vs 16.7-16.8
+ (128,4) vs 16.7-16.9 (128,8), median-of-3 @120f; VGPR identical (134), no
+ spills either way — the hint only constrained the scheduler. KEPT.
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
+ EP DTS->Clip fusion fix (2026-09-06): kernel had do_clip but host never
  set it (4-u32 params vs 7-u32 DtsParams) => 8x8 output all zeros, 256
  maxdiff 0.97 vs CPU. Fixed by Pass 3 (fuse trailing Clip into DTS) +
  full DtsPush. After fix: 256 maxdiff 0.00098, 1920 maxdiff 0.00134,
  HIP 66.8 vs MIGX 67.1 ms/iter. KEPT.

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
- R6 direct-from-register writeback, INLINE patch attempt (2026-09-06):
  replaced the staged writeback with per-lane direct stores (ko=k_base+
  lane/16+2e). WRONG VALUES (photo mean 0.49846 but maxdiff 0.0039/2.3% px
  vs staged — the D-fragment row->ko mapping needs the per-wpi FOLD4 decode,
  not the assumed linear map) AND slower (16.55 vs 16.9). REVERTED; R6 stays
  open only as a separately-validated kernel.
- Strip half16 b128 widening (2026-09-06): dst[2]=src[2] dropped (copy-paste:
  2x half16 covers all 16 ch, the extra 2 lines were stale) => black output
  (mean 0.0037), slower (16.4). REVERTED; the 4x half4 path was already b128
  in disasm (2x global_load_b128 + 2x ds_store_b128 per cell) — nothing left.
- Transposed v_lds, SECOND attempt (2026-09-06): CORRECT (maxdiff 0.0059 vs
  MIGX) but 5.0 fps (-70%). Cause found: VGPR 134 -> 192 WITH spills
  (161 scratch_ ops); the 4 extra half16 row pointers + 4 extra resident
  A-frag sets blew the register budget. Layout idea is sound, register cost
  is not — retry only with a leaner schedule (recompute pointers per wpi,
  stream A per-group). REVERTED.
  ==> RESOLVED 2026-09-10, NOW SHIPPED: the register blowup was wpi-loop
  unrolling (all 16 A-fragments hoisted), not the layout. `#pragma unroll 1`
  on the wpi loop -> VGPR 135, no spills, +19-20%. See CURRENT BEST at top.
- A-fragment pre-staging across wpi (2026-09-06): same failure signature
  (192 VGPR + spills, 5 fps). 16 extra live half16 across the wpi loop does
  not fit. REVERTED.
- B-gather u32-pair widening (2026-09-06): BLACK output. The u32 reinterpret
  of the stride-24 LDS row confused the compiler's alias analysis (spills
  appeared, WMMA operands moved to v89+). REVERTED.
- Writeback-stage half2 widening (2026-09-06): BLACK output. Staging rows
  are stride-2 in ko (koff+2e), so element pairs (e,e+1) are NOT adjacent
  rows — a half2 store writes the wrong row. Not pairable; REVERTED.
- WG=256 reshape, DRAFTED then abandoned pre-build (2026-09-06): splitting
  8 waves over (kgroup x nt-quad) needs the y-fragment->ko mapping re-derived
  (each wave owns 16ko x 4nt, not 16ko x 16nt; staging/writeback all change).
  My draft got the staging indices incoherent (leftover koff algebra,
  yb collisions); stopped before building rather than debugging blind.
  Needs a clean-sheet derivation of the per-wave (ko, nt) ownership first.
- HIP-graph capture of the 29-kernel frame (2026-09-06): BLACK/NaN output.
  Capture wraps kernels-only (upload before begin, download after end) yet
  frame 1 comes out NaN — the capture path on this stack (ROCm 7.2) silently
  breaks something (no API error; diagnostics fprintf never even appeared,
  suggesting the capture rundll path misbehaves). REVERTED whole (engine
  files back to HEAD); revisit only with a minimal 2-kernel repro proving
  capture works on this machine first.
- LDS XOR-swizzle of v_lds nt lanes (2026-09-06): CORRECT but 4.7 fps.
  The per-access xor blocked the compiler's LDS address CSE: VGPR 134->192
  WITH spills (166 scratch ops), ds_load_u16 64->256. Same over-budget
  signature as transpose/prefetch. REVERTED.
- Padded-c v_lds stride 24->48 (2026-09-06): CORRECT but 12.1 fps (-29%).
  24KB LDS halves residency 4->2 WGs/CU and the extra live waves don't
  compensate — occupancy loss dominates the conflict win. REVERTED.
  Lesson: LDS budget is load-bearing; stop trading it away.

BIG-LEVER ANALYSIS (2026-09-06, all measured, no code):
- F(4x4,3x3): KILLED numerically. fp16-transform error ~1-6 per element vs
  signal ~1-2 (1000x worse than F(2x2)'s ~1e-3); matches published experience
  (F(4,3) overflows fp16, costs undetected mAP). Would need fp32 transforms
  (register/LDS blowup) or a different tile (F(2x4) needs LDS redesign).
- Megakernel/persistent: KILLED by scaling data. Time is linear in pixels
  (34 Mpix/s from 0.26 to 2 Mpix/frame; 2-wide batch = 2x serial) and each
  conv's working set (265MB in + 265MB out) exceeds the 6MB L2 AND the 96MB
  Infinity Cache — NO reuse is possible across convs (weights 0.13MB are the only reusable bytes, and
  they already stay L2-resident within a conv). Gap-hunting bounds the prize:
  s=1-vs-s=2 overlap is worth 9% and we already bank it; residual graph
  overhead is ~1%. NOT next.
- True roofline (corrected 2026-09-06): 68 GFLOP/conv, MMA floor 1.1ms @61T
  vs 4.35ms measured = 26% MMA util (13% of the 123T WMMA peak); DRAM floor
  0.55ms (8x below). LATENCY-bound at 26%: the B-gather (64 serialized
  scalar LDS loads + waits per wpi) is the single biggest exposed stall.

MEASURED FACTS (rocprofv3, 8 frames @1080p, no-bounds build 2026-09-06):
- winograd: 208 dispatches, 915ms total = 4.40ms true exec each (26 convs x 8).
- direct_conv: 8.7ms total (~1.1ms each). dts_in_f32: 0.82ms (~100us each).
  cast_f32_to_f16: 1.5ms total. Convs = 98.8% of GPU time.
- VSHIP_PROFILE steady-state (frame 128): op0 (direct) 1.47ms, ops1-20 flat
  4.35ms, ops21-25 taper 4.2/3.3/2.2/2.2/2.2 (smaller H/feature maps deeper
  in the chain), op26 4.2, DTS 1.34ms. Event spans include queue gaps; trust
  shape (flat-then-taper), not absolutes (sums exceed wall time).
- VGPR (winograd.s disasm, gfx1100): 135 in the current build, no spills;
  rocprofv3 reports 136/128 and LDS_BLOCK_SIZE 13824. The old 134/15744
  pair predates the stride-20 V and the tail-wave guard.
  WMMA VGPR banks (LLVM#204254): 3 of 4 WMMAs have A/C on bank 1 (collision,
  +2c each); B on bank 2. Fix needs inline asm (can't steer regalloc from
  source) — expected gain ~6% of WMMA time only (~1.2ms of 4.4ms), NOT next.
- Frame math: 26 x 4.4 = 114ms?? vs 300 frames in ~17.5s = 58ms/frame wall.
  (Event/rocprof sums double-count overlapped streams — GPU 100% busy, the
  convs ARE the frame; ~15% over MIGX, ~5% under the 3080/TensorRT target.)

NEXT (not tried, in priority order):
1. VGPR-bank fix (LLVM#204254) via inline-asm WMMA wrapper with pinned
   registers (A/B/C forced to distinct banks): only ~6% of WMMA time, but
   near-free once written. Validate vs staged output bit-exact first.
2. Fused tail writeback+DTS (R7-lite): tail conv writes DTS-swizzled output
   directly (saves the 45us DTS + one 8M intermediate). Needs care with the
   D-fragment corner mapping (previous E-series failures); build as a
   SEPARATE kernel validated standalone first.
3. Bigger levers if the kernel stalls: persistent megakernel (grid.sync,
   weights stay in L2 across 26 convs), or IMPLICIT-GEMM reformulation
   (CUTLASS-style, kills the Winograd transform + LDS traffic entirely).
4. Lock clocks for benchmarking (rocm-smi --setperflevel high); all numbers
   above at sustained 2245MHz but the card idles at 130MHz between runs.

## Multi-channel models (2026-09-06: chroma SHIPPED, dehalo SHIPPED)

DEHALO 3ch->3ch (ArtCNN_R8F64_YCbCr_DEHALO.onnx @ workspace root, 2026-09-06):
engine needed NO changes (27 Conv + SiLU + Adds, tail Clip, C=3 in/out —
all already supported; tail weight [3,64,3,3] exercises the M<16
vector-tail writeback path). Pure PLUGIN gaps, fixed in vs_hip.cpp:
+ accept cmRGB input clips (was: gray/YUV only -> hard error on RGBS).
+ out_c==3 non-flex declares + writes a 3-plane RGB frame (mirrors vsmigx
  setDimensions: C==3 non-flex -> cmRGB; C==1-or-flex -> gray). Each output
  channel goes to its own plane; flex path untouched.
Accuracy vs MIGX GenericOnnxScaler on identical random RGBS (limiter-only
preprocess both sides): 64px maxdiff 0.0039/0.0032/0.0029, 320x180 maxdiff
0.0034/0.0044/0.0039, 0% px >0.02 all planes — BETTER than luma class.
Luma bit-identical pre/post change (acc_dump self-check maxdiff 0.0).
NOTE: model resolves relative to the VS host cwd (vsscale resolves its own
models); pass an absolute path to hip.Model for workspace-root models.
(The first "frame not of declared format" crash was mine: GetFrame wrote RGB
while hipInit still declared gray — both sides now share the out_c==3 rule.)

CHROMA RESOLVED 2026-09-06: chroma was NEVER broken in the engine. HIP-vs-MIGX via
vsscale on identical random YUV444PS: 64x64 (U maxdiff 0.005/V 0.008),
320x180 (U 0.0095/V 0.0088, 0% px >0.02), 1080p random (U 0.0117/V 0.0112,
0% px >0.02) — fp16-rounding class plus tail-Clip saturation divergence,
same class as luma (0.0057). Luma unaffected (320p maxdiff 0.00571,
EP multires 256: HIP-vs-CPU 0.00098 = MIGX-vs-CPU 0.00098).
Root causes of the false alarm, in order:
1. STALE PLUGIN BINARY: /usr/lib/.../plugins/libhip.so (04:50) predated the
   working tree's transpose fix — always rebuild+reinstall before concluding
   (cd src/vapoursynth && ./build_hip.sh && cp ...). After reinstall, VS-vs-EP
   on the identical constant-0.3 tensor matches to 1e-5 (ladder step 2 PASSES).
2. BROKEN TEST HARNESSES, not the engine:
   - chroma_vspack.py fed a RAW constant-0.3 tensor through the VS plugin,
     skipping vsscale's preprocess (Y clamp, UV x0.5+0.5); raw Blanks are
     Y=0/U=0/V=0 (NOT 0/0.5/0.5), BlankClip YUV444PS reads back garbage float
     bit patterns (~-3.7e19) without ctypes per-plane reads. And the "EP" half
     ran through the HIP ORT EP .so which no longer registers on this box
     (falls back to CPU with a warning) — so it compared garbage-VS vs CPU-EP.
   - chroma_acc.py's read idiom (ctypes.cast(get_read_ptr)) is only safe for
     tightly-packed frames; with stride padding it segfaults in numpy — use
     acc_dump.py's idiom only when stride == w*bytes, else row-copy via
     ptr.value + get_stride(0). The migx-after-hip same-process teardown
     segfault (OPEN QUESTION below) also poisoned several runs.
   - Correct harness: vsscale ArtCNN.R8F64_Chroma both backends on the SAME
     ModifyFrame clip (tmp/chroma_acc.py fixed idiom), stride-aware reads,
     per-backend processes.
3. The ACTUAL historical bug (already fixed in tree, 04:50): NCHW-vs-NHWC
   transpose missing on the fp32 bridge paths. Shipped fix: plugin packs
   NCHW plane-major + cast_f32_to_f16_transpose upload; NHWC->NCHW via
   cast_f16_to_f32_transpose download. C=1 identical (no-op), C=3 fixed.
   (tmp/chroma_vspack.py constant-0.3 VS-vs-EP ladder: VS means ~1e-28/1e-40/
   1e-13 vs EP 0.30 was the pre-fix signature. plus the fp16/fp32 Clip-bound
   misread fixed earlier: chroma stores float bounds, half-reads gave
   clip max=1.9e-3 clamping all output to ~0.)
Lesson: distrust any chroma number not produced by (a) freshly installed
libhip.so (check timestamps), (b) vsscale preprocess on both sides, (c) the
same random clip, (d) separate processes per backend.

Shipped (working, keep):
+ flexible_output_prop protocol (matches vsmigx): Model returns MAP
  {clip, num_planes} when the arg is passed; frames carry MlrtFlexibleN
  FRAME props (PropToClip needs frames, not bytes — data props fail) +
  num_planes. Verified: chroma vspipe runs, u/v split works.
+ Multi-channel pack: NCHW in_shape (clip planes or model C), NHWC
  pixel-interleaved pack, NCHW out buffers, Input/OutputChannels() from
  weight shapes, YUV clip acceptance.
+ fp32 weight/clip-bound reads dtype-aware (WeightFloat/clip_scalar):
  chroma stores float initializers; half-reads gave clip max=1.9e-3 (~all
  output clamped to 0). Fixed that stage (output went 0 → full-range garbage).
+ fp32 bridge transpose (THIS SESSION): NCHW plane-major pack in vs_hip.cpp
  (host_fp32 branch) + cast_f32_to_f16_transpose upload +
  cast_f16_to_f32_transpose download in hip_engine.cc. C=1 path unchanged.

GOTCHAS logged: /tmp is per-command tmpfs (backups vanish — use tmp/);
  `git checkout -- <file>` nukes ALL uncommitted work in it (lost the plugin
  recovery twice — commit or `git diff > tmp/` first); heat-soak skews benches.

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

### Kernel tuning log (winograd_conv in src/common/hip_kernels.h)
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
