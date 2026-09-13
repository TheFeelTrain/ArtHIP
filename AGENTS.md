# AGENTS.md - ArtHIP (standalone HIP Winograd WMMA inference)

Goal: make HIP inference on the AMD RX 7900 XTX (gfx1100, RDNA3) FASTER than
MIGraphX on the same card at 1920x1080 — ultimately matching RTX 3080
TensorRT on this model (18-19 fps, similar on-paper fp16 TFLOPS, so headroom
exists). Small accuracy tradeoffs are acceptable if the output stays "close
enough" (photo max absdiff vs MIGX ~1e-3 class); this is the bottleneck of a
VapourSynth filter chain, so raw speed wins.

## Current State (representative; see notes for authoritative results)

- HIP leads MIGX ~10%: 17.2-17.4 vs 15.7-15.8 fps (300 blank 1080p frames,
  interleaved x2). Photo accuracy vs MIGX: max absdiff 0.0024 (unchanged).
- Kernel: F(2x2,3x3) Winograd WMMA, WG=128 (4 wave32s), fp16 compute / fp32
  clip IO (GRAYS in → GRAYS out, like MIGX). True exec 2.30ms per 64→64
  conv (rocprofv3); the 26 convs ARE the frame (GPU 100% busy).
- Authoritative state, kept optimizations, rejected experiments, and next
  steps all live in `NOTES.md` — read it before touching kernels,
  and do not duplicate it here.

## Build / Test

- Correctness suite (the primary gate; it builds and installs what it needs):
  `python tests/correctness/run.py` — see `tests/correctness/README.md`. Run it
  before and after any kernel/engine change; add a test when fixing a bug.
- Plugin: `cd src/vapoursynth && ./build_hip.sh` (`hipcc
  --offload-arch=gfx1100`), then `cp build/libhip.so
  /usr/lib/python3.14/site-packages/vapoursynth/plugins/` (world-writable,
  no root needed).
- Provider (ORT EP): `cd src/onnxruntime-hip && ./build.sh`.
- VapourSynth comparison: `MANGOHUD=0 VS_BACKEND=<hip|migx> vspipe -p
  "tests/vs_test.py" --` (500 blank 1920x1080 frames; vspipe R79 ignores
  script args, so the backend comes from VS_BACKEND, default hip).
  NOTE: ~20s per backend — run infrequently; use `tests/multires.py` and
  short vspipe runs for iteration.
- EP accuracy/speed: `MANGOHUD=0 python tests/multires.py 256` (needs an
  ORT venv; until one exists here, run from the parent checkout).
- Run everything with `MANGOHUD=0`. Numbers are GPU-contention sensitive;
  verify ~2.2GHz sclk during a bench and re-run if a number looks off.

## Working Style

- USE WEB SEARCH OFTEN. Before and during optimization work, search for
  outside help (RDNA3 WMMA performance, Winograd/GEMM tuning,
  MIGraphX/MIOpen techniques, fast SiLU/exp approximations). The web has
  sourced real wins here (occupancy cliffs, VGPR-bank fix); verify findings
  against published results and keep searching even when not stuck.
- KEEP `NOTES.md` UPDATED AND CLEAN at all times. Record every
  experiment (change, result, correctness, status) there immediately after
  confirming it — mandatory before ending a session. Rules:
  - Failed/rejected experiments go in ONE place (the rejected ledger).
    Never spread them across sections.
  - "Current Status" holds ONLY the actual current state; edit old facts in
    place instead of appending corrections.
  - Keep it concise — it is the primary context for future sessions, so
    every line should earn its place.
- NEVER accept a speed result without an accuracy check (photo max absdiff
  vs MIGX + `tests/multires.py`). Broken or inaccurate variants are invalid,
  timings ignored. Micro-benches/harnesses may locate a bug, but every
  conclusion must still be confirmed on the real model before trusting it.
- OPTIMIZE AGGRESSIVELY. Do not stop at +10% — keep pushing variants,
  measuring, and iterating toward the TensorRT-level target.
