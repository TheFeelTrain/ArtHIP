# ArtHIP — standalone HIP Winograd WMMA inference for AMD GPUs

Runs fp16 conv-chain ONNX models (ArtCNN-style: 3x3 convs + DepthToSpace)
directly on AMD GPUs through hand-written gfx1100 Winograd F(2x2,3x3) WMMA
kernels. No Vulkan, no DirectML — HIP only. Currently beats the MIGraphX
reference plugin (~17.3 vs ~15.7 fps @1080p, ArtCNN R8F64).

## Layout

- `src/hip/` — kernels (`hip_kernels.h`, shared by both paths below),
  the ONNX Runtime execution provider (`hip_graph.cc`, ...), micro-benches,
  and `HIP_NOTES.md` (experiment log — read this before touching kernels).
- `src/vapoursynth/` — the standalone VapourSynth plugin (`hip/`: `vs_hip.cpp`
  + `hip_engine.cc/h`, no ORT involved), shared ONNX helpers (`common/`),
  and `build_hip.sh`.
- `tests/` — `benchmark.py` + `multires.py` (HIP EP vs MIGraphX EP via ORT),
  `vs_test.py` (`VS_BACKEND=hip|migx` VapourSynth comparison), test images,
  `.mxr_cache/` (MIGraphX compiled programs — 7.6GB, saves recompiles).

## Build / Test

- Plugin: `cd src/vapoursynth && ./build_hip.sh` (`hipcc
  --offload-arch=gfx1100`), then copy the result into your VapourSynth
  plugins dir, e.g. `cp hip/build/libhip.so
  /usr/lib/python3.14/site-packages/vapoursynth/plugins/libhip.so`
- Provider (ORT EP): `cd src/hip && ./build.sh`
- VapourSynth comparison: `MANGOHUD=0 VS_BACKEND=<hip|migx> vspipe -p
  tests/vs_test.py --` (500 blank 1920x1080 frames)
- EP accuracy/speed: `MANGOHUD=0 .venv/bin/python tests/multires.py 256`
  (run from the original checkout until a venv is set up here)
- Run everything with `MANGOHUD=0` (the overlay skews clocks/measurement).

## Notes

- Compute is fp16; clip IO follows the model like the MIGX plugin: GRAYS
  in → GRAYS out, GRAYH/int in → GRAYH out.
- `reference/` (migraphx/llama.cpp/dml baselines) and the Vulkan EP were
  deliberately left behind in the parent repo — this tree is HIP-only.
