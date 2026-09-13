# ArtHIP — standalone HIP Winograd WMMA inference for AMD GPUs

Runs fp16 conv-chain ONNX models (ArtCNN-style: 3x3 convs + DepthToSpace)
directly on AMD GPUs through hand-written gfx1100 Winograd F(2x2,3x3) WMMA
kernels. Currently beats the MIGraphX reference plugin (~23.0 vs ~15.7 fps @1080p, ArtCNN R8F64).

## Layout

- `src/common/` — `hip_kernels.h`, the gfx1100 Winograd WMMA kernels shared by
  both backends below.
- `src/onnxruntime-hip/` — the ONNX Runtime execution provider
  (`hip_graph.cc`, `hip_execution_provider.cc`, `hip_provider_factory.cc`),
  its `build.sh`, and standalone WMMA micro-benches.
- `src/vapoursynth/` — the standalone VapourSynth plugin (`vs_hip.cpp`
  + `hip_engine.cc/h`, no ORT involved), shared ONNX helpers
  (`onnx_utils`, `convert_float_to_float16`), and `build_hip.sh`.
- `NOTES.md` — experiment log; read this before touching kernels.
- `tests/` — `benchmark.py` + `multires.py` (HIP EP vs MIGraphX EP via ORT),
  `vs_test.py` (`VS_BACKEND=hip|migx` VapourSynth comparison), test images,
  `.mxr_cache/` (MIGraphX compiled programs — 7.6GB, saves recompiles).

## Build / Test

- Plugin: `cd src/vapoursynth && ./build_hip.sh` (`hipcc
  --offload-arch=gfx1100`), then copy the result into your VapourSynth
  plugins dir, e.g. `cp build/libhip.so
  /usr/lib/python3.14/site-packages/vapoursynth/plugins/libhip.so`
- Provider (ORT EP): `cd src/onnxruntime-hip && ./build.sh`
- VapourSynth comparison: `VS_BACKEND=<hip|migx> vspipe -p
  tests/vs_test.py --` (500 blank 1920x1080 frames)
- EP accuracy/speed: `python tests/multires.py 1920`

## Notes

- Compute is fp16; clip IO follows the model like the MIGX plugin: GRAYS
  in → GRAYS out, GRAYH/int in → GRAYH out.