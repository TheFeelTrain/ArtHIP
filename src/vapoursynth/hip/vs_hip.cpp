// Standalone HIP ML filter for VapourSynth (Backend.HIP).
//
// Runs an fp16 conv-chain ONNX model (ArtCNN-style: 3x3 convs + DepthToSpace)
// directly on the AMD GPU through the standalone HipEngine - no ONNX Runtime.
// The engine shares the winograd WMMA kernels with the HIP execution provider.
//
// Filter: hip.Model(clip, network_path, overlap, tilesize, device_id,
//                   num_streams, fp16)
//   - accepts GRAY8 / GRAY16 / GRAYH (fp16) / GRAYS (fp32) input; fp32 clips
//     with an fp32-io model are passed through raw (the engine converts on
//     the host inside Run); everything else is packed to fp16 [0,1] bits here
//   - compute is fp16; the output format follows the model IO like the MIGX
//     plugin: GRAYS (fp32) for fp32 clips, GRAYH (fp16 [0,1]) otherwise, at
//     2x the input size (model-defined); GRAY16 when fp16=false
//   - fp16=true (default) converts fp32 models to fp16 at load time
//   - num_streams engines run round-robin so concurrent frame requests
//     overlap (fmParallel-safe; each engine owns its stream + mutex)

#include <VapourSynth.h>
#include <VSHelper.h>

#include <atomic>
#include <cstdint>
#include <cstring>
#include <string>
#include <unordered_set>
#include <vector>

#include <onnx/onnx_pb.h>

#include <hip/hip_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>

#include "../common/convert_float_to_float16.h"
#include "../common/onnx_utils.h"
#include "hip_engine.h"

static const VSPlugin* myself = nullptr;

struct HipData {
  std::vector<vship::HipEngine*> engines;
  std::atomic<int> next_engine{0};
  std::string network_path;
  int device_id = 0;
  int num_streams = 1;
  bool output_fp16 = true;
  bool output_fp32 = false;  // fp16 compute, fp32 (GRAYS) output like MIGX
  const VSVideoInfo* vi = nullptr;
  VSNodeRef* node = nullptr;
};

static std::string vs_error(const std::string& what) {
  return "hip.Model: " + what;
}

// ---------------------------------------------------------------------------
// Frame processing
// ---------------------------------------------------------------------------

static const VSFrameRef* VS_CC hipGetFrame(
    int n, int activationReason, void** instanceData, void** frameData,
    VSFrameContext* frameCtx, VSCore* core, const VSAPI* vsapi) {
  auto* d = static_cast<HipData*>(*instanceData);
  if (activationReason == arInitial) {
    vsapi->requestFrameFilter(n, d->node, frameCtx);
    return nullptr;
  }
  if (activationReason != arAllFramesReady) {
    return nullptr;
  }

  const VSFrameRef* src = vsapi->getFrameFilter(n, d->node, frameCtx);
  if (!src) {
    return nullptr;
  }
  VSFrameRef* dst = nullptr;
  try {
  const auto* fmt = vsapi->getFrameFormat(src);
  const int in_w = vsapi->getFrameWidth(src, 0);
  const int in_h = vsapi->getFrameHeight(src, 0);

  // Round-robin over the engine pool; each engine serializes internally.
  vship::HipEngine* eng =
      d->engines[d->next_engine.fetch_add(1) % static_cast<int>(d->engines.size())];

  std::vector<int64_t> in_shape{1, 1, in_h, in_w};
  std::vector<int64_t> out_shape;
  if (!eng->GetOutputShape(in_shape, out_shape) || out_shape.size() != 4) {
    vsapi->setFilterError(vs_error("shape propagation failed").c_str(), frameCtx);
    vsapi->freeFrame(src);
    return nullptr;
  }
  const int out_w = static_cast<int>(out_shape[3]);
  const int out_h = static_cast<int>(out_shape[2]);

  const VSFormat* out_fmt = d->output_fp32 ? vsapi->getFormatPreset(pfGrayS, core)
                                : d->output_fp16 ? vsapi->getFormatPreset(pfGrayH, core)
                                                 : vsapi->getFormatPreset(pfGray16, core);
  dst = vsapi->newVideoFrame(out_fmt, out_w, out_h, src, core);

  // Pack the source frame into a contiguous host buffer. fp32 clips on
  // fp32-input models pass through raw; everything else becomes fp16 [0,1].
  const bool host_fp32 = eng->InputIsFp32();
  const size_t in_elem = host_fp32 ? 4 : 2;
  std::vector<uint8_t> in_pack(static_cast<size_t>(in_w) * in_h * in_elem);

  const uint8_t* src_ptr = vsapi->getReadPtr(src, 0);
  const ptrdiff_t src_stride = vsapi->getStride(src, 0);
  switch (fmt->sampleType) {
    case stInteger: {
      // Rare path (GRAY8/16 sources): scale + RNE convert to fp16.
      if (fmt->bitsPerSample == 8) {
        const float scale = 1.0f / 255.0f;
        for (int y = 0; y < in_h; ++y) {
          const uint8_t* row = src_ptr + y * src_stride;
          auto* dst_row = reinterpret_cast<uint16_t*>(in_pack.data()) +
                          static_cast<size_t>(y) * in_w;
          for (int x = 0; x < in_w; ++x) {
            uint16_t h;
            vship::HipEngine::FloatToHalfBitsRNE(row[x] * scale, h);
            dst_row[x] = h;
          }
        }
      } else {
        const float scale = 1.0f / 65535.0f;
        for (int y = 0; y < in_h; ++y) {
          const uint16_t* row = reinterpret_cast<const uint16_t*>(src_ptr + y * src_stride);
          auto* dst_row = reinterpret_cast<uint16_t*>(in_pack.data()) +
                          static_cast<size_t>(y) * in_w;
          for (int x = 0; x < in_w; ++x) {
            uint16_t h;
            vship::HipEngine::FloatToHalfBitsRNE(row[x] * scale, h);
            dst_row[x] = h;
          }
        }
      }
      break;
    }
    case stFloat: {
      if (fmt->bitsPerSample == 16) {  // fp16, already [0,1]
        for (int y = 0; y < in_h; ++y) {
          std::memcpy(in_pack.data() + static_cast<size_t>(y) * in_w * in_elem,
                      src_ptr + y * src_stride, static_cast<size_t>(in_w) * 2);
        }
      } else if (host_fp32) {  // fp32 raw pass-through (engine converts host-side)
        for (int y = 0; y < in_h; ++y) {
          std::memcpy(in_pack.data() + static_cast<size_t>(y) * in_w * in_elem,
                      src_ptr + y * src_stride, static_cast<size_t>(in_w) * 4);
        }
      } else {  // fp32 source into an fp16-input model: precision convert
        for (int y = 0; y < in_h; ++y) {
          const float* row = reinterpret_cast<const float*>(src_ptr + y * src_stride);
          auto* dst_row = reinterpret_cast<uint16_t*>(in_pack.data()) +
                          static_cast<size_t>(y) * in_w;
          for (int x = 0; x < in_w; ++x) {
            uint16_t h;
            vship::HipEngine::FloatToHalfBitsRNE(row[x], h);
            dst_row[x] = h;
          }
        }
      }
      break;
    }
  }

  // Output buffer element type follows the engine: fp32 (GRAYS) when the
  // model keeps a float output (fp16 compute, like MIGX), else fp16.
  std::vector<float> out_f32;
  std::vector<uint16_t> out_fp16;
  void* out_ptr = nullptr;
  if (d->output_fp32) {
    out_f32.resize(static_cast<size_t>(out_w) * out_h);
    out_ptr = out_f32.data();
  } else {
    out_fp16.resize(static_cast<size_t>(out_w) * out_h);
    out_ptr = out_fp16.data();
  }
  static const bool dbg = [] { const char* e = std::getenv("VSHIP_TRACE"); return e && *e == '1'; }();
  auto now = [] { return std::chrono::steady_clock::now().time_since_epoch().count(); };
  if (dbg) fprintf(stderr, "[trace] enter eng=%d n=%d t=%lld\n",
                   d->next_engine.load(std::memory_order_relaxed) % (int)d->engines.size(), n, (long long)now());
  std::string error;
  bool ok = eng->Run(in_pack.data(), in_shape, out_ptr, error);
  if (dbg) { int ei = 0; for (size_t q = 0; q < d->engines.size(); ++q) if (d->engines[q] == eng) ei = (int)q;
    fprintf(stderr, "[trace] leave eng=%d n=%d t=%lld\n", ei, n, (long long)now()); }
  if (!ok) {
    vsapi->setFilterError(vs_error(error).c_str(), frameCtx);
    vsapi->freeFrame(src);
    vsapi->freeFrame(dst);
    return nullptr;
  }

  // Write the output to the destination frame (fp16 [0,1] or fp32).
  uint8_t* dst_ptr = vsapi->getWritePtr(dst, 0);
  const ptrdiff_t dst_stride = vsapi->getStride(dst, 0);
  if (d->output_fp32) {
    for (int y = 0; y < out_h; ++y) {
      float* row = reinterpret_cast<float*>(dst_ptr + y * dst_stride);
      std::memcpy(row, &out_f32[static_cast<size_t>(y) * out_w], static_cast<size_t>(out_w) * 4);
    }
  } else if (d->output_fp16) {
    for (int y = 0; y < out_h; ++y) {
      uint16_t* row = reinterpret_cast<uint16_t*>(dst_ptr + y * dst_stride);
      std::memcpy(row, &out_fp16[static_cast<size_t>(y) * out_w], static_cast<size_t>(out_w) * 2);
    }
  } else {
    for (int y = 0; y < out_h; ++y) {
      uint16_t* row = reinterpret_cast<uint16_t*>(dst_ptr + y * dst_stride);
      for (int x = 0; x < out_w; ++x) {
        const float v = vship::HipEngine::HalfBitsToFloat(out_fp16[static_cast<size_t>(y) * out_w + x]);
        row[x] = static_cast<uint16_t>(v * 65535.0f + 0.5f);
      }
    }
  }

  vsapi->freeFrame(src);
  return dst;
  } catch (const std::exception& e) {
    vsapi->setFilterError(vs_error(std::string("exception: ") + e.what()).c_str(), frameCtx);
    vsapi->freeFrame(src);
    vsapi->freeFrame(dst);
    return nullptr;
  } catch (...) {
    vsapi->setFilterError(vs_error("unknown exception").c_str(), frameCtx);
    vsapi->freeFrame(src);
    vsapi->freeFrame(dst);
    return nullptr;
  }
}

// ---------------------------------------------------------------------------
// Filter creation
// ---------------------------------------------------------------------------

static void VS_CC hipInit(VSMap* in, VSMap* out, void** instanceData, VSNode* node,
                          VSCore* core, const VSAPI* vsapi) {
  auto* d = static_cast<HipData*>(*instanceData);
  try {
  std::vector<int64_t> probe_in{1, 1, 64, 64};
  std::vector<int64_t> probe_out;
  int err = 0;
  const std::string network_path = vsapi->propGetData(in, "network_path", 0, &err);
  d->device_id = static_cast<int>(vsapi->propGetInt(in, "device_id", 0, &err));
  d->num_streams = static_cast<int>(vsapi->propGetInt(in, "num_streams", 0, &err));
  if (d->num_streams < 1) d->num_streams = 1;
  d->output_fp16 = !!vsapi->propGetInt(in, "fp16", 0, &err);

  auto set_error = [&](const std::string& msg) {
    vsapi->setError(out, vs_error(msg).c_str());
    delete d;
    *instanceData = nullptr;
  };

  auto load_result = loadONNX(network_path, 0, 0, false);
  if (std::holds_alternative<std::string>(load_result)) {
    set_error(std::get<std::string>(load_result));
    return;
  }
  auto model = std::move(std::get<ONNX_NAMESPACE::ModelProto>(load_result));

  // fp16=true: convert fp32 models to fp16 at load time. An fp32 source
  // clip keeps its io type on both ends (like the MIGX plugin and vsort's
  // output_format=0): the converter inserts an input Cast node, which the
  // engine folds into a device-side fp32 -> fp16 convert of the raw upload,
  // and an output Cast node, which the engine folds into a device-side
  // fp16 -> fp32 convert of the download. Compute is fp16 either way; only
  // the clip format differs (GRAYS for fp32 clips, GRAYH otherwise).
  const bool clip_fp32 = d->vi->format && d->vi->format->sampleType == stFloat &&
                         d->vi->format->bitsPerSample == 32;
  if (d->output_fp16) {
    std::unordered_set<std::string> blacklist;
    convert_float_to_float16(model, /*force_fp16_initializers=*/true, blacklist,
                             /*cast_input=*/clip_fp32, /*cast_output=*/clip_fp32);
  }

  d->engines.reserve(static_cast<size_t>(d->num_streams));
  for (int i = 0; i < d->num_streams; ++i) {
    auto* e = new vship::HipEngine(d->device_id);
    std::string error;
    if (!e->Build(model, error)) {
      delete e;
      for (auto* prev : d->engines) delete prev;
      set_error(error);
      return;
    }
    if (!e->GetOutputShape(probe_in, probe_out) || probe_out.size() != 4) {
      delete e;
      for (auto* prev : d->engines) delete prev;
      set_error("model shape probe failed");
      return;
    }
    d->engines.push_back(e);
  }
  d->output_fp32 = !d->engines.empty() && d->engines[0]->OutputIsFp32();

  const int in_w = d->vi->width;
  const int in_h = d->vi->height;
  const int scale_w = static_cast<int>(probe_out[3] / probe_in[3]);
  const int scale_h = static_cast<int>(probe_out[2] / probe_in[2]);
  VSVideoInfo out_vi = *d->vi;
  out_vi.width = in_w * scale_w;
  out_vi.height = in_h * scale_h;
  out_vi.format = d->output_fp32 ? vsapi->getFormatPreset(pfGrayS, core)
                : d->output_fp16 ? vsapi->getFormatPreset(pfGrayH, core)
                                 : vsapi->getFormatPreset(pfGray16, core);
  vsapi->setVideoInfo(&out_vi, 1, node);
  } catch (const std::exception& e) {
    vsapi->setError(out, vs_error(std::string("exception: ") + e.what()).c_str());
    delete d;
    *instanceData = nullptr;
  } catch (...) {
    vsapi->setError(out, vs_error("unknown exception").c_str());
    delete d;
    *instanceData = nullptr;
  }
}

static void VS_CC hipFree(void* instanceData, VSCore* core, const VSAPI* vsapi) {
  auto* d = static_cast<HipData*>(instanceData);
  if (d) {
    vsapi->freeNode(d->node);
    for (auto* e : d->engines) delete e;
    delete d;
  }
}

static void VS_CC hipCreate(const VSMap* in, VSMap* out, void* userData, VSCore* core,
                            const VSAPI* vsapi) {
  auto set_error = [&](const std::string& msg) {
    vsapi->setError(out, vs_error(msg).c_str());
  };

  const int num_clips = vsapi->propNumElements(in, "clips");
  if (num_clips != 1) {
    set_error("expects exactly 1 input clip");
    return;
  }
  auto* d = new HipData;
  d->node = vsapi->propGetNode(in, "clips", 0, nullptr);
  d->vi = vsapi->getVideoInfo(d->node);

  // Validate the clip format: single-plane 8/16-bit int or fp32/fp16 float.
  const auto* fmt = d->vi->format;
  if (!fmt || fmt->numPlanes != 1 || fmt->colorFamily != cmGray) {
    set_error("expects a single-plane gray clip");
    delete d;
    return;
  }
  if (!((fmt->sampleType == stInteger && (fmt->bitsPerSample == 8 || fmt->bitsPerSample == 16)) ||
        (fmt->sampleType == stFloat && (fmt->bitsPerSample == 16 || fmt->bitsPerSample == 32)))) {
    set_error("expects GRAY8/GRAY16/GRAYH/GRAYS input");
    delete d;
    return;
  }

  vsapi->createFilter(in, out, "Model", hipInit, hipGetFrame, hipFree, fmParallel, 0, d, core);
}

// ---------------------------------------------------------------------------
// Version + device properties helpers
// ---------------------------------------------------------------------------

static void VS_CC hipVersion(const VSMap*, VSMap* out, void*, VSCore*, const VSAPI* vsapi) {
  vsapi->propSetData(out, "version", "0.2.0", -1, paReplace);
  int count = 0;
  if (hipGetDeviceCount(&count) == hipSuccess) {
    vsapi->propSetInt(out, "device_count", count, paReplace);
  }
}

static void VS_CC hipDeviceProperties(const VSMap* in, VSMap* out, void*, VSCore*,
                                      const VSAPI* vsapi) {
  const int device_id = static_cast<int>(vsapi->propGetInt(in, "device_id", 0, nullptr));
  hipDeviceProp_t props{};
  if (hipGetDeviceProperties(&props, device_id) != hipSuccess) {
    vsapi->setError(out, "hipDeviceProperties: failed to query device");
    return;
  }
  vsapi->propSetData(out, "name", props.name, -1, paReplace);
  vsapi->propSetInt(out, "pci_device_id", static_cast<int64_t>(props.pciDeviceID), paReplace);
  vsapi->propSetInt(out, "clock_rate", static_cast<int64_t>(props.clockRate), paReplace);
  vsapi->propSetInt(out, "global_memory", static_cast<int64_t>(props.totalGlobalMem), paReplace);
}

// ---------------------------------------------------------------------------
// Plugin entry point
// ---------------------------------------------------------------------------

VS_EXTERNAL_API(void) VapourSynthPluginInit(VSConfigPlugin configFunc,
                                            VSRegisterFunction registerFunc,
                                            VSPlugin* plugin) {
  myself = plugin;

  configFunc("io.github.thefeeltrain.vs_hip", "hip",
             "Standalone HIP ML filter (winograd WMMA, no ORT)",
             VAPOURSYNTH_API_VERSION, 1, plugin);

  registerFunc("Model",
               "clips:clip[];"
               "network_path:data;"
               "overlap:int[]:opt;"
               "tilesize:int[]:opt;"
               "device_id:int:opt;"
               "num_streams:int:opt;"
               "fp16:int:opt;",
               hipCreate, nullptr, plugin);

  registerFunc("Version", "", hipVersion, nullptr, plugin);
  registerFunc("DeviceProperties", "device_id:int:opt;", hipDeviceProperties, nullptr, plugin);
}
