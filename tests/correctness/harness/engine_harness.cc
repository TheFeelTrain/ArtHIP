// ctypes driver for the standalone HipEngine (tests/correctness).
//
// The Python side owns input generation and all reference math; this driver
// only maps the C++ engine API onto a small C ABI so tests can exercise the
// real production source (src/vapoursynth/hip_engine.cc) plus the real
// gfx1100 kernels.
//
// It is built as a shared library, mirroring src/vapoursynth/build_hip.sh:
// protobuf/absl symbols resolve through libonnx.so at load time, which a plain
// executable cannot do with the system link line.
//
// Return codes: 0 = ok, 1 = build/shape/argument failure, 2 = run failure.
//   "failure" means the engine rejected the model or the request (for example a
//   1x1 Conv, batch > 1, or a shape that cannot be scheduled). Tests assert on
//   the message text, so it is surfaced verbatim.
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <hip/hip_runtime.h>
#include <onnx/onnx_pb.h>

#include "hip_engine.h"

namespace {

constexpr int kMaxRank = 8;

bool LoadModel(const char* path, ONNX_NAMESPACE::ModelProto& model) {
  FILE* f = fopen(path, "rb");
  if (!f) return false;
  const bool ok = model.ParseFromFileDescriptor(fileno(f));
  fclose(f);
  return ok;
}

void CopyOut(const std::string& s, char* dst, int len) {
  if (!dst || len <= 0) return;
  size_t n = s.size();
  if (n > static_cast<size_t>(len) - 1) n = static_cast<size_t>(len) - 1;
  std::memcpy(dst, s.data(), n);
  dst[n] = '\0';
}

void WriteShape(const std::vector<int64_t>& shape, int64_t* out_shape, int* out_rank) {
  if (out_rank) *out_rank = static_cast<int>(shape.size());
  if (!out_shape) return;
  for (size_t i = 0; i < shape.size() && i < kMaxRank; ++i) out_shape[i] = shape[i];
}

size_t ElemSize(bool fp32) { return fp32 ? 4u : 2u; }

}  // namespace

extern "C" int engine_device_count(int* count) {
  if (!count) return 1;
  if (hipInit(0) != hipSuccess) return 1;
  int n = 0;
  if (hipGetDeviceCount(&n) != hipSuccess) return 1;
  *count = n;
  return 0;
}

extern "C" int engine_probe(const char* model_path, int N, int H, int W, int C, int device,
                            char* err, int errlen,
                            int* in_fp32, int* out_fp32, int* cin, int* cout,
                            int64_t* out_shape, int* out_rank) {
  ONNX_NAMESPACE::ModelProto model;
  if (!LoadModel(model_path, model)) {
    CopyOut("failed to parse model file", err, errlen);
    return 1;
  }
  vship::HipEngine eng(device);
  std::string error;
  if (!eng.Build(model, error)) {
    CopyOut(error, err, errlen);
    return 1;
  }
  if (in_fp32) *in_fp32 = eng.InputIsFp32() ? 1 : 0;
  if (out_fp32) *out_fp32 = eng.OutputIsFp32() ? 1 : 0;
  if (cin) *cin = eng.InputChannels();
  if (cout) *cout = eng.OutputChannels();

  std::vector<int64_t> in_shape{N, C, H, W};
  std::vector<int64_t> shape;
  if (!eng.GetOutputShape(in_shape, shape)) {
    CopyOut("shape propagation failed", err, errlen);
    return 1;
  }
  WriteShape(shape, out_shape, out_rank);
  CopyOut("", err, errlen);
  return 0;
}

extern "C" int engine_run(const char* model_path, int N, int H, int W, int C, int device,
                          const void* input, size_t input_bytes,
                          void* out, size_t cap, size_t* out_bytes, int* out_fp32,
                          int64_t* out_shape, int* out_rank,
                          char* err, int errlen) {
  ONNX_NAMESPACE::ModelProto model;
  if (!LoadModel(model_path, model)) {
    CopyOut("failed to parse model file", err, errlen);
    return 1;
  }
  vship::HipEngine eng(device);
  std::string error;
  if (!eng.Build(model, error)) {
    CopyOut(error, err, errlen);
    return 1;
  }

  std::vector<int64_t> in_shape{N, C, H, W};
  std::vector<int64_t> shape;
  if (!eng.GetOutputShape(in_shape, shape) || shape.size() != 4) {
    CopyOut("shape propagation failed", err, errlen);
    return 1;
  }
  const bool out_is_fp32 = eng.OutputIsFp32();
  const size_t out_numel = static_cast<size_t>(shape[0]) * shape[1] * shape[2] * shape[3];
  const size_t needed = out_numel * ElemSize(out_is_fp32);
  if (needed > cap) {
    CopyOut("output buffer too small", err, errlen);
    return 1;
  }
  // The engine memcpys the whole input; a short buffer would read out of bounds.
  const size_t expected_in = static_cast<size_t>(N) * C * H * W * ElemSize(eng.InputIsFp32());
  if (input == nullptr || input_bytes != expected_in) {
    CopyOut("input buffer size mismatch (expected " + std::to_string(expected_in) + " bytes, got " +
                std::to_string(input_bytes) + ")",
            err, errlen);
    return 1;
  }

  std::string run_err;
  if (!eng.Run(input, in_shape, out, run_err)) {
    CopyOut(run_err, err, errlen);
    return 2;
  }
  if (out_bytes) *out_bytes = needed;
  if (out_fp32) *out_fp32 = out_is_fp32 ? 1 : 0;
  WriteShape(shape, out_shape, out_rank);
  CopyOut("", err, errlen);
  return 0;
}

// P1-5: an oversized activation allocation must be reported, and the same
// engine must still serve a valid frame afterwards (partial device state
// unwound). The failing call allocates before reading input, so only the
// recovery input is needed; the recovery shape is fixed at 16x16.
extern "C" int engine_alloc_failure_then_run(const char* model_path,
                                             int N, int bigH, int bigW, int C, int device,
                                             const void* recovery_input,
                                             size_t recovery_input_bytes,
                                             void* out, size_t cap, size_t* out_bytes,
                                             int* out_fp32, int64_t* out_shape, int* out_rank,
                                             char* failerr, int failerrlen,
                                             char* runerr, int runerrlen) {
  ONNX_NAMESPACE::ModelProto model;
  if (!LoadModel(model_path, model)) {
    CopyOut("failed to parse model file", failerr, failerrlen);
    return 1;
  }
  vship::HipEngine eng(device);
  std::string error;
  if (!eng.Build(model, error)) {
    CopyOut("build failed: " + error, failerr, failerrlen);
    return 1;
  }

  char dummy[8] = {0};
  std::string first_err;
  std::vector<int64_t> big_shape{N, C, bigH, bigW};
  if (eng.Run(nullptr, big_shape, dummy, first_err)) {
    CopyOut("expected the oversized run to fail but it succeeded", failerr, failerrlen);
    return 1;
  }
  CopyOut(first_err, failerr, failerrlen);

  const int recH = 16;
  const int recW = 16;
  std::vector<int64_t> in_shape{N, C, recH, recW};
  std::vector<int64_t> shape;
  if (!eng.GetOutputShape(in_shape, shape) || shape.size() != 4) {
    CopyOut("recovery shape probe failed", runerr, runerrlen);
    return 2;
  }
  const bool out_is_fp32 = eng.OutputIsFp32();
  const size_t out_numel = static_cast<size_t>(shape[0]) * shape[1] * shape[2] * shape[3];
  const size_t needed = out_numel * ElemSize(out_is_fp32);
  if (needed > cap) {
    CopyOut("recovery output buffer too small", runerr, runerrlen);
    return 2;
  }
  const size_t expected_in = static_cast<size_t>(N) * C * recH * recW * ElemSize(eng.InputIsFp32());
  if (recovery_input == nullptr || recovery_input_bytes != expected_in) {
    CopyOut("recovery input buffer size mismatch", runerr, runerrlen);
    return 2;
  }
  std::string second_err;
  if (!eng.Run(recovery_input, in_shape, out, second_err)) {
    CopyOut(second_err, runerr, runerrlen);
    return 2;
  }
  if (out_bytes) *out_bytes = needed;
  if (out_fp32) *out_fp32 = out_is_fp32 ? 1 : 0;
  WriteShape(shape, out_shape, out_rank);
  CopyOut("", runerr, runerrlen);
  return 0;
}
