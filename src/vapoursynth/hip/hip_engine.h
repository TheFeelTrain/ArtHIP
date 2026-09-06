// Standalone HIP inference engine for the ArtCNN-style fp16 conv chain.
// Parses an ONNX model directly (ModelProto) and runs the fused graph on the
// GPU with the same winograd WMMA kernels as the HIP execution provider
// (src/hip/hip_kernels.h) - no ONNX Runtime involved.
#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <onnx/onnx_pb.h>

namespace vship {

enum class OpType {
  Conv,
  Sigmoid,
  Mul,
  Add,
  Clip,
  DepthToSpace,
};

struct OpSpec {
  OpType type{OpType::Conv};
  std::string in0;
  std::string in1;  // empty for unary ops; for Conv this is the weight name
  std::string out;

  std::string conv_bias;  // Conv: bias initializer name (may be empty)
  bool do_silu = false;   // fused epilogue: out = silu(conv_out)
  bool do_add = false;    // fused epilogue: out = conv_out + add_input
  std::string add_input;

  int stride_h = 1;
  int stride_w = 1;
  int pad_h = 0;
  int pad_w = 0;
  int dil_h = 1;
  int dil_w = 1;
  int group = 1;

  float clip_min = 0.0f;
  float clip_max = 1.0f;

  int blocksize = 1;
  bool fuse_clip = false;  // DepthToSpace: fold a following Clip into the kernel
  bool fuse_dts = false;   // Conv: write the DTS output directly (tail conv)
};

struct WeightSpec {
  std::vector<uint8_t> data;
  std::vector<int64_t> shape;
};

struct TensorInfo {
  std::vector<int64_t> shape;
  int buffer_index = -1;
};

struct HipBuffer {
  void* ptr = nullptr;
  size_t size = 0;
};

struct DeviceOp {
  OpType type{OpType::Conv};
  std::vector<uint32_t> params;
  uint32_t dispatch_x = 1;
  uint32_t dispatch_y = 1;
  uint32_t dispatch_z = 1;
  std::vector<int> input_buffers;
  int output_buffer = -1;
};

// Standalone engine: compile the model once (Build), allocate device state
// per input shape (EnsureBuilt), then run frames (Run). Thread-safe via an
// internal mutex (VapourSynth may call from multiple worker threads).
class HipEngine {
 public:
  HipEngine(int device_id = 0);
  ~HipEngine();

  // Parse + fuse the model. Returns false with a message on failure.
  bool Build(const ONNX_NAMESPACE::ModelProto& model, std::string& error);

  // Compute the output shape for a concrete input shape (NCHW).
  bool GetOutputShape(const std::vector<int64_t>& input_shape,
                      std::vector<int64_t>& output_shape) const;

  // Run inference. input/output hold NCHW [1,1,H,W] data in the layout
  // the model expects (the engine handles the internal NHWC transpose).
  // Element type follows the model IO: fp16 normally, raw fp32 when
  // InputIsFp32()/OutputIsFp32() (fp32-input models / fp32-output models).
  bool Run(const void* input_data, const std::vector<int64_t>& input_shape,
           void* output_data, std::string& error);

  int64_t OutputElements(const std::vector<int64_t>& input_shape);

  // Host-side helpers for the plugin's frame conversion.
  static bool FloatToHalfBitsRNE(float f, uint16_t& h);
  static float HalfBitsToFloat(uint16_t h);

  size_t InputBytes(const std::vector<int64_t>& input_shape) const;
  size_t OutputBytes(const std::vector<int64_t>& input_shape) const;

  // True when the model takes an fp32 input (Run() then expects raw fp32
  // host data, converted to fp16 on the host inside Run()).
  bool InputIsFp32() const { return input_is_fp32_; }

  // True when the model produces an fp32 output (internal compute is still
  // fp16; Run() converts on-device and returns raw fp32 host data).
  bool OutputIsFp32() const { return output_is_fp32_; }

 private:
  bool PropagateShapes(const std::vector<int64_t>& input_shape,
                       std::unordered_map<std::string, std::vector<int64_t>>& shapes) const;
  bool EnsureBuilt(const std::vector<int64_t>& input_shape, std::string& error);
  void DestroyDeviceState();

  int device_id_ = 0;
  void* stream_ = nullptr;

  std::string input_name_;
  std::string output_name_;
  std::vector<OpSpec> ops_;
  std::unordered_map<std::string, WeightSpec> weights_;

  bool built_ = false;
  std::vector<int64_t> built_input_shape_;
  std::vector<HipBuffer> buffers_;
  std::unordered_map<std::string, TensorInfo> tensor_map_;
  std::vector<DeviceOp> device_ops_;
  int input_buffer_ = -1;
  int output_buffer_ = -1;
  void* input_staging_ = nullptr;
  void* output_staging_ = nullptr;
  // fp32-input models: pinned host + device fp32 buffers; a small kernel
  // converts to the fp16 input buffer on-device.
  void* input_staging_f32_ = nullptr;
  void* input_f32_dev_ = nullptr;
  // fp32-output models: device fp32 buffer (on-device cast target) + pinned
  // host staging for the float download.
  void* output_staging_f32_ = nullptr;
  void* output_f32_dev_ = nullptr;
  size_t input_bytes_ = 0;
  size_t output_bytes_ = 0;
  bool input_is_fp32_ = false;
  mutable bool input_is_fp32_dbg_done_ = false;
  bool output_is_fp32_ = false;
  size_t input_ort_bytes_ = 0;
  size_t output_ort_bytes_ = 0;

  std::mutex mutex_;
};

}  // namespace vship
