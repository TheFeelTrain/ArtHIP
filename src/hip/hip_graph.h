// HIPExecutionProvider - Compiled graph representation + HIP execution engine.
#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/providers/shared_library/provider_api.h"

namespace onnxruntime {

class GraphViewer;

namespace hip {

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

  // Conv: bias initializer name (may be empty if the model has no bias).
  std::string conv_bias;
  // Conv epilogues (fused activations).
  bool do_silu = false;           // out = silu(conv_out)
  bool do_add = false;            // out = conv_out + add_input
  std::string add_input;          // tensor added to the conv output

  // Conv attributes.
  int stride_h = 1;
  int stride_w = 1;
  int pad_h = 0;
  int pad_w = 0;
  int dil_h = 1;
  int dil_w = 1;
  int group = 1;

  // Clip bounds.
  float clip_min = 0.0f;
  float clip_max = 1.0f;

  // DepthToSpace (+ optional fused trailing Clip).
  int blocksize = 1;
  bool do_clip = false;
};

struct WeightSpec {
  std::vector<uint8_t> data;
  std::vector<int64_t> shape;
  int buffer_index = -1;
};

// Per-tensor device-side info used at run time.
struct TensorInfo {
  std::vector<int64_t> shape;
  int buffer_index = -1;
  bool is_constant = false;
};

// A HIP device allocation.
struct HipBuffer {
  void* ptr = nullptr;
  size_t size = 0;
};

// One executable op instance (opaque params for the HIP launch).
struct DeviceOp {
  OpType type{OpType::Conv};
  // Opaque push-constant-style payload, sized per op type.
  std::vector<uint32_t> params;
  uint32_t dispatch_x = 1;
  uint32_t dispatch_y = 1;
  uint32_t dispatch_z = 1;
  // Buffer indices this op reads from / writes to.
  std::vector<int> input_buffers;
  int output_buffer = -1;
};

// Minimal HIP context: device handle + a persistent stream.
class HipContext {
 public:
  HipContext();
  ~HipContext();

  bool Initialize();
  void* Stream() const { return stream_; }

 private:
  void* stream_ = nullptr;
  int device_ = 0;
  bool initialized_ = false;
};

// The compiled program for a single fused node.
class HipGraph {
 public:
  explicit HipGraph(std::shared_ptr<HipContext> ctx);
  ~HipGraph();

  // Build the symbolic representation from the fused subgraph (called during Compile).
  common::Status CompileGraph(const GraphViewer& graph);

  // Run inference for the given input. input_shape describes the concrete graph input.
  common::Status Run(const void* input_data, size_t input_bytes,
                     const std::vector<int64_t>& input_shape,
                     void* output_data, size_t output_bytes);

  // Compute the output shape for a given concrete input shape.
  common::Status GetOutputShape(const std::vector<int64_t>& input_shape,
                                std::vector<int64_t>& output_shape);

 private:
  common::Status PropagateShapes(const std::vector<int64_t>& input_shape,
                                 std::unordered_map<std::string, std::vector<int64_t>>& shapes) const;
  common::Status EnsureBuilt(const std::vector<int64_t>& input_shape);
  void DestroyDeviceState();

  std::shared_ptr<HipContext> ctx_;

  std::string input_name_;
  std::string output_name_;
  std::vector<OpSpec> ops_;
  std::unordered_map<std::string, WeightSpec> weights_;

  // Device-side state (valid when built_ is true).
  bool built_ = false;
  std::vector<int64_t> built_input_shape_;
  std::vector<HipBuffer> buffers_;
  std::unordered_map<std::string, TensorInfo> tensor_map_;
  std::vector<DeviceOp> device_ops_;
  int input_buffer_ = -1;
  int output_buffer_ = -1;
  void* input_staging_ = nullptr;
  void* output_staging_ = nullptr;
  size_t input_bytes_ = 0;
  size_t output_bytes_ = 0;
  // Boundary Cast nodes (fp32<->fp16) are folded into the IO conversion.
  bool input_is_fp32_ = false;
  bool output_is_fp32_ = false;
  size_t input_ort_bytes_ = 0;
  size_t output_ort_bytes_ = 0;

  std::mutex mutex_;
  bool profiling_ = false;
};

}  // namespace hip
}  // namespace onnxruntime
