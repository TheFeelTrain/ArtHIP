// HIPExecutionProvider - Compiled graph + HIP execution engine implementation.
#include "hip_graph.h"

#include <cstring>
#include <chrono>
#include <cstdlib>
#include <numeric>
#include <unordered_set>

#include <hip/hip_runtime.h>

#include "core/common/common.h"
#include "hip_kernels.h"

namespace onnxruntime {
namespace hip {

namespace {

constexpr uint32_t kPointwiseLocal = 256;

uint32_t DivCeil(uint64_t a, uint32_t b) {
  return static_cast<uint32_t>((a + b - 1) / b);
}

// Winograd WMMA: uint H,W,C,M,M_pad,C_pad,tiles_w,do_silu,do_add (36 bytes)
struct WinogradPush {
  uint32_t h, w, c, m, m_pad, c_pad, tiles_w, do_silu, do_add;
};

// Unary (sigmoid/clip): uint numel, op; float min, max
struct UnaryPush {
  uint32_t numel;
  uint32_t op;
  float min_val;
  float max_val;
};

// Binary (add/mul): uint numel, op
struct BinaryPush {
  uint32_t numel;
  uint32_t op;
};

// DepthToSpace: uint C,H,W,B (+ clip fusion: uint do_clip, float min, max)
struct DtsPush {
  uint32_t c, h, w, b;
  uint32_t do_clip;
  float min_val, max_val;
};

// The conv shader assumes kernel 3x3, stride 1, dilation 1 and equal padding of 1.
bool IsSupportedConv(const OpSpec& op) {
  return op.stride_h == 1 && op.stride_w == 1 && op.dil_h == 1 && op.dil_w == 1 &&
         op.pad_h == 1 && op.pad_w == 1 && op.group == 1;
}

int64_t NumElements(const std::vector<int64_t>& shape) {
  return std::accumulate(shape.begin(), shape.end(), int64_t{1}, std::multiplies<int64_t>());
}

}  // namespace

// ---------------------------------------------------------------------------
// HipContext
// ---------------------------------------------------------------------------

HipContext::HipContext() = default;
HipContext::~HipContext() {
  if (stream_) {
    hipSetDevice(device_);
    hipStreamDestroy(static_cast<hipStream_t>(stream_));
  }
}

common::Status HipContext::SetDevice() const {
  hipError_t e = hipSetDevice(device_);
  if (e != hipSuccess) {
    return Status(common::ONNXRUNTIME, common::FAIL,
                  "HIP EP: hipSetDevice(" + std::to_string(device_) + ") failed: " +
                      hipGetErrorString(e));
  }
  return Status::OK();
}

bool HipContext::Initialize(int device_id) {
  if (initialized_) {
    return true;
  }
  device_ = device_id;
  if (hipInit(0) != hipSuccess) {
    fprintf(stderr, "[hip] hipInit failed\n");
    return false;
  }
  int device_count = 0;
  if (hipGetDeviceCount(&device_count) != hipSuccess || device_count == 0) {
    fprintf(stderr, "[hip] no HIP devices found\n");
    return false;
  }
  if (device_ < 0 || device_ >= device_count) {
    fprintf(stderr, "[hip] requested device %d is out of range (found %d devices)\n", device_,
            device_count);
    return false;
  }
  hipError_t se = hipSetDevice(device_);
  if (se != hipSuccess) {
    fprintf(stderr, "[hip] hipSetDevice(%d) failed: %s\n", device_, hipGetErrorString(se));
    return false;
  }
  hipDeviceProp_t props{};
  if (hipGetDeviceProperties(&props, device_) != hipSuccess) {
    fprintf(stderr, "[hip] hipGetDeviceProperties(%d) failed\n", device_);
    return false;
  }
  fprintf(stderr, "[hip] using device %d: %s\n", device_, props.name);
  if (hipStreamCreateWithFlags(reinterpret_cast<hipStream_t*>(&stream_),
                               hipStreamNonBlocking) != hipSuccess) {
    fprintf(stderr, "[hip] hipStreamCreate failed\n");
    stream_ = nullptr;
    return false;
  }
  initialized_ = true;
  return true;
}

// ---------------------------------------------------------------------------
// HipGraph
// ---------------------------------------------------------------------------

HipGraph::HipGraph(std::shared_ptr<HipContext> ctx) : ctx_(std::move(ctx)) {
  const char* prof = std::getenv("HIP_PROFILE");
  profiling_ = (prof != nullptr && prof[0] == '1');
}

HipGraph::~HipGraph() {
  DestroyDeviceState();
}

void HipGraph::DestroyDeviceState() {
  // Freeing device memory requires the owning device to be current on this
  // thread; the destructor and rebuild paths can run on any ORT thread.
  if (ctx_) hipSetDevice(ctx_->DeviceId());
  for (auto& buf : buffers_) {
    if (buf.ptr) {
      hipFree(buf.ptr);
    }
  }
  buffers_.clear();
  tensor_map_.clear();
  device_ops_.clear();
  // A partially built engine must never retain a stale buffer index.
  input_buffer_ = -1;
  output_buffer_ = -1;
  if (input_staging_) {
    hipFreeHost(input_staging_);
    input_staging_ = nullptr;
  }
  if (output_staging_) {
    hipFreeHost(output_staging_);
    output_staging_ = nullptr;
  }
  built_ = false;
}

common::Status HipGraph::CompileGraph(const GraphViewer& graph) {
  const auto& inputs = graph.GetInputs();
  const auto& outputs = graph.GetOutputs();
  if (inputs.empty() || outputs.empty()) {
    return Status(common::ONNXRUNTIME, common::FAIL, "HIP EP: graph must have at least one input and output.");
  }
  input_name_ = inputs[0]->Name();
  output_name_ = outputs[0]->Name();

  // Fold boundary Cast nodes (fp32<->fp16) into the IO conversion so the
  // whole conv chain stays on this EP.
  std::unordered_set<NodeIndex> consumed;
  {
    const auto& bnodes = graph.GetNodesInTopologicalOrder();
    for (const auto& node_index : bnodes) {
      const Node* node = graph.GetNode(node_index);
      if (node == nullptr || node->OpType() != "Cast") {
        continue;
      }
      auto id = node->InputDefs();
      auto od = node->OutputDefs();
      if (id.empty() || od.empty() || id[0] == nullptr || od[0] == nullptr ||
          !id[0]->Exists() || !od[0]->Exists() || !id[0]->TypeAsProto() || !od[0]->TypeAsProto() ||
          !id[0]->TypeAsProto()->has_tensor_type() || !od[0]->TypeAsProto()->has_tensor_type()) {
        continue;
      }
      const int in_t = id[0]->TypeAsProto()->tensor_type().elem_type();
      const int out_t = od[0]->TypeAsProto()->tensor_type().elem_type();
      if (in_t == ONNX_NAMESPACE::TensorProto_DataType_FLOAT &&
          out_t == ONNX_NAMESPACE::TensorProto_DataType_FLOAT16) {
        // Input cast: the fp16 tensor after it becomes the graph input.
        bool feeds_graph_input = false;
        for (const auto* gi : inputs) {
          if (gi != nullptr && gi->Exists() && gi->Name() == id[0]->Name()) {
            feeds_graph_input = true;
            break;
          }
        }
        if (feeds_graph_input) {
          input_is_fp32_ = true;
          input_name_ = od[0]->Name();
          consumed.insert(node_index);
        }
      } else if (in_t == ONNX_NAMESPACE::TensorProto_DataType_FLOAT16 &&
                 out_t == ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
        // Output cast: the fp16 tensor before it becomes the graph output.
        bool produces_graph_output = false;
        for (const auto* go : outputs) {
          if (go != nullptr && go->Exists() && go->Name() == od[0]->Name()) {
            produces_graph_output = true;
            break;
          }
        }
        if (produces_graph_output) {
          output_is_fp32_ = true;
          output_name_ = id[0]->Name();
          consumed.insert(node_index);
        }
      }
    }
  }

  auto read_initializer = [&graph](const std::string& name, std::vector<uint8_t>& data, std::vector<int64_t>& dims) -> common::Status {
    const ONNX_NAMESPACE::TensorProto* tensor_proto = nullptr;
    if (!graph.GetInitializedTensor(name, tensor_proto)) {
      return Status(common::ONNXRUNTIME, common::FAIL,
                    "HIP EP: expected initializer '" + name + "' was not found.");
    }
    data.clear();
    auto st = utils::UnpackInitializerData(*tensor_proto, data);
    if (!st.IsOK()) {
      return st;
    }
    dims.clear();
    const auto& d = tensor_proto->dims();
    for (int i = 0; i < d.size(); ++i) {
      dims.push_back(d[i]);
    }
    return Status::OK();
  };

  // Nodes that have been folded into a fused conv op are skipped in the main loop.
  const auto& nodes_in_order = graph.GetNodesInTopologicalOrder();

  for (const auto& node_index : nodes_in_order) {
    if (consumed.count(node_index)) {
      continue;
    }
    const Node* node = graph.GetNode(node_index);
    if (node == nullptr) {
      continue;
    }
    const std::string& optype = node->OpType();
    auto input_defs = node->InputDefs();
    auto output_defs = node->OutputDefs();
    if (output_defs.empty() || output_defs[0]->Name().empty()) {
      continue;
    }

    auto in_name = [&](size_t i) -> std::string {
      if (i < input_defs.size() && input_defs[i] != nullptr) {
        return input_defs[i]->Name();
      }
      return "";
    };

    OpSpec op;
    op.out = output_defs[0]->Name();

    if (optype == "Conv") {
      op.type = OpType::Conv;
      op.in0 = in_name(0);
      const std::string weight_name = in_name(1);
      const std::string bias_name = in_name(2);
      op.in1 = weight_name;
      op.conv_bias = bias_name;

      const auto& attrs = node->GetAttributes();
      if (attrs.count("strides")) {
        const auto& a = attrs.at("strides");
        if (a.ints_size() >= 2) {
          op.stride_h = static_cast<int>(a.ints(0));
          op.stride_w = static_cast<int>(a.ints(1));
        }
      }
      if (attrs.count("dilations")) {
        const auto& a = attrs.at("dilations");
        if (a.ints_size() >= 2) {
          op.dil_h = static_cast<int>(a.ints(0));
          op.dil_w = static_cast<int>(a.ints(1));
        }
      }
      if (attrs.count("group")) {
        op.group = static_cast<int>(attrs.at("group").i());
      }
      if (attrs.count("auto_pad")) {
        const std::string auto_pad = attrs.at("auto_pad").s();
        if (auto_pad == "SAME_UPPER" || auto_pad == "SAME_LOWER") {
          op.pad_h = 1;
          op.pad_w = 1;
        }
      }
      if (attrs.count("pads")) {
        const auto& a = attrs.at("pads");
        if (a.ints_size() >= 4) {
          op.pad_h = static_cast<int>(a.ints(0));
          op.pad_w = static_cast<int>(a.ints(2));
        }
      }

      if (!IsSupportedConv(op)) {
        return Status(common::ONNXRUNTIME, common::FAIL,
                      "HIP EP: unsupported Conv configuration (only 3x3 stride 1, pad 1 supported).");
      }

      ORT_RETURN_IF_ERROR(read_initializer(weight_name, weights_[weight_name].data, weights_[weight_name].shape));
      // Validate the weight geometry, dtype and storage size up front: the
      // fixed 3x3 packing loops read nine coefficients unconditionally.
      {
        const WeightSpec& w = weights_[weight_name];
        if (w.shape.size() != 4 || w.shape[2] != 3 || w.shape[3] != 3) {
          return Status(common::ONNXRUNTIME, common::FAIL,
                        "HIP EP: Conv weight '" + weight_name + "' must be a rank-4 3x3 tensor.");
        }
        if (w.shape[0] <= 0 || w.shape[1] <= 0) {
          return Status(common::ONNXRUNTIME, common::FAIL,
                        "HIP EP: Conv weight '" + weight_name + "' has a non-positive channel count.");
        }
        if (w.data.size() < static_cast<size_t>(NumElements(w.shape)) * 2) {
          return Status(common::ONNXRUNTIME, common::FAIL,
                        "HIP EP: Conv weight '" + weight_name + "' storage is truncated.");
        }
        if (w.shape[0] > 64) {
          return Status(common::ONNXRUNTIME, common::FAIL,
                        "HIP EP: Conv weight '" + weight_name +
                            "' requests more than 64 output channels, which the kernels do not cover.");
        }
      }
      if (!bias_name.empty()) {
        ORT_RETURN_IF_ERROR(read_initializer(bias_name, weights_[bias_name].data, weights_[bias_name].shape));
        const WeightSpec& b = weights_[bias_name];
        if (b.shape.size() != 1 || b.shape[0] != weights_[weight_name].shape[0]) {
          return Status(common::ONNXRUNTIME, common::FAIL,
                        "HIP EP: Conv bias '" + bias_name +
                            "' must be rank-1 with the weight's output-channel count.");
        }
        if (b.data.size() < static_cast<size_t>(b.shape[0]) * 2) {
          return Status(common::ONNXRUNTIME, common::FAIL,
                        "HIP EP: Conv bias '" + bias_name + "' storage is truncated.");
        }
      }

      // Fuse the SiLU (Sigmoid + Mul) epilogue into the conv.
      const std::string conv_out = op.out;
      for (auto it = node->OutputNodesBegin(); it != node->OutputNodesEnd(); ++it) {
        const Node& consumer = *it;
        if (consumer.OpType() == "Sigmoid") {
          const std::string& s_out = consumer.OutputDefs()[0]->Name();
          for (auto it2 = consumer.OutputNodesBegin(); it2 != consumer.OutputNodesEnd(); ++it2) {
            const Node& mul_node = *it2;
            if (mul_node.OpType() == "Mul" && mul_node.InputDefs().size() == 2) {
              auto md = mul_node.InputDefs();
              const std::string& n0 = md[0]->Name();
              const std::string& n1 = md[1]->Name();
              if ((n0 == conv_out && n1 == s_out) || (n0 == s_out && n1 == conv_out)) {
                op.do_silu = true;
                op.out = mul_node.OutputDefs()[0]->Name();
                consumed.insert(consumer.Index());  // Sigmoid
                consumed.insert(mul_node.Index());  // Mul
                break;
              }
            }
          }
        }
      }

    } else if (optype == "Sigmoid") {
      op.type = OpType::Sigmoid;
      op.in0 = in_name(0);
    } else if (optype == "Mul") {
      op.type = OpType::Mul;
      op.in0 = in_name(0);
      op.in1 = in_name(1);
    } else if (optype == "Add") {
      op.type = OpType::Add;
      op.in0 = in_name(0);
      op.in1 = in_name(1);
    } else if (optype == "Clip") {
      op.type = OpType::Clip;
      op.in0 = in_name(0);
      op.clip_min = 0.0f;
      op.clip_max = 1.0f;
      if (input_defs.size() > 1) {
        std::string min_name = in_name(1);
        if (!min_name.empty()) {
          std::vector<uint8_t> raw;
          std::vector<int64_t> dims;
          auto st = read_initializer(min_name, raw, dims);
          if (st.IsOK() && raw.size() >= 2) {
            MLFloat16 v;
            std::memcpy(&v, raw.data(), 2);
            op.clip_min = v.ToFloat();
          }
        }
      }
      if (input_defs.size() > 2) {
        std::string max_name = in_name(2);
        if (!max_name.empty()) {
          std::vector<uint8_t> raw;
          std::vector<int64_t> dims;
          auto st = read_initializer(max_name, raw, dims);
          if (st.IsOK() && raw.size() >= 2) {
            MLFloat16 v;
            std::memcpy(&v, raw.data(), 2);
            op.clip_max = v.ToFloat();
          }
        }
      }
    } else if (optype == "DepthToSpace") {
      op.type = OpType::DepthToSpace;
      op.in0 = in_name(0);
      const auto& attrs = node->GetAttributes();
      if (attrs.count("blocksize")) {
        op.blocksize = static_cast<int>(attrs.at("blocksize").i());
      }
      if (attrs.count("mode")) {
        const std::string mode = attrs.at("mode").s();
        if (mode != "DCR") {
          return Status(common::ONNXRUNTIME, common::FAIL,
                        "HIP EP: only DepthToSpace mode DCR is supported.");
        }
      }
    } else {
      return Status(common::ONNXRUNTIME, common::FAIL,
                    "HIP EP: unsupported op '" + optype + "' in fused graph.");
    }

    ops_.push_back(std::move(op));
  }

  // Pass 2: fuse residual Add nodes into the conv that produces their main input.
  {
    std::unordered_map<std::string, size_t> producer;
    for (size_t i = 0; i < ops_.size(); ++i) {
      producer[ops_[i].out] = i;
    }
    std::vector<size_t> remove_adds;
    for (size_t i = 0; i < ops_.size(); ++i) {
      if (ops_[i].type != OpType::Add) {
        continue;
      }
      const auto it_a = producer.find(ops_[i].in0);
      const auto it_b = producer.find(ops_[i].in1);
      if (it_a == producer.end() || it_b == producer.end()) {
        continue;
      }
      size_t main = (it_a->second >= it_b->second) ? it_a->second : it_b->second;
      if (ops_[main].type == OpType::Conv) {
        const std::string main_out = (it_a->second >= it_b->second) ? ops_[i].in0 : ops_[i].in1;
        if (ops_[main].out == main_out) {
          ops_[main].do_add = true;
          ops_[main].add_input = (it_a->second >= it_b->second) ? ops_[i].in1 : ops_[i].in0;
          ops_[main].out = ops_[i].out;
          remove_adds.push_back(i);
        }
      }
    }
    for (auto it = remove_adds.rbegin(); it != remove_adds.rend(); ++it) {
      ops_.erase(ops_.begin() + static_cast<std::ptrdiff_t>(*it));
    }
  }

  // Pass 3: fuse a trailing Clip into the DepthToSpace (the tail is always
  // DTS -> Clip; the kernel applies the clip inline, saving a full pass).
  {
    std::unordered_map<std::string, size_t> producer;
    for (size_t i = 0; i < ops_.size(); ++i) {
      producer[ops_[i].out] = i;
    }
    std::vector<size_t> remove_clips;
    for (size_t i = 0; i < ops_.size(); ++i) {
      if (ops_[i].type != OpType::Clip) {
        continue;
      }
      const auto it = producer.find(ops_[i].in0);
      if (it == producer.end() || ops_[it->second].type != OpType::DepthToSpace) {
        continue;
      }
      // The Clip input must have no other consumer.
      bool shared = false;
      for (size_t j = 0; j < ops_.size(); ++j) {
        if (j == i || j == it->second) {
          continue;
        }
        const OpSpec& other = ops_[j];
        if (other.in0 == ops_[i].in0 || other.in1 == ops_[i].in0 ||
            (other.do_add && other.add_input == ops_[i].in0)) {
          shared = true;
          break;
        }
      }
      if (shared) {
        continue;
      }
      OpSpec& dts = ops_[it->second];
      dts.clip_min = ops_[i].clip_min;
      dts.clip_max = ops_[i].clip_max;
      dts.do_clip = true;
      dts.out = ops_[i].out;
      remove_clips.push_back(i);
    }
    for (auto it = remove_clips.rbegin(); it != remove_clips.rend(); ++it) {
      ops_.erase(ops_.begin() + static_cast<std::ptrdiff_t>(*it));
    }
  }

  // Validate binary-op operands: only same-shape elementwise Mul/Add between
  // two activations produced inside this partition is implemented. Constants
  // and graph inputs would bind a null device buffer and fault in the kernel.
  {
    std::unordered_set<std::string> produced;
    for (const auto& op : ops_) produced.insert(op.out);
    auto require_activation = [&](const std::string& what, const std::string& name) -> common::Status {
      if (produced.count(name) == 0) {
        return Status(common::ONNXRUNTIME, common::FAIL,
                      "HIP EP: " + what + " operand '" + name +
                          "' is not an activation produced in this partition; constants, graph "
                          "inputs and broadcasting are not supported.");
      }
      return Status::OK();
    };
    for (const OpSpec& op : ops_) {
      if (op.type == OpType::Sigmoid || op.type == OpType::Clip) {
        ORT_RETURN_IF_ERROR(require_activation("unary op", op.in0));
      } else if (op.type == OpType::Mul) {
        ORT_RETURN_IF_ERROR(require_activation("Mul", op.in0));
        ORT_RETURN_IF_ERROR(require_activation("Mul", op.in1));
      } else if (op.type == OpType::Add) {
        ORT_RETURN_IF_ERROR(require_activation("Add", op.in0));
        ORT_RETURN_IF_ERROR(require_activation("Add", op.in1));
      } else if (op.type == OpType::Conv && op.do_add) {
        ORT_RETURN_IF_ERROR(require_activation("fused residual Add", op.add_input));
      }
    }
  }

  return Status::OK();
}

common::Status HipGraph::PropagateShapes(const std::vector<int64_t>& input_shape,
                                         std::unordered_map<std::string, std::vector<int64_t>>& shapes) const {
  if (input_shape.size() != 4) {
    return Status(common::ONNXRUNTIME, common::FAIL, "HIP EP: expected 4D input tensor.");
  }
  if (input_shape[0] != 1) {
    return Status(common::ONNXRUNTIME, common::FAIL,
                  "HIP EP: batch size must be 1 (the kernels have no batch dimension).");
  }
  shapes.clear();
  // All internal tensors use the NHWC [N,H,W,C] layout.
  shapes[input_name_] = {input_shape[0], input_shape[2], input_shape[3], input_shape[1]};

  for (const auto& op : ops_) {
    const auto& in = shapes.find(op.in0);
    if (in == shapes.end()) {
      return Status(common::ONNXRUNTIME, common::FAIL, "HIP EP: missing shape for '" + op.in0 + "'.");
    }
    const auto& s = in->second;
    if (s.size() != 4) {
      return Status(common::ONNXRUNTIME, common::FAIL, "HIP EP: unexpected tensor rank.");
    }
    std::vector<int64_t> out_shape;
    switch (op.type) {
      case OpType::Conv: {
        const auto& w = weights_.find(op.in1);
        if (w == weights_.end()) {
          return Status(common::ONNXRUNTIME, common::FAIL, "HIP EP: missing weights for '" + op.in1 + "'.");
        }
        if (w->second.shape.size() != 4 || w->second.shape[1] != s[3]) {
          return Status(common::ONNXRUNTIME, common::FAIL,
                        "HIP EP: Conv weight '" + op.in1 +
                            "' channel count does not match its input activation.");
        }
        int64_t m = w->second.shape[0];
        int64_t oh = (s[1] + 2 * op.pad_h - op.dil_h * 2 - 1) / op.stride_h + 1;
        int64_t ow = (s[2] + 2 * op.pad_w - op.dil_w * 2 - 1) / op.stride_w + 1;
        if (oh <= 0 || ow <= 0) {
          return Status(common::ONNXRUNTIME, common::FAIL, "HIP EP: degenerate Conv output shape.");
        }
        out_shape = {s[0], oh, ow, m};
        if (op.do_add) {
          const auto& add_shape = shapes.find(op.add_input);
          if (add_shape == shapes.end() || add_shape->second != out_shape) {
            return Status(common::ONNXRUNTIME, common::FAIL,
                          "HIP EP: fused residual add input must match the conv output shape.");
          }
        }
        break;
      }
      case OpType::Sigmoid:
      case OpType::Clip:
        out_shape = s;
        break;
      case OpType::Mul:
      case OpType::Add: {
        // Same-shape elementwise only; broadcasting would read out of bounds.
        const auto& rhs = shapes.find(op.in1);
        if (rhs == shapes.end() || rhs->second != s) {
          return Status(common::ONNXRUNTIME, common::FAIL,
                        "HIP EP: binary op requires both operands to have identical shapes "
                        "(broadcasting is not implemented).");
        }
        out_shape = s;
        break;
      }
      case OpType::DepthToSpace: {
        int64_t b = op.blocksize;
        if (b < 1) {
          return Status(common::ONNXRUNTIME, common::FAIL, "HIP EP: DepthToSpace block size must be positive.");
        }
        if (s[3] % (b * b) != 0) {
          return Status(common::ONNXRUNTIME, common::FAIL,
                        "HIP EP: DepthToSpace requires the input channel count to be divisible by blocksize^2.");
        }
        out_shape = {s[0], s[1] * b, s[2] * b, s[3] / (b * b)};
        break;
      }
    }
    shapes[op.out] = out_shape;
  }
  return Status::OK();
}

common::Status HipGraph::GetOutputShape(const std::vector<int64_t>& input_shape,
                                        std::vector<int64_t>& output_shape) {
  std::unordered_map<std::string, std::vector<int64_t>> shapes;
  ORT_RETURN_IF_ERROR(PropagateShapes(input_shape, shapes));
  const auto& it = shapes.find(output_name_);
  if (it == shapes.end()) {
    return Status(common::ONNXRUNTIME, common::FAIL, "HIP EP: missing output shape.");
  }
  const auto& nhwc = it->second;
  if (nhwc.size() != 4) {
    return Status(common::ONNXRUNTIME, common::FAIL, "HIP EP: unexpected output rank.");
  }
  // Return the output in NCHW order for the ORT tensor.
  output_shape = {nhwc[0], nhwc[3], nhwc[1], nhwc[2]};
  return Status::OK();
}

common::Status HipGraph::EnsureBuilt(const std::vector<int64_t>& input_shape) {
  if (built_ && input_shape == built_input_shape_) {
    return Status::OK();
  }
  if (!ctx_ || !ctx_->initialized()) {
    return Status(common::ONNXRUNTIME, common::FAIL, "HIP EP: device context is not initialized.");
  }
  // ORT may call Run() from any thread; HIP's current device is thread-local.
  ORT_RETURN_IF_ERROR(ctx_->SetDevice());

  if (input_shape.size() != 4) {
    return Status(common::ONNXRUNTIME, common::FAIL, "HIP EP: expected 4D input tensor.");
  }

  DestroyDeviceState();
  built_input_shape_ = input_shape;

  // --- Propagate shapes ---
  std::unordered_map<std::string, std::vector<int64_t>> shapes;
  ORT_RETURN_IF_ERROR(PropagateShapes(input_shape, shapes));

  if (shapes.find(output_name_) == shapes.end()) {
    return Status(common::ONNXRUNTIME, common::FAIL, "HIP EP: missing output shape.");
  }

  // --- Allocate buffers with liveness-based reuse ---
  std::unordered_map<std::string, int> last_use;
  for (size_t oi = 0; oi < ops_.size(); ++oi) {
    const auto& op = ops_[oi];
    last_use[op.in0] = static_cast<int>(oi);
    if (!op.in1.empty() && !weights_.count(op.in1)) {
      last_use[op.in1] = static_cast<int>(oi);
    }
    if (op.do_add && !op.add_input.empty() && !weights_.count(op.add_input)) {
      last_use[op.add_input] = static_cast<int>(oi);
    }
  }
  last_use[output_name_] = static_cast<int>(ops_.size());

  // (last_use, buffer_index) of tensors that can be reused once dead.
  std::vector<std::pair<int, int>> slots;
  auto alloc_tensor = [&](const std::string& name, const std::vector<int64_t>& shape,
                          const void* constant_data, size_t constant_bytes,
                          int producer) -> int {
    if (tensor_map_.count(name)) {
      return tensor_map_[name].buffer_index;
    }
    int64_t numel = NumElements(shape);
    size_t bytes = static_cast<size_t>(numel * 2);
    int buffer_index = -1;
    for (size_t s = 0; s < slots.size(); ++s) {
      if (slots[s].first < producer && buffers_[slots[s].second].size >= bytes) {
        buffer_index = slots[s].second;
        slots[s].first = last_use[name];
        break;
      }
    }
    if (buffer_index < 0) {
      void* ptr = nullptr;
      hipError_t err = hipMalloc(&ptr, bytes);
      if (err != hipSuccess) {
        fprintf(stderr, "[hip] hipMalloc(%zu) failed for '%s': %s\n", bytes, name.c_str(),
                hipGetErrorString(err));
        return -1;
      }
      HipBuffer buf{ptr, bytes};
      buffer_index = static_cast<int>(buffers_.size());
      buffers_.push_back(buf);
      slots.emplace_back(last_use[name], buffer_index);
    }
    TensorInfo info;
    info.shape = shape;
    info.buffer_index = buffer_index;
    info.is_constant = constant_data != nullptr;
    tensor_map_[name] = info;
    if (constant_data != nullptr) {
      hipError_t err = hipMemcpy(buffers_[buffer_index].ptr, constant_data, constant_bytes, hipMemcpyHostToDevice);
      if (err != hipSuccess) {
        fprintf(stderr, "[hip] hipMemcpy of constant '%s' failed: %s\n", name.c_str(),
                hipGetErrorString(err));
        return -1;
      }
    }
    return buffer_index;
  };

  auto fail_build = [&](const std::string& what) -> common::Status {
    DestroyDeviceState();
    return Status(common::ONNXRUNTIME, common::FAIL, "HIP EP: " + what);
  };

  input_buffer_ = alloc_tensor(input_name_, shapes[input_name_], nullptr, 0, 0);
  if (input_buffer_ < 0) return fail_build("failed to allocate the input buffer.");
  for (size_t oi = 0; oi < ops_.size(); ++oi) {
    const auto& op = ops_[oi];
    const int producer = static_cast<int>(oi);
    const auto& in_shape = shapes[op.in0];
    if (alloc_tensor(op.in0, in_shape, nullptr, 0, producer) < 0 ||
        (!op.in1.empty() && !weights_.count(op.in1) &&
         alloc_tensor(op.in1, shapes.count(op.in1) ? shapes[op.in1] : in_shape, nullptr, 0, producer) < 0) ||
        (op.do_add && !op.add_input.empty() && !weights_.count(op.add_input) &&
         alloc_tensor(op.add_input, shapes.count(op.add_input) ? shapes[op.add_input] : in_shape, nullptr, 0,
                      producer) < 0) ||
        alloc_tensor(op.out, shapes[op.out], nullptr, 0, producer) < 0) {
      return fail_build("failed to allocate an activation buffer.");
    }
  }
  output_buffer_ = tensor_map_[output_name_].buffer_index;

  // --- Build device ops ---
  device_ops_.reserve(ops_.size());
  for (const auto& op : ops_) {
    DeviceOp dop;
    dop.type = op.type;
    const auto& in_shape = shapes[op.in0];

    switch (op.type) {
      case OpType::Conv: {
        int in_idx = tensor_map_[op.in0].buffer_index;
        // Winograd F(2x2,3x3) weight transform: U = G g G^T, laid out as
        // U[wp][ko][c] fp16 (ko padded to M_pad, a multiple of 16) so the
        // WMMA A-operand is a contiguous 16x16 block per wp. Bias is appended
        // at the end (offset 16*M_pad*C).
        const auto& wspec = weights_.at(op.in1);
        const uint32_t M = static_cast<uint32_t>(wspec.shape[0]);
        const uint32_t C = static_cast<uint32_t>(wspec.shape[1]);
        const uint32_t H = static_cast<uint32_t>(in_shape[1]);
        const uint32_t W = static_cast<uint32_t>(in_shape[2]);
        const uint32_t M_pad = (M + 31u) / 32u * 32u;
        const uint32_t C_pad = (C + 15u) / 16u * 16u;
        const uint32_t kb_stride = M_pad / 16u;  // k-blocks per wp
        const uint32_t C_blocks = C_pad / 16u;   // c-blocks

        if (C == 1u) {
          // Direct 3x3 kernel for the special-case first conv (C=1): raw
          // weights re-laid out as [t][c][co] fp16 (co contiguous), bias
          // appended.
          const uint32_t wt_size = 9u * C * M + M;
          std::vector<uint16_t> d_buf(wt_size, 0);
          const uint16_t* src = reinterpret_cast<const uint16_t*>(wspec.data.data());
          for (uint32_t t = 0; t < 9u; ++t) {
            for (uint32_t c = 0; c < C; ++c) {
              for (uint32_t ko = 0; ko < M; ++ko) {
                d_buf[(t * C + c) * M + ko] =
                    src[static_cast<size_t>(ko) * C * 9u + c * 9u + t];
              }
            }
          }
          if (!op.conv_bias.empty() && weights_.count(op.conv_bias)) {
            const auto& bs = weights_[op.conv_bias];
            const uint16_t* bsrc = reinterpret_cast<const uint16_t*>(bs.data.data());
            for (uint32_t ko = 0; ko < M; ++ko) {
              MLFloat16 bh;
              std::memcpy(&bh, bsrc + ko, 2);
              MLFloat16 uh(bh.ToFloat());
              std::memcpy(d_buf.data() + 9u * C * M + ko, &uh, 2);
            }
          }
          void* wt_d = nullptr;
          {
            hipError_t e = hipMalloc(&wt_d, d_buf.size() * 2);
            if (e != hipSuccess) {
              return fail_build(std::string("hipMalloc of direct-conv weights failed: ") +
                                hipGetErrorString(e));
            }
            e = hipMemcpy(wt_d, d_buf.data(), d_buf.size() * 2, hipMemcpyHostToDevice);
            if (e != hipSuccess) {
              return fail_build(std::string("hipMemcpy of direct-conv weights failed: ") +
                                hipGetErrorString(e));
            }
          }
          int wt_idx = static_cast<int>(buffers_.size());
          buffers_.push_back(HipBuffer{wt_d, d_buf.size() * 2});

          int add_idx = wt_idx;  // dummy binding when unused
          if (op.do_add && !op.add_input.empty() && tensor_map_.count(op.add_input)) {
            add_idx = tensor_map_[op.add_input].buffer_index;
          }
          {
            WinogradPush p{};
            p.h = H;
            p.w = W;
            p.c = C;
            p.m = M;
            p.m_pad = 0;
            p.c_pad = 0;
            p.tiles_w = 0;
            p.do_silu = op.do_silu ? 1u : 0u;
            p.do_add = op.do_add ? 1u : 0u;
            dop.params.resize(sizeof(WinogradPush) / sizeof(uint32_t));
            std::memcpy(dop.params.data(), &p, sizeof(p));
          }
          dop.dispatch_x = DivCeil(W, 8u);
          dop.dispatch_y = H;
          dop.dispatch_z = 1;
          dop.input_buffers = {in_idx, wt_idx, add_idx};
          dop.output_buffer = tensor_map_[op.out].buffer_index;
          break;
        }
        // G (4x3)
        const float G[4][3] = {{1.0f, 0.0f, 0.0f},
                               {0.5f, 0.5f, 0.5f},
                               {0.5f, -0.5f, 0.5f},
                               {0.0f, 0.0f, 1.0f}};
        // U[kb][wp_i][cb][wp_local][ko_local][c_local]: each 16x16 (ko,c) block
        // contiguous for the WMMA A-operand, and the 4 wp_local matrices of a
        // (kb, wp_i) group are contiguous (1KB).
        std::vector<uint16_t> u_buf(kb_stride * 4u * C_blocks * 4u * 256u + M_pad, 0);
        const uint16_t* src = reinterpret_cast<const uint16_t*>(wspec.data.data());
        for (uint32_t ko = 0; ko < M; ++ko) {
          for (uint32_t c = 0; c < C; ++c) {
            float g[9];
            for (uint32_t t = 0; t < 9; ++t) {
              MLFloat16 h;
              std::memcpy(&h, src + (static_cast<size_t>(ko) * C + c) * 9 + t, 2);
              g[t] = h.ToFloat();
            }
            float Gg[4][3];
            for (uint32_t r = 0; r < 4; ++r) {
              for (uint32_t cc = 0; cc < 3; ++cc) {
                Gg[r][cc] = 0.0f;
                for (uint32_t k = 0; k < 3; ++k) {
                  Gg[r][cc] += G[r][k] * g[k * 3 + cc];
                }
              }
            }
            for (uint32_t r = 0; r < 4; ++r) {
              for (uint32_t cc = 0; cc < 4; ++cc) {
                float u = 0.0f;
                for (uint32_t k = 0; k < 3; ++k) {
                  u += Gg[r][k] * G[cc][k];  // G^T[k][cc] == G[cc][k]
                }
                uint32_t wp = r * 4 + cc;
                MLFloat16 uh(u);
                std::memcpy(u_buf.data() + (ko / 16u) * (4u * C_blocks * 4u * 256u) + (wp / 4u) * (C_blocks * 4u * 256u) +
                                (c / 16u) * 1024u + (wp % 4u) * 256u + (ko % 16u) * 16u + (c % 16u),
                            &uh, 2);
              }
            }
          }
        }
        if (!op.conv_bias.empty() && weights_.count(op.conv_bias)) {
          const auto& bs = weights_[op.conv_bias];
          const uint16_t* bsrc = reinterpret_cast<const uint16_t*>(bs.data.data());
          for (uint32_t ko = 0; ko < M; ++ko) {
            MLFloat16 bh;
            std::memcpy(&bh, bsrc + ko, 2);
            MLFloat16 uh(bh.ToFloat());
            std::memcpy(u_buf.data() + kb_stride * 4u * C_blocks * 4u * 256u + ko, &uh, 2);
          }
        }
        void* wt_u = nullptr;
        {
          hipError_t e = hipMalloc(&wt_u, u_buf.size() * 2);
          if (e != hipSuccess) {
            return fail_build(std::string("hipMalloc of winograd weights failed: ") +
                              hipGetErrorString(e));
          }
          e = hipMemcpy(wt_u, u_buf.data(), u_buf.size() * 2, hipMemcpyHostToDevice);
          if (e != hipSuccess) {
            return fail_build(std::string("hipMemcpy of winograd weights failed: ") +
                              hipGetErrorString(e));
          }
        }
        int wt_idx = static_cast<int>(buffers_.size());
        buffers_.push_back(HipBuffer{wt_u, u_buf.size() * 2});

        // Optional residual-add input (fused epilogue).
        int add_idx = wt_idx;  // dummy binding when unused
        if (op.do_add && !op.add_input.empty() && tensor_map_.count(op.add_input)) {
          add_idx = tensor_map_[op.add_input].buffer_index;
        }
        {
          WinogradPush p{};
          p.h = H;
          p.w = W;
          p.c = C;
          p.m = M;
          p.m_pad = M_pad;
          p.c_pad = C_pad;
          // Round the tile counts up (the kernel guards partial writes) so odd
          // and non-multiple-of-16 sizes are still fully scheduled.
          p.tiles_w = (W + 1u) / 2u;
          p.do_silu = op.do_silu ? 1u : 0u;
          p.do_add = op.do_add ? 1u : 0u;
          static const char* sk = std::getenv("HIP_SKIPPHASE");
          if (sk != nullptr && sk[0] == 'w') p.tiles_w = 0xDEAD0001u;  // skip writeback
          else if (sk != nullptr && sk[0] == 'v') p.tiles_w = 0xDEAD0002u;  // skip V (stub)
          dop.params.resize(sizeof(WinogradPush) / sizeof(uint32_t));
          std::memcpy(dop.params.data(), &p, sizeof(p));
        }
        {
          // Same workgroup geometry as the standalone engine: a workgroup
          // covers a 2-tile-row x 8-tile-column region for every k-group.
          const uint32_t tiles_w8 = (((W + 1u) / 2u) + 7u) / 8u;
          const uint32_t row_pairs = (((H + 1u) / 2u) + 1u) / 2u;
          const uint32_t kgroups = M_pad / 32u;
          dop.dispatch_x = tiles_w8 * row_pairs * ((kgroups + 1u) / 2u);
        }
        dop.dispatch_y = 1u;
        dop.dispatch_z = 1u;
        dop.input_buffers = {in_idx, wt_idx, add_idx};
        dop.output_buffer = tensor_map_[op.out].buffer_index;
        break;
      }
      case OpType::Sigmoid:
      case OpType::Clip: {
        int64_t numel = NumElements(in_shape);
        UnaryPush p{};
        p.numel = static_cast<uint32_t>(numel);
        p.op = op.type == OpType::Sigmoid ? 0u : 1u;
        p.min_val = op.clip_min;
        p.max_val = op.clip_max;
        dop.params.resize(4);
        std::memcpy(dop.params.data(), &p, sizeof(p));
        dop.dispatch_x = DivCeil(numel, kPointwiseLocal);
        dop.input_buffers = {tensor_map_[op.in0].buffer_index};
        dop.output_buffer = tensor_map_[op.out].buffer_index;
        break;
      }
      case OpType::Mul:
      case OpType::Add: {
        int64_t numel = NumElements(in_shape);
        BinaryPush p{};
        p.numel = static_cast<uint32_t>(numel);
        p.op = op.type == OpType::Mul ? 1u : 0u;
        dop.params.resize(2);
        std::memcpy(dop.params.data(), &p, sizeof(p));
        dop.dispatch_x = DivCeil(numel, kPointwiseLocal);
        dop.input_buffers = {tensor_map_[op.in0].buffer_index, tensor_map_[op.in1].buffer_index};
        dop.output_buffer = tensor_map_[op.out].buffer_index;
        break;
      }
      case OpType::DepthToSpace: {
        int64_t b = op.blocksize;
        int64_t numel = NumElements(tensor_map_[op.out].shape);
        DtsPush p{};
        p.c = static_cast<uint32_t>(in_shape[3] / (b * b));  // NHWC: C is innermost
        p.h = static_cast<uint32_t>(in_shape[1]);
        p.w = static_cast<uint32_t>(in_shape[2]);
        p.b = static_cast<uint32_t>(b);
        p.do_clip = op.do_clip ? 1u : 0u;
        p.min_val = op.clip_min;
        p.max_val = op.clip_max;
        dop.params.resize(sizeof(DtsPush) / sizeof(uint32_t));
        std::memcpy(dop.params.data(), &p, sizeof(p));
        dop.dispatch_x = DivCeil(numel, kPointwiseLocal);
        dop.input_buffers = {tensor_map_[op.in0].buffer_index};
        dop.output_buffer = tensor_map_[op.out].buffer_index;
        break;
      }
    }
    device_ops_.push_back(std::move(dop));
  }

  // --- Record tensor byte sizes ---
  input_bytes_ = static_cast<size_t>(NumElements(shapes[input_name_]) * 2);
  output_bytes_ = static_cast<size_t>(NumElements(shapes[output_name_]) * 2);
  // With a boundary Cast, the ORT-side IO is fp32 (4 bytes/elem) but the
  // staged/internal tensors stay fp16.
  input_ort_bytes_ = input_bytes_ * (input_is_fp32_ ? 2 : 1);
  output_ort_bytes_ = output_bytes_ * (output_is_fp32_ ? 2 : 1);

  // --- Persistent pinned staging buffers (allocated once, reused every run) ---
  {
    hipError_t e = hipHostMalloc(&input_staging_, input_bytes_, hipHostMallocDefault);
    if (e != hipSuccess) {
      return fail_build(std::string("hipHostMalloc of the input staging buffer failed: ") +
                        hipGetErrorString(e));
    }
    e = hipHostMalloc(&output_staging_, output_bytes_, hipHostMallocDefault);
    if (e != hipSuccess) {
      return fail_build(std::string("hipHostMalloc of the output staging buffer failed: ") +
                        hipGetErrorString(e));
    }
    e = hipDeviceSynchronize();
    if (e != hipSuccess) {
      return fail_build(std::string("hipDeviceSynchronize after setup failed: ") +
                        hipGetErrorString(e));
    }
  }
  built_ = true;
  return Status::OK();
}

// Bit-level view of an _Float16 value; lets the IO conversion loops
// vectorize to vcvtps2ph/vcvtph2ps (build with -mf16c -mavx2).
union H16 {
  uint16_t u;
  _Float16 h;
  H16() = default;
  explicit H16(_Float16 v) : h(v) {}
  explicit H16(uint16_t v) : u(v) {}
};

static inline uint32_t F32Bits(float f) {
  uint32_t u;
  std::memcpy(&u, &f, sizeof(u));
  return u;
}

static inline float BitsF32(uint32_t u) {
  float f;
  std::memcpy(&f, &u, sizeof(f));
  return f;
}

static inline uint16_t FloatToHalfBitsRNE(float f) {
  const uint32_t b = F32Bits(f);
  const uint32_t sign = (b >> 16u) & 0x8000u;
  const uint32_t e = (b >> 23u) & 0xffu;
  const uint32_t m = b & 0x7fffffu;
  if (e == 0xffu) {
    return static_cast<uint16_t>(sign | 0x7c00u | (m ? 0x200u : 0u));
  }
  int e16 = static_cast<int>(e) - 127 + 15;
  if (e16 >= 31) {
    return static_cast<uint16_t>(sign | 0x7c00u);  // overflow -> inf
  }
  if (e16 <= 0) {
    if (e16 < -10) {
      return static_cast<uint16_t>(sign);  // flush to zero
    }
    uint32_t hm = m | 0x800000u;
    const uint32_t shift = static_cast<uint32_t>(14 - e16);
    uint32_t half = hm >> shift;
    const uint32_t rem = hm & ((1u << shift) - 1u);
    const uint32_t halfway = 1u << (shift - 1u);
    if (rem > halfway || (rem == halfway && (half & 1u))) {
      ++half;
    }
    return static_cast<uint16_t>(sign | half);
  }
  uint32_t m16 = m >> 13u;
  const uint32_t rem = m & 0x1fffu;
  if (rem > 0x1000u || (rem == 0x1000u && (m16 & 1u))) {
    ++m16;
  }
  if (m16 == 0x400u) {
    m16 = 0u;
    ++e16;
  }
  if (e16 >= 31) {
    return static_cast<uint16_t>(sign | 0x7c00u);
  }
  return static_cast<uint16_t>(sign | (static_cast<uint32_t>(e16) << 10u) | m16);
}

static inline float HalfBitsToFloat(uint16_t h) {
  const uint32_t sign = (static_cast<uint32_t>(h) & 0x8000u) << 16u;
  const uint32_t e = (h >> 10u) & 0x1fu;
  const uint32_t m = h & 0x3ffu;
  if (e == 0u) {
    if (m == 0u) {
      return BitsF32(sign);
    }
    uint32_t e32 = 127u - 15u + 1u;
    uint32_t hm = m;
    while ((hm & 0x400u) == 0u) {
      hm <<= 1u;
      --e32;
    }
    return BitsF32(sign | (e32 << 23u) | ((hm & 0x3ffu) << 13u));
  }
  if (e == 0x1fu) {
    return BitsF32(sign | 0x7f800000u | (m ? 0x400000u : 0u));
  }
  return BitsF32(sign | ((e + (127u - 15u)) << 23u) | (m << 13u));
}

common::Status HipGraph::Run(const void* input_data, size_t input_bytes,
                             const std::vector<int64_t>& input_shape,
                             void* output_data, size_t output_bytes) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto prof_t0 = std::chrono::high_resolution_clock::now();
  // Consume any error left in the thread-local last-error slot by an earlier
  // failed call on this thread so the launch check cannot misattribute it.
  (void)hipGetLastError();
  ORT_RETURN_IF_ERROR(EnsureBuilt(input_shape));
  auto prof_t1 = std::chrono::high_resolution_clock::now();

  if (input_bytes != input_ort_bytes_ || output_bytes != output_ort_bytes_) {
    return Status(common::ONNXRUNTIME, common::FAIL, "HIP EP: input/output size mismatch.");
  }

  hipStream_t stream = static_cast<hipStream_t>(ctx_->Stream());

  auto run_fail = [](const char* what, hipError_t e) -> common::Status {
    return Status(common::ONNXRUNTIME, common::FAIL,
                  std::string("HIP EP: ") + what + " failed: " + hipGetErrorString(e));
  };

  // Upload the input into the persistent pinned staging buffer, converting
  // fp32 -> fp16 on the host when the graph input is behind a Cast.
  if (input_is_fp32_) {
    const float* src = static_cast<const float*>(input_data);
    uint16_t* dst = static_cast<uint16_t*>(input_staging_);
    const size_t n = input_bytes_ / 2;
    for (size_t i = 0; i < n; ++i) {
      dst[i] = FloatToHalfBitsRNE(src[i]);
    }
  } else {
    std::memcpy(input_staging_, input_data, input_bytes);
  }

  auto prof_t2 = std::chrono::high_resolution_clock::now();

  {
    hipError_t e = hipMemcpyAsync(buffers_[input_buffer_].ptr, input_staging_, input_bytes_,
                                  hipMemcpyHostToDevice, stream);
    if (e != hipSuccess) return run_fail("input upload", e);
  }

  for (size_t dop_i = 0; dop_i < device_ops_.size(); ++dop_i) {
    const auto& dop = device_ops_[dop_i];
    void* in_bufs[3] = {nullptr, nullptr, nullptr};
    void* out_buf = buffers_[dop.output_buffer].ptr;
    for (size_t i = 0; i < dop.input_buffers.size() && i < 3; ++i) {
      if (dop.input_buffers[i] >= 0) {
        in_bufs[i] = buffers_[dop.input_buffers[i]].ptr;
      }
    }
    switch (dop.type) {
      case OpType::Conv: {
        ConvParams p{};
        std::memcpy(&p, dop.params.data(), sizeof(p));
        if (p.C == 1u) {
          hipLaunchKernelGGL(direct_conv, dim3(dop.dispatch_x, dop.dispatch_y), dim3(64), 0, stream,
                             static_cast<const _Float16*>(in_bufs[0]),
                             static_cast<const _Float16*>(in_bufs[1]),
                             static_cast<_Float16*>(out_buf),
                             static_cast<const _Float16*>(in_bufs[2]), p);
        } else {
          hipLaunchKernelGGL(winograd_conv, dim3(dop.dispatch_x), dim3(128), 0, stream,
                             static_cast<const _Float16*>(in_bufs[0]),
                             static_cast<const _Float16*>(in_bufs[1]),
                             static_cast<_Float16*>(out_buf),
                             static_cast<const _Float16*>(in_bufs[2]), p);
        }
        break;
      }
      case OpType::Sigmoid:
      case OpType::Clip: {
        UnaryParams p{};
        std::memcpy(&p, dop.params.data(), sizeof(p));
        hipLaunchKernelGGL(unary_kernel, dim3(dop.dispatch_x), dim3(kPointwiseLocal), 0, stream,
                           static_cast<const _Float16*>(in_bufs[0]),
                           static_cast<_Float16*>(out_buf), p);
        break;
      }
      case OpType::Mul:
      case OpType::Add: {
        BinaryParams p{};
        std::memcpy(&p, dop.params.data(), sizeof(p));
        hipLaunchKernelGGL(binary_kernel, dim3(dop.dispatch_x), dim3(kPointwiseLocal), 0, stream,
                           static_cast<const _Float16*>(in_bufs[0]),
                           static_cast<const _Float16*>(in_bufs[1]),
                           static_cast<_Float16*>(out_buf), p);
        break;
      }
      case OpType::DepthToSpace: {
        DtsParams p{};
        std::memcpy(&p, dop.params.data(), sizeof(p));
        hipLaunchKernelGGL(dts_kernel, dim3(dop.dispatch_x), dim3(kPointwiseLocal), 0, stream,
                           static_cast<const _Float16*>(in_bufs[0]),
                           static_cast<_Float16*>(out_buf), p);
        break;
      }
    }
    // hipLaunchKernelGGL discards its status; read it before the next HIP call.
    {
      hipError_t le = hipGetLastError();
      if (le != hipSuccess) {
        return Status(common::ONNXRUNTIME, common::FAIL,
                      "HIP EP: kernel launch for op " + std::to_string(dop_i) + " failed: " +
                          hipGetErrorString(le));
      }
    }
    if (std::getenv("HIP_DUMPOPS")) {
      fprintf(stderr, "[dump] op %zu type=%d dispatch=(%u,%u,%u) in=", dop_i,
              static_cast<int>(dop.type), dop.dispatch_x, dop.dispatch_y, dop.dispatch_z);
      for (int bi : dop.input_buffers) fprintf(stderr, "%d(%p,%.2fMB) ", bi,
              buffers_[bi].ptr, buffers_[bi].size / 1048576.0);
      fprintf(stderr, "out=%d(%p,%.2fMB)\n", dop.output_buffer,
              buffers_[dop.output_buffer].ptr, buffers_[dop.output_buffer].size / 1048576.0);
    }
  }

  auto prof_t2b = std::chrono::high_resolution_clock::now();

  {
    hipError_t e = hipMemcpyAsync(output_data, buffers_[output_buffer_].ptr, output_bytes_,
                                  hipMemcpyDeviceToHost, stream);
    if (e != hipSuccess) return run_fail("output download", e);
    e = hipStreamSynchronize(stream);
    if (e != hipSuccess) return run_fail("stream synchronize", e);
  }
  auto prof_tafter_submit = std::chrono::high_resolution_clock::now();

  if (profiling_) {
    fprintf(stderr, "[hipprof] build=%.3fms memin=%.3fms launch=%.3fms readback+wait=%.3fms\n",
            std::chrono::duration<double, std::milli>(prof_t1 - prof_t0).count(),
            std::chrono::duration<double, std::milli>(prof_t2 - prof_t1).count(),
            std::chrono::duration<double, std::milli>(prof_t2b - prof_t2).count(),
            std::chrono::duration<double, std::milli>(prof_tafter_submit - prof_t2b).count());
  }

  // The output is read straight into the ORT tensor; fp16->fp32 conversion is
  // only needed when the graph output is behind a Cast (fp32 ORT tensor).
  if (output_is_fp32_) {
    // Keep the half data in the pinned staging copy and read from THERE: the
    // half source and the float destination share output_data, so reading the
    // source in place would consume bytes already overwritten by earlier
    // (wider) float writes.
    std::memcpy(output_staging_, output_data, output_bytes_);
    const uint16_t* src = static_cast<const uint16_t*>(output_staging_);
    float* dst = static_cast<float*>(output_data);
    const size_t count = output_bytes_ / 2;
    for (size_t i = 0; i < count; ++i) {
      dst[i] = HalfBitsToFloat(src[i]);
    }
  }
  auto prof_t3 = std::chrono::high_resolution_clock::now();

  if (profiling_) {
    fprintf(stderr, "[hipprof] memout=%.3fms total=%.3fms\n",
            std::chrono::duration<double, std::milli>(prof_t3 - prof_tafter_submit).count(),
            std::chrono::duration<double, std::milli>(prof_t3 - prof_t0).count());
  }
  return Status::OK();
}

}  // namespace hip
}  // namespace onnxruntime
