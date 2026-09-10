// Standalone HIP inference engine implementation.
#include "hip_engine.h"

#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <unordered_set>

#include "hip_kernels.h"

namespace vship {

namespace {

int64_t NumElements(const std::vector<int64_t>& shape) {
  return std::accumulate(shape.begin(), shape.end(), int64_t{1}, std::multiplies<int64_t>());
}

uint32_t DivCeil(uint64_t a, uint32_t b) {
  return static_cast<uint32_t>((a + b - 1) / b);
}

// ONNX element size for the dtypes the engine can consume; 0 for unknown.
size_t OnnxDtypeSize(int dtype) {
  switch (dtype) {
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT: return 4;
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16: return 2;
    case ONNX_NAMESPACE::TensorProto_DataType_DOUBLE: return 8;
    case ONNX_NAMESPACE::TensorProto_DataType_INT64: return 8;
    case ONNX_NAMESPACE::TensorProto_DataType_UINT32:
    case ONNX_NAMESPACE::TensorProto_DataType_INT32: return 4;
    case ONNX_NAMESPACE::TensorProto_DataType_UINT16:
    case ONNX_NAMESPACE::TensorProto_DataType_INT16:
    case ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16: return 2;
    case ONNX_NAMESPACE::TensorProto_DataType_UINT8:
    case ONNX_NAMESPACE::TensorProto_DataType_INT8:
    case ONNX_NAMESPACE::TensorProto_DataType_BOOL: return 1;
    default: return 0;
  }
}

// Materialize one initializer's storage with its declared dtype preserved.
// FLOAT16 values are stored in int32_data() as 16-bit bit patterns (that is
// what the ONNX schema and ArtHIP's own converter do), so they must NOT be
// interpreted as numeric integers. External data cannot be materialized by
// this parser and is rejected here.
bool MaterializeInitializer(const ONNX_NAMESPACE::TensorProto& init,
                            WeightSpec& ws, std::string& error) {
  const std::string& name = init.name();
  ws.dtype = init.data_type();
  ws.data.clear();
  ws.shape.clear();
  for (int d = 0; d < init.dims_size(); ++d) ws.shape.push_back(init.dims(d));

  if (init.data_location() == ONNX_NAMESPACE::TensorProto_DataLocation_EXTERNAL) {
    error = "initializer '" + name + "' uses external data, which is not supported";
    return false;
  }

  const std::string& raw = init.raw_data();
  if (!raw.empty()) {
    if (OnnxDtypeSize(ws.dtype) == 0) {
      error = "initializer '" + name + "' has an unsupported dtype";
      return false;
    }
    ws.data.assign(raw.begin(), raw.end());
    return true;
  }

  if (init.float_data_size() > 0) {
    if (ws.dtype != ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
      error = "initializer '" + name + "' has float_data with a non-FLOAT dtype";
      return false;
    }
    ws.data.resize(static_cast<size_t>(init.float_data_size()) * 4);
    std::memcpy(ws.data.data(), init.float_data().data(), ws.data.size());
  } else if (init.double_data_size() > 0) {
    if (ws.dtype != ONNX_NAMESPACE::TensorProto_DataType_DOUBLE) {
      error = "initializer '" + name + "' has double_data with a non-DOUBLE dtype";
      return false;
    }
    ws.data.resize(static_cast<size_t>(init.double_data_size()) * 8);
    std::memcpy(ws.data.data(), init.double_data().data(), ws.data.size());
  } else if (init.int32_data_size() > 0) {
    // int8/int16/bool/int32/uint* and FLOAT16/BFLOAT16 all share int32_data as
    // one slot per element. FLOAT16 slots hold 16-bit BIT PATTERNS, not numeric
    // integers, so the declared element width governs: keep the low
    // size-of-dtype bytes of each slot verbatim (little-endian host).
    const int n = init.int32_data_size();
    const size_t elem = OnnxDtypeSize(ws.dtype);
    if (elem == 0) {
      error = "initializer '" + name + "' has an unsupported dtype";
      return false;
    }
    ws.data.resize(static_cast<size_t>(n) * elem);
    for (int k = 0; k < n; ++k) {
      const uint32_t v = static_cast<uint32_t>(init.int32_data(k));
      std::memcpy(ws.data.data() + static_cast<size_t>(k) * elem, &v, elem);
    }
  } else if (init.int64_data_size() > 0) {
    const int n = init.int64_data_size();
    ws.data.resize(static_cast<size_t>(n) * 8);
    std::memcpy(ws.data.data(), init.int64_data().data(), ws.data.size());
  }
  // An empty typed field is legal for a zero-element tensor; every consumer
  // validates that the bytes it reads are present.
  return true;
}

struct WinogradPush {
  uint32_t h, w, c, m, m_pad, c_pad, tiles_w, do_silu, do_add;
};
struct UnaryPush {
  uint32_t numel, op;
  float min_val, max_val;
};
struct BinaryPush {
  uint32_t numel, op;
};
struct DtsPush {
  uint32_t c, h, w, b;
  uint32_t do_clip;
  float min_val, max_val;
};

bool IsSupportedConv(const OpSpec& op) {
  return op.stride_h == 1 && op.stride_w == 1 && op.dil_h == 1 && op.dil_w == 1 &&
         op.pad_h == 1 && op.pad_w == 1 && op.group == 1;
}

// fp32 -> fp16 conversion for fp32-input models (raw host upload, on-device
// cast; replaces the old per-pixel CPU conversion loop).
__global__ void cast_f32_to_f16(const float* __restrict__ in,
                                _Float16* __restrict__ out, uint32_t n) {
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = (_Float16)in[i];
}

// fp16 -> fp32 conversion for fp32-output models (compute stays fp16; the
// float result matches what the MIGX plugin downloads).
__global__ void cast_f16_to_f32(const _Float16* __restrict__ in,
                                float* __restrict__ out, uint32_t n) {
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = (float)in[i];
}

// Fused NCHW->NHWC transpose + fp32->fp16 cast for multi-channel fp32 inputs:
// the ORT-layout upload is NCHW, the compute buffer is NHWC. One thread per
// (n, h, w) pixel, loops over C (C<=3 for our models; C=1 is a copy).
__global__ void cast_f32_to_f16_transpose(const float* __restrict__ in,
                                          _Float16* __restrict__ out,
                                          uint32_t N, uint32_t C,
                                          uint32_t H, uint32_t W) {
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N * H * W) return;
  const uint32_t n = i / (H * W);
  const uint32_t hw = i - n * H * W;
  for (uint32_t c = 0; c < C; ++c) {
    out[(n * H * W + hw) * C + c] = (_Float16)in[(n * C + c) * H * W + hw];
  }
}

// Fused NHWC->NCHW transpose + fp16->fp32 cast for multi-channel fp32
// outputs: the compute buffer is NHWC, the download is ORT NCHW.
__global__ void cast_f16_to_f32_transpose(const _Float16* __restrict__ in,
                                          float* __restrict__ out,
                                          uint32_t N, uint32_t C,
                                          uint32_t H, uint32_t W) {
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N * H * W) return;
  const uint32_t n = i / (H * W);
  const uint32_t hw = i - n * H * W;
  for (uint32_t c = 0; c < C; ++c) {
    out[(n * C + c) * H * W + hw] = (float)in[(n * H * W + hw) * C + c];
  }
}

}  // namespace

HipEngine::HipEngine(int device_id) : device_id_(device_id) {}

HipEngine::~HipEngine() {
  DestroyDeviceState();
  if (stream_) {
    hipStreamDestroy(static_cast<hipStream_t>(stream_));
  }
}

void HipEngine::DestroyDeviceState() {
  // Freeing device memory requires the owning device to be current on THIS
  // thread; the destructor and rebuild paths can run on any thread.
  hipSetDevice(device_id_);
  for (auto& buf : buffers_) {
    if (buf.ptr) hipFree(buf.ptr);
  }
  buffers_.clear();
  tensor_map_.clear();
  device_ops_.clear();
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
  if (input_staging_f32_) {
    hipFreeHost(input_staging_f32_);
    input_staging_f32_ = nullptr;
  }
  if (input_f32_dev_) {
    hipFree(input_f32_dev_);
    input_f32_dev_ = nullptr;
  }
  if (output_staging_f32_) {
    hipFreeHost(output_staging_f32_);
    output_staging_f32_ = nullptr;
  }
  if (output_f32_dev_) {
    hipFree(output_f32_dev_);
    output_f32_dev_ = nullptr;
  }
  built_ = false;
}

bool HipEngine::Build(const ONNX_NAMESPACE::ModelProto& model, std::string& error) {
  const auto& graph = model.graph();

  // Basic IO setup.
  if (graph.input_size() < 1 || graph.output_size() < 1) {
    error = "model must have at least one input and output";
    return false;
  }
  input_name_ = graph.input(0).name();
  output_name_ = graph.output(0).name();

  // Read all initializers into raw bytes with their declared dtype preserved
  // (float, half, ...). NOTE: small scalars (Clip bounds) are often stored in
  // float_data(), not raw_data() — materialize those too. FLOAT16 tensors use
  // int32_data() for raw 16-bit patterns, not numeric integers.
  for (int ii = 0; ii < graph.initializer_size(); ++ii) {
    const auto& init = graph.initializer(ii);
    if (init.name().empty()) {
      error = "initializer #" + std::to_string(ii) + " has no name";
      return false;
    }
    if (!MaterializeInitializer(init, weights_[init.name()], error)) return false;
  }

  // Attribute lookup helper.
  auto attr = [](const ONNX_NAMESPACE::NodeProto& n, const std::string& name) {
    for (int j = 0; j < n.attribute_size(); ++j) {
      if (n.attribute(j).name() == name) return &n.attribute(j);
    }
    return static_cast<const ONNX_NAMESPACE::AttributeProto*>(nullptr);
  };

  // Consumers map (output tensor name -> node indices) for the fusions.
  std::unordered_map<std::string, std::vector<int>> consumers;
  for (int ni = 0; ni < graph.node_size(); ++ni) {
    const auto& n = graph.node(ni);
    for (int j = 0; j < n.input_size(); ++j) {
      if (!n.input(j).empty()) consumers[n.input(j)].push_back(ni);
    }
  }
  auto out_nodes = [&](const std::string& t) -> const std::vector<int>& {
    static const std::vector<int> empty;
    auto it = consumers.find(t);
    return it == consumers.end() ? empty : it->second;
  };
  auto node_of = [&](int idx) -> const ONNX_NAMESPACE::NodeProto& {
    return graph.node(idx);
  };

  // fp32 model I/O detection (with converter-inserted boundary casts) happens
  // further down, in the same pass that marks the Cast nodes consumed.

  // Clip-bound scalar: fp32 (float), fp16 (half) or double initializer -> float.
  // (The fp16 luma model stores half bounds; the fp32 chroma model stores
  // float bounds. Reading float bits as half gave min=-0.0/max=1.9e-3,
  // clamping all chroma output to ~0.)
  auto clip_scalar = [&](const std::string& name, float& dst) -> bool {
    const auto w = weights_.find(name);
    if (w == weights_.end() || w->second.data.empty()) return true;  // omit -> keep default
    const auto& ws = w->second;
    switch (ws.dtype) {
      case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
        if (ws.data.size() >= 4) {
          float v;
          std::memcpy(&v, ws.data.data(), 4);
          dst = v;
          return true;
        }
        break;
      case ONNX_NAMESPACE::TensorProto_DataType_DOUBLE:
        if (ws.data.size() >= 8) {
          double v;
          std::memcpy(&v, ws.data.data(), 8);
          dst = static_cast<float>(v);
          return true;
        }
        break;
      case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16:
        if (ws.data.size() >= 2) {
          uint16_t bits;
          std::memcpy(&bits, ws.data.data(), 2);
          dst = HipEngine::HalfBitsToFloat(bits);
          return true;
        }
        break;
      default:
        break;
    }
    error = "Clip bound initializer '" + name + "' has an unsupported dtype or truncated storage";
    return false;
  };

  auto read_initializer = [&](const std::string& name, std::vector<uint8_t>& data,
                              std::vector<int64_t>& dims) -> bool {
    auto it = weights_.find(name);
    if (it == weights_.end()) {
      error = "expected initializer '" + name + "' was not found";
      return false;
    }
    data = it->second.data;
    dims = it->second.shape;
    return true;
  };

  // Skip the boundary Cast nodes. Detection and consumption must happen in
  // ONE pass: input_name_/output_name_ are rewritten during detection, so a
  // later name-based pass would miss the converter-inserted casts.
  // Models whose IO is natively fp32 (no Cast nodes) still need the fp32
  // bridge paths — detect bare fp32 IO directly.
  if (graph.input(0).type().tensor_type().elem_type() == ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
    input_is_fp32_ = true;
  }
  if (graph.output(0).type().tensor_type().elem_type() == ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
    output_is_fp32_ = true;
  }
  std::vector<bool> consumed(graph.node_size(), false);
  if (graph.input(0).type().tensor_type().elem_type() == ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
    for (int ni = 0; ni < graph.node_size(); ++ni) {
      const auto& n = graph.node(ni);
      if (n.op_type() != "Cast") continue;
      if (n.input_size() >= 1 && n.input(0) == input_name_ && n.output_size() >= 1) {
        input_name_ = n.output(0);
        input_is_fp32_ = true;
        consumed[ni] = true;
        break;
      }
    }
  }
  if (graph.output(0).type().tensor_type().elem_type() == ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
    for (int ni = 0; ni < graph.node_size(); ++ni) {
      const auto& n = graph.node(ni);
      if (n.op_type() != "Cast") continue;
      if (n.output_size() >= 1 && n.output(0) == output_name_ && n.input_size() >= 1) {
        output_name_ = n.input(0);
        output_is_fp32_ = true;
        consumed[ni] = true;
        break;
      }
    }
  }

  for (int ni = 0; ni < graph.node_size(); ++ni) {
    if (consumed[ni]) continue;
    const auto& n = graph.node(ni);
    const std::string& optype = n.op_type();
    if (n.output_size() < 1 || n.output(0).empty()) continue;

    auto in_name = [&](int i) -> std::string {
      return (i < n.input_size()) ? n.input(i) : "";
    };

    OpSpec op;
    op.out = n.output(0);

    if (optype == "Conv") {
      op.type = OpType::Conv;
      op.in0 = in_name(0);
      op.in1 = in_name(1);
      op.conv_bias = in_name(2);
      if (op.in0.empty() || op.in1.empty()) {
        error = "Conv missing inputs";
        return false;
      }
      if (auto a = attr(n, "strides"); a && a->ints_size() >= 2) {
        op.stride_h = static_cast<int>(a->ints(0));
        op.stride_w = static_cast<int>(a->ints(1));
      }
      if (auto a = attr(n, "dilations"); a && a->ints_size() >= 2) {
        op.dil_h = static_cast<int>(a->ints(0));
        op.dil_w = static_cast<int>(a->ints(1));
      }
      if (auto a = attr(n, "group")) op.group = static_cast<int>(a->i());
      if (auto a = attr(n, "auto_pad"); a && (a->s() == "SAME_UPPER" || a->s() == "SAME_LOWER")) {
        op.pad_h = 1;
        op.pad_w = 1;
      }
      if (auto a = attr(n, "pads"); a && a->ints_size() >= 4) {
        op.pad_h = static_cast<int>(a->ints(0));
        op.pad_w = static_cast<int>(a->ints(2));
      }
      if (!IsSupportedConv(op)) {
        error = "unsupported Conv configuration (only 3x3 stride 1, pad 1 supported)";
        return false;
      }
      std::vector<uint8_t> wdata;
      std::vector<int64_t> wdims;
      if (!read_initializer(op.in1, wdata, wdims)) return false;

      // Validate weight geometry / dtype / backing bytes BEFORE any shape
      // probing or the fixed 3x3 packing loops, which read nine coefficients
      // unconditionally.
      if (wdims.size() != 4) {
        error = "Conv weight '" + op.in1 + "' must be a rank-4 tensor";
        return false;
      }
      if (wdims[2] != 3 || wdims[3] != 3) {
        error = "Conv weight '" + op.in1 + "' must be 3x3 (got " + std::to_string(wdims[2]) +
                "x" + std::to_string(wdims[3]) + ")";
        return false;
      }
      if (wdims[0] <= 0 || wdims[1] <= 0) {
        error = "Conv weight '" + op.in1 + "' has a non-positive channel count";
        return false;
      }
      const int wdtype = weights_[op.in1].dtype;
      if (wdtype != ONNX_NAMESPACE::TensorProto_DataType_FLOAT &&
          wdtype != ONNX_NAMESPACE::TensorProto_DataType_FLOAT16) {
        error = "Conv weight '" + op.in1 + "' has unsupported dtype " + std::to_string(wdtype);
        return false;
      }
      const size_t wbytes = static_cast<size_t>(NumElements(wdims)) * OnnxDtypeSize(wdtype);
      if (wdata.size() < wbytes) {
        error = "Conv weight '" + op.in1 + "' storage is truncated (" + std::to_string(wdata.size()) +
                " bytes, need " + std::to_string(wbytes) + ")";
        return false;
      }
      if (!op.conv_bias.empty()) {
        std::vector<uint8_t> bdata;
        std::vector<int64_t> bdims;
        if (!read_initializer(op.conv_bias, bdata, bdims)) return false;
        if (bdims.size() != 1 || bdims[0] != wdims[0]) {
          error = "Conv bias '" + op.conv_bias + "' must be a rank-1 tensor with the weight's channel count";
          return false;
        }
        const int bdtype = weights_[op.conv_bias].dtype;
        if (bdtype != ONNX_NAMESPACE::TensorProto_DataType_FLOAT &&
            bdtype != ONNX_NAMESPACE::TensorProto_DataType_FLOAT16) {
          error = "Conv bias '" + op.conv_bias + "' has unsupported dtype " + std::to_string(bdtype);
          return false;
        }
        if (bdata.size() < static_cast<size_t>(bdims[0]) * OnnxDtypeSize(bdtype)) {
          error = "Conv bias '" + op.conv_bias + "' storage is truncated";
          return false;
        }
      }

      // Fuse the SiLU (Sigmoid + Mul) epilogue into the conv.
      const std::string conv_out = op.out;
      for (int ci : out_nodes(conv_out)) {
        const auto& consumer = node_of(ci);
        if (consumer.op_type() == "Sigmoid") {
          const std::string& s_out = consumer.output(0);
          for (int ci2 : out_nodes(s_out)) {
            const auto& mul_node = node_of(ci2);
            if (mul_node.op_type() == "Mul" && mul_node.input_size() == 2) {
              const std::string& n0 = mul_node.input(0);
              const std::string& n1 = mul_node.input(1);
              if ((n0 == conv_out && n1 == s_out) || (n0 == s_out && n1 == conv_out)) {
                op.do_silu = true;
                op.out = mul_node.output(0);
                consumed[ci] = true;   // Sigmoid
                consumed[ci2] = true;  // Mul
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
      if (auto a = attr(n, "min")) op.clip_min = a->f();
      if (auto a = attr(n, "max")) op.clip_max = a->f();
      if (n.input_size() > 1 && !n.input(1).empty() && !clip_scalar(n.input(1), op.clip_min)) return false;
      if (n.input_size() > 2 && !n.input(2).empty() && !clip_scalar(n.input(2), op.clip_max)) return false;
    } else if (optype == "DepthToSpace") {
      op.type = OpType::DepthToSpace;
      op.in0 = in_name(0);
      if (auto a = attr(n, "blocksize")) op.blocksize = static_cast<int>(a->i());
      if (op.blocksize < 1) {
        error = "DepthToSpace blocksize must be positive";
        return false;
      }
      if (auto a = attr(n, "mode"); a && a->s() != "DCR") {
        error = "only DepthToSpace mode DCR is supported";
        return false;
      }
      // Fuse an immediately-following Clip into this op (elementwise).
      for (int ci : out_nodes(op.out)) {
        const auto& cn = node_of(ci);
        if (cn.op_type() != "Clip" || consumed[ci]) continue;
        if (auto a = attr(cn, "min")) op.clip_min = a->f();
        if (auto a = attr(cn, "max")) op.clip_max = a->f();
        if (cn.input_size() > 1 && !cn.input(1).empty() && !clip_scalar(cn.input(1), op.clip_min)) return false;
        if (cn.input_size() > 2 && !cn.input(2).empty() && !clip_scalar(cn.input(2), op.clip_max)) return false;
        op.fuse_clip = true;
        consumed[ci] = true;
        op.out = cn.output(0);  // produce the Clip's output tensor directly
        break;
      }
    } else {
      error = "unsupported op '" + optype + "' in fused graph";
      return false;
    }
    ops_.push_back(std::move(op));
  }

  // Pass 2: fuse residual Add nodes into the conv that produces their main input.
  {
    std::unordered_map<std::string, size_t> producer;
    for (size_t i = 0; i < ops_.size(); ++i) producer[ops_[i].out] = i;
    std::vector<size_t> remove_adds;
    for (size_t i = 0; i < ops_.size(); ++i) {
      if (ops_[i].type != OpType::Add) continue;
      const auto it_a = producer.find(ops_[i].in0);
      const auto it_b = producer.find(ops_[i].in1);
      if (it_a == producer.end() || it_b == producer.end()) continue;
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

  // Validate binary-op operands: the engine implements same-shape elementwise
  // Mul/Add between two engine-produced activations only. Initializer-backed
  // constants (including the ubiquitous `x * 0.25`) and ONNX broadcasting are
  // not implemented, so reject them here instead of launching with a null
  // activation pointer (which dereferences null in the kernel). Runs after the
  // Add fusion so the fused residual input is validated too.
  {
    std::unordered_set<std::string> produced;
    for (const auto& op : ops_) produced.insert(op.out);
    auto require_activation = [&](const std::string& what, const std::string& name) -> bool {
      if (produced.count(name) == 0) {
        error = what + " operand '" + name +
                "' is not an activation produced in this graph; initializer constants, graph "
                "inputs and broadcasting are not supported";
        return false;
      }
      return true;
    };
    for (const OpSpec& op : ops_) {
      if (op.type == OpType::Sigmoid || op.type == OpType::Clip) {
        if (!require_activation("unary op", op.in0)) return false;
      } else if (op.type == OpType::Mul) {
        if (!require_activation("Mul", op.in0) || !require_activation("Mul", op.in1)) return false;
      } else if (op.type == OpType::Add) {
        if (!require_activation("Add", op.in0) || !require_activation("Add", op.in1)) return false;
      } else if (op.type == OpType::Conv && op.do_add) {
        if (!require_activation("fused residual Add", op.add_input)) return false;
      }
    }
  }

  // The engine uses the same GPU as the EP: init once here.
  if (hipInit(0) != hipSuccess) {
    error = "hipInit failed";
    return false;
  }
  if (!SetDevice(error)) return false;
  if (hipStreamCreateWithFlags(reinterpret_cast<hipStream_t*>(&stream_), hipStreamNonBlocking) != hipSuccess) {
    error = "hipStreamCreate failed";
    return false;
  }

  return true;
}

bool HipEngine::SetDevice(std::string& error) const {
  hipError_t e = hipSetDevice(device_id_);
  if (e != hipSuccess) {
    error = "hipSetDevice(" + std::to_string(device_id_) + ") failed: " + hipGetErrorString(e);
    return false;
  }
  return true;
}

bool HipEngine::PropagateShapes(const std::vector<int64_t>& input_shape,
                                std::unordered_map<std::string, std::vector<int64_t>>& shapes) const {
  if (input_shape.size() != 4) return false;
  if (input_shape[0] != 1) return false;  // kernels have no batch dimension
  shapes.clear();
  // Internal tensors use the NHWC [N,H,W,C] layout.
  shapes[input_name_] = {input_shape[0], input_shape[2], input_shape[3], input_shape[1]};
  for (const auto& op : ops_) {
    const auto& in = shapes.find(op.in0);
    if (in == shapes.end()) return false;
    const auto& s = in->second;
    if (s.size() != 4) return false;
    std::vector<int64_t> out_shape;
    switch (op.type) {
      case OpType::Conv: {
        const auto& w = weights_.find(op.in1);
        if (w == weights_.end()) return false;
        if (w->second.shape.size() != 4) return false;
        // Channel agreement: the weight's input channel count must match the
        // activation's channel count, or the kernels read the wrong stride.
        if (w->second.shape[1] != s[3]) return false;
        int64_t m = w->second.shape[0];
        int64_t oh = (s[1] + 2 * op.pad_h - op.dil_h * 2 - 1) / op.stride_h + 1;
        int64_t ow = (s[2] + 2 * op.pad_w - op.dil_w * 2 - 1) / op.stride_w + 1;
        if (oh <= 0 || ow <= 0) return false;
        out_shape = {s[0], oh, ow, m};
        // A fused residual input must match the conv output exactly: the
        // kernel indexes it with the conv output's geometry.
        if (op.do_add) {
          const auto& add_shape = shapes.find(op.add_input);
          if (add_shape == shapes.end() || add_shape->second != out_shape) return false;
        }
        break;
      }
      case OpType::Sigmoid:
      case OpType::Clip:
        out_shape = s;
        break;
      case OpType::Mul:
      case OpType::Add: {
        // The kernels implement same-shape elementwise ops only; ONNX
        // broadcasting would read the second operand out of bounds.
        const auto& rhs = shapes.find(op.in1);
        if (rhs == shapes.end()) return false;
        if (rhs->second != s) return false;
        out_shape = s;
        break;
      }
      case OpType::DepthToSpace: {
        int64_t b = op.blocksize;
        if (b < 1) return false;
        if (s[3] % (b * b) != 0) return false;  // C must split into whole blocks
        out_shape = {s[0], s[1] * b, s[2] * b, s[3] / (b * b)};
        break;
      }
    }
    shapes[op.out] = out_shape;
  }
  return true;
}

bool HipEngine::GetOutputShape(const std::vector<int64_t>& input_shape,
                               std::vector<int64_t>& output_shape) const {
  std::unordered_map<std::string, std::vector<int64_t>> shapes;
  if (!PropagateShapes(input_shape, shapes)) return false;
  const auto& it = shapes.find(output_name_);
  if (it == shapes.end() || it->second.size() != 4) return false;
  const auto& nhwc = it->second;
  output_shape = {nhwc[0], nhwc[3], nhwc[1], nhwc[2]};
  return true;
}

int64_t HipEngine::OutputElements(const std::vector<int64_t>& input_shape) {
  std::vector<int64_t> o;
  if (!GetOutputShape(input_shape, o)) return 0;
  return NumElements(o);
}

int HipEngine::InputChannels() const {
  for (const auto& op : ops_) {
    if (op.type != OpType::Conv) continue;
    const auto it = weights_.find(op.in1);
    if (it != weights_.end() && it->second.shape.size() >= 2) {
      return static_cast<int>(it->second.shape[1]);
    }
  }
  return 1;
}

int HipEngine::OutputChannels() const {
  for (auto it = ops_.rbegin(); it != ops_.rend(); ++it) {
    if (it->type != OpType::Conv) continue;
    const auto w = weights_.find(it->in1);
    if (w != weights_.end() && !w->second.shape.empty()) {
      return static_cast<int>(w->second.shape[0]);
    }
  }
  return 1;
}

size_t HipEngine::InputBytes(const std::vector<int64_t>& input_shape) const {
  return static_cast<size_t>(NumElements(input_shape)) * 2;
}

size_t HipEngine::OutputBytes(const std::vector<int64_t>& input_shape) const {
  std::vector<int64_t> o;
  if (!GetOutputShape(input_shape, o)) return 0;
  return static_cast<size_t>(NumElements(o)) * 2;
}

bool HipEngine::EnsureBuilt(const std::vector<int64_t>& input_shape, std::string& error) {
  if (built_ && input_shape == built_input_shape_) return true;
  // HIP's current device is thread-local and this runs on VapourSynth worker
  // threads, so re-select before creating or destroying any resource.
  if (!SetDevice(error)) return false;
  if (input_shape.size() != 4) {
    error = "expected 4D input tensor";
    return false;
  }

  // The kernels have no batch dimension: they index only H/W/C/M, so N must
  // be 1 or only the first image would be produced.
  if (input_shape[0] != 1) {
    error = "batch size must be 1 (the kernels do not implement a batch dimension)";
    return false;
  }
  // The Winograd strip geometry needs at least one 2x2 output tile row/column.
  if (input_shape[2] < 2 || input_shape[3] < 2) {
    error = "input height and width must be at least 2";
    return false;
  }

  DestroyDeviceState();
  built_input_shape_ = input_shape;

  std::unordered_map<std::string, std::vector<int64_t>> shapes;
  if (!PropagateShapes(input_shape, shapes) || shapes.find(output_name_) == shapes.end()) {
    error = "shape propagation failed (unsupported convolution geometry, channel mismatch, or "
            "DepthToSpace block size)";
    return false;
  }

  // --- Buffer allocation with liveness-based reuse ---
  std::unordered_map<std::string, int> last_use;
  for (size_t oi = 0; oi < ops_.size(); ++oi) {
    const auto& op = ops_[oi];
    last_use[op.in0] = static_cast<int>(oi);
    if (!op.in1.empty() && !weights_.count(op.in1)) last_use[op.in1] = static_cast<int>(oi);
    if (op.do_add && !op.add_input.empty() && !weights_.count(op.add_input)) {
      last_use[op.add_input] = static_cast<int>(oi);
    }
  }
  last_use[output_name_] = static_cast<int>(ops_.size());

  // Any HIP failure below unwinds the partially created device state instead
  // of leaving null staging pointers or invalid tensor indices behind.
  auto hip_fail = [&](const char* what, hipError_t e) -> bool {
    error = std::string(what) + " failed: " + hipGetErrorString(e);
    DestroyDeviceState();
    return false;
  };

  std::vector<std::pair<int, int>> slots;
  auto alloc_tensor = [&](const std::string& name, const std::vector<int64_t>& shape,
                          const void* constant_data, size_t constant_bytes, int producer) -> int {
    if (tensor_map_.count(name)) return tensor_map_[name].buffer_index;
    size_t bytes = static_cast<size_t>(NumElements(shape) * 2);
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
      hipError_t e = hipMalloc(&ptr, bytes);
      if (e != hipSuccess) {
        error = std::string("hipMalloc of ") + std::to_string(bytes) + " bytes for tensor '" + name +
                "' failed: " + hipGetErrorString(e);
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
    tensor_map_[name] = info;
    if (constant_data != nullptr) {
      hipError_t e = hipMemcpy(buffers_[buffer_index].ptr, constant_data, constant_bytes,
                               hipMemcpyHostToDevice);
      if (e != hipSuccess) {
        error = std::string("hipMemcpy of constant '") + name + "' failed: " + hipGetErrorString(e);
        return -1;
      }
    }
    return buffer_index;
  };

  input_buffer_ = alloc_tensor(input_name_, shapes[input_name_], nullptr, 0, 0);
  if (input_buffer_ < 0) {
    DestroyDeviceState();
    return false;
  }
  for (size_t oi = 0; oi < ops_.size(); ++oi) {
    const auto& op = ops_[oi];
    const int producer = static_cast<int>(oi);
    if (alloc_tensor(op.in0, shapes[op.in0], nullptr, 0, producer) < 0 ||
        (!op.in1.empty() && !weights_.count(op.in1) &&
         alloc_tensor(op.in1, shapes[op.in1], nullptr, 0, producer) < 0) ||
        (op.do_add && !op.add_input.empty() && !weights_.count(op.add_input) &&
         alloc_tensor(op.add_input, shapes[op.add_input], nullptr, 0, producer) < 0) ||
        alloc_tensor(op.out, shapes[op.out], nullptr, 0, producer) < 0) {
      DestroyDeviceState();
      return false;
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
        const auto& wspec = weights_.at(op.in1);
        const uint32_t M = static_cast<uint32_t>(wspec.shape[0]);
        const uint32_t C = static_cast<uint32_t>(wspec.shape[1]);
        const uint32_t H = static_cast<uint32_t>(in_shape[1]);
        const uint32_t W = static_cast<uint32_t>(in_shape[2]);
        const uint32_t M_pad = (M + 31u) / 32u * 32u;
        const uint32_t C_pad = (C + 15u) / 16u * 16u;
        const uint32_t kb_stride = M_pad / 16u;
        const uint32_t C_blocks = C_pad / 16u;
        const int in_idx = tensor_map_[op.in0].buffer_index;

        // The compact kernels cover at most 64 output channels: direct_conv
        // writes co in [0,56]+8 and winograd's four wave32s own exactly four
        // 16-ko k-blocks. Larger M would silently leave channels unwritten.
        if (M > 64u) {
          error = "conv supports at most 64 output channels (got " + std::to_string(M) + ")";
          return false;
        }
        if (C == 1u) {
          // Direct 3x3 kernel for the C=1 first conv: [t][c][co] + bias.
          const uint32_t wt_size = 9u * C * M + M;
          std::vector<uint16_t> d_buf(wt_size, 0);
          for (uint32_t t = 0; t < 9u; ++t)
            for (uint32_t c = 0; c < C; ++c)
              for (uint32_t ko = 0; ko < M; ++ko)
                FloatToHalfBitsRNE(
                    HipEngine::WeightFloat(wspec.data, wspec.dtype, static_cast<size_t>(ko) * C * 9u + c * 9u + t),
                    d_buf[(t * C + c) * M + ko]);
          if (!op.conv_bias.empty() && weights_.count(op.conv_bias)) {
            const auto& bs = weights_[op.conv_bias];
            for (uint32_t ko = 0; ko < M; ++ko) {
              uint16_t uh;
              FloatToHalfBitsRNE(HipEngine::WeightFloat(bs.data, bs.dtype, ko), uh);
              std::memcpy(d_buf.data() + 9u * C * M + ko, &uh, 2);
            }
          }
          void* wt_d = nullptr;
          {
            hipError_t e = hipMalloc(&wt_d, d_buf.size() * 2);
            if (e != hipSuccess) return hip_fail("hipMalloc of direct-conv weights", e);
            e = hipMemcpy(wt_d, d_buf.data(), d_buf.size() * 2, hipMemcpyHostToDevice);
            if (e != hipSuccess) return hip_fail("hipMemcpy of direct-conv weights", e);
          }
          int wt_idx = static_cast<int>(buffers_.size());
          buffers_.push_back(HipBuffer{wt_d, d_buf.size() * 2});

          int add_idx = wt_idx;
          if (op.do_add && !op.add_input.empty() && tensor_map_.count(op.add_input)) {
            add_idx = tensor_map_[op.add_input].buffer_index;
          }
          WinogradPush p{};
          p.h = H; p.w = W; p.c = C; p.m = M;
          p.do_silu = op.do_silu ? 1u : 0u;
          p.do_add = op.do_add ? 1u : 0u;
          dop.params.resize(sizeof(WinogradPush) / sizeof(uint32_t));
          std::memcpy(dop.params.data(), &p, sizeof(p));
          // 8 pixels per block, 2D grid; ceil so widths that are not a
          // multiple of 8 still schedule their final partial block.
          dop.dispatch_x = DivCeil(W, 8u);
          dop.dispatch_y = H;
          dop.dispatch_z = 1;
          dop.input_buffers = {in_idx, wt_idx, add_idx};
          dop.output_buffer = tensor_map_[op.out].buffer_index;
          break;
        }
        const float G[4][3] = {{1.0f, 0.0f, 0.0f},
                               {0.5f, 0.5f, 0.5f},
                               {0.5f, -0.5f, 0.5f},
                               {0.0f, 0.0f, 1.0f}};
        std::vector<uint16_t> u_buf(kb_stride * 4u * C_blocks * 4u * 256u + M_pad, 0);
        for (uint32_t ko = 0; ko < M; ++ko) {
          for (uint32_t c = 0; c < C; ++c) {
            float g[9];
            for (uint32_t t = 0; t < 9; ++t) {
              g[t] = HipEngine::WeightFloat(wspec.data, wspec.dtype, (static_cast<size_t>(ko) * C + c) * 9 + t);
            }
            float Gg[4][3];
            for (uint32_t r = 0; r < 4; ++r)
              for (uint32_t cc = 0; cc < 3; ++cc) {
                Gg[r][cc] = 0.0f;
                for (uint32_t k = 0; k < 3; ++k) Gg[r][cc] += G[r][k] * g[k * 3 + cc];
              }
            for (uint32_t r = 0; r < 4; ++r)
              for (uint32_t cc = 0; cc < 4; ++cc) {
                float u = 0.0f;
                for (uint32_t k = 0; k < 3; ++k) u += Gg[r][k] * G[cc][k];
                uint32_t wp = r * 4 + cc;
                uint16_t uh;
                FloatToHalfBitsRNE(u, uh);
                std::memcpy(u_buf.data() + (ko / 16u) * (4u * C_blocks * 4u * 256u) + (wp / 4u) * (C_blocks * 4u * 256u) +
                                (c / 16u) * 1024u + (wp % 4u) * 256u + (ko % 16u) * 16u + (c % 16u),
                            &uh, 2);
              }
          }
        }
        if (!op.conv_bias.empty() && weights_.count(op.conv_bias)) {
          const auto& bs = weights_[op.conv_bias];
          for (uint32_t ko = 0; ko < M; ++ko) {
            uint16_t uh;
            FloatToHalfBitsRNE(HipEngine::WeightFloat(bs.data, bs.dtype, ko), uh);
            std::memcpy(u_buf.data() + kb_stride * 4u * C_blocks * 4u * 256u + ko, &uh, 2);
          }
        }
        void* wt_u = nullptr;
        {
          hipError_t e = hipMalloc(&wt_u, u_buf.size() * 2);
          if (e != hipSuccess) return hip_fail("hipMalloc of winograd weights", e);
          e = hipMemcpy(wt_u, u_buf.data(), u_buf.size() * 2, hipMemcpyHostToDevice);
          if (e != hipSuccess) return hip_fail("hipMemcpy of winograd weights", e);
        }
        int wt_idx = static_cast<int>(buffers_.size());
        buffers_.push_back(HipBuffer{wt_u, u_buf.size() * 2});

        int add_idx = wt_idx;
        if (op.do_add && !op.add_input.empty() && tensor_map_.count(op.add_input)) {
          add_idx = tensor_map_[op.add_input].buffer_index;
        }
        if (op.do_add && !op.add_input.empty() && add_idx == wt_idx) {
          error = "fused residual add input '" + op.add_input + "' has no device buffer";
          return false;
        }
        WinogradPush p{};
        p.h = H; p.w = W; p.c = C; p.m = M;
        p.m_pad = M_pad; p.c_pad = C_pad;
        // Kernel tile geometry: each workgroup covers a 2-row x 8-tile-column
        // region (16 F(2x2) tiles). Round the tile counts UP so odd H/W and
        // partial regions are still scheduled; the kernel guards the writes.
        const uint32_t tiles_w = (W + 1u) / 2u;
        const uint32_t tiles_h = (H + 1u) / 2u;
        p.tiles_w = tiles_w;
        p.do_silu = op.do_silu ? 1u : 0u;
        p.do_add = op.do_add ? 1u : 0u;
        dop.params.resize(sizeof(WinogradPush) / sizeof(uint32_t));
        std::memcpy(dop.params.data(), &p, sizeof(p));
        const uint32_t tiles_w8 = (tiles_w + 7u) / 8u;   // workgroups per tile row
        const uint32_t row_pairs = (tiles_h + 1u) / 2u;  // 2 tile rows per workgroup
        const uint32_t kgroups = M_pad / 32u;
        dop.dispatch_x = tiles_w8 * row_pairs * ((kgroups + 1u) / 2u);
        dop.dispatch_y = 1u;
        dop.dispatch_z = 1u;
        dop.input_buffers = {in_idx, wt_idx, add_idx};
        dop.output_buffer = tensor_map_[op.out].buffer_index;
        break;
      }
      case OpType::Sigmoid:
      case OpType::Clip: {
        UnaryPush p{};
        p.numel = static_cast<uint32_t>(NumElements(in_shape));
        p.op = op.type == OpType::Sigmoid ? 0u : 1u;
        p.min_val = op.clip_min;
        p.max_val = op.clip_max;
        dop.params.resize(4);
        std::memcpy(dop.params.data(), &p, sizeof(p));
        dop.dispatch_x = DivCeil(p.numel, 256u);
        dop.input_buffers = {tensor_map_[op.in0].buffer_index};
        dop.output_buffer = tensor_map_[op.out].buffer_index;
        break;
      }
      case OpType::Mul:
      case OpType::Add: {
        BinaryPush p{};
        p.numel = static_cast<uint32_t>(NumElements(in_shape));
        p.op = op.type == OpType::Mul ? 1u : 0u;
        dop.params.resize(2);
        std::memcpy(dop.params.data(), &p, sizeof(p));
        dop.dispatch_x = DivCeil(p.numel, 256u);
        dop.input_buffers = {tensor_map_[op.in0].buffer_index, tensor_map_[op.in1].buffer_index};
        dop.output_buffer = tensor_map_[op.out].buffer_index;
        break;
      }
      case OpType::DepthToSpace: {
        DtsPush p{};
        p.c = static_cast<uint32_t>(in_shape[3] / (op.blocksize * op.blocksize));
        p.h = static_cast<uint32_t>(in_shape[1]);
        p.w = static_cast<uint32_t>(in_shape[2]);
        p.b = static_cast<uint32_t>(op.blocksize);
        p.do_clip = op.fuse_clip ? 1u : 0u;
        p.min_val = op.clip_min;
        p.max_val = op.clip_max;
        dop.params.resize(sizeof(DtsPush) / sizeof(uint32_t));
        std::memcpy(dop.params.data(), &p, sizeof(p));
        // 2D grid over the output plane for dts_kernel_2d[_f32] (32x8 threads).
        const auto& osh = tensor_map_[op.out].shape;  // NHWC
        dop.dispatch_x = DivCeil(static_cast<uint64_t>(osh[2]), 32u);
        dop.dispatch_y = DivCeil(static_cast<uint64_t>(osh[1]), 8u);
        dop.input_buffers = {tensor_map_[op.in0].buffer_index};
        dop.output_buffer = tensor_map_[op.out].buffer_index;
        break;
      }
    }
    device_ops_.push_back(std::move(dop));
  }

  input_bytes_ = static_cast<size_t>(NumElements(shapes[input_name_]) * 2);
  output_bytes_ = static_cast<size_t>(NumElements(shapes[output_name_]) * 2);
  // fp32 input staging is sized from the NCHW input shape (element count
  // is layout-independent; the NHWC internal map holds the same count).
  input_ort_bytes_ = input_is_fp32_
                         ? static_cast<size_t>(NumElements(built_input_shape_)) * 4
                         : 0;
  output_ort_bytes_ = output_bytes_ * (output_is_fp32_ ? 2 : 1);

  {
    hipError_t e = hipHostMalloc(&input_staging_, input_bytes_, hipHostMallocDefault);
    if (e != hipSuccess) return hip_fail("hipHostMalloc of input staging", e);
    e = hipHostMalloc(&output_staging_, output_bytes_, hipHostMallocDefault);
    if (e != hipSuccess) return hip_fail("hipHostMalloc of output staging", e);
    if (input_is_fp32_) {
      // Raw-fp32 upload path: pinned host staging + device fp32 buffer.
      e = hipHostMalloc(&input_staging_f32_, input_ort_bytes_, hipHostMallocDefault);
      if (e != hipSuccess) return hip_fail("hipHostMalloc of fp32 input staging", e);
      e = hipMalloc(&input_f32_dev_, input_ort_bytes_);
      if (e != hipSuccess) return hip_fail("hipMalloc of fp32 input buffer", e);
    }
    if (output_is_fp32_) {
      // fp32 download path: device fp32 buffer (on-device cast target) +
      // pinned host staging. The device fp16 output buffer stays the kernels'
      // write target; the cast reads it after the last op.
      e = hipHostMalloc(&output_staging_f32_, output_ort_bytes_, hipHostMallocDefault);
      if (e != hipSuccess) return hip_fail("hipHostMalloc of fp32 output staging", e);
      e = hipMalloc(&output_f32_dev_, output_ort_bytes_);
      if (e != hipSuccess) return hip_fail("hipMalloc of fp32 output buffer", e);
    }
    e = hipDeviceSynchronize();
    if (e != hipSuccess) return hip_fail("hipDeviceSynchronize after setup", e);
  }

  {
    static const bool dbg2 = [] { const char* e = std::getenv("VSHIP_DEBUG"); return e && *e == '1'; }();
    if (dbg2) {
      for (size_t i = 0; i < device_ops_.size(); ++i) {
        uint32_t dts = 0, clp = 0, mm = 0;
        if (device_ops_[i].params.size() >= 13) {
          std::memcpy(&dts, device_ops_[i].params.data() + 9, 4);
          std::memcpy(&clp, device_ops_[i].params.data() + 10, 4);
          std::memcpy(&mm,  device_ops_[i].params.data() + 3, 4);
        }
        std::fprintf(stderr, "[vship] devop%-2zu type=%d dx=%u ob=%d M=%u dts=%u clip=%u\n",
                     i, static_cast<int>(device_ops_[i].type), device_ops_[i].dispatch_x,
                     device_ops_[i].output_buffer, mm, dts, clp);
      }
      std::fprintf(stderr, "[vship] output_buffer_=%d input_buffer_=%d\n", output_buffer_, input_buffer_);
      fflush(stderr);
    }
  }

  built_ = true;
  return true;
}

bool HipEngine::FloatToHalfBitsRNE(float f, uint16_t& h) {
  uint32_t b;
  std::memcpy(&b, &f, sizeof(b));
  const uint32_t sign = (b >> 16u) & 0x8000u;
  const uint32_t e = (b >> 23u) & 0xffu;
  const uint32_t m = b & 0x7fffffu;
  if (e == 0xffu) {
    h = static_cast<uint16_t>(sign | 0x7c00u | (m ? 0x200u : 0u));
    return true;
  }
  int e16 = static_cast<int>(e) - 127 + 15;
  if (e16 >= 31) {
    h = static_cast<uint16_t>(sign | 0x7c00u);
    return true;
  }
  if (e16 <= 0) {
    if (e16 < -10) {
      h = static_cast<uint16_t>(sign);
      return true;
    }
    uint32_t hm = m | 0x800000u;
    const uint32_t shift = static_cast<uint32_t>(14 - e16);
    uint32_t half = hm >> shift;
    const uint32_t rem = hm & ((1u << shift) - 1u);
    const uint32_t halfway = 1u << (shift - 1u);
    if (rem > halfway || (rem == halfway && (half & 1u))) ++half;
    h = static_cast<uint16_t>(sign | half);
    return true;
  }
  uint32_t m16 = m >> 13u;
  const uint32_t rem = m & 0x1fffu;
  if (rem > 0x1000u || (rem == 0x1000u && (m16 & 1u))) ++m16;
  if (m16 == 0x400u) {
    m16 = 0u;
    ++e16;
  }
  if (e16 >= 31) {
    h = static_cast<uint16_t>(sign | 0x7c00u);
    return true;
  }
  h = static_cast<uint16_t>(sign | (static_cast<uint32_t>(e16) << 10u) | m16);
  return true;
}

float HipEngine::WeightFloat(const std::vector<uint8_t>& data, int dtype, size_t idx) {
  // Build() validates dtype and element count before any transform reads
  // these; the guards below only keep a future caller from walking off the
  // end of the buffer.
  if (dtype == ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
    if ((idx + 1) * 4 > data.size()) return 0.0f;
    float v;
    std::memcpy(&v, data.data() + idx * 4, 4);
    return v;
  }
  if (dtype != ONNX_NAMESPACE::TensorProto_DataType_FLOAT16) return 0.0f;
  if ((idx + 1) * 2 > data.size()) return 0.0f;
  uint16_t bits;
  std::memcpy(&bits, data.data() + idx * 2, 2);
  return HalfBitsToFloat(bits);
}

float HipEngine::HalfBitsToFloat(uint16_t h) {
  const uint32_t sign = (static_cast<uint32_t>(h) & 0x8000u) << 16u;
  const uint32_t e = (h >> 10u) & 0x1fu;
  const uint32_t m = h & 0x3ffu;
  uint32_t out;
  if (e == 0u) {
    if (m == 0u) {
      out = sign;
    } else {
      uint32_t e32 = 127u - 15u + 1u;
      uint32_t hm = m;
      while ((hm & 0x400u) == 0u) {
        hm <<= 1u;
        --e32;
      }
      out = sign | (e32 << 23u) | ((hm & 0x3ffu) << 13u);
    }
  } else if (e == 0x1fu) {
    out = sign | 0x7f800000u | (m ? 0x400000u : 0u);
  } else {
    out = sign | ((e + (127u - 15u)) << 23u) | (m << 13u);
  }
  float f;
  std::memcpy(&f, &out, sizeof(f));
  return f;
}

bool HipEngine::Run(const void* input_data, const std::vector<int64_t>& input_shape,
                    void* output_data, std::string& error) {
  static const bool trc = [] { const char* e = std::getenv("VSHIP_TRACE"); return e && *e == '1'; }();
  static uint64_t run_no = 0;
  uint64_t my_no = 0;
  if (trc) { my_no = ++run_no; fprintf(stderr, "[run] enter #%llu in=%p\n",
                                       (unsigned long long)my_no, input_data); fflush(stderr); }
  std::lock_guard<std::mutex> lock(mutex_);
  // Consume any error left in the thread-local last-error slot by an earlier
  // failed call on this thread (e.g. a reported hipMalloc failure): otherwise
  // the launch check below would misattribute it to this frame's kernel.
  (void)hipGetLastError();
  // Re-select the owning device: HIP's current device is thread-local and
  // VapourSynth may hand a frame to a different worker thread each call.
  if (!SetDevice(error)) return false;
  if (!EnsureBuilt(input_shape, error)) return false;
  if (trc) fprintf(stderr, "[run] built #%llu\n", (unsigned long long)my_no), fflush(stderr);

  hipStream_t stream = static_cast<hipStream_t>(stream_);

  // Every HIP call below is checked: a failed transfer, launch or
  // synchronization must surface as a frame error instead of a silently
  // stale/unwritten output buffer.
  auto run_fail = [&](const char* what, hipError_t e) -> bool {
    error = std::string(what) + " failed: " + hipGetErrorString(e);
    return false;
  };

  if (input_is_fp32_) {
    // Raw fp32 upload + on-device cast (no CPU per-pixel conversion).
    std::memcpy(input_staging_f32_, input_data, input_ort_bytes_);
    {
      hipError_t e = hipMemcpyAsync(input_f32_dev_, input_staging_f32_, input_ort_bytes_,
                                    hipMemcpyHostToDevice, stream);
      if (e != hipSuccess) return run_fail("HIP fp32 input upload", e);
    }
    // NCHW upload -> NHWC compute buffer: fused transpose + cast.
    // (built_input_shape_ is NCHW; C=1 makes this a plain cast.)
    {
      const uint32_t iN = static_cast<uint32_t>(built_input_shape_[0]);
      const uint32_t iC = static_cast<uint32_t>(built_input_shape_[1]);
      const uint32_t iH = static_cast<uint32_t>(built_input_shape_[2]);
      const uint32_t iW = static_cast<uint32_t>(built_input_shape_[3]);
      hipLaunchKernelGGL(cast_f32_to_f16_transpose,
                         dim3(DivCeil(static_cast<uint64_t>(iN) * iH * iW, 256u)),
                         dim3(256), 0, stream,
                         static_cast<const float*>(input_f32_dev_),
                         static_cast<_Float16*>(buffers_[input_buffer_].ptr),
                         iN, iC, iH, iW);
      hipError_t e = hipGetLastError();
      if (e != hipSuccess) return run_fail("HIP f32->f16 input cast launch", e);
    }
    {
      static const bool dbg = [] { const char* e = std::getenv("VSHIP_DEBUG"); return e && *e == '1'; }();
      if (dbg && !input_is_fp32_dbg_done_) {
        input_is_fp32_dbg_done_ = true;
        std::fprintf(stderr, "[vship] fp32 path: numel=%u ort=%zu f32dev=%p infmt_ptr=%p\n",
                     (uint32_t)(input_bytes_ / 2), input_ort_bytes_,
                     input_f32_dev_, buffers_[input_buffer_].ptr);
      }
    }
  } else {
    std::memcpy(input_staging_, input_data, input_bytes_);
    hipError_t e = hipMemcpyAsync(buffers_[input_buffer_].ptr, input_staging_, input_bytes_,
                                  hipMemcpyHostToDevice, stream);
    if (e != hipSuccess) return run_fail("HIP input upload", e);
  }

  // VSHIP_PROFILE=1: per-op GPU timestamps via HIP events.
  static const bool prof = [] { const char* e = std::getenv("VSHIP_PROFILE"); return e && *e == '1'; }();
  std::vector<hipEvent_t> ev;
  if (prof) {
    ev.resize(2u * device_ops_.size() + 2u);
    for (auto& e : ev) {
      hipError_t ce = hipEventCreate(&e);
      if (ce != hipSuccess) {
        for (auto& d : ev) {
          if (d) hipEventDestroy(d);
        }
        return run_fail("HIP event creation", ce);
      }
    }
    hipError_t re = hipEventRecord(ev[0], stream);
    if (re != hipSuccess) {
      for (auto& d : ev) hipEventDestroy(d);
      return run_fail("HIP event record", re);
    }
  }
  // Releases profiling events on any early return.
  auto destroy_events = [&]() {
    if (prof) {
      for (auto& d : ev) hipEventDestroy(d);
    }
  };

  if (trc) fprintf(stderr, "[run] #%llu cast launched\n", (unsigned long long)my_no), fflush(stderr);
  bool dts_fused_fp32 = false;
  for (size_t dop_i = 0; dop_i < device_ops_.size(); ++dop_i) {
    const auto& dop = device_ops_[dop_i];
    if (trc) fprintf(stderr, "[run] #%llu op%zu type=%d\n", (unsigned long long)my_no, dop_i, (int)dop.type), fflush(stderr);
    if (prof) {
      hipError_t re = hipEventRecord(ev[1u + 2u * dop_i], stream);
      if (re != hipSuccess) {
        destroy_events();
        return run_fail("HIP event record", re);
      }
    }
    void* in_bufs[3] = {nullptr, nullptr, nullptr};
    for (size_t i = 0; i < dop.input_buffers.size() && i < 3; ++i) {
      if (dop.input_buffers[i] >= 0) in_bufs[i] = buffers_[dop.input_buffers[i]].ptr;
    }
    void* out_buf = buffers_[dop.output_buffer].ptr;
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
        hipLaunchKernelGGL(unary_kernel, dim3(dop.dispatch_x), dim3(256), 0, stream,
                           static_cast<const _Float16*>(in_bufs[0]), static_cast<_Float16*>(out_buf), p);
        break;
      }
      case OpType::Mul:
      case OpType::Add: {
        BinaryParams p{};
        std::memcpy(&p, dop.params.data(), sizeof(p));
        hipLaunchKernelGGL(binary_kernel, dim3(dop.dispatch_x), dim3(256), 0, stream,
                           static_cast<const _Float16*>(in_bufs[0]),
                           static_cast<const _Float16*>(in_bufs[1]), static_cast<_Float16*>(out_buf), p);
        break;
      }
      case OpType::DepthToSpace: {
        DtsParams p{};
        std::memcpy(&p, dop.params.data(), sizeof(p));
        if (output_is_fp32_ && dop.output_buffer == output_buffer_ && p.b == 2u) {
          // Final-output DTS on an fp32-output model: input-centric pass
          // writes floats directly, skipping the separate cast over 8M px.
          // Restricted to blocksize 2, which is what the kernel's whole-block
          // half4 load and B*B layout implement.
          hipLaunchKernelGGL(dts_kernel_in_f32, dim3(DivCeil(p.w, 32u), DivCeil(p.h, 8u)), dim3(32, 8), 0, stream,
                             static_cast<const _Float16*>(in_bufs[0]), static_cast<float*>(output_f32_dev_), p);
          dts_fused_fp32 = true;
        } else {
          // Generic fp16 DTS; a non-2 block size on an fp32-output model is
          // converted by the trailing cast below.
          hipLaunchKernelGGL(dts_kernel_2d, dim3(dop.dispatch_x, dop.dispatch_y), dim3(32, 8), 0, stream,
                             static_cast<const _Float16*>(in_bufs[0]), static_cast<_Float16*>(out_buf), p);
        }
        break;
      }
    }
    // hipLaunchKernelGGL discards its launch status; read it before the next
    // HIP call can overwrite the last-error slot.
    {
      hipError_t le = hipGetLastError();
      if (le != hipSuccess) {
        destroy_events();
        error = "HIP kernel launch for op " + std::to_string(dop_i) + " failed: " +
                hipGetErrorString(le);
        return false;
      }
    }
    if (prof) {
      hipError_t re = hipEventRecord(ev[2u + 2u * dop_i], stream);
      if (re != hipSuccess) {
        destroy_events();
        return run_fail("HIP event record", re);
      }
    }
  }

  if (output_is_fp32_) {
    // fp16 compute, fp32 download: cast on-device (unless the final DTS
    // already wrote floats), then download floats. NHWC compute buffer ->
    // NCHW download: fused transpose + cast (C=1 is a plain cast).
    if (!dts_fused_fp32) {
      std::unordered_map<std::string, std::vector<int64_t>> oshapes;
      PropagateShapes(built_input_shape_, oshapes);
      const std::vector<int64_t>& onhwc = oshapes[output_name_];  // N,H,W,C
      const uint32_t oN = static_cast<uint32_t>(onhwc[0]);
      const uint32_t oH = static_cast<uint32_t>(onhwc[1]);
      const uint32_t oW = static_cast<uint32_t>(onhwc[2]);
      const uint32_t oC = static_cast<uint32_t>(onhwc[3]);
      hipLaunchKernelGGL(cast_f16_to_f32_transpose,
                         dim3(DivCeil(static_cast<uint64_t>(oN) * oH * oW, 256u)),
                         dim3(256), 0, stream,
                         static_cast<const _Float16*>(buffers_[output_buffer_].ptr),
                         static_cast<float*>(output_f32_dev_),
                         oN, oC, oH, oW);
      hipError_t le = hipGetLastError();
      if (le != hipSuccess) {
        destroy_events();
        return run_fail("HIP f16->f32 output cast launch", le);
      }
    }
    hipError_t e = hipMemcpyAsync(output_staging_f32_, output_f32_dev_, output_ort_bytes_,
                                  hipMemcpyDeviceToHost, stream);
    if (e != hipSuccess) {
      destroy_events();
      return run_fail("HIP fp32 output download", e);
    }
  } else {
    hipError_t e = hipMemcpyAsync(output_staging_, buffers_[output_buffer_].ptr, output_bytes_,
                                  hipMemcpyDeviceToHost, stream);
    if (e != hipSuccess) {
      destroy_events();
      return run_fail("HIP output download", e);
    }
  }
  if (prof) {
    hipError_t re = hipEventRecord(ev[2u * device_ops_.size()], stream);
    if (re != hipSuccess) {
      destroy_events();
      return run_fail("HIP final event record", re);
    }
    hipError_t se = hipEventSynchronize(ev[2u * device_ops_.size()]);
    if (se != hipSuccess) {
      destroy_events();
      return run_fail("HIP profiling event synchronize", se);
    }
    float ms = 0.f;
    static uint64_t frame_no = 0;
    const bool report = frame_no++ < 3 || (frame_no % 128 == 0);
    if (report) {
      std::fprintf(stderr, "[vship-prof] frame %llu\n", static_cast<unsigned long long>(frame_no));
      for (size_t i = 0; i < device_ops_.size(); ++i) {
        ms = 0.f;
        hipEventElapsedTime(&ms, ev[1u + 2u * i], ev[2u + 2u * i]);
        std::fprintf(stderr, "[vship-prof] op%-2zu type=%d %8.3f ms\n",
                     i, static_cast<int>(device_ops_[i].type), ms);
      }
      fflush(stderr);
    }
    destroy_events();
  }
  // Propagate execution failures: without this a kernel fault (or a device
  // loss) still looks like a successful frame over stale staging data.
  {
    hipError_t e = hipStreamSynchronize(stream);
    if (e != hipSuccess) return run_fail("HIP stream synchronize", e);
  }

  if (output_is_fp32_) {
    std::memcpy(output_data, output_staging_f32_, output_ort_bytes_);
  } else {
    std::memcpy(output_data, output_staging_, output_bytes_);
  }
  if (trc) fprintf(stderr, "[run] done    #%llu out=%p\n", (unsigned long long)my_no, output_data), fflush(stderr);
  return true;
}

}  // namespace vship
