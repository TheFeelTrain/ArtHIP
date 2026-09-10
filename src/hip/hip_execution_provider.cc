// HIPExecutionProvider - execution provider implementation.
#include "hip_execution_provider.h"

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "core/common/common.h"
#include "core/session/onnxruntime_cxx_api.h"

#include "hip_graph.h"

namespace onnxruntime {

namespace {

// Whether a single node can be executed by the HIP EP.
bool IsNodeSupported(const onnxruntime::GraphViewer& graph, const Node* node) {
  const std::string& optype = node->OpType();
  // Boundary Cast nodes (fp32->fp16 at the input, fp16->fp32 at the output)
  // are folded into the EP's IO conversion; any other Cast stays on CPU.
  if (optype == "Cast") {
    auto id = node->InputDefs();
    auto od = node->OutputDefs();
    if (id.empty() || od.empty() || id[0] == nullptr || od[0] == nullptr ||
        !id[0]->Exists() || !od[0]->Exists() || !id[0]->TypeAsProto() || !od[0]->TypeAsProto() ||
        !id[0]->TypeAsProto()->has_tensor_type() || !od[0]->TypeAsProto()->has_tensor_type()) {
      return false;
    }
    const int in_t = id[0]->TypeAsProto()->tensor_type().elem_type();
    const int out_t = od[0]->TypeAsProto()->tensor_type().elem_type();
    if (in_t == ONNX_NAMESPACE::TensorProto_DataType_FLOAT &&
        out_t == ONNX_NAMESPACE::TensorProto_DataType_FLOAT16) {
      for (const auto* gi : graph.GetInputs()) {
        if (gi != nullptr && gi->Exists() && gi->Name() == id[0]->Name()) {
          return true;
        }
      }
      return false;
    }
    if (in_t == ONNX_NAMESPACE::TensorProto_DataType_FLOAT16 &&
        out_t == ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
      for (const auto* go : graph.GetOutputs()) {
        if (go != nullptr && go->Exists() && go->Name() == od[0]->Name()) {
          return true;
        }
      }
      return false;
    }
    return false;
  }
  if (optype != "Conv" && optype != "Sigmoid" && optype != "Mul" && optype != "Add" &&
      optype != "DepthToSpace" && optype != "Clip") {
    return false;
  }
  if (!node->Domain().empty()) {
    return false;
  }

  // Only fp16 tensors are supported for now.
  bool fp16_only = true;
  node->ForEachDef([&fp16_only](const onnxruntime::NodeArg& node_arg, bool /*is_input*/) {
    const auto* type_proto = node_arg.TypeAsProto();
    if (type_proto == nullptr || !type_proto->has_tensor_type() ||
        type_proto->tensor_type().elem_type() != ONNX_NAMESPACE::TensorProto_DataType_FLOAT16) {
      fp16_only = false;
    }
  });
  if (!fp16_only) {
    return false;
  }

  if (optype == "Conv") {
    const auto& attrs = node->GetAttributes();
    if (attrs.count("strides")) {
      const auto& a = attrs.at("strides");
      if (a.ints_size() >= 2 && (a.ints(0) != 1 || a.ints(1) != 1)) {
        return false;
      }
    }
    if (attrs.count("dilations")) {
      const auto& a = attrs.at("dilations");
      if (a.ints_size() >= 2 && (a.ints(0) != 1 || a.ints(1) != 1)) {
        return false;
      }
    }
    if (attrs.count("group") && attrs.at("group").i() != 1) {
      return false;
    }
    bool has_pads = false;
    int pad_h = 0;
    int pad_w = 0;
    if (attrs.count("pads")) {
      const auto& a = attrs.at("pads");
      if (a.ints_size() >= 4) {
        pad_h = static_cast<int>(a.ints(0));
        pad_w = static_cast<int>(a.ints(2));
        has_pads = true;
      }
    }
    if (attrs.count("auto_pad")) {
      const std::string auto_pad = attrs.at("auto_pad").s();
      if (auto_pad == "SAME_UPPER" || auto_pad == "SAME_LOWER") {
        pad_h = 1;
        pad_w = 1;
        has_pads = true;
      }
    }
    if (has_pads && (pad_h != 1 || pad_w != 1)) {
      return false;
    }

    // Kernel must be 3x3.
    auto defs = node->InputDefs();
    if (defs.size() < 2) {
      return false;
    }
    const std::string& weight_name = defs[1]->Name();
    const ONNX_NAMESPACE::TensorProto* tensor_proto = nullptr;
    if (!graph.GetInitializedTensor(weight_name, tensor_proto)) {
      return false;
    }
    const auto& dims = tensor_proto->dims();
    if (dims.size() != 4 || dims[2] != 3 || dims[3] != 3) {
      return false;
    }
  }

  return true;
}

}  // namespace

HipExecutionProvider::HipExecutionProvider(const HipExecutionProviderInfo& info)
    : IExecutionProvider(onnxruntime::kHipExecutionProvider, OrtDevice()), info_(info) {
  InitProviderOrtApi();
  context_ = std::make_shared<hip::HipContext>();
  // Wire the configured device into the context and refuse capabilities if the
  // context could not be created: a null context must never be dereferenced.
  if (!context_->Initialize(info_.device_id)) {
    fprintf(stderr, "[hip] context initialization failed for device %d; HIP EP disabled\n",
            info_.device_id);
    context_ = nullptr;
  }
}

HipExecutionProvider::~HipExecutionProvider() = default;

std::vector<std::unique_ptr<ComputeCapability>>
HipExecutionProvider::GetCapability(const onnxruntime::GraphViewer& graph_viewer,
                                    const IKernelLookup& /*kernel_lookup*/,
                                    const GraphOptimizerRegistry& /*graph_optimizer_registry*/,
                                    IResourceAccountant* /*resource_accountant*/) const {
  std::vector<std::unique_ptr<ComputeCapability>> result;

  // Without a usable context the EP cannot execute anything; claiming nodes
  // would only move work from a working provider onto a broken one.
  if (!context_ || !context_->initialized()) {
    return result;
  }

  const auto& nodes_in_order = graph_viewer.GetNodesInTopologicalOrder();
  if (nodes_in_order.empty()) {
    return result;
  }

  // Partition the graph into maximal consecutive runs of supported nodes.
  std::vector<std::vector<NodeIndex>> runs;
  {
    std::vector<NodeIndex> current;
    for (const auto& node_idx : nodes_in_order) {
      const Node* node = graph_viewer.GetNode(node_idx);
      if (node != nullptr && IsNodeSupported(graph_viewer, node)) {
        current.push_back(node_idx);
      } else if (!current.empty()) {
        runs.push_back(std::move(current));
        current.clear();
      }
    }
    if (!current.empty()) {
      runs.push_back(std::move(current));
    }
  }

  static std::atomic<int> metadef_id{0};
  for (auto& run : runs) {
    std::unordered_set<std::string> produced;
    std::unordered_set<std::string> consumed;
    for (const auto& node_idx : run) {
      const Node* node = graph_viewer.GetNode(node_idx);
      for (const auto* out : node->OutputDefs()) {
        if (out != nullptr && out->Exists()) {
          produced.insert(out->Name());
        }
      }
    }
    for (const auto& node_idx : run) {
      const Node* node = graph_viewer.GetNode(node_idx);
      for (const auto* in : node->InputDefs()) {
        if (in != nullptr && in->Exists()) {
          consumed.insert(in->Name());
        }
      }
    }

    auto sub_graph = onnxruntime::IndexedSubGraph::Create();
    for (const auto& node_idx : run) {
      sub_graph->Nodes().push_back(node_idx);
    }
    auto meta_def = IndexedSubGraph_MetaDef::Create();
    meta_def->name() = "HipKernel_graph_" + std::to_string(metadef_id.fetch_add(1));
    meta_def->domain() = kMSDomain;
    meta_def->since_version() = 1;
    for (const auto& node_idx : run) {
      const Node* node = graph_viewer.GetNode(node_idx);
      for (const auto* in : node->InputDefs()) {
        if (in != nullptr && in->Exists() && produced.count(in->Name()) == 0) {
          meta_def->inputs().push_back(in->Name());
        }
      }
      for (const auto* out : node->OutputDefs()) {
        if (out != nullptr && out->Exists() && consumed.count(out->Name()) == 0) {
          meta_def->outputs().push_back(out->Name());
        }
      }
    }
    sub_graph->SetMetaDef(std::move(meta_def));
    result.push_back(ComputeCapability::Create(std::move(sub_graph)));
  }
  return result;
}

common::Status HipExecutionProvider::Compile(const std::vector<FusedNodeAndGraph>& fused_nodes_and_graphs,
                                             std::vector<NodeComputeInfo>& node_compute_funcs) {
  if (!context_ || !context_->initialized()) {
    return Status(common::ONNXRUNTIME, common::FAIL,
                  "HIP EP: device context is not initialized (device " +
                      std::to_string(info_.device_id) + ")");
  }
  for (const auto& fused_node_graph : fused_nodes_and_graphs) {
    const GraphViewer& graph_body_viewer = fused_node_graph.filtered_graph;

    auto hip_graph = std::make_shared<hip::HipGraph>(context_);
    ORT_RETURN_IF_ERROR(hip_graph->CompileGraph(graph_body_viewer));

    NodeComputeInfo compute_info;
    compute_info.create_state_func = [hip_graph](ComputeContext* /*context*/, FunctionState* state) {
      *state = static_cast<FunctionState>(hip_graph.get());
      return 0;
    };
    compute_info.release_state_func = [](FunctionState /*state*/) {
      // The HipGraph is kept alive by the shared_ptr captured in the lambdas below.
    };
    compute_info.compute_func = [hip_graph](FunctionState /*state*/, const OrtApi* /*api*/,
                                            OrtKernelContext* context) -> Status {
      auto t0 = std::chrono::high_resolution_clock::now();
      Ort::KernelContext ctx(context);
      auto input = ctx.GetInput(0);
      auto input_info = input.GetTensorTypeAndShapeInfo();
      std::vector<int64_t> input_shape = input_info.GetShape();
      if (input_shape.size() != 4) {
        return Status(common::ONNXRUNTIME, common::FAIL,
                      "HIP EP: expected a 4D input tensor, got " + std::to_string(input_shape.size()) + "D.");
      }
      const void* input_data = input.GetTensorRawData();
      // The boundary Cast fold means the ORT-side IO may be fp32.
      const int in_elem_type = input.GetTensorTypeAndShapeInfo().GetElementType();
      const size_t in_elem_size =
          (in_elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) ? sizeof(float) : sizeof(uint16_t);
      size_t input_bytes = static_cast<size_t>(input_info.GetElementCount()) * in_elem_size;
      auto t1 = std::chrono::high_resolution_clock::now();

      std::vector<int64_t> output_shape;
      ORT_RETURN_IF_ERROR(hip_graph->GetOutputShape(input_shape, output_shape));

      auto output = ctx.GetOutput(0, output_shape.data(), output_shape.size());
      void* output_data = output.GetTensorMutableRawData();
      auto t2 = std::chrono::high_resolution_clock::now();

      int64_t output_numel = 1;
      for (auto d : output_shape) {
        output_numel *= d;
      }
      const int out_elem_type = output.GetTensorTypeAndShapeInfo().GetElementType();
      const size_t out_elem_size =
          (out_elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) ? sizeof(float) : sizeof(uint16_t);
      size_t output_bytes = static_cast<size_t>(output_numel) * out_elem_size;

      auto st = hip_graph->Run(input_data, input_bytes, input_shape, output_data, output_bytes);
      auto t3 = std::chrono::high_resolution_clock::now();
      if (std::getenv("HIP_PROFILE")) {
        fprintf(stderr, "[hipprof] ort_io=%.3fms getout=%.3fms run=%.3fms\n",
                std::chrono::duration<double, std::milli>(t1 - t0).count(),
                std::chrono::duration<double, std::milli>(t2 - t1).count(),
                std::chrono::duration<double, std::milli>(t3 - t2).count());
      }
      return st;
    };
    node_compute_funcs.push_back(std::move(compute_info));
  }
  return Status::OK();
}

}  // namespace onnxruntime
