// HipExecutionProvider - ONNX Runtime execution provider.
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "core/providers/shared_library/provider_api.h"

namespace onnxruntime {

constexpr const char* kHipExecutionProvider = "HIPExecutionProvider";

namespace hip {
class HipContext;
class HipGraph;
}  // namespace hip

// Information needed to construct the HIP execution provider.
struct HipExecutionProviderInfo {
  int device_id{0};

  explicit HipExecutionProviderInfo(int device_id = 0) : device_id(device_id) {}
};

// Logical device representation.
class HipExecutionProvider : public IExecutionProvider {
 public:
  explicit HipExecutionProvider(const HipExecutionProviderInfo& info);
  ~HipExecutionProvider() override;

  std::vector<std::unique_ptr<ComputeCapability>> GetCapability(
      const onnxruntime::GraphViewer& graph_viewer,
      const IKernelLookup& kernel_lookup,
      const GraphOptimizerRegistry& graph_optimizer_registry,
      IResourceAccountant* resource_accountant) const override;

  common::Status Compile(const std::vector<FusedNodeAndGraph>& fused_nodes_and_graphs,
                         std::vector<NodeComputeInfo>& node_compute_funcs) override;

  int GetDeviceId() const override { return info_.device_id; }

 private:
  HipExecutionProviderInfo info_;
  std::shared_ptr<hip::HipContext> context_;
};

}  // namespace onnxruntime
