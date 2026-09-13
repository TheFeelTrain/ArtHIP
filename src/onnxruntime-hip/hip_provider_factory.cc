// HipExecutionProvider - provider factory, bridge entry points and EP plugin factory.
#include <memory>
#include <string>

#include "core/providers/shared_library/provider_api.h"
#define ORT_API_MANUAL_INIT
#include "core/session/onnxruntime_cxx_api.h"

#include "hip_execution_provider.h"

namespace onnxruntime {

struct HipProviderFactory : IExecutionProviderFactory {
  explicit HipProviderFactory(HipExecutionProviderInfo info) : info_(std::move(info)) {}
  ~HipProviderFactory() override = default;

  std::unique_ptr<IExecutionProvider> CreateProvider() override {
    return std::make_unique<HipExecutionProvider>(info_);
  }

 private:
  HipExecutionProviderInfo info_;
};

// Provider bridge implementation (the classic shared-library provider entry point).
struct Hip_Provider final : Provider {
  void* GetInfo() override { return nullptr; }

  std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory(int device_id) override {
    return std::make_shared<HipProviderFactory>(HipExecutionProviderInfo(device_id));
  }

  std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory(const void* provider_options) override {
    if (provider_options != nullptr) {
      const auto& options = *static_cast<const ProviderOptions*>(provider_options);
      HipExecutionProviderInfo info;
      const auto it = options.find("device_id");
      if (it != options.end()) {
        info.device_id = std::stoi(it->second);
      }
      return std::make_shared<HipProviderFactory>(std::move(info));
    }
    return nullptr;
  }

  void Initialize() override {}
  void Shutdown() override {}

  Status CreateIExecutionProvider(const OrtHardwareDevice* const* /*devices*/,
                                  const OrtKeyValuePairs* const* /*ep_metadata*/,
                                  size_t /*num_devices*/,
                                  ProviderOptions& provider_options,
                                  const OrtSessionOptions& /*session_options*/,
                                  const OrtLogger& /*logger*/,
                                  std::unique_ptr<IExecutionProvider>& ep) override {
    HipExecutionProviderInfo info;
    const auto it = provider_options.find("device_id");
    if (it != provider_options.end()) {
      info.device_id = std::stoi(it->second);
    }
    ep = std::make_unique<HipExecutionProvider>(info);
    return Status::OK();
  }
} g_provider;

// OrtEpFactory implementation so the EP can also be loaded via the EP plugin path.
struct HipEpFactory : OrtEpFactory {
  explicit HipEpFactory(const OrtApi& ort_api_in) : ort_api(ort_api_in) {
    ort_version_supported = ORT_API_VERSION;
    GetName = GetNameImpl;
    GetVendor = GetVendorImpl;
    GetVendorId = GetVendorIdImpl;
    GetVersion = GetVersionImpl;
    GetSupportedDevices = GetSupportedDevicesImpl;
    CreateEp = CreateEpImpl;
    ReleaseEp = ReleaseEpImpl;
    CreateAllocator = CreateAllocatorImpl;
    ReleaseAllocator = ReleaseAllocatorImpl;
    CreateDataTransfer = CreateDataTransferImpl;
    IsStreamAware = IsStreamAwareImpl;
    CreateSyncStreamForDevice = CreateSyncStreamForDeviceImpl;
  }

  static const char* GetNameImpl(const OrtEpFactory* /*this_ptr*/) noexcept {
    return kHipExecutionProvider;
  }

  static const char* GetVendorImpl(const OrtEpFactory* /*this_ptr*/) noexcept {
    return "onnxruntime";
  }

  static uint32_t GetVendorIdImpl(const OrtEpFactory* /*this_ptr*/) noexcept {
    return 0;
  }

  static const char* GetVersionImpl(const OrtEpFactory* /*this_ptr*/) noexcept {
    return "1.0.0";
  }

  static OrtStatus* GetSupportedDevicesImpl(OrtEpFactory* this_ptr,
                                            const OrtHardwareDevice* const* devices,
                                            size_t num_devices,
                                            OrtEpDevice** ep_devices,
                                            size_t max_ep_devices,
                                            size_t* num_ep_devices) noexcept {
    size_t& num = *num_ep_devices;
    num = 0;
    auto* factory = static_cast<HipEpFactory*>(this_ptr);
    for (size_t i = 0; i < num_devices && num < max_ep_devices; ++i) {
      if (factory->ort_api.HardwareDevice_Type(devices[i]) == OrtHardwareDeviceType_GPU) {
        OrtStatus* st = factory->ort_api.GetEpApi()->CreateEpDevice(factory, devices[i], nullptr, nullptr,
                                                                    &ep_devices[num]);
        if (st != nullptr) {
          return st;
        }
        ++num;
      }
    }
    return nullptr;
  }

  static OrtStatus* CreateEpImpl(OrtEpFactory* /*this_ptr*/,
                                 const OrtHardwareDevice* const* /*devices*/,
                                 const OrtKeyValuePairs* const* /*ep_metadata*/,
                                 size_t /*num_devices*/,
                                 const OrtSessionOptions* /*session_options*/,
                                 const OrtLogger* /*logger*/,
                                 OrtEp** /*ep*/) noexcept {
    return onnxruntime::CreateStatus(ORT_INVALID_ARGUMENT,
                                     "Hip EP: CreateEp is not used; use the provider bridge instead.");
  }

  static void ReleaseEpImpl(OrtEpFactory* /*this_ptr*/, OrtEp* /*ep*/) noexcept {
    // no-op; we never create OrtEp instances.
  }

  static OrtStatus* CreateAllocatorImpl(OrtEpFactory* /*this_ptr*/,
                                        const OrtMemoryInfo* /*memory_info*/,
                                        const OrtKeyValuePairs* /*allocator_options*/,
                                        OrtAllocator** allocator) noexcept {
    *allocator = nullptr;
    return onnxruntime::CreateStatus(ORT_INVALID_ARGUMENT,
                                     "Hip EP: CreateAllocator is not implemented.");
  }

  static void ReleaseAllocatorImpl(OrtEpFactory* /*this_ptr*/, OrtAllocator* /*allocator*/) noexcept {
    // no-op
  }

  static OrtStatus* CreateDataTransferImpl(OrtEpFactory* /*this_ptr*/,
                                           OrtDataTransferImpl** data_transfer) noexcept {
    *data_transfer = nullptr;
    return nullptr;
  }

  static bool IsStreamAwareImpl(const OrtEpFactory* /*this_ptr*/) noexcept {
    return false;
  }

  static OrtStatus* CreateSyncStreamForDeviceImpl(OrtEpFactory* this_ptr,
                                                  const OrtMemoryDevice* /*memory_device*/,
                                                  const OrtKeyValuePairs* /*stream_options*/,
                                                  OrtSyncStreamImpl** stream) noexcept {
    auto* factory = static_cast<HipEpFactory*>(this_ptr);
    *stream = nullptr;
    return factory->ort_api.CreateStatus(ORT_INVALID_ARGUMENT,
                                         "Hip EP: CreateSyncStreamForDevice is not implemented.");
  }

  const OrtApi& ort_api;
};

}  // namespace onnxruntime

extern "C" {
//
// Public symbols
//

// Provide the ORT C API base locally by delegating to the host (which is
// loaded by the main onnxruntime library). This avoids depending on the
// versioned OrtGetApiBase symbol that is not visible to dlopen'd libraries.
__attribute__((visibility("default"))) const OrtApiBase* OrtGetApiBase() {
  return Provider_GetHost()->OrtGetApiBase();
}

__attribute__((visibility("default"))) OrtStatus* CreateEpFactories(const char* /*registration_name*/, const OrtApiBase* ort_api_base,
                             const OrtLogger* /*default_logger*/,
                             OrtEpFactory** factories, size_t max_factories, size_t* num_factories) noexcept {
  const OrtApi* ort_api = ort_api_base->GetApi(ORT_API_VERSION);
  if (ort_api == nullptr) {
    return onnxruntime::CreateStatus(ORT_INVALID_ARGUMENT, "Hip EP: failed to get OrtApi.");
  }
  if (max_factories < 1) {
    return ort_api->CreateStatus(ORT_INVALID_ARGUMENT,
                                 "Hip EP: not enough space to return the EP factory.");
  }
  std::unique_ptr<onnxruntime::HipEpFactory> factory =
      std::make_unique<onnxruntime::HipEpFactory>(*ort_api);
  factories[0] = factory.release();
  *num_factories = 1;
  return nullptr;
}

__attribute__((visibility("default"))) OrtStatus* ReleaseEpFactory(OrtEpFactory* factory) noexcept {
  delete static_cast<onnxruntime::HipEpFactory*>(factory);
  return nullptr;
}

__attribute__((visibility("default"))) onnxruntime::Provider* GetProvider() {
  return &onnxruntime::g_provider;
}
}  // extern "C"
