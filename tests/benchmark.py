import onnxruntime as ort
import numpy as np
import cv2
import time
import sys
from pathlib import Path

# ============================================================
# Configuration
# ============================================================

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent

# Cache MIGraphX's compiled program. The model has dynamic input dims, so the
# ORT cache key is identical across sizes -- use a per-size dir.
_mxr_cache_root = HERE / ".mxr_cache"
_mxr_cache_root.mkdir(exist_ok=True)

model_path = HERE / "ArtCNN_R8F64_fp16.onnx"
input_image_path = sys.argv[1] if len(sys.argv) > 1 else str(HERE / "test_1024.png")
num_iterations = 100

providers_to_test = [
    #"CPUExecutionProvider",
    "MIGraphXExecutionProvider",
    "HIPExecutionProvider",
    #"WebGpuExecutionProvider"
    #"IREE",
]

# The HIPExecutionProvider is shipped as a shared library provider.
# Register it (if present) so it shows up in the list of available providers.
from onnxruntime.capi import _pybind_state
import os as _os

# Disable the MANGOHUD overlay (it skews GPU clocks/measurement).
_os.environ["MANGOHUD"] = "0"

# Errors only for the DEFAULT logger (used by provider init banners e.g.
# "[MIGraphX EP] MIGraphX ENV Override Variables Set:").
_pybind_state.set_default_logger_severity(3)

_hip_provider_so = _os.environ.get(
    "HIP_PROVIDER_SO",
    str(PROJECT_ROOT / "src/hip/build/libonnxruntime_providers_hip.so"),
)
if _os.path.exists(_hip_provider_so):
    _pybind_state.register_execution_provider_library(
        "HIPExecutionProvider", _hip_provider_so
    )

# The IREE execution provider (https://github.com/iree-org/onnxruntime-ep-iree)
# is an optional external plugin. It needs the IREE compiler shared library
# (libIREECompiler.so) at runtime, which is loaded process-globally.
_iree_provider_so = _os.environ.get("IREE_PROVIDER_SO", "/tmp/onnxruntime-ep-iree/build/libonnxruntime_ep_iree.so")
_iree_compiler_so = _os.environ.get("IREE_COMPILER_SO", "/tmp/iree_dist/lib/libIREECompiler.so")
_iree_target_arch = _os.environ.get("IREE_TARGET_ARCH", "gfx1100")
_iree_available = _os.path.exists(_iree_provider_so) and _os.path.exists(_iree_compiler_so)
if _iree_available:
    _pybind_state.register_execution_provider_library("IREE", _iree_provider_so)

# Set this to True if the ONNX model itself is FP16.
# If False, the script will use the model's declared input type.
FORCE_FP16_INPUT = True


# ============================================================
# Load image
# ============================================================

input_img = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)

if input_img is None:
    raise FileNotFoundError(f"Could not load image: {input_image_path}")

# Normalize to [0, 1]
input_img = input_img.astype(np.float32) / 255.0
input_img = np.clip(input_img, 0.0, 1.0)

# NCHW: [1, 1, H, W]
input_img = input_img[None, None, :, :]

if FORCE_FP16_INPUT:
    input_img = input_img.astype(np.float16)
else:
    input_img = input_img.astype(np.float32)


# ============================================================
# Available providers
# ============================================================

available_providers = ort.get_available_providers()

print("Available providers:")
for provider in available_providers:
    print(f"  {provider}")

print()


# ============================================================
# Benchmark
# ============================================================

# Silence ORT provider init chatter (e.g. MIGraphX's env-var banner): errors only.
_quiet_sess_opts = ort.SessionOptions()
_quiet_sess_opts.log_severity_level = 3


def _strict_session(provider, provider_opts):
    """A session that fails unless `provider` claims the WHOLE graph.

    ``get_providers()`` still lists a provider that claimed nothing (those nodes
    simply run elsewhere), so a provider name proves nothing about placement.
    Disabling the CPU fallback turns "this EP did not take the graph" into an
    explicit session-creation failure.  A plugin execution provider - the HIP EP
    is one - is also absent from ``get_available_providers()``, so availability
    is decided by trying the session rather than by that list.
    """
    options = ort.SessionOptions()
    options.log_severity_level = 3
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    session = ort.InferenceSession(
        str(model_path), options, providers=[(provider, provider_opts)]
    )
    claimed = session.get_providers()
    if not claimed or claimed[0] != provider:
        raise RuntimeError(f"{provider} did not claim the graph (providers: {claimed})")
    return session


results = {}
failures = []

for provider in providers_to_test:

    if provider == "IREE" and not _iree_available:
        print("[SKIP]  IREE is not available (set IREE_PROVIDER_SO / IREE_COMPILER_SO).\n")
        continue

    session = None

    try:
        print(f"[RUN]   {provider}")

        if provider == "IREE":
            # IREE is a device-based provider: select the Vulkan device and
            # specialize the dynamic H/W dims to this image's size.
            iree_devices = ort.get_ep_devices()
            vk_dev = next(
                (d for d in iree_devices if d.device.metadata.get("iree.driver") == "vulkan"),
                None,
            )
            if vk_dev is None:
                print("[ERROR] IREE: no Vulkan device found.\n")
                continue
            h, w = input_img.shape[2], input_img.shape[3]
            sess_options = ort.SessionOptions()
            sess_options.log_severity_level = 3
            sess_options.add_provider_for_devices(
                [vk_dev],
                {
                    "target_arch": _iree_target_arch,
                    "opt_level": "O3",
                    "compiler_lib_path": _iree_compiler_so,
                    "dim_specs": f"H({h},{h}),W({w},{w})",
                },
            )
            session = ort.InferenceSession(str(model_path), sess_options)
        else:
            opts = {}
            if provider == "MIGraphXExecutionProvider":
                h, w = input_img.shape[2], input_img.shape[3]
                _mxr_cache = _mxr_cache_root / f"{h}x{w}"
                _mxr_cache.mkdir(exist_ok=True)
                opts["migraphx_model_cache_dir"] = str(_mxr_cache)
            # Placement is proven, not assumed: this raises when the provider
            # declines a node instead of silently timing a fallback.
            session = _strict_session(provider, opts)

        input_info = session.get_inputs()[0]
        input_name = input_info.name
        model_input_type = input_info.type

        # --------------------------------------------------------
        # Make sure our NumPy dtype matches the ONNX input type.
        # --------------------------------------------------------

        if model_input_type == "tensor(float16)":
            benchmark_input = input_img.astype(np.float16, copy=False)

        elif model_input_type == "tensor(float)":
            benchmark_input = input_img.astype(np.float32, copy=False)

        else:
            raise RuntimeError(
                f"Unsupported model input type: {model_input_type}"
            )

        # --------------------------------------------------------
        # Warmup
        # --------------------------------------------------------

        session.run(None, {input_name: benchmark_input})

        # --------------------------------------------------------
        # Timed runs
        # --------------------------------------------------------

        start_time = time.perf_counter()

        for _ in range(num_iterations):
            pred = session.run(
                None,
                {input_name: benchmark_input}
            )

        end_time = time.perf_counter()

        elapsed = end_time - start_time
        per_iter_ms = (elapsed / num_iterations) * 1000.0

        results[provider] = elapsed

        print(
            f"        {num_iterations} forward steps took "
            f"{elapsed:.16f} s "
            f"({per_iter_ms:.16f} ms/iter)"
        )

        # --------------------------------------------------------
        # Save output
        # --------------------------------------------------------

        pred_out = np.asarray(pred[0])
        pred_out = np.squeeze(pred_out)

        # Convert FP16 output back to FP32 for image processing.
        pred_out = pred_out.astype(np.float32)

        pred_out = np.clip(pred_out, 0.0, 1.0)

        output_img = np.round(pred_out * 255.0).astype(np.uint8)

        out_path = (
            str(HERE / f"{model_path.stem}_{provider}_fp16.png")
        )

        success = cv2.imwrite(out_path, output_img)

        if not success:
            raise RuntimeError(
                f"Failed to save output image: {out_path}"
            )

        print(f"        Saved output to {out_path}\n")

    except Exception as e:
        # Fail fast: report, remember, and exit non-zero below. A swallowed
        # failure used to leave the summary looking like a successful run.
        print(f"[ERROR] Failed to benchmark {provider}:")
        print(f"        {type(e).__name__}: {e}\n")
        failures.append(f"{provider}: {type(e).__name__}: {e}")

    finally:
        if session is not None:
            del session


# ============================================================
# Summary
# ============================================================

print("=" * 70)
print("Benchmark summary (fastest first)")
print("=" * 70)

for provider, elapsed in sorted(
    results.items(),
    key=lambda x: x[1]
):
    per_iter_ms = (elapsed / num_iterations) * 1000.0

    print(
        f"  {provider:35s} : "
        f"{elapsed:7.16f} s  "
        f"({per_iter_ms:7.16f} ms/iter)"
    )

if failures:
    print()
    print(f"{len(failures)} provider(s) failed:")
    for line in failures:
        print(f"  {line}")
    sys.exit(1)
