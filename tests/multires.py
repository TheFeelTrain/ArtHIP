import onnxruntime as ort
import numpy as np
import cv2
import time
import sys
from pathlib import Path
import os

# ============================================================
# Multi-resolution accuracy + speed benchmark for the HIP EP.
# For each resolution (256/512/1024 by default), it compares the
# HIPExecutionProvider output against MIGraphXExecutionProvider
# (fp16 reference) and reports per-iteration latency.
#
# Usage:  python3 multires.py [256 512 1024]
# ============================================================

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent

model_path = HERE / "ArtCNN_R8F64_fp16.onnx"

# Cache MIGraphX's compiled program. The model has dynamic input dims, so the
# ORT cache key is identical across resolutions -- use a per-resolution dir so
# a cached program for one size is never loaded for another.
_mxr_cache_root = HERE / ".mxr_cache"
_mxr_cache_root.mkdir(exist_ok=True)

# The HIPExecutionProvider is shipped as a shared library provider.
# Register it (if present) so it shows up in the list of available providers.
from onnxruntime.capi import _pybind_state

# Disable the MANGOHUD overlay (it skews GPU clocks/measurement).
os.environ["MANGOHUD"] = "0"

# Errors only for the DEFAULT logger (used by provider init banners e.g.
# "[MIGraphX EP] MIGraphX ENV Override Variables Set:").
_pybind_state.set_default_logger_severity(3)

_hip_provider_so = os.environ.get(
    "HIP_PROVIDER_SO",
    str(PROJECT_ROOT / "src/hip/build/libonnxruntime_providers_hip.so"),
)
if os.path.exists(_hip_provider_so):
    _pybind_state.register_execution_provider_library(
        "HIPExecutionProvider", _hip_provider_so
    )


def load(p):
    img = cv2.imread(p, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
    return np.clip(img, 0, 1)[None, None, :, :].astype(np.float16)


# Silence ORT provider init chatter (e.g. MIGraphX's env-var banner): errors only.
_sess_opts = ort.SessionOptions()
_sess_opts.log_severity_level = 3


resolutions = [int(a) for a in sys.argv[1:]] or [256, 512, 1024]

for res in resolutions:
    path = HERE / f"test_{res}.png"
    x = load(path)
    _, _, H, W = x.shape
    print(f"=== {W}x{H} -> {W*2}x{H*2} ===")

    # Reference: CPU provider (runs once, purely for the accuracy check).
    # The CPU conv is slow at large sizes, so cache its output per resolution.
    cpu_ref_path = HERE / f".cpu_ref_{res}.npy"
    if cpu_ref_path.exists():
        ref = np.load(cpu_ref_path)
    else:
        try:
            cpu = ort.InferenceSession(str(model_path), _sess_opts, providers=["CPUExecutionProvider"])
            ref = np.asarray(cpu.run(None, {"input": x})[0])
            np.save(cpu_ref_path, ref)
        except Exception as e:
            print(f"  CPU failed: {e}")
            continue

    hip = ort.InferenceSession(str(model_path), _sess_opts, providers=["HIPExecutionProvider"])
    _mxr_cache = _mxr_cache_root / f"res{res}"
    _mxr_cache.mkdir(exist_ok=True)
    mig = ort.InferenceSession(
        str(model_path),
        _sess_opts,
        providers=[("MIGraphXExecutionProvider", {"migraphx_model_cache_dir": str(_mxr_cache)})],
    )

    # Accuracy vs CPU (reference), for both HIP and MIGraphX.
    for name, sess in [("HIP", hip), ("MIGraphX", mig)]:
        ov = sess.run(None, {"input": x})[0]
        d = np.abs(ref.astype(np.float32) - ov.astype(np.float32))
        rc = np.clip(np.round(ref * 255), 0, 255).astype(np.uint8)
        rv = np.clip(np.round(ov * 255), 0, 255).astype(np.uint8)
        mm = (rc != rv).sum()
        print(
            f"  {name} vs CPU: maxdiff {d.max():.5f} meandiff {d.mean():.6f} "
            f"uint8 mismatch {mm}/{rc.size} ({100 * mm / rc.size:.2f}%)"
        )

    # Speed. Time 100 runs (matching benchmark.py); a short warmup re-boosts
    # the GPU clock after the CPU reference run.
    for name, sess in [("MIGraphX", mig), ("HIP", hip)]:
        for _ in range(10):
            sess.run(None, {"input": x})
        n = 100
        t0 = time.perf_counter()
        for _ in range(n):
            sess.run(None, {"input": x})
        print(f"  {name}: {(time.perf_counter() - t0) / n * 1000:.2f} ms/iter")

    del mig, hip
