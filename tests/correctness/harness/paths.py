"""Repository layout and artifact locations for the correctness suite."""

from __future__ import annotations

from pathlib import Path

# tests/correctness/harness/paths.py -> repo root
REPO_ROOT = Path(__file__).resolve().parents[3]
TESTS_DIR = REPO_ROOT / "tests"

# Suite-local scratch (fixtures, built harness, generated scripts). Ignored by
# git through the repository-wide "build/" rule.
SUITE_DIR = Path(__file__).resolve().parents[1]
BUILD_DIR = SUITE_DIR / "build"
FIXTURE_DIR = BUILD_DIR / "fixtures"
SCRIPT_DIR = BUILD_DIR / "scripts"

# Shipped models used by the plugin checks.
#   luma    1 input plane  -> 1 output plane (GRAY, 2x upscale, tail DTS)
#   dehalo  3 input planes -> 3 output planes (RGB/YUV, 1:1)
#   chroma  3 input planes -> 2 output planes (YUV 4:4:4, emits U and V)
SHIPPED_MODELS = {
    "luma": TESTS_DIR / "ArtCNN_R8F64_fp16.onnx",
    "dehalo": TESTS_DIR / "ArtCNN_R8F64_YCbCr_DEHALO.onnx",
    "chroma": TESTS_DIR / "ArtCNN_R8F64_Chroma.onnx",
}

# Production artifacts and their build scripts.
ENGINE_SOURCES = REPO_ROOT / "src/vapoursynth/hip"
HIP_SOURCES = REPO_ROOT / "src/hip"
COMMON_SOURCES = REPO_ROOT / "src/vapoursynth/common"
PLUGIN_SO = ENGINE_SOURCES / "build/libhip.so"
PLUGIN_BUILD_SCRIPT = REPO_ROOT / "src/vapoursynth/build_hip.sh"
EP_SO = HIP_SOURCES / "build/libonnxruntime_providers_hip.so"
EP_BUILD_SCRIPT = HIP_SOURCES / "build.sh"
ENGINE_HARNESS_SO = BUILD_DIR / "libengine_harness.so"
