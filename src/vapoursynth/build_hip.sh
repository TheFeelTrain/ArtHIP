#!/usr/bin/env bash
# Builds libhip.so - the standalone HIP VapourSynth plugin (Backend.HIP).
# No ONNX Runtime involved: the ONNX model is parsed directly and run with
# the winograd WMMA kernels from src/hip/hip_kernels.h.
#
# Usage: ./build_hip.sh [output_dir]
set -euo pipefail

SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/hip" && pwd)"
OUT_DIR="${1:-${SRC_DIR}/build}"
mkdir -p "${OUT_DIR}"

HIPCC="${HIPCC:-hipcc}"
CXXFLAGS="-std=c++17 -O3 -ffast-math -fPIC -shared -w -DONNX_ML -DONNX_NAMESPACE=onnx"

VS_INC="${VS_INC:-/usr/include/vapoursynth}"
ONNX_INC="${ONNX_INC:-/usr/include}"

LIBS="-lonnx -lonnx_proto -lprotobuf -lpthread -ldl"

echo "[build] VapourSynth headers: ${VS_INC}"
echo "[build] ONNX headers: ${ONNX_INC}"

"${HIPCC}" --offload-arch=gfx1100 ${CXXFLAGS} \
  -I"${SRC_DIR}" \
  -I"${SRC_DIR}/.." \
  -I"${SRC_DIR}/../../hip" \
  -I"${VS_INC}" \
  -I"${ONNX_INC}" \
  "${SRC_DIR}/vs_hip.cpp" \
  "${SRC_DIR}/hip_engine.cc" \
  "${SRC_DIR}/../common/onnx_utils.cpp" \
  "${SRC_DIR}/../common/convert_float_to_float16.cpp" \
  -o "${OUT_DIR}/libhip.so" \
  ${LIBS}

echo "[build] done: ${OUT_DIR}/libhip.so"
