"""NumPy references and comparison helpers for the correctness suite.

Reference math is float64 so that an fp16 engine result is compared against a
meaningful ground truth rather than against another fp16 rounding. Tolerances
live with the tests that use them.

The synthetic input is a deterministic ramp shared by every fixture; because it
covers [0, 1) densely it exposes dropped rows/columns and NaN as large errors
instead of hiding them behind smooth data.
"""

from __future__ import annotations

import numpy as np

RAMP_MODULUS = 251
RAMP_MULT = 37
RAMP_ADD = 11

#: Default tolerance for a full fp16 convolution chain (fp16 inputs, fp16
#: accumulation order). Observed worst case on the fixture sweep is ~2.2e-3.
FP16_CONV_TOL = 2e-2


def ramp(numel: int) -> np.ndarray:
    """The shared deterministic input ramp, as float32 in [0, 1)."""
    return (
        ((np.arange(numel, dtype=np.int64) * RAMP_MULT + RAMP_ADD) % RAMP_MODULUS).astype(np.float32)
        / RAMP_MODULUS
    )


def raw_input(c: int, h: int, w: int, in_fp32: bool) -> np.ndarray:
    """Flat host buffer in the layout ``HipEngine::Run`` expects.

    fp32-input models take NCHW; fp16-input models take NHWC (the engine's
    compute buffers are NHWC and the plugin packs accordingly).
    """
    flat = ramp(c * h * w)
    return flat.astype(np.float32) if in_fp32 else flat.astype(np.float16)


def logical_input(c: int, h: int, w: int, in_fp32: bool) -> np.ndarray:
    """The same data as a logical NCHW tensor, for reference math.

    fp16-input models receive the ramp already rounded to fp16, so the reference
    must round identically -- otherwise an exact kernel (an identity conv, say)
    appears to be off by a rounding step.
    """
    flat = ramp(c * h * w)
    if in_fp32:
        return flat.reshape(1, c, h, w).astype(np.float64)
    half = flat.astype(np.float16)
    return half.reshape(1, h, w, c).transpose(0, 3, 1, 2).astype(np.float64)


def decode_output(buf: bytes, shape_nchw, out_fp32: bool) -> np.ndarray:
    """Decode a host output buffer into a logical NCHW float64 tensor.

    The engine returns fp32 outputs in NCHW but fp16 outputs in NHWC (a
    documented engine boundary quirk, REVIEW.md P2-11). Channel counts above
    one therefore decode differently per dtype; update this helper together
    with that finding.
    """
    n, c, h, w = (int(v) for v in shape_nchw)
    if out_fp32:
        return np.frombuffer(buf, dtype=np.float32).reshape(n, c, h, w).astype(np.float64)
    return (
        np.frombuffer(buf, dtype=np.float16)
        .reshape(n, h, w, c)
        .transpose(0, 3, 1, 2)
        .astype(np.float64)
    )


def conv3x3(x: np.ndarray, weight: np.ndarray, bias: np.ndarray | None = None, pad: int = 1) -> np.ndarray:
    """Reference 3x3 SAME convolution (the only geometry the kernels support)."""
    m, c, kh, kw = weight.shape
    n, _, h, w = x.shape
    x = x.astype(np.float64)
    xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    out = np.zeros((n, m, h, w), dtype=np.float64)
    for ko in range(m):
        for ci in range(c):
            for r in range(kh):
                for s in range(kw):
                    out[:, ko] += weight[ko, ci, r, s] * xp[:, ci, r : r + h, s : s + w]
        if bias is not None:
            out[:, ko] += bias[ko]
    return out


def depth_to_space_dcr(x: np.ndarray, blocksize: int) -> np.ndarray:
    """Reference ONNX DepthToSpace in DCR mode."""
    n, c, h, w = x.shape
    b = blocksize
    assert c % (b * b) == 0, "channel count must be divisible by blocksize^2"
    return (
        x.reshape(n, b, b, c // (b * b), h, w)
        .transpose(0, 3, 4, 1, 5, 2)
        .reshape(n, c // (b * b), h * b, w * b)
    )


def max_abs_diff(got: np.ndarray, ref: np.ndarray) -> float:
    """Largest absolute difference, or ``inf`` when the output is not finite.

    NaN must never be treated as "close": a kernel that writes nothing at all
    would otherwise pass every comparison.
    """
    if not np.isfinite(got).all():
        return float("inf")
    return float(np.abs(got.astype(np.float64) - ref.astype(np.float64)).max())


def assert_close(got: np.ndarray, ref: np.ndarray, tol: float, label: str) -> float:
    """Assert finiteness and closeness; returns the measured max difference."""
    assert np.isfinite(got).all(), f"{label}: output contains non-finite values"
    diff = max_abs_diff(got, ref)
    assert diff < tol, f"{label}: max abs diff {diff:.6g} >= tolerance {tol:g}"
    return diff
