"""VapourSynth plugin checks.

Covers REVIEW.md P1-1 (subsampled planes and channel mismatches must be refused)
plus the shipped multi-plane model paths:

* ``luma``   - 1 input plane  -> 1 output plane (2x upscale, tail DTS)
* ``dehalo`` - 3 input planes -> 3 output planes (RGB or YUV, 1:1)
* ``chroma`` - 3 input planes -> 2 output planes (emits U and V through the
  flexible-output protocol, exactly how vsscale's ``R8F64_Chroma`` consumes it)

Simple load/format checks run through ``vspipe``; anything that needs plane data
or frame properties runs as a Python probe through :mod:`harness.vsrun`. Both run
in a child process, which mirrors real usage and guarantees the plugin under test
is the installed binary (VapourSynth auto-loads plugins by id, so an explicit
``LoadPlugin`` of a second copy is rejected).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import textwrap

import pytest

from harness import paths, vsrun

pytestmark = pytest.mark.skipif(shutil.which("vspipe") is None, reason="vspipe is not on PATH")

TIMEOUT_S = 600

#: Plugin computes in fp16 by default while the reference below is fp32, so the
#: comparison carries fp16 rounding error (measured ~0.006 on the chroma model).
CHROMA_TOL = 0.02


# ---------------------------------------------------------------------------
# vspipe-based load / format checks
# ---------------------------------------------------------------------------


def _script(fmt: str, model, *, flexible: bool = False, width: int = 64, height: int = 64) -> str:
    output = 'out["clip"].set_output()' if flexible else "out.set_output()"
    extra = ", flexible_output_prop='MlrtFlexible'" if flexible else ""
    return textwrap.dedent(
        f"""
        import vapoursynth as vs
        core = vs.core
        core.max_cache_size = 256
        clip = core.std.BlankClip(width={width}, height={height}, format=vs.{fmt}, length=1)
        out = core.hip.Model(clip, {str(model)!r}, fp16=1{extra})
        {output}
        """
    ).strip()


def _run_vspipe(name: str, script_body: str) -> subprocess.CompletedProcess:
    paths.SCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    script = paths.SCRIPT_DIR / f"{name}.vpy"
    script.write_text(script_body)
    env = dict(os.environ, MANGOHUD="0")
    return subprocess.run(
        ["vspipe", "-p", str(script), "--"],
        capture_output=True, text=True, env=env, timeout=TIMEOUT_S,
    )


def _expect_accept(name: str, fmt: str, model, *, flexible: bool = False) -> None:
    result = _run_vspipe(name, _script(fmt, model, flexible=flexible))
    assert result.returncode == 0, (
        f"{name}: vspipe failed for a valid clip/model pair\n{result.stdout}{result.stderr}"
    )


def _expect_reject(name: str, fmt: str, model, match: str, *, flexible: bool = False) -> None:
    result = _run_vspipe(name, _script(fmt, model, flexible=flexible))
    combined = result.stdout + result.stderr
    assert result.returncode != 0, f"{name}: expected rejection but vspipe succeeded"
    assert match in combined, f"{name}: expected {match!r} in the error, got:\n{combined}"


# ---------------------------------------------------------------------------
# Accepted combinations (the validation must not be over-strict)
# ---------------------------------------------------------------------------

ACCEPT_CASES = [
    ("accept_gray_luma", "GRAY8", "luma", False),
    ("accept_yuv444_dehalo", "YUV444PS", "dehalo", False),
    ("accept_rgb_dehalo", "RGBS", "dehalo", False),
    ("accept_yuv444_chroma", "YUV444PS", "chroma", True),
]


@pytest.mark.parametrize("name,fmt,model_key,flexible", ACCEPT_CASES,
                         ids=[c[0] for c in ACCEPT_CASES])
def test_clip_format_is_accepted(plugin, shipped_models, name, fmt, model_key, flexible):
    _expect_accept(name, fmt, shipped_models[model_key], flexible=flexible)


# ---------------------------------------------------------------------------
# P1-1: subsampled planes
# ---------------------------------------------------------------------------

SUBSAMPLED_CASES = [
    ("dehalo", "YUV420P8"),
    ("dehalo", "YUV422P8"),
    ("chroma", "YUV420P8"),
    ("chroma", "YUV422P8"),
]


@pytest.mark.parametrize("model_key,fmt", SUBSAMPLED_CASES,
                         ids=[f"{m}-{f}" for m, f in SUBSAMPLED_CASES])
def test_p1_01_subsampled_clip_is_rejected(plugin, shipped_models, model_key, fmt):
    """Chroma planes are half size; packing them with the luma dimensions reads
    past the plane."""
    _expect_reject(f"reject_{fmt.lower()}_{model_key}", fmt, shipped_models[model_key], "subsampled")


# ---------------------------------------------------------------------------
# P1-1: plane count vs model input channels
# ---------------------------------------------------------------------------

PLANE_COUNT_CASES = [
    ("dehalo", "GRAY8"),      # 1 plane into a 3-channel model
    ("chroma", "GRAY8"),      # 1 plane into the 3-channel chroma model
    ("luma", "YUV444PS"),     # 3 planes into the 1-channel luma model
]


@pytest.mark.parametrize("model_key,fmt", PLANE_COUNT_CASES,
                         ids=[f"{m}-{f}" for m, f in PLANE_COUNT_CASES])
def test_p1_01_plane_count_mismatch_is_rejected(plugin, shipped_models, model_key, fmt):
    _expect_reject(f"reject_{fmt.lower()}_{model_key}", fmt, shipped_models[model_key], "plane(s)")


def test_p1_01_more_channels_than_planes_is_rejected(plugin, fixtures):
    """A model wanting more channels than a clip can provide is refused at
    creation, before any device buffers are sized from it."""
    model = fixtures.get("conv_c4m4").model  # four input channels
    _expect_reject("reject_four_channel_model", "YUV444PS", model, "input channels")


# ---------------------------------------------------------------------------
# Chroma model: flexible output and chroma-plane values
# ---------------------------------------------------------------------------

CHROMA_PLANES_PROBE = """
import ctypes
import json
import os

import numpy as np
import onnxruntime as ort
import vapoursynth as vs

MARK = "__VS_RESULT__"
model = os.environ["VS_MODEL"]
size = int(os.environ.get("VS_SIZE", "64"))
seed = int(os.environ.get("VS_SEED", "1234"))

core = vs.core
core.max_cache_size = 256
rng = np.random.RandomState(seed)
data = rng.uniform(0.0, 1.0, (3, size, size)).astype(np.float32)

base = core.std.BlankClip(width=size, height=size, format=vs.YUV444PS, length=1)


def fill(n, f):
    nf = f.copy()
    for p in range(3):
        ctypes.memmove(nf.get_write_ptr(p), data[p].tobytes(), data[p].nbytes)
    return nf


clip = base.std.ModifyFrame(base, fill)
# Same call vsscale's R8F64_Chroma makes: fp16 compute + flexible outputs.
out = core.hip.Model(clip, model, fp16=1, flexible_output_prop="MlrtFlexible")
num_planes = int(out["num_planes"])
frame = out["clip"].get_frame(0)


def plane32(f, p=0):
    stride = f.get_stride(p)
    w, h = f.width, f.height
    ptr = f.get_read_ptr(p)
    addr = ptr.value if hasattr(ptr, "value") else int(ptr)
    raw = ctypes.string_at(addr, stride * h)
    arr = np.frombuffer(raw, np.float32, count=(stride // 4) * h).reshape(h, stride // 4)
    return arr[:, :w].copy()


planes = [plane32(frame.props["MlrtFlexible%d" % i]) for i in range(num_planes)]
main = plane32(frame)

# Independent reference: ONNX Runtime on the CPU with the original fp32 model.
options = ort.SessionOptions()
options.log_severity_level = 3
cpu = ort.InferenceSession(model, options, providers=["CPUExecutionProvider"])
expected = cpu.run(None, {"input": np.stack(data)[None]})[0]

result = {
    "num_planes": num_planes,
    "reference_channels": int(expected.shape[1]),
    "size": size,
    "finite": [bool(np.isfinite(p).all()) for p in planes],
    "maxdiff": [float(np.abs(planes[i] - expected[0, i]).max()) for i in range(num_planes)],
    "mean": [float(p.mean()) for p in planes],
    "reference_mean": [float(expected[0, i].mean()) for i in range(expected.shape[1])],
    "main_matches_plane0": float(np.abs(main - planes[0]).max()) if planes else None,
}
print(MARK, json.dumps(result))
"""


@pytest.fixture(scope="module")
def chroma_probe(plugin, shipped_models) -> dict:
    """One chroma run, shared by the flexible-output and accuracy checks."""
    pytest.importorskip("onnxruntime", reason="the chroma reference runs on the ORT CPU EP")
    run = vsrun.run_script("chroma_planes", CHROMA_PLANES_PROBE,
                           env={"VS_MODEL": str(shipped_models["chroma"])})
    return run.require_ok("chroma model probe")


def test_chroma_model_outputs_two_planes(chroma_probe):
    """The chroma model emits U and V; the flexible protocol must publish both."""
    assert chroma_probe["reference_channels"] == 2, "fixture model is expected to output U and V"
    assert chroma_probe["num_planes"] == chroma_probe["reference_channels"]


def test_chroma_planes_are_finite(chroma_probe):
    assert all(chroma_probe["finite"]), f"non-finite chroma plane: {chroma_probe['finite']}"


@pytest.mark.parametrize("plane", [0, 1], ids=["U", "V"])
def test_chroma_planes_match_cpu_reference(chroma_probe, plane):
    """Both chroma planes must match an independent fp32 CPU run of the same
    model. This is what catches a channel/layout mix-up: a swapped or
    de-interleaved plane is off by far more than the fp16 tolerance."""
    diff = chroma_probe["maxdiff"][plane]
    ref_mean = chroma_probe["reference_mean"][plane]
    assert diff < CHROMA_TOL, (
        f"chroma plane {plane} maxdiff {diff:.6f} >= {CHROMA_TOL}; "
        f"got mean {chroma_probe['mean'][plane]:.5f} vs reference {ref_mean:.5f}"
    )


def test_chroma_planes_are_distinct(chroma_probe):
    """U and V are separate outputs; identical planes would mean one was lost."""
    assert chroma_probe["mean"][0] != pytest.approx(chroma_probe["mean"][1], abs=1e-6)


def test_chroma_main_frame_matches_flexible_plane0(chroma_probe):
    """The primary frame plane 0 and MlrtFlexible0 must be the same data."""
    assert chroma_probe["main_matches_plane0"] == 0.0
