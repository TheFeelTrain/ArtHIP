#!/usr/bin/env python3
"""Intermittent-corruption hunter for the HIP kernels.

The shared WMMA convolutions once produced a rare, per-process output corruption
(see ``NOTES.md``).  Root cause, 2026-09-13: ``winograd_conv`` wrote its fp16
input strip to shared memory and then hit the top-of-loop ``__syncthreads()``
with those LDS stores still in flight, so another wave could pass the barrier
and read stale shared memory.  ``S_BARRIER`` does not drain LDS, and LLVM only
emitted the required ``s_waitcnt lgkmcnt(0)`` on the entry path, not on the
loop-back edge where the strip prefetch's stores arrive.  The fix is the
explicit ``lds_barrier()`` helper in ``src/common/hip_kernels.h``.

This tool is how the window was found and how a fix is graded.  The defect is
environment-dependent -- the same binary and input produced 0 % or ~85 %
corrupt frames depending on the host path and machine state -- so a single run
proves nothing and a plain pass/fail gate cannot see it:

* every sample is one **fresh child process** running **one** frame, because the
  defect shows up in the first frame of a process,
* the reference is an ONNX Runtime **CPU** run of the same model and input,
  cached under a hash of both, so a sample can never be compared against a
  stale reference,
* a sample is reported corrupt when the maximum absolute difference is above
  ``--tolerance`` *or* the output is not finite,
* corrupt samples are saved to ``tests/.cache/corrupt/`` for offline analysis,
* when comparing two builds, interleave them (``--ab A.so B.so``): the rate
  drifts with machine state, so sequential runs are not comparable.

Two routes, because the two host paths differ:

* ``ep``     -- the ONNX Runtime execution provider, driven directly.
* ``plugin`` -- the VapourSynth plugin, which is the only route that exercises
  the multi-channel models (the EP claims single-activation-input graphs only).

Usage
-----
    python tests/tools/corruption_hunt.py                       # luma1920, 50 samples
    python tests/tools/corruption_hunt.py --rounds 100
    python tests/tools/corruption_hunt.py --cases chroma64 chroma512
    python tests/tools/corruption_hunt.py --ab buildA.so buildB.so --rounds 40
    python tests/tools/corruption_hunt.py --list

Exit status is 1 when at least one corrupt sample was found, so it can be used
as a gate once the rate is known to be non-zero in the current environment.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent  # tests/tools
TESTS_DIR = HERE.parent  # tests/
PROJECT_ROOT = TESTS_DIR.parent  # repository root
FIXTURES_DIR = TESTS_DIR / "fixtures"
MODELS_DIR = FIXTURES_DIR / "models"
IMAGES_DIR = FIXTURES_DIR / "images"
CACHE_DIR = TESTS_DIR / ".cache"  # generated, git-ignored
CORRUPT_DIR = CACHE_DIR / "corrupt"
MARK = "__ARTHIP_HUNT__"

MODELS = {
    "luma": MODELS_DIR / "ArtCNN_R8F64_fp16.onnx",
    "chroma": MODELS_DIR / "ArtCNN_R8F64_Chroma.onnx",
    "dehalo": MODELS_DIR / "ArtCNN_R8F64_YCbCr_DEHALO.onnx",
}

#: case name -> (route, model key, width, height).  ``ep`` cases drive the ONNX
#: Runtime provider; ``plugin`` cases drive the VapourSynth plugin, which is the
#: only route that can exercise the multi-channel models.  The photo cases use
#: the real test image at its native size; the luma model doubles its input, so
#: a plugin luma case reports the output at 2x.
CASES: dict[str, tuple[str, str, int, int]] = {
    "luma64": ("ep", "luma", 64, 64),
    "luma512": ("ep", "luma", 512, 512),
    "luma1920": ("ep", "luma", 1920, 1080),
    "chroma64": ("plugin", "chroma", 64, 64),
    "chroma512": ("plugin", "chroma", 512, 512),
    "luma1080": ("plugin", "luma", 1920, 1080),
}

HIP_PROVIDER_SO = Path(
    os.environ.get("HIP_PROVIDER_SO",
                   str(PROJECT_ROOT / "src/onnxruntime-hip/build/libonnxruntime_providers_hip.so"))
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def make_input(model_key: str, w: int, h: int) -> np.ndarray:
    """A deterministic planar (C, H, W) float32 input for the case.

    Real content is used where a matching test image exists (it exercises the
    values the production filter sees) and seeded noise otherwise.
    """
    channels = 1 if model_key == "luma" else 3
    img_path = IMAGES_DIR / f"test_{w}.png"
    if img_path.exists():
        import cv2

        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise SystemExit(f"could not read {img_path}")
        base = np.clip(img.astype(np.float32) / 255.0, 0.0, 1.0)
        if base.shape != (h, w):
            base = cv2.resize(base, (w, h), interpolation=cv2.INTER_AREA)
        if channels == 1:
            x = base[None]
        else:
            rng = np.random.RandomState(1234)
            x = np.empty((channels, h, w), np.float32)
            x[0] = base
            for c in (1, 2):
                x[c] = rng.uniform(0.0, 1.0, (h, w))
    else:
        rng = np.random.RandomState(1234)
        x = rng.uniform(0.0, 1.0, (channels, h, w)).astype(np.float32)
    return x


def ref_paths(case: str) -> tuple[Path, Path]:
    return (CACHE_DIR / f"hunt_ref_{case}.npy", CACHE_DIR / f"hunt_ref_{case}.json")


def load_reference(case: str, inputs: list[np.ndarray]) -> np.ndarray | None:
    npy, sidecar = ref_paths(case)
    if not (npy.exists() and sidecar.exists()):
        return None
    try:
        meta = json.loads(sidecar.read_text())
    except (OSError, ValueError):
        return None
    digests = [hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest() for v in inputs]
    if meta.get("model_sha256") != sha256_file(MODELS[CASES[case][1]]):
        return None
    if meta.get("input_sha256") != digests:
        return None
    return np.load(npy)


def build_reference(case: str) -> None:
    """One ORT CPU run of the case model, cached under model+input hashes."""
    os.environ["MANGOHUD"] = "0"
    import onnxruntime as ort

    route, model_key, w, h = CASES[case]
    x = make_input(model_key, w, h)
    # The EP talks to the provider in the model's IO dtype; the plugin always
    # feeds the model fp16 (its engine computes in fp16) from a float clip.
    ep_input = x.astype(np.float16) if model_key == "luma" else x
    if load_reference(case, [ep_input]) is not None:
        return
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    cpu = ort.InferenceSession(str(MODELS[model_key]), opts, providers=["CPUExecutionProvider"])
    ref = np.asarray(cpu.run(None, {"input": ep_input[None]})[0])
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    npy, sidecar = ref_paths(case)
    np.save(npy, ref)
    sidecar.write_text(json.dumps({
        "model_sha256": sha256_file(MODELS[model_key]),
        "input_sha256": [hashlib.sha256(np.ascontiguousarray(ep_input).tobytes()).hexdigest()],
        "shape": list(ref.shape),
        "route": route,
    }, indent=2) + "\n")


# ---------------------------------------------------------------------------
# Child: one sample
# ---------------------------------------------------------------------------


def _plane32(frame, plane: int = 0) -> np.ndarray:
    """A VapourSynth float plane (32-bit or half) as a dense float32 array."""
    fmt = frame.format
    stride = frame.get_stride(plane)
    w, h = frame.width, frame.height
    ptr = frame.get_read_ptr(plane)
    addr = ptr.value if hasattr(ptr, "value") else int(ptr)
    raw = ctypes.string_at(addr, stride * h)
    if fmt.bytes_per_sample == 2:  # half-float formats (GRAYH)
        arr = np.frombuffer(raw, np.float16, count=(stride // 2) * h).reshape(h, stride // 2)
    else:
        arr = np.frombuffer(raw, np.float32, count=(stride // 4) * h).reshape(h, stride // 4)
    return arr[:, :w].astype(np.float32).copy()


def _child_plugin(case: str, tolerance: float, save_dir: Path | None) -> int:
    import vapoursynth as vs

    _, model_key, w, h = CASES[case]
    x = make_input(model_key, w, h)
    ref = load_reference(case, [x.astype(np.float16) if model_key == "luma" else x])
    if ref is None:
        raise SystemExit("no cached reference; run the parent once (it builds them)")

    core = vs.core
    core.max_cache_size = 256
    nch = x.shape[0]
    # The plugin feeds the model with as many planes as it has input channels:
    # the luma model wants one, the chroma/dehalo models three.
    fmt = vs.GRAYS if nch == 1 else vs.YUV444PS
    base = core.std.BlankClip(width=w, height=h, format=fmt, length=1)

    def fill(n, f):
        nf = f.copy()
        for p in range(nch):
            ctypes.memmove(nf.get_write_ptr(p), x[p].tobytes(), x[p].nbytes)
        return nf

    clip = base.std.ModifyFrame(base, fill)
    out = core.hip.Model(clip, str(MODELS[model_key]), fp16=1, flexible_output_prop="MlrtFlexible")
    num_planes = int(out["num_planes"]) if "num_planes" in out else 1
    frame = out["clip"].get_frame(0)
    if num_planes > 1:
        planes = [_plane32(frame.props[f"MlrtFlexible{i}"], 0) for i in range(num_planes)]
    else:
        planes = [_plane32(frame, 0)]

    ref = ref.astype(np.float32)
    finite = all(bool(np.isfinite(p).all()) for p in planes)
    diffs = [float(np.abs(p - ref[0, i]).max()) for i, p in enumerate(planes)]
    maxdiff = max(diffs) if finite else float("inf")
    bad = (not finite) or maxdiff > tolerance
    result = {"bad": bad, "finite": finite, "maxdiff": maxdiff, "planes": diffs}
    if bad and save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        tag = save_dir / f"{case}_{os.getpid()}"
        np.save(str(tag) + ".got.npy", np.stack(planes))
        np.save(str(tag) + ".in.npy", x)
        result["saved"] = str(tag)
    print(f"{MARK} {json.dumps(result)}", flush=True)
    return 0


def _child_ep(case: str, tolerance: float, save_dir: Path | None) -> int:
    import onnxruntime as ort
    from onnxruntime.capi import _pybind_state

    _pybind_state.set_default_logger_severity(3)
    _, model_key, w, h = CASES[case]
    x = make_input(model_key, w, h)
    ep_input = x.astype(np.float16) if model_key == "luma" else x
    ref = load_reference(case, [ep_input])
    if ref is None:
        raise SystemExit("no cached reference; run the parent once (it builds them)")

    if not HIP_PROVIDER_SO.exists():
        raise SystemExit(f"HIP provider library not found: {HIP_PROVIDER_SO}")
    _pybind_state.register_execution_provider_library("HIPExecutionProvider", str(HIP_PROVIDER_SO))
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    opts.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    sess = ort.InferenceSession(str(MODELS[model_key]), opts, providers=["HIPExecutionProvider"])
    got = np.asarray(sess.run(None, {"input": ep_input[None]})[0])
    ref = ref.astype(np.float32)
    gotf = got.astype(np.float32)
    finite = bool(np.isfinite(got).all())
    d = np.abs(gotf - ref)
    maxdiff = float(d.max()) if finite else float("inf")
    bad = (not finite) or maxdiff > tolerance
    result = {"bad": bad, "finite": finite, "maxdiff": maxdiff}
    if bad and save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        tag = save_dir / f"{case}_{os.getpid()}"
        np.save(str(tag) + ".got.npy", got)
        np.save(str(tag) + ".ref.npy", ref)
        np.save(str(tag) + ".in.npy", x)
        result["saved"] = str(tag)
    print(f"{MARK} {json.dumps(result)}", flush=True)
    return 0


def child(case: str, tolerance: float, save_dir: Path | None) -> int:
    os.environ["MANGOHUD"] = "0"
    route = CASES[case][0]
    return _child_plugin(case, tolerance, save_dir) if route == "plugin" else _child_ep(case, tolerance, save_dir)


# ---------------------------------------------------------------------------
# Parent
# ---------------------------------------------------------------------------


def _sample(case: str, tolerance: float, env: dict[str, str]) -> dict | None:
    cmd = [sys.executable, str(Path(__file__).resolve()), "--child",
           "--case", case, "--tolerance", str(tolerance)]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT), env=env)
    line = next((l for l in proc.stdout.splitlines() if l.startswith(MARK)), None)
    if line is None:
        detail = (proc.stdout + proc.stderr).strip().splitlines()[-1:] or ["no output"]
        if "fallback to CPU EP has been explicitly disabled" in detail[0]:
            return {"skip": "the HIP EP does not claim this graph"}
        return {"error": detail[0]}
    return json.loads(line[len(MARK):])


def _one(case: str, rounds: int, tolerance: float, env: dict[str, str], label: str) -> int:
    bad = 0
    for r in range(rounds):
        res = _sample(case, tolerance, env)
        if res is None or res.get("error"):
            print(f"  {label}{case} sample {r}: FAILED TO RUN: {(res or {}).get('error')}")
            bad += 1
        elif res.get("skip"):
            print(f"{case}: SKIP - {res['skip']}")
            return -1
        elif res["bad"]:
            bad += 1
            print(f"  {label}{case} sample {r}: CORRUPT maxdiff {res['maxdiff']:.5f} "
                  f"finite={res['finite']}"
                  + (f" saved={res.get('saved')}" if res.get("saved") else ""), flush=True)
    print(f"{label}{case}: {bad}/{rounds} corrupt samples")
    return bad


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cases", nargs="*", default=None,
                        help=f"cases to hunt (default luma1920); known: {', '.join(sorted(CASES))}")
    parser.add_argument("--rounds", type=int, default=50, help="fresh-process samples per case")
    parser.add_argument("--tolerance", type=float, default=None,
                        help="max absdiff above which a sample counts as corrupt "
                             "(default 0.003 for ep, 0.02 for plugin)")
    parser.add_argument("--ab", nargs=2, metavar=("SO_A", "SO_B"),
                        help="interleave two builds and grade both: the .so is loaded as the HIP "
                             "provider for ep cases and installed as the VapourSynth plugin for "
                             "plugin cases")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--case", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args.list:
        for name, (route, model_key, w, h) in sorted(CASES.items()):
            print(f"{name:12s} route={route:7s} model={model_key:7s} {w}x{h}")
        return 0

    if args.child:
        tol = args.tolerance if args.tolerance is not None else (
            0.02 if CASES[args.case][0] == "plugin" else 0.003)
        return child(args.case, tol, CORRUPT_DIR)

    cases = args.cases or ["luma1920"]
    unknown = [c for c in cases if c not in CASES]
    if unknown:
        print(f"unknown case(s): {', '.join(unknown)}; see --list", file=sys.stderr)
        return 2

    print("building CPU references (one ORT CPU run per case) ...", flush=True)
    for name in cases:
        build_reference(name)
        print(f"  {name} ready", flush=True)

    def default_tol(name: str) -> float:
        return 0.02 if CASES[name][0] == "plugin" else 0.003

    total_bad = 0
    if args.ab:
        # ONNX Runtime resolves the provider path against its own library
        # directory, so a relative path must be made absolute here.
        import shutil

        import vapoursynth

        a, b = (str(Path(p).resolve()) for p in args.ab)
        plugin_dest = Path(vapoursynth.__file__).resolve().parent / "plugins" / "libhip.so"
        a_bad = b_bad = 0
        for r in range(args.rounds):
            for so, label in ((a, "A "), (b, "B ")):
                env = dict(os.environ, HIP_PROVIDER_SO=so)
                for name in cases:
                    # The EP route takes the library from HIP_PROVIDER_SO; the
                    # plugin route can only be switched by replacing the
                    # installed plugin, which VapourSynth auto-loads.
                    if CASES[name][0] == "plugin":
                        shutil.copy2(so, plugin_dest)
                    tol = args.tolerance if args.tolerance is not None else default_tol(name)
                    res = _sample(name, tol, env)
                    if res is None or res.get("error"):
                        print(f"  pair {r}: {label}{name}: FAILED TO RUN: {(res or {}).get('error')}")
                        continue
                    if res.get("skip"):
                        continue
                    if res["bad"]:
                        if label == "A ":
                            a_bad += 1
                        else:
                            b_bad += 1
                        print(f"  pair {r}: {label}{name} CORRUPT maxdiff {res['maxdiff']:.5f}", flush=True)
        samples = args.rounds * len(cases)
        print(f"\nA {a}: {a_bad}/{samples} corrupt\nB {b}: {b_bad}/{samples} corrupt")
        return 1 if a_bad or b_bad else 0

    for name in cases:
        tol = args.tolerance if args.tolerance is not None else default_tol(name)
        bad = _one(name, args.rounds, tol, dict(os.environ), "")
        if bad > 0:
            total_bad += bad

    print()
    if total_bad:
        print(f"FOUND {total_bad} corrupt sample(s); outputs saved under {CORRUPT_DIR}")
        return 1
    print("no corrupt samples in this run (the defect is intermittent: absence is not proof)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
