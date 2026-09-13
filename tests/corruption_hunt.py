#!/usr/bin/env python3
"""Intermittent-corruption hunter for the HIP execution provider.

There is a rare, timing-dependent output corruption in the shared WMMA
convolutions (see ``NOTES.md``, "intermittent corruption").  It is
environment-dependent: the same binary and input produce 0 % or up to ~40 %
corrupt frames depending on machine state, so a single run proves nothing and a
plain pass/fail gate cannot see it.  This script hunts for it:

* every sample is one **fresh child process** running **one** frame, because the
  defect shows up in the first frame of a process,
* the reference is an ONNX Runtime **CPU** run of the same model and input,
  cached under a hash of both, so a sample can never be compared against a
  stale reference,
* a sample is reported corrupt when the maximum absolute difference is above
  ``--tolerance`` (default 0.003) *or* the output is not finite,
* corrupt samples are saved to ``tests/.corrupt/`` for offline analysis.

Usage
-----
    python tests/corruption_hunt.py                 # luma 1920, 50 samples
    python tests/corruption_hunt.py --rounds 100
    python tests/corruption_hunt.py --cases luma64 chroma64 dehalo64
    python tests/corruption_hunt.py --list

Exit status is 1 when at least one corrupt sample was found, so it can be used
as a gate once the rate is known to be non-zero in the current environment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
CORRUPT_DIR = HERE / ".corrupt"
MARK = "__ARTHIP_HUNT__"

MODELS = {
    "luma": HERE / "ArtCNN_R8F64_fp16.onnx",
    "chroma": HERE / "ArtCNN_R8F64_Chroma.onnx",
    "dehalo": HERE / "ArtCNN_R8F64_YCbCr_DEHALO.onnx",
}

#: (case name, model key, size in px).  A case name is ``<model><size>``.
CASES: dict[str, tuple[str, int]] = {
    "luma64": ("luma", 64),
    "luma512": ("luma", 512),
    "luma1920": ("luma", 1920),
}

#: Multi-channel models are exercised through the VapourSynth plugin, not here:
#: the execution provider deliberately claims only single-activation-input
#: regions, so it refuses the chroma/dehalo graphs when the CPU fallback is
#: disabled (see REVIEW.md P2-17 and ``tests/correctness/test_ep.py``).  The
#: plugin route is ``tmp/p2probe/find.py`` (chroma, 64x64), described in
#: ``NOTES.md``.


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


def make_input(model_key: str, size: int) -> np.ndarray:
    """A deterministic NCHW input for the case, in the model's IO dtype.

    The luma model takes one plane; the chroma/dehalo models take three.  Real
    content is used where a matching test image exists (it exercises the same
    values the production filter sees) and seeded noise otherwise.
    """
    channels = 1 if model_key == "luma" else 3
    img_path = HERE / f"test_{size}.png"
    if img_path.exists():
        import cv2

        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise SystemExit(f"could not read {img_path}")
        base = np.clip(img.astype(np.float32) / 255.0, 0.0, 1.0)
        if channels == 1:
            x = base[None, None]
        else:
            rng = np.random.RandomState(1234)
            x = np.empty((1, 3, size, size), np.float32)
            x[0, 0] = base
            for c in (1, 2):
                x[0, c] = rng.uniform(0.0, 1.0, (size, size))
    else:
        rng = np.random.RandomState(1234)
        x = rng.uniform(0.0, 1.0, (1, channels, size, size)).astype(np.float32)
    # The shipped luma model is fp16 IO; the chroma and dehalo models are fp32.
    return x.astype(np.float16 if model_key == "luma" else np.float32)


def ref_paths(model_key: str, size: int) -> tuple[Path, Path]:
    return (HERE / f".hunt_ref_{model_key}{size}.npy",
            HERE / f".hunt_ref_{model_key}{size}.json")


def load_reference(model_key: str, size: int, model_sha: str, input_sha: str):
    npy, sidecar = ref_paths(model_key, size)
    if not (npy.exists() and sidecar.exists()):
        return None
    try:
        meta = json.loads(sidecar.read_text())
    except (OSError, ValueError):
        return None
    if meta.get("model_sha256") != model_sha or meta.get("input_sha256") != input_sha:
        return None
    return np.load(npy)


def child(model_key: str, size: int, tolerance: float, save_dir: Path | None) -> int:
    os.environ["MANGOHUD"] = "0"
    import onnxruntime as ort
    from onnxruntime.capi import _pybind_state

    _pybind_state.set_default_logger_severity(3)
    model = MODELS[model_key]
    x = make_input(model_key, size)
    model_sha = sha256_file(model)
    input_sha = hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
    ref = load_reference(model_key, size, model_sha, input_sha)
    if ref is None:
        raise SystemExit("no cached reference; run the parent once (it builds them)")

    if not HIP_PROVIDER_SO.exists():
        raise SystemExit(f"HIP provider library not found: {HIP_PROVIDER_SO}")
    _pybind_state.register_execution_provider_library("HIPExecutionProvider", str(HIP_PROVIDER_SO))
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    opts.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    sess = ort.InferenceSession(str(model), opts, providers=["HIPExecutionProvider"])
    got = np.asarray(sess.run(None, {"input": x})[0])
    ref = ref.astype(np.float32)
    gotf = got.astype(np.float32)
    finite = bool(np.isfinite(got).all())
    d = np.abs(gotf - ref)
    maxdiff = float(d.max()) if finite else float("inf")
    bad = (not finite) or maxdiff > tolerance
    global_max = float(np.abs(ref).max()) or 1.0
    result = {"bad": bad, "finite": finite, "maxdiff": maxdiff, "rel": maxdiff / global_max}
    if bad and save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        tag = save_dir / f"{model_key}{size}_{os.getpid()}"
        np.save(tag.with_suffix(".got.npy"), got)
        np.save(tag.with_suffix(".ref.npy"), ref)
        np.save(tag.with_suffix(".in.npy"), x)
        result["saved"] = str(tag)
    print(f"{MARK} {json.dumps(result)}", flush=True)
    return 0


def build_reference(model_key: str, size: int) -> None:
    os.environ["MANGOHUD"] = "0"
    import onnxruntime as ort

    model = MODELS[model_key]
    x = make_input(model_key, size)
    model_sha = sha256_file(model)
    input_sha = hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
    if load_reference(model_key, size, model_sha, input_sha) is not None:
        return
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    cpu = ort.InferenceSession(str(model), opts, providers=["CPUExecutionProvider"])
    ref = np.asarray(cpu.run(None, {"input": x})[0])
    npy, sidecar = ref_paths(model_key, size)
    np.save(npy, ref)
    sidecar.write_text(json.dumps(
        {"model_sha256": model_sha, "input_sha256": input_sha, "shape": list(ref.shape)}, indent=2) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cases", nargs="*", default=["luma1920"],
                        help=f"cases to hunt (default luma1920); known: {', '.join(sorted(CASES))}")
    parser.add_argument("--rounds", type=int, default=50, help="fresh-process samples per case")
    parser.add_argument("--tolerance", type=float, default=0.003,
                        help="max absdiff above which a sample counts as corrupt")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--model", help=argparse.SUPPRESS)
    parser.add_argument("--size", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args.list:
        for name, (model_key, size) in sorted(CASES.items()):
            print(f"{name:12s} model={model_key:7s} size={size}")
        return 0

    if args.child:
        return child(args.model, args.size, args.tolerance, CORRUPT_DIR)

    unknown = [c for c in args.cases if c not in CASES]
    if unknown:
        print(f"unknown case(s): {', '.join(unknown)}; see --list", file=sys.stderr)
        return 2

    print("building CPU references (one ORT CPU run per case) ...", flush=True)
    for name in args.cases:
        model_key, size = CASES[name]
        build_reference(model_key, size)
        print(f"  {name} ready", flush=True)

    total_bad = 0
    for name in args.cases:
        model_key, size = CASES[name]
        bad = 0
        for r in range(args.rounds):
            cmd = [sys.executable, str(Path(__file__).resolve()), "--child",
                   "--model", model_key, "--size", str(size),
                   "--tolerance", str(args.tolerance)]
            proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT))
            line = next((l for l in proc.stdout.splitlines() if l.startswith(MARK)), None)
            if line is None:
                detail = (proc.stdout + proc.stderr).strip().splitlines()[-1:] or ["no output"]
                if "fallback to CPU EP has been explicitly disabled" in detail[0]:
                    print(f"{name}: SKIP - the HIP EP does not claim this graph")
                    bad = -1
                    break
                print(f"  {name} sample {r}: FAILED TO RUN: {detail[0]}")
                bad += 1
                continue
            res = json.loads(line[len(MARK):])
            if res["bad"]:
                bad += 1
                print(f"  {name} sample {r}: CORRUPT maxdiff {res['maxdiff']:.5f} "
                      f"(rel {res['rel']:.4f}) finite={res['finite']}"
                      + (f" saved={res.get('saved')}" if res.get("saved") else ""))
        if bad < 0:
            continue
        total_bad += bad
        print(f"{name}: {bad}/{args.rounds} corrupt samples")

    print()
    if total_bad:
        print(f"FOUND {total_bad} corrupt sample(s); outputs saved under {CORRUPT_DIR}")
        return 1
    print("no corrupt samples in this run (the defect is intermittent: absence is not proof)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
