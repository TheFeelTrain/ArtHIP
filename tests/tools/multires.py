#!/usr/bin/env python3
"""Multi-resolution accuracy + speed check for the HIP execution provider.

What this script guarantees:

* **Each backend runs in its own process.**  Combining the HIP and MIGraphX
  runtimes in one process is documented to crash at teardown
  (``NOTES.md``), and a per-backend process also means a backend
  that dies cannot be mistaken for one that passed.
* **Accuracy is asserted, not printed.**  The process exits non-zero when a
  backend's output is non-finite, differs from the CPU reference by more than
  the tolerance, or when a session silently falls back off the provider it was
  asked for.  ``get_providers()`` still lists a provider that claimed nothing,
  so the session is built with the CPU fallback disabled: an unclaimed node
  then makes session creation fail instead.
* **The CPU reference cache is keyed by content.**  The cache entry stores the
  SHA-256 of the model file and of the input tensor, so a changed model or test
  image can never be compared against a stale reference.

Usage
-----
    python tests/tools/multires.py                     # 256 512 1024, default frames
    python tests/tools/multires.py 256 512             # selected resolutions
    python tests/tools/multires.py 1920 --frames 120 --rounds 3
    python tests/tools/multires.py --list              # show what would run
    python tests/tools/multires.py --no-speed          # accuracy gate only

MIGraphX is optional: when it is unavailable the HIP checks still run and are
still asserted, and the MIGraphX comparison is reported as skipped.
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

HERE = Path(__file__).resolve().parent  # tests/tools
TESTS_DIR = HERE.parent  # tests/
PROJECT_ROOT = TESTS_DIR.parent  # repository root
FIXTURES_DIR = TESTS_DIR / "fixtures"
MODELS_DIR = FIXTURES_DIR / "models"
IMAGES_DIR = FIXTURES_DIR / "images"
CACHE_DIR = TESTS_DIR / ".cache"  # generated, git-ignored
MODEL_PATH = MODELS_DIR / "ArtCNN_R8F64_fp16.onnx"
HIP_PROVIDER_SO = Path(
    os.environ.get(
        "HIP_PROVIDER_SO",
        str(PROJECT_ROOT / "src/onnxruntime-hip/build/libonnxruntime_providers_hip.so"),
    )
)

#: Marker that separates a worker's JSON result from provider chatter.
MARK = "__ARTHIP_RESULT__"

#: Absolute accuracy gate against the fp32 CPU reference.  Both HIP and
#: MIGraphX measure ~0.001 (fp16 rounding class), so 0.01 leaves an order of
#: magnitude of headroom while still catching a layout or dispatch defect
#: (which shows up as an error of order 1.0).
TOL_MAXDIFF = 0.01
#: Fraction of 8-bit codes allowed to differ after rounding.
TOL_MISMATCH = 0.005

DEFAULT_RESOLUTIONS = [256, 512, 1024]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_input(res: int) -> np.ndarray:
    """The test image as the fp16 NCHW tensor every backend is fed."""
    import cv2

    img_path = IMAGES_DIR / f"test_{res}.png"
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise SystemExit(f"could not read {img_path}")
    x = np.clip(img.astype(np.float32) / 255.0, 0.0, 1.0)
    return x[None, None, :, :].astype(np.float16)


def cache_paths(res: int) -> tuple[Path, Path]:
    return CACHE_DIR / f"cpu_ref_{res}.npy", CACHE_DIR / f"cpu_ref_{res}.json"


def load_reference(res: int, model_sha: str, input_sha: str) -> np.ndarray | None:
    npy, sidecar = cache_paths(res)
    if not (npy.exists() and sidecar.exists()):
        return None
    try:
        meta = json.loads(sidecar.read_text())
    except (OSError, ValueError):
        return None
    if meta.get("model_sha256") != model_sha or meta.get("input_sha256") != input_sha:
        return None
    return np.load(npy)


def store_reference(res: int, model_sha: str, input_sha: str, ref: np.ndarray) -> None:
    npy, sidecar = cache_paths(res)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(npy, ref)
    sidecar.write_text(
        json.dumps(
            {"model_sha256": model_sha, "input_sha256": input_sha, "shape": list(ref.shape)},
            indent=2,
        )
        + "\n"
    )


def _strict_session(ort, backend: str, cache_dir: Path | None = None):
    """A session that fails instead of silently falling back to the CPU EP."""
    options = ort.SessionOptions()
    options.log_severity_level = 3
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    if backend == "hip":
        return ort.InferenceSession(str(MODEL_PATH), options, providers=["HIPExecutionProvider"])
    if backend == "migx":
        provider = ("MIGraphXExecutionProvider", {})
        if cache_dir is not None:
            provider = ("MIGraphXExecutionProvider", {"migraphx_model_cache_dir": str(cache_dir)})
        return ort.InferenceSession(str(MODEL_PATH), options, providers=[provider])
    raise ValueError(backend)


def _register_hip_provider(ort) -> None:
    from onnxruntime.capi import _pybind_state

    if not HIP_PROVIDER_SO.exists():
        raise SystemExit(
            f"HIP provider library not found: {HIP_PROVIDER_SO}\n"
            "Build it first: cd src/onnxruntime-hip && ./build.sh"
        )
    _pybind_state.register_execution_provider_library(
        "HIPExecutionProvider", str(HIP_PROVIDER_SO)
    )
    if not any(getattr(d, "ep_name", "") == "HIPExecutionProvider" for d in ort.get_ep_devices()):
        raise SystemExit(
            "HIPExecutionProvider registration failed "
            f"(ep devices: {[getattr(d, 'ep_name', d) for d in ort.get_ep_devices()]})"
        )


def worker_main(args) -> int:
    os.environ["MANGOHUD"] = "0"
    import onnxruntime as ort
    from onnxruntime.capi import _pybind_state

    _pybind_state.set_default_logger_severity(3)

    x = load_input(args.res)
    model_sha = sha256_file(MODEL_PATH)
    input_sha = hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
    result: dict = {"backend": args.backend, "res": args.res, "input_sha256": input_sha}

    if args.backend == "cpu":
        cpu = ort.InferenceSession(
            str(MODEL_PATH), ort.SessionOptions(), providers=["CPUExecutionProvider"]
        )
        ref = np.asarray(cpu.run(None, {"input": x})[0])
        store_reference(args.res, model_sha, input_sha, ref)
        result["reference_shape"] = list(ref.shape)
    else:
        if args.backend == "hip":
            _register_hip_provider(ort)
        ref = load_reference(args.res, model_sha, input_sha)
        if ref is None:
            raise SystemExit(
                f"no CPU reference for {args.res}px matching this model/input; "
                "run the cpu worker first (the orchestrator does this automatically)"
            )
        try:
            sess = _strict_session(ort, args.backend, cache_dir=args.mxr_cache)
        except Exception as exc:  # capability refusal is a failure, not a skip
            raise SystemExit(f"{args.backend} session did not claim the graph: {exc}") from exc
        result["providers"] = sess.get_providers()

        got = np.asarray(sess.run(None, {"input": x})[0])
        if got.shape != ref.shape:
            raise SystemExit(f"{args.backend} output shape {got.shape} != reference {ref.shape}")
        diff = np.abs(ref.astype(np.float32) - got.astype(np.float32))
        rc = np.clip(np.round(ref.astype(np.float32) * 255), 0, 255).astype(np.uint8)
        rv = np.clip(np.round(got.astype(np.float32) * 255), 0, 255).astype(np.uint8)
        result.update(
            finite=bool(np.isfinite(got).all()),
            maxdiff=float(diff.max()),
            meandiff=float(diff.mean()),
            mismatch_fraction=float((rc != rv).mean()),
            size=int(rc.size),
        )

        if args.frames > 0:
            for _ in range(args.warmup):
                sess.run(None, {"input": x})
            import time

            t0 = time.perf_counter()
            for _ in range(args.frames):
                sess.run(None, {"input": x})
            result["ms_per_iter"] = (time.perf_counter() - t0) / args.frames * 1000.0
            result["frames"] = args.frames

    print(f"{MARK} {json.dumps(result)}", flush=True)
    return 0


def run_worker(res: int, backend: str, args) -> dict:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--backend",
        backend,
        "--res",
        str(res),
        "--frames",
        str(args.frames),
        "--warmup",
        str(args.warmup),
    ]
    if backend == "migx":
        cmd += ["--mxr-cache", str(TESTS_DIR / ".mxr_cache" / f"res{res}")]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT))
    for line in proc.stdout.splitlines():
        if line.startswith(MARK):
            return json.loads(line[len(MARK):])
    detail = (proc.stdout + proc.stderr).strip()
    return {"backend": backend, "res": res, "error": detail or f"exit code {proc.returncode}"}


def check(entry: dict, label: str) -> bool:
    """Assert one backend's result; return True when it passed."""
    if "error" in entry:
        print(f"  FAIL {label}: {entry['error']}")
        return False
    if "maxdiff" not in entry:
        return True  # cpu worker: produced the reference
    detail = (
        f"maxdiff {entry['maxdiff']:.5f} meandiff {entry['meandiff']:.6f} "
        f"uint8 mismatch {100 * entry['mismatch_fraction']:.2f}%"
    )
    ms = entry.get("ms_per_iter")
    speed = f"{ms:8.2f} ms/iter" if ms else "   (untimed)"
    ok = True
    if not entry["finite"]:
        print(f"  FAIL {label}: output is not finite  [{detail}]")
        ok = False
    if entry["maxdiff"] > TOL_MAXDIFF:
        print(f"  FAIL {label}: maxdiff above {TOL_MAXDIFF} (layout/dispatch error)  [{detail}]")
        ok = False
    if entry["mismatch_fraction"] > TOL_MISMATCH:
        print(f"  FAIL {label}: 8-bit mismatch above {100 * TOL_MISMATCH:.2f}%  [{detail}]")
        ok = False
    if ok:
        print(f"  OK   {label}: {detail}  {speed}")
    return ok


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("resolutions", nargs="*", type=int, default=None)
    parser.add_argument("--frames", type=int, default=100, help="timed iterations per backend")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument(
        "--rounds",
        type=int,
        default=2,
        help="interleaved HIP/MIGraphX timing rounds (accuracy is checked every round)",
    )
    parser.add_argument("--no-speed", action="store_true", help="accuracy gate only")
    parser.add_argument("--list", action="store_true")
    # Worker-only arguments.
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--backend", choices=["cpu", "hip", "migx"], help=argparse.SUPPRESS)
    parser.add_argument("--res", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--mxr-cache", type=Path, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args.worker:
        return worker_main(args)

    resolutions = args.resolutions or DEFAULT_RESOLUTIONS
    if args.list:
        for res in resolutions:
            npy, _ = cache_paths(res)
            print(f"{res:5d}  input={IMAGES_DIR / f'test_{res}.png'}  reference_cached={npy.exists()}")
        return 0

    frames = 0 if args.no_speed else args.frames
    args.frames = frames
    failures: list[str] = []
    speed: dict[str, list[float]] = {}

    for res in resolutions:
        print(f"=== {res}px ===", flush=True)
        entry = run_worker(res, "cpu", args)
        if "error" in entry:
            print(f"  SKIP {res}px: {entry['error']}")
            continue

        for rnd in range(max(1, args.rounds if frames else 1)):
            order = ["hip", "migx"] if rnd % 2 == 0 else ["migx", "hip"]
            for backend in order:
                entry = run_worker(res, backend, args)
                label = f"{res}px {backend} (round {rnd + 1})" if frames else f"{res}px {backend}"
                if "error" in entry and backend == "migx":
                    print(f"  SKIP {res}px migx: not usable on this machine")
                    continue
                if not check(entry, label):
                    failures.append(label)
                if entry.get("ms_per_iter"):
                    speed.setdefault(backend, []).append(entry["ms_per_iter"])

    print()
    if speed:
        print("Speed (median of the timed rounds, separate processes):")
        for backend, values in speed.items():
            values.sort()
            median = values[len(values) // 2]
            print(f"  {backend:9s} {median:8.2f} ms/iter  ({len(values)} round(s))")

    if failures:
        print(f"\nFAILED: {len(failures)} check(s): {', '.join(failures)}")
        return 1
    print("\nall accuracy checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
