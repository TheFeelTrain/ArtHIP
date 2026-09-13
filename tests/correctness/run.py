#!/usr/bin/env python3
"""Run the ArtHIP correctness suite.

Each suite runs in its own process. The HIP runtime, the ONNX Runtime provider
and VapourSynth do not always tear down cleanly when combined in one process
(see NOTES.md), so process-per-suite keeps runs reliable.

Usage
-----
    python tests/correctness/run.py                  # every suite
    python tests/correctness/run.py engine ep        # selected suites
    python tests/correctness/run.py --list           # show suites
    python tests/correctness/run.py --rebuild        # force artifact rebuilds
    python tests/correctness/run.py engine -- -k conv -x   # extra pytest args

Anything after ``--`` is forwarded to pytest unchanged.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

SUITE_DIR = Path(__file__).resolve().parent
SUITES = {
    "engine": "test_engine.py",
    "ep": "test_ep.py",
    "plugin": "test_plugin.py",
}


def _split_pytest_args(argv: list[str]) -> tuple[list[str], list[str]]:
    if "--" in argv:
        idx = argv.index("--")
        return argv[:idx], argv[idx + 1 :]
    return argv, []


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Run the ArtHIP correctness suite (one process per suite).",
        epilog="Arguments after -- are passed to pytest.",
    )
    parser.add_argument("suites", nargs="*", metavar="SUITE",
                        help=f"suites to run (default: all); choose from {', '.join(SUITES)}")
    parser.add_argument("--list", action="store_true", help="list the suites and exit")
    parser.add_argument("--rebuild", action="store_true",
                        help="rebuild every test artifact before running")
    parser.add_argument("--no-build", action="store_true",
                        help="use existing artifacts without rebuilding")
    parser.add_argument("--no-install-plugin", action="store_true",
                        help="do not copy libhip.so into the VapourSynth plugin directory")
    parser.add_argument("--device", type=int, default=0, help="HIP device id (default: 0)")
    return parser


def main(argv: list[str] | None = None) -> int:
    main_argv, pytest_argv = _split_pytest_args(list(sys.argv[1:] if argv is None else argv))
    args = _parser().parse_args(main_argv)

    if args.list:
        for name, filename in SUITES.items():
            print(f"{name:8s} {filename}")
        return 0

    unknown = [s for s in args.suites if s not in SUITES]
    if unknown:
        print(f"unknown suite(s): {', '.join(unknown)}; choose from {', '.join(SUITES)}",
              file=sys.stderr)
        return 2
    selected = args.suites or list(SUITES)

    flags = [f"--device={args.device}"]
    if args.rebuild:
        flags.append("--rebuild")
    if args.no_build:
        flags.append("--no-build")
    if args.no_install_plugin:
        flags.append("--no-install-plugin")

    results: dict[str, int] = {}
    summaries: dict[str, str] = {}
    for name in selected:
        print(f"\n{'=' * 72}\n== ArtHIP correctness: {name}\n{'=' * 72}", flush=True)
        cmd = [sys.executable, "-m", "pytest", str(SUITE_DIR / SUITES[name]),
               "-v", "-ra", "-p", "no:cacheprovider", *flags, *pytest_argv]
        proc = subprocess.run(cmd, cwd=str(SUITE_DIR), capture_output=True, text=True)
        print(proc.stdout, end="")
        if proc.stderr.strip():
            print(proc.stderr, end="", file=sys.stderr)
        results[name] = proc.returncode
        summaries[name] = _summary_line(proc.stdout)

    print(f"\n{'=' * 72}\n== Summary\n{'=' * 72}")
    for name in selected:
        status = "PASS" if results[name] == 0 else "FAIL"
        detail = f"  ({summaries[name]})" if summaries[name] else ""
        print(f"  {status}  {name}{detail}")
    failed = [n for n, rc in results.items() if rc != 0]
    if failed:
        print(f"\n{len(failed)} suite(s) failed: {', '.join(failed)}")
        return 1
    print(f"\nall {len(selected)} suite(s) passed")
    return 0


def _summary_line(output: str) -> str:
    """The pytest outcome line, e.g. ``94 passed in 31.20s``."""
    for line in reversed(output.strip().splitlines()):
        stripped = line.strip().strip("=").strip()
        if stripped and ("passed" in stripped or "failed" in stripped or "error" in stripped):
            return stripped
    return ""


if __name__ == "__main__":
    sys.exit(main())
