"""Build helpers for the correctness suite.

Every artifact is rebuilt when a source it depends on is newer than the
artifact, so a normal run always tests the current tree. ``force=True``
(``--rebuild``) rebuilds unconditionally, and ``no_build=True`` (``--no-build``)
trusts whatever already exists -- useful for repeat runs and for CI that has
already built everything.

Artifacts that cannot be built raise :class:`BuildError`; the fixtures turn
that into a skip with the underlying reason rather than a confusing failure.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
from pathlib import Path

from . import paths


class BuildError(RuntimeError):
    """An artifact could not be built (or is missing with --no-build)."""


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_stale(artifact: Path, deps) -> bool:
    """True when the artifact is missing or older than any dependency."""
    if not artifact.exists():
        return True
    try:
        built = artifact.stat().st_mtime
    except OSError:
        return True
    return any(dep.exists() and dep.stat().st_mtime > built for dep in deps)


def _glob(*patterns: str) -> list[Path]:
    found: list[Path] = []
    for pattern in patterns:
        found.extend(paths.REPO_ROOT.glob(pattern))
    return found


def run_command(cmd: list[str], cwd: Path | None = None, what: str = "command") -> None:
    """Run a build command, raising BuildError with the output tail on failure."""
    result = subprocess.run(cmd, cwd=str(cwd) if cwd else None, capture_output=True, text=True)
    if result.returncode != 0:
        tail = "\n".join((result.stdout + result.stderr).strip().splitlines()[-25:])
        raise BuildError(f"{what} failed (exit {result.returncode}):\n{tail}")


def gpu_available() -> tuple[bool, str]:
    """Cheap precondition check; the real probe happens once the harness loads."""
    if shutil.which("hipcc") is None:
        return False, "hipcc is not on PATH (ROCm toolchain required)"
    if not Path("/dev/kfd").exists():
        return False, "/dev/kfd is missing (no AMD compute device on this host)"
    return True, ""


def _ensure(artifact: Path, deps, build, *, what: str, force: bool, no_build: bool) -> Path:
    rebuild = force or is_stale(artifact, deps)
    if not rebuild:
        return artifact
    if no_build:
        raise BuildError(
            f"{what}: {artifact} is missing or stale and --no-build was requested"
        )
    build()
    if not artifact.exists():
        raise BuildError(f"{what}: {artifact} was not produced by the build")
    return artifact


# ---------------------------------------------------------------------------
# Engine driver
# ---------------------------------------------------------------------------


def build_engine_harness(*, force: bool = False, no_build: bool = False) -> Path:
    """Compile the ctypes driver together with the real engine sources."""
    harness_cc = Path(__file__).resolve().parent / "engine_harness.cc"
    deps = [
        harness_cc,
        paths.ENGINE_SOURCES / "hip_engine.cc",
        paths.ENGINE_SOURCES / "hip_engine.h",
        paths.HIP_SOURCES / "hip_kernels.h",
        paths.COMMON_SOURCES / "onnx_utils.cpp",
        paths.COMMON_SOURCES / "onnx_utils.h",
        paths.COMMON_SOURCES / "convert_float_to_float16.cpp",
        paths.COMMON_SOURCES / "convert_float_to_float16.h",
    ]

    def build() -> None:
        paths.BUILD_DIR.mkdir(parents=True, exist_ok=True)
        print(f"[correctness] building engine driver -> {paths.ENGINE_HARNESS_SO.name}")
        run_command(
            [
                "hipcc", "--offload-arch=gfx1100", "-std=c++17", "-O2", "-g",
                "-fPIC", "-shared", "-w", "-DONNX_ML", "-DONNX_NAMESPACE=onnx",
                f"-I{paths.ENGINE_SOURCES}",
                f"-I{paths.HIP_SOURCES}",
                "-I/usr/include",
                str(harness_cc),
                str(paths.ENGINE_SOURCES / "hip_engine.cc"),
                str(paths.COMMON_SOURCES / "onnx_utils.cpp"),
                str(paths.COMMON_SOURCES / "convert_float_to_float16.cpp"),
                "-o", str(paths.ENGINE_HARNESS_SO),
                "-lonnx", "-lonnx_proto", "-lprotobuf", "-lpthread", "-ldl",
            ],
            cwd=paths.REPO_ROOT,
            what="engine driver build",
        )

    return _ensure(paths.ENGINE_HARNESS_SO, deps, build,
                   what="engine driver", force=force, no_build=no_build)


# ---------------------------------------------------------------------------
# ONNX Runtime execution provider
# ---------------------------------------------------------------------------


def build_ep(*, force: bool = False, no_build: bool = False) -> Path:
    """Build libonnxruntime_providers_hip.so via the provider's own script."""
    deps = _glob("src/hip/*.cc", "src/hip/*.h")

    def build() -> None:
        print(f"[correctness] building HIP execution provider -> {paths.EP_SO.name}")
        run_command(["bash", str(paths.EP_BUILD_SCRIPT)], cwd=paths.REPO_ROOT,
                    what="HIP execution provider build")

    return _ensure(paths.EP_SO, deps, build,
                   what="HIP execution provider", force=force, no_build=no_build)


# ---------------------------------------------------------------------------
# VapourSynth plugin
# ---------------------------------------------------------------------------


def build_plugin(*, force: bool = False, no_build: bool = False) -> Path:
    """Build libhip.so via the plugin's own script."""
    deps = (
        _glob("src/vapoursynth/hip/*.cpp", "src/vapoursynth/hip/*.h")
        + _glob("src/vapoursynth/common/*.cpp", "src/vapoursynth/common/*.h")
        + [paths.HIP_SOURCES / "hip_kernels.h"]
    )

    def build() -> None:
        print(f"[correctness] building VapourSynth plugin -> {paths.PLUGIN_SO.name}")
        run_command(["bash", str(paths.PLUGIN_BUILD_SCRIPT)], cwd=paths.REPO_ROOT,
                    what="VapourSynth plugin build")

    return _ensure(paths.PLUGIN_SO, deps, build,
                   what="VapourSynth plugin", force=force, no_build=no_build)


def plugin_install_dir() -> Path:
    """Directory VapourSynth auto-loads plugins from."""
    import vapoursynth  # local import: only the plugin suite needs it

    return Path(vapoursynth.__file__).resolve().parent / "plugins"


def install_plugin(lib: Path, *, force: bool = False) -> tuple[Path, bool]:
    """Install the built plugin so VapourSynth loads THIS binary.

    VapourSynth auto-loads plugins by id, so an explicit ``LoadPlugin`` of a
    second copy fails ("already loaded") -- the installed file has to be the
    artifact under test. Returns ``(destination, changed)``.
    """
    dest = plugin_install_dir() / lib.name
    if dest.exists() and not force and sha256(dest) == sha256(lib):
        return dest, False
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(lib, dest)
    except OSError as exc:
        raise BuildError(
            f"could not install {lib} to {dest} ({exc}); "
            f"install it manually or pass --no-install-plugin"
        ) from exc
    if sha256(dest) != sha256(lib):
        raise BuildError(f"installed {dest} does not match the built plugin")
    return dest, True
