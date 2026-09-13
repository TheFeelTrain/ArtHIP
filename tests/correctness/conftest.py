"""Shared fixtures and options for the ArtHIP correctness suite.

Suites are plain pytest modules grouped by the component they exercise:

========================  ==================================================
``test_kernel.py``       kernel memory-ordering invariants (source + ISA)
``test_engine.py``       the standalone ``HipEngine`` (real GPU kernels)
``test_ep.py``           the ONNX Runtime execution provider
``test_plugin.py``       the VapourSynth plugin, through ``vspipe``
========================  ==================================================

Run everything with ``python tests/correctness/run.py`` (one process per suite,
which keeps the HIP and ONNX Runtime lifetimes apart) or an individual suite
with ``python -m pytest tests/correctness/test_engine.py``.

Artifacts are rebuilt when a source they depend on is newer (see
``harness/build.py``); ``--rebuild`` forces it and ``--no-build`` disables it.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import pytest

# GPU measurements and the reference tooling both assume no overlay.
os.environ.setdefault("MANGOHUD", "0")

from harness import build, paths  # noqa: E402  (after the env setup above)
from harness.engine import EngineHarness  # noqa: E402
# NOTE: harness.fixtures, never harness.modelgen -- the latter imports the onnx
# Python package, which cannot coexist with the VapourSynth plugin load that the
# vsengine pytest plugin triggers. See harness/fixtures.py.
from harness.fixtures import FixtureError, Fixtures  # noqa: E402


def pytest_addoption(parser) -> None:
    group = parser.getgroup("arthip")
    group.addoption("--rebuild", action="store_true", default=False,
                    help="rebuild every test artifact before running")
    group.addoption("--no-build", action="store_true", default=False,
                    help="use existing artifacts without rebuilding")
    group.addoption("--no-install-plugin", action="store_true", default=False,
                    help="do not copy libhip.so into the VapourSynth plugin directory")
    group.addoption("--device", action="store", type=int, default=0,
                    help="HIP device id used by the tests (default: 0)")


@dataclass(frozen=True)
class BuildOptions:
    force: bool
    no_build: bool
    install_plugin: bool
    device: int


@pytest.fixture(scope="session")
def build_options(request) -> BuildOptions:
    get = request.config.getoption
    return BuildOptions(
        force=get("--rebuild"),
        no_build=get("--no-build"),
        install_plugin=not get("--no-install-plugin"),
        device=get("--device"),
    )


@pytest.fixture(scope="session")
def device(build_options: BuildOptions) -> int:
    """HIP device id the tests should use."""
    return build_options.device


@pytest.fixture(scope="session")
def kernel_asm(build_options: BuildOptions) -> Path:
    """gfx1100 assembly for ``src/common/hip_kernels.h`` (needs hipcc, no GPU).

    Used by the memory-ordering checks in ``test_kernel.py``; a machine without
    the ROCm toolchain skips them rather than failing.
    """
    if shutil.which("hipcc") is None:
        pytest.skip("hipcc is not on PATH (ROCm toolchain required)")
    try:
        return build.build_kernel_asm(force=build_options.force, no_build=build_options.no_build)
    except build.BuildError as exc:
        pytest.skip(str(exc))


@pytest.fixture(scope="session")
def engine(build_options: BuildOptions) -> EngineHarness:
    """The real ``vship::HipEngine``, compiled from the current sources."""
    available, reason = build.gpu_available()
    if not available:
        pytest.skip(reason)
    try:
        lib = build.build_engine_harness(force=build_options.force, no_build=build_options.no_build)
    except build.BuildError as exc:
        pytest.skip(str(exc))
    harness = EngineHarness(lib)
    count = harness.device_count()
    if count <= 0:
        pytest.skip(f"no usable HIP device (hipGetDeviceCount -> {count})")
    return harness


@pytest.fixture(scope="session")
def fixtures() -> Fixtures:
    """Tiny ONNX fixtures, generated in a subprocess (see harness/fixtures.py)."""
    fx = Fixtures(paths.FIXTURE_DIR)
    try:
        fx.ensure()
    except FixtureError as exc:
        pytest.skip(str(exc))
    return fx


@pytest.fixture(scope="session")
def ep_library(build_options: BuildOptions):
    """Build and register the HIP execution provider; returns its path."""
    available, reason = build.gpu_available()
    if not available:
        pytest.skip(reason)
    ort = pytest.importorskip("onnxruntime", reason="onnxruntime is required for the EP suite")
    try:
        so = build.build_ep(force=build_options.force, no_build=build_options.no_build)
    except build.BuildError as exc:
        pytest.skip(str(exc))

    from onnxruntime.capi import _pybind_state

    _pybind_state.set_default_logger_severity(3)
    _pybind_state.register_execution_provider_library("HIPExecutionProvider", str(so))
    if not any(getattr(d, "ep_name", "") == "HIPExecutionProvider" for d in ort.get_ep_devices()):
        pytest.skip("HIPExecutionProvider registration failed")
    return so


@pytest.fixture(scope="session")
def plugin(build_options: BuildOptions):
    """Build the plugin and make sure VapourSynth will load that exact binary.

    Returns ``(built_lib, installed_lib)``.
    """
    available, reason = build.gpu_available()
    if not available:
        pytest.skip(reason)
    if shutil.which("vspipe") is None:
        pytest.skip("vspipe is not on PATH")
    try:
        lib = build.build_plugin(force=build_options.force, no_build=build_options.no_build)
    except build.BuildError as exc:
        pytest.skip(str(exc))

    dest = build.plugin_install_dir() / lib.name
    if build_options.install_plugin:
        try:
            installed, changed = build.install_plugin(lib, force=build_options.force)
        except build.BuildError as exc:
            pytest.skip(str(exc))
        if changed:
            print(f"[correctness] installed {lib.name} -> {installed}")
    else:
        if not dest.exists() or build.sha256(dest) != build.sha256(lib):
            pytest.skip(
                "the installed libhip.so differs from the build and "
                "--no-install-plugin was given; rerun without that flag"
            )
        installed = dest
    return lib, installed


@pytest.fixture(scope="session")
def shipped_models() -> dict:
    """Paths to the shipped ArtCNN models; skips when a model is absent."""
    missing = [name for name, path in paths.SHIPPED_MODELS.items() if not path.exists()]
    if missing:
        pytest.skip(f"shipped model(s) missing: {', '.join(missing)}")
    return dict(paths.SHIPPED_MODELS)
