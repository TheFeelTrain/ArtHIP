"""Fixture loading for the correctness suite.

The ONNX builders live in :mod:`harness.modelgen`, which imports the ``onnx``
Python package. That import must not happen in the test process: pytest loads
the ``vsengine`` plugin, which imports VapourSynth, which auto-loads
``libhip.so`` and with it the C++ protobuf/ONNX descriptors. Importing the
``onnx`` Python bindings on top of that aborts the interpreter with a
duplicate-descriptor check ("File already exists in database:
onnx/onnx-ml.proto").

Generation therefore runs in a subprocess (``python -m harness.modelgen``) and
this module only reads the resulting manifest plus the ``.npy`` arrays.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import paths

MANIFEST_NAME = "fixtures.json"


class FixtureError(RuntimeError):
    """Fixtures could not be generated."""


@dataclass(frozen=True)
class Fixture:
    """A generated model plus the arrays needed to compute its reference."""

    name: str
    model: Path
    input_channels: int
    output_channels: int
    #: fp16 weights/bias exactly as stored in the model (cast for reference math).
    weights: np.ndarray | None = None
    bias: np.ndarray | None = None
    #: DepthToSpace block size for DTS fixtures, else 0.
    blocksize: int = 0
    #: True when the model's graph IO is fp32 (boundary Casts or native fp32).
    fp32_io: bool = False


class Fixtures:
    """Reads (and lazily generates) the fixture set for one session."""

    def __init__(self, out_dir: Path):
        self.out_dir = Path(out_dir)
        self._manifest: dict[str, dict] | None = None

    # -- generation -------------------------------------------------------
    @property
    def manifest_path(self) -> Path:
        return self.out_dir / MANIFEST_NAME

    def _is_stale(self) -> bool:
        if not self.manifest_path.exists():
            return True
        built = self.manifest_path.stat().st_mtime
        sources = [Path(__file__).resolve().parent / "modelgen.py", Path(__file__).resolve()]
        return any(src.stat().st_mtime > built for src in sources)

    def ensure(self, *, force: bool = False) -> None:
        """Generate the fixture set in a subprocess when it is missing or stale."""
        if not force and not self._is_stale():
            return
        self.out_dir.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(
            [str(paths.SUITE_DIR), env.get("PYTHONPATH", "")]
        ).strip(os.pathsep)
        result = subprocess.run(
            [sys.executable, "-m", "harness.modelgen", str(self.out_dir)],
            cwd=str(paths.SUITE_DIR), capture_output=True, text=True, env=env,
        )
        if result.returncode != 0:
            tail = "\n".join((result.stdout + result.stderr).strip().splitlines()[-20:])
            raise FixtureError(f"fixture generation failed (exit {result.returncode}):\n{tail}")

    # -- access -----------------------------------------------------------
    def _load(self) -> dict[str, dict]:
        if self._manifest is None:
            self.ensure()
            try:
                self._manifest = json.loads(self.manifest_path.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                raise FixtureError(f"cannot read {self.manifest_path}: {exc}") from exc
        return self._manifest

    def names(self) -> list[str]:
        return sorted(self._load())

    def get(self, name: str) -> Fixture:
        entry = self._load().get(name)
        if entry is None:
            raise KeyError(f"unknown fixture {name!r}; known: {self.names()}")

        def _array(key: str):
            rel = entry.get(key)
            if not rel:
                return None
            return np.load(self.out_dir / rel)

        return Fixture(
            name=name,
            model=self.out_dir / entry["model"],
            input_channels=entry["input_channels"],
            output_channels=entry["output_channels"],
            weights=_array("weights"),
            bias=_array("bias"),
            blocksize=entry.get("blocksize", 0),
            fp32_io=entry.get("fp32_io", False),
        )
