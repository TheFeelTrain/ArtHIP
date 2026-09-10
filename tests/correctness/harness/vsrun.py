"""Run VapourSynth scripts in a subprocess and read structured results.

``vspipe`` is enough for "does this clip/model pair load" checks, but reading
plane data, frame properties and the flexible-output map needs the Python API.
Those probes run as real scripts in a child process, which also keeps the
VapourSynth runtime out of the pytest process.

A probe prints one line of the form ``<marker> {json}``; the last such line is
parsed and returned as :attr:`ScriptResult.result`.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path

from . import paths

RESULT_MARKER = "__VS_RESULT__"


@dataclass(frozen=True)
class ScriptResult:
    """Outcome of one probe run."""

    rc: int
    stdout: str
    stderr: str
    result: dict | None

    @property
    def output(self) -> str:
        """Combined stdout/stderr, for failure messages."""
        return f"{self.stdout}{self.stderr}".strip()

    def require_ok(self, label: str) -> dict:
        """Fail the calling test unless the probe ran and reported a result."""
        if self.rc != 0 or self.result is None:
            raise AssertionError(f"{label}: probe failed (rc={self.rc}):\n{self.output}")
        return self.result


def run_script(name: str, body: str, *, env: dict | None = None, timeout: int = 600) -> ScriptResult:
    """Write ``body`` to ``build/scripts/<name>.py`` and run it."""
    paths.SCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    script = paths.SCRIPT_DIR / f"{name}.py"
    script.write_text(textwrap.dedent(body).lstrip("\n") + "\n")

    run_env = dict(os.environ, MANGOHUD="0")
    if env:
        run_env.update({k: str(v) for k, v in env.items()})
    proc = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True, text=True, env=run_env,
        cwd=str(paths.REPO_ROOT), timeout=timeout,
    )
    parsed: dict | None = None
    for line in proc.stdout.splitlines():
        if line.startswith(RESULT_MARKER):
            parsed = json.loads(line[len(RESULT_MARKER):])
    return ScriptResult(proc.returncode, proc.stdout, proc.stderr, parsed)


def first_missing(model_paths: dict) -> list[str]:
    """Names of shipped models that are not present on disk."""
    return [name for name, path in model_paths.items() if not Path(path).exists()]
