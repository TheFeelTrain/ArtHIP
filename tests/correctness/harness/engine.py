"""ctypes wrapper around the C++ engine driver (``engine_harness.cc``).

The wrapper keeps the test modules free of ctypes plumbing: ``probe`` and
``run`` raise :class:`EngineHarnessError` with the engine's own message when it
rejects a model or a request, which is what the rejection tests assert on.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import ctypes

import numpy as np

from . import reference

_ERR_LEN = 1024
_MAX_RANK = 8
_DEVICE_COUNT_UNKNOWN = -1


class EngineHarnessError(RuntimeError):
    """The engine rejected the model, the shape, or the run."""


@dataclass(frozen=True)
class Probe:
    """What the engine reports for one concrete input shape."""

    in_fp32: bool
    out_fp32: bool
    in_channels: int
    out_channels: int
    shape: tuple[int, ...]  # NCHW


@dataclass(frozen=True)
class RunResult:
    """A completed frame: logical NCHW float64 plus the NCHW shape."""

    output: np.ndarray
    shape: tuple[int, ...]


class EngineHarness:
    """Thin, reusable binding to the real ``vship::HipEngine``."""

    def __init__(self, lib_path: Path):
        self._lib = ctypes.CDLL(str(lib_path))
        self._bind()

    # -- binding ----------------------------------------------------------
    def _bind(self) -> None:
        lib = self._lib
        i32, i64 = ctypes.c_int, ctypes.c_int64
        i32p, i64p = ctypes.POINTER(i32), ctypes.POINTER(i64)

        lib.engine_device_count.argtypes = [i32p]
        lib.engine_device_count.restype = i32

        lib.engine_probe.argtypes = [
            ctypes.c_char_p, i32, i32, i32, i32, i32,
            ctypes.c_char_p, i32,
            i32p, i32p, i32p, i32p, i64p, i32p,
        ]
        lib.engine_probe.restype = i32

        lib.engine_run.argtypes = [
            ctypes.c_char_p, i32, i32, i32, i32, i32,
            ctypes.c_void_p, ctypes.c_size_t,
            ctypes.c_void_p, ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_size_t), i32p, i64p, i32p,
            ctypes.c_char_p, i32,
        ]
        lib.engine_run.restype = i32

        lib.engine_alloc_failure_then_run.argtypes = [
            ctypes.c_char_p, i32, i32, i32, i32, i32,
            ctypes.c_void_p, ctypes.c_size_t,
            ctypes.c_void_p, ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_size_t), i32p, i64p, i32p,
            ctypes.c_char_p, i32, ctypes.c_char_p, i32,
        ]
        lib.engine_alloc_failure_then_run.restype = i32

    # -- helpers ----------------------------------------------------------
    @staticmethod
    def _err_buf() -> ctypes.Array:
        return ctypes.create_string_buffer(_ERR_LEN)

    @staticmethod
    def _decode_err(buf: ctypes.Array, fallback: str) -> str:
        text = bytes(buf.value).decode(errors="replace").strip()
        return text or fallback

    @staticmethod
    def _shape_out() -> tuple[ctypes.Array, ctypes.c_int]:
        return (ctypes.c_int64 * _MAX_RANK)(), ctypes.c_int(0)

    @staticmethod
    def _read_shape(shape: ctypes.Array, rank: ctypes.c_int) -> tuple[int, ...]:
        return tuple(int(shape[i]) for i in range(rank.value))

    # -- API --------------------------------------------------------------
    def device_count(self) -> int:
        """Number of HIP devices, or ``-1`` when the runtime is unusable."""
        count = ctypes.c_int(_DEVICE_COUNT_UNKNOWN)
        if self._lib.engine_device_count(ctypes.byref(count)) != 0:
            return _DEVICE_COUNT_UNKNOWN
        return count.value

    def probe(self, model: Path | str, *, n: int = 1, h: int = 16, w: int = 16, c: int = 1,
              device: int = 0) -> Probe:
        """Build the model and propagate one shape; raises on rejection."""
        err = self._err_buf()
        in_fp32, out_fp32 = ctypes.c_int(0), ctypes.c_int(0)
        cin, cout = ctypes.c_int(0), ctypes.c_int(0)
        shape, rank = self._shape_out()
        rc = self._lib.engine_probe(
            str(model).encode(), n, h, w, c, device, err, _ERR_LEN,
            ctypes.byref(in_fp32), ctypes.byref(out_fp32),
            ctypes.byref(cin), ctypes.byref(cout), shape, ctypes.byref(rank),
        )
        if rc != 0:
            raise EngineHarnessError(self._decode_err(err, f"engine_probe failed (rc={rc})"))
        return Probe(
            in_fp32=bool(in_fp32.value),
            out_fp32=bool(out_fp32.value),
            in_channels=cin.value,
            out_channels=cout.value,
            shape=self._read_shape(shape, rank),
        )

    def run(self, model: Path | str, *, n: int = 1, h: int = 16, w: int = 16, c: int = 1,
            device: int = 0, raw_input: np.ndarray | None = None) -> RunResult:
        """Run one frame; ``raw_input`` overrides the default ramp input."""
        probe = self.probe(model, n=n, h=h, w=w, c=c, device=device)
        if raw_input is None:
            raw_input = reference.raw_input(c, h, w, probe.in_fp32)
        raw_input = np.ascontiguousarray(raw_input)
        return self._run_prepared(model, probe, raw_input, n=n, h=h, w=w, c=c, device=device)

    def _run_prepared(self, model, probe: Probe, raw_input: np.ndarray, *, n, h, w, c,
                      device) -> RunResult:
        numel = int(np.prod(probe.shape))
        cap = numel * (4 if probe.out_fp32 else 2)
        out = ctypes.create_string_buffer(cap)
        out_bytes = ctypes.c_size_t(0)
        out_fp32 = ctypes.c_int(0)
        shape, rank = self._shape_out()
        err = self._err_buf()
        rc = self._lib.engine_run(
            str(model).encode(), n, h, w, c, device,
            ctypes.c_void_p(raw_input.ctypes.data), raw_input.nbytes,
            ctypes.cast(out, ctypes.c_void_p), cap,
            ctypes.byref(out_bytes), ctypes.byref(out_fp32), shape, ctypes.byref(rank),
            err, _ERR_LEN,
        )
        if rc != 0:
            raise EngineHarnessError(self._decode_err(err, f"engine_run failed (rc={rc})"))
        result_shape = self._read_shape(shape, rank)
        data = out.raw[: out_bytes.value]
        return RunResult(output=reference.decode_output(data, result_shape, bool(out_fp32.value)),
                         shape=result_shape)

    def alloc_failure_then_run(self, model: Path | str, *, big_h: int = 200_000,
                               big_w: int = 200_000, n: int = 1, c: int = 1, h: int = 16,
                               w: int = 16, device: int = 0) -> tuple[str, RunResult]:
        """Force an oversized allocation, then run a valid frame on the SAME
        engine. Returns the reported first failure and the recovery result."""
        probe = self.probe(model, n=n, h=h, w=w, c=c, device=device)
        raw_input = np.ascontiguousarray(reference.raw_input(c, h, w, probe.in_fp32))
        numel = int(np.prod(probe.shape))
        cap = numel * (4 if probe.out_fp32 else 2)
        out = ctypes.create_string_buffer(cap)
        out_bytes = ctypes.c_size_t(0)
        out_fp32 = ctypes.c_int(0)
        shape, rank = self._shape_out()
        failerr, runerr = self._err_buf(), self._err_buf()
        rc = self._lib.engine_alloc_failure_then_run(
            str(model).encode(), n, big_h, big_w, c, device,
            ctypes.c_void_p(raw_input.ctypes.data), raw_input.nbytes,
            ctypes.cast(out, ctypes.c_void_p), cap,
            ctypes.byref(out_bytes), ctypes.byref(out_fp32), shape, ctypes.byref(rank),
            failerr, _ERR_LEN, runerr, _ERR_LEN,
        )
        fail_msg = self._decode_err(failerr, "")
        if rc != 0:
            raise EngineHarnessError(
                f"driver rc={rc}; reported failure={fail_msg!r}; "
                f"recovery={self._decode_err(runerr, '')!r}"
            )
        result_shape = self._read_shape(shape, rank)
        data = out.raw[: out_bytes.value]
        return fail_msg, RunResult(
            output=reference.decode_output(data, result_shape, bool(out_fp32.value)),
            shape=result_shape,
        )
