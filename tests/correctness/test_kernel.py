"""Kernel memory-ordering invariants — the intermittent-corruption gate.

``NOTES.md`` documents a rare, per-process output corruption that reproduced in
both the VapourSynth plugin and the execution provider, at a random
convolution, with bit-identical inputs.  The root cause was an LDS
producer/consumer race inside ``winograd_conv``:

* ``load_strip`` writes the fp16 input strip to shared memory (``strip4``);
* the V transform of every wave reads that strip;
* ``S_BARRIER`` synchronizes control flow but does **not** retire outstanding
  LDS stores, so a wave could arrive at the top-of-loop barrier with its
  ``strip4`` stores still in flight while another wave passed the barrier and
  read stale shared memory.

LLVM's ``SIInsertWaitcnts`` pass normally inserts ``s_waitcnt lgkmcnt(0)``
before such a barrier, and it did for three of the four barriers in this
kernel — but not on the loop-back edge, where the strip prefetch's stores
arrive.  The fix is the ``lds_barrier()`` helper in ``src/common/hip_kernels.h``:
it emits the drain explicitly, so the ordering no longer depends on the
compiler's waitcnt analysis.

These checks are deliberately *deterministic and GPU-free*: they inspect the
source and the emitted gfx1100 assembly, so they catch the regression (and any
future one) even though the runtime symptom is a ~1-in-10 event that a single
correctness run usually misses.  ``test_engine.py`` adds an end-to-end check on
top (fresh engines never agree on a corrupt first frame).
"""

from __future__ import annotations

from pathlib import Path

from harness import paths

#: How far back from an ``s_barrier`` the LDS drain may sit. The plugin build
#: puts it on the immediately preceding line; a little slack keeps the check
#: from breaking on harmless scheduling differences.
_DRAIN_WINDOW = 6


# ---------------------------------------------------------------------------
# Assembly helpers
# ---------------------------------------------------------------------------


def _strip_comment(line: str) -> str:
    return line.split(";", 1)[0].rstrip()


def _symbol_instructions(asm: str, symbol_fragment: str) -> list[str]:
    """Instruction lines of the emitted function whose label contains the name.

    Labels and assembler directives are dropped, so the result is the straight
    instruction stream the hardware executes.
    """
    lines = asm.splitlines()
    start = None
    for i, line in enumerate(lines):
        text = _strip_comment(line).strip()
        if text.endswith(":") and symbol_fragment in text:
            start = i + 1
            break
    assert start is not None, f"symbol containing {symbol_fragment!r} not found in the ISA dump"

    instructions: list[str] = []
    for line in lines[start:]:
        text = _strip_comment(line).strip()
        if text.startswith(".size"):
            break
        if not text or text.endswith(":") or text.startswith("."):
            continue
        instructions.append(text)
    assert instructions, f"no instructions between {symbol_fragment!r} and .size"
    return instructions


def _drains_lds(instruction: str) -> bool:
    return instruction.startswith("s_waitcnt") and "lgkmcnt(0)" in instruction


def _strip_cpp_comments(text: str) -> str:
    """Drop ``//`` and ``/* ... */`` comments so prose cannot satisfy a check."""
    out: list[str] = []
    i = 0
    while i < len(text):
        if text.startswith("//", i):
            end = text.find("\n", i)
            i = len(text) if end < 0 else end
        elif text.startswith("/*", i):
            end = text.find("*/", i + 2)
            i = len(text) if end < 0 else end + 2
        else:
            out.append(text[i])
            i += 1
    return "".join(out)


def test_every_winograd_conv_barrier_is_preceded_by_an_lds_drain(kernel_asm: Path):
    """No wave may pass a ``winograd_conv`` barrier with an LDS store in flight.

    This is the invariant the corruption violated: on the loop-back edge the
    strip4 stores reached the top-of-loop ``s_barrier`` with no preceding
    ``s_waitcnt lgkmcnt(0)``.
    """
    instructions = _symbol_instructions(kernel_asm.read_text(), "winograd_conv")
    barriers = [i for i, ins in enumerate(instructions) if ins.startswith("s_barrier")]
    assert len(barriers) == 4, (
        f"winograd_conv should have 4 workgroup barriers, found {len(barriers)}"
    )
    for index in barriers:
        window = instructions[max(0, index - _DRAIN_WINDOW):index]
        assert any(_drains_lds(ins) for ins in window), (
            f"barrier at instruction {index} has no LDS drain in the preceding "
            f"{_DRAIN_WINDOW} instructions: {window}"
        )


def test_winograd_conv_uses_the_draining_barrier_helper():
    """The kernel must not call ``__syncthreads()`` directly.

    Relying on the compiler to insert the LDS drain is exactly what failed, so
    the source has to say what it needs.  This catches a revert that happens to
    still emit a drain under the current toolchain but would silently lose it
    under another.
    """
    source = (paths.COMMON_SOURCES / "hip_kernels.h").read_text()
    start = source.index("__global__ void winograd_conv(")
    end = source.index("__global__", start + 1)
    body = _strip_cpp_comments(source[start:end])
    assert "__syncthreads" not in body, (
        "winograd_conv calls __syncthreads(); cross-wave LDS ordering must use "
        "lds_barrier() so the drain cannot be dropped by the waitcnt pass"
    )
    assert body.count("lds_barrier()") >= 4, (
        f"winograd_conv should drain at all 4 barriers, found {body.count('lds_barrier()')}"
    )
