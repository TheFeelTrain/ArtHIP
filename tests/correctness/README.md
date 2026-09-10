# ArtHIP correctness suite

Executable checks for the HIP plugin and execution provider. They run against
the **real** engine, the real gfx1100 kernels, the real ONNX Runtime provider
and the real VapourSynth plugin — no CPU shim.

Every test is named after the `REVIEW.md` finding it covers (`test_p1_03_...` =
finding 3), so a failure points straight back at the review item that motivated
the check.

## Quick start

```bash
python tests/correctness/run.py                 # all suites
python tests/correctness/run.py engine ep       # selected suites
python tests/correctness/run.py --list          # show suites
python tests/correctness/run.py --rebuild       # force artifact rebuilds
python tests/correctness/run.py engine -- -k geometry -x   # extra pytest args
```

Anything after `--` is forwarded to pytest. A single suite can also be run
directly, which is the fastest way to iterate:

```bash
cd tests/correctness
python -m pytest test_engine.py -v
python -m pytest -k blocksize
```

Requirements: `hipcc`/`/dev/kfd` (a machine without them is skipped, not
failed), `pytest`, `onnxruntime` (EP suite and the chroma CPU reference), and
`vapoursynth` + `vspipe` (plugin suite).

## Suites

| Suite | File | Covers |
|---|---|---|
| `engine` | `test_engine.py` | The standalone `HipEngine`: kernel geometry, weight/bias validation, typed initializers, binary ops, DTS block size, device selection, allocation failures. 40 checks. |
| `ep` | `test_ep.py` | The ONNX Runtime provider: fp32 boundary conversion, device fallback, shape rebuilds. |
| `plugin` | `test_plugin.py` | The VapourSynth plugin: which clip formats/plane counts are accepted and rejected, plus the shipped multi-plane models and their chroma planes. 18 checks. |

Performance is deliberately out of scope — see `tests/multires.py`,
`tests/benchmark.py` and `tests/vs_test.py` for speed/accuracy runs.

## Shipped models

The plugin checks run the real ArtCNN models in `tests/`:

| Model | Graph IO | What it exercises |
|---|---|---|
| `ArtCNN_R8F64_fp16.onnx` (`luma`) | 1 plane → 1 plane | 2x upscale, tail DepthToSpace, GRAY in/out |
| `ArtCNN_R8F64_YCbCr_DEHALO.onnx` (`dehalo`) | 3 planes → 3 planes | Multi-channel RGB/YUV, 1:1 |
| `ArtCNN_R8F64_Chroma.onnx` (`chroma`) | 3 planes → **2** planes | Chroma-only output (U and V) through the flexible-output protocol, consumed exactly as vsscale's `R8F64_Chroma` does |

The chroma model is the strongest layout check in the suite: it returns two
chroma planes, and each is compared against an independent ONNX Runtime **CPU**
run of the same model on the same input. Correct pairing measures ~0.006; a
swapped, shifted or de-interleaved plane measures 1.0 — 50x the tolerance — so
the check has real teeth. `harness/vsrun.py` runs these Python probes in a child
process and reads back a JSON result.

## How artifacts are built

`harness/build.py` compiles the engine driver, the provider and the plugin from
the current sources. An artifact is rebuilt when any source it depends on is
newer, so a normal run tests the working tree:

- `--rebuild` always rebuilds.
- `--no-build` never rebuilds (fails with a skip when an artifact is stale).
- `--device N` selects the HIP device (default 0).
- `--no-install-plugin` leaves the system `libhip.so` alone.

A failed build becomes a **skip with the compiler output**, not a confusing
test failure.

The plugin is installed into VapourSynth's plugin directory before the plugin
suite runs: VapourSynth auto-loads plugins by id, so explicitly loading a second
copy of `libhip.so` fails with "already loaded" — the installed file has to be
the artifact under test. The suite verifies the installed file matches the build.

## Extending

**Add a test.** Add a function to the relevant `test_*.py`. Use the shared
fixtures and the helpers in `harness/reference.py`:

```python
def test_my_case(engine, fixtures, device):
    fx = fixtures.get("conv_c4m4")
    result = engine.run(fx.model, h=16, w=16, c=4, device=device)
    expected = ref.conv3x3(ref.logical_input(4, 16, 16, in_fp32=False), fx.weights, fx.bias)
    ref.assert_close(result.output, expected, ref.FP16_CONV_TOL, "my case")
```

`engine.probe()` / `engine.run()` raise `EngineHarnessError` with the engine's
own message when it rejects a model, so rejection tests just do:

```python
with pytest.raises(EngineHarnessError, match="3x3"):
    engine.probe(fx.model, h=16, w=16, c=4, device=device)
```

**Add a fixture.** Register a builder in `harness/modelgen.py`:

```python
@fixture("my_model")
def _my_model(out: Path) -> Fixture:
    ...
```

`fixtures.get("my_model")` then returns it; the manifest and `.npy` weight/bias
arrays are written automatically. Keep fixtures tiny — they probe engine
behaviour, not accuracy.

**Add a suite.** Drop a `test_<name>.py` in this directory and add it to
`SUITES` in `run.py`.

**Add a shipped-model check.** Register the model in `SHIPPED_MODELS`
(`harness/paths.py`) and add the case to the relevant list in
`test_plugin.py`. For checks that need plane data or frame properties, write a
Python probe and run it through `harness/vsrun.run_script` — see
`CHROMA_PLANES_PROBE` for the pattern (it reads the flexible-output frames and
compares them against an independent ONNX Runtime CPU run).

## Design notes

**One process per suite.** `run.py` invokes pytest separately per suite because
the HIP runtime, the ONNX Runtime provider and VapourSynth do not always tear
down cleanly when combined (see `src/hip/HIP_NOTES.md`).

**Fixtures are generated in a subprocess.** The `vsengine` pytest plugin imports
VapourSynth in every pytest process, which auto-loads `libhip.so` and with it the
C++ protobuf/ONNX descriptors. Importing the `onnx` Python package on top of
that aborts the interpreter with a duplicate-descriptor check. So the ONNX
builders in `harness/modelgen.py` only ever run as `python -m harness.modelgen`,
and the test process reads `harness/fixtures.py` (no `onnx` import). Keep that
split.

**Scope of the EP suite.** The EP's supported layout is a single activation
input/output, and its multi-channel buffer conversion is still open
(`REVIEW.md` P2-11). The P1-4/P1-6 checks therefore use one input and one output
channel, and the multi-channel gap is recorded as a **strict xfail**
(`test_p2_11_multichannel_layout_is_converted`): when P2-11 is fixed that test
XPASSes and fails the run until the marker is removed — the intended prompt.

`tests/README.md`-style reference data and the shipped ArtCNN models live in
`tests/`; this suite reuses them for the plugin checks.

## Files

```
tests/correctness/
├── run.py                  # one command, process per suite
├── conftest.py             # options and session fixtures
├── test_engine.py          # HipEngine checks
├── test_ep.py              # execution-provider checks
├── test_plugin.py          # VapourSynth plugin checks
├── harness/
│   ├── paths.py            # repo/build locations and shipped models
│   ├── build.py            # build + install helpers
│   ├── reference.py        # NumPy references and comparison helpers
│   ├── fixtures.py         # fixture loading (no onnx import)
│   ├── modelgen.py         # ONNX fixture builders (subprocess only)
│   ├── vsrun.py            # run VapourSynth probes, read JSON results
│   ├── engine.py           # ctypes wrapper
│   └── engine_harness.cc   # C++ driver over the real engine
└── build/                  # generated: fixtures, harness .so, scripts
```

`build/` is generated and ignored by git (via the repository-wide `build/`
rule).
