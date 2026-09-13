# tests/

Fixtures, manual tools, and the automated gate. Nothing in this tree is built
by the repository itself.

## Layout

| path | what |
|---|---|
| `fixtures/models/` | shipped ONNX models (ArtCNN R8F64 luma / chroma / dehalo) |
| `fixtures/images/` | input photos (`test_<res>.png`) that drive the swept resolutions |
| `correctness/` | the pytest gate — see `correctness/README.md` |
| `tools/` | manual benches and comparisons; not run by any automated flow |
| `.cache/` | everything generated: CPU references, `corrupt/` samples, `migraphx/` compiled programs — git-ignored |

## Tools

Each tool resolves its own paths, so it can be run from anywhere.

| tool | purpose |
|---|---|
| `tools/multires.py [res]` | HIP EP vs MIGraphX EP accuracy + speed per resolution |
| `tools/vs_test.py` | VapourSynth script for `vspipe` (`VS_BACKEND=hip\|migx`) |
| `tools/corruption_hunt.py` | hunts the known intermittent corruption |

## Caches

`.cache/` holds everything generated and is git-ignored; nothing in it is
committed. `multires.py` and `corruption_hunt.py` key every CPU reference by the
SHA-256 of the model file *and* the input tensor, so a stale entry is detected
and recomputed rather than silently trusted. The `migraphx/` subtree holds
compiled programs keyed per resolution and is expensive to rebuild — leave it in
place.

## Gate

`python tests/correctness/run.py` is the only automated suite. Run it before and
after any kernel or engine change.
