"""Tiny ONNX fixtures for the correctness suite (generation side).

Fixtures are small, deterministic models that each exercise one engine property
(kernel geometry, weight validation, DTS block size, ...). Keep them tiny (tens
of pixels, a handful of channels): they probe engine behaviour, they are not
accuracy benchmarks. The shipped ArtCNN models are used by the plugin checks.

This module imports the ``onnx`` Python package, so it is only ever run as a
subprocess by :class:`harness.fixtures.Fixtures` -- see that module for why it
cannot be imported inside the pytest process. Run it directly with::

    python -m harness.modelgen <output_dir>

Extending
---------
Add a function decorated with ``@fixture("name")`` that takes the output
directory and returns a :class:`harness.fixtures.Fixture`. It is then available
to tests as ``fixtures.get("name")``; the manifest and ``.npy`` arrays are
written automatically.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from .fixtures import MANIFEST_NAME, Fixture

OPSET = 17

_REGISTRY: dict[str, Callable[[Path], Fixture]] = {}


def fixture(name: str):
    """Register a fixture builder under ``name``."""

    def decorate(fn: Callable[[Path], Fixture]) -> Callable[[Path], Fixture]:
        if name in _REGISTRY:
            raise ValueError(f"duplicate fixture name: {name}")
        _REGISTRY[name] = fn
        return fn

    return decorate


# ---------------------------------------------------------------------------
# ONNX helpers
# ---------------------------------------------------------------------------


def _vi(name: str, shape, dtype=TensorProto.FLOAT16):
    return helper.make_tensor_value_info(name, dtype, list(shape))


def _save(model, path: Path, *, external: bool = False) -> Path:
    onnx.checker.check_model(model)
    if external:
        onnx.save_model(
            model,
            str(path),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=f"{path.stem}.bin",
            size_threshold=0,
        )
        assert (path.parent / f"{path.stem}.bin").exists(), f"external data missing for {path}"
    else:
        onnx.save(model, str(path))
    return path


def _model(nodes, input_name, input_shape, input_type, output_name, output_shape, output_type, inits):
    graph = helper.make_graph(
        nodes,
        "fixture",
        [_vi(input_name, input_shape, input_type)],
        [_vi(output_name, output_shape, output_type)],
        inits,
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)])


def _half(name: str, array: np.ndarray):
    return numpy_helper.from_array(np.asarray(array, dtype=np.float16), name)


def _float(name: str, array: np.ndarray):
    return numpy_helper.from_array(np.asarray(array, dtype=np.float32), name)


def _typed_half(name: str, array: np.ndarray):
    """A FLOAT16 initializer whose values live in ``int32_data`` as raw 16-bit
    patterns -- exactly what ArtHIP's own converter emits."""
    half = np.asarray(array, dtype=np.float16)
    bits = half.view(np.uint16).astype(np.int32).ravel()
    proto = TensorProto()
    proto.name = name
    proto.data_type = TensorProto.FLOAT16
    proto.dims.extend(list(half.shape))
    proto.int32_data.extend(bits.tolist())
    return proto


def _conv_node(x, w, b, out, *, kernel=3, pads=(1, 1, 1, 1)):
    inputs = [x, w] + ([b] if b else [])
    return helper.make_node(
        "Conv",
        inputs,
        [out],
        kernel_shape=[kernel, kernel],
        pads=list(pads),
        strides=[1, 1],
        dilations=[1, 1],
        group=1,
    )


def _kern(seed: int, m: int, c: int, kernel: int = 3) -> np.ndarray:
    return np.random.RandomState(seed).uniform(-0.5, 0.5, (m, c, kernel, kernel)).astype(np.float32)


def _linspace(m: int, lo: float = -0.1, hi: float = 0.1) -> np.ndarray:
    return np.linspace(lo, hi, m).astype(np.float32)


# ---------------------------------------------------------------------------
# Single-conv fixtures
# ---------------------------------------------------------------------------


def _single_conv(out: Path, name: str, *, m: int, c: int, seed: int,
                 bias: str | int = "full", kernel: int = 3, pads=(1, 1, 1, 1),
                 external: bool = False) -> Fixture:
    """One 3x3 SAME conv. ``bias`` is "full", "none", or an explicit length."""
    w = _kern(seed, m, c, kernel)
    fields: dict = {"weights": w.astype(np.float16)}
    inits = [_half("w", w)]
    bias_name = None
    if bias != "none":
        blen = m if bias == "full" else int(bias)
        b = _linspace(blen)
        inits.append(_half("b", b))
        bias_name = "b"
        if blen == m:
            fields["bias"] = b.astype(np.float16)
    node = _conv_node("x", "w", bias_name, "y", kernel=kernel, pads=pads)
    model = _model([node], "x", [1, c, "h", "w"], TensorProto.FLOAT16,
                   "y", [1, m, "h", "w"], TensorProto.FLOAT16, inits)
    return Fixture(name=name, model=_save(model, out / f"{name}.onnx", external=external),
                   input_channels=c, output_channels=m, **fields)


@fixture("conv_c4m4")
def _conv_c4m4(out: Path) -> Fixture:
    """4->4 channels, the general winograd path."""
    return _single_conv(out, "conv_c4m4", m=4, c=4, seed=11)


@fixture("conv_c4m8")
def _conv_c4m8(out: Path) -> Fixture:
    """4->8 channels."""
    return _single_conv(out, "conv_c4m8", m=8, c=4, seed=12)


@fixture("conv_c4m64")
def _conv_c4m64(out: Path) -> Fixture:
    """4->64 channels: the largest output count the kernels cover."""
    return _single_conv(out, "conv_c4m64", m=64, c=4, seed=13)


@fixture("conv_c1m8")
def _conv_c1m8(out: Path) -> Fixture:
    """1->8 channels: the direct (non-WMMA) first-conv path."""
    return _single_conv(out, "conv_c1m8", m=8, c=1, seed=21)


@fixture("conv_m128")
def _conv_m128(out: Path) -> Fixture:
    """128 output channels: beyond what the four-wave kernels cover."""
    return _single_conv(out, "conv_m128", m=128, c=4, seed=51, bias="none")


@fixture("conv_1x1")
def _conv_1x1(out: Path) -> Fixture:
    """A legal 1x1 Conv with explicit pad 1 (REVIEW.md P1-2 crash case)."""
    return _single_conv(out, "conv_1x1", m=4, c=4, seed=31, bias="none", kernel=1)


@fixture("conv_short_bias")
def _conv_short_bias(out: Path) -> Fixture:
    """Bias shorter than the weight's output-channel count."""
    return _single_conv(out, "conv_short_bias", m=4, c=4, seed=41, bias=2)


@fixture("conv_external")
def _conv_external(out: Path) -> Fixture:
    """Weights stored as ONNX external data (unsupported by the parser)."""
    return _single_conv(out, "conv_external", m=4, c=4, seed=95, bias="none", external=True)


# ---------------------------------------------------------------------------
# Typed-initializer fixtures (P1-7)
# ---------------------------------------------------------------------------


def _identity_fixture(out: Path, name: str, *, typed: bool) -> Fixture:
    ident = np.zeros((1, 1, 3, 3), dtype=np.float32)
    ident[0, 0, 1, 1] = 1.0
    init = _typed_half("w", ident) if typed else _half("w", ident)
    node = _conv_node("x", "w", None, "y")
    model = _model([node], "x", [1, 1, "h", "w"], TensorProto.FLOAT16,
                   "y", [1, 1, "h", "w"], TensorProto.FLOAT16, [init])
    return Fixture(name=name, model=_save(model, out / f"{name}.onnx"),
                   input_channels=1, output_channels=1, weights=ident.astype(np.float16))


@fixture("identity_raw")
def _identity_raw(out: Path) -> Fixture:
    """3x3 identity conv with raw fp16 weight storage."""
    return _identity_fixture(out, "identity_raw", typed=False)


@fixture("identity_typed_half")
def _identity_typed_half(out: Path) -> Fixture:
    """Identical kernel stored in int32_data as fp16 bit patterns."""
    return _identity_fixture(out, "identity_typed_half", typed=True)


# ---------------------------------------------------------------------------
# Binary-op fixtures (P1-8)
# ---------------------------------------------------------------------------


@fixture("add_const")
def _add_const(out: Path) -> Fixture:
    """x = conv(x); y = x + 0.25 (initializer operand)."""
    w = _kern(61, 4, 4)
    k = np.full((1, 1, 1, 1), 0.25, dtype=np.float32)
    n0 = _conv_node("x", "w", None, "raw")
    n1 = helper.make_node("Add", ["raw", "k"], ["y"])
    model = _model([n0, n1], "x", [1, 4, "h", "w"], TensorProto.FLOAT16,
                   "y", [1, 4, "h", "w"], TensorProto.FLOAT16, [_half("w", w), _float("k", k)])
    return Fixture(name="add_const", model=_save(model, out / "add_const.onnx"),
                   input_channels=4, output_channels=4, weights=w.astype(np.float16))


@fixture("add_broadcast")
def _add_broadcast(out: Path) -> Fixture:
    """Add of a 4-channel and a 1-channel activation (legal ONNX broadcast)."""
    w = _kern(71, 4, 4)
    w2 = _kern(72, 1, 4)
    n0 = _conv_node("x", "w", None, "a")
    n1 = _conv_node("x", "w2", None, "b")
    n2 = helper.make_node("Add", ["a", "b"], ["y"])
    model = _model([n0, n1, n2], "x", [1, 4, "h", "w"], TensorProto.FLOAT16,
                   "y", [1, 4, "h", "w"], TensorProto.FLOAT16, [_half("w", w), _half("w2", w2)])
    return Fixture(name="add_broadcast", model=_save(model, out / "add_broadcast.onnx"),
                   input_channels=4, output_channels=4, weights=w.astype(np.float16))


# ---------------------------------------------------------------------------
# DepthToSpace fixtures (P1-9)
# ---------------------------------------------------------------------------


def _dts_fixture(out: Path, name: str, blocksize: int) -> Fixture:
    m = blocksize * blocksize
    w = _kern(80 + blocksize, m, 1)
    b = _linspace(m, -0.05, 0.05)
    n0 = _conv_node("x", "w", "b", "raw")
    n1 = helper.make_node("DepthToSpace", ["raw"], ["sq"], blocksize=blocksize, mode="DCR")
    n2 = helper.make_node("Cast", ["sq"], ["y"], to=TensorProto.FLOAT)
    model = _model([n0, n1, n2], "x", [1, 1, "h", "w"], TensorProto.FLOAT16,
                   "y", [1, 1, "h", "w"], TensorProto.FLOAT,
                   [_half("w", w), _half("b", b)])
    return Fixture(name=name, model=_save(model, out / f"{name}.onnx"), input_channels=1,
                   output_channels=1, weights=w.astype(np.float16), bias=b.astype(np.float16),
                   blocksize=blocksize, fp32_io=True)


@fixture("dts_b1")
def _dts_b1(out: Path) -> Fixture:
    """DepthToSpace blocksize 1 (identity; the fp32 fast path must decline it)."""
    return _dts_fixture(out, "dts_b1", 1)


@fixture("dts_b2")
def _dts_b2(out: Path) -> Fixture:
    """DepthToSpace blocksize 2 (the fp32 fused fast path)."""
    return _dts_fixture(out, "dts_b2", 2)


@fixture("dts_b3")
def _dts_b3(out: Path) -> Fixture:
    """DepthToSpace blocksize 3 (generic path; the fast path mis-indexed it)."""
    return _dts_fixture(out, "dts_b3", 3)


def _dts_multi_fixture(out: Path, name: str, *, blocksize: int, cout: int,
                       fp32_out: bool) -> Fixture:
    """DepthToSpace with several OUTPUT channels (P2-10).

    The DCR and CRD channel orders coincide only for a single output channel, so
    a multi-channel tail is what distinguishes them.
    """
    m = cout * blocksize * blocksize
    w = _kern(90 + blocksize * 10 + cout, m, 1)
    b = _linspace(m, -0.05, 0.05)
    nodes = [
        _conv_node("x", "w", "b", "raw"),
        helper.make_node("DepthToSpace", ["raw"], ["sq"], blocksize=blocksize, mode="DCR"),
    ]
    if fp32_out:
        nodes.append(helper.make_node("Cast", ["sq"], ["y"], to=TensorProto.FLOAT))
        out_name, out_type = "y", TensorProto.FLOAT
    else:
        out_name, out_type = "sq", TensorProto.FLOAT16
    model = _model(nodes, "x", [1, 1, "h", "w"], TensorProto.FLOAT16,
                   out_name, [1, cout, "h", "w"], out_type, [_half("w", w), _half("b", b)])
    return Fixture(name=name, model=_save(model, out / f"{name}.onnx"), input_channels=1,
                   output_channels=cout, weights=w.astype(np.float16), bias=b.astype(np.float16),
                   blocksize=blocksize, fp32_io=fp32_out)


@fixture("dts_multi_b2")
def _dts_multi_b2(out: Path) -> Fixture:
    """DTS blocksize 2 with two output channels, fp32 output (generic kernel)."""
    return _dts_multi_fixture(out, "dts_multi_b2", blocksize=2, cout=2, fp32_out=True)


@fixture("dts_multi_b2_f16")
def _dts_multi_b2_f16(out: Path) -> Fixture:
    """DTS blocksize 2 with two output channels, fp16 output (NCHW boundary)."""
    return _dts_multi_fixture(out, "dts_multi_b2_f16", blocksize=2, cout=2, fp32_out=False)


@fixture("dts_multi_b3")
def _dts_multi_b3(out: Path) -> Fixture:
    """DTS blocksize 3 with two output channels, fp32 output."""
    return _dts_multi_fixture(out, "dts_multi_b3", blocksize=3, cout=2, fp32_out=True)


# ---------------------------------------------------------------------------
# Fusion with shared intermediates (P2-13)
# ---------------------------------------------------------------------------


@fixture("silu_shared")
def _silu_shared(out: Path) -> Fixture:
    """raw = conv(x); z = raw * sigmoid(raw); y = raw + z.

    The conv output has consumers besides the SiLU pair, so the SiLU fusion must
    be declined; fusing it would delete ``raw`` and break the Add.
    """
    w = _kern(63, 4, 4)
    b = _linspace(4)
    n0 = _conv_node("x", "w", "b", "raw")
    n1 = helper.make_node("Sigmoid", ["raw"], ["sig"])
    n2 = helper.make_node("Mul", ["raw", "sig"], ["z"])
    n3 = helper.make_node("Add", ["raw", "z"], ["y"])
    model = _model([n0, n1, n2, n3], "x", [1, 4, "h", "w"], TensorProto.FLOAT16,
                   "y", [1, 4, "h", "w"], TensorProto.FLOAT16, [_half("w", w), _half("b", b)])
    return Fixture(name="silu_shared", model=_save(model, out / "silu_shared.onnx"),
                   input_channels=4, output_channels=4,
                   weights=w.astype(np.float16), bias=b.astype(np.float16))


@fixture("add_shared_conv_out")
def _add_shared_conv_out(out: Path) -> Fixture:
    """a = conv(x); bb = conv(a); s = a + bb; y = bb * s.

    The conv producing ``bb`` feeds both the residual Add and the Mul, so the
    Add must not be fused into it (that would rename ``bb`` away).
    """
    w = (_kern(64, 4, 4) * 0.2).astype(np.float32)  # keep the reference in fp16 range
    b = _linspace(4)
    n0 = _conv_node("x", "w", "b", "a")
    n1 = _conv_node("a", "w", "b", "bb")
    n2 = helper.make_node("Add", ["a", "bb"], ["s"])
    n3 = helper.make_node("Mul", ["bb", "s"], ["y"])
    model = _model([n0, n1, n2, n3], "x", [1, 4, "h", "w"], TensorProto.FLOAT16,
                   "y", [1, 4, "h", "w"], TensorProto.FLOAT16, [_half("w", w), _half("b", b)])
    return Fixture(name="add_shared_conv_out", model=_save(model, out / "add_shared_conv_out.onnx"),
                   input_channels=4, output_channels=4,
                   weights=w.astype(np.float16), bias=b.astype(np.float16))


# ---------------------------------------------------------------------------
# Conv padding / Clip bounds (P2-14)
# ---------------------------------------------------------------------------


@fixture("conv_pads_asym")
def _conv_pads_asym(out: Path) -> Fixture:
    """A 3x3 Conv padded [top=1, left=0, bottom=1, right=0].

    Reading pads[0] and pads[2] as (h, w) silently treated this as full
    padding; ONNX semantics make it a different, smaller-output convolution.
    """
    return _single_conv(out, "conv_pads_asym", m=4, c=4, seed=33, bias="none",
                        pads=(1, 0, 1, 0))


@fixture("clip_no_bounds")
def _clip_no_bounds(out: Path) -> Fixture:
    """Conv -> Clip with no bounds: ONNX means the type's extrema, so the clip
    must be a no-op rather than clamping to [0, 1]."""
    w = _kern(34, 4, 4)
    b = np.full((4,), 2.0, dtype=np.float32)  # push the output well above 1.0
    n0 = _conv_node("x", "w", "b", "raw")
    n1 = helper.make_node("Clip", ["raw"], ["y"])
    model = _model([n0, n1], "x", [1, 4, "h", "w"], TensorProto.FLOAT16,
                   "y", [1, 4, "h", "w"], TensorProto.FLOAT16, [_half("w", w), _half("b", b)])
    return Fixture(name="clip_no_bounds", model=_save(model, out / "clip_no_bounds.onnx"),
                   input_channels=4, output_channels=4,
                   weights=w.astype(np.float16), bias=b.astype(np.float16))


@fixture("clip_dynamic_bound")
def _clip_dynamic_bound(out: Path) -> Fixture:
    """Clip whose lower bound is an activation, not an initializer."""
    w = _kern(35, 4, 4)
    n0 = _conv_node("x", "w", None, "raw")
    n1 = helper.make_node("Clip", ["raw", "raw"], ["y"])
    model = _model([n0, n1], "x", [1, 4, "h", "w"], TensorProto.FLOAT16,
                   "y", [1, 4, "h", "w"], TensorProto.FLOAT16, [_half("w", w)])
    return Fixture(name="clip_dynamic_bound", model=_save(model, out / "clip_dynamic_bound.onnx"),
                   input_channels=4, output_channels=4, weights=w.astype(np.float16))


# ---------------------------------------------------------------------------
# Execution-provider partition boundaries (P2-17)
# ---------------------------------------------------------------------------


@fixture("ep_two_inputs")
def _ep_two_inputs(out: Path) -> Fixture:
    """Two activation inputs added together: the boundary cannot be represented
    by the EP's single-activation-input compiled graph."""
    x1 = _vi("x1", [1, 4, "h", "w"], TensorProto.FLOAT16)
    x2 = _vi("x2", [1, 4, "h", "w"], TensorProto.FLOAT16)
    graph = helper.make_graph(
        [helper.make_node("Add", ["x1", "x2"], ["y"])],
        "fixture", [x1, x2],
        [_vi("y", [1, 4, "h", "w"], TensorProto.FLOAT16)], [],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)])
    return Fixture(name="ep_two_inputs", model=_save(model, out / "ep_two_inputs.onnx"),
                   input_channels=4, output_channels=4)


@fixture("ep_two_convs_one_input")
def _ep_two_convs_one_input(out: Path) -> Fixture:
    """One graph input feeding two convs whose outputs are added.

    The external input list must be deduplicated, otherwise the run looks like a
    two-input region and is (correctly) refused.
    """
    w1 = _kern(81, 4, 4)
    w2 = _kern(82, 4, 4)
    b1 = _linspace(4)
    b2 = _linspace(4, -0.2, 0.2)
    n0 = _conv_node("x", "w1", "b1", "ca")
    n1 = _conv_node("x", "w2", "b2", "cb")
    n2 = helper.make_node("Add", ["ca", "cb"], ["y"])
    model = _model([n0, n1, n2], "x", [1, 4, "h", "w"], TensorProto.FLOAT16,
                   "y", [1, 4, "h", "w"], TensorProto.FLOAT16,
                   [_half("w1", w1), _half("b1", b1), _half("w2", w2), _half("b2", b2)])
    return Fixture(name="ep_two_convs_one_input",
                   model=_save(model, out / "ep_two_convs_one_input.onnx"),
                   input_channels=4, output_channels=4, weights=w1.astype(np.float16),
                   bias=b1.astype(np.float16))


@fixture("identity_fp32io")
def _identity_fp32io(out: Path) -> Fixture:
    """fp32 graph IO around an fp16 identity conv (P2-12).

    The plugin must widen an integer or half clip to fp32 for this model;
    writing the packed half values into the four-byte staging buffer produced
    ~3e-5 instead of ~0.502 for a GRAY8 frame of 128.
    """
    ident = np.zeros((1, 1, 3, 3), dtype=np.float32)
    ident[0, 0, 1, 1] = 1.0
    n0 = helper.make_node("Cast", ["x"], ["xh"], to=TensorProto.FLOAT16)
    n1 = _conv_node("xh", "w", None, "raw")
    n2 = helper.make_node("Cast", ["raw"], ["y"], to=TensorProto.FLOAT)
    model = _model([n0, n1, n2], "x", [1, 1, "h", "w"], TensorProto.FLOAT,
                   "y", [1, 1, "h", "w"], TensorProto.FLOAT, [_half("w", ident)])
    return Fixture(name="identity_fp32io", model=_save(model, out / "identity_fp32io.onnx"),
                   input_channels=1, output_channels=1, weights=ident.astype(np.float16),
                   fp32_io=True)


# ---------------------------------------------------------------------------
# Execution-provider fixture (P1-4)
# ---------------------------------------------------------------------------


@fixture("ep_conv_fp32io")
def _ep_conv_fp32io(out: Path) -> Fixture:
    """fp32 graph IO with fp16 compute: Cast -> Conv -> Cast, the shape the
    converter emits for an fp32 clip.

    One input AND one output channel on purpose: the EP's supported layout is
    single-activation (REVIEW.md P1-17) and its multi-channel buffer conversion
    is a separate open finding (P2-11). Keeping both at 1 isolates the fp16->fp32
    boundary conversion, which is what the P1-4 checks are about.
    """
    w = _kern(5, 1, 1)
    b = _linspace(1)
    n0 = helper.make_node("Cast", ["x"], ["xh"], to=TensorProto.FLOAT16)
    n1 = _conv_node("xh", "w", "b", "raw")
    n2 = helper.make_node("Cast", ["raw"], ["y"], to=TensorProto.FLOAT)
    model = _model([n0, n1, n2], "x", [1, 1, "h", "w"], TensorProto.FLOAT,
                   "y", [1, 1, "h", "w"], TensorProto.FLOAT, [_half("w", w), _half("b", b)])
    return Fixture(name="ep_conv_fp32io", model=_save(model, out / "ep_conv_fp32io.onnx"),
                   input_channels=1, output_channels=1,
                   weights=w.astype(np.float16), bias=b.astype(np.float16), fp32_io=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def build_all(out_dir: Path) -> dict[str, dict]:
    """Generate every registered fixture and write the manifest."""
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, dict] = {}
    for name in sorted(_REGISTRY):
        fx = _REGISTRY[name](out_dir)
        entry: dict = {
            "model": fx.model.name,
            "input_channels": fx.input_channels,
            "output_channels": fx.output_channels,
            "blocksize": fx.blocksize,
            "fp32_io": fx.fp32_io,
        }
        if fx.weights is not None:
            rel = f"{name}.w.npy"
            np.save(out_dir / rel, fx.weights)
            entry["weights"] = rel
        if fx.bias is not None:
            rel = f"{name}.b.npy"
            np.save(out_dir / rel, fx.bias)
            entry["bias"] = rel
        manifest[name] = entry
    (out_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print("usage: python -m harness.modelgen <output_dir>", file=sys.stderr)
        return 2
    out_dir = Path(argv[0])
    manifest = build_all(out_dir)
    print(f"[correctness] generated {len(manifest)} fixtures in {out_dir}")
    for name in sorted(manifest):
        print(f"  {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
