"""Standalone ``HipEngine`` correctness checks.

Test names carry the ``REVIEW.md`` finding they cover (``p1_03`` = finding 3) so
a failure points straight back at the review item that motivated the check.
"""

from __future__ import annotations

import numpy as np
import pytest

from harness import reference as ref
from harness.engine import EngineHarnessError

# Odd sizes, sizes that are not multiples of 8/16, and the two cases called out
# by the review: a 17-wide direct conv and a 1082-row winograd band. Before the
# dispatch fix these lost whole rows/columns (max diff ~3.0).
GEOMETRY_SIZES = [
    (8, 8),
    (8, 17),
    (8, 24),
    (8, 25),
    (17, 16),
    (17, 17),
    (33, 17),
    (16, 33),
    (33, 33),
    (64, 64),
    (1082, 64),
    (54, 1920),
]


def _check_conv(engine, fx, h, w, device, tol=ref.FP16_CONV_TOL, label="") -> float:
    """Run a conv fixture and compare against the float64 reference."""
    result = engine.run(fx.model, h=h, w=w, c=fx.input_channels, device=device)
    logical = ref.logical_input(fx.input_channels, h, w, in_fp32=False)
    expected = ref.conv3x3(logical, fx.weights, fx.bias)
    assert result.shape == (1, fx.output_channels, h, w), f"{label}: unexpected output shape"
    return ref.assert_close(result.output, expected, tol, label or f"{fx.name} {h}x{w}")


def _expect_rejected(engine, fx, *, c, match, device):
    with pytest.raises(EngineHarnessError, match=match):
        engine.probe(fx.model, h=16, w=16, c=c, device=device)


# ---------------------------------------------------------------------------
# P1-3: dispatch geometry / partial writes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("h,w", GEOMETRY_SIZES)
def test_p1_03_direct_conv_covers_geometry(engine, fixtures, device, h, w):
    """The direct (C=1) path must write every output pixel."""
    _check_conv(engine, fixtures.get("conv_c1m8"), h, w, device)


@pytest.mark.parametrize("h,w", GEOMETRY_SIZES)
def test_p1_03_winograd_conv_covers_geometry(engine, fixtures, device, h, w):
    """The winograd path must schedule the final partial tile band."""
    _check_conv(engine, fixtures.get("conv_c4m4"), h, w, device)


@pytest.mark.parametrize("name", ["conv_c4m8", "conv_c4m64"])
def test_p1_03_conv_channel_counts(engine, fixtures, device, name):
    """Small and maximum supported output-channel counts."""
    _check_conv(engine, fixtures.get(name), 16, 16, device)


def test_p1_03_batch_greater_than_one_rejected(engine, fixtures, device):
    """The kernels have no batch dimension, so N>1 must be refused."""
    fx = fixtures.get("conv_c4m4")
    with pytest.raises(EngineHarnessError, match="shape propagation failed"):
        engine.probe(fx.model, n=2, h=16, w=16, c=4, device=device)


def test_p1_03_excessive_output_channels_rejected(engine, fixtures, device):
    """M>64 exceeds what the four-wave kernels cover and must be refused."""
    _expect_rejected(engine, fixtures.get("conv_m128"), c=4,
                     match="64 output channels", device=device)


# ---------------------------------------------------------------------------
# P1-2: weight / bias / external-data validation
# ---------------------------------------------------------------------------


def test_p1_02_non_3x3_weight_rejected(engine, fixtures, device):
    """A legal 1x1 Conv must not reach the fixed nine-tap packing loop."""
    _expect_rejected(engine, fixtures.get("conv_1x1"), c=4, match="3x3", device=device)


def test_p1_02_short_bias_rejected(engine, fixtures, device):
    _expect_rejected(engine, fixtures.get("conv_short_bias"), c=4,
                     match="bias", device=device)


def test_p1_02_external_data_rejected(engine, fixtures, device):
    _expect_rejected(engine, fixtures.get("conv_external"), c=4,
                     match="external data", device=device)


# ---------------------------------------------------------------------------
# P1-7: typed fp16 initializers
# ---------------------------------------------------------------------------


def test_p1_07_typed_half_weights_are_bit_patterns(engine, fixtures, device):
    """FLOAT16 values in int32_data must be unpacked as bit patterns, not read
    as numeric integers. The typed fixture and the raw fixture are the same
    kernel, so they must agree exactly with each other and with the input."""
    h, w = 17, 8
    raw = engine.run(fixtures.get("identity_raw").model, h=h, w=w, c=1, device=device)
    typed = engine.run(fixtures.get("identity_typed_half").model, h=h, w=w, c=1, device=device)
    expected = ref.logical_input(1, h, w, in_fp32=False)
    ref.assert_close(raw.output, expected, 1e-6, "raw fp16 identity")
    ref.assert_close(typed.output, expected, 1e-6, "typed fp16 identity")
    np.testing.assert_array_equal(raw.output, typed.output)


# ---------------------------------------------------------------------------
# P1-8: constant / broadcast binary operands
# ---------------------------------------------------------------------------


def test_p1_08_initializer_operand_rejected(engine, fixtures, device):
    """A checked model computing x + 0.25 must be refused, not run with a null
    device buffer."""
    _expect_rejected(engine, fixtures.get("add_const"), c=4,
                     match="not an activation produced", device=device)


def test_p1_08_broadcast_operand_rejected(engine, fixtures, device):
    """Add of a 4-channel and a 1-channel activation is legal ONNX but is not
    implemented; it must be refused rather than read out of bounds."""
    _expect_rejected(engine, fixtures.get("add_broadcast"), c=4,
                     match="shape propagation failed", device=device)


# ---------------------------------------------------------------------------
# P1-9: DepthToSpace block size
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("blocksize", [1, 2, 3])
def test_p1_09_depth_to_space_blocksize(engine, fixtures, device, blocksize):
    """Blocksize 2 uses the fused fp32 fast path; 1 and 3 must fall back to the
    generic kernel. All three must match the DCR reference."""
    fx = fixtures.get(f"dts_b{blocksize}")
    h = w = 16
    result = engine.run(fx.model, h=h, w=w, c=1, device=device)
    conv_out = ref.conv3x3(ref.logical_input(1, h, w, in_fp32=False), fx.weights, fx.bias)
    expected = ref.depth_to_space_dcr(conv_out, blocksize)
    assert result.shape == (1, 1, h * blocksize, w * blocksize)
    ref.assert_close(result.output, expected, ref.FP16_CONV_TOL, f"DepthToSpace b={blocksize}")


# ---------------------------------------------------------------------------
# P1-6: device selection
# ---------------------------------------------------------------------------


def test_p1_06_selected_device_is_used(engine, fixtures, device):
    """A supported device id must build, propagate and run."""
    fx = fixtures.get("conv_c4m4")
    probe = engine.probe(fx.model, h=16, w=16, c=4, device=device)
    assert (probe.in_channels, probe.out_channels) == (4, 4)
    assert not probe.in_fp32 and not probe.out_fp32
    _check_conv(engine, fx, 16, 16, device)


def test_p1_06_invalid_device_is_reported(engine, fixtures, device):
    """An out-of-range device must fail with the HIP error, not run on device 0."""
    with pytest.raises(EngineHarnessError, match="hipSetDevice|invalid device"):
        engine.probe(fixtures.get("conv_c4m4").model, h=16, w=16, c=4, device=999)


# ---------------------------------------------------------------------------
# P1-5: reported failures and recovery
# ---------------------------------------------------------------------------


def test_p1_05_allocation_failure_is_reported_then_recovers(engine, fixtures, device):
    """A failing allocation must surface the HIP error, and the same engine must
    still serve a valid frame afterwards (partial state unwound)."""
    fx = fixtures.get("conv_c4m4")
    message, result = engine.alloc_failure_then_run(fx.model, c=4, device=device)
    assert "hipMalloc" in message, f"unexpected failure message: {message!r}"
    expected = ref.conv3x3(ref.logical_input(4, 16, 16, in_fp32=False), fx.weights, fx.bias)
    ref.assert_close(result.output, expected, ref.FP16_CONV_TOL,
                     "recovery run after an allocation failure")
