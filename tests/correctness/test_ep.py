"""HIP execution-provider checks.

Covers P1-4 (the fp16->fp32 boundary conversion), P1-6 (device wiring
and capability refusal after a failed context init) and the dispatch regression
that the checked launch status exposes.

The fixture keeps one input channel: the EP's multi-channel input/output layout
is a separate, still-open finding (P2-11), so these checks stay inside the
behaviour the EP claims to support.
"""

from __future__ import annotations

import numpy as np
import pytest

from harness import reference as ref

ort = pytest.importorskip("onnxruntime", reason="onnxruntime is required for the EP suite")

# The prebuilt ORT wheel does not list out-of-tree providers, so the Python API
# warns on every session even though registration succeeded.
pytestmark = pytest.mark.filterwarnings("ignore:.*not in available provider names.*")

PROVIDER = "HIPExecutionProvider"


def _session(model, providers=None):
    options = ort.SessionOptions()
    options.log_severity_level = 3
    return ort.InferenceSession(str(model), options, providers=providers or [PROVIDER])


def _hip_only_session(model):
    """Build a session that succeeds only when the HIP EP claims every node.

    ``get_providers()`` still lists the HIP EP on a graph it claimed nothing of,
    so capability is observed by disabling the CPU fallback: an unclaimed node
    then makes session creation fail with an explicit error.
    """
    options = ort.SessionOptions()
    options.log_severity_level = 3
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    return ort.InferenceSession(str(model), options, providers=[PROVIDER])


@pytest.fixture(scope="module")
def hip_session(ep_library, fixtures):
    """A HIP-EP session over the fp32-boundary conv fixture."""
    return _session(fixtures.get("ep_conv_fp32io").model)


@pytest.fixture(scope="module")
def ep_case(fixtures):
    """The fixture, its fp32 NCHW input, and the float64 reference output."""
    fx = fixtures.get("ep_conv_fp32io")
    h = w = 16
    x = ref.raw_input(fx.input_channels, h, w, in_fp32=True).reshape(1, fx.input_channels, h, w)
    expected = ref.conv3x3(x.astype(np.float16), fx.weights, fx.bias)
    return fx, x, expected


def test_p1_04_hip_provider_claims_the_graph(hip_session):
    assert hip_session.get_providers()[0] == PROVIDER, hip_session.get_providers()


def test_p1_04_fp32_output_conversion_is_not_aliased(hip_session, ep_case):
    """The fp16->fp32 write must not read half values that the wider float
    writes have already overwritten. Pre-fix this produced ~512.0 of error and
    corrupted every element after the first."""
    fx, x, expected = ep_case
    got = hip_session.run(None, {"x": x})[0].astype(np.float64)
    assert got.shape == expected.shape
    ref.assert_close(got, expected, ref.FP16_CONV_TOL, "EP fp32 output")

    # The aliasing damaged everything from the second element onwards, so check
    # that region explicitly for a failure message that names the cause.
    first = got.reshape(-1)[:4]
    first_ref = expected.reshape(-1)[:4]
    assert np.allclose(first, first_ref, atol=ref.FP16_CONV_TOL), (
        f"leading output elements look aliased-corrupted: got {first} want {first_ref}"
    )


def test_p1_06_invalid_device_refuses_capability(ep_library, fixtures, ep_case):
    """An unusable device must not be advertised: ORT falls back to another
    provider instead of dereferencing an uninitialized HIP context."""
    fx, x, expected = ep_case
    session = _session(fx.model,
                       providers=[(PROVIDER, {"device_id": "999"}), "CPUExecutionProvider"])
    assert PROVIDER not in session.get_providers(), session.get_providers()
    got = session.run(None, {"x": x})[0].astype(np.float64)
    ref.assert_close(got, expected, ref.FP16_CONV_TOL, "CPU fallback output")


def test_ep_rebuilds_for_a_new_input_shape(hip_session, ep_case):
    """A second, smaller shape must re-plan its dispatch instead of reusing the
    first plan. W=4 used to schedule a zero-sized grid and fail to launch."""
    fx, x, _ = ep_case
    small = np.ascontiguousarray(x[:, :, :4, :4])
    got = hip_session.run(None, {"x": small})[0].astype(np.float64)
    expected = ref.conv3x3(small.astype(np.float16), fx.weights, fx.bias)
    assert got.shape == (1, fx.output_channels, 4, 4)
    ref.assert_close(got, expected, ref.FP16_CONV_TOL, "EP 4x4 rebuild")


def test_p2_11_multichannel_output_layout_is_converted(ep_library, fixtures):
    """P2-11: the EP's download path must transpose the NHWC compute
    output into ORT's NCHW tensor. C=1 input (identical layout) with M=8 output
    isolates the output conversion.

    This check was a strict xfail while the layout conversion was missing; it
    XPASSed (failing the run) as soon as the conversion was added.
    """
    fx = fixtures.get("conv_c1m8")  # C=1 (input layout identical), M=8 output
    h = w = 16
    x = ref.raw_input(1, h, w, in_fp32=False).reshape(1, 1, h, w)
    session = _session(fx.model)
    got = session.run(None, {"x": x})[0].astype(np.float64)
    expected = ref.conv3x3(x.astype(np.float16), fx.weights, fx.bias)
    ref.assert_close(got, expected, ref.FP16_CONV_TOL, "EP multi-channel output layout")


def test_p2_11_multichannel_input_layout_is_converted(ep_library, fixtures):
    """The upload path needs the same NCHW -> NHWC conversion: three input
    channels with a single output channel."""
    fx = fixtures.get("conv_c4m4")
    h = w = 16
    x = ref.raw_input(4, h, w, in_fp32=False).reshape(1, 4, h, w)
    session = _session(fx.model)
    got = session.run(None, {"x": x})[0].astype(np.float64)
    expected = ref.conv3x3(x.astype(np.float16), fx.weights, fx.bias)
    ref.assert_close(got, expected, ref.FP16_CONV_TOL, "EP multi-channel input layout")


# ---------------------------------------------------------------------------
# P2-17: partition boundaries must fit the single-input / single-output engine
# ---------------------------------------------------------------------------


def test_p2_17_two_activation_inputs_are_not_claimed(ep_library, fixtures):
    """A region with two activation inputs cannot be represented by the compiled
    graph (which reads input 0), so the EP must refuse it and let ORT fall back
    instead of running it with one input silently dropped."""
    fx = fixtures.get("ep_two_inputs")
    with pytest.raises(Exception, match="fallback"):
        _hip_only_session(fx.model)

    session = _session(fx.model, providers=[PROVIDER, "CPUExecutionProvider"])
    h = w = 8
    x1 = ref.raw_input(4, h, w, in_fp32=False).reshape(1, 4, h, w)
    x2 = np.ascontiguousarray(x1 + np.float16(0.25))
    got = session.run(None, {"x1": x1, "x2": x2})[0].astype(np.float64)
    expected = x1.astype(np.float64) + x2.astype(np.float64)
    ref.assert_close(got, expected, 1e-3, "fallback result for a two-input region")


def test_p2_17_shared_graph_input_is_deduplicated(ep_library, fixtures):
    """One graph input feeding two convs is a single-activation-input region;
    appending it twice used to make the run look unsupported."""
    fx = fixtures.get("ep_two_convs_one_input")
    h = w = 8
    x = ref.raw_input(4, h, w, in_fp32=False).reshape(1, 4, h, w)

    hip_only = _hip_only_session(fx.model)  # succeeds only if the EP claims it
    got = hip_only.run(None, {"x": x})[0].astype(np.float64)

    # Independent reference from the same model on the CPU provider.
    cpu = _session(fx.model, providers=["CPUExecutionProvider"])
    expected = cpu.run(None, {"x": x})[0].astype(np.float64)
    assert got.shape == expected.shape
    ref.assert_close(got, expected, ref.FP16_CONV_TOL, "deduplicated shared input")

