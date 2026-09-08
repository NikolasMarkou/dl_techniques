"""Contract and numerical-parity suite for :class:`GatedDConvFeedForward`.

The centrepiece is `_gdfn_reference`, an independent NumPy transcription of
Restormer's PyTorch GDFN (`models/restormer_arch.py:76-93`). It was written
from the upstream spec -- `hidden = int(dim * factor)`, `project_in` to
`2*hidden`, a fully depthwise 3x3, `chunk(2, dim=1)` with the GATE first, and
`project_out` from `hidden` -- and NOT from the Keras layer, because a
reference derived from the subject is a second copy, not an oracle.

The reference carries two deliberate MUTATION KNOBS (`gate_chunk`,
`gelu_form`). Each is one of the choices that shape, serialization and
gradient tests are all structurally blind to, and each has a guard below
asserting that the layer is FAR from the mutant while being equal to the
correct reference. Those two halves together are what makes the parity test
non-vacuous: agreeing with a reference is only evidence if disagreeing with
its neighbours is also demonstrated.

The third invisible choice, the `int()` truncation of the hidden width, cannot
be caught numerically -- a wrong width still runs. It is caught structurally
instead, by reading the BUILT weight shapes and comparing them against
hard-coded literals (`test_hidden_width_truncates`). Recomputing
`int(dim * 2.66)` on the assertion side would test nothing at all.

TF32 is scoped OFF for this module (`tests/test_layers/conftest.py`'s
`tf32_disabled`). The parity bound is derived from `eps_f32 = 1.19e-07`; one
TF32 ulp is ~4100x larger, so on a TF32-capable GPU the bound would be
unattainable and every parity assertion here would be measuring the toggle
instead of the layer.
"""

import keras
import numpy as np
import pytest
from scipy.special import erf

from dl_techniques.layers.ffn.gated_dconv_ffn import GatedDConvFeedForward
from tests.numerics import reassociation_atol
from tests.test_models.test_sam.dead_component_oracle import (
    fit_one_step_moved_variables,
)

pytestmark = pytest.mark.usefixtures("tf32_disabled")

# ---------------------------------------------------------------------
# Fixed parity subject. `dim=8` with the shipped factor gives
# `int(8 * 2.66) == 21`, an ODD hidden width -- deliberately, because an even
# one would let a `2*hidden`-vs-`hidden` slip still split cleanly.
# ---------------------------------------------------------------------

_DIM = 8
_FACTOR = 2.66
_HIDDEN = int(_DIM * _FACTOR)  # 21
_HEIGHT = 4
_WIDTH = 5

# The four widths DocRes actually instantiates, with the hidden width each one
# must truncate to. These literals come from D-008's measurement, not from
# re-evaluating the expression the layer evaluates.
_DOCRES_WIDTHS = [(48, 127), (96, 255), (192, 510), (384, 1021)]

# ---------------------------------------------------------------------
# NumPy reference
# ---------------------------------------------------------------------


def _conv1x1(x: np.ndarray, kernel: np.ndarray, bias) -> np.ndarray:
    """Apply a 1x1 convolution given a Keras `(1, 1, C_in, C_out)` kernel."""
    y = np.einsum("bhwi,io->bhwo", x, kernel[0, 0])
    return y if bias is None else y + bias


def _depthwise3x3(x: np.ndarray, kernel: np.ndarray, bias) -> np.ndarray:
    """Apply a fully depthwise 3x3 'same' convolution (Keras cross-correlation).

    `kernel` has the Keras `DepthwiseConv2D` shape `(3, 3, C, 1)` with
    `depth_multiplier=1`, which is the upstream `Conv2d(C, C, 3, groups=C)`:
    output channel `c` sees only input channel `c`.
    """
    padded = np.pad(x, ((0, 0), (1, 1), (1, 1), (0, 0)))
    out = np.zeros_like(x)
    for u in range(3):
        for v in range(3):
            out += (
                padded[:, u:u + x.shape[1], v:v + x.shape[2], :]
                * kernel[u, v, :, 0]
            )
    return out if bias is None else out + bias


def _gelu(x: np.ndarray, form: str) -> np.ndarray:
    """GELU in either the exact/erf form (correct) or the tanh approximation.

    D-008 measured the two forms `4.12e-04` apart over the probed range, three
    orders above this module's parity bound, so the choice is discriminable.
    """
    if form == "erf":
        return 0.5 * x * (1.0 + erf(x / np.sqrt(2.0)))
    if form == "tanh":
        inner = np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)
        return 0.5 * x * (1.0 + np.tanh(inner))
    raise ValueError(f"unknown gelu form {form!r}")


def _gdfn_reference(
    x: np.ndarray,
    *,
    w_in: np.ndarray,
    b_in,
    w_dw: np.ndarray,
    b_dw,
    w_out: np.ndarray,
    b_out,
    gate_chunk: str = "first",
    gelu_form: str = "erf",
) -> np.ndarray:
    """Independent NumPy GDFN, in float64, from the PyTorch spec.

    :param x: Input, shape `(B, H, W, dim)`.
    :param gate_chunk: `"first"` (correct: `gelu(x1) * x2`) or `"second"`
        (mutant: `x1 * gelu(x2)`).
    :param gelu_form: `"erf"` (correct, exact) or `"tanh"` (mutant).
    :return: Output, shape `(B, H, W, dim)`.
    """
    x = np.asarray(x, dtype=np.float64)

    h = _conv1x1(x, w_in, b_in)
    h = _depthwise3x3(h, w_dw, b_dw)

    x1, x2 = np.split(h, 2, axis=-1)
    if gate_chunk == "first":
        gated = _gelu(x1, gelu_form) * x2
    elif gate_chunk == "second":
        gated = x1 * _gelu(x2, gelu_form)
    else:
        raise ValueError(f"unknown gate_chunk {gate_chunk!r}")

    return _conv1x1(gated, w_out, b_out)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _randomized_layer(use_bias: bool, seed: int = 0):
    """Build a GDFN layer and give every weight a distinct random value.

    Randomized so that a defect cannot hide behind a symmetric parameter.
    Values are read back through the same attribute accessors the reference is
    fed with, so no `set_weights` ordering is assumed.
    """
    rng = np.random.default_rng(seed)
    layer = GatedDConvFeedForward(
        dim=_DIM, ffn_expansion_factor=_FACTOR, use_bias=use_bias
    )
    layer.build((None, _HEIGHT, _WIDTH, _DIM))

    def fill(var, low=-0.5, high=0.5):
        var.assign(rng.uniform(low, high, size=var.shape).astype("float32"))

    fill(layer.project_in.kernel)
    fill(layer.dwconv.kernel)
    fill(layer.project_out.kernel)
    if use_bias:
        fill(layer.project_in.bias)
        fill(layer.dwconv.bias)
        fill(layer.project_out.bias)
    return layer


def _reference_kwargs(layer) -> dict:
    """Read the layer's weights into the reference's keyword arguments."""
    np_ = keras.ops.convert_to_numpy
    return dict(
        w_in=np_(layer.project_in.kernel).astype(np.float64),
        b_in=(
            np_(layer.project_in.bias).astype(np.float64)
            if layer.use_bias
            else None
        ),
        w_dw=np_(layer.dwconv.kernel).astype(np.float64),
        b_dw=(
            np_(layer.dwconv.bias).astype(np.float64)
            if layer.use_bias
            else None
        ),
        w_out=np_(layer.project_out.kernel).astype(np.float64),
        b_out=(
            np_(layer.project_out.bias).astype(np.float64)
            if layer.use_bias
            else None
        ),
    )


def _parity_atol(scale: float) -> float:
    """Derived float32 bound for the layer-vs-float64-reference comparison.

    Not a pasted `1e-6`. `tests.numerics.reassociation_atol` charges an
    8-sigma random walk of `u = eps_f32 / 2` over the rounded multiply-adds on
    the compared path; the contraction lengths below are exactly GDFN's, in
    call order:

    ==========================  ======================
    stage                       contraction length
    ==========================  ======================
    project_in 1x1 conv         `_DIM` (8)
    depthwise 3x3               9
    gating multiply             1
    project_out 1x1 conv        `_HIDDEN` (21)
    ==========================  ======================

    The helper already charges BOTH sides of the comparison (`2 * num_steps`),
    which is conservative here: only one side (the layer) runs in float32, the
    reference being float64.
    """
    return reassociation_atol(
        reduction_lengths=[_DIM, 9, 1, _HIDDEN],
        num_steps=1,
        scale=scale,
    )


def _fixed_input(seed: int = 7, batch: int = 2) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(-1.0, 1.0, size=(batch, _HEIGHT, _WIDTH, _DIM)).astype(
        "float32"
    )


# ---------------------------------------------------------------------
# 1. Numerical parity
# ---------------------------------------------------------------------


@pytest.mark.parametrize("use_bias", [False, True])
def test_matches_the_numpy_reference(use_bias):
    """The layer reproduces the independent NumPy GDFN at `rtol=0`."""
    layer = _randomized_layer(use_bias)
    x = _fixed_input()

    actual = keras.ops.convert_to_numpy(layer(x)).astype(np.float64)
    expected = _gdfn_reference(x, **_reference_kwargs(layer))

    assert actual.shape == expected.shape == (2, _HEIGHT, _WIDTH, _DIM)
    np.testing.assert_allclose(
        actual,
        expected,
        rtol=0.0,
        atol=_parity_atol(float(np.max(np.abs(expected)))),
    )


# ---------------------------------------------------------------------
# 2. Mutation guards -- the two choices a shape test cannot see
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "mutation,margin",
    [
        # MEASURED separations on this fixed subject, in multiples of the
        # derived parity bound: chunk swap 4.79e+04x, tanh GELU 21.7x. The
        # margins below sit under the measurement with headroom; they are NOT
        # a uniform pasted constant, because a uniform 100x would be vacuous
        # for the first mutation and UNREACHABLE for the second.
        #
        # The tanh separation is small for a reason worth stating: the two
        # GELU forms differ by at most 4.12e-04 (D-008) at their WORST point,
        # this layer feeds them mid-scale pre-activations, and `project_out`
        # then averages over 21 channels whose errors partly cancel. 21.7x the
        # float32 noise floor is still an unambiguous separation -- the parity
        # test passes at 1x against the erf reference -- but it is why the
        # guard must be calibrated rather than assumed generous.
        pytest.param(
            {"gate_chunk": "second"}, 100.0, id="gelu_on_the_second_chunk"
        ),
        pytest.param({"gelu_form": "tanh"}, 10.0, id="tanh_approximate_gelu"),
    ],
)
def test_the_layer_is_far_from_the_mutant(mutation, margin):
    """The layer must NOT match a reference built with the wrong choice.

    Each mutation is a plausible port slip that leaves every shape, config and
    serialization assertion in this file green. The separation is measured
    against the same derived tolerance the parity test passes at, not against
    zero.
    """
    layer = _randomized_layer(use_bias=False)
    x = _fixed_input()

    actual = keras.ops.convert_to_numpy(layer(x)).astype(np.float64)
    kwargs = _reference_kwargs(layer)
    correct = _gdfn_reference(x, **kwargs)
    mutant = _gdfn_reference(x, **kwargs, **mutation)

    atol = _parity_atol(float(np.max(np.abs(correct))))

    # Anti-vacuity: the mutant must actually differ from the correct
    # reference, or this guard could pass on two identical arrays.
    assert np.max(np.abs(mutant - correct)) > margin * atol, (
        f"mutation {mutation} did not change the reference; the guard would "
        "be vacuous"
    )
    assert np.max(np.abs(actual - mutant)) > margin * atol, (
        f"the layer matches the {mutation} mutant, i.e. it implements the "
        "wrong variant"
    )


# ---------------------------------------------------------------------
# 3. Hidden-width truncation
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "dim,expected_hidden",
    _DOCRES_WIDTHS,
    ids=[f"dim{d}" for d, _ in _DOCRES_WIDTHS],
)
def test_hidden_width_truncates(dim, expected_hidden):
    """`int(dim * 2.66)`, read off the BUILT weights, not recomputed.

    `expected_hidden` is a literal from D-008. Asserting against a freshly
    evaluated `int(dim * 2.66)` would evaluate the same expression on both
    sides of the comparison and could not fail.
    """
    layer = GatedDConvFeedForward(dim=dim, ffn_expansion_factor=2.66)
    layer.build((None, None, None, dim))

    assert layer.hidden_features == expected_hidden
    assert tuple(layer.project_in.kernel.shape) == (
        1, 1, dim, 2 * expected_hidden,
    )
    assert tuple(layer.dwconv.kernel.shape) == (3, 3, 2 * expected_hidden, 1)
    # The output projection reads `hidden`, NOT `2 * hidden`: the gate has
    # already halved the tensor.
    assert tuple(layer.project_out.kernel.shape) == (1, 1, expected_hidden, dim)


def test_the_width_parametrisation_can_discriminate_round_from_int():
    """Anti-vacuity for the test above: `round()` must disagree somewhere.

    `round(96 * 2.66) == 255` and `round(384 * 2.66) == 1021`, i.e. two of the
    four DocRes widths CANNOT tell a `round()`-instead-of-`int()` port defect
    from a correct one. The other two can, and this asserts they are in the
    parametrisation -- otherwise `test_hidden_width_truncates` would be a
    guard that a real defect walks straight past.
    """
    discriminating = [
        dim
        for dim, expected in _DOCRES_WIDTHS
        if round(dim * 2.66) != expected
    ]
    assert discriminating == [48, 192], (
        "the widths that separate round() from int() changed; "
        f"got {discriminating}"
    )
    assert round(48 * 2.66) == 128 and round(192 * 2.66) == 511


# ---------------------------------------------------------------------
# 4. The depthwise stage really is `groups == 2 * hidden`
# ---------------------------------------------------------------------


def test_the_dwconv_is_fully_depthwise():
    """`DepthwiseConv2D(3, depth_multiplier=1)` == upstream `groups=2*hidden`.

    Two independent readings: the weight count (`3 * 3 * 2 * hidden * 1`, i.e.
    no cross-channel term) and per-channel independence measured by perturbing
    one input channel and observing which outputs move.
    """
    layer = _randomized_layer(use_bias=False)
    kernel = keras.ops.convert_to_numpy(layer.dwconv.kernel)

    assert kernel.shape == (3, 3, 2 * _HIDDEN, 1)
    assert layer.dwconv.count_params() == 3 * 3 * 2 * _HIDDEN

    dw = layer.dwconv
    base = np.zeros((1, _HEIGHT, _WIDTH, 2 * _HIDDEN), dtype="float32")
    y0 = keras.ops.convert_to_numpy(dw(base))
    for probe_channel in (0, 5, 2 * _HIDDEN - 1):
        perturbed = base.copy()
        perturbed[0, 1, 1, probe_channel] = 1.0
        y1 = keras.ops.convert_to_numpy(dw(perturbed))
        moved = np.nonzero(np.abs(y1 - y0).max(axis=(0, 1, 2)) > 0.0)[0]
        assert moved.tolist() == [probe_channel], (
            f"perturbing input channel {probe_channel} moved output channels "
            f"{moved.tolist()}; the convolution is not fully depthwise"
        )


# ---------------------------------------------------------------------
# 5. Serialization
# ---------------------------------------------------------------------


def test_serialization_round_trip_preserves_outputs(tmp_path):
    layer = _randomized_layer(use_bias=True, seed=11)
    inputs = keras.Input(shape=(_HEIGHT, _WIDTH, _DIM))
    model = keras.Model(inputs, layer(inputs))

    x = _fixed_input(seed=13)
    original = keras.ops.convert_to_numpy(model(x, training=False))

    path = tmp_path / "gdfn_model.keras"
    model.save(path)
    loaded = keras.models.load_model(path)
    restored = keras.ops.convert_to_numpy(loaded(x, training=False))

    np.testing.assert_allclose(restored, original, rtol=0.0, atol=0.0)


def test_weights_survive_a_round_trip_before_the_first_call(tmp_path):
    """Weight VALUES compare at `atol=0.0` BEFORE the loaded model is called.

    Comparing after a call would let a re-initialized variable be masked by a
    lazy rebuild; comparing at `atol=0.0` is what makes this a byte-level
    check rather than a numerical one.
    """
    layer = _randomized_layer(use_bias=True, seed=3)
    inputs = keras.Input(shape=(_HEIGHT, _WIDTH, _DIM))
    model = keras.Model(inputs, layer(inputs))

    before = [
        keras.ops.convert_to_numpy(w)
        for w in (
            layer.project_in.kernel,
            layer.dwconv.kernel,
            layer.project_out.kernel,
        )
    ]

    path = tmp_path / "gdfn.keras"
    model.save(path)
    loaded = keras.models.load_model(path)

    restored_layer = next(
        sub for sub in loaded.layers if isinstance(sub, GatedDConvFeedForward)
    )
    after = [
        keras.ops.convert_to_numpy(w)
        for w in (
            restored_layer.project_in.kernel,
            restored_layer.dwconv.kernel,
            restored_layer.project_out.kernel,
        )
    ]
    for lhs, rhs in zip(before, after):
        np.testing.assert_allclose(rhs, lhs, rtol=0.0, atol=0.0)


def test_get_config_round_trips_every_constructor_argument():
    layer = GatedDConvFeedForward(
        dim=16, ffn_expansion_factor=1.5, use_bias=True, name="gdfn_cfg"
    )
    config = layer.get_config()
    assert config["dim"] == 16
    assert config["ffn_expansion_factor"] == 1.5
    assert config["use_bias"] is True

    clone = GatedDConvFeedForward.from_config(config)
    assert (clone.dim, clone.ffn_expansion_factor, clone.use_bias) == (
        16, 1.5, True,
    )
    assert clone.hidden_features == 24


def test_compute_output_shape_matches_a_real_call():
    layer = GatedDConvFeedForward(dim=_DIM, ffn_expansion_factor=_FACTOR)
    declared = layer.compute_output_shape((None, _HEIGHT, _WIDTH, _DIM))
    actual = keras.ops.convert_to_numpy(layer(_fixed_input())).shape
    assert declared == (None, _HEIGHT, _WIDTH, _DIM)
    assert actual[1:] == declared[1:]


# ---------------------------------------------------------------------
# 6. Gradient flow after ONE real optimizer step
# ---------------------------------------------------------------------


def test_every_trainable_variable_moves_after_one_step():
    """Reuses the shared instrument; it reports moved variables BY NAME."""
    layer = GatedDConvFeedForward(
        dim=_DIM, ffn_expansion_factor=_FACTOR, use_bias=True
    )
    inputs = keras.Input(shape=(_HEIGHT, _WIDTH, _DIM))
    model = keras.Model(inputs, layer(inputs))
    model.compile(optimizer=keras.optimizers.Adam(1e-2), loss="mse")

    x = _fixed_input(seed=17, batch=4)
    y = _fixed_input(seed=19, batch=4)

    report = fit_one_step_moved_variables(model, x, y, batch_size=4)
    assert report.unmoved == (), (
        f"variables did not move under a real optimizer step: {report}"
    )
    assert any("dwconv" in name for name in report.moved)


# ---------------------------------------------------------------------
# 7. Dynamic spatial extents
# ---------------------------------------------------------------------


def test_builds_with_unknown_spatial_dims_and_runs_at_two_sizes():
    """DocRes runs at arbitrary page sizes; `H`/`W` are unknown at build time."""
    layer = GatedDConvFeedForward(dim=_DIM, ffn_expansion_factor=_FACTOR)
    inputs = keras.Input(shape=(None, None, _DIM))
    model = keras.Model(inputs, layer(inputs))
    assert model.output_shape == (None, None, None, _DIM)

    rng = np.random.default_rng(23)
    for height, width in ((4, 5), (9, 3)):
        x = rng.uniform(-1.0, 1.0, size=(2, height, width, _DIM)).astype(
            "float32"
        )
        y = keras.ops.convert_to_numpy(model(x))
        assert y.shape == (2, height, width, _DIM)
        assert np.all(np.isfinite(y))


def test_the_dynamic_path_agrees_with_the_reference():
    """A dynamically-built layer computes the same values as the static one."""
    layer = _randomized_layer(use_bias=False, seed=29)
    inputs = keras.Input(shape=(None, None, _DIM))
    model = keras.Model(inputs, layer(inputs))

    x = _fixed_input(seed=31)
    actual = keras.ops.convert_to_numpy(model(x)).astype(np.float64)
    expected = _gdfn_reference(x, **_reference_kwargs(layer))
    np.testing.assert_allclose(
        actual,
        expected,
        rtol=0.0,
        atol=_parity_atol(float(np.max(np.abs(expected)))),
    )


# ---------------------------------------------------------------------
# 8. Precision arms
# ---------------------------------------------------------------------


def test_output_is_finite_under_every_dtype_policy(dtype_policy):
    """`mixed_float16` and `float64` must both produce finite output."""
    layer = GatedDConvFeedForward(dim=_DIM, ffn_expansion_factor=_FACTOR)
    x = keras.ops.cast(_fixed_input(seed=37), layer.compute_dtype)
    y = layer(x)
    assert bool(keras.ops.all(keras.ops.isfinite(y))), (
        f"non-finite output under policy {dtype_policy}"
    )
    assert keras.ops.convert_to_numpy(y).shape == (2, _HEIGHT, _WIDTH, _DIM)


# ---------------------------------------------------------------------
# 9. Configuration validation
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "dim,factor,match",
    [
        (0, 2.66, "dim must be positive"),
        (-8, 2.66, "dim must be positive"),
        (8, 0.0, "ffn_expansion_factor must be positive"),
        (8, -1.0, "ffn_expansion_factor must be positive"),
    ],
)
def test_non_positive_configuration_raises(dim, factor, match):
    with pytest.raises(ValueError, match=match):
        GatedDConvFeedForward(dim=dim, ffn_expansion_factor=factor)


@pytest.mark.parametrize(
    "dim,factor",
    [(1, 0.5), (3, 0.25), (2, 0.4)],
    ids=["dim1_f0.5", "dim3_f0.25", "dim2_f0.4"],
)
def test_a_hidden_width_that_truncates_to_zero_raises(dim, factor):
    """A small dim with a small factor is a real degenerate case, not theory.

    The message must name BOTH values, since neither alone explains the
    truncation.
    """
    with pytest.raises(ValueError) as excinfo:
        GatedDConvFeedForward(dim=dim, ffn_expansion_factor=factor)
    message = str(excinfo.value)
    assert "truncates to 0" in message
    assert f"dim={dim}" in message
    assert f"ffn_expansion_factor={factor}" in message


def test_a_channel_mismatch_is_caught_at_build():
    layer = GatedDConvFeedForward(dim=_DIM, ffn_expansion_factor=_FACTOR)
    with pytest.raises(ValueError, match="match"):
        layer.build((None, _HEIGHT, _WIDTH, _DIM + 1))


def test_a_non_rank_4_input_is_caught_at_build():
    layer = GatedDConvFeedForward(dim=_DIM, ffn_expansion_factor=_FACTOR)
    with pytest.raises(ValueError, match="4D input shape"):
        layer.build((None, _HEIGHT * _WIDTH, _DIM))
