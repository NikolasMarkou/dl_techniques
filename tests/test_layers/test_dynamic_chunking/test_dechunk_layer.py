"""Tests for ``DeChunkLayer`` -- H-Net's EMA smoother and full-resolution scatter.

The grading oracle is ``hnet_reference_numpy.dechunk_reference``: the float64
NumPy transcription of ``_reference/dc.py:239-313`` (with the EMA taken from the
reference's own kernel-free ``step()`` at ``dc.py:333``), written *before* this
layer existed and RED-proven by its own injected mutations (plan decision D-011).
No second reference is written here.

Four guards are owed by plan step 6:

(i)   parity against ``dechunk_reference`` at a DERIVED tolerance;
(ii)  ``test_a_non_boundary_position_repeats_the_last_chunk_value``, with the
      "something changed" twin asserting a boundary position takes a NEW value;
(iii) the long-chunk underflow arm at ``p = 1e-4, inner_len = 512``, asserting
      finiteness AND monotone decay -- ``p`` is DRIVEN to its clamp, never
      sampled, because a long-sequence accumulator arm at a random ``p``
      accumulates no more than two steps would and proves nothing;
(iv)  the fp32-vs-float64 delta reported explicitly, so the shipped tolerance is
      measured rather than assumed.

Tolerance derivation
--------------------

``rtol = 0`` throughout. The absolute bound is a function of the compared
quantity's own scale, not a pasted constant::

    ema_atol(inner) = ROUNDED_OPS_PER_EMA_STEP * ERROR_GROWTH_ALLOWANCE
                      * eps_float32 * max|inner|

- ``ROUNDED_OPS_PER_EMA_STEP = 4``. One EMA step is ``h = p*x + (1-p)*h``: a
  subtract, two multiplies and an add -- four rounded float32 operations on
  quantities bounded by ``max|inner|``, because the recurrence is a CONVEX
  combination and cannot amplify its inputs (``|h_t| <= max_s |x_s|``).
- ``ERROR_GROWTH_ALLOWANCE = 2``. The carried error is multiplied by
  ``(1 - p_t) <= 1`` at every step, so it contracts rather than compounds, and
  the measured growth from ``M = 8`` to ``M = 512`` is a factor of ~3 rather
  than ~64. Two units of headroom cover that observed growth.
- The scale is ``max|inner|`` and NOT ``max|output|``. That distinction is
  measured, not assumed: in the ``p = 1e-4`` regime the outputs are ~5e-3 while
  the inputs are ~4, and an output-scaled bound would be 49x too tight there.

This is deliberately NOT ``tests/numerics.reassociation_atol``: step 4 MEASURED
that helper UNDER-counting against a float64 oracle (D-012), and this layer's
error model is a sequential recurrence, not a reassociated reduction.

Both halves of the bound are proven by execution rather than argued.
``test_the_derived_bound_is_attainable_across_every_regime`` measures the worst
``max|f32 layer - f64 oracle| / (eps * max|inner|)`` over 24 regime x length x
width cells and asserts it is below the shipped coefficient with real headroom;
``test_the_parity_instrument_can_report_a_mismatch`` shows the same comparison
failing on the two defects this recurrence is realistically got wrong (a
``p <-> 1-p`` swap and an off-by-one in ``p``).

The fp32-vs-float64 arm (guard iv) is the reason the bound can be this tight:
run at ``dtype="float64"`` the layer reproduces the float64 oracle EXACTLY
(``max|delta| = 0.0``) in every one of those 24 cells, so the fp32 residue is
float32 rounding and nothing else.

Masks are built from ``keras.ops.arange`` broadcasts and never from
``keras.ops.tril``/``triu``: those raise ``TypeError: ('pred must not be a Python
bool', True)`` under plain ``tf.function`` as well as under XLA (measured,
D-010(b)), while eager ``tril`` is bitwise equal to the ``arange`` form -- so a
wrong call looks correct until it is traced.
"""

import numpy as np
import pytest
import keras
import tensorflow as tf

from dl_techniques.layers.dynamic_chunking.dechunk_layer import (
    DEFAULT_CLAMP_MAX,
    DEFAULT_CLAMP_MIN,
    DeChunkLayer,
)

from .hnet_reference_numpy import dechunk_reference

# ---------------------------------------------------------------------
# Derived tolerance -- see the module docstring for the derivation
# ---------------------------------------------------------------------

#: float32 machine epsilon, read from NumPy rather than typed out.
EPS32 = float(np.finfo(np.float32).eps)

#: ``h = p*x + (1-p)*h`` -- subtract, multiply, multiply, add.
ROUNDED_OPS_PER_EMA_STEP = 4

#: Headroom for the observed sub-linear growth of the residue in ``M``.
ERROR_GROWTH_ALLOWANCE = 2

#: The shipped coefficient, in units of ``eps * max|inner|``.
EMA_ATOL_COEFFICIENT = ROUNDED_OPS_PER_EMA_STEP * ERROR_GROWTH_ALLOWANCE


def ema_atol(inner: np.ndarray) -> float:
    """The derived absolute bound for one comparison, from the input's own scale.

    :param inner: The ``(B, M, D)`` inner hidden states fed to the layer.
    :type inner: numpy.ndarray
    :return: Absolute tolerance to use with ``rtol=0``.
    :rtype: float
    """
    return EMA_ATOL_COEFFICIENT * EPS32 * float(np.max(np.abs(inner)))


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------


def _boundaries(rng, batch_size: int, seq_len: int, count: int) -> np.ndarray:
    """A ``(B, L)`` boundary mask with position 0 forced and exactly ``count`` set.

    Position 0 is always a boundary (``dc.py:95-96`` forces ``p = 1.0`` there),
    so ``cumsum(mask) - 1`` is never negative and the port's ``max(idx, 0)`` clip
    is inert on every well-formed input. The rows that make it fire get their own
    test.

    :param rng: Source of randomness.
    :type rng: numpy.random.Generator
    :param batch_size: Number of rows.
    :type batch_size: int
    :param seq_len: Full-resolution length.
    :type seq_len: int
    :param count: Boundaries per row, including position 0.
    :type count: int
    :return: ``(B, L)`` boolean array.
    :rtype: numpy.ndarray
    """
    out = np.zeros((batch_size, seq_len), dtype=bool)
    out[:, 0] = True
    for row in range(batch_size):
        if count > 1:
            picks = rng.choice(np.arange(1, seq_len), size=count - 1, replace=False)
            out[row, picks] = True
    return out


def _padding_mask(batch_size: int, seq_len: int, lengths) -> np.ndarray:
    """Boolean ``(B, L)`` validity mask built from an ``arange`` broadcast.

    Never ``tril``/``triu`` -- see this module's docstring.

    :param batch_size: Number of rows.
    :type batch_size: int
    :param seq_len: Sequence length.
    :type seq_len: int
    :param lengths: Per-row count of valid leading positions.
    :type lengths: Sequence[int]
    :return: ``(B, L)`` boolean array, ``True`` = valid.
    :rtype: numpy.ndarray
    """
    positions = keras.ops.arange(seq_len)[None, :]
    limits = keras.ops.convert_to_tensor(np.asarray(lengths, dtype="int32"))[:, None]
    out = keras.ops.convert_to_numpy(keras.ops.less(positions, limits))
    assert out.shape == (batch_size, seq_len)
    return out.astype(bool)


def _run(layer, inner, prob, boundary, mask=None, dtype="float32") -> np.ndarray:
    """Call ``layer`` and return the output as a float64 NumPy array.

    :param layer: The :class:`DeChunkLayer` under test.
    :type layer: DeChunkLayer
    :param inner: ``(B, M, D)`` inner hidden states.
    :type inner: numpy.ndarray
    :param prob: ``(B, L)`` or ``(B, L, 2)`` boundary probabilities.
    :type prob: numpy.ndarray
    :param boundary: ``(B, L)`` boolean boundary mask.
    :type boundary: numpy.ndarray
    :param mask: ``(B, L)`` validity mask or ``None``.
    :type mask: numpy.ndarray or None
    :param dtype: Float dtype to feed the layer with.
    :type dtype: str
    :return: ``(B, L, D)`` float64 array.
    :rtype: numpy.ndarray
    """
    kwargs = {
        "boundary_prob": keras.ops.convert_to_tensor(np.asarray(prob, dtype=dtype)),
        "boundary_mask": keras.ops.convert_to_tensor(boundary),
    }
    if mask is not None:
        kwargs["mask"] = keras.ops.convert_to_tensor(mask)
    out = layer(keras.ops.convert_to_tensor(np.asarray(inner, dtype=dtype)), **kwargs)
    return keras.ops.convert_to_numpy(out).astype(np.float64)


def _closed_form_ema(p: np.ndarray, x: np.ndarray) -> np.ndarray:
    """The REJECTED reassociation, in float64, kept only to show it dies.

    ``A_t = prod_{s<=t}(1 - p_s)``, ``h_t = A_t * sum_{s<=t} p_s x_s / A_s``.
    Algebraically identical to the recurrence and numerically unusable at the
    clamp bound -- see ``TestGuardThreeUnderflowAtTheClamp``.

    :param p: ``(B, M)`` gate.
    :type p: numpy.ndarray
    :param x: ``(B, M, D)`` values.
    :type x: numpy.ndarray
    :return: ``(B, M, D)`` -- ``nan`` once ``A_t`` underflows.
    :rtype: numpy.ndarray
    """
    p = np.asarray(p, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        cumulative = np.cumprod(1.0 - p, axis=1)
        terms = (p / cumulative)[:, :, None] * x
        return cumulative[:, :, None] * np.cumsum(terms, axis=1)


#: The regimes the tolerance is derived over. ``PMIN``/``PMAX`` are the two
#: values ``dc.py:256``'s own clamp produces, not random draws.
_REGIMES = {
    "uniform": lambda rng, shape: rng.uniform(
        DEFAULT_CLAMP_MIN, DEFAULT_CLAMP_MAX, shape
    ),
    "pmax": lambda rng, shape: np.full(shape, DEFAULT_CLAMP_MAX),
    "pmin": lambda rng, shape: np.full(shape, DEFAULT_CLAMP_MIN),
    "beta": lambda rng, shape: np.clip(
        rng.beta(0.2, 0.2, shape), DEFAULT_CLAMP_MIN, DEFAULT_CLAMP_MAX
    ),
}


# =====================================================================
# Construction, configuration, registration
# =====================================================================


class TestInitAndConfig:
    """Constructor state, the weightless contract, config round trip, registration."""

    def test_init_stores_the_reference_clamp_bounds(self):
        layer = DeChunkLayer()
        assert layer.clamp_min == 1e-4
        assert layer.clamp_max == 1.0 - 1e-4
        assert layer.built is False
        assert layer.weights == []

    def test_the_layer_is_deliberately_weightless_after_build(self):
        """A per-weight gradient check is vacuous here, so the ABSENCE is asserted.

        ``DeChunkLayer`` is a clamp, a gather, a scan and a scatter. It owns no
        variables, and this test says so on purpose rather than silently skipping
        the gradient-flow requirement: if a weight ever appears, the step-6
        design has changed and this assertion is where that surfaces.
        """
        layer = DeChunkLayer()
        layer.build((None, 4, 5))
        assert layer.built is True
        assert layer.weights == []
        assert layer.trainable_variables == []
        assert layer.non_trainable_variables == []

    def test_get_config_round_trip_reproduces_behaviour(self):
        rng = np.random.default_rng(0)
        inner = rng.standard_normal((2, 3, 4)).astype("float32")
        prob = rng.uniform(0.0, 1.0, (2, 9)).astype("float32")
        boundary = _boundaries(rng, 2, 9, 3)

        original = DeChunkLayer(clamp_min=0.01, clamp_max=0.99)
        config = original.get_config()
        assert config["clamp_min"] == 0.01
        assert config["clamp_max"] == 0.99
        clone = DeChunkLayer.from_config(config)

        first = _run(original, inner, prob, boundary)
        second = _run(clone, inner, prob, boundary)
        np.testing.assert_allclose(first, second, rtol=0, atol=0.0)

        # DIFFER twin: the comparison above is not comparing constants -- a clone
        # built from DIFFERENT bounds does not reproduce it. The probabilities
        # above include values outside [0.01, 0.99], so the bounds are live.
        other = DeChunkLayer.from_config(
            {**config, "clamp_min": 0.4, "clamp_max": 0.6}
        )
        assert float(np.max(np.abs(_run(other, inner, prob, boundary) - first))) > 1e-3

    def test_layer_is_registered_under_the_package_qualified_key(self):
        key = keras.saving.get_registered_name(DeChunkLayer)
        assert key == "dl_techniques.layers.dynamic_chunking.dechunk_layer>DeChunkLayer"
        assert keras.saving.get_registered_object(key) is DeChunkLayer


class TestValidation:
    """Every raise, and the inputs that must reach one."""

    @pytest.mark.parametrize(
        "clamp_min,clamp_max",
        [(0.0, 0.9), (-1e-4, 0.9), (0.5, 0.5), (0.9, 0.1), (0.1, 1.5)],
    )
    def test_out_of_order_or_out_of_range_bounds_raise(self, clamp_min, clamp_max):
        with pytest.raises(ValueError, match="clamp bounds must satisfy"):
            DeChunkLayer(clamp_min=clamp_min, clamp_max=clamp_max)

    @pytest.mark.parametrize("bad", ["0.1", None, True])
    def test_non_float_bounds_raise(self, bad):
        with pytest.raises(ValueError, match="must be a float"):
            DeChunkLayer(clamp_min=bad)

    def test_rank_two_input_raises(self):
        layer = DeChunkLayer()
        with pytest.raises(ValueError, match="rank-3"):
            layer.build((None, 8))

    def test_a_missing_boundary_prob_raises(self):
        layer = DeChunkLayer()
        inner = keras.ops.convert_to_tensor(np.zeros((2, 3, 4), dtype="float32"))
        boundary = keras.ops.convert_to_tensor(np.ones((2, 8), dtype=bool))
        with pytest.raises(ValueError, match="requires boundary_prob"):
            layer(inner, boundary_mask=boundary)

    def test_a_missing_boundary_mask_raises(self):
        layer = DeChunkLayer()
        inner = keras.ops.convert_to_tensor(np.zeros((2, 3, 4), dtype="float32"))
        prob = keras.ops.convert_to_tensor(np.ones((2, 8), dtype="float32"))
        with pytest.raises(ValueError, match="requires boundary_prob"):
            layer(inner, boundary_prob=prob)

    def test_a_correct_shape_does_not_raise(self):
        layer = DeChunkLayer()
        layer.build((None, 4, 6))
        assert layer.built is True


# =====================================================================
# Forward-pass contract
# =====================================================================


class TestForwardPass:
    """Shapes, dtypes, the two accepted probability layouts, the mask, training."""

    @pytest.mark.parametrize(
        "batch_size,seq_len,inner_len,d_model",
        [(1, 4, 2, 3), (3, 16, 5, 8), (2, 33, 33, 4), (4, 12, 1, 6)],
    )
    def test_output_shape_and_dtype(self, batch_size, seq_len, inner_len, d_model):
        rng = np.random.default_rng(301)
        inner = rng.standard_normal((batch_size, inner_len, d_model)).astype("float32")
        prob = rng.uniform(0.0, 1.0, (batch_size, seq_len)).astype("float32")
        boundary = _boundaries(rng, batch_size, seq_len, inner_len)
        out = _run(DeChunkLayer(), inner, prob, boundary)
        assert out.shape == (batch_size, seq_len, d_model)
        assert np.isfinite(out).all()

    def test_compute_output_shape_matches_the_real_call(self):
        rng = np.random.default_rng(302)
        inner = rng.standard_normal((2, 4, 7)).astype("float32")
        prob = rng.uniform(0.0, 1.0, (2, 11)).astype("float32")
        boundary = _boundaries(rng, 2, 11, 4)
        layer = DeChunkLayer()
        declared = layer.compute_output_shape(
            (2, 4, 7), boundary_prob_shape=(2, 11), boundary_mask_shape=(2, 11)
        )
        assert declared == (2, 11, 7)
        assert _run(layer, inner, prob, boundary).shape == declared

    def test_a_rank_three_probability_reads_the_boundary_channel(self):
        """``boundary_prob[..., -1]`` -- ``dc.py:256`` -- for a ``(B, L, 2)`` input.

        The router emits ``(B, L, 2)``; the layer must read channel ``-1`` and
        not channel ``0``. The twin makes the channel choice observable by
        putting DIFFERENT numbers in the two channels.
        """
        rng = np.random.default_rng(303)
        inner = rng.standard_normal((2, 4, 5)).astype("float32")
        boundary = _boundaries(rng, 2, 12, 4)
        p = rng.uniform(0.05, 0.95, (2, 12)).astype("float32")
        stacked = np.stack([1.0 - p, p], axis=-1).astype("float32")

        rank_three = _run(DeChunkLayer(), inner, stacked, boundary)
        rank_two = _run(DeChunkLayer(), inner, p, boundary)
        np.testing.assert_allclose(rank_three, rank_two, rtol=0, atol=0.0)

        # DIFFER twin: reading channel 0 instead would give a different answer,
        # so the arm above is not satisfied by an implementation that ignores the
        # channel index.
        wrong_channel = _run(DeChunkLayer(), inner, 1.0 - p, boundary)
        assert float(np.max(np.abs(rank_three - wrong_channel))) > 1e-3

    def test_the_padding_mask_removes_a_boundary_the_router_left_set(self):
        """``mask`` is ANDed into ``boundary_mask``, exactly as ``ChunkLayer`` does.

        Both layers must partition on the SAME predicate or inner column ``j``
        stops meaning the same position in the two of them.
        """
        rng = np.random.default_rng(304)
        seq_len, inner_len, d_model = 10, 4, 3
        inner = rng.standard_normal((1, inner_len, d_model)).astype("float32")
        prob = rng.uniform(0.2, 0.8, (1, seq_len)).astype("float32")
        boundary = np.zeros((1, seq_len), dtype=bool)
        boundary[0, [0, 3, 8]] = True
        mask = _padding_mask(1, seq_len, [6])  # position 8 is padding

        masked = _run(DeChunkLayer(), inner, prob, boundary, mask=mask)
        equivalent = _run(DeChunkLayer(), inner, prob, boundary & mask)
        np.testing.assert_allclose(masked, equivalent, rtol=0, atol=0.0)

        # DIFFER twin: `mask` is not a no-op on this input -- dropping the
        # boundary at position 8 moves the tail of the row.
        unmasked = _run(DeChunkLayer(), inner, prob, boundary)
        assert float(np.max(np.abs(masked[:, 8:] - unmasked[:, 8:]))) > 1e-3

    def test_an_all_true_mask_is_the_same_as_no_mask(self):
        rng = np.random.default_rng(305)
        inner = rng.standard_normal((2, 3, 4)).astype("float32")
        prob = rng.uniform(0.1, 0.9, (2, 9)).astype("float32")
        boundary = _boundaries(rng, 2, 9, 3)
        with_mask = _run(
            DeChunkLayer(), inner, prob, boundary, mask=np.ones((2, 9), dtype=bool)
        )
        without = _run(DeChunkLayer(), inner, prob, boundary)
        np.testing.assert_allclose(with_mask, without, rtol=0, atol=0.0)

    def test_an_integer_boundary_mask_is_accepted(self):
        rng = np.random.default_rng(306)
        inner = rng.standard_normal((2, 3, 4)).astype("float32")
        prob = rng.uniform(0.1, 0.9, (2, 9)).astype("float32")
        boundary = _boundaries(rng, 2, 9, 3)
        as_int = _run(DeChunkLayer(), inner, prob, boundary.astype("int32"))
        as_bool = _run(DeChunkLayer(), inner, prob, boundary)
        np.testing.assert_allclose(as_int, as_bool, rtol=0, atol=0.0)

    def test_the_training_flag_does_not_change_the_output(self):
        rng = np.random.default_rng(307)
        inner = rng.standard_normal((2, 3, 4)).astype("float32")
        prob = rng.uniform(0.1, 0.9, (2, 9)).astype("float32")
        boundary = _boundaries(rng, 2, 9, 3)
        layer = DeChunkLayer()
        kwargs = {
            "boundary_prob": keras.ops.convert_to_tensor(prob),
            "boundary_mask": keras.ops.convert_to_tensor(boundary),
        }
        inner_tensor = keras.ops.convert_to_tensor(inner)
        train = keras.ops.convert_to_numpy(layer(inner_tensor, training=True, **kwargs))
        infer = keras.ops.convert_to_numpy(layer(inner_tensor, training=False, **kwargs))
        np.testing.assert_array_equal(train, infer)

    def test_the_clamp_fires_on_probabilities_outside_the_bounds(self):
        """``dc.py:256`` -- ``p`` never reaches 0 or 1, whatever the router says.

        Fed ``p = 0`` the un-clamped recurrence would freeze (``h_t = h_{t-1}``,
        so every output is exactly zero); fed the clamp it moves by ``1e-4``.
        """
        inner = np.ones((1, 4, 2), dtype="float32")
        boundary = np.zeros((1, 4), dtype=bool)
        boundary[0, :] = True
        out = _run(DeChunkLayer(), inner, np.zeros((1, 4), "float32"), boundary)
        np.testing.assert_allclose(out[0, 0], DEFAULT_CLAMP_MIN, rtol=0, atol=1e-9)
        assert float(np.min(np.abs(out))) > 0.0

        # DIFFER twin: the upper clamp is live too -- at p = 1 the un-clamped
        # recurrence would copy the input exactly, and the clamp keeps it short
        # of that by exactly 1e-4 of the previous state.
        # (``abs=1e-6``, not ``1e-9``: ``1 - 1e-4`` is not exactly representable
        # in float32 and rounds to ``0.99989998``, so a float64-tight bound here
        # would be measuring the literal's representation, not the clamp.)
        top = _run(DeChunkLayer(), inner, np.ones((1, 4), "float32"), boundary)
        assert float(top[0, 0, 0]) == pytest.approx(DEFAULT_CLAMP_MAX, abs=1e-6)
        assert float(top[0, 0, 0]) < 1.0


# =====================================================================
# Guard (i): parity with the step-3 float64 oracle
# =====================================================================


class TestGuardOneOracleParity:
    """Parity with ``dechunk_reference`` at the DERIVED tolerance."""

    @pytest.mark.parametrize("regime", sorted(_REGIMES))
    @pytest.mark.parametrize(
        "seed,batch_size,seq_len,inner_len,d_model",
        [
            (401, 1, 4, 2, 3),
            (402, 3, 16, 8, 6),
            (403, 2, 64, 32, 5),
            (404, 4, 24, 24, 7),
        ],
    )
    def test_parity_without_a_mask(
        self, regime, seed, batch_size, seq_len, inner_len, d_model
    ):
        rng = np.random.default_rng(seed)
        inner = rng.standard_normal((batch_size, inner_len, d_model)).astype("float32")
        prob = _REGIMES[regime](rng, (batch_size, seq_len)).astype("float32")
        boundary = _boundaries(rng, batch_size, seq_len, inner_len)

        got = _run(DeChunkLayer(), inner, prob, boundary)
        expected = dechunk_reference(inner, prob, boundary)
        np.testing.assert_allclose(got, expected, rtol=0, atol=ema_atol(inner))

    @pytest.mark.parametrize("seed,lengths", [(411, [12, 7, 4]), (412, [3, 12, 12])])
    def test_parity_with_a_padding_mask(self, seed, lengths):
        rng = np.random.default_rng(seed)
        batch_size, seq_len, inner_len, d_model = len(lengths), 12, 3, 5
        inner = rng.standard_normal((batch_size, inner_len, d_model)).astype("float32")
        prob = rng.uniform(0.05, 0.95, (batch_size, seq_len)).astype("float32")
        boundary = _boundaries(rng, batch_size, seq_len, inner_len)
        mask = _padding_mask(batch_size, seq_len, lengths)

        got = _run(DeChunkLayer(), inner, prob, boundary, mask=mask)
        # The oracle takes a single boundary mask; the layer ANDs `mask` in, so
        # the equivalent oracle input is the conjunction.
        expected = dechunk_reference(inner, prob, boundary & mask)
        np.testing.assert_allclose(got, expected, rtol=0, atol=ema_atol(inner))

    def test_parity_when_fewer_boundaries_than_the_inner_width(self):
        """A row with one chunk while ``max_chunks`` is much larger (D-007).

        The surplus inner columns are garbage the ``inner_mask`` marks invalid;
        the scatter must still be defined, and ``cumsum - 1`` sends every
        position to column 0.
        """
        rng = np.random.default_rng(413)
        inner = rng.standard_normal((2, 8, 4)).astype("float32")
        prob = rng.uniform(0.05, 0.95, (2, 10)).astype("float32")
        boundary = np.zeros((2, 10), dtype=bool)
        boundary[:, 0] = True

        got = _run(DeChunkLayer(), inner, prob, boundary)
        expected = dechunk_reference(inner, prob, boundary)
        np.testing.assert_allclose(got, expected, rtol=0, atol=ema_atol(inner))
        # Every full-resolution position carries the SAME single chunk value.
        for position in range(1, 10):
            np.testing.assert_allclose(got[:, position], got[:, 0], rtol=0, atol=0.0)

    def test_parity_when_the_probability_is_rank_three(self):
        rng = np.random.default_rng(414)
        inner = rng.standard_normal((3, 6, 5)).astype("float32")
        boundary = _boundaries(rng, 3, 20, 6)
        p = rng.uniform(0.05, 0.95, (3, 20))
        stacked = np.stack([1.0 - p, p], axis=-1).astype("float32")

        got = _run(DeChunkLayer(), inner, stacked, boundary)
        expected = dechunk_reference(inner, stacked, boundary)
        np.testing.assert_allclose(got, expected, rtol=0, atol=ema_atol(inner))

    def test_the_parity_instrument_can_report_a_mismatch(self):
        """An instrument that cannot report a difference has measured nothing.

        The two arms are the two ways this recurrence is realistically got wrong:
        a swapped gate (``p`` where ``1-p`` belongs) and an off-by-one in ``p``.
        Both must exceed the derived bound by orders of magnitude.
        """
        rng = np.random.default_rng(415)
        inner = rng.standard_normal((3, 16, 6)).astype("float32")
        prob = rng.uniform(0.05, 0.95, (3, 40)).astype("float32")
        boundary = _boundaries(rng, 3, 40, 16)
        got = _run(DeChunkLayer(), inner, prob, boundary)
        bound = ema_atol(inner)

        swapped = dechunk_reference(inner, 1.0 - prob, boundary)
        assert float(np.max(np.abs(got - swapped))) > 1e3 * bound

        rolled = dechunk_reference(inner, np.roll(prob, 1, axis=1), boundary)
        assert float(np.max(np.abs(got - rolled))) > 1e3 * bound

    def test_the_derived_bound_is_attainable_across_every_regime(self):
        """Attainability, measured: the worst residue over 24 cells is well inside.

        This is the arm that makes the docstring's derivation a measurement. It
        reports the worst ``max|f32 - f64 oracle| / (eps * max|inner|)`` over
        every regime x length x width cell and asserts it is below the shipped
        coefficient of 8 -- with the observed value printed into the assertion
        message so a future regression says by how much it moved.
        """
        rng = np.random.default_rng(416)
        worst_ratio = 0.0
        worst_delta = 0.0
        for regime in sorted(_REGIMES):
            for seq_len, inner_len in ((16, 8), (128, 64), (1024, 512)):
                for d_model in (6, 32):
                    inner = rng.standard_normal(
                        (3, inner_len, d_model)
                    ).astype("float32")
                    prob = _REGIMES[regime](rng, (3, seq_len)).astype("float32")
                    boundary = _boundaries(rng, 3, seq_len, inner_len)
                    got = _run(DeChunkLayer(), inner, prob, boundary)
                    expected = dechunk_reference(inner, prob, boundary)
                    delta = float(np.max(np.abs(got - expected)))
                    scale = EPS32 * float(np.max(np.abs(inner)))
                    worst_delta = max(worst_delta, delta)
                    worst_ratio = max(worst_ratio, delta / scale)

        assert worst_ratio < EMA_ATOL_COEFFICIENT, (
            f"worst residue {worst_ratio:.3f} eps*|x| exceeds the shipped "
            f"coefficient {EMA_ATOL_COEFFICIENT}"
        )
        # Anti-vacuity for the bound itself: the residue is not identically zero
        # in float32, so the tolerance is bounding something real rather than
        # being slack around an exact computation.
        assert worst_delta > 0.0


# =====================================================================
# Guard (ii): the scatter semantics
# =====================================================================


class TestGuardTwoScatterSemantics:
    """``plug_back_idx = cumsum(boundary_mask) - 1`` -- ``dc.py:302-308``."""

    def test_a_non_boundary_position_repeats_the_last_chunk_value(self):
        """Positions 1,2 repeat chunk 0; positions 4..7 repeat chunk 1.

        The values are compared at ``atol = 0``: the scatter is a gather of
        identical rows, so anything other than bit-equality is a defect, not
        rounding.
        """
        seq_len, d_model = 8, 3
        inner = np.array(
            [[[1.0, 2.0, 3.0], [-4.0, -5.0, -6.0]]], dtype="float32"
        )  # (1, 2, 3) -- the two chunks are far apart in value
        boundary = np.zeros((1, seq_len), dtype=bool)
        boundary[0, [0, 3]] = True
        prob = np.full((1, seq_len), 0.9, dtype="float32")

        out = _run(DeChunkLayer(), inner, prob, boundary)
        for position in (1, 2):
            np.testing.assert_allclose(out[0, position], out[0, 0], rtol=0, atol=0.0)
        for position in (4, 5, 6, 7):
            np.testing.assert_allclose(out[0, position], out[0, 3], rtol=0, atol=0.0)

    def test_a_boundary_position_takes_a_new_value(self):
        """The "something changed" twin: without it the arm above is satisfiable
        by a layer that broadcasts one constant everywhere.

        The margin is asserted against the derived bound, not against zero, so a
        rounding-scale difference could not pass for a new value.
        """
        seq_len = 8
        inner = np.array(
            [[[1.0, 2.0, 3.0], [-4.0, -5.0, -6.0]]], dtype="float32"
        )
        boundary = np.zeros((1, seq_len), dtype=bool)
        boundary[0, [0, 3]] = True
        prob = np.full((1, seq_len), 0.9, dtype="float32")

        out = _run(DeChunkLayer(), inner, prob, boundary)
        moved = float(np.max(np.abs(out[0, 3] - out[0, 2])))
        assert moved > 1e3 * ema_atol(inner)
        # And the move is the EMA of the second chunk, not an arbitrary change:
        # h_1 = p*x_1 + (1-p)*h_0 with h_0 = p*x_0.
        p = DEFAULT_CLAMP_MAX if 0.9 > DEFAULT_CLAMP_MAX else 0.9
        h0 = p * inner[0, 0]
        h1 = p * inner[0, 1] + (1.0 - p) * h0
        np.testing.assert_allclose(out[0, 0], h0, rtol=0, atol=1e-6)
        np.testing.assert_allclose(out[0, 3], h1, rtol=0, atol=1e-6)

    def test_the_scatter_index_is_a_cumsum_and_not_a_position(self):
        """A boundary at position ``k`` maps to chunk ``rank(k)``, not to chunk ``k``.

        With boundaries at ``{0, 5}`` the second chunk is column 1 of the inner
        sequence; an implementation indexing by POSITION would read column 5 and
        run off a 2-column inner sequence (or, clipped, read the wrong column).
        """
        seq_len = 7
        inner = np.array([[[10.0], [20.0], [30.0], [40.0]]], dtype="float32")
        boundary = np.zeros((1, seq_len), dtype=bool)
        boundary[0, [0, 5]] = True
        prob = np.ones((1, seq_len), dtype="float32")  # clamped to 1 - 1e-4

        out = _run(DeChunkLayer(), inner, prob, boundary)
        # p ~ 1 makes the EMA an (almost) exact copy, so the columns are readable.
        np.testing.assert_allclose(out[0, 0, 0], 10.0, rtol=0, atol=1e-2)
        np.testing.assert_allclose(out[0, 5, 0], 20.0, rtol=0, atol=1e-2)
        # DIFFER twin: chunk index 5 would have been 40.0-ish (or clipped to
        # 40.0 at the last column), which this row is measurably not.
        assert abs(float(out[0, 5, 0]) - 40.0) > 1.0

    def test_a_row_without_a_boundary_at_position_zero_is_clipped_not_wrapped(self):
        """The port's ``max(idx, 0)`` divergence, guarded.

        ``dc.py:95-96`` forces position 0 to be a boundary so upstream never sees
        ``-1``; a caller who bypasses the router would, and NumPy wraps a ``-1``
        gather to the LAST column while TensorFlow's behaviour is undefined. The
        port clips to column 0 instead. This test pins the clip and shows it is
        NOT the wrap.
        """
        seq_len = 6
        inner = np.array([[[1.0], [2.0], [3.0]]], dtype="float32")
        boundary = np.zeros((1, seq_len), dtype=bool)
        boundary[0, [2, 4]] = True  # position 0 is deliberately NOT a boundary
        prob = np.ones((1, seq_len), dtype="float32")

        out = _run(DeChunkLayer(), inner, prob, boundary)
        assert np.isfinite(out).all()
        # Positions 0 and 1 precede every boundary: clipped to chunk 0 (~1.0).
        np.testing.assert_allclose(out[0, 0, 0], 1.0, rtol=0, atol=1e-2)
        np.testing.assert_allclose(out[0, 1, 0], 1.0, rtol=0, atol=1e-2)
        # DIFFER twin: NumPy's wrap would have put the LAST column (~3.0) there,
        # and the oracle -- which is NumPy -- does exactly that. The divergence
        # is real and is recorded, not accidental.
        wrapped = dechunk_reference(inner, prob, boundary)
        assert abs(float(wrapped[0, 0, 0]) - float(out[0, 0, 0])) > 1.0

    def test_more_boundaries_than_inner_columns_is_clipped_to_the_last_column(self):
        """The other half of the D-007 divergence: ``cumsum - 1`` can exceed ``M-1``.

        Under a FIXED ``max_chunks`` a row may carry more boundaries than the
        inner sequence has columns. Upstream this cannot happen (``M`` is the
        batch maximum). The port clips to the last column; the guard is that the
        result is defined and finite, and that it is the last column rather than
        an out-of-range read.
        """
        seq_len = 8
        inner = np.array([[[1.0], [2.0]]], dtype="float32")  # M = 2
        boundary = np.ones((1, seq_len), dtype=bool)  # 8 boundaries
        prob = np.ones((1, seq_len), dtype="float32")

        out = _run(DeChunkLayer(), inner, prob, boundary)
        assert np.isfinite(out).all()
        np.testing.assert_allclose(out[0, 0, 0], 1.0, rtol=0, atol=1e-2)
        for position in range(1, seq_len):
            np.testing.assert_allclose(out[0, position, 0], out[0, 1, 0], rtol=0, atol=0.0)
        # DIFFER twin: the clipped tail is the SECOND column, not the first.
        assert abs(float(out[0, 7, 0]) - float(out[0, 0, 0])) > 0.5


# =====================================================================
# Guard (iii): the underflow arm, at the clamp and nowhere else
# =====================================================================


class TestGuardThreeUnderflowAtTheClamp:
    """512 steps at ``p`` DRIVEN to its clamp -- not sampled, not near it."""

    @staticmethod
    def _impulse_inputs(inner_len: int, raw_prob: float):
        """One unit impulse at inner column 0, zeros after; ``p`` forced to a clamp.

        Passing a RAW probability outside ``[1e-4, 1-1e-4]`` makes the layer's own
        clamp produce the exact bound, so the regime is driven by the code under
        test rather than by the test's arithmetic.

        :param inner_len: Inner sequence length ``M``.
        :type inner_len: int
        :param raw_prob: The pre-clamp probability to feed (e.g. ``0.0``).
        :type raw_prob: float
        :return: ``(inner, prob, boundary)``.
        :rtype: tuple
        """
        inner = np.zeros((1, inner_len, 1), dtype="float32")
        inner[0, 0, 0] = 1.0
        boundary = np.ones((1, inner_len), dtype=bool)
        prob = np.full((1, inner_len), raw_prob, dtype="float32")
        return inner, prob, boundary

    def test_the_carried_state_decays_monotonically_and_stays_finite(self):
        """``p = 1e-4``, ``M = 512``: finite AND strictly decreasing, and the decay
        is the one 512 steps produce, not the one two steps produce.

        Finiteness alone is near-vacuous -- a layer that never accumulated
        anything is also finite. After the impulse, ``h_t = (1-p) h_{t-1}``, so
        the closed-form carried weight after ``M-1`` decay steps is
        ``(1-p)**(M-1)``. The arm asserts (a) monotone decay at every one of the
        511 steps, (b) agreement with that closed form, and (c) that the total
        decay is two orders of magnitude larger than a 2-step run's -- which is
        the assertion that the accumulation actually happened.
        """
        inner_len = 512
        inner, prob, boundary = self._impulse_inputs(inner_len, 0.0)
        out = _run(DeChunkLayer(), inner, prob, boundary)[0, :, 0]

        assert np.isfinite(out).all()
        assert float(out[0]) == pytest.approx(DEFAULT_CLAMP_MIN, rel=1e-5)
        assert np.all(np.diff(out) < 0.0), "the carried state must decay at every step"

        expected_tail = DEFAULT_CLAMP_MIN * (1.0 - DEFAULT_CLAMP_MIN) ** (inner_len - 1)
        assert float(out[-1]) == pytest.approx(expected_tail, rel=1e-4)

        total_decay = 1.0 - float(out[-1]) / float(out[0])
        two_step_decay = 1.0 - (1.0 - DEFAULT_CLAMP_MIN) ** 1
        assert total_decay > 100.0 * two_step_decay, (
            f"512 steps decayed by {total_decay:.6f}, which is not measurably "
            f"more than the {two_step_decay:.6f} a single step gives -- the "
            f"accumulation did not happen"
        )

    def test_the_same_construction_at_two_steps_does_NOT_show_the_decay(self):
        """Anti-vacuity twin for the arm above: the probe distinguishes M.

        If the 512-step assertion also passed at ``M = 2`` it would be measuring
        the construction rather than the accumulation.
        """
        inner, prob, boundary = self._impulse_inputs(2, 0.0)
        out = _run(DeChunkLayer(), inner, prob, boundary)[0, :, 0]
        total_decay = 1.0 - float(out[-1]) / float(out[0])
        two_step_decay = 1.0 - (1.0 - DEFAULT_CLAMP_MIN) ** 1
        assert total_decay == pytest.approx(two_step_decay, rel=1e-3)
        assert not total_decay > 100.0 * two_step_decay

    def test_the_upper_clamp_at_512_steps_stays_finite(self):
        """``p = 1 - 1e-4``, ``M = 512`` -- the regime that kills the closed form.

        Here the per-step carry is ``1 - p = 1e-4``, so the cumulative product
        ``A_t`` underflows. The recurrence does not care; it never forms ``A_t``.
        """
        inner_len = 512
        rng = np.random.default_rng(501)
        inner = rng.standard_normal((2, inner_len, 4)).astype("float32")
        boundary = np.ones((2, inner_len), dtype=bool)
        prob = np.ones((2, inner_len), dtype="float32")  # clamped to 1 - 1e-4

        out = _run(DeChunkLayer(), inner, prob, boundary)
        assert np.isfinite(out).all()
        expected = dechunk_reference(inner, prob, boundary)
        assert np.isfinite(expected).all()
        np.testing.assert_allclose(out, expected, rtol=0, atol=ema_atol(inner))

    def test_the_rejected_closed_form_is_nan_on_that_same_input(self):
        """What makes the D-014 anchor a guard rather than a comment.

        The anchor says: do not reassociate this loop into
        ``A_t = cumprod(1-p); h_t = A_t * cumsum(p x / A)``. This arm runs that
        expression, in FLOAT64, on the very input the arm above passes on, and
        shows it is non-finite -- so the rejection is executable evidence rather
        than an assertion in a comment.
        """
        inner_len = 512
        rng = np.random.default_rng(502)
        inner = rng.standard_normal((2, inner_len, 4))
        p = np.full((2, inner_len), DEFAULT_CLAMP_MAX)

        closed = _closed_form_ema(p, inner)
        assert not np.isfinite(closed).all(), (
            "the closed form is expected to die at the clamp bound; if it no "
            "longer does, D-010(c)'s measurement and D-014 need re-deriving"
        )
        assert float(np.min(np.abs(np.cumprod(1.0 - p, axis=1)[:, -1]))) == 0.0

        # DIFFER twin: the closed form is NOT simply broken -- at a short length
        # where A_t has not underflowed it agrees with the recurrence, which is
        # what makes it a tempting and dangerous "optimization".
        short_inner = inner[:, :8]
        short_p = p[:, :8]
        short_closed = _closed_form_ema(short_p, short_inner)
        assert np.isfinite(short_closed).all()
        boundary = np.ones((2, 8), dtype=bool)
        recurrence = dechunk_reference(short_inner, short_p, boundary)
        np.testing.assert_allclose(short_closed, recurrence, rtol=0, atol=1e-9)


# =====================================================================
# Guard (iv): the measured fp32-vs-float64 delta
# =====================================================================


class TestGuardFourPrecisionDelta:
    """The shipped tolerance is measured, and the measurement is a test."""

    @pytest.mark.parametrize("regime", sorted(_REGIMES))
    def test_the_float64_layer_reproduces_the_float64_oracle_exactly(self, regime):
        """``max|delta| == 0.0`` in float64 -- so the fp32 residue IS float32 rounding.

        This is the arm that licenses the tight fp32 bound. If the layer had an
        algorithmic divergence from the oracle -- a different association order,
        a different clamp, an off-by-one -- it would show up here at float64
        precision instead of hiding under a float32 tolerance.
        """
        rng = np.random.default_rng(601)
        seq_len, inner_len, d_model = 256, 128, 6
        inner = rng.standard_normal((3, inner_len, d_model))
        prob = _REGIMES[regime](rng, (3, seq_len))
        boundary = _boundaries(rng, 3, seq_len, inner_len)

        layer = DeChunkLayer(dtype="float64")
        got = _run(layer, inner, prob, boundary, dtype="float64")
        assert layer.compute_dtype == "float64"
        expected = dechunk_reference(inner, prob, boundary)
        assert float(np.max(np.abs(got - expected))) == 0.0

    def test_the_fp32_residue_is_strictly_smaller_than_the_bound_and_strictly_positive(
        self,
    ):
        """The reported delta, as an executable statement.

        MEASURED on this host (TF 2.18.0 / Keras 3.8.0, CPU): worst
        ``max|f32 - f64 oracle| = 6.079338e-07`` over 24 regime x length x width
        cells, at ``max|inner| ~ 3.6``, i.e. ``1.408 * eps32 * max|inner|`` --
        against a shipped coefficient of 8. The float64 arm above measures
        ``0.0`` on the same construction. This test re-derives the pair rather
        than quoting it.
        """
        rng = np.random.default_rng(602)
        seq_len, inner_len, d_model = 1024, 512, 6
        inner = rng.standard_normal((3, inner_len, d_model))
        prob = np.clip(
            rng.beta(0.2, 0.2, (3, seq_len)), DEFAULT_CLAMP_MIN, DEFAULT_CLAMP_MAX
        )
        boundary = _boundaries(rng, 3, seq_len, inner_len)
        expected = dechunk_reference(inner, prob, boundary)

        fp32 = _run(DeChunkLayer(), inner.astype("float32"), prob, boundary)
        fp64 = _run(DeChunkLayer(dtype="float64"), inner, prob, boundary, dtype="float64")

        delta32 = float(np.max(np.abs(fp32 - expected)))
        delta64 = float(np.max(np.abs(fp64 - expected)))

        assert delta64 == 0.0
        assert 0.0 < delta32 < ema_atol(inner.astype("float32"))
        # The fp32 residue is many orders above the float64 one: the two dtypes
        # are genuinely different measurements, not the same one twice.
        assert delta32 > 1e-8


# =====================================================================
# Gradients
# =====================================================================


class TestGradientFlow:
    """The layer is weightless; the differentiable paths that DO exist are pinned.

    ``keras.ops.take_along_axis`` reads ``.ndim`` off its first argument and a
    ``tf.Variable`` has none, so these tests use ``tf.constant`` + ``tape.watch``
    rather than a Variable (the trap step 5 measured; ``RoutingModule`` does not
    hit it because ``Dense`` converts first).
    """

    def test_the_layer_has_no_trainable_weights_to_receive_a_gradient(self):
        layer = DeChunkLayer()
        rng = np.random.default_rng(701)
        _run(
            layer,
            rng.standard_normal((2, 3, 4)).astype("float32"),
            rng.uniform(0.1, 0.9, (2, 9)).astype("float32"),
            _boundaries(rng, 2, 9, 3),
        )
        assert layer.trainable_variables == []
        assert layer.weights == []

    def test_gradient_reaches_every_inner_column_that_is_scattered(self):
        rng = np.random.default_rng(702)
        seq_len, inner_len, d_model = 9, 3, 4
        inner = tf.constant(
            rng.standard_normal((1, inner_len, d_model)).astype("float32")
        )
        prob = tf.constant(np.full((1, seq_len), 0.5, dtype="float32"))
        boundary = np.zeros((1, seq_len), dtype=bool)
        boundary[0, [0, 3, 6]] = True

        layer = DeChunkLayer()
        with tf.GradientTape() as tape:
            tape.watch(inner)
            out = layer(
                inner,
                boundary_prob=prob,
                boundary_mask=tf.constant(boundary),
            )
            loss = tf.reduce_sum(out)
        grad = np.asarray(tape.gradient(loss, inner))

        assert np.isfinite(grad).all()
        assert float(np.min(np.abs(grad))) > 0.0

    def test_gradient_reaches_the_boundary_probability_inside_the_clamp(self):
        """``p`` is differentiable -- it is the layer's only learnable pathway.

        The gate multiplies the values, so ``d out / d p`` is non-zero for any
        ``p`` strictly inside the clamp.
        """
        rng = np.random.default_rng(703)
        seq_len, inner_len, d_model = 6, 3, 2
        inner = tf.constant(
            rng.standard_normal((1, inner_len, d_model)).astype("float32")
        )
        prob = tf.constant(np.full((1, seq_len), 0.5, dtype="float32"))
        boundary = np.zeros((1, seq_len), dtype=bool)
        boundary[0, [0, 2, 4]] = True

        layer = DeChunkLayer()
        with tf.GradientTape() as tape:
            tape.watch(prob)
            out = layer(
                inner, boundary_prob=prob, boundary_mask=tf.constant(boundary)
            )
            loss = tf.reduce_sum(out)
        grad = np.asarray(tape.gradient(loss, prob))

        assert np.isfinite(grad).all()
        assert float(np.max(np.abs(grad[0, [0, 2, 4]]))) > 0.0

    def test_a_probability_outside_the_clamp_receives_no_gradient(self):
        """The DIFFER twin for the arm above, and the clamp's own guard.

        ``clip`` has zero derivative outside its bounds, so a router that has
        saturated past ``1 - 1e-4`` gets no signal back through this layer. That
        is the reference's behaviour (``torch.clamp``) and it is asserted rather
        than assumed, because it is the difference between a clamp and a
        soft-bounded rescale.
        """
        rng = np.random.default_rng(704)
        inner = tf.constant(rng.standard_normal((1, 3, 2)).astype("float32"))
        prob = tf.constant(np.full((1, 6), 2.0, dtype="float32"))  # far above clamp_max
        boundary = np.zeros((1, 6), dtype=bool)
        boundary[0, [0, 2, 4]] = True

        layer = DeChunkLayer()
        with tf.GradientTape() as tape:
            tape.watch(prob)
            out = layer(
                inner, boundary_prob=prob, boundary_mask=tf.constant(boundary)
            )
            loss = tf.reduce_sum(out)
        grad = np.asarray(tape.gradient(loss, prob))
        np.testing.assert_array_equal(grad, np.zeros_like(grad))

    def test_the_boundary_mask_carries_no_gradient(self):
        """The hard decision is boolean and must stay non-differentiable."""
        rng = np.random.default_rng(705)
        inner = tf.constant(rng.standard_normal((1, 3, 2)).astype("float32"))
        prob = tf.constant(np.full((1, 6), 0.5, dtype="float32"))
        boundary_float = tf.constant(
            np.array([[1.0, 0.0, 1.0, 0.0, 1.0, 0.0]], dtype="float32")
        )

        layer = DeChunkLayer()
        with tf.GradientTape() as tape:
            tape.watch(boundary_float)
            out = layer(
                inner, boundary_prob=prob, boundary_mask=boundary_float
            )
            loss = tf.reduce_sum(out)
        assert tape.gradient(loss, boundary_float) is None


# =====================================================================
# Graph / XLA
# =====================================================================


class TestGraphSafety:
    """The scan must survive tracing, not only eager execution.

    ``keras.ops.while_loop`` + ``slice_update`` is exactly the construct a graph
    or XLA backend is most likely to refuse, so this is a guard and not an
    assumption inherited from the step-2 probe.
    """

    @pytest.mark.parametrize("jit", [False, True])
    def test_the_forward_pass_traces_and_matches_eager(self, jit):
        rng = np.random.default_rng(801)
        seq_len, inner_len, d_model = 14, 6, 5
        inner_np = rng.standard_normal((3, inner_len, d_model)).astype("float32")
        prob_np = rng.uniform(0.05, 0.95, (3, seq_len)).astype("float32")
        boundary_np = _boundaries(rng, 3, seq_len, inner_len)
        mask_np = _padding_mask(3, seq_len, [14, 9, 6])
        layer = DeChunkLayer()

        @tf.function(jit_compile=jit)
        def traced(x, p, b, m):
            return layer(x, boundary_prob=p, boundary_mask=b, mask=m)

        eager = _run(layer, inner_np, prob_np, boundary_np, mask_np)
        graph = np.asarray(
            traced(
                keras.ops.convert_to_tensor(inner_np),
                keras.ops.convert_to_tensor(prob_np),
                keras.ops.convert_to_tensor(boundary_np),
                keras.ops.convert_to_tensor(mask_np),
            )
        ).astype(np.float64)
        if jit:
            # XLA fuses the multiply-add of the recurrence, so graph and eager
            # are NOT bit-identical here -- measured 1.19e-07 worst case, one
            # float32 ulp at this scale. Bit-equality is asserted only for the
            # un-jitted trace, where no reassociation happens; the jitted arm is
            # graded at the module's derived bound like any other numeric arm.
            np.testing.assert_allclose(
                graph, eager, rtol=0, atol=ema_atol(inner_np)
            )
        else:
            np.testing.assert_array_equal(graph, eager)

    def test_xla_reassociates_the_recurrence_but_only_within_the_bound(self):
        """The measurement behind the branch above, stated as its own claim.

        ``jit_compile=True`` is not bit-identical to eager on this recurrence,
        and pretending otherwise would either make the graph arm flaky or force a
        tolerance nobody derived. The worst observed deviation is one float32 ulp
        at the data's own scale, which is inside the module's derived bound.
        """
        rng = np.random.default_rng(803)
        inner_np = rng.standard_normal((3, 64, 8)).astype("float32")
        prob_np = rng.uniform(0.05, 0.95, (3, 128)).astype("float32")
        boundary_np = _boundaries(rng, 3, 128, 64)
        layer = DeChunkLayer()

        @tf.function(jit_compile=True)
        def traced(x, p, b):
            return layer(x, boundary_prob=p, boundary_mask=b)

        graph = np.asarray(
            traced(
                keras.ops.convert_to_tensor(inner_np),
                keras.ops.convert_to_tensor(prob_np),
                keras.ops.convert_to_tensor(boundary_np),
            )
        ).astype(np.float64)
        eager = _run(layer, inner_np, prob_np, boundary_np)
        deviation = float(np.max(np.abs(graph - eager)))
        assert deviation < ema_atol(inner_np)
        # And both are within the bound of the float64 oracle, which is the claim
        # that matters -- the XLA path is not a second, unvalidated algorithm.
        expected = dechunk_reference(inner_np, prob_np, boundary_np)
        np.testing.assert_allclose(graph, expected, rtol=0, atol=ema_atol(inner_np))

    def test_the_backward_pass_compiles_under_xla(self):
        """The gradient, not only the forward pass, must survive ``jit_compile``.

        This is the arm that pins ``maximum_iterations``: without it the XLA
        backward pass raises ``XLA compilation requires a fixed tensor list
        size``, and the forward-only graph test would not have seen it.
        """
        rng = np.random.default_rng(804)
        inner = tf.constant(rng.standard_normal((2, 6, 3)).astype("float32"))
        prob = tf.constant(rng.uniform(0.1, 0.9, (2, 10)).astype("float32"))
        boundary = tf.constant(_boundaries(rng, 2, 10, 6))
        layer = DeChunkLayer()

        @tf.function(jit_compile=True)
        def traced_gradient(x, p, b):
            with tf.GradientTape() as tape:
                tape.watch(x)
                loss = tf.reduce_sum(
                    layer(x, boundary_prob=p, boundary_mask=b)
                )
            return tape.gradient(loss, x)

        grad = np.asarray(traced_gradient(inner, prob, boundary))
        assert np.isfinite(grad).all()
        assert float(np.min(np.abs(grad))) > 0.0

    def test_the_forward_pass_traces_at_unknown_batch_and_sequence_length(self):
        """``(None, None, D)`` -- the shape the recursive model will build at."""
        layer = DeChunkLayer()

        @tf.function(
            input_signature=[
                tf.TensorSpec((None, None, 3), tf.float32),
                tf.TensorSpec((None, None), tf.float32),
                tf.TensorSpec((None, None), tf.bool),
            ]
        )
        def traced(x, p, b):
            return layer(x, boundary_prob=p, boundary_mask=b)

        rng = np.random.default_rng(802)
        for seq_len, inner_len in ((7, 4), (19, 11)):
            inner = rng.standard_normal((2, inner_len, 3)).astype("float32")
            prob = rng.uniform(0.05, 0.95, (2, seq_len)).astype("float32")
            boundary = _boundaries(rng, 2, seq_len, inner_len)
            out = np.asarray(
                traced(
                    keras.ops.convert_to_tensor(inner),
                    keras.ops.convert_to_tensor(prob),
                    keras.ops.convert_to_tensor(boundary),
                )
            ).astype(np.float64)
            assert out.shape == (2, seq_len, 3)
            np.testing.assert_array_equal(out, _run(layer, inner, prob, boundary))


# =====================================================================
# Serialization
# =====================================================================


class TestSerialization:
    """The ``.keras`` round trip, compared on VALUES with ``training=False``."""

    def test_keras_save_load_round_trip_on_values(self, tmp_path):
        seq_len, inner_len, d_model = 13, 5, 4
        inner_input = keras.Input(shape=(inner_len, d_model), dtype="float32")
        prob_input = keras.Input(shape=(seq_len,), dtype="float32")
        boundary_input = keras.Input(shape=(seq_len,), dtype="bool")
        output = DeChunkLayer(name="dechunker")(
            inner_input, boundary_prob=prob_input, boundary_mask=boundary_input
        )
        model = keras.Model(
            inputs=[inner_input, prob_input, boundary_input], outputs=output
        )

        rng = np.random.default_rng(901)
        inner = rng.standard_normal((3, inner_len, d_model)).astype("float32")
        prob = rng.uniform(0.05, 0.95, (3, seq_len)).astype("float32")
        boundary = _boundaries(rng, 3, seq_len, inner_len)

        before = keras.ops.convert_to_numpy(
            model([inner, prob, boundary], training=False)
        )

        path = tmp_path / "dechunk_layer.keras"
        model.save(path)
        loaded = keras.models.load_model(path)
        after = keras.ops.convert_to_numpy(
            loaded([inner, prob, boundary], training=False)
        )

        np.testing.assert_allclose(before, after, rtol=0, atol=0.0)
        assert loaded.get_layer("dechunker").clamp_min == 1e-4
        assert loaded.get_layer("dechunker").clamp_max == 1.0 - 1e-4

        # DIFFER twin: the loaded model is live, not a replayed cache.
        other = rng.standard_normal((3, inner_len, d_model)).astype("float32")
        other_out = keras.ops.convert_to_numpy(
            loaded([other, prob, boundary], training=False)
        )
        assert float(np.max(np.abs(other_out - after))) > 1e-2

    def test_the_restored_clamp_is_the_saved_one_and_not_a_default(self, tmp_path):
        """Anti-vacuity for the round trip: different bounds give a different model."""
        seq_len, inner_len, d_model = 10, 4, 3
        rng = np.random.default_rng(902)
        inner = rng.standard_normal((2, inner_len, d_model)).astype("float32")
        prob = rng.uniform(0.0, 1.0, (2, seq_len)).astype("float32")
        boundary = _boundaries(rng, 2, seq_len, inner_len)

        outputs = {}
        for clamp_min, clamp_max in ((1e-4, 1.0 - 1e-4), (0.4, 0.6)):
            inner_input = keras.Input(shape=(inner_len, d_model), dtype="float32")
            prob_input = keras.Input(shape=(seq_len,), dtype="float32")
            boundary_input = keras.Input(shape=(seq_len,), dtype="bool")
            output = DeChunkLayer(
                clamp_min=clamp_min, clamp_max=clamp_max, name="dechunker"
            )(inner_input, boundary_prob=prob_input, boundary_mask=boundary_input)
            model = keras.Model(
                inputs=[inner_input, prob_input, boundary_input], outputs=output
            )
            path = tmp_path / f"dechunk_{clamp_min}.keras"
            model.save(path)
            loaded = keras.models.load_model(path)
            assert loaded.get_layer("dechunker").clamp_min == clamp_min
            assert loaded.get_layer("dechunker").clamp_max == clamp_max
            outputs[clamp_min] = keras.ops.convert_to_numpy(
                loaded([inner, prob, boundary], training=False)
            )

        assert float(np.max(np.abs(outputs[1e-4] - outputs[0.4]))) > 1e-3
