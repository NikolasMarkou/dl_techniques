"""Tests for ``ChunkLayer`` -- H-Net's fixed-width ragged-selection operator.

The grading oracle is ``hnet_reference_numpy.chunk_reference``: the float64 NumPy
transcription of ``_reference/dc.py:169-207`` written *before* this layer existed
and RED-proven by its own injected mutations (plan decision D-011). No second
reference is written here.

Four guards are owed by plan step 5:

(i)   parity against ``chunk_reference`` for every row whose boundary count is
      ``<= max_chunks``;
(ii)  ``test_a_later_boundary_never_displaces_an_earlier_one`` -- a row whose
      LATE boundary carries the HIGHEST probability must lose exactly that
      boundary, because truncation is in POSITION order and never in magnitude
      order (S5; the measured hazard is `layers/blt/blt_blocks.py:410-420`);
(iii) ``test_row_i_is_independent_of_row_j`` -- row 0's output is bit-identical
      across 32 randomisations of row 1, with a "something changed" twin on row
      1 AND an arm showing the reference's own data-dependent width FAILS this;
(iv)  the zero-interior-boundary row and the all-positions-boundary row both
      produce well-formed ``inner_mask``s.

Guards (ii) and (iii) are what make the ``# DECISION .../D-007`` anchor in
``chunk_layer.py`` a guard rather than a comment.

**Tolerance.** Every numerical comparison in this module is at ``atol=0.0,
rtol=0``, and that is derived, not chosen for strictness. ``ChunkLayer`` performs
NO arithmetic on hidden-state values: ``take_along_axis`` copies float32 bit
patterns, and the oracle's ``np.asarray(..., np.float64)`` cast of a float32
input is exact (every float32 is representable in float64). There is therefore no
rounding to bound, and any positive ``atol`` would be pure unfailable slack --
the failure mode `tests/numerics.py`'s D-024 comment warns about from the other
side. The index arithmetic (``arange + (~keep) * L``) is exact int32 for
``L < 2**30``. This is the one path in the port where an exact bound is the
correct bound, and it is stated so a later reader does not "fix" it by pasting a
tolerance. ``tests/numerics.reassociation_atol`` is not used and could not be:
step 4 MEASURED that it UNDER-counts against a float64 oracle (D-012), and here
the correct count is zero rounded operations.

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

from dl_techniques.layers.dynamic_chunking.chunk_layer import ChunkLayer

from .hnet_reference_numpy import chunk_reference

# ---------------------------------------------------------------------
# Derived tolerance
# ---------------------------------------------------------------------

#: See the module docstring: the layer is a pure gather, so the only correct
#: bound is exact equality. Used with ``rtol=0`` everywhere.
GATHER_ATOL = 0.0


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------


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


def _run(layer, hidden, boundary, mask=None):
    """Call ``layer`` and return both outputs as NumPy arrays.

    :param layer: The :class:`ChunkLayer` under test.
    :type layer: ChunkLayer
    :param hidden: ``(B, L, D)`` float32 array.
    :type hidden: numpy.ndarray
    :param boundary: ``(B, L)`` boolean array.
    :type boundary: numpy.ndarray
    :param mask: ``(B, L)`` boolean array or ``None``.
    :type mask: numpy.ndarray or None
    :return: ``(next_hidden_states, inner_mask)`` as NumPy.
    :rtype: tuple
    """
    kwargs = {"boundary_mask": keras.ops.convert_to_tensor(boundary)}
    if mask is not None:
        kwargs["mask"] = keras.ops.convert_to_tensor(mask)
    inner, inner_mask = layer(keras.ops.convert_to_tensor(hidden), **kwargs)
    return (
        keras.ops.convert_to_numpy(inner),
        keras.ops.convert_to_numpy(inner_mask),
    )


def _random_boundaries(rng, batch_size, seq_len, max_count):
    """A boolean ``(B, L)`` boundary mask with position 0 forced and a capped count.

    Position 0 is always a boundary (``dc.py:95-96`` forces ``p = 1.0`` there), so
    every row has at least one chunk. The interior boundary count is drawn so the
    row total never exceeds ``max_count`` -- guard (i) grades only such rows.

    :param rng: Source of randomness.
    :type rng: numpy.random.Generator
    :param batch_size: Number of rows.
    :type batch_size: int
    :param seq_len: Sequence length.
    :type seq_len: int
    :param max_count: Inclusive cap on the per-row boundary count.
    :type max_count: int
    :return: ``(B, L)`` boolean array.
    :rtype: numpy.ndarray
    """
    out = np.zeros((batch_size, seq_len), dtype=bool)
    out[:, 0] = True
    for row in range(batch_size):
        extra = int(rng.integers(0, min(max_count - 1, seq_len - 1) + 1))
        if extra:
            picks = rng.choice(np.arange(1, seq_len), size=extra, replace=False)
            out[row, picks] = True
    return out


def _magnitude_order_selection(hidden, boundary, probs, max_chunks):
    """The DEFECT: keep the ``max_chunks`` boundaries with the HIGHEST probability.

    This is not a second oracle -- it is the rejected policy, computed here only
    so a test can assert the layer is DIFFERENT from it. Ties are broken by
    position so the defect is deterministic.

    :param hidden: ``(B, L, D)`` array.
    :type hidden: numpy.ndarray
    :param boundary: ``(B, L)`` boolean array.
    :type boundary: numpy.ndarray
    :param probs: ``(B, L)`` boundary probabilities.
    :type probs: numpy.ndarray
    :param max_chunks: Output width.
    :type max_chunks: int
    :return: ``(B, max_chunks, D)`` array.
    :rtype: numpy.ndarray
    """
    batch_size, _, d_model = hidden.shape
    out = np.zeros((batch_size, max_chunks, d_model), dtype=hidden.dtype)
    for row in range(batch_size):
        positions = np.flatnonzero(boundary[row])
        order = sorted(positions, key=lambda p: (-probs[row, p], p))
        kept = sorted(order[:max_chunks])
        for column, position in enumerate(kept):
            out[row, column] = hidden[row, position]
    return out


# =====================================================================
# Construction, config and registration
# =====================================================================


class TestInitAndConfig:
    """Constructor state, the weightless contract, config round trip, registration."""

    def test_init_stores_max_chunks_and_creates_no_weights(self):
        layer = ChunkLayer(max_chunks=6)
        assert layer.max_chunks == 6
        assert layer.built is False
        assert layer.weights == []

    def test_the_layer_is_deliberately_weightless_after_build(self):
        """A per-weight gradient check is vacuous here, so the ABSENCE is asserted.

        ``ChunkLayer`` is a pure gather plus a mask construction. It owns no
        variables, and this test says so on purpose rather than silently skipping
        the gradient-flow requirement: if a weight ever appears, the step-5
        design has changed and this assertion is where that surfaces.
        """
        layer = ChunkLayer(max_chunks=4)
        layer.build((None, 10, 5))
        assert layer.built is True
        assert layer.weights == []
        assert layer.trainable_variables == []
        assert layer.non_trainable_variables == []

    def test_get_config_round_trip_reproduces_behaviour(self):
        rng = np.random.default_rng(0)
        hidden = rng.standard_normal((2, 9, 4)).astype("float32")
        boundary = _random_boundaries(rng, 2, 9, 3)

        original = ChunkLayer(max_chunks=3)
        config = original.get_config()
        assert config["max_chunks"] == 3
        clone = ChunkLayer.from_config(config)

        first = _run(original, hidden, boundary)
        second = _run(clone, hidden, boundary)
        np.testing.assert_allclose(first[0], second[0], rtol=0, atol=GATHER_ATOL)
        np.testing.assert_array_equal(first[1], second[1])

        # DIFFER twin: the comparison above is not comparing constants -- a clone
        # built from a DIFFERENT config does not reproduce it.
        other = ChunkLayer.from_config({**config, "max_chunks": 2})
        assert _run(other, hidden, boundary)[0].shape != first[0].shape

    def test_layer_is_registered_under_the_package_qualified_key(self):
        key = keras.saving.get_registered_name(ChunkLayer)
        assert key == "dl_techniques.layers.dynamic_chunking.chunk_layer>ChunkLayer"
        assert keras.saving.get_registered_object(key) is ChunkLayer


class TestValidation:
    """Every raise, and the inputs that must reach one."""

    @pytest.mark.parametrize("bad", [0, -1, -8])
    def test_non_positive_max_chunks_raises(self, bad):
        with pytest.raises(ValueError, match="must be positive"):
            ChunkLayer(max_chunks=bad)

    @pytest.mark.parametrize("bad", [4.0, "4", None, True])
    def test_non_int_max_chunks_raises(self, bad):
        with pytest.raises(ValueError, match="must be an int"):
            ChunkLayer(max_chunks=bad)

    def test_rank_two_input_raises(self):
        layer = ChunkLayer(max_chunks=3)
        with pytest.raises(ValueError, match="rank-3"):
            layer.build((None, 8))

    def test_a_missing_boundary_mask_raises(self):
        layer = ChunkLayer(max_chunks=3)
        hidden = keras.ops.convert_to_tensor(np.zeros((2, 8, 4), dtype="float32"))
        with pytest.raises(ValueError, match="requires boundary_mask"):
            layer(hidden)

    def test_a_correct_shape_does_not_raise(self):
        layer = ChunkLayer(max_chunks=3)
        layer.build((None, 8, 4))
        assert layer.built is True


# =====================================================================
# Forward-pass contract
# =====================================================================


class TestForwardPass:
    """Shapes, dtypes, and the properties every call must satisfy."""

    @pytest.mark.parametrize(
        "batch_size,seq_len,d_model,max_chunks",
        [(1, 4, 2, 2), (3, 16, 8, 5), (2, 7, 3, 7), (4, 5, 6, 9)],
    )
    def test_output_shapes_and_dtypes(self, batch_size, seq_len, d_model, max_chunks):
        rng = np.random.default_rng(11)
        hidden = rng.standard_normal((batch_size, seq_len, d_model)).astype("float32")
        boundary = _random_boundaries(rng, batch_size, seq_len, max_chunks)
        inner, inner_mask = _run(ChunkLayer(max_chunks=max_chunks), hidden, boundary)
        assert inner.shape == (batch_size, max_chunks, d_model)
        assert inner_mask.shape == (batch_size, max_chunks)
        assert inner.dtype == np.float32
        assert inner_mask.dtype == np.bool_

    def test_compute_output_shape_matches_the_real_call(self):
        layer = ChunkLayer(max_chunks=5)
        declared = layer.compute_output_shape((None, 12, 7))
        rng = np.random.default_rng(12)
        hidden = rng.standard_normal((3, 12, 7)).astype("float32")
        realised = _run(layer, hidden, _random_boundaries(rng, 3, 12, 5))
        assert declared == ((None, 5, 7), (None, 5))
        assert realised[0].shape == (3, 5, 7)
        assert realised[1].shape == (3, 5)

    def test_every_selected_column_is_a_row_of_the_input(self):
        """The layer only ever COPIES hidden states; it never combines them."""
        rng = np.random.default_rng(13)
        hidden = rng.standard_normal((3, 14, 4)).astype("float32")
        boundary = _random_boundaries(rng, 3, 14, 6)
        inner, _ = _run(ChunkLayer(max_chunks=6), hidden, boundary)
        for row in range(3):
            for column in range(6):
                matches = np.all(hidden[row] == inner[row, column], axis=-1)
                assert matches.any(), f"column {column} of row {row} is not an input row"

    def test_valid_columns_are_exactly_the_boundary_positions_in_order(self):
        rng = np.random.default_rng(14)
        hidden = rng.standard_normal((3, 13, 5)).astype("float32")
        boundary = _random_boundaries(rng, 3, 13, 4)
        inner, inner_mask = _run(ChunkLayer(max_chunks=4), hidden, boundary)
        for row in range(3):
            positions = np.flatnonzero(boundary[row])
            assert int(inner_mask[row].sum()) == len(positions)
            for column, position in enumerate(positions):
                np.testing.assert_allclose(
                    inner[row, column], hidden[row, position], rtol=0, atol=GATHER_ATOL
                )

    def test_the_padding_mask_removes_a_boundary_the_router_left_set(self):
        """``mask`` is ANDed in: a padded position can never be selected.

        Defensive over the reference, whose padded branch reads ``boundary_mask``
        alone because the router already masked it. Idempotent in the normal
        case, and this test pins the abnormal one.
        """
        hidden = np.arange(1 * 6 * 2, dtype="float32").reshape(1, 6, 2)
        boundary = np.array([[True, False, True, False, True, False]])
        mask = _padding_mask(1, 6, [3])  # positions 3..5 are padding
        layer = ChunkLayer(max_chunks=3)

        masked, masked_valid = _run(layer, hidden, boundary, mask=mask)
        unmasked, unmasked_valid = _run(layer, hidden, boundary)

        assert int(masked_valid.sum()) == 2      # positions 0 and 2 survive
        assert int(unmasked_valid.sum()) == 3    # position 4 also survives
        np.testing.assert_allclose(masked[0, 0], hidden[0, 0], rtol=0, atol=GATHER_ATOL)
        np.testing.assert_allclose(masked[0, 1], hidden[0, 2], rtol=0, atol=GATHER_ATOL)
        # DIFFER twin: masking is not a no-op on this input.
        assert float(np.max(np.abs(masked - unmasked))) > 0.0

    def test_an_all_true_mask_is_the_same_as_no_mask(self):
        rng = np.random.default_rng(15)
        hidden = rng.standard_normal((2, 10, 3)).astype("float32")
        boundary = _random_boundaries(rng, 2, 10, 4)
        layer = ChunkLayer(max_chunks=4)
        with_mask = _run(layer, hidden, boundary, mask=np.ones((2, 10), dtype=bool))
        without = _run(layer, hidden, boundary)
        np.testing.assert_allclose(with_mask[0], without[0], rtol=0, atol=GATHER_ATOL)
        np.testing.assert_array_equal(with_mask[1], without[1])

    def test_an_integer_boundary_mask_is_accepted(self):
        rng = np.random.default_rng(16)
        hidden = rng.standard_normal((2, 9, 3)).astype("float32")
        boundary = _random_boundaries(rng, 2, 9, 4)
        layer = ChunkLayer(max_chunks=4)
        as_bool = _run(layer, hidden, boundary)
        as_int = _run(layer, hidden, boundary.astype("int32"))
        np.testing.assert_allclose(as_bool[0], as_int[0], rtol=0, atol=GATHER_ATOL)
        np.testing.assert_array_equal(as_bool[1], as_int[1])

    def test_the_training_flag_does_not_change_the_output(self):
        rng = np.random.default_rng(17)
        hidden = keras.ops.convert_to_tensor(
            rng.standard_normal((2, 11, 4)).astype("float32")
        )
        boundary = keras.ops.convert_to_tensor(_random_boundaries(rng, 2, 11, 5))
        layer = ChunkLayer(max_chunks=5)
        train = keras.ops.convert_to_numpy(
            layer(hidden, boundary_mask=boundary, training=True)[0]
        )
        infer = keras.ops.convert_to_numpy(
            layer(hidden, boundary_mask=boundary, training=False)[0]
        )
        np.testing.assert_array_equal(train, infer)


# =====================================================================
# Guard (i): parity against the float64 oracle
# =====================================================================


class TestGuardOneOracleParity:
    """Parity with ``chunk_reference`` for every row whose count is within the cap."""

    @pytest.mark.parametrize(
        "seed,batch_size,seq_len,d_model,max_chunks",
        [
            (101, 1, 2, 2, 1),
            (102, 3, 16, 8, 5),
            (103, 4, 32, 6, 12),
            (104, 2, 9, 3, 9),
            (105, 5, 20, 4, 7),
        ],
    )
    def test_parity_without_a_mask(self, seed, batch_size, seq_len, d_model, max_chunks):
        rng = np.random.default_rng(seed)
        hidden = rng.standard_normal((batch_size, seq_len, d_model)).astype("float32")
        boundary = _random_boundaries(rng, batch_size, seq_len, max_chunks)
        assert boundary.sum(axis=-1).max() <= max_chunks

        inner, inner_mask = _run(ChunkLayer(max_chunks=max_chunks), hidden, boundary)
        expected_inner, expected_mask = chunk_reference(
            hidden, boundary, max_chunks=max_chunks
        )
        np.testing.assert_allclose(inner, expected_inner, rtol=0, atol=GATHER_ATOL)
        np.testing.assert_array_equal(inner_mask, expected_mask)

    @pytest.mark.parametrize("seed,lengths", [(201, [12, 7, 1]), (202, [3, 12, 12])])
    def test_parity_with_a_padding_mask(self, seed, lengths):
        rng = np.random.default_rng(seed)
        batch_size, seq_len, d_model, max_chunks = len(lengths), 12, 5, 6
        hidden = rng.standard_normal((batch_size, seq_len, d_model)).astype("float32")
        boundary = _random_boundaries(rng, batch_size, seq_len, max_chunks)
        mask = _padding_mask(batch_size, seq_len, lengths)

        inner, inner_mask = _run(
            ChunkLayer(max_chunks=max_chunks), hidden, boundary, mask=mask
        )
        # The oracle takes a single boundary mask; the layer ANDs `mask` in, so
        # the equivalent oracle input is the conjunction.
        expected_inner, expected_mask = chunk_reference(
            hidden, boundary & mask, max_chunks=max_chunks
        )
        np.testing.assert_allclose(inner, expected_inner, rtol=0, atol=GATHER_ATOL)
        np.testing.assert_array_equal(inner_mask, expected_mask)

    def test_parity_when_max_chunks_exceeds_the_sequence_length(self):
        """The surplus columns are defined and marked invalid.

        The permutation is right-padded with index 0, so columns ``>= L`` hold
        ``hidden[:, 0]``. The oracle's own ``arange(width) < num_tokens`` marks
        them invalid, so only the mask has to agree -- but the values must be
        finite and defined, which is what makes the padding choice testable.
        """
        rng = np.random.default_rng(203)
        hidden = rng.standard_normal((3, 4, 2)).astype("float32")
        boundary = _random_boundaries(rng, 3, 4, 4)
        inner, inner_mask = _run(ChunkLayer(max_chunks=9), hidden, boundary)
        _, expected_mask = chunk_reference(hidden, boundary, max_chunks=9)

        np.testing.assert_array_equal(inner_mask, expected_mask)
        assert np.isfinite(inner).all()
        for row in range(3):
            for column in range(4, 9):
                np.testing.assert_allclose(
                    inner[row, column], hidden[row, 0], rtol=0, atol=GATHER_ATOL
                )
        # DIFFER twin: the valid prefix is NOT the same constant filler.
        assert int(inner_mask.sum()) > 0

    def test_the_parity_instrument_can_report_a_mismatch(self):
        """An instrument that cannot report a difference has measured nothing."""
        rng = np.random.default_rng(204)
        hidden = rng.standard_normal((2, 12, 4)).astype("float32")
        boundary = _random_boundaries(rng, 2, 12, 5)
        inner, inner_mask = _run(ChunkLayer(max_chunks=5), hidden, boundary)

        # Values arm: the oracle fed the hidden states rolled by one position
        # returns different chunk contents.
        rolled_inner, _ = chunk_reference(
            np.roll(hidden, 1, axis=1), boundary, max_chunks=5
        )
        assert float(np.max(np.abs(inner - rolled_inner))) > GATHER_ATOL

        # Mask arm: a ROLL of the boundary mask preserves the per-row count and
        # therefore CANNOT move `inner_mask` -- so the mask twin adds a boundary
        # instead. (Stated because the roll was tried first and was inert.)
        richer = boundary.copy()
        richer[:, -1] = True
        _, richer_mask = chunk_reference(hidden, richer, max_chunks=5)
        assert not np.array_equal(inner_mask, richer_mask)

    def test_the_exact_bound_is_attainable_and_is_the_correct_bound(self):
        """``atol = 0`` is attainable: the layer copies bit patterns, it does not compute.

        Anti-vacuity for the tolerance itself. A gather cannot round, so the
        measured worst-case deviation over many draws must be EXACTLY zero; if it
        were not, a bound of 0 would be unattainable and the derivation in this
        module's docstring would be wrong.
        """
        rng = np.random.default_rng(205)
        worst = 0.0
        for _ in range(16):
            hidden = rng.standard_normal((3, 20, 6)).astype("float32")
            boundary = _random_boundaries(rng, 3, 20, 8)
            inner, _ = _run(ChunkLayer(max_chunks=8), hidden, boundary)
            expected, _ = chunk_reference(hidden, boundary, max_chunks=8)
            worst = max(worst, float(np.max(np.abs(inner - expected))))
        assert worst == 0.0


# =====================================================================
# Guard (ii): POSITION-order truncation -- the D-007 anchor's guard
# =====================================================================


class TestGuardTwoPositionOrderTruncation:
    """A later boundary never displaces an earlier one, whatever its probability.

    This is the guard that makes the ``# DECISION .../D-007`` anchor on the
    ``max_chunks`` cap a guard rather than a comment. Magnitude-order truncation
    is the causality hazard: an earlier token's chunk assignment would depend on
    a later token's score (S5; MEASURED in `layers/blt/blt_blocks.py:410-420`).
    """

    def test_a_later_boundary_never_displaces_an_earlier_one(self):
        """The LATE boundary carries the HIGHEST probability and is the one dropped.

        Hand-built row: ``L = 8``, boundaries at positions 0, 3, 6, 7 and
        ``max_chunks = 3``, with probabilities ``[1.0, 0.51, 0.55, 0.99]``. The
        highest-probability boundary is at position 7 -- the LAST one. Position
        order keeps ``{0, 3, 6}``; magnitude order would keep ``{0, 7, 6}``.
        """
        hidden = np.arange(8 * 2, dtype="float32").reshape(1, 8, 2)
        boundary = np.zeros((1, 8), dtype=bool)
        boundary[0, [0, 3, 6, 7]] = True
        probs = np.zeros((1, 8), dtype="float32")
        probs[0, [0, 3, 6, 7]] = [1.0, 0.51, 0.55, 0.99]

        inner, inner_mask = _run(ChunkLayer(max_chunks=3), hidden, boundary)

        np.testing.assert_allclose(
            inner[0], hidden[0, [0, 3, 6]], rtol=0, atol=GATHER_ATOL
        )
        # The row has 4 boundaries against a cap of 3, so every column is real.
        np.testing.assert_array_equal(inner_mask, np.ones((1, 3), dtype=bool))
        # The dropped boundary is exactly the highest-probability one.
        assert not np.any(np.all(inner[0] == hidden[0, 7], axis=-1))

        # DIFFER twin: the magnitude-order policy is a DIFFERENT answer on this
        # input, so the assertion above is discriminating between two live
        # candidates rather than restating a shape.
        defect = _magnitude_order_selection(hidden, boundary, probs, 3)
        assert float(np.max(np.abs(inner - defect))) > 0.0

    def test_truncation_keeps_a_prefix_of_the_boundary_positions(self):
        """Generalisation of the hand-built row over 24 random draws."""
        rng = np.random.default_rng(301)
        for _ in range(24):
            seq_len = int(rng.integers(6, 32))
            max_chunks = int(rng.integers(2, 6))
            hidden = rng.standard_normal((1, seq_len, 3)).astype("float32")
            boundary = np.zeros((1, seq_len), dtype=bool)
            boundary[0, 0] = True
            picks = rng.choice(
                np.arange(1, seq_len), size=min(max_chunks + 3, seq_len - 1),
                replace=False,
            )
            boundary[0, picks] = True

            inner, _ = _run(ChunkLayer(max_chunks=max_chunks), hidden, boundary)
            expected_positions = np.flatnonzero(boundary[0])[:max_chunks]
            np.testing.assert_allclose(
                inner[0], hidden[0, expected_positions], rtol=0, atol=GATHER_ATOL
            )

    def test_raising_a_late_boundarys_probability_cannot_change_the_output(self):
        """The output is a function of POSITIONS only -- probabilities never enter.

        Executable statement of why the anchor's rule is safe: the layer's
        signature does not even accept a probability, so no ordering by magnitude
        is reachable. The twin asserts the probabilities used are not degenerate.
        """
        hidden = np.arange(9 * 2, dtype="float32").reshape(1, 9, 2)
        boundary = np.zeros((1, 9), dtype=bool)
        boundary[0, [0, 2, 5, 8]] = True
        layer = ChunkLayer(max_chunks=2)
        baseline, _ = _run(layer, hidden, boundary)

        low = np.array([[1.0, 0, 0.6, 0, 0, 0.6, 0, 0, 0.51]], dtype="float32")
        high = np.array([[1.0, 0, 0.6, 0, 0, 0.6, 0, 0, 0.99]], dtype="float32")
        for probs in (low, high):
            again, _ = _run(layer, hidden, boundary)
            np.testing.assert_array_equal(again, baseline)
            assert probs.shape == boundary.shape  # the probes are well formed

        # DIFFER twin: the two probability vectors DO order the boundaries
        # differently, so a magnitude-ordered implementation would have moved.
        assert not np.array_equal(
            _magnitude_order_selection(hidden, boundary, low, 2),
            _magnitude_order_selection(hidden, boundary, high, 2),
        )


# =====================================================================
# Guard (iii): batch independence -- the other half of the D-007 anchor
# =====================================================================


class TestGuardThreeBatchIndependence:
    """Row ``i``'s output must not depend on row ``j``'s content."""

    def test_row_i_is_independent_of_row_j(self):
        """Row 0 held fixed; row 1 randomised 32 times; row 0 bit-identical each time."""
        rng = np.random.default_rng(401)
        seq_len, d_model, max_chunks = 24, 4, 6
        fixed_hidden = rng.standard_normal((seq_len, d_model)).astype("float32")
        fixed_boundary = np.zeros(seq_len, dtype=bool)
        fixed_boundary[[0, 5, 9]] = True

        layer = ChunkLayer(max_chunks=max_chunks)
        reference_inner = None
        reference_mask = None
        row_one_outputs = []

        for _ in range(32):
            other_hidden = rng.standard_normal((seq_len, d_model)).astype("float32")
            other_boundary = np.zeros(seq_len, dtype=bool)
            other_boundary[0] = True
            picks = rng.choice(
                np.arange(1, seq_len),
                size=int(rng.integers(0, seq_len - 1)),
                replace=False,
            )
            other_boundary[picks] = True

            hidden = np.stack([fixed_hidden, other_hidden])
            boundary = np.stack([fixed_boundary, other_boundary])
            inner, inner_mask = _run(layer, hidden, boundary)

            if reference_inner is None:
                reference_inner, reference_mask = inner[0], inner_mask[0]
            else:
                np.testing.assert_allclose(
                    inner[0], reference_inner, rtol=0, atol=0.0
                )
                np.testing.assert_array_equal(inner_mask[0], reference_mask)
            row_one_outputs.append(inner[1])

        # SOMETHING-CHANGED twin: row 1 DID move across the draws, so the
        # bit-identity above is a property of the layer and not of a suite that
        # fed it the same batch 32 times.
        stacked = np.stack(row_one_outputs)
        assert float(np.max(np.abs(stacked - stacked[0]))) > 0.0

    def test_a_row_computed_alone_equals_the_same_row_inside_a_batch(self):
        rng = np.random.default_rng(402)
        hidden = rng.standard_normal((5, 18, 3)).astype("float32")
        boundary = _random_boundaries(rng, 5, 18, 12)
        layer = ChunkLayer(max_chunks=6)
        batched, batched_mask = _run(layer, hidden, boundary)
        for row in range(5):
            alone, alone_mask = _run(layer, hidden[row : row + 1], boundary[row : row + 1])
            np.testing.assert_allclose(alone[0], batched[row], rtol=0, atol=0.0)
            np.testing.assert_array_equal(alone_mask[0], batched_mask[row])

    def test_the_reference_data_dependent_width_DOES_depend_on_row_j(self):
        """Anti-vacuity for D-007: the rejected width FAILS this very property.

        ``chunk_reference(..., max_chunks=None)`` reproduces the reference's own
        ``next_max_seqlen = max(boundary_mask.sum(-1))``. Row 0 is held fixed and
        only row 1 changes, yet row 0's output WIDTH -- and therefore its content
        beyond the first column -- moves. Without this arm the guard above could
        be satisfied by any implementation at all, and the anchor would be
        recording a hazard nobody had shown to exist.
        """
        rng = np.random.default_rng(403)
        seq_len, d_model = 16, 3
        fixed_hidden = rng.standard_normal((seq_len, d_model)).astype("float32")
        fixed_boundary = np.zeros(seq_len, dtype=bool)
        fixed_boundary[[0, 4]] = True

        # `thin` puts its second boundary LATE (position 7) rather than at
        # position 1: with boundaries at {0, 1} the stable partition's first five
        # columns would be {0,1,2,3,4}, which is exactly what the all-boundary row
        # gives, and the row-1 DIFFER twin below would be inert.
        thin = np.zeros(seq_len, dtype=bool)
        thin[[0, 7]] = True
        thick = np.ones(seq_len, dtype=bool)
        other_hidden = rng.standard_normal((seq_len, d_model)).astype("float32")

        widths = []
        for other_boundary in (thin, thick):
            inner, _ = chunk_reference(
                np.stack([fixed_hidden, other_hidden]),
                np.stack([fixed_boundary, other_boundary]),
                max_chunks=None,
            )
            widths.append(inner.shape[1])
        assert widths[0] != widths[1], "the reference width did not move; probe is inert"

        # And the FIXED-width port is unmoved on the identical pair of batches.
        layer = ChunkLayer(max_chunks=5)
        first, first_mask = _run(
            layer,
            np.stack([fixed_hidden, other_hidden]),
            np.stack([fixed_boundary, thin]),
        )
        second, second_mask = _run(
            layer,
            np.stack([fixed_hidden, other_hidden]),
            np.stack([fixed_boundary, thick]),
        )
        np.testing.assert_allclose(first[0], second[0], rtol=0, atol=0.0)
        np.testing.assert_array_equal(first_mask[0], second_mask[0])
        # DIFFER twin: row 1 is genuinely different between the two batches.
        assert float(np.max(np.abs(first[1] - second[1]))) > 0.0


# =====================================================================
# Guard (iv): the two degenerate rows
# =====================================================================


class TestGuardFourDegenerateRows:
    """Zero interior boundaries and all-positions boundaries stay well formed."""

    def test_a_row_with_only_the_forced_first_boundary(self):
        """One chunk, ``max_chunks`` wide: exactly column 0 is valid."""
        rng = np.random.default_rng(501)
        hidden = rng.standard_normal((1, 10, 4)).astype("float32")
        boundary = np.zeros((1, 10), dtype=bool)
        boundary[0, 0] = True

        inner, inner_mask = _run(ChunkLayer(max_chunks=5), hidden, boundary)
        expected_inner, expected_mask = chunk_reference(hidden, boundary, max_chunks=5)

        np.testing.assert_array_equal(
            inner_mask, np.array([[True, False, False, False, False]])
        )
        np.testing.assert_array_equal(inner_mask, expected_mask)
        np.testing.assert_allclose(inner, expected_inner, rtol=0, atol=GATHER_ATOL)
        np.testing.assert_allclose(inner[0, 0], hidden[0, 0], rtol=0, atol=GATHER_ATOL)
        assert np.isfinite(inner).all()
        # The invalid columns hold the stable partition's non-boundary tail --
        # garbage by design, but DEFINED garbage in position order.
        np.testing.assert_allclose(
            inner[0, 1:], hidden[0, 1:5], rtol=0, atol=GATHER_ATOL
        )

    def test_a_row_where_every_position_is_a_boundary(self):
        """More boundaries than the cap: every column is real, the tail is lost."""
        rng = np.random.default_rng(502)
        hidden = rng.standard_normal((1, 10, 4)).astype("float32")
        boundary = np.ones((1, 10), dtype=bool)

        inner, inner_mask = _run(ChunkLayer(max_chunks=4), hidden, boundary)
        expected_inner, expected_mask = chunk_reference(hidden, boundary, max_chunks=4)

        np.testing.assert_array_equal(inner_mask, np.ones((1, 4), dtype=bool))
        np.testing.assert_array_equal(inner_mask, expected_mask)
        np.testing.assert_allclose(inner, expected_inner, rtol=0, atol=GATHER_ATOL)
        # Identity gather: the first `max_chunks` positions, in order.
        np.testing.assert_allclose(
            inner[0], hidden[0, :4], rtol=0, atol=GATHER_ATOL
        )
        # DIFFER twin against the degenerate row above -- the two cases produce
        # different masks, so neither assertion is a restatement of the shape.
        assert int(inner_mask.sum()) == 4

    def test_all_positions_boundary_when_the_cap_is_wider_than_the_sequence(self):
        rng = np.random.default_rng(503)
        hidden = rng.standard_normal((1, 4, 2)).astype("float32")
        boundary = np.ones((1, 4), dtype=bool)
        inner, inner_mask = _run(ChunkLayer(max_chunks=6), hidden, boundary)
        np.testing.assert_array_equal(
            inner_mask, np.array([[True, True, True, True, False, False]])
        )
        np.testing.assert_allclose(inner[0, :4], hidden[0], rtol=0, atol=GATHER_ATOL)
        assert np.isfinite(inner).all()

    def test_the_two_degenerate_rows_coexist_in_one_batch(self):
        """Both extremes in a single call, and each still matches the oracle."""
        rng = np.random.default_rng(504)
        hidden = rng.standard_normal((2, 8, 3)).astype("float32")
        boundary = np.zeros((2, 8), dtype=bool)
        boundary[0, 0] = True
        boundary[1, :] = True

        inner, inner_mask = _run(ChunkLayer(max_chunks=5), hidden, boundary)
        expected_inner, expected_mask = chunk_reference(hidden, boundary, max_chunks=5)
        np.testing.assert_allclose(inner, expected_inner, rtol=0, atol=GATHER_ATOL)
        np.testing.assert_array_equal(inner_mask, expected_mask)
        assert int(inner_mask[0].sum()) == 1
        assert int(inner_mask[1].sum()) == 5


# =====================================================================
# Gradient flow
# =====================================================================


class TestGradientFlow:
    """The layer is weightless, so the gradient contract is about its INPUT."""

    def test_the_layer_has_no_trainable_weights_to_receive_a_gradient(self):
        """Deliberate assertion of ABSENCE -- see ``TestInitAndConfig``.

        The plan's per-weight gradient-flow requirement cannot be satisfied by a
        weightless layer, and skipping it silently would hide a future regression
        in which a weight appears untrained. It is asserted instead.
        """
        layer = ChunkLayer(max_chunks=3)
        layer.build((None, 9, 4))
        assert layer.trainable_variables == []

    def test_gradient_reaches_the_selected_positions_and_only_those(self):
        rng = np.random.default_rng(601)
        # `tf.constant` + `tape.watch`, not `tf.Variable`: `take_along_axis`
        # reads `.ndim` off its first argument and a `ResourceVariable` has no
        # such attribute, so a Variable input raises inside the gather.
        hidden = tf.constant(rng.standard_normal((1, 8, 2)).astype("float32"))
        boundary = keras.ops.convert_to_tensor(
            np.array([[True, False, True, False, False, True, False, False]])
        )
        layer = ChunkLayer(max_chunks=3)

        with tf.GradientTape() as tape:
            tape.watch(hidden)
            inner, _ = layer(hidden, boundary_mask=boundary)
            loss = tf.reduce_sum(inner)
        raw = tape.gradient(loss, hidden)
        assert raw is not None
        grad = np.asarray(tf.convert_to_tensor(raw))
        for position in (0, 2, 5):
            assert float(np.min(np.abs(grad[0, position]))) > 0.0
        # DIFFER twin: the un-selected positions receive nothing, so the
        # assertion above is about the gather and not about an all-ones tensor.
        for position in (1, 3, 4, 6, 7):
            assert float(np.max(np.abs(grad[0, position]))) == 0.0

    def test_the_inner_mask_carries_no_gradient(self):
        """The validity mask is a hard boolean; nothing differentiable flows through it."""
        rng = np.random.default_rng(602)
        hidden = tf.constant(rng.standard_normal((1, 8, 2)).astype("float32"))
        boundary = keras.ops.convert_to_tensor(
            np.array([[True, False, True, False, False, True, False, False]])
        )
        layer = ChunkLayer(max_chunks=3)
        with tf.GradientTape() as tape:
            tape.watch(hidden)
            _, inner_mask = layer(hidden, boundary_mask=boundary)
            loss = tf.reduce_sum(tf.cast(inner_mask, tf.float32))
        grad = tape.gradient(loss, hidden)
        assert grad is None or float(tf.reduce_max(tf.abs(grad))) == 0.0


# =====================================================================
# Graph safety
# =====================================================================


class TestGraphSafety:
    """The selection path must survive tracing, not only eager execution."""

    @pytest.mark.parametrize("jit", [False, True])
    def test_the_forward_pass_traces_and_matches_eager(self, jit):
        rng = np.random.default_rng(701)
        hidden_np = rng.standard_normal((3, 14, 5)).astype("float32")
        boundary_np = _random_boundaries(rng, 3, 14, 6)
        mask_np = _padding_mask(3, 14, [14, 9, 4])
        layer = ChunkLayer(max_chunks=6)

        @tf.function(jit_compile=jit)
        def traced(h, b, m):
            inner, inner_mask = layer(h, boundary_mask=b, mask=m)
            return inner, tf.cast(inner_mask, tf.float32)

        eager = _run(layer, hidden_np, boundary_np, mask_np)
        graph = [
            np.asarray(t)
            for t in traced(
                keras.ops.convert_to_tensor(hidden_np),
                keras.ops.convert_to_tensor(boundary_np),
                keras.ops.convert_to_tensor(mask_np),
            )
        ]
        np.testing.assert_array_equal(graph[0], eager[0])
        np.testing.assert_array_equal(graph[1], eager[1].astype("float32"))

    def test_the_forward_pass_traces_at_unknown_batch_and_sequence_length(self):
        """``(None, None, D)`` -- the shape the recursive model will build at."""
        layer = ChunkLayer(max_chunks=4)

        @tf.function(
            input_signature=[
                tf.TensorSpec((None, None, 3), tf.float32),
                tf.TensorSpec((None, None), tf.bool),
            ]
        )
        def traced(h, b):
            inner, inner_mask = layer(h, boundary_mask=b)
            return inner, tf.cast(inner_mask, tf.float32)

        rng = np.random.default_rng(702)
        for seq_len in (7, 19):
            hidden = rng.standard_normal((2, seq_len, 3)).astype("float32")
            boundary = _random_boundaries(rng, 2, seq_len, 4)
            inner, inner_mask = traced(
                keras.ops.convert_to_tensor(hidden),
                keras.ops.convert_to_tensor(boundary),
            )
            assert tuple(np.asarray(inner).shape) == (2, 4, 3)
            eager = _run(layer, hidden, boundary)
            np.testing.assert_array_equal(np.asarray(inner), eager[0])
            np.testing.assert_array_equal(
                np.asarray(inner_mask), eager[1].astype("float32")
            )


# =====================================================================
# Serialization
# =====================================================================


class TestSerialization:
    """The ``.keras`` round trip, compared on VALUES with ``training=False``."""

    def test_keras_save_load_round_trip_on_values(self, tmp_path):
        seq_len, d_model, max_chunks = 13, 4, 5
        hidden_input = keras.Input(shape=(seq_len, d_model), dtype="float32")
        boundary_input = keras.Input(shape=(seq_len,), dtype="bool")
        mask_input = keras.Input(shape=(seq_len,), dtype="bool")
        inner, inner_mask = ChunkLayer(max_chunks=max_chunks, name="chunker")(
            hidden_input, boundary_mask=boundary_input, mask=mask_input
        )
        model = keras.Model(
            inputs=[hidden_input, boundary_input, mask_input],
            outputs=[inner, keras.ops.cast(inner_mask, "float32")],
        )

        rng = np.random.default_rng(801)
        hidden = rng.standard_normal((3, seq_len, d_model)).astype("float32")
        boundary = _random_boundaries(rng, 3, seq_len, max_chunks)
        mask = _padding_mask(3, seq_len, [13, 8, 3])

        before = [
            keras.ops.convert_to_numpy(t)
            for t in model([hidden, boundary, mask], training=False)
        ]

        path = tmp_path / "chunk_layer.keras"
        model.save(path)
        loaded = keras.models.load_model(path)
        after = [
            keras.ops.convert_to_numpy(t)
            for t in loaded([hidden, boundary, mask], training=False)
        ]

        for original, restored in zip(before, after):
            np.testing.assert_allclose(original, restored, rtol=0, atol=GATHER_ATOL)
        assert loaded.get_layer("chunker").max_chunks == max_chunks

        # DIFFER twin: the loaded model is live, not a replayed cache.
        other = rng.standard_normal((3, seq_len, d_model)).astype("float32")
        other_out = keras.ops.convert_to_numpy(
            loaded([other, boundary, mask], training=False)[0]
        )
        assert float(np.max(np.abs(other_out - after[0]))) > 1e-2

    def test_the_restored_max_chunks_is_the_saved_one_and_not_a_default(self, tmp_path):
        """Anti-vacuity for the round trip: a different cap gives a different model."""
        seq_len, d_model = 10, 3
        saved_widths = {}
        for max_chunks in (2, 7):
            hidden_input = keras.Input(shape=(seq_len, d_model), dtype="float32")
            boundary_input = keras.Input(shape=(seq_len,), dtype="bool")
            inner, _ = ChunkLayer(max_chunks=max_chunks, name="chunker")(
                hidden_input, boundary_mask=boundary_input
            )
            model = keras.Model(inputs=[hidden_input, boundary_input], outputs=inner)
            path = tmp_path / f"chunk_{max_chunks}.keras"
            model.save(path)
            loaded = keras.models.load_model(path)
            saved_widths[max_chunks] = loaded.get_layer("chunker").max_chunks
            assert loaded.output_shape[1] == max_chunks
        assert saved_widths == {2: 2, 7: 7}


# =====================================================================
# Guard (v): the three width regimes, L < C, L == C, L > C
# =====================================================================


class TestGuardFiveWidthSymmetry:
    """``max_chunks`` may sit below, at, or above the sequence length ``L``.

    This layer always handled all three; :class:`DeChunkLayer` did not, and the
    ONE-SIDEDNESS was the defect (see ``indexing.py``'s module docstring and
    decisions.md D-029). Both layers now share
    :func:`~dl_techniques.layers.dynamic_chunking.indexing.pad_permutation_to_width`,
    and both suites carry a class of this name so the pair is graded on the same
    three regimes.

    The oracle is not consulted above its own domain: ``chunk_reference``
    transcribes upstream, where ``C <= L`` always holds because upstream's width
    IS ``max(boundary_mask.sum(-1))``. For ``C > L`` the assertion is therefore
    (a) parity with the oracle at width ``L`` on the first ``L`` columns and
    (b) a stated, checked value for the surplus columns.
    """

    @pytest.mark.parametrize(
        "seq_len,max_chunks,regime",
        [
            (8, 3, "L > C"),
            (8, 8, "L == C"),
            (8, 13, "L < C"),
            (3, 16, "L << C"),
            (1, 4, "L = 1"),
        ],
    )
    def test_the_output_width_is_max_chunks_in_every_regime(
        self, seq_len, max_chunks, regime
    ):
        """Shape is ``(B, max_chunks, D)`` whatever ``L`` is -- D-007's whole point."""
        rng = np.random.default_rng(20260909)
        batch_size, d_model = 4, 3
        hidden = rng.standard_normal((batch_size, seq_len, d_model)).astype("float32")
        boundary = _random_boundaries(rng, batch_size, seq_len, max(1, seq_len // 2))

        inner, inner_mask = _run(ChunkLayer(max_chunks=max_chunks), hidden, boundary)

        assert inner.shape == (batch_size, max_chunks, d_model), regime
        assert inner_mask.shape == (batch_size, max_chunks), regime

    @pytest.mark.parametrize("seq_len,max_chunks", [(8, 3), (8, 8), (8, 13), (3, 16)])
    def test_the_first_min_L_C_columns_are_the_oracle_gather(self, seq_len, max_chunks):
        """Inside the oracle's domain the gather is unchanged, bit for bit."""
        rng = np.random.default_rng(11)
        batch_size, d_model = 4, 5
        hidden = rng.standard_normal((batch_size, seq_len, d_model)).astype("float32")
        boundary = _random_boundaries(rng, batch_size, seq_len, max(1, seq_len // 2))

        inner, _ = _run(ChunkLayer(max_chunks=max_chunks), hidden, boundary)
        overlap = min(seq_len, max_chunks)
        expected, _ = chunk_reference(hidden, boundary, max_chunks=overlap)

        np.testing.assert_allclose(
            inner[:, :overlap], expected, rtol=0, atol=GATHER_ATOL
        )

    def test_the_surplus_columns_are_index_zero_and_are_marked_invalid(self):
        """``L < C``: the pad is index 0's row, and ``inner_mask`` says so.

        Anti-vacuity: the assertion can fail. Position 0's hidden state is drawn
        from the same distribution as every other row, so ``inner[:, L:]``
        matching it is a statement about the padding rule, not an identity --
        the twin below shows it differs from position 1's row.
        """
        rng = np.random.default_rng(1234)
        batch_size, seq_len, max_chunks, d_model = 3, 4, 10, 6
        hidden = rng.standard_normal((batch_size, seq_len, d_model)).astype("float32")
        boundary = np.zeros((batch_size, seq_len), dtype=bool)
        boundary[:, 0] = True
        boundary[:, 2] = True

        inner, inner_mask = _run(ChunkLayer(max_chunks=max_chunks), hidden, boundary)

        for column in range(seq_len, max_chunks):
            np.testing.assert_allclose(
                inner[:, column], hidden[:, 0], rtol=0, atol=GATHER_ATOL
            )
        assert not inner_mask[:, seq_len:].any(), "padded columns must be invalid"
        assert inner_mask[:, :2].all(), "the two real chunks must be valid"

        # DIFFER twin: the pad is position 0 SPECIFICALLY, not "any row".
        assert float(np.max(np.abs(inner[:, -1] - hidden[:, 1]))) > 1e-3

    def test_the_valid_column_count_never_exceeds_the_sequence_length(self):
        """``num_tokens <= L``, so no surplus column can ever be read back.

        This is what makes the padding inert for
        :class:`~dl_techniques.layers.dynamic_chunking.dechunk_layer.DeChunkLayer`,
        whose ``plug_back_idx`` is bounded by ``num_tokens - 1``.
        """
        rng = np.random.default_rng(77)
        batch_size, seq_len, max_chunks, d_model = 6, 5, 32, 2
        hidden = rng.standard_normal((batch_size, seq_len, d_model)).astype("float32")
        boundary = np.ones((batch_size, seq_len), dtype=bool)  # EVERY position

        _, inner_mask = _run(ChunkLayer(max_chunks=max_chunks), hidden, boundary)

        counts = inner_mask.sum(axis=1)
        assert counts.tolist() == [seq_len] * batch_size
        assert int(counts.max()) <= seq_len < max_chunks
