"""Behavioural guards for :class:`DocScannerFlowSequenceLoss` (Eq. 9-14).

The loss under test has no upstream implementation to differential-test
against: the released DocScanner repository ships inference only. Every claim
here is therefore either an ANALYTIC case (a fixture whose exact loss value can
be written down by hand) or a GOLDEN REFERENCE (an independent numpy
transcription of the paper's equations, written in this file and deliberately
NOT importing the code under test), because a self-consistency check against
the implementation would pass under every mutation this file exists to catch.

The four claims that carry the file:

1. **The weight direction.** ``gamma^(K-k)``, so the LAST iteration carries
   weight exactly 1.0. The inverted exponent ``gamma^(k-1)`` is one character
   away, keeps the loss finite and decreasing, serializes, trains -- and
   optimizes the wrong end of the refinement sequence.
2. **A perfect prediction costs exactly 0.0**, both terms, on an analytic
   identity fixture.
3. **The line term penalizes CURVATURE, not displacement.** The curved and
   straight fixtures are built to carry the IDENTICAL L1 term (the same
   per-pixel ``|offset|`` everywhere, only its sign pattern differs), so the
   whole difference between their losses is the circle-consistency term and
   setting ``alpha = 0`` must collapse it to exactly zero.
4. **It trains through stock ``compile``/``fit``** with no custom
   ``train_step`` -- the wiring claim, and the only end-to-end proof that a
   ``(B, K, H, W, 2)`` model output and a ``(B, H, W, 4)`` target survive
   Keras' loss plumbing.

Every fixture that could hide an axis confusion is NON-SQUARE, for the reason
``tests/test_models/test_doc_scanner/test_components.py`` states at length.
"""

import numpy as np
import pytest
import keras
import tensorflow as tf

from dl_techniques.losses import DocScannerFlowSequenceLoss
from dl_techniques.losses.doc_scanner_flow_sequence_loss import (
    DocScannerFlowSequenceLoss as _DirectImport,
)
from dl_techniques.models.vision.image_restoration.doc_scanner.components import (
    LINE_LOSS_WEIGHT,
    REFINE_ITERATIONS,
    SEQUENCE_LOSS_GAMMA,
)
from dl_techniques.models.vision.image_restoration.doc_scanner.model import (
    DocScannerRectifier,
)
from dl_techniques.models.vision.image_restoration.doc_scanner.warp import (
    coords_grid,
)

# The tiny rectifier configuration the rest of this port's suite uses. Imported
# rather than re-typed: its widths satisfy a real constructor invariant
# (`gru_input_dim == context_dim + motion_output_dim`) and a private copy here
# would drift out of it silently.
from ..test_models.test_doc_scanner.test_model import _SMALL

# ---------------------------------------------------------------------

BATCH = 2
HEIGHT, WIDTH = 6, 9            # NON-SQUARE
ITERS = 4

#: The repo convention for a value assertion: an explicit absolute tolerance
#: and no relative component.
ATOL = 1e-6

# ---------------------------------------------------------------------


def _identity_field(batch: int = BATCH, height: int = HEIGHT,
                    width: int = WIDTH) -> np.ndarray:
    """``(B, H, W, 2)`` identity coordinate grid, in ``(x, y)`` order."""
    return np.asarray(
        keras.ops.convert_to_numpy(coords_grid(batch, height, width)),
        dtype="float32",
    )


def _targets(flow_gt: np.ndarray, forward_gt: np.ndarray) -> np.ndarray:
    """Stack ``[f_gt(2), g(2)]`` into the 4-channel ``y_true``."""
    return np.concatenate([flow_gt, forward_gt], axis=-1).astype("float32")


def _repeat_as_sequence(field: np.ndarray, iters: int = ITERS) -> np.ndarray:
    """``(B, H, W, 2)`` -> ``(B, K, H, W, 2)``, the same field every step."""
    return np.repeat(field[:, None], iters, axis=1).astype("float32")


# ---------------------------------------------------------------------
# The GOLDEN REFERENCE: the paper's equations, transcribed independently.
# ---------------------------------------------------------------------


def _reference_bilinear(field: np.ndarray, coords: np.ndarray) -> np.ndarray:
    """Bilinear read of ``field`` at absolute pixel ``coords``, in ``(x, y)``.

    An independent numpy transcription of the sampling convention -- NOT a call
    into ``warp.py``. Every fixture that reaches it keeps its coordinates
    strictly inside ``[1, S - 2]``, so the edge-clamping branch is never
    exercised and this reference does not have to model it.
    """
    height, width = field.shape[1], field.shape[2]
    pix_x = coords[..., 0]
    pix_y = coords[..., 1]
    assert pix_x.min() >= 0.0 and pix_x.max() <= width - 1.0
    assert pix_y.min() >= 0.0 and pix_y.max() <= height - 1.0

    x0 = np.floor(pix_x).astype("int64")
    y0 = np.floor(pix_y).astype("int64")
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    fx = (pix_x - x0)[..., None]
    fy = (pix_y - y0)[..., None]

    out = np.empty(coords.shape[:-1] + (field.shape[-1],), dtype="float64")
    for b in range(field.shape[0]):
        top_left = field[b][y0[b], x0[b]]
        top_right = field[b][y0[b], x1[b]]
        bottom_left = field[b][y1[b], x0[b]]
        bottom_right = field[b][y1[b], x1[b]]
        top = top_left * (1.0 - fx[b]) + top_right * fx[b]
        bottom = bottom_left * (1.0 - fx[b]) + bottom_right * fx[b]
        out[b] = top * (1.0 - fy[b]) + bottom * fy[b]
    return out


def _reference_loss(
        flow_gt: np.ndarray,
        forward_gt: np.ndarray,
        sequence: np.ndarray,
        gamma: float = SEQUENCE_LOSS_GAMMA,
        alpha: float = LINE_LOSS_WEIGHT,
) -> np.ndarray:
    """Eq. 9-14, written out from the paper. Returns the per-sample loss.

    Deliberately written as an explicit double loop over samples and
    iterations, with the exponent spelled ``iters - 1 - k``, so that a reader
    can check it against the paper line by line without holding a vectorized
    fold in their head.
    """
    iters = sequence.shape[1]
    total = np.zeros((sequence.shape[0],), dtype="float64")
    for k in range(iters):
        pred = sequence[:, k]
        flow_term = np.mean(
            np.abs(flow_gt.astype("float64") - pred.astype("float64")),
            axis=(1, 2, 3),
        )
        composed = _reference_bilinear(forward_gt.astype("float64"),
                                       pred.astype("float64"))
        # A row is straight iff its Y coordinate is constant along it; a column
        # is straight iff its X coordinate is constant down it.
        row_variance = np.var(composed[..., 1], axis=2).mean(axis=1)
        column_variance = np.var(composed[..., 0], axis=1).mean(axis=1)
        line_term = row_variance + column_variance
        total += (gamma ** (iters - 1 - k)) * (flow_term + alpha * line_term)
    return total


# ---------------------------------------------------------------------


class TestTheContract:
    """Construction, validation and the config round trip."""

    def test_the_defaults_are_the_papers_numbers(self):
        loss = DocScannerFlowSequenceLoss()
        assert loss.iters == REFINE_ITERATIONS == 12
        assert loss.gamma == SEQUENCE_LOSS_GAMMA == 0.85
        assert loss.line_weight == LINE_LOSS_WEIGHT == 0.5

    def test_the_public_export_is_the_same_object(self):
        assert DocScannerFlowSequenceLoss is _DirectImport

    def test_it_is_registered_under_its_own_module_path(self):
        assert keras.saving.get_registered_name(
            DocScannerFlowSequenceLoss
        ) == (
            "dl_techniques.losses.doc_scanner_flow_sequence_loss>"
            "DocScannerFlowSequenceLoss"
        )

    @pytest.mark.parametrize("kwargs", [
        {"iters": 0},
        {"iters": -3},
        {"gamma": 0.0},
        {"gamma": -0.5},
        {"gamma": 1.5},
        {"line_weight": -1e-6},
    ])
    def test_an_out_of_range_knob_raises(self, kwargs):
        with pytest.raises(ValueError):
            DocScannerFlowSequenceLoss(**kwargs)

    def test_gamma_of_exactly_one_is_allowed(self):
        """An unweighted sequence loss is a legitimate ablation, not an error."""
        loss = DocScannerFlowSequenceLoss(iters=4, gamma=1.0)
        assert loss._iteration_weights() == [1.0, 1.0, 1.0, 1.0]

    def test_the_config_round_trips_all_three_knobs(self):
        original = DocScannerFlowSequenceLoss(
            iters=7, gamma=0.5, line_weight=0.125)
        config = original.get_config()
        assert config["iters"] == 7
        assert config["gamma"] == 0.5
        assert config["line_weight"] == 0.125

        restored = DocScannerFlowSequenceLoss.from_config(config)
        assert restored.iters == 7
        assert restored.gamma == 0.5
        assert restored.line_weight == 0.125

    def test_it_survives_a_keras_serialization_round_trip(self):
        original = DocScannerFlowSequenceLoss(
            iters=5, gamma=0.9, line_weight=0.25)
        restored = keras.saving.deserialize_keras_object(
            keras.saving.serialize_keras_object(original)
        )
        assert isinstance(restored, DocScannerFlowSequenceLoss)
        assert (restored.iters, restored.gamma, restored.line_weight) == (
            5, 0.9, 0.25)

        identity = _identity_field()
        sequence = _repeat_as_sequence(identity, iters=5)
        sequence = sequence + 0.75
        y_true = _targets(identity, identity)
        np.testing.assert_allclose(
            float(original(y_true, sequence)),
            float(restored(y_true, sequence)),
            atol=ATOL, rtol=0,
        )


class TestTheShapeContractIsEnforced:
    """The three ways a caller can wire this loss up wrongly."""

    def test_a_rank_four_prediction_names_the_training_flag(self):
        identity = _identity_field()
        loss = DocScannerFlowSequenceLoss(iters=ITERS)
        with pytest.raises(ValueError, match="training=True"):
            loss(_targets(identity, identity), identity)

    def test_a_sequence_of_the_wrong_length_raises(self):
        identity = _identity_field()
        loss = DocScannerFlowSequenceLoss(iters=ITERS)
        with pytest.raises(ValueError, match="iterations"):
            loss(_targets(identity, identity),
                 _repeat_as_sequence(identity, iters=ITERS + 1))

    def test_a_two_channel_target_raises_naming_the_forward_map(self):
        identity = _identity_field()
        loss = DocScannerFlowSequenceLoss(iters=ITERS)
        with pytest.raises(ValueError, match="forward map"):
            loss(identity, _repeat_as_sequence(identity))


class TestTheLastIterationCarriesWeightOne:
    """THE headline guard: the weight direction of Eq. 9.

    Three independent readings of the same claim, because the inverted exponent
    ``gamma ** k`` is shape-, dtype-, finiteness- and serialization-invisible:

    * the weight vector itself, pinned element-wise;
    * a loss-level measurement -- one wrong iteration at a time, whose cost must
      be STRICTLY INCREASING in the iteration's position;
    * the exact analytic value of that cost, which pins the base AND the
      exponent, not merely their ordering.
    """

    def _one_wrong_iteration(self, index, offset=0.5, **kwargs):
        """A sequence that is perfect except at ``index``, and its loss."""
        identity = _identity_field()
        sequence = _repeat_as_sequence(identity)
        sequence[:, index, ..., 1] += offset
        loss = DocScannerFlowSequenceLoss(iters=ITERS, **kwargs)
        return float(loss(_targets(identity, identity), sequence))

    def test_the_weight_vector_ends_at_exactly_one(self):
        weights = DocScannerFlowSequenceLoss(
            iters=ITERS, gamma=0.85)._iteration_weights()
        assert len(weights) == ITERS
        assert weights[-1] == 1.0
        np.testing.assert_allclose(
            weights,
            [0.85 ** 3, 0.85 ** 2, 0.85 ** 1, 1.0],
            atol=ATOL, rtol=0,
        )

    def test_the_weights_increase_toward_the_last_iteration(self):
        weights = DocScannerFlowSequenceLoss(
            iters=ITERS, gamma=0.85)._iteration_weights()
        assert all(
            weights[i] < weights[i + 1] for i in range(len(weights) - 1)
        ), weights

    def test_a_late_error_costs_strictly_more_than_an_early_one(self):
        measured = [self._one_wrong_iteration(k) for k in range(ITERS)]
        assert all(
            measured[k] < measured[k + 1] for k in range(ITERS - 1)
        ), measured

    def test_the_cost_of_one_wrong_iteration_is_exactly_the_papers_weight(self):
        """Pins the base and the exponent, not just the ordering.

        The offset is applied to the Y channel only, so the mean absolute error
        over the 2-channel field is ``offset / 2``; the offset is constant, so
        the round-tripped lines stay straight and the line term is exactly 0.
        """
        offset = 0.5
        for k in range(ITERS):
            expected = (offset / 2.0) * (0.85 ** (ITERS - 1 - k))
            np.testing.assert_allclose(
                self._one_wrong_iteration(k, offset=offset, gamma=0.85),
                expected, atol=ATOL, rtol=0,
                err_msg=f"iteration {k} of {ITERS}",
            )

    def test_the_ratio_between_neighbouring_iterations_is_gamma(self):
        measured = [
            self._one_wrong_iteration(k, gamma=0.5) for k in range(ITERS)
        ]
        for k in range(ITERS - 1):
            np.testing.assert_allclose(
                measured[k] / measured[k + 1], 0.5, atol=ATOL, rtol=0)


class TestAPerfectPredictionCostsExactlyZero:
    """The analytic identity case: both terms, exactly 0.0, not merely small."""

    def test_the_identity_round_trip_is_free(self):
        identity = _identity_field()
        loss = DocScannerFlowSequenceLoss(iters=ITERS)
        value = float(
            loss(_targets(identity, identity), _repeat_as_sequence(identity)))
        assert value == 0.0

    def test_it_is_free_for_every_line_weight(self):
        identity = _identity_field()
        for alpha in (0.0, 0.5, 1.0, 4.0):
            loss = DocScannerFlowSequenceLoss(iters=ITERS, line_weight=alpha)
            assert float(
                loss(_targets(identity, identity),
                     _repeat_as_sequence(identity))
            ) == 0.0

    def test_an_imperfect_prediction_is_not_free(self):
        """Anti-vacuity: the fixture is capable of scoring above zero."""
        identity = _identity_field()
        sequence = _repeat_as_sequence(identity) + 0.25
        loss = DocScannerFlowSequenceLoss(iters=ITERS)
        assert float(loss(_targets(identity, identity), sequence)) > 0.0


class TestTheLineTermPenalizesCurvature:
    """Curvature, not displacement -- and the whole difference is ``alpha``.

    ``_straight`` and ``_curved`` carry the IDENTICAL per-pixel ``|offset|``,
    so their L1 terms are equal by construction (asserted, not assumed). The
    only thing that separates them is the SIGN pattern along each row, i.e. the
    curvature of the round-tripped line. At ``alpha = 0`` they must therefore
    score exactly the same.
    """

    OFFSET = 0.25

    def _fixture(self, curved: bool, height=HEIGHT, width=WIDTH):
        identity = _identity_field(BATCH, height, width)
        pattern = np.ones((height, width), dtype="float32")
        if curved:
            pattern[:, 1::2] = -1.0
        # Interior rows only: the offsets must not push a coordinate outside
        # the field, or edge clamping -- not curvature -- would drive the term.
        interior = np.zeros((height, width), dtype="float32")
        interior[1:-1, :] = 1.0
        field = identity.copy()
        field[..., 1] += self.OFFSET * interior * pattern
        return identity, field

    def _loss_of(self, field, identity, **kwargs):
        loss = DocScannerFlowSequenceLoss(iters=ITERS, **kwargs)
        return float(loss(_targets(identity, identity),
                          _repeat_as_sequence(field)))

    def test_the_two_fixtures_carry_the_same_l1_term(self):
        identity, straight = self._fixture(curved=False)
        _, curved = self._fixture(curved=True)
        np.testing.assert_allclose(
            self._loss_of(straight, identity, line_weight=0.0),
            self._loss_of(curved, identity, line_weight=0.0),
            atol=ATOL, rtol=0,
        )

    def test_a_curved_round_trip_scores_higher_than_a_straight_one(self):
        identity, straight = self._fixture(curved=False)
        _, curved = self._fixture(curved=True)
        assert (
            self._loss_of(curved, identity)
            > self._loss_of(straight, identity) + 1e-3
        )

    def test_a_straight_displacement_costs_nothing_extra(self):
        """The line term is exactly 0 on the straight fixture, at any alpha."""
        identity, straight = self._fixture(curved=False)
        baseline = self._loss_of(straight, identity, line_weight=0.0)
        for alpha in (0.5, 1.0, 4.0):
            np.testing.assert_allclose(
                self._loss_of(straight, identity, line_weight=alpha),
                baseline, atol=ATOL, rtol=0)

    def test_at_alpha_zero_the_distinction_collapses_exactly(self):
        identity, straight = self._fixture(curved=False)
        _, curved = self._fixture(curved=True)
        assert (
            self._loss_of(curved, identity, line_weight=0.0)
            == self._loss_of(straight, identity, line_weight=0.0)
        )

    def test_curvature_in_the_rows_beyond_the_width_still_counts(self):
        """The S-5 reading, asserted rather than left in a docstring.

        Eq. 14's ar5iv extraction indexes the row sum ``i = 1..W``; this port
        reads it as ``i = 1..H`` and says so. The difference is only observable
        off-square, so the fixture is TALL (H > W) and its curvature lives
        exclusively in rows ``W .. H-1`` -- rows a literal ``1..W`` reading
        never visits. Under that reading this test scores exactly the straight
        fixture and fails.
        """
        tall_h, tall_w = 12, 5
        identity = _identity_field(BATCH, tall_h, tall_w)
        pattern = np.zeros((tall_h, tall_w), dtype="float32")
        pattern[tall_w:tall_h - 1, 1::2] = -1.0
        pattern[tall_w:tall_h - 1, 0::2] = 1.0
        field = identity.copy()
        field[..., 1] += self.OFFSET * pattern

        with_line = self._loss_of(field, identity, line_weight=0.5)
        without_line = self._loss_of(field, identity, line_weight=0.0)
        assert with_line > without_line + 1e-3, (with_line, without_line)


class TestTheLineWeightIsAppliedExactlyOnce:
    """``alpha`` multiplies the line term once -- not twice, not zero times."""

    def _loss_at(self, alpha):
        rng = np.random.default_rng(11)
        identity = _identity_field()
        field = identity + rng.uniform(
            -0.4, 0.4, identity.shape).astype("float32")
        field = np.clip(field, 1.0, min(HEIGHT, WIDTH) - 2.0)
        loss = DocScannerFlowSequenceLoss(iters=ITERS, line_weight=alpha)
        return float(loss(_targets(identity, identity),
                          _repeat_as_sequence(field)))

    def test_the_loss_is_affine_in_alpha(self):
        """``L(a) = L(0) + a * line``: doubling ``a`` doubles the increment.

        A squared or twice-applied factor breaks this; so does dropping it.
        """
        base = self._loss_at(0.0)
        half = self._loss_at(0.5)
        one = self._loss_at(1.0)
        two = self._loss_at(2.0)
        assert one - base > 1e-4, "the line term is inert on this fixture"
        np.testing.assert_allclose(
            (half - base) * 2.0, one - base, atol=1e-5, rtol=0)
        np.testing.assert_allclose(
            (one - base) * 2.0, two - base, atol=1e-5, rtol=0)

    def test_the_default_alpha_is_the_papers_one_half(self):
        base = self._loss_at(0.0)
        one = self._loss_at(1.0)
        # Anti-vacuity: with an inert line term every alpha scores the same and
        # the assertion below would hold for a loss that dropped the term.
        assert one - base > 1e-4, (base, one)
        rng = np.random.default_rng(11)
        identity = _identity_field()
        field = identity + rng.uniform(
            -0.4, 0.4, identity.shape).astype("float32")
        field = np.clip(field, 1.0, min(HEIGHT, WIDTH) - 2.0)
        default = float(
            DocScannerFlowSequenceLoss(iters=ITERS)(
                _targets(identity, identity), _repeat_as_sequence(field))
        )
        np.testing.assert_allclose(
            default, base + 0.5 * (one - base), atol=1e-5, rtol=0)


class TestItMatchesAnIndependentTranscriptionOfTheEquations:
    """The golden-reference arm: an ASYMMETRIC fixture, checked end to end.

    This is the arm that pins the composition ORDER. ``g(f^k)`` and ``f^k(g)``
    are shape-identical, both finite, both differentiable, and both exactly
    zero on every identity fixture in this file -- only a fixture in which
    ``f`` and ``g`` are independent, non-identity fields can tell them apart.
    All coordinates are kept strictly inside the field so that no edge clamping
    is involved and the numpy reference does not have to model it.
    """

    def _fixture(self, seed):
        rng = np.random.default_rng(seed)
        identity = _identity_field()
        low, high = 1.0, float(min(HEIGHT, WIDTH) - 2)
        flow_gt = rng.uniform(low, high, identity.shape).astype("float32")
        forward_gt = rng.uniform(low, high, identity.shape).astype("float32")
        sequence = rng.uniform(
            low, high, (BATCH, ITERS) + identity.shape[1:]).astype("float32")
        return flow_gt, forward_gt, sequence

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_the_per_sample_values_match(self, seed):
        flow_gt, forward_gt, sequence = self._fixture(seed)
        loss = DocScannerFlowSequenceLoss(
            iters=ITERS, gamma=0.85, line_weight=0.5, reduction=None)
        measured = np.asarray(keras.ops.convert_to_numpy(
            loss(_targets(flow_gt, forward_gt), sequence)))
        expected = _reference_loss(flow_gt, forward_gt, sequence)
        assert measured.shape == (BATCH,)
        np.testing.assert_allclose(measured, expected, atol=1e-4, rtol=0)

    def test_the_reduced_value_is_the_mean_over_the_batch(self):
        flow_gt, forward_gt, sequence = self._fixture(3)
        scalar = float(DocScannerFlowSequenceLoss(iters=ITERS)(
            _targets(flow_gt, forward_gt), sequence))
        expected = float(
            np.mean(_reference_loss(flow_gt, forward_gt, sequence)))
        np.testing.assert_allclose(scalar, expected, atol=1e-4, rtol=0)

    def test_the_two_samples_of_the_fixture_really_differ(self):
        """Anti-vacuity for the per-sample arm: a batch-collapsing bug would
        otherwise be invisible against a reference that also collapsed."""
        flow_gt, forward_gt, sequence = self._fixture(0)
        per_sample = _reference_loss(flow_gt, forward_gt, sequence)
        assert abs(per_sample[0] - per_sample[1]) > 1e-3, per_sample


# ---------------------------------------------------------------------
# The end-to-end wiring claim.
# ---------------------------------------------------------------------

_FIT_HEIGHT, _FIT_WIDTH = 32, 24        # NON-SQUARE, both multiples of 8
_FIT_ITERS = 2


def _fit_subject():
    """A tiny rectifier and a synthetic batch, at the port's own test widths."""
    keras.utils.set_random_seed(0)
    model = DocScannerRectifier(**{**_SMALL, "iters": _FIT_ITERS})
    rng = np.random.default_rng(0)
    x = rng.uniform(0.0, 1.0,
                    (BATCH, _FIT_HEIGHT, _FIT_WIDTH, 3)).astype("float32")
    identity = _identity_field(BATCH, _FIT_HEIGHT, _FIT_WIDTH)
    y = _targets(identity, identity)
    return model, x, y


class TestItTrainsThroughStockFit:
    """No custom ``train_step`` anywhere: ``compile(loss=...)`` is enough.

    This is the claim the whole step exists to make good on. The sequence
    arrives as a model OUTPUT under ``training=True``, so Keras' own
    ``fit`` loop can carry it -- and a `(B, K, H, W, 2)` prediction against a
    `(B, H, W, 4)` target is exactly the pairing that would force a custom
    training step if the loss plumbing rejected it.
    """

    def test_the_loss_decreases_over_a_few_steps_and_stays_finite(self):
        model, x, y = _fit_subject()
        model.compile(
            optimizer=keras.optimizers.Adam(1e-3),
            loss=DocScannerFlowSequenceLoss(iters=_FIT_ITERS),
        )
        history = model.fit(x, y, batch_size=BATCH, epochs=4, verbose=0)
        curve = history.history["loss"]
        assert len(curve) == 4
        assert all(np.isfinite(curve)), curve
        assert curve[-1] < curve[0], curve

    def test_no_class_in_the_port_overrides_train_step(self):
        """The invariant stated as a property, not as prose in a docstring."""
        for cls in (DocScannerRectifier,):
            assert "train_step" not in vars(cls)

    def test_the_gradient_reaches_every_trainable_weight(self):
        model, x, _ = _fit_subject()
        rng = np.random.default_rng(1)
        identity = _identity_field(BATCH, _FIT_HEIGHT, _FIT_WIDTH)
        flow_gt = identity + rng.uniform(
            -2.0, 2.0, identity.shape).astype("float32")
        y_true = _targets(flow_gt, identity)
        loss = DocScannerFlowSequenceLoss(iters=_FIT_ITERS)

        with tf.GradientTape() as tape:
            sequence = model(tf.constant(x), training=True)
            value = loss(tf.constant(y_true), sequence)
        gradients = tape.gradient(value, model.trainable_variables)

        assert model.trainable_variables, "the subject has no weights"
        dead = [
            variable.path
            for variable, gradient in zip(model.trainable_variables, gradients)
            if gradient is None
            or float(np.max(np.abs(keras.ops.convert_to_numpy(gradient)))) == 0.0
        ]
        assert not dead, dead

    def test_the_line_term_alone_still_moves_the_weights(self):
        """The circle-consistency term is differentiable w.r.t. the prediction.

        With ``f_gt`` set to the model's own prediction the L1 term contributes
        nothing, so any gradient that survives came through the two-step warp.
        """
        model, x, _ = _fit_subject()
        identity = _identity_field(BATCH, _FIT_HEIGHT, _FIT_WIDTH)
        loss = DocScannerFlowSequenceLoss(
            iters=_FIT_ITERS, line_weight=1.0)

        with tf.GradientTape() as tape:
            sequence = model(tf.constant(x), training=True)
            flow_gt = tf.stop_gradient(sequence[:, -1])
            y_true = keras.ops.concatenate(
                [flow_gt, tf.constant(identity)], axis=-1)
            value = loss(y_true, sequence)
        gradients = tape.gradient(value, model.trainable_variables)

        live = [
            g for g in gradients
            if g is not None
            and float(np.max(np.abs(keras.ops.convert_to_numpy(g)))) > 0.0
        ]
        assert live, "no weight receives a gradient from the line term"
