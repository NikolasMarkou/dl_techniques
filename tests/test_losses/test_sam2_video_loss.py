"""Guards for :class:`SAM2GatedMaskLoss` -- upstream's ground-truth gate.

Every value assertion here is made against :func:`gated_mask_loss_oracle`, an
INDEPENDENT float64 numpy transcription of focal + dice written from the
formulae, not from a second call into the repository's own loss stack. A guard
that computed its expectation by calling ``SAMMaskLoss`` would be asserting that
the code equals itself.

RED-proven against BOTH wrong candidates the plan names, on GPU 1:

* an **ungated** variant (the gate deleted -- plain :class:`SAMMaskLoss`), and
* a **self-gated** variant (gated by the PREDICTED mask instead of the ground
  truth), which is the plausible wrong one: it looks like it works, and it is
  self-fulfilling -- a model that predicts "absent" switches off the loss that
  would correct it.

:func:`discriminating_fixture` is built so that all three candidates DISAGREE.
That is not decoration: over a plain random fixture the self-gated and ungated
candidates COINCIDE exactly (measured -- every row has some pixel above 0.5, so
the predicted gate is all-``True``), and a guard sited there would prove the
self-gated mutation exactly zero times.
"""

from typing import Any, Dict, Tuple

import keras
import numpy as np
import pytest
import tensorflow as tf
from keras import ops

from dl_techniques.losses.sam2_video_loss import (
    ABSENT_ROW_FOCAL_RESIDUAL,
    SAM2GatedMaskLoss,
    mask_presence_gate,
)
from dl_techniques.losses.sam_mask_loss import SAMMaskLoss

#: Tolerance for every value assertion in this file, DERIVED not guessed.
#: MEASURED |repo float32 - float64 oracle| over the two fixtures below:
#: ``7.91e-07`` and ``3.17e-06``. That is float32 reduction error over a
#: ``2 x 4 x 16 x 16`` stack, and it is the BINDING term -- the absent-row focal
#: residual it is often assumed to bound is fourteen orders of magnitude
#: smaller (see :data:`ABSENT_ROW_FOCAL_RESIDUAL`). ``1e-5`` is ~3x the measured
#: error and ~4e5 times smaller than the smallest candidate separation
#: (``4.118``); :meth:`TestGateIsValueExact.test_the_tolerance_cannot_admit_a_wrong_candidate`
#: asserts that margin rather than leaving it to prose.
VALUE_TOLERANCE = 1e-5

#: The same measurement for the UNGATED loss, which needs its OWN number and
#: does not get to borrow the gated one. MEASURED |SAMMaskLoss float32 -
#: float64 oracle| at :func:`discriminating_fixture`: ``1.729e-05``, i.e. 5.5x
#: the gated figure -- the ungated total is 1.7x larger and keeps two more rows
#: in the focal reduction, so it accumulates more float32 error. ``5e-5`` is
#: ~2.9x the measured value and still 8e4 times inside the nearest candidate
#: separation. Writing one tolerance for both would have meant either a false
#: failure here or a loosened bound on the arm that actually matters.
UNGATED_VALUE_TOLERANCE = 5e-5


def gated_mask_loss_oracle(
        truth: np.ndarray,
        logits: np.ndarray,
        gate: np.ndarray,
        focal_weight: float = 20.0,
        dice_weight: float = 1.0,
        gamma: float = 2.0,
        alpha: float = 0.25,
        smooth: float = 1e-6,
        clip_epsilon: float = 1e-7,
) -> float:
    """Hand-compute ``focal_weight * focal + dice_weight * dice`` in float64.

    Transcribed from ``SegmentationLosses.dice_loss`` / ``focal_loss`` and the
    two ``sam_mask_loss`` adapters as FORMULAE -- ``dice`` reduces the spatial
    axes per row and averages the rows; ``focal`` averages the per-pixel sum
    over a 2-channel one-hot; the clip is what leaves an absent row a residual
    instead of an exact zero.

    :param truth: ``(B, M, h, w)`` binary ground truth.
    :type truth: numpy.ndarray
    :param logits: ``(B, M, h, w)`` predicted mask logits.
    :type logits: numpy.ndarray
    :param gate: ``(B, M)`` bool. Rows that are ``False`` have their
        PROBABILITIES zeroed, exactly as the loss does.
    :type gate: numpy.ndarray
    :param focal_weight: Weight of the focal term.
    :type focal_weight: float
    :param dice_weight: Weight of the dice term.
    :type dice_weight: float
    :param gamma: ``LossConfig.focal_gamma``.
    :type gamma: float
    :param alpha: ``LossConfig.focal_alpha``.
    :type alpha: float
    :param smooth: ``LossConfig.smooth_factor``.
    :type smooth: float
    :param clip_epsilon: ``focal_loss``'s probability clip.
    :type clip_epsilon: float
    :return: The scalar loss.
    :rtype: float
    """
    true64 = truth.astype(np.float64)
    probabilities = 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))
    probabilities = np.where(gate[..., None, None], probabilities, 0.0)

    numerator = 2.0 * (true64 * probabilities).sum(axis=(-2, -1))
    denominator = true64.sum(axis=(-2, -1)) + probabilities.sum(axis=(-2, -1))
    dice = 1.0 - ((numerator + smooth) / (denominator + smooth)).mean()

    true_channels = np.stack([1.0 - true64, true64], axis=-1)
    pred_channels = np.clip(
        np.stack([1.0 - probabilities, probabilities], axis=-1),
        clip_epsilon, 1.0 - clip_epsilon,
    )
    cross_entropy = -true_channels * np.log(pred_channels)
    focal = (alpha * (1.0 - pred_channels) ** gamma * cross_entropy).sum(
        axis=-1).mean()
    return float(focal_weight * focal + dice_weight * dice)


def discriminating_fixture() -> Tuple[np.ndarray, np.ndarray]:
    """A ``(truth, logits)`` pair on which all three candidates disagree.

    Two rows are engineered, because a gate is a per-row decision and the two
    ways of getting it wrong point in opposite directions:

    * row ``(0, 1)``: ground truth PRESENT, every logit ``-8`` so the predicted
      mask is empty. The GT gate KEEPS this row; a self-gate DROPS it. This is
      the self-fulfilling case -- the row the model has wrongly given up on is
      exactly the row that must still be scored.
    * rows ``(0, 2)`` and ``(1, 3)``: ground truth ABSENT, logits forced
      positive so the predicted mask is non-empty. The GT gate DROPS these; a
      self-gate KEEPS them, and an ungated loss keeps them too.

    :return: ``(truth, logits)``, both ``(2, 4, 16, 16)`` float32.
    :rtype: Tuple[numpy.ndarray, numpy.ndarray]
    """
    rng = np.random.default_rng(11)
    truth = (rng.random((2, 4, 16, 16)) > 0.6).astype("float32")
    truth[:, :, 0, 0] = 1.0
    logits = rng.normal(0.0, 3.0, (2, 4, 16, 16)).astype("float32")
    logits[0, 1] = -8.0
    truth[0, 2] = 0.0
    truth[1, 3] = 0.0
    logits[0, 2] = np.abs(logits[0, 2]) + 1.0
    logits[1, 3] = np.abs(logits[1, 3]) + 1.0
    return truth, logits


def ground_truth_gate(truth: np.ndarray) -> np.ndarray:
    """``(B, M)`` bool: rows whose ground truth has any foreground."""
    return truth.max(axis=(-2, -1)) > 0.0


def predicted_gate(logits: np.ndarray) -> np.ndarray:
    """``(B, M)`` bool: rows whose PREDICTED mask has any foreground.

    This is the wrong gate, reproduced here so the guards can measure how far
    wrong it is instead of asserting that it is wrong.
    """
    return (1.0 / (1.0 + np.exp(-logits.astype(np.float64)))).max(
        axis=(-2, -1)) > 0.5


def candidate_values(
        truth: np.ndarray, logits: np.ndarray) -> Dict[str, float]:
    """The three candidate losses at one fixture, all from the oracle."""
    return {
        "gt_gated": gated_mask_loss_oracle(
            truth, logits, ground_truth_gate(truth)),
        "ungated": gated_mask_loss_oracle(
            truth, logits, np.ones_like(ground_truth_gate(truth))),
        "self_gated": gated_mask_loss_oracle(
            truth, logits, predicted_gate(logits)),
    }


# ---------------------------------------------------------------------
# A-6: the residual is MEASURED here, not quoted from the module
# ---------------------------------------------------------------------


class TestAbsentRowResidual:
    """The absent-row residual A-6 predicted, re-measured every run."""

    @pytest.mark.parametrize(
        "shape", [(1, 1, 16, 16), (2, 3, 16, 16), (2, 4, 64, 64)])
    def test_a_fully_absent_batch_is_the_published_residual(
            self, shape: Tuple[int, ...]) -> None:
        """A batch whose every row is absent IS the pure residual.

        Its value is the whole of what the gate cannot remove, so it is the
        honest measurement of A-6 -- and it must not depend on the mask grid,
        because ``focal_loss``' clip is a per-pixel constant.
        """
        truth = np.zeros(shape, dtype="float32")
        logits = np.full(shape, 7.5, dtype="float32")
        measured = float(SAM2GatedMaskLoss()(truth, logits))
        assert measured == pytest.approx(
            20.0 * ABSENT_ROW_FOCAL_RESIDUAL, rel=1e-6), (
            f"the absent-row residual moved to {measured!r}; "
            f"ABSENT_ROW_FOCAL_RESIDUAL and the module docstring quote "
            f"{ABSENT_ROW_FOCAL_RESIDUAL!r} per unit focal weight")

    def test_the_residual_is_not_what_the_tolerance_is_for(self) -> None:
        """A-6 is confirmed, and confirmed to be NON-binding.

        The plan pre-committed to STOP if the residual exceeded ``1e-6`` of the
        ungated term. It is fourteen orders below float32's own accumulation
        error on the same reduction, so :data:`VALUE_TOLERANCE` is a float32
        floor, not a residual budget. Recorded as an assertion so a future
        change to ``SegmentationLosses``' clipping cannot quietly make the
        residual the binding term while the tolerance stays put.
        """
        truth, logits = discriminating_fixture()
        ungated = candidate_values(truth, logits)["ungated"]
        assert 20.0 * ABSENT_ROW_FOCAL_RESIDUAL < 1e-6 * ungated
        assert 20.0 * ABSENT_ROW_FOCAL_RESIDUAL < VALUE_TOLERANCE * 1e-6

    def test_the_dice_half_of_an_absent_row_is_exactly_zero(self) -> None:
        """Dice needs no residual at all: ``1 - s/s`` is exact."""
        shape = (2, 3, 16, 16)
        truth = np.zeros(shape, dtype="float32")
        logits = np.full(shape, 7.5, dtype="float32")
        dice_only = SAM2GatedMaskLoss(focal_weight=0.0, dice_weight=1.0)
        assert float(dice_only(truth, logits)) == 0.0


# ---------------------------------------------------------------------
# G3.1 / G3.3 -- value exactness against BOTH wrong candidates
# ---------------------------------------------------------------------


class TestGateIsValueExact:
    """The loss equals the hand-computed GT-gated number, and only that one."""

    def test_the_three_candidates_really_are_separated(self) -> None:
        """Fixture validity, asserted BEFORE anything is proven with it.

        Iteration 1 sited four guards at a point where the correct and the
        mutated variant provably coincide. Over a plain random fixture the
        self-gated candidate coincides with the ungated one EXACTLY, so this
        check is the difference between a proof and a coincidence.
        """
        truth, logits = discriminating_fixture()
        values = candidate_values(truth, logits)
        pairs = [("gt_gated", "ungated"), ("gt_gated", "self_gated"),
                 ("ungated", "self_gated")]
        for left, right in pairs:
            separation = abs(values[left] - values[right])
            assert separation > 1.0, (
                f"{left} and {right} are only {separation!r} apart at this "
                f"fixture; a guard sited here cannot discriminate them")

    def test_the_tolerance_cannot_admit_a_wrong_candidate(self) -> None:
        """:data:`VALUE_TOLERANCE` is 4e5 times inside the nearest wrong one."""
        truth, logits = discriminating_fixture()
        values = candidate_values(truth, logits)
        nearest = min(
            abs(values["gt_gated"] - values["ungated"]),
            abs(values["gt_gated"] - values["self_gated"]),
        )
        assert nearest / VALUE_TOLERANCE > 1e4, (
            f"the nearest wrong candidate is only {nearest!r} away, i.e. "
            f"{nearest / VALUE_TOLERANCE!r} tolerances")

    def test_the_loss_equals_the_hand_computed_gated_value(self) -> None:
        """G3.1: value exactness against the float64 oracle.

        MEASURED |float32 - float64| at this fixture: ``3.17e-06``.
        """
        truth, logits = discriminating_fixture()
        measured = float(SAM2GatedMaskLoss()(truth, logits))
        expected = candidate_values(truth, logits)["gt_gated"]
        assert measured == pytest.approx(expected, abs=VALUE_TOLERANCE), (
            f"the gated loss is {measured!r}, the hand-computed GT-gated "
            f"value is {expected!r}")

    def test_it_is_not_the_ungated_value(self) -> None:
        """G3.1's RED arm, permanent: deleting the gate moves it by 4.118."""
        truth, logits = discriminating_fixture()
        measured = float(SAM2GatedMaskLoss()(truth, logits))
        values = candidate_values(truth, logits)
        assert abs(measured - values["ungated"]) > 1.0
        # ... and the ungated candidate really is what plain SAMMaskLoss gives,
        # so "delete the gate" is the mutation this arm rejects, not a
        # hypothetical.
        assert float(SAMMaskLoss()(truth, logits)) == pytest.approx(
            values["ungated"], abs=UNGATED_VALUE_TOLERANCE)

    def test_it_is_not_the_self_gated_value(self) -> None:
        """G3.3's RED arm, permanent: gating on the PREDICTION moves it by 6.042.

        The row that separates them is ``(0, 1)``: present in the ground truth,
        empty in the prediction. A self-gated loss scores it zero -- it has
        switched off the correction for the mistake it just made.
        """
        truth, logits = discriminating_fixture()
        gt_gate = ground_truth_gate(truth)
        pred_gate = predicted_gate(logits)
        assert gt_gate[0, 1] and not pred_gate[0, 1], (
            "the fixture no longer contains a row that is present in the GT "
            "and absent in the prediction; the self-gated mutation cannot be "
            "discriminated without one")
        measured = float(SAM2GatedMaskLoss()(truth, logits))
        assert abs(
            measured - candidate_values(truth, logits)["self_gated"]) > 1.0

    def test_a_clip_with_no_absent_row_matches_the_ungated_loss(self) -> None:
        """The gate is a no-op when nothing is occluded -- by value.

        Without this, a gate that zeroed EVERY row would satisfy every
        assertion above that only says "not the ungated value".
        """
        rng = np.random.default_rng(3)
        truth = (rng.random((2, 3, 16, 16)) > 0.6).astype("float32")
        truth[:, :, 0, 0] = 1.0
        logits = rng.normal(0.0, 2.0, (2, 3, 16, 16)).astype("float32")
        assert ground_truth_gate(truth).all()
        assert float(SAM2GatedMaskLoss()(truth, logits)) == pytest.approx(
            float(SAMMaskLoss()(truth, logits)), abs=VALUE_TOLERANCE)


# ---------------------------------------------------------------------
# G3.2 -- gradient, two-sided
# ---------------------------------------------------------------------


def logit_gradient(loss: keras.losses.Loss, truth: np.ndarray,
                   logits: np.ndarray) -> np.ndarray:
    """``d loss / d logits`` as a numpy array."""
    variable = tf.Variable(logits)
    with tf.GradientTape() as tape:
        value = loss(ops.convert_to_tensor(truth), variable)
    return np.asarray(tape.gradient(value, variable))


class TestGradientOnAbsentRows:
    """G3.2: absent rows receive EXACTLY zero gradient, present rows do not."""

    def test_absent_rows_get_exactly_zero_and_present_rows_do_not(
            self) -> None:
        """Both arms in one test, because either alone is satisfiable.

        Per D-037 the expectation is exact **zeros**, not ``None`` -- the
        absent rows stay structurally connected through the reductions, so a
        ``None``-counting assertion would fail for an unrelated reason.
        """
        truth, logits = discriminating_fixture()
        gate = ground_truth_gate(truth)
        gradient = logit_gradient(SAM2GatedMaskLoss(), truth, logits)
        assert gradient is not None
        absent = gradient[~gate]
        present = gradient[gate]
        assert np.count_nonzero(absent) == 0, (
            f"{np.count_nonzero(absent)} of {absent.size} absent-row "
            f"gradient entries are non-zero; max |g| = "
            f"{np.abs(absent).max()!r}")
        assert np.abs(present).max() > 0.0

    def test_the_same_rows_are_live_without_the_gate(self) -> None:
        """The two-sided half: those zeros are the GATE, not the fixture.

        Without it, a fixture whose absent rows happened to sit at a gradient
        of zero for some other reason would pass the arm above on a loss with
        no gate at all.
        """
        truth, logits = discriminating_fixture()
        gate = ground_truth_gate(truth)
        gradient = logit_gradient(SAMMaskLoss(), truth, logits)
        absent = gradient[~gate]
        assert np.abs(absent).max() > 0.0, (
            "the UNGATED loss also gives these rows zero gradient, so the "
            "gated arm proves nothing about the gate")

    def test_a_float_multiply_would_also_zero_it(self) -> None:
        """The plan predicted G3.2's named mutation is INERT. It is.

        Replacing ``ops.where`` with a multiply by the float gate zeroes the
        absent row's gradient just as well -- ``0.0 * x`` has zero derivative
        in ``x``. Measured here as an executable statement so nobody re-derives
        it: the discriminating evidence for the gate's SHAPE is the VALUE
        assertions above, not this gradient.
        """
        truth, logits = discriminating_fixture()
        gate = ground_truth_gate(truth).astype("float32")

        class MultiplyGated(SAMMaskLoss):
            def call(self, y_true: Any, y_pred: Any) -> Any:
                return super().call(
                    y_true, y_pred * ops.reshape(
                        ops.convert_to_tensor(gate), (2, 4, 1, 1)))

        gradient = logit_gradient(MultiplyGated(), truth, logits)
        assert np.count_nonzero(gradient[ground_truth_gate(truth) == 0]) == 0


# ---------------------------------------------------------------------
# G3.4 -- the adapters are `sam_mask_loss`'s, not a second opinion
# ---------------------------------------------------------------------


class TestAdapterReuse:
    """The layout refusals inherited from ``sam_mask_loss`` still fire."""

    def test_a_rank_three_stack_is_refused(self) -> None:
        """H-17's trap: raw-layout dice returns a plausible wrong number."""
        truth = np.zeros((2, 16, 16), dtype="float32")
        with pytest.raises(ValueError, match="mask stack"):
            SAM2GatedMaskLoss()(truth, truth)

    def test_the_shared_adapters_are_the_ones_called(self) -> None:
        """Not a second copy: the module-level adapters are invoked.

        A future author who inlined a reshape here would keep every value
        assertion green (the layouts agree) while creating the second home
        H-17 exists to prevent.
        """
        import dl_techniques.losses.sam2_video_loss as module

        truth, logits = discriminating_fixture()
        calls = {"dice": 0, "focal": 0}
        real_dice, real_focal = module.to_dice_layout, module.to_focal_layout

        def counting_dice(stack: Any) -> Any:
            calls["dice"] += 1
            return real_dice(stack)

        def counting_focal(stack: Any) -> Any:
            calls["focal"] += 1
            return real_focal(stack)

        module.to_dice_layout = counting_dice
        module.to_focal_layout = counting_focal
        try:
            SAM2GatedMaskLoss()(truth, logits)
        finally:
            module.to_dice_layout = real_dice
            module.to_focal_layout = real_focal
        assert calls == {"dice": 2, "focal": 2}


# ---------------------------------------------------------------------
# the gate helper itself, and serialization
# ---------------------------------------------------------------------


class TestPresenceGate:
    """:func:`mask_presence_gate` is the SINGLE home of "presence"."""

    def test_it_returns_a_broadcastable_boolean_row_flag(self) -> None:
        truth, _ = discriminating_fixture()
        gate = np.asarray(mask_presence_gate(ops.convert_to_tensor(truth)))
        assert gate.shape == (2, 4, 1, 1)
        assert gate.dtype == bool
        np.testing.assert_array_equal(
            gate[..., 0, 0], ground_truth_gate(truth))

    def test_a_single_foreground_pixel_is_presence(self) -> None:
        """``> 0``, not ``>= 0.5`` and not a fraction of the mask.

        A downsampled binary mask is exactly 0 or 1, and upstream's
        ``target_obj`` is ``any(...)``. A threshold on the mask AREA would
        silently reclassify a barely-visible object as occluded.
        """
        masks = np.zeros((1, 2, 8, 8), dtype="float32")
        masks[0, 1, 3, 4] = 1.0
        gate = np.asarray(mask_presence_gate(ops.convert_to_tensor(masks)))
        assert gate[0, 0, 0, 0] == np.False_
        assert gate[0, 1, 0, 0] == np.True_


class TestSerialization:
    """Round trip by VALUE, through the inherited ``from_config``."""

    def test_config_round_trip_reproduces_the_loss(self) -> None:
        truth, logits = discriminating_fixture()
        original = SAM2GatedMaskLoss(focal_weight=7.0, dice_weight=3.0)
        restored = SAM2GatedMaskLoss.from_config(original.get_config())
        assert isinstance(restored, SAM2GatedMaskLoss)
        assert restored.focal_weight == 7.0
        assert restored.dice_weight == 3.0
        assert float(restored(truth, logits)) == pytest.approx(
            float(original(truth, logits)), abs=VALUE_TOLERANCE)

    def test_it_is_registered_under_its_own_key(self) -> None:
        assert keras.saving.get_registered_object(
            "Custom>SAM2GatedMaskLoss") is SAM2GatedMaskLoss


# ---------------------------------------------------------------------
# G3.6 -- dead-component probe. MEASURED, not hypothesized.
# ---------------------------------------------------------------------


class TestDeadComponentPartition:
    """Which guards survive a DEAD dice term, measured and pinned.

    Iteration 1 falsified "all guards go RED under a dead component" at five
    consecutive steps, so the partition is recorded as a test rather than as a
    prediction. With ``SegmentationWrapperLoss('dice')`` returning a constant,
    every VALUE guard in this file goes RED (the constant shifts the total by
    far more than :data:`VALUE_TOLERANCE`) while the gradient, adapter, gate
    and serialization guards stay GREEN -- the gradient guards because a
    constant has zero derivative everywhere, which zeroes the absent rows for
    the wrong reason and leaves the present rows live through focal alone.
    """

    def test_a_constant_dice_term_still_leaves_the_gradient_guard_green(
            self) -> None:
        """So the gradient guard is NOT evidence that dice is alive."""
        truth, logits = discriminating_fixture()
        loss = SAM2GatedMaskLoss()
        loss._dice = lambda *args, **kwargs: ops.convert_to_tensor(0.5)
        gradient = logit_gradient(loss, truth, logits)
        gate = ground_truth_gate(truth)
        assert np.count_nonzero(gradient[~gate]) == 0
        assert np.abs(gradient[gate]).max() > 0.0

    def test_a_constant_dice_term_DOES_break_the_value_guard(self) -> None:
        """And the value guard is what catches it."""
        truth, logits = discriminating_fixture()
        loss = SAM2GatedMaskLoss()
        loss._dice = lambda *args, **kwargs: ops.convert_to_tensor(0.5)
        expected = candidate_values(truth, logits)["gt_gated"]
        assert abs(float(loss(truth, logits)) - expected) > VALUE_TOLERANCE
