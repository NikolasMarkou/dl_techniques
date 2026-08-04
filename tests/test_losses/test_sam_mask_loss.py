"""
Guards for `SAMMaskLoss` / `SAMIoULoss` -- and for the two silent defects they
exist to contain.

The probes ARE the deliverable
------------------------------
`SegmentationLosses` cannot be pointed at SAM's mask stack as-is, and neither
defect is visible to a test that checks shapes, finiteness, or "the loss went
down":

* ``focal_loss`` on a 1-channel binary map is **bit-identically** blind to
  negatives. ``TestFocalNegativeBlindness`` reproduces that as a control, so
  the 2-channel adapter is justified by a measurement rather than by the
  finding that reported it.
* ``dice_loss`` on the raw ``(B, M, h, w)`` layout reduces ``(M, h)`` instead of
  ``(h, w)`` and returns a plausible number. Crucially it also *moves* under
  both destroy probes -- so a "does the loss react" test alone would certify
  the bug. ``TestRawLayoutIsRefused`` pins both halves: the wrong number, and
  the refusal.

Every loss path here carries a destroy-negatives AND a destroy-positives probe
with its two measured numbers, per invariant I-5.

Measured on GPU 1 (RTX 4070), keras 3.8.0 / tf 2.18.0.
"""

from typing import Any, Dict, Tuple

import keras
import numpy as np
import pytest
from keras import ops

from dl_techniques.losses.segmentation_loss import LossConfig, SegmentationLosses
from dl_techniques.losses.sam_mask_loss import (
    DICE_CHANNELS,
    FOCAL_CHANNELS,
    SAMIoULoss,
    SAMMaskLoss,
    to_dice_layout,
    to_focal_layout,
)
from dl_techniques.models.sam.training_model import achieved_mask_iou

from tests.test_models.test_sam.dead_component_oracle import (
    destroy_negatives,
    destroy_positives,
)

# ---------------------------------------------------------------------------
# Fixed probe data. A filled rectangle rather than a random mask: a random
# "mask" has no positive/negative STRUCTURE, and a destroy-one-class probe
# against it is close to destroying everything.
# ---------------------------------------------------------------------------
BATCH, MASKS, HEIGHT, WIDTH = 2, 1, 64, 64


def _probe_arrays(seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(gt, probs)`` -- a rectangular GT mask and random predictions."""
    gt = np.zeros((BATCH, MASKS, HEIGHT, WIDTH), dtype="float32")
    gt[:, :, 12:40, 20:52] = 1.0
    probs = 1.0 / (
        1.0
        + np.exp(
            -np.random.RandomState(seed)
            .normal(0.0, 1.0, (BATCH, MASKS, HEIGHT, WIDTH))
            .astype("float32")
        )
    )
    return gt, probs.astype("float32")


def _to_logits(probs: np.ndarray) -> np.ndarray:
    """Inverse sigmoid, so the same probabilities can drive a from_logits loss."""
    clipped = np.clip(probs, 1e-6, 1.0 - 1e-6)
    return np.log(clipped / (1.0 - clipped)).astype("float32")


def _scalar(value: Any) -> float:
    """Convert a Keras scalar to a Python float."""
    return float(ops.convert_to_numpy(value))


def _t(array: np.ndarray) -> Any:
    """Shorthand tensor conversion."""
    return ops.convert_to_tensor(array)


class TestProbeDataIsNotDegenerate:
    """Premise checks. A degenerate probe corpus passes against any loss."""

    def test_ground_truth_has_both_classes_in_useful_proportion(self) -> None:
        gt, _ = _probe_arrays()
        fraction = float(gt.mean())
        assert 0.05 < fraction < 0.5, (
            f"probe GT foreground fraction {fraction:.4f} is degenerate; a "
            "destroy-one-class probe against it would be near-total corruption"
        )

    def test_predictions_are_not_already_perfect_or_already_dead(self) -> None:
        _, probs = _probe_arrays()
        assert 0.01 < float(probs.min()) and float(probs.max()) < 0.99


class TestFocalNegativeBlindness:
    """
    A-5, reproduced as a CONTROL: the unadapted 1-channel focal path.

    Measured on this data: base ``0.018317``; after destroying every negative
    pixel, ``0.018317`` -- identical to six decimals. Destroying positives moves
    it to ``0.246834``, so the loss is alive, just structurally blind to one
    class. Values are recorded in decisions.md; the ASSERTION here is the
    invariance and the direction, never a cross-process constant.
    """

    def _focal_1ch(self, gt: np.ndarray, probs: np.ndarray) -> float:
        losses = SegmentationLosses(LossConfig(num_classes=1))
        return _scalar(losses.focal_loss(to_dice_layout(_t(gt)), to_dice_layout(_t(probs))))

    def test_one_channel_focal_is_bit_identically_blind_to_negatives(self) -> None:
        gt, probs = _probe_arrays()
        base = self._focal_1ch(gt, probs)
        destroyed = self._focal_1ch(gt, destroy_negatives(probs, gt))
        assert destroyed == base, (
            "the 1-channel focal path reacted to destroyed negatives "
            f"({base:.6f} -> {destroyed:.6f}); if this now holds, the 2-channel "
            "adapter's justification must be re-derived (assumption A-5)"
        )

    def test_one_channel_focal_is_alive_on_positives(self) -> None:
        """
        The control's own control: the blindness is class-specific, not a dead
        loss object. Without this, the test above would also pass for a loss
        that always returns a constant.
        """
        gt, probs = _probe_arrays()
        base = self._focal_1ch(gt, probs)
        destroyed = self._focal_1ch(gt, destroy_positives(probs, gt))
        assert destroyed > base * 2.0, f"{base:.6f} -> {destroyed:.6f}"


class TestAdaptedFocalPath:
    """The 2-channel one-hot adapter, with both destroy probes."""

    def _focal_2ch(self, gt: np.ndarray, probs: np.ndarray) -> float:
        losses = SegmentationLosses(LossConfig(num_classes=FOCAL_CHANNELS))
        return _scalar(
            losses.focal_loss(to_focal_layout(_t(gt)), to_focal_layout(_t(probs)))
        )

    def test_adapter_produces_a_complementary_one_hot(self) -> None:
        gt, probs = _probe_arrays()
        adapted = ops.convert_to_numpy(to_focal_layout(_t(probs)))
        assert adapted.shape == (BATCH * MASKS, HEIGHT, WIDTH, FOCAL_CHANNELS)
        assert np.allclose(adapted.sum(axis=-1), 1.0, atol=1e-6)
        assert np.allclose(adapted[..., 1], probs.reshape(-1, HEIGHT, WIDTH), atol=0.0)

    def test_destroying_negatives_moves_the_adapted_focal_loss(self) -> None:
        """Measured: ``0.084166 -> 0.899866``, i.e. ~10.7x."""
        gt, probs = _probe_arrays()
        base = self._focal_2ch(gt, probs)
        destroyed = self._focal_2ch(gt, destroy_negatives(probs, gt))
        assert destroyed > base * 3.0, f"{base:.6f} -> {destroyed:.6f}"

    def test_destroying_positives_moves_the_adapted_focal_loss(self) -> None:
        """Measured: ``0.084166 -> 0.312682``."""
        gt, probs = _probe_arrays()
        base = self._focal_2ch(gt, probs)
        destroyed = self._focal_2ch(gt, destroy_positives(probs, gt))
        assert destroyed > base * 1.5, f"{base:.6f} -> {destroyed:.6f}"


class TestAdaptedDicePath:
    """The ``(B*M, h, w, 1)`` adapter, with both destroy probes."""

    def _dice(self, gt: np.ndarray, probs: np.ndarray) -> float:
        losses = SegmentationLosses(LossConfig(num_classes=DICE_CHANNELS))
        return _scalar(
            losses.dice_loss(to_dice_layout(_t(gt)), to_dice_layout(_t(probs)))
        )

    def test_adapter_shape(self) -> None:
        _, probs = _probe_arrays()
        adapted = to_dice_layout(_t(probs))
        assert tuple(adapted.shape) == (BATCH * MASKS, HEIGHT, WIDTH, DICE_CHANNELS)

    def test_destroying_negatives_moves_dice(self) -> None:
        """Measured: ``0.693310 -> 0.800957``."""
        gt, probs = _probe_arrays()
        base = self._dice(gt, probs)
        destroyed = self._dice(gt, destroy_negatives(probs, gt))
        assert destroyed > base + 0.05, f"{base:.6f} -> {destroyed:.6f}"

    def test_destroying_positives_moves_dice(self) -> None:
        """Measured: ``0.693310 -> 0.992800``."""
        gt, probs = _probe_arrays()
        base = self._dice(gt, probs)
        destroyed = self._dice(gt, destroy_positives(probs, gt))
        assert destroyed > 0.95, f"{base:.6f} -> {destroyed:.6f}"

    def test_a_perfect_prediction_drives_dice_to_nearly_zero(self) -> None:
        """A calibration point: without it, "dice moved" says nothing about direction."""
        gt, _ = _probe_arrays()
        assert self._dice(gt, gt) < 1e-4


class TestRawLayoutIsRefused:
    """
    SC-5's second half. The raw ``(B, M, h, w)`` layout must be REFUSED, not
    silently reduced over ``(M, h)``.
    """

    def test_raw_layout_dice_returns_a_plausible_but_different_number(self) -> None:
        """
        The defect itself, pinned as a control. Measured: raw ``0.765062`` vs
        adapted ``0.693310`` on identical data -- both plausible, one wrong.
        """
        gt, probs = _probe_arrays()
        losses = SegmentationLosses(LossConfig(num_classes=1))
        raw = _scalar(losses.dice_loss(_t(gt), _t(probs)))
        adapted = _scalar(
            losses.dice_loss(to_dice_layout(_t(gt)), to_dice_layout(_t(probs)))
        )
        assert 0.0 < raw < 1.0, "the raw layout did not even look plausible"
        assert abs(raw - adapted) > 1e-3, (
            f"raw {raw:.6f} and adapted {adapted:.6f} agree, so this data cannot "
            "discriminate the wrong-axis reduction"
        )

    def test_the_raw_layout_also_reacts_to_both_destroy_probes(self) -> None:
        """
        Why a liveness probe alone is NOT enough, stated as an assertion.

        The wrong-axis reduction moves under destroy-negatives AND under
        destroy-positives, so an I-5 probe applied to the raw layout would go
        green while computing the wrong quantity. The refusal below is what
        actually protects the training path.
        """
        gt, probs = _probe_arrays()
        losses = SegmentationLosses(LossConfig(num_classes=1))
        base = _scalar(losses.dice_loss(_t(gt), _t(probs)))
        neg = _scalar(losses.dice_loss(_t(gt), _t(destroy_negatives(probs, gt))))
        pos = _scalar(losses.dice_loss(_t(gt), _t(destroy_positives(probs, gt))))
        assert neg > base and pos > base, f"{base:.6f} / {neg:.6f} / {pos:.6f}"

    def test_sam_mask_loss_refuses_a_non_rank_four_stack(self) -> None:
        gt, probs = _probe_arrays()
        loss = SAMMaskLoss()
        flat = probs.reshape(BATCH, MASKS * HEIGHT, WIDTH)
        with pytest.raises(ValueError, match="mask stack"):
            loss(_t(gt.reshape(BATCH, MASKS * HEIGHT, WIDTH)), _t(flat))

    def test_the_channels_last_guard_refuses_a_raw_mask_stack(self) -> None:
        """
        The guard's own discrimination: it must reject a ``(B, M, h, w)`` stack
        and accept the adapted layout, with a message naming the reduction.
        """
        from dl_techniques.losses.sam_mask_loss import _require_channels_last

        _, probs = _probe_arrays()
        with pytest.raises(ValueError, match="reduces axis=\\[1, 2\\]"):
            _require_channels_last(_t(probs), DICE_CHANNELS, "probe")
        # The adapted layout passes, so the guard is not simply always-raising.
        _require_channels_last(to_dice_layout(_t(probs)), DICE_CHANNELS, "probe")


class TestSAMMaskLoss:
    """The shipped loss, end to end, from logits."""

    def _loss(self, gt: np.ndarray, probs: np.ndarray, **kwargs: Any) -> float:
        return _scalar(SAMMaskLoss(**kwargs)(_t(gt), _t(_to_logits(probs))))

    def test_destroying_negatives_moves_the_shipped_loss(self) -> None:
        """Measured at 20:1: ``2.376630 -> 18.798296``."""
        gt, probs = _probe_arrays()
        base = self._loss(gt, probs)
        destroyed = self._loss(gt, destroy_negatives(probs, gt))
        assert destroyed > base * 3.0, f"{base:.6f} -> {destroyed:.6f}"

    def test_destroying_positives_moves_the_shipped_loss(self) -> None:
        """Measured at 20:1: ``2.376630 -> 7.246445``."""
        gt, probs = _probe_arrays()
        base = self._loss(gt, probs)
        destroyed = self._loss(gt, destroy_positives(probs, gt))
        assert destroyed > base * 1.5, f"{base:.6f} -> {destroyed:.6f}"

    def test_both_terms_are_live_in_the_mix(self) -> None:
        """
        A weight of zero must silence its term exactly, and neither term may be
        a no-op at the shipped defaults -- the dead-knob defect class.
        """
        gt, probs = _probe_arrays()
        focal_only = self._loss(gt, probs, focal_weight=1.0, dice_weight=0.0)
        dice_only = self._loss(gt, probs, focal_weight=0.0, dice_weight=1.0)
        both = self._loss(gt, probs, focal_weight=1.0, dice_weight=1.0)
        assert focal_only > 0.0 and dice_only > 0.0
        assert both == pytest.approx(focal_only + dice_only, rel=1e-5)

    def test_measured_term_scales_support_the_papers_ratio(self) -> None:
        """
        SC-6. The mixing weights are RE-MEASURED on the shipped code rather than
        pasted from the paper.

        Derivation command (recorded in decisions.md D-036)::

            CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m pytest \\
                tests/test_losses/test_sam_mask_loss.py -q -k measured_term_scales

        Measured on this data: unweighted focal ``0.084166``, unweighted dice
        ``0.693310`` -- dice is ``8.24x`` focal, so the two terms are NOT on the
        same scale and an unweighted sum would be dice-dominated. Applying the
        paper's 20:1 makes focal's contribution ``1.683320`` against dice's
        ``0.693310``, i.e. focal leads by ``2.43x`` in loss VALUE. The assertions
        are those two structural facts, not the constants.
        """
        gt, probs = _probe_arrays()
        focal = self._loss(gt, probs, focal_weight=1.0, dice_weight=0.0)
        dice = self._loss(gt, probs, focal_weight=0.0, dice_weight=1.0)
        assert dice > focal * 3.0, (
            f"focal {focal:.6f} and dice {dice:.6f} are on the same scale here, "
            "so the 20:1 weighting would make focal dominate outright and the "
            "shipped defaults must be re-derived"
        )
        weighted_focal = 20.0 * focal
        assert weighted_focal > dice, (
            f"at the shipped 20:1 the focal term ({weighted_focal:.6f}) no longer "
            f"leads dice ({dice:.6f}); the ratio is not doing what the paper says"
        )
        assert weighted_focal < dice * 10.0, (
            f"at the shipped 20:1 focal ({weighted_focal:.6f}) swamps dice "
            f"({dice:.6f}) by more than 10x; the ratio is nonsensical on this code"
        )

    def test_defaults_are_the_papers_ratio(self) -> None:
        loss = SAMMaskLoss()
        assert (loss.focal_weight, loss.dice_weight) == (20.0, 1.0)

    def test_config_round_trip(self) -> None:
        loss = SAMMaskLoss(focal_weight=3.0, dice_weight=7.0, from_logits=False)
        restored = SAMMaskLoss.from_config(loss.get_config())
        assert restored.focal_weight == 3.0
        assert restored.dice_weight == 7.0
        assert restored.from_logits is False
        gt, probs = _probe_arrays()
        assert _scalar(restored(_t(gt), _t(probs))) == pytest.approx(
            _scalar(loss(_t(gt), _t(probs))), rel=1e-6
        )

    def test_from_logits_false_consumes_probabilities_directly(self) -> None:
        """A knob that changes nothing is the defect class iteration 1 fixed."""
        gt, probs = _probe_arrays()
        as_logits = _scalar(SAMMaskLoss(from_logits=True)(_t(gt), _t(_to_logits(probs))))
        as_probs = _scalar(SAMMaskLoss(from_logits=False)(_t(gt), _t(probs)))
        assert as_logits == pytest.approx(as_probs, rel=1e-4)
        # ... and feeding logits to the probability path is NOT the same number.
        wrong = _scalar(SAMMaskLoss(from_logits=False)(_t(gt), _t(_to_logits(probs))))
        assert abs(wrong - as_probs) > 1e-3


class TestAchievedMaskIoU:
    """The IoU target the head is trained against."""

    def test_perfect_prediction_gives_one(self) -> None:
        gt, _ = _probe_arrays()
        perfect = (gt * 20.0 - 10.0).astype("float32")
        iou = ops.convert_to_numpy(achieved_mask_iou(_t(perfect), _t(gt)))
        assert np.allclose(iou, 1.0, atol=1e-5)

    def test_inverted_prediction_gives_zero(self) -> None:
        gt, _ = _probe_arrays()
        inverted = ((1.0 - gt) * 20.0 - 10.0).astype("float32")
        iou = ops.convert_to_numpy(achieved_mask_iou(_t(inverted), _t(gt)))
        assert np.allclose(iou, 0.0, atol=1e-5)

    def test_matches_a_hand_computed_iou(self) -> None:
        """A closed-form control; a plausible-looking number is not enough."""
        gt = np.zeros((1, 1, 4, 4), "float32")
        gt[0, 0, :2, :] = 1.0  # 8 positives
        pred = np.full((1, 1, 4, 4), -1.0, "float32")
        pred[0, 0, :3, :] = 1.0  # 12 predicted, 8 overlap, union 12
        iou = float(ops.convert_to_numpy(achieved_mask_iou(_t(pred), _t(gt)))[0, 0])
        assert iou == pytest.approx(8.0 / 12.0, abs=1e-5)

    def test_empty_union_is_one_not_nan(self) -> None:
        empty = np.zeros((1, 1, 4, 4), "float32")
        pred = np.full((1, 1, 4, 4), -1.0, "float32")
        iou = float(ops.convert_to_numpy(achieved_mask_iou(_t(pred), _t(empty)))[0, 0])
        assert iou == pytest.approx(1.0, abs=1e-5)


class TestSAMIoULoss:
    """MSE between the predicted IoU and the achieved IoU, read from one tensor."""

    @staticmethod
    def _pair(predicted: float, achieved: float) -> Any:
        return _t(
            np.stack(
                [
                    np.full((BATCH, MASKS), predicted, "float32"),
                    np.full((BATCH, MASKS), achieved, "float32"),
                ],
                axis=-1,
            )
        )

    def test_a_perfect_iou_prediction_is_exactly_zero(self) -> None:
        loss = SAMIoULoss()
        dummy = _t(np.zeros((BATCH, MASKS, 2), "float32"))
        assert _scalar(loss(dummy, self._pair(0.4, 0.4))) == 0.0

    def test_the_loss_is_the_squared_error(self) -> None:
        loss = SAMIoULoss()
        dummy = _t(np.zeros((BATCH, MASKS, 2), "float32"))
        assert _scalar(loss(dummy, self._pair(0.9, 0.4))) == pytest.approx(0.25, abs=1e-6)
        assert _scalar(loss(dummy, self._pair(0.5, 0.4))) == pytest.approx(0.01, abs=1e-6)

    def test_destroying_the_prediction_moves_the_loss(self) -> None:
        """The I-5 probe for this path: a corrupted prediction must be punished."""
        loss = SAMIoULoss()
        dummy = _t(np.zeros((BATCH, MASKS, 2), "float32"))
        base = _scalar(loss(dummy, self._pair(0.42, 0.40)))
        destroyed = _scalar(loss(dummy, self._pair(0.99, 0.40)))
        assert destroyed > base * 10.0, f"{base:.6f} -> {destroyed:.6f}"

    def test_a_mis_wired_output_key_is_refused(self) -> None:
        """
        Routing this loss to ``iou_predictions`` (shape ``(B, M)``) instead of
        ``iou_supervision`` would otherwise train the head against garbage.
        """
        loss = SAMIoULoss()
        with pytest.raises(ValueError, match="iou_supervision"):
            loss(_t(np.zeros((BATCH, MASKS), "float32")), _t(np.zeros((BATCH, MASKS), "float32")))

    def test_config_round_trip(self) -> None:
        loss = SAMIoULoss(name="custom_iou")
        restored = SAMIoULoss.from_config(loss.get_config())
        assert restored.name == "custom_iou"


class TestMaskAxisAlignmentAcrossRefinementRounds:
    """
    Plan step 4. ``SAMTrainingModel`` concatenates every refinement round's
    logits on the mask axis, so ``y_pred`` is ``(B, M*R, h, w)`` while the data
    pipeline's ``y_true`` stays ``(B, M, h, w)`` -- and must, or every data
    source would have to know the model's round count.
    """

    @staticmethod
    def _two_distinct_masks() -> np.ndarray:
        """A GT stack whose two masks are DIFFERENT, so mis-ordering shows."""
        gt = np.zeros((BATCH, 2, HEIGHT, WIDTH), dtype="float32")
        gt[:, 0, 8:24, 8:24] = 1.0
        gt[:, 1, 40:56, 40:56] = 1.0
        return gt

    def test_a_single_instance_gt_is_repeated_across_rounds(self) -> None:
        gt, probs = _probe_arrays()
        logits = _to_logits(probs)
        rounds = 3
        repeated_pred = np.concatenate([logits] * rounds, axis=1)
        repeated_true = np.concatenate([gt] * rounds, axis=1)
        loss = SAMMaskLoss()
        broadcast = _scalar(loss(_t(gt), _t(repeated_pred)))
        explicit = _scalar(loss(_t(repeated_true), _t(repeated_pred)))
        assert broadcast == pytest.approx(explicit, abs=1e-6), (
            f"repeating the GT internally gave {broadcast} but an explicitly "
            f"repeated y_true gave {explicit}"
        )

    def test_the_repeat_is_round_major_not_interleaved(self) -> None:
        """
        The one alignment error that returns a plausible number. With two
        distinct GT masks and two rounds, ``concatenate``/``tile`` order is
        ``[m0, m1, m0, m1]`` while ``ops.repeat`` would give ``[m0, m0, m1,
        m1]``. Feed a prediction that is PERFECT under the first ordering: the
        loss must be near its floor, and would be far from it under the other.
        """
        gt = self._two_distinct_masks()
        perfect = np.concatenate([gt, gt], axis=1) * 20.0 - 10.0
        interleaved = np.concatenate(
            [gt[:, 0:1], gt[:, 0:1], gt[:, 1:2], gt[:, 1:2]], axis=1
        ) * 20.0 - 10.0
        loss = SAMMaskLoss()
        aligned = _scalar(loss(_t(gt), _t(perfect)))
        misaligned = _scalar(loss(_t(gt), _t(interleaved)))
        assert misaligned > aligned * 10.0, (
            f"round-major alignment gave {aligned} and the interleaved "
            f"ordering gave {misaligned} -- this test cannot tell the two apart"
        )

    def test_a_non_multiple_mask_axis_is_refused(self) -> None:
        gt = self._two_distinct_masks()
        pred = np.concatenate([gt, gt[:, 0:1]], axis=1)
        with pytest.raises(ValueError, match="not a whole multiple"):
            SAMMaskLoss()(_t(gt), _t(pred))

    def test_matching_mask_axes_are_left_alone(self) -> None:
        """Non-firing control: the pre-step-4 shape must be untouched."""
        gt, probs = _probe_arrays()
        assert np.isfinite(_scalar(SAMMaskLoss()(_t(gt), _t(_to_logits(probs)))))
