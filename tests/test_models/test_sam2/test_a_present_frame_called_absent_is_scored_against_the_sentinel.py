"""A present frame the score head calls absent is scored against the sentinel.

What is pinned here, by value and end to end, is what this port shares with
upstream SAM 2's training loss on a frame whose object is PRESENT in the ground
truth while the object-score head calls it ABSENT (D-250 item 1, closed by
plan-2026-10-14T062616-81938e81 D-281):

* the mask logits of the frame are the ``NO_OBJ_SCORE = -1024`` sentinel
  (``SAM2._suppress_absent_object``; upstream ``sam2_base.py``
  ``torch.where(is_obj_appearing[:, None, None], low_res_multimasks,
  NO_OBJ_SCORE)``);
* the focal and dice terms are gated by GROUND-TRUTH presence, never by the
  predicted score (upstream ``_update_losses``: ``target_obj`` from
  ``target_masks``, then ``loss_mask * target_obj`` and ``loss_dice *
  target_obj``), so the frame is scored, sentinel against a present mask, and
  the loss VALUE is large;
* no gradient reaches the mask path through the unselected ``where`` branch;
* the object-score BCE is ungated and is the term that moves the score head
  (upstream ``loss_class``: focal with gamma 0, alpha -1, i.e. a plain BCE).

The upstream lines are the transcription F-12 records (fetched 2026-10-22 from
facebookresearch/sam2 main; ``findings/iter3-sam2-omnipoint-cliffordclip.md``
"Item 9"), and, since D-283, upstream's ``training/loss_fns.py`` (retrieved
2026-10-22T23:16:15Z, sha256
``457d7bd5c90fbab7c1a9012abc01e7f5c6583289bf51e366dea0a22f04e6ef38``) and the
``sam2.1_hiera_b+_MOSE_finetune.yaml`` training config (retrieved
2026-10-22T23:19:11Z, sha256
``d96919e157527116c3a80c6ec961151bda98e5eb988ae4dd85be6e7efbd2b51f``): the
dice smoothing 1 (``loss_fns.py:46``), the focal default ``alpha = 0.25``,
``gamma = 2`` (lines 56-57), the IoU target ``area_i / clamp(area_u, min=1)``
(line 115), the L1 IoU loss (line 118, yaml ``iou_use_l1_loss: true``) and the
weights 20, 1, 1, 1 (yaml lines 285-288). D-281's oracle took the dice
smoothing (``1e-6``) and the IoU target from this repository's documentation;
D-283 found both, and the MSE IoU loss, to differ from upstream and fixed them.

The oracle is a float64 numpy transcription of the FORMULAE, written in the
textbook forms rather than the forms the loss uses, so it does not assert that
the code equals itself: BCE as ``t * softplus(-x) + (1 - t) * softplus(x)``
(the loss spells it ``softplus(-x) + x * (1 - t)``), and upstream's row factor
``alpha_t = alpha * t + (1 - alpha) * (1 - t)`` where the loss applies a scalar
``alpha`` (D-073 of plan 2026-08-04T044628-4c240b4c). On a sentinel row the two
alpha forms agree: a background pixel's cross entropy is
``softplus(-1024) = 0`` exactly, so only foreground pixels, where
``alpha_t = alpha``, contribute. Hence the per-row value
``20 * 0.25 * 1024 * f + dice``, with ``f`` the foreground fraction and the
dice term ``1 - smooth / (N_fg + smooth) = 1 - 1 / (N_fg + 1)``: 1280.9846 at
f = 1/4 (1281.0 before D-283) and 160.8889 at f = 1/32.

RED proofs (in place, sha256 restored, D-281): gating ``per_mask`` by the
prediction instead of the ground truth fails the value arms; replacing
``_suppress_absent_object`` with the identity fails the gradient arm.
"""

from typing import Any, Dict, Tuple

import keras
import numpy as np
import pytest
import tensorflow as tf

from dl_techniques.losses.sam2_video_loss import SAM2GatedMaskLoss
from dl_techniques.models.vision_language.sam.sam2.model import NO_OBJ_SCORE
from dl_techniques.models.vision_language.sam.sam2.training_model import (
    SAM2_IOU_SUPERVISION,
    SAM2_LOW_RES_LOGITS,
    SAM2_OBJECT_SCORE_LOGITS,
    SAM2TrainingModel,
    compile_sam2_video_trainer,
)

from .test_training_model import (
    BATCH,
    FRAMES,
    clip_inputs,
    head_gradient,
    pin_predicted_iou,
    trainer,
)

#: Upstream's mask-loss weights, ``weight_dict`` ``loss_mask: 20`` and
#: ``loss_dice: 1`` (training yaml lines 285-286). Written here as numbers, not
#: read from the loss, so a changed default fails.
UPSTREAM_FOCAL_WEIGHT = 20.0
UPSTREAM_DICE_WEIGHT = 1.0
#: ``sigmoid_focal_loss``'s balancing factor and focusing exponent.
UPSTREAM_ALPHA = 0.25
UPSTREAM_GAMMA = 2.0
#: Upstream's ``dice_loss`` smoothing, ``loss_fns.py:46`` (D-283; D-281 used
#: the repository's ``1e-6``).
DICE_SMOOTH = 1.0

#: The object-score logit that calls every frame absent, and its opposite.
CALLED_ABSENT = -5.0
CALLED_PRESENT = 5.0

#: Mask grid of the per-row arm and its two foreground counts: f = 1/4 and
#: f = 1/32 (the latter gives D-250's "about 160" focal term).
GRID = 16
FOREGROUND_PIXELS = (64, 8)

#: Relative tolerance of every float32-vs-float64 value comparison here. The
#: per-row values are sums of exact float32 terms (256 per foreground pixel)
#: divided once, so the measured error is a few float32 ulps.
VALUE_RTOL = 1e-6


def softplus64(x: np.ndarray) -> np.ndarray:
    """``log(1 + exp(x))`` in float64, without forming ``exp(x)``.

    :param x: Logits.
    :type x: numpy.ndarray
    :return: ``softplus(x)``.
    :rtype: numpy.ndarray
    """
    return np.logaddexp(0.0, x)


def upstream_rows_oracle(truth: np.ndarray, logits: np.ndarray) -> np.ndarray:
    """Per-row ``20 * focal + dice`` with upstream's ground-truth gate, float64.

    Focal: upstream ``sigmoid_focal_loss`` (``alpha_t * ce * (1 - p_t)^gamma``,
    pixel mean per row). Dice: ``1 - (2 * sum(p t) + s) / (sum(p) + sum(t) + s)``
    per row. Both multiplied by ``target_obj = any(t > 0)`` per row, upstream's
    gate. The sigmoid is ``0.5 * (1 + tanh(x / 2))``, which does not overflow at
    the sentinel.

    :param truth: ``(..., h, w)`` binary ground truth.
    :type truth: numpy.ndarray
    :param logits: ``(..., h, w)`` mask logits, same shape.
    :type logits: numpy.ndarray
    :return: ``(...)`` per-row loss.
    :rtype: numpy.ndarray
    """
    t = truth.astype(np.float64)
    x = logits.astype(np.float64)
    probability = 0.5 * (1.0 + np.tanh(0.5 * x))
    cross_entropy = t * softplus64(-x) + (1.0 - t) * softplus64(x)
    p_t = probability * t + (1.0 - probability) * (1.0 - t)
    alpha_t = UPSTREAM_ALPHA * t + (1.0 - UPSTREAM_ALPHA) * (1.0 - t)
    focal = (alpha_t * cross_entropy * (1.0 - p_t) ** UPSTREAM_GAMMA).mean(
        axis=(-2, -1))
    numerator = 2.0 * (probability * t).sum(axis=(-2, -1))
    denominator = probability.sum(axis=(-2, -1)) + t.sum(axis=(-2, -1))
    dice = 1.0 - (numerator + DICE_SMOOTH) / (denominator + DICE_SMOOTH)
    target_obj = (t.max(axis=(-2, -1)) > 0.0).astype(np.float64)
    return target_obj * (UPSTREAM_FOCAL_WEIGHT * focal + UPSTREAM_DICE_WEIGHT * dice)


def foreground_block(pixels: int) -> np.ndarray:
    """A ``(GRID, GRID)`` mask whose first ``pixels`` cells are foreground.

    :param pixels: Foreground cell count.
    :type pixels: int
    :return: Binary float32 mask.
    :rtype: numpy.ndarray
    """
    mask = np.zeros(GRID * GRID, dtype="float32")
    mask[:pixels] = 1.0
    return mask.reshape(GRID, GRID)


def every_gradient(model: SAM2TrainingModel, inputs: Dict[str, np.ndarray]
                   ) -> Tuple[Any, ...]:
    """The wrapper's mask loss and its gradient on every trainable variable.

    :param model: The wrapper, its score pinned.
    :type model: SAM2TrainingModel
    :param inputs: A clip.
    :type inputs: Dict[str, np.ndarray]
    :return: ``(outputs, loss, [(variable, gradient), ...])``.
    :rtype: Tuple[Any, ...]
    """
    with tf.GradientTape() as tape:
        outputs = model(inputs, training=True)
        loss = model.mask_loss(inputs["gt_masks"], outputs[SAM2_LOW_RES_LOGITS])
    variables = model.trainable_variables
    gradients = tape.gradient(loss, variables)
    return outputs, loss, list(zip(variables, gradients))


def largest(gradient: Any) -> float:
    """``max|gradient|``, ``0.0`` for ``None``.

    :param gradient: A gradient as ``tape.gradient`` returns it.
    :type gradient: Any
    :return: The largest absolute element.
    :rtype: float
    """
    if gradient is None:
        return 0.0
    return float(np.max(np.abs(np.asarray(tf.convert_to_tensor(gradient)))))


class TestThePerRowValue:
    """(a) ``SAM2GatedMaskLoss.per_mask`` against the float64 oracle."""

    def test_a_sentinel_row_scores_upstreams_focal_and_dice(self) -> None:
        """Present and called absent: ``20 * 0.25 * 1024 * f + 1 - 1 / (N_fg + 1)``.

        Rows: f = 1/4 and f = 1/32 at the sentinel, then two ground-truth
        absent rows, one called present (logit +5) and one at the sentinel.
        """
        truth = np.stack([
            foreground_block(FOREGROUND_PIXELS[0]),
            foreground_block(FOREGROUND_PIXELS[1]),
            np.zeros((GRID, GRID), dtype="float32"),
            np.zeros((GRID, GRID), dtype="float32"),
        ])[None]
        logits = np.full(truth.shape, NO_OBJ_SCORE, dtype="float32")
        logits[0, 2] = CALLED_PRESENT

        rows = np.asarray(SAM2GatedMaskLoss().per_mask(
            keras.ops.convert_to_tensor(truth),
            keras.ops.convert_to_tensor(logits)))
        expected = upstream_rows_oracle(truth, logits)

        # The explorer's hypothesis is the FOCAL part of the oracle, exactly.
        fractions = np.asarray(FOREGROUND_PIXELS, dtype=np.float64) / GRID ** 2
        focal_only = UPSTREAM_FOCAL_WEIGHT * UPSTREAM_ALPHA * 1024.0 * fractions
        np.testing.assert_allclose(
            expected[0, :2] - focal_only,
            1.0 - DICE_SMOOTH / (np.asarray(FOREGROUND_PIXELS) + DICE_SMOOTH),
            rtol=0.0, atol=1e-9)

        np.testing.assert_allclose(rows[0, :2], expected[0, :2], rtol=VALUE_RTOL)
        assert rows[0, 2] == 0.0, (
            f"a ground-truth-absent row the model calls present scored "
            f"{rows[0, 2]!r}; the gate must come from the ground truth")
        assert rows[0, 3] == 0.0


class TestThroughTheTrainer:
    """(b) and (c): the same frame through ``SAM2TrainingModel``."""

    @staticmethod
    def _called(score: float) -> Tuple[SAM2TrainingModel, Dict[str, np.ndarray]]:
        """A seeded ``tiny`` wrapper with every frame's score pinned, and a clip.

        Frame 2 is ground-truth absent; frames 0 and 1 are present.

        :param score: The pinned object-score logit.
        :type score: float
        :return: ``(model, inputs)``.
        :rtype: Tuple[SAM2TrainingModel, Dict[str, numpy.ndarray]]
        """
        keras.utils.set_random_seed(0)
        model = trainer(object_score=score)
        return model, clip_inputs(model, absent_frames=(2,))

    def test_b_the_mask_loss_equals_the_oracle(self) -> None:
        """Every frame is the sentinel; the loss is the oracle's row mean."""
        model, inputs = self._called(CALLED_ABSENT)
        outputs, loss, _ = every_gradient(model, inputs)
        logits = np.asarray(outputs[SAM2_LOW_RES_LOGITS])
        np.testing.assert_array_equal(logits, NO_OBJ_SCORE)

        expected = float(upstream_rows_oracle(inputs["gt_masks"], logits).mean())
        assert float(loss) == pytest.approx(expected, rel=VALUE_RTOL), (
            f"mask loss {float(loss)!r}, oracle {expected!r}")
        assert expected > 100.0, "the fixture no longer gives a large value"

    def test_b_no_gradient_reaches_any_variable_through_the_sentinel(
            self) -> None:
        """``None`` or exactly 0 on every variable; positive when called present."""
        model, inputs = self._called(CALLED_ABSENT)
        _, _, pairs = every_gradient(model, inputs)
        decoder = {id(v) for v in model.sam2.mask_decoder.trainable_variables}
        assert any(id(v) in decoder for v, _ in pairs)
        moved = [(v.path, largest(g)) for v, g in pairs if largest(g) != 0.0]
        assert not moved, (
            f"{len(moved)} variables got a mask-loss gradient through the "
            f"-1024 sentinel, e.g. {moved[:3]}")

        model, inputs = self._called(CALLED_PRESENT)
        outputs, _, pairs = every_gradient(model, inputs)
        assert max(largest(g) for v, g in pairs if id(v) in {
            id(w) for w in model.sam2.mask_decoder.trainable_variables}) > 0.0
        rows = np.asarray(model.mask_loss.per_mask(
            keras.ops.convert_to_tensor(inputs["gt_masks"]),
            outputs[SAM2_LOW_RES_LOGITS]))
        assert (rows[:, 2] == 0.0).all() and (rows[:, :2] > 0.0).all(), (
            f"called present, frame 2 (ground-truth absent) must score 0 and "
            f"frames 0-1 must not: {rows!r}")

    def test_c_the_bce_moves_the_score_head_towards_present(self) -> None:
        """The ungated BCE: its value and its bias gradient, by value."""
        model, inputs = self._called(CALLED_ABSENT)
        present = (np.max(inputs["gt_masks"], axis=(-2, -1)) > 0.0)
        targets = {SAM2_OBJECT_SCORE_LOGITS:
                   present.astype("float32")[..., None]}
        bce = keras.losses.BinaryCrossentropy(from_logits=True)

        assert head_gradient(
            model, {SAM2_OBJECT_SCORE_LOGITS: bce}, inputs, targets) > 0.0

        bias = model.sam2.mask_decoder.pred_obj_score_head.layers[-1].bias
        with tf.GradientTape() as tape:
            scores = model(inputs, training=True)[SAM2_OBJECT_SCORE_LOGITS]
            value = bce(targets[SAM2_OBJECT_SCORE_LOGITS], scores)
        gradient = np.asarray(tape.gradient(value, bias))

        sigma = 1.0 / (1.0 + np.exp(-CALLED_ABSENT))
        target = present.astype(np.float64)
        expected_value = float(np.where(
            target > 0.0, softplus64(-CALLED_ABSENT),
            softplus64(CALLED_ABSENT)).mean())
        expected_gradient = float((sigma - target).mean())
        assert float(value) == pytest.approx(expected_value, rel=VALUE_RTOL)
        np.testing.assert_allclose(gradient, [expected_gradient], rtol=1e-5)
        assert expected_gradient < 0.0, "descent must raise the score"


class TestTheIoUHead:
    """(d) What the IoU head is trained towards on a present-called-absent frame."""

    def test_d_the_achieved_iou_is_the_empty_mask_and_the_head_is_pushed_down(
            self) -> None:
        """Target exactly ``0``, upstream's ``area_i / clamp(area_u, min=1)``.

        The sentinel mask thresholds to empty, so ``area_i = 0`` and the
        achieved IoU is exactly 0 (``loss_fns.py:115``; before D-283 the target
        was SAM 1's ``1e-6 / (N_fg + 1e-6)``). The frame is ground-truth
        present, so the pair is NOT zeroed, and the trainer's L1 IoU loss
        (``iou_use_l1_loss: true``, ``loss_fns.py:118``) regresses the IoU
        head's predicted value (pinned 0.9) towards it.
        """
        keras.utils.set_random_seed(0)
        model = trainer(object_score=CALLED_ABSENT, multimask_output_in_sam=False)
        pin_predicted_iou(model, 0.9)
        compile_sam2_video_trainer(model)
        iou_loss = model.loss[SAM2_IOU_SUPERVISION]
        inputs = clip_inputs(model, absent_frames=(2,))
        bias = model.sam2.mask_decoder.iou_prediction_head.layers[-1].bias
        with tf.GradientTape() as tape:
            packed = model(inputs, training=True)[SAM2_IOU_SUPERVISION]
            value = iou_loss(
                np.zeros((BATCH, FRAMES, 2), dtype="float32"), packed)
        gradient = np.asarray(tape.gradient(value, bias))
        packed = np.asarray(packed)

        present = np.max(inputs["gt_masks"], axis=(-2, -1)) > 0.0
        achieved = np.zeros(present.shape)
        predicted = np.where(present, np.float64(np.float32(0.9)), 0.0)

        np.testing.assert_array_equal(packed[:, 2], 0.0)
        np.testing.assert_array_equal(packed[:, :2, 0], np.float32(0.9))
        np.testing.assert_array_equal(packed[..., 1], achieved)

        absolute = np.abs(predicted - achieved)
        assert float(value) == pytest.approx(
            float(absolute.mean()), rel=VALUE_RTOL)
        expected_gradient = float(np.sign(predicted - achieved).sum()
                                  / (BATCH * FRAMES))
        # One bias entry per mask token; a single-mask frame reads token 0 only.
        expected_bias = np.zeros(gradient.shape)
        expected_bias[0] = expected_gradient
        np.testing.assert_allclose(gradient, expected_bias, rtol=1e-5)
        assert expected_gradient > 0.0, "descent must lower the predicted IoU"
