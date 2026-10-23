"""SAM 2's dice, IoU target and IoU loss equal upstream's ``loss_fns.py`` by value.

Upstream sources, fetched from facebookresearch/sam2 ``main`` (plan
2026-10-14T062616-81938e81, D-282 / D-283; untrusted copies under that plan's
``scratch_iter3/upstream_sam2/``):

* ``training/loss_fns.py``, retrieved 2026-10-22T23:16:15Z, sha256
  ``457d7bd5c90fbab7c1a9012abc01e7f5c6583289bf51e366dea0a22f04e6ef38``;
* ``sam2/configs/sam2.1_training/sam2.1_hiera_b+_MOSE_finetune.yaml``,
  retrieved 2026-10-22T23:19:11Z, sha256
  ``d96919e157527116c3a80c6ec961151bda98e5eb988ae4dd85be6e7efbd2b51f``.

The oracles are float64 numpy transcriptions of:

* ``dice_loss`` (``loss_fns.py:20-49``): per mask
  ``1 - (2 * sum(p * t) + 1) / (sum(p) + sum(t) + 1)``, smoothing **1**
  (line 46);
* ``iou_loss`` (``loss_fns.py:93-123``): hard masks ``inputs > 0`` and
  ``targets > 0`` (lines 111-112), ``actual_ious = area_i / clamp(area_u,
  min=1)`` (line 115), and the **L1** loss (line 118) that the training config
  selects (``iou_use_l1_loss: true``, yaml line 290). ``supervise_all_iou:
  true`` (yaml line 289) averages the candidates (``_update_losses`` line 280);
  the term is gated by ``target_obj`` (line 291) and summed over objects
  divided by ``num_objects`` (lines 122 and 296), i.e. a mean over objects;
* ``sigmoid_focal_loss`` (``loss_fns.py:52-90``) for the full per-row value
  ``20 * focal + 1 * dice`` (weights: yaml lines 285-286), with the
  repository's SCALAR ``alpha`` in place of upstream's ``alpha_t`` (lines
  82-84). That one difference is the recorded ruling D-073 of plan
  2026-08-04T044628-4c240b4c and is not reopened (D-283).

The repository computes these terms in float32 (``dice_per_sample`` casts its
ground truth to float32, and a float64 prediction is then refused by the
backend), so every fixture is float32-representable and the oracle sees
exactly the values the loss sees. Each tolerance is stated beside the float32
error measured after the fix and the divergence measured at c09d3df03.

RED at c09d3df03 (this file against that commit's source, D-283): the dice arms (smoothing ``1e-6``, off by
up to ``0.5`` per row), the IoU-target arm (``(I + 1e-6) / (U + 1e-6)``: off by
``1e-6`` at ``I = 0``) and the IoU-loss arm (MSE ``0.2049`` against L1
``0.2778``) fail; the arms that pin what must NOT move pass.
"""

from typing import Dict

import keras
import numpy as np
import pytest
import tensorflow as tf

from dl_techniques.losses.sam2_video_loss import SAM2GatedMaskLoss
from dl_techniques.losses.sam_mask_loss import SAMIoULoss, SAMMaskLoss
from dl_techniques.losses.segmentation_loss import LossConfig
from dl_techniques.models.vision_language.sam.sam2.model import NO_OBJ_SCORE
from dl_techniques.models.vision_language.sam.sam2.training_model import (
    SAM2_IOU_SUPERVISION,
    SAM2TrainingModel,
    compile_sam2_video_trainer,
)

from .test_training_model import trainer

#: ``dice_loss``'s smoothing, ``loss_fns.py:46``. A number, not read from the
#: loss, so a changed default fails.
UPSTREAM_DICE_SMOOTH = 1.0
#: ``weight_dict`` ``loss_mask`` / ``loss_dice``, yaml lines 285-286.
UPSTREAM_FOCAL_WEIGHT = 20.0
UPSTREAM_DICE_WEIGHT = 1.0
#: ``sigmoid_focal_loss`` defaults, ``loss_fns.py:56-57`` (and 130-131).
UPSTREAM_ALPHA = 0.25
UPSTREAM_GAMMA = 2.0
#: The repository's shared default, which SAM 1 and every other segmentation
#: user keep (D-082, D-282). Pinned so the SAM 2 fix cannot be made by moving it.
SHARED_SMOOTH_FACTOR = 1e-6

GRID = 16
#: Foreground counts of the dice fixture: an absent row, one pixel, f = 1/32,
#: f = 1/4. Each appears once with random logits and once at the sentinel.
FOREGROUND_COUNTS = (0, 1, 8, 64)

#: Dice rows: MEASURED float32 error after the fix ``3.0e-08``; the smallest
#: HEAD divergence on a present row is ``3.8e-03`` (random logits, N_fg 64).
DICE_ATOL = 1e-6
#: Full per-row value (up to ``1280.98``): MEASURED float32 relative error
#: after the fix ``6.7e-08``; the HEAD divergence at the f = 1/4 sentinel row is
#: ``1.2e-05`` relative.
ROW_RTOL = 1e-6
#: IoU target: ``I / U`` rounded to float32 is within ``6e-8`` relative
#: (MEASURED after the fix: ``3.0e-08`` at ``I, U = 2, 3``); HEAD's
#: ``(I + 1e-6) / (U + 1e-6)`` is ``4.8e-07`` relative off at ``I, U = 1, 2``.
IOU_TARGET_RTOL = 1e-7


def sigmoid64(x: np.ndarray) -> np.ndarray:
    """``0.5 * (1 + tanh(x / 2))``: no overflow at the ``-1024`` sentinel.

    :param x: Logits.
    :type x: numpy.ndarray
    :return: ``sigmoid(x)`` in float64.
    :rtype: numpy.ndarray
    """
    return 0.5 * (1.0 + np.tanh(0.5 * x))


def upstream_dice_rows(truth: np.ndarray, logits: np.ndarray) -> np.ndarray:
    """``dice_loss`` per mask (``loss_on_multimask``, ``num_objects = 1``), gated.

    Lines 34-46: sigmoid, flatten the spatial axes, ``2 * sum(p t)`` over
    ``sum(p) + sum(t)``, smoothing 1 on both. Multiplied by ``target_obj``
    (line 290).

    :param truth: ``(..., h, w)`` binary ground truth.
    :type truth: numpy.ndarray
    :param logits: ``(..., h, w)`` mask logits.
    :type logits: numpy.ndarray
    :return: ``(...)`` per-row dice loss.
    :rtype: numpy.ndarray
    """
    t = truth.astype(np.float64)
    p = sigmoid64(logits.astype(np.float64))
    numerator = 2.0 * (p * t).sum(axis=(-2, -1))
    denominator = p.sum(axis=(-2, -1)) + t.sum(axis=(-2, -1))
    dice = 1.0 - (numerator + UPSTREAM_DICE_SMOOTH) / (
        denominator + UPSTREAM_DICE_SMOOTH)
    return dice * (t.max(axis=(-2, -1)) > 0.0)


def upstream_rows_with_scalar_alpha(truth: np.ndarray,
                                    logits: np.ndarray) -> np.ndarray:
    """``20 * focal + 1 * dice`` per row, gated; focal with the scalar alpha.

    Focal is ``sigmoid_focal_loss`` (lines 77-80 and 89: BCE with logits times
    ``(1 - p_t)^gamma``, pixel mean per mask) with ``alpha`` applied as a scalar
    (D-073 of plan 2026-08-04T044628-4c240b4c) instead of lines 82-84's
    ``alpha_t``. BCE is written ``t * softplus(-x) + (1 - t) * softplus(x)``.

    :param truth: ``(..., h, w)`` binary ground truth.
    :type truth: numpy.ndarray
    :param logits: ``(..., h, w)`` mask logits.
    :type logits: numpy.ndarray
    :return: ``(...)`` per-row loss.
    :rtype: numpy.ndarray
    """
    t = truth.astype(np.float64)
    x = logits.astype(np.float64)
    p = sigmoid64(x)
    cross_entropy = t * np.logaddexp(0.0, -x) + (1.0 - t) * np.logaddexp(0.0, x)
    p_t = p * t + (1.0 - p) * (1.0 - t)
    focal = (UPSTREAM_ALPHA * cross_entropy
             * (1.0 - p_t) ** UPSTREAM_GAMMA).mean(axis=(-2, -1))
    focal = focal * (t.max(axis=(-2, -1)) > 0.0)
    return (UPSTREAM_FOCAL_WEIGHT * focal
            + UPSTREAM_DICE_WEIGHT * upstream_dice_rows(truth, logits))


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


def dice_fixture() -> Dict[str, np.ndarray]:
    """One ``(1, 8, GRID, GRID)`` stack: four random rows, four sentinel rows.

    :return: ``{"truth": ..., "logits": ...}``, float32.
    :rtype: Dict[str, numpy.ndarray]
    """
    rng = np.random.default_rng(0)
    truth = np.stack([foreground_block(n) for n in FOREGROUND_COUNTS] * 2)
    logits = np.concatenate([
        rng.normal(0.0, 3.0, (4, GRID, GRID)),
        np.full((4, GRID, GRID), NO_OBJ_SCORE),
    ]).astype("float32")
    return {"truth": truth[None], "logits": logits[None]}


class TestTheDiceTerm:
    """``dice_loss``'s smoothing 1, for SAM 2 only."""

    def test_the_dice_rows_are_upstreams_dice_loss(self) -> None:
        """Absent, 1, 8 and 64 foreground pixels; random logits and the sentinel."""
        fixture = dice_fixture()
        rows = np.asarray(SAM2GatedMaskLoss(
            focal_weight=0.0, dice_weight=1.0).per_mask(
                keras.ops.convert_to_tensor(fixture["truth"]),
                keras.ops.convert_to_tensor(fixture["logits"])))
        expected = upstream_dice_rows(fixture["truth"], fixture["logits"])
        # The sentinel rows in closed form: p = 0, so 1 - 1 / (N_fg + 1).
        np.testing.assert_allclose(
            expected[0, 5:], [1.0 - 1.0 / (n + 1.0) for n in (1, 8, 64)],
            rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(rows, expected, rtol=0.0, atol=DICE_ATOL)
        assert rows[0, 0] == 0.0 and rows[0, 4] == 0.0

    def test_the_per_row_value_is_upstreams_with_the_scalar_alpha(self) -> None:
        """The default loss: ``20 * focal + dice``; at f = 1/4 the sentinel row is 1280.98."""
        fixture = dice_fixture()
        rows = np.asarray(SAM2GatedMaskLoss().per_mask(
            keras.ops.convert_to_tensor(fixture["truth"]),
            keras.ops.convert_to_tensor(fixture["logits"])))
        expected = upstream_rows_with_scalar_alpha(
            fixture["truth"], fixture["logits"])
        assert expected[0, 7] == pytest.approx(
            20.0 * 0.25 * 1024.0 * 0.25 + 1.0 - 1.0 / 65.0, rel=1e-12)
        np.testing.assert_allclose(rows, expected, rtol=ROW_RTOL, atol=1e-6)

    def test_the_smoothing_survives_a_round_trip(self) -> None:
        """The SAM 2 default is carried by the loss's own config."""
        restored = SAM2GatedMaskLoss.from_config(SAM2GatedMaskLoss().get_config())
        assert restored.config.smooth_factor == UPSTREAM_DICE_SMOOTH

    def test_sam_1_and_the_shared_default_do_not_move(self) -> None:
        """The fix is SAM 2's; ``LossConfig`` and ``SAMMaskLoss`` keep ``1e-6``."""
        assert LossConfig(num_classes=2).smooth_factor == SHARED_SMOOTH_FACTOR
        assert SAMMaskLoss().config.smooth_factor == SHARED_SMOOTH_FACTOR


@pytest.fixture(scope="module")
def built_trainer() -> SAM2TrainingModel:
    """One built ``tiny`` wrapper for the IoU arms (one slot per frame).

    :return: The wrapper.
    :rtype: SAM2TrainingModel
    """
    keras.utils.set_random_seed(0)
    return trainer(object_score=5.0, multimask_output_in_sam=False)


#: ``(intersection, union)`` cell counts of the IoU-target rows. ``I = 0``
#: rows are where HEAD's smoothing is visible as a non-zero target.
IOU_ROWS = ((1, 2), (1, 3), (2, 3), (5, 64), (0, 1), (0, 64), (63, 64),
            (64, 64))


class TestTheIoUTarget:
    """``actual_ious = area_i / clamp(area_u, min=1)`` on hard masks."""

    def test_the_achieved_iou_is_area_i_over_the_clamped_union(
            self, built_trainer: SAM2TrainingModel) -> None:
        """Through ``SAM2TrainingModel._iou_supervision``, the trainer's call site.

        Each row predicts its first ``I`` cells (logit +4, the rest -4); its
        ground truth is the first ``U`` cells, so ``area_i = I`` and
        ``area_u = U``. A ground-truth-absent row is appended: its pair is
        zeroed by the gate, as upstream multiplies by ``target_obj``.
        """
        logits = np.full((len(IOU_ROWS) + 1, GRID * GRID), -4.0, "float32")
        truth = np.zeros((len(IOU_ROWS) + 1, GRID * GRID), "float32")
        for row, (inter, union) in enumerate(IOU_ROWS):
            logits[row, :inter] = 4.0
            truth[row, :union] = 1.0
        logits[-1, :5] = 4.0
        logits = logits.reshape(-1, 1, GRID, GRID)
        truth = truth.reshape(-1, 1, GRID, GRID)

        packed = np.asarray(built_trainer._iou_supervision(
            {"low_res_logits": keras.ops.convert_to_tensor(logits),
             "iou_predictions": keras.ops.zeros((logits.shape[0], 1))},
            keras.ops.convert_to_tensor(truth)))
        achieved = packed[:, 0, 1]

        predicted_mask = logits.reshape(logits.shape[0], -1) > 0.0
        true_mask = truth.reshape(truth.shape[0], -1) > 0.0
        area_i = (predicted_mask & true_mask).sum(-1).astype(np.float64)
        area_u = (predicted_mask | true_mask).sum(-1).astype(np.float64)
        expected = area_i / np.maximum(area_u, 1.0)
        expected[-1] = 0.0

        np.testing.assert_array_equal(area_i[:-1], [r[0] for r in IOU_ROWS])
        np.testing.assert_array_equal(area_u[:-1], [r[1] for r in IOU_ROWS])
        empty = area_i == 0.0
        np.testing.assert_array_equal(achieved[empty], 0.0)
        np.testing.assert_allclose(
            achieved[~empty], expected[~empty], rtol=IOU_TARGET_RTOL, atol=0.0)


#: ``(objects, candidates)`` predicted and achieved IoU pairs of the IoU-loss
#: arm, exact in float32. Object 2 is ground-truth absent, so the model has
#: zeroed both channels.
PREDICTED_IOU = np.array([[0.875, 0.25, 0.5], [0.125, 0.625, 1.0],
                          [0.0, 0.0, 0.0]], dtype="float32")
ACHIEVED_IOU = np.array([[0.25, 0.5, 0.5], [0.75, 0.625, 0.0],
                         [0.0, 0.0, 0.0]], dtype="float32")


def upstream_iou_loss(predicted: np.ndarray, achieved: np.ndarray) -> float:
    """``iou_loss`` with ``use_l1_loss`` and ``supervise_all_iou``, one frame.

    Lines 118 and 122: ``|pred - actual| / num_objects`` per candidate; line
    280: mean over candidates; lines 291 and 296: times ``target_obj``, summed
    over objects. ``num_objects`` is the object count.

    :param predicted: ``(N, K)`` predicted IoU.
    :type predicted: numpy.ndarray
    :param achieved: ``(N, K)`` achieved IoU.
    :type achieved: numpy.ndarray
    :return: The frame's IoU loss.
    :rtype: float
    """
    target_obj = np.array([1.0, 1.0, 0.0])
    per_object = np.abs(predicted.astype(np.float64)
                        - achieved.astype(np.float64)).mean(-1)
    return float((per_object * target_obj).sum() / predicted.shape[0])


class TestTheIoULoss:
    """The trainer's compiled IoU loss is upstream's L1, not MSE."""

    def test_the_compiled_iou_loss_is_upstreams_l1(
            self, built_trainer: SAM2TrainingModel) -> None:
        """Value and gradient of ``compile_sam2_video_trainer``'s IoU loss."""
        compile_sam2_video_trainer(built_trainer)
        loss = built_trainer.loss[SAM2_IOU_SUPERVISION]

        predicted = tf.Variable(PREDICTED_IOU)
        with tf.GradientTape() as tape:
            packed = keras.ops.stack(
                [predicted, keras.ops.convert_to_tensor(ACHIEVED_IOU)], axis=-1)
            value = loss(np.zeros((3, 2), dtype="float32"), packed)
        gradient = np.asarray(tape.gradient(value, predicted))

        expected = upstream_iou_loss(PREDICTED_IOU, ACHIEVED_IOU)
        assert expected == pytest.approx(0.2777777777777778, rel=1e-12)
        assert float(value) == pytest.approx(expected, rel=1e-6)
        sign = np.sign(PREDICTED_IOU.astype(np.float64) - ACHIEVED_IOU)
        np.testing.assert_allclose(gradient, sign / (3 * 3), rtol=1e-6, atol=0.0)

    def test_sam_1s_iou_loss_stays_mse(self) -> None:
        """``SAMIoULoss()``'s default, which SAM 1 compiles, does not move."""
        packed = np.stack([PREDICTED_IOU, ACHIEVED_IOU], axis=-1)
        squared = (PREDICTED_IOU.astype(np.float64) - ACHIEVED_IOU) ** 2
        assert float(SAMIoULoss()(np.zeros((3, 2), "float32"), packed)) == (
            pytest.approx(float(squared.mean()), rel=1e-6))
