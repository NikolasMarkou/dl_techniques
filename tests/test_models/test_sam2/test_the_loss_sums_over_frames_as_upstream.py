"""SAM 2's training loss sums over frames and averages over objects, as upstream does.

Upstream source (plan 2026-10-14T062616-81938e81, D-284 / D-285; untrusted copy
under that plan's ``scratch_iter3/upstream_sam2/``): facebookresearch/sam2
``main`` ``training/loss_fns.py``, retrieved 2026-10-22T23:16:15Z, sha256
``457d7bd5c90fbab7c1a9012abc01e7f5c6583289bf51e366dea0a22f04e6ef38``.

:func:`upstream_forward` transcribes ``MultiStepMultiMasksAndIous.forward``
(lines 167-182): ``num_objects = clamp(targets_batch.shape[1], min=1)``, the
objects of one frame, then a loop over the frames that ADDS each frame's losses
(``losses[k] += v``). Each frame's terms are ``_update_losses`` (lines
218-297): every per-object term is divided by ``num_objects`` and summed over
the objects (lines 88-90, 47-49, 121-123 and 293-297), ``supervise_all_iou``
averages the candidates (line 280), the focal, dice and IoU terms are
multiplied by ``target_obj`` (lines 289-291), and ``loss_class`` is the BCE
mean over objects (line 90 at ``alpha = -1``, ``gamma = 0``). The weights are
20, 1, 1, 1 (the ``sam2.1`` training config, D-283). This port has one object
per clip, so ``num_objects`` is the batch size and the target reduction is a
sum over ``T`` of a mean over ``B``.

The per-row formulas are the ones Step 5.1 already pinned
(``test_the_loss_terms_match_upstream_loss_fns.py``): dice smoothing 1, the
focal with the repository's scalar ``alpha`` (D-073 of plan
2026-08-04T044628-4c240b4c), the L1 IoU loss. The IoU pairs and the selected
frame-0 candidate are read from the model's own outputs; what this file pins
is the REDUCTION over ``(B, T)``, through the real training model, its
``compile_sam2_video_trainer`` losses and ``compute_loss``.

RED at 318829aea (the port's mean over ``(B, T)``): every term and the total
are ``1 / T`` of the oracle (ratio 2.000000 and 4.000000), so ``T = 2`` and
``T = 4`` fail (8 of 12) and ``T = 1`` passes. In place after D-285 (saved
bytes written back, sha256 equal): ``frame_sum`` taking the mean over frames
again fails the same 8; the three compiled losses at ``reduction="sum"`` (a
sum over the batch) fail all 12, ``T = 1`` included.
"""

import functools
from collections import defaultdict
from typing import Dict, Tuple

import keras
import numpy as np
import pytest

from dl_techniques.models.vision_language.sam.sam2.training_model import (
    SAM2_IOU_SUPERVISION,
    SAM2_LOW_RES_LOGITS,
    SAM2_OBJECT_SCORE_LOGITS,
    compile_sam2_video_trainer,
)

from .test_the_loss_terms_match_upstream_loss_fns import (
    upstream_dice_rows,
    upstream_rows_with_scalar_alpha,
)
from .test_training_model import BATCH, clip_inputs, trainer

#: ``weight_dict`` of the ``sam2.1`` training config (yaml lines 285-288).
UPSTREAM_WEIGHTS = {"loss_mask": 20.0, "loss_dice": 1.0, "loss_iou": 1.0,
                    "loss_class": 1.0}
#: Pinned object-score logit: positive, so the masks are live (a negative score
#: makes every logit the ``-1024`` sentinel; ``test_training_model.py``).
OBJECT_SCORE = 5.0
#: Clip lengths. ``T = 1`` is the control: there a sum and a mean agree.
FRAME_COUNTS = (1, 2, 4)
#: MEASURED float32 relative error of every term against this float64 oracle
#: after D-285: at most 6.9e-08 (``T = 4``, the IoU term); a missing frame sum
#: is off by ``(T - 1) / T >= 0.5`` and a sum over the batch by ``B - 1 = 1``.
RTOL = 1e-5

TERMS = ("mask", "object_score", "iou", "total")


def softplus64(x: np.ndarray) -> np.ndarray:
    """``log(1 + exp(x))`` in float64, stable at large ``|x|``.

    :param x: Input.
    :type x: numpy.ndarray
    :return: ``softplus(x)``.
    :rtype: numpy.ndarray
    """
    return np.logaddexp(0.0, x)


def upstream_frame(truth: np.ndarray, logits: np.ndarray, scores: np.ndarray,
                   iou_pairs: np.ndarray, num_objects: float
                   ) -> Dict[str, float]:
    """``_update_losses`` for one frame of ``N`` objects, one prediction step.

    :param truth: ``(N, h, w)`` binary ground truth of the frame.
    :type truth: numpy.ndarray
    :param logits: ``(N, h, w)`` the supervised mask (the lowest-loss
        candidate on a multimask frame, selected by the model).
    :type logits: numpy.ndarray
    :param scores: ``(N,)`` object-score logits.
    :type scores: numpy.ndarray
    :param iou_pairs: ``(N, K, 2)`` predicted and achieved IoU of the frame's
        ``K`` candidates.
    :type iou_pairs: numpy.ndarray
    :param num_objects: ``forward``'s clamped object count.
    :type num_objects: float
    :return: ``loss_mask``, ``loss_dice``, ``loss_iou``, ``loss_class``.
    :rtype: Dict[str, float]
    """
    target_obj = (truth.astype(np.float64).max(axis=(-2, -1)) > 0.0).astype(
        np.float64)
    dice = upstream_dice_rows(truth, logits)
    rows = upstream_rows_with_scalar_alpha(truth, logits)
    focal = (rows - UPSTREAM_WEIGHTS["loss_dice"] * dice) / UPSTREAM_WEIGHTS[
        "loss_mask"]
    predicted = iou_pairs[..., 0].astype(np.float64)
    achieved = iou_pairs[..., 1].astype(np.float64)
    iou = (np.abs(predicted - achieved) / num_objects).mean(axis=-1)
    x = scores.astype(np.float64)
    bce = target_obj * softplus64(-x) + (1.0 - target_obj) * softplus64(x)
    return {
        "loss_mask": float((focal / num_objects * target_obj).sum()),
        "loss_dice": float((dice / num_objects * target_obj).sum()),
        "loss_iou": float((iou * target_obj).sum()),
        "loss_class": float(bce.sum() / num_objects),
    }


def upstream_forward(truth: np.ndarray, outputs: Dict[str, np.ndarray]
                     ) -> Dict[str, float]:
    """``MultiStepMultiMasksAndIous.forward`` (lines 167-182) over a clip batch.

    ``targets_batch`` is frames-first, ``[T, N, h, w]``; ``num_objects`` is its
    second axis clamped at 1, and the frames' losses are ADDED.

    :param truth: ``(B, T, h, w)`` ground truth (one object per clip).
    :type truth: numpy.ndarray
    :param outputs: The wrapper's outputs, as numpy.
    :type outputs: Dict[str, numpy.ndarray]
    :return: The four summed terms and the weighted ``core_loss``.
    :rtype: Dict[str, float]
    """
    targets_batch = np.transpose(truth, (1, 0, 2, 3))
    num_objects = max(float(targets_batch.shape[1]), 1.0)
    frames = targets_batch.shape[0]
    pairs = outputs[SAM2_IOU_SUPERVISION]
    slots = pairs.shape[1] // frames
    losses: Dict[str, float] = defaultdict(float)
    for t, targets in enumerate(targets_batch):
        current = upstream_frame(
            targets, outputs[SAM2_LOW_RES_LOGITS][:, t],
            outputs[SAM2_OBJECT_SCORE_LOGITS][:, t, 0],
            pairs[:, t * slots:(t + 1) * slots], num_objects)
        for key, value in current.items():
            losses[key] += value
    losses["core_loss"] = sum(
        losses[key] * weight for key, weight in UPSTREAM_WEIGHTS.items())
    return dict(losses)


@functools.lru_cache(maxsize=None)
def measured(frames: int) -> Tuple[Dict[str, float], Dict[str, float]]:
    """The compiled terms and ``compute_loss`` total, and the oracle, at ``T``.

    One ``tiny`` wrapper per ``T``, seeded, the object score pinned positive,
    a single-click frame 0 (three candidates, ``K = 3`` IoU slots on frame 0)
    and, for ``T > 1``, the last frame ground-truth absent. Inference mode:
    the outputs are read once and every loss is evaluated on those arrays.

    :param frames: ``T``.
    :type frames: int
    :return: ``(port, oracle)``, each keyed by :data:`TERMS`.
    :rtype: Tuple[Dict[str, float], Dict[str, float]]
    """
    keras.utils.set_random_seed(0)
    model = trainer(num_frames=frames, object_score=OBJECT_SCORE)
    absent = (frames - 1,) if frames > 1 else ()
    inputs = clip_inputs(model, absent_frames=absent)
    outputs = {key: np.asarray(value)
               for key, value in model(inputs, training=False).items()}
    truth = inputs["gt_masks"]
    present = (truth.max(axis=(-2, -1)) > 0.0).astype("float32")[..., None]
    y_true = {
        SAM2_LOW_RES_LOGITS: truth,
        SAM2_OBJECT_SCORE_LOGITS: present,
        SAM2_IOU_SUPERVISION: np.zeros_like(outputs[SAM2_IOU_SUPERVISION]),
    }
    compile_sam2_video_trainer(model)
    terms = {
        "mask": SAM2_LOW_RES_LOGITS,
        "object_score": SAM2_OBJECT_SCORE_LOGITS,
        "iou": SAM2_IOU_SUPERVISION,
    }
    port = {name: float(model.loss[key](y_true[key], outputs[key]))
            for name, key in terms.items()}
    port["total"] = float(model.compute_loss(
        x=None, y=y_true, y_pred=outputs, training=False))

    upstream = upstream_forward(truth, outputs)
    oracle = {
        "mask": (UPSTREAM_WEIGHTS["loss_mask"] * upstream["loss_mask"]
                 + UPSTREAM_WEIGHTS["loss_dice"] * upstream["loss_dice"]),
        "object_score": upstream["loss_class"],
        "iou": upstream["loss_iou"],
        "total": upstream["core_loss"],
    }
    return port, oracle


@pytest.mark.parametrize("frames", FRAME_COUNTS)
@pytest.mark.parametrize("term", TERMS)
def test_the_compiled_loss_is_upstreams_sum_over_frames(
        frames: int, term: str) -> None:
    """Each compiled term, and ``compute_loss``, equals the transcription."""
    port, oracle = measured(frames)
    assert oracle[term] > 0.0, f"the {term} oracle is 0: the fixture is vacuous"
    assert port[term] == pytest.approx(oracle[term], rel=RTOL), (
        f"T = {frames}, B = {BATCH}: the port's {term} loss is {port[term]!r}, "
        f"upstream's frame sum {oracle[term]!r} (ratio "
        f"{oracle[term] / port[term] if port[term] else float('inf'):.6f})")


def test_the_fixture_has_an_absent_frame_and_three_slots_on_frame_zero() -> None:
    """The oracle's gate and its ``K``-slot grouping are exercised, not assumed."""
    keras.utils.set_random_seed(0)
    model = trainer(num_frames=2, object_score=OBJECT_SCORE)
    inputs = clip_inputs(model, absent_frames=(1,))
    packed = np.asarray(model(inputs, training=False)[SAM2_IOU_SUPERVISION])
    assert packed.shape == (BATCH, 2 * 3, 2)
    np.testing.assert_array_equal(packed[:, 3:], 0.0)
    assert (packed[:, :3, 0] != 0.0).all()
