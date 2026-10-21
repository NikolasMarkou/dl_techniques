"""SAM 2's multimask tokens train the way the reference trains them.

Guard for plan-2026-10-14T062616-81938e81 iteration 2 step 12.6(b) (D-249).
``SAM2TrainingModel`` decoded every frame with ``multimask_output=False``
(D-051 of plan 4c240b4c), so mask tokens 1-3 and their hypernetwork MLPs never
received a gradient, and the inference fallback to them answered with
untrained tokens (D-245). Upstream SAM 2 (facebookresearch/sam2 2b90b9f5,
``training/loss_fns.py`` ``MultiStepMultiMasksAndIous`` and ``sam2_base.py``
``_use_multimask``) decodes the three multimask candidates on a single-click
initial frame, back-propagates focal and dice through the candidate of lowest
``20 * focal + dice`` only, and supervises the IoU head on every candidate
(``supervise_all_iou: true``).

The fixture is ``test_training_model.py``'s: one click on frame 0, the object
score pinned positive (at initialisation the head says "absent" and every mask
logit is the ``-1024`` sentinel, which carries no gradient).
"""

from typing import Any, Dict, List

import keras
import numpy as np
import pytest
import tensorflow as tf

from dl_techniques.losses.sam2_video_loss import (
    SAM2GatedMaskLoss,
    select_lowest_loss_mask,
)
from dl_techniques.models.vision_language.sam.sam1.training_model import achieved_mask_iou
from dl_techniques.models.vision_language.sam.sam2.training_model import (
    SAM2_IOU_SUPERVISION,
    SAM2_LOW_RES_LOGITS,
)

from .test_training_model import clip_inputs, trainer

CANDIDATES = 3


def _hand_selection(truth: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """Per row, the candidate whose scalar ``SAM2GatedMaskLoss`` is lowest.

    Each loss is the loss class's own ``__call__`` on ONE ``(1, 1, h, w)`` pair,
    so the reduction under test (``per_mask``) is not used to build the answer.
    """
    loss = SAM2GatedMaskLoss()
    chosen = []
    for row in range(truth.shape[0]):
        values = [
            float(loss(truth[row:row + 1], candidates[row:row + 1, k:k + 1]))
            for k in range(candidates.shape[1])
        ]
        chosen.append(candidates[row, int(np.argmin(values))])
    return np.stack(chosen)[:, None]


def _recording_decode(model) -> List[Dict[str, Any]]:
    """Record every ``SAM2._decode`` output dict the wrapper produces."""
    seen: List[Dict[str, Any]] = []
    original = model.sam2._decode

    def spy(*args, **kwargs):
        out = original(*args, **kwargs)
        seen.append(out)
        return out

    model.sam2._decode = spy
    return seen


def test_the_selection_equals_a_hand_composition_by_value():
    rng = np.random.default_rng(3)
    truth = (rng.random((4, 1, 16, 16)) > 0.6).astype("float32")
    truth[3] = 0.0  # an absent row: every candidate scores 0, the first is taken
    candidates = rng.normal(0.0, 3.0, (4, CANDIDATES, 16, 16)).astype("float32")
    # Row 1's best candidate is made its last one, so the selection is not index 0
    # everywhere by accident.
    candidates[1, 2] = np.where(truth[1, 0] > 0, 6.0, -6.0)

    got = keras.ops.convert_to_numpy(
        select_lowest_loss_mask(SAM2GatedMaskLoss(), truth, candidates))
    want = _hand_selection(truth, candidates)
    np.testing.assert_array_equal(got, want)
    np.testing.assert_array_equal(got[1, 0], candidates[1, 2])
    np.testing.assert_array_equal(got[3, 0], candidates[3, 0])


def test_the_loss_mean_of_per_mask_is_the_scalar_loss():
    rng = np.random.default_rng(4)
    truth = (rng.random((2, 3, 16, 16)) > 0.5).astype("float32")
    truth[1, 2] = 0.0
    logits = rng.normal(0.0, 2.0, (2, 3, 16, 16)).astype("float32")
    loss = SAM2GatedMaskLoss()
    rows = keras.ops.convert_to_numpy(loss.per_mask(truth, logits))
    assert rows.shape == (2, 3) and rows[1, 2] == 0.0
    for b in range(2):
        for m in range(3):
            np.testing.assert_allclose(
                rows[b, m], float(loss(truth[b:b + 1, m:m + 1], logits[b:b + 1, m:m + 1])),
                rtol=1e-6, atol=0)


def test_frame_zero_supervises_its_lowest_loss_candidate_and_every_iou():
    model = trainer(num_frames=2, object_score=8.0)
    seen = _recording_decode(model)
    inputs = clip_inputs(model)
    out = model(inputs, training=False)

    frame0 = seen[0]
    candidates = keras.ops.convert_to_numpy(frame0["low_res_logits"])
    assert candidates.shape[1] == CANDIDATES
    assert keras.ops.convert_to_numpy(seen[1]["low_res_logits"]).shape[1] == 1

    supervised = keras.ops.convert_to_numpy(out[SAM2_LOW_RES_LOGITS])
    np.testing.assert_array_equal(
        supervised[:, 0:1], _hand_selection(inputs["gt_masks"][:, 0:1], candidates))

    iou = keras.ops.convert_to_numpy(out[SAM2_IOU_SUPERVISION])
    assert iou.shape == (inputs["image"].shape[0], 2 * CANDIDATES, 2)
    np.testing.assert_array_equal(
        iou[:, :CANDIDATES, 0], keras.ops.convert_to_numpy(frame0["iou_predictions"]))
    np.testing.assert_array_equal(
        iou[:, :CANDIDATES, 1],
        keras.ops.convert_to_numpy(
            achieved_mask_iou(candidates, inputs["gt_masks"][:, 0:1])))
    # Frame 1 decoded one mask; its pair fills its slots.
    frame1 = keras.ops.convert_to_numpy(seen[1]["iou_predictions"])
    for slot in range(CANDIDATES, 2 * CANDIDATES):
        np.testing.assert_array_equal(iou[:, slot, 0], frame1[:, 0])


@pytest.mark.parametrize("token", [1, 2, 3])
def test_every_multimask_token_trains_when_its_candidate_is_the_lowest_loss_one(token):
    """Each multimask token's hypernetwork MLP and embedding row receive a nonzero
    gradient once its candidate is the one frame 0 supervises; mask token 0 is
    trained by the single-mask frame after it.

    Seeded (D-252): the probe makes a candidate the lowest-loss one by using its
    own foreground as the ground truth, which needs that candidate to have a
    foreground and to beat the other two on it. At unseeded initial weights 4
    of 36 (seed, token) draws failed one of those two preconditions (seeds 0-11:
    seed 1 token 3, seed 5 token 2, seeds 6 and 8 token 1), so the test failed
    about one run in nine; seed 0 meets both for every token.
    """
    keras.utils.set_random_seed(0)
    model = trainer(num_frames=2, object_score=8.0)
    inputs = clip_inputs(model, batch=1)
    seen = _recording_decode(model)
    model(inputs, training=True)
    candidate = keras.ops.convert_to_numpy(seen[0]["low_res_logits"])[0, token - 1]
    # Ground truth = this candidate's own foreground (logit > 0): it is then the
    # lowest-loss candidate, which the assertion below checks rather than assumes.
    gt = inputs["gt_masks"].copy()
    gt[0, 0] = (candidate > 0.0).astype("float32")
    assert gt[0, 0].any(), "the candidate has no foreground; the probe is vacuous"
    ranks = keras.ops.convert_to_numpy(model.mask_loss.per_mask(
        gt[:, 0:1], keras.ops.convert_to_numpy(seen[0]["low_res_logits"])))
    assert int(np.argmin(ranks[0])) == token - 1, ranks
    inputs["gt_masks"] = gt

    decoder = model.sam2.mask_decoder
    watched = (list(decoder.output_hypernetworks_mlps[token].trainable_weights)
               + list(decoder.output_hypernetworks_mlps[0].trainable_weights)
               + [decoder.mask_tokens.embeddings])
    with tf.GradientTape() as tape:
        out = model(inputs, training=True)
        value = model.mask_loss(gt, out[SAM2_LOW_RES_LOGITS])
    grads = tape.gradient(value, watched)
    n = len(decoder.output_hypernetworks_mlps[token].trainable_weights)
    mlp_k = sum(float(np.abs(g).sum()) for g in grads[:n])
    mlp_0 = sum(float(np.abs(g).sum()) for g in grads[n:-1])
    rows = np.abs(keras.ops.convert_to_numpy(tf.convert_to_tensor(grads[-1]))).sum(axis=1)
    assert mlp_k > 0.0, f"hypernetwork_mlp_{token} got no gradient"
    assert mlp_0 > 0.0, "hypernetwork_mlp_0 got no gradient from frame 1"
    assert rows[token] > 0.0 and rows[0] > 0.0, rows


def test_a_box_or_a_second_point_keeps_frame_zero_on_token_zero():
    """Upstream's rule: one point only (a box counts as two)."""
    model = trainer(num_frames=1, object_score=8.0)
    seen = _recording_decode(model)
    inputs = clip_inputs(model)
    inputs["point_coords"] = np.concatenate([inputs["point_coords"]] * 2, axis=1)
    inputs["point_labels"] = np.ones((inputs["image"].shape[0], 2), "int32")
    out = model(inputs, training=False)
    assert keras.ops.convert_to_numpy(seen[-1]["low_res_logits"]).shape[1] == 1
    assert tuple(out[SAM2_IOU_SUPERVISION].shape)[1] == 1

    boxed = clip_inputs(model)
    boxed["boxes"] = np.array([[[2.0, 2.0, 20.0, 20.0]]] * boxed["image"].shape[0], "float32")
    model(boxed, training=False)
    assert keras.ops.convert_to_numpy(seen[-1]["low_res_logits"]).shape[1] == 1


def test_the_switch_off_restores_single_mask_training_and_round_trips():
    model = trainer(num_frames=1, object_score=8.0)
    off = trainer(num_frames=1, object_score=8.0, multimask_output_in_sam=False)
    assert off.get_config()["sam2"]["config"]["multimask_output_in_sam"] is False
    assert model.get_config()["sam2"]["config"]["multimask_output_in_sam"] is True
    assert type(off).from_config(off.get_config()).sam2.multimask_output_in_sam is False
    seen = _recording_decode(off)
    off(clip_inputs(model), training=False)
    assert keras.ops.convert_to_numpy(seen[0]["low_res_logits"]).shape[1] == 1


def test_an_absent_frame_zeroes_all_its_slots():
    """The D-052 gate holds on the K-slot pack: an occluded frame's K pairs are 0."""
    model = trainer(num_frames=3, object_score=8.0)
    inputs = clip_inputs(model, absent_frames=(1,))
    iou = keras.ops.convert_to_numpy(model(inputs, training=False)[SAM2_IOU_SUPERVISION])
    per_frame = iou.reshape(iou.shape[0], 3, CANDIDATES, 2)
    np.testing.assert_array_equal(per_frame[:, 1], 0.0)
    assert (per_frame[:, 0, :, 0] != 0.0).all() and (per_frame[:, 2, :, 0] != 0.0).all()
