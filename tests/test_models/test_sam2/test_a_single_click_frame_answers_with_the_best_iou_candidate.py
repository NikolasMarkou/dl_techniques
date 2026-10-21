"""A single-click frame answers with its best predicted-IoU candidate (D-252).

Upstream ``SAM2Base._forward_sam_heads`` (facebookresearch/sam2 2b90b9f5):
when ``_use_multimask(is_init_cond_frame, point_inputs)`` holds, the decoder
returns its three candidates, and the frame's mask (``low_res_masks``, what
``track_step`` returns as ``pred_masks`` and hands to the memory encoder) is
the candidate of highest predicted IoU; the object pointer comes from that
candidate's output token. ``_use_multimask`` is ``multimask_output_in_sam and
(is_init_cond_frame or multimask_output_for_tracking) and min_pt <= num_pts
<= max_pt``, a box counting as two points. This port trains that rule with
the SAM2Base defaults for the last three (1, 1, tracking off; D-249), so
inference applies the same rule: a conditioning frame (``call``, or
``stream_step(is_conditioning=True)``) with one point and no box decodes the
three candidates and answers with the best one, shape ``(B, 1, h, w)``
unchanged. ``multimask_output=True`` still returns all three.

Every check is by value against a hand argmax over the IoU head's own
predictions for the same weights and inputs. The IoU head's last bias is
driven so that the winner sits at slice position 0 and then 2, so a selection
by position cannot pass.
"""

from typing import Any, Dict, List

import keras
import numpy as np
import pytest
from unittest import mock

from dl_techniques.models.vision_language.sam.sam2.memory_encoder import SAM2MemoryEncoder
from dl_techniques.models.vision_language.sam.sam2.model import SAM2
from dl_techniques.models.vision_language.sam.sam2.training_model import SAM2TrainingModel

from .test_model import images, pin_object_score, tiny_model

BATCH = 3


def _click(batch: int = BATCH, points: int = 1, seed: int = 0) -> Any:
    """``points`` positive clicks per row, inside the 64 px image."""
    rng = np.random.default_rng(seed)
    coords = rng.uniform(4.0, 60.0, (batch, points, 2)).astype("float32")
    return coords, np.ones((batch, points), "int32")


def _force_winner(model: SAM2, slice_position: int) -> None:
    """Make multimask slice ``slice_position`` (token ``p + 1``) win the IoU argmax."""
    head = model.mask_decoder.iou_prediction_head.layers[-1]
    bias = np.zeros(head.bias.shape, dtype="float32")
    bias[slice_position + 1] = 100.0
    head.bias.assign(bias)


def _hand_selection(candidates: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Pick each row's candidate by a numpy argmax over its predicted IoU."""
    iou = keras.ops.convert_to_numpy(candidates["iou_predictions"])
    logits = keras.ops.convert_to_numpy(candidates["low_res_logits"])
    best = np.argmax(iou, axis=1)
    rows = np.arange(iou.shape[0])
    return {
        "best": best,
        "low_res_logits": logits[rows, best][:, None],
        "iou_predictions": iou[rows, best][:, None],
    }


def _hand_pointer(model: SAM2, image: np.ndarray, points: Any, best: np.ndarray) -> np.ndarray:
    """The object pointer built from the hand-selected candidate's output token."""
    encoded = model.image_encoder(image, training=False)
    sparse, dense = model.prompt_encoder(points=points, boxes=None, masks=None, training=False)
    fpn = encoded["backbone_fpn"]
    _, _, score, tokens = model.mask_decoder(
        image_embeddings=encoded["vision_features"],
        image_pe=model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse,
        dense_prompt_embeddings=dense,
        multimask_output=True,
        high_res_features=[fpn[0], fpn[1]],
        training=False,
    )
    tokens = keras.ops.convert_to_numpy(tokens)
    assert tokens.shape[1] == 3, "the pointer must come from a multimask token"
    picked = tokens[np.arange(tokens.shape[0]), best]
    pointer = model.obj_ptr_proj(picked, training=False)
    return keras.ops.convert_to_numpy(model._blend_object_pointer(pointer, score))


@pytest.fixture(scope="module")
def model() -> SAM2:
    built = tiny_model()
    pin_object_score(built, 5.0)
    return built


@pytest.mark.parametrize("winner", [0, 2])
def test_call_answers_a_single_click_with_the_best_iou_candidate(model, winner):
    _force_winner(model, winner)
    image, points = images(BATCH, seed=1), _click()
    candidates = model({"image": image, "points": points}, multimask_output=True)
    hand = _hand_selection(candidates)
    assert (hand["best"] == winner).all()

    answer = model({"image": image, "points": points})
    assert tuple(answer["low_res_logits"].shape) == (BATCH, 1, 16, 16)
    np.testing.assert_array_equal(
        keras.ops.convert_to_numpy(answer["low_res_logits"]), hand["low_res_logits"])
    np.testing.assert_array_equal(
        keras.ops.convert_to_numpy(answer["iou_predictions"]), hand["iou_predictions"])
    np.testing.assert_allclose(
        keras.ops.convert_to_numpy(answer["object_pointer"]),
        _hand_pointer(model, image, points, hand["best"]), atol=1e-5, rtol=0)


def test_the_natural_ranking_is_followed_row_by_row():
    """No forced bias: each row takes its own argmax, and the rows disagree.

    Seeded: the rows' natural ranking is a property of the initial weights,
    and an unseeded build takes whatever the earlier tests left in the RNG.
    """
    keras.utils.set_random_seed(7)
    model = tiny_model()
    pin_object_score(model, 5.0)
    image, points = images(8, seed=3), _click(8, seed=3)
    hand = _hand_selection(model({"image": image, "points": points}, multimask_output=True))
    assert len(set(hand["best"].tolist())) > 1, f"rows agree: {hand['best']}"
    answer = model({"image": image, "points": points})
    np.testing.assert_array_equal(
        keras.ops.convert_to_numpy(answer["low_res_logits"]), hand["low_res_logits"])


def test_multimask_output_still_returns_every_candidate(model):
    out = model({"image": images(BATCH), "points": _click()}, multimask_output=True)
    assert tuple(out["low_res_logits"].shape)[1] == 3
    assert tuple(out["iou_predictions"].shape)[1] == 3


@pytest.mark.parametrize("prompt", ["none", "two_points", "box"])
def test_other_prompts_decode_mask_token_zero(model, prompt):
    """Upstream's point-count rule: 0 points, 2 points, or a box (= 2 points)."""
    inputs = {"image": images(BATCH)}
    if prompt == "two_points":
        inputs["points"] = _click(points=2)
    elif prompt == "box":
        inputs["boxes"] = np.array([[[2.0, 2.0, 40.0, 40.0]]] * BATCH, "float32")
    seen = _spy_decoder(model)
    out = model(inputs)
    assert seen == [False]
    assert tuple(out["low_res_logits"].shape)[1] == 1


def _spy_decoder(model: SAM2) -> List[bool]:
    """Record the ``multimask_output`` the decoder is called with."""
    seen: List[bool] = []
    original = type(model.mask_decoder).call

    def spy(self, *args, **kwargs):
        seen.append(bool(kwargs["multimask_output"]))
        return original(self, *args, **kwargs)

    patcher = mock.patch.object(type(model.mask_decoder), "call", spy)
    patcher.start()
    _PATCHERS.append(patcher)
    return seen


_PATCHERS: List[Any] = []


@pytest.fixture(autouse=True)
def _stop_patchers():
    yield
    while _PATCHERS:
        _PATCHERS.pop().stop()


def test_the_switch_off_keeps_token_zero_and_round_trips():
    on = tiny_model()
    assert on.multimask_output_in_sam is True
    off = tiny_model(multimask_output_in_sam=False)
    assert off.get_config()["multimask_output_in_sam"] is False
    assert SAM2.from_config(off.get_config()).multimask_output_in_sam is False
    seen = _spy_decoder(off)
    off({"image": images(BATCH), "points": _click()})
    assert seen == [False]


class TestStreaming:
    """``stream_step``: the selected candidate is what memory and pointer see."""

    @pytest.mark.parametrize("winner", [0, 2])
    def test_the_memory_encoder_and_pointer_take_the_selected_candidate(self, winner):
        model = tiny_model()
        pin_object_score(model, 5.0)
        _force_winner(model, winner)
        image, points = images(BATCH, seed=5), _click(seed=5)
        hand = _hand_selection(model({"image": image, "points": points}, multimask_output=True))

        masks_seen: List[np.ndarray] = []
        original = SAM2MemoryEncoder.call

        def spy(self, inputs, *args, **kwargs):
            masks_seen.append(keras.ops.convert_to_numpy(inputs[1]))
            return original(self, inputs, *args, **kwargs)

        model.stream_reset()
        with mock.patch.object(SAM2MemoryEncoder, "call", spy):
            out = model.stream_step(image, frame_idx=0, points=points, is_conditioning=True)

        np.testing.assert_array_equal(
            keras.ops.convert_to_numpy(out["low_res_logits"]), hand["low_res_logits"])
        expected_memory_mask = keras.ops.convert_to_numpy(keras.ops.image.resize(
            np.transpose(hand["low_res_logits"], (0, 2, 3, 1)),
            (model.image_size, model.image_size), interpolation="bilinear"))
        assert len(masks_seen) == 1
        np.testing.assert_allclose(masks_seen[0], expected_memory_mask, atol=1e-5, rtol=0)

        hand_pointer = _hand_pointer(model, image, points, hand["best"])
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(out["object_pointer"]), hand_pointer, atol=1e-5, rtol=0)
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(model.memory_bank.cond_frames[0].obj_ptr),
            hand_pointer, atol=1e-5, rtol=0)

    def test_tracking_and_non_conditioning_clicks_decode_token_zero(self):
        """Upstream: multimask only on an initial conditioning frame (tracking off)."""
        model = tiny_model()
        pin_object_score(model, 5.0)
        seen = _spy_decoder(model)
        model.stream_reset()
        model.stream_step(images(BATCH, seed=0), frame_idx=0, points=_click(), is_conditioning=True)
        model.stream_step(images(BATCH, seed=1), frame_idx=1)
        model.stream_step(images(BATCH, seed=2), frame_idx=2, points=_click(seed=2))
        assert seen == [True, False, False]


def test_the_training_wrapper_reads_the_same_switch():
    """One home for the rule: the wrapper's frame-0 choice is SAM2's."""
    from .test_training_model import clip_inputs

    for flag, expected in ((True, [True, False]), (False, [False, False])):
        sam2 = tiny_model(multimask_output_in_sam=flag)
        pin_object_score(sam2, 5.0)
        wrapper = SAM2TrainingModel(sam2, num_frames=2)
        wrapper.build(None)
        seen = _spy_decoder(sam2)
        wrapper(clip_inputs(wrapper), training=False)
        assert seen == expected
        _PATCHERS.pop().stop()
