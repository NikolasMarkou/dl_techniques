"""Every VLM head that pools or attends over a sequence honours its padding mask.

Guard for plan-2026-10-14T062616-81938e81 step 11.2 (the user's heads review,
``research/reviews/20261015_heads_review.md`` item VLM-3). Measured at
``8b4182f9d``: no VLM head read a mask. The captioning cross-attention, the
VQA pooling, the grounding text mean and the image-text-matching means all ran
over padded positions; moving the padded text tokens of an
``ImageTextMatchingHead`` moved ``'match_score'`` by 0.83.

The masks ride in the input dict next to the features they describe, as the
keep predicate the attention layers use (1 = a real token, 0 = padding,
shape ``(batch, length)``): ``vision_mask`` for ``vision_features``,
``text_mask`` for ``text_features`` and ``question_mask`` for VQA's
``question_features``.

Each test fills the padded positions with large values, runs the head again,
and asserts that no output that belongs to a real position moved. A mask on a
rank-2 (already pooled) input is refused, because there is nothing to mask.
"""

import os
import tempfile
from typing import Dict

import keras
import numpy as np
import pytest

from dl_techniques.layers.heads.vlm import (
    VLMTaskConfig,
    VLMTaskType,
    create_multi_task_vlm_head,
    create_vlm_head,
)

# The prefix comparison runs one row against a batch of three; TF32 matmuls
# round those differently (measured 1.9e-4 on 'match_score'), so the module
# compares in full float32.
pytestmark = pytest.mark.usefixtures("tf32_disabled")

DIM = 16
B, N_VIS, S = 3, 6, 5
#: Per row: how many leading positions are real.
VIS_KEEP = (6, 4, 2)
TXT_KEEP = (5, 3, 1)


def _mask(keep, length) -> np.ndarray:
    m = np.zeros((len(keep), length), "float32")
    for row, k in enumerate(keep):
        m[row, :k] = 1.0
    return m


def _features(seed, length) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=(B, length, DIM)).astype("float32")


def _flood(x: np.ndarray, mask: np.ndarray, seed: int) -> np.ndarray:
    """Overwrite every padded position with large random values."""
    junk = 50.0 * np.random.default_rng(seed).normal(size=x.shape).astype("float32")
    return np.where(mask[..., None] > 0, x, junk)


def _head(task: VLMTaskType, **kwargs):
    extra = {
        VLMTaskType.IMAGE_CAPTIONING: {"vocab_size": 11},
        VLMTaskType.VISUAL_QUESTION_ANSWERING: {"num_classes": 7},
    }.get(task, {})
    config = VLMTaskConfig(name="m", task_type=task, hidden_size=DIM,
                           dropout_rate=0.0, **extra)
    if task is VLMTaskType.IMAGE_CAPTIONING:
        kwargs = {"num_layers": 2, "num_heads": 4, **kwargs}
    return create_vlm_head(config, **kwargs)


def _np(out) -> Dict[str, np.ndarray]:
    return {k: keras.ops.convert_to_numpy(v) for k, v in out.items()}


def _assert_close(a, b, what) -> None:
    np.testing.assert_allclose(a, b, rtol=0, atol=1e-5, err_msg=what)


# ---------------------------------------------------------------------
# Image-text matching: both means
# ---------------------------------------------------------------------

def test_matching_ignores_padded_vision_and_text_positions() -> None:
    head = _head(VLMTaskType.IMAGE_TEXT_MATCHING)
    vm, tm = _mask(VIS_KEEP, N_VIS), _mask(TXT_KEEP, S)
    v, t = _features(1, N_VIS), _features(2, S)
    clean = _np(head({"vision_features": v, "text_features": t,
                      "vision_mask": vm, "text_mask": tm}))
    flooded = _np(head({"vision_features": _flood(v, vm, 3),
                        "text_features": _flood(t, tm, 4),
                        "vision_mask": vm, "text_mask": tm}))
    for key in clean:
        _assert_close(clean[key], flooded[key], f"matching '{key}' moved")


def test_a_matching_mean_is_the_mean_of_the_real_positions() -> None:
    head = _head(VLMTaskType.IMAGE_TEXT_MATCHING)
    vm, tm = _mask(VIS_KEEP, N_VIS), _mask(TXT_KEEP, S)
    v, t = _features(1, N_VIS), _features(2, S)
    masked = _np(head({"vision_features": v, "text_features": t,
                       "vision_mask": vm, "text_mask": tm}))
    # Row by row, the same head on the unpadded prefix alone.
    for row in range(B):
        alone = _np(head({"vision_features": v[row:row + 1, :VIS_KEEP[row]],
                          "text_features": t[row:row + 1, :TXT_KEEP[row]]}))
        _assert_close(masked["match_score"][row], alone["match_score"][0],
                      f"row {row}: masked mean differs from the prefix mean")
        _assert_close(masked["vision_embeddings"][row], alone["vision_embeddings"][0],
                      f"row {row}: vision embedding")


# ---------------------------------------------------------------------
# Visual grounding: the text mean, and the region pick
# ---------------------------------------------------------------------

def test_grounding_ignores_padded_text_and_never_picks_a_padded_region() -> None:
    head = _head(VLMTaskType.VISUAL_GROUNDING)
    vm, tm = _mask(VIS_KEEP, N_VIS), _mask(TXT_KEEP, S)
    v, t = _features(1, N_VIS), _features(2, S)
    clean = _np(head({"vision_features": v, "text_features": t,
                      "vision_mask": vm, "text_mask": tm}))
    flooded = _np(head({"vision_features": _flood(v, vm, 3),
                        "text_features": _flood(t, tm, 4),
                        "vision_mask": vm, "text_mask": tm}))
    _assert_close(clean["bbox"], flooded["bbox"], "grounding 'bbox' moved")
    _assert_close(clean["grounded_features"], flooded["grounded_features"],
                  "grounding 'grounded_features' moved")
    # 'confidence' keeps one logit per region; only the real regions' are owed.
    real = vm > 0
    _assert_close(clean["confidence"][real], flooded["confidence"][real],
                  "grounding 'confidence' of a real region moved")


@pytest.mark.parametrize("strategy", ["concatenation", "addition", "multiplication",
                                      "gated", "bilinear", "tensor_fusion"])
def test_the_grounding_pick_is_a_real_region_for_every_fusion(strategy) -> None:
    head = _head(VLMTaskType.VISUAL_GROUNDING, fusion_strategy=strategy)
    vm = _mask(VIS_KEEP, N_VIS)
    v, t = _features(1, N_VIS), _features(2, S)
    payload = {"vision_features": v, "text_features": t, "vision_mask": vm}
    clean = _np(head(payload))
    flooded = _np(head({**payload, "vision_features": _flood(v, vm, 9)}))
    _assert_close(clean["grounded_features"], flooded["grounded_features"],
                  f"{strategy}: the picked region changed when padding moved")
    real = vm > 0
    _assert_close(clean["confidence"][real], flooded["confidence"][real],
                  f"{strategy}: a real region's logit depends on padding")


# ---------------------------------------------------------------------
# VQA: all three pooling strategies
# ---------------------------------------------------------------------

@pytest.mark.parametrize("strategy", ["mean", "max", "attention"])
def test_vqa_ignores_padded_vision_and_question_positions(strategy) -> None:
    head = _head(VLMTaskType.VISUAL_QUESTION_ANSWERING, pooling_strategy=strategy)
    vm, qm = _mask(VIS_KEEP, N_VIS), _mask(TXT_KEEP, S)
    v, q = _features(1, N_VIS), _features(2, S)
    clean = _np(head({"vision_features": v, "question_features": q,
                      "vision_mask": vm, "question_mask": qm}))
    flooded = _np(head({"vision_features": _flood(v, vm, 3),
                        "question_features": _flood(q, qm, 4),
                        "vision_mask": vm, "question_mask": qm}))
    _assert_close(clean["answer_logits"], flooded["answer_logits"],
                  f"vqa ({strategy}) 'answer_logits' moved")


# ---------------------------------------------------------------------
# Captioning: cross-attention onto vision, self-attention over left padding
# ---------------------------------------------------------------------

def test_captioning_cross_attention_ignores_padded_vision_tokens() -> None:
    head = _head(VLMTaskType.IMAGE_CAPTIONING)
    vm = _mask(VIS_KEEP, N_VIS)
    v, t = _features(1, N_VIS), _features(2, S)
    clean = _np(head({"vision_features": v, "text_features": t, "vision_mask": vm}))
    flooded = _np(head({"vision_features": _flood(v, vm, 3), "text_features": t,
                        "vision_mask": vm}))
    for key in clean:
        _assert_close(clean[key], flooded[key], f"captioning '{key}' moved")


def test_captioning_self_attention_ignores_left_padded_text() -> None:
    head = _head(VLMTaskType.IMAGE_CAPTIONING)
    # Left padding: the real tokens sit at the END of each row, where the
    # causal mask alone lets them attend to every padded position before them.
    tm = _mask(TXT_KEEP, S)[:, ::-1].copy()
    v, t = _features(1, N_VIS), _features(2, S)
    clean = _np(head({"vision_features": v, "text_features": t, "text_mask": tm}))
    flooded = _np(head({"vision_features": v, "text_features": _flood(t, tm, 4),
                        "text_mask": tm}))
    real = tm > 0
    for key in clean:
        _assert_close(clean[key][real], flooded[key][real],
                      f"captioning '{key}' at a real token moved")


# ---------------------------------------------------------------------
# All heads: an all-keep mask is no mask; a mask on a pooled input is refused
# ---------------------------------------------------------------------

_TEXT_KEY = {
    VLMTaskType.VISUAL_QUESTION_ANSWERING: ("question_features", "question_mask"),
}


@pytest.mark.parametrize("task", [
    VLMTaskType.IMAGE_CAPTIONING, VLMTaskType.VISUAL_QUESTION_ANSWERING,
    VLMTaskType.VISUAL_GROUNDING, VLMTaskType.IMAGE_TEXT_MATCHING,
], ids=lambda t: t.value)
def test_an_all_keep_mask_matches_no_mask(task) -> None:
    head = _head(task)
    feat_key, mask_key = _TEXT_KEY.get(task, ("text_features", "text_mask"))
    payload = {"vision_features": _features(1, N_VIS), feat_key: _features(2, S)}
    bare = _np(head(payload))
    masked = _np(head({**payload, "vision_mask": np.ones((B, N_VIS), "float32"),
                       mask_key: np.ones((B, S), "bool")}))
    for key in bare:
        np.testing.assert_allclose(bare[key], masked[key], rtol=1e-5, atol=1e-5,
                                   err_msg=f"{task.value} '{key}'")


@pytest.mark.parametrize("task,key", [
    (VLMTaskType.IMAGE_TEXT_MATCHING, "vision"),
    (VLMTaskType.IMAGE_TEXT_MATCHING, "text"),
    (VLMTaskType.VISUAL_GROUNDING, "text"),
])
def test_a_mask_on_a_pooled_input_is_refused(task, key) -> None:
    head = _head(task)
    payload = {"vision_features": _features(1, N_VIS), "text_features": _features(2, S)}
    payload[f"{key}_features"] = payload[f"{key}_features"][:, 0, :]
    payload[f"{key}_mask"] = np.ones((B, 1), "float32")
    with pytest.raises(ValueError, match=rf"{key}_mask.*rank"):
        head(payload)


# ---------------------------------------------------------------------
# Masks reach every head of a multi-task wrapper, and survive a graph
# ---------------------------------------------------------------------

def test_the_multi_task_wrapper_hands_the_masks_to_every_head() -> None:
    configs = [
        VLMTaskConfig(name="itm", task_type=VLMTaskType.IMAGE_TEXT_MATCHING,
                      hidden_size=DIM, dropout_rate=0.0),
        VLMTaskConfig(name="grd", task_type=VLMTaskType.VISUAL_GROUNDING,
                      hidden_size=DIM, dropout_rate=0.0),
    ]
    head = create_multi_task_vlm_head(configs)
    vm, tm = _mask(VIS_KEEP, N_VIS), _mask(TXT_KEEP, S)
    v, t = _features(1, N_VIS), _features(2, S)
    clean = head({"vision_features": v, "text_features": t,
                  "vision_mask": vm, "text_mask": tm})
    flooded = head({"vision_features": _flood(v, vm, 3),
                    "text_features": _flood(t, tm, 4),
                    "vision_mask": vm, "text_mask": tm})
    _assert_close(keras.ops.convert_to_numpy(clean["itm"]["match_score"]),
                  keras.ops.convert_to_numpy(flooded["itm"]["match_score"]),
                  "multi-task itm moved")
    _assert_close(keras.ops.convert_to_numpy(clean["grd"]["bbox"]),
                  keras.ops.convert_to_numpy(flooded["grd"]["bbox"]),
                  "multi-task grounding moved")


def test_a_masked_functional_model_round_trips_by_value() -> None:
    head = _head(VLMTaskType.IMAGE_TEXT_MATCHING)
    inputs = {
        "vision_features": keras.Input(shape=(N_VIS, DIM), name="vision_features"),
        "text_features": keras.Input(shape=(S, DIM), name="text_features"),
        "vision_mask": keras.Input(shape=(N_VIS,), name="vision_mask"),
        "text_mask": keras.Input(shape=(S,), name="text_mask"),
    }
    model = keras.Model(inputs, head(inputs))
    xs = {"vision_features": _features(1, N_VIS), "text_features": _features(2, S),
          "vision_mask": _mask(VIS_KEEP, N_VIS), "text_mask": _mask(TXT_KEEP, S)}
    before = model.predict(xs, verbose=0)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "masked.keras")
        model.save(path)
        restored = keras.models.load_model(path)
    after = restored.predict(xs, verbose=0)
    for key in before:
        np.testing.assert_allclose(before[key], after[key], rtol=0, atol=1e-6)
    flooded = model.predict({**xs, "text_features": _flood(xs["text_features"],
                                                            xs["text_mask"], 4)},
                            verbose=0)
    _assert_close(before["match_score"], flooded["match_score"],
                  "the graph path ignores the mask")


# ---------------------------------------------------------------------
# The poolers that carry the masks change no fixed-seed initialization
# ---------------------------------------------------------------------

def test_a_mask_pooler_draws_no_random_seed_and_owns_no_weight() -> None:
    """``_weightless_pooling`` builds a ``SequencePooling`` that takes no seed.

    ``SequencePooling.__init__`` resolves a ``'glorot_uniform'`` default, and a
    seedless Keras initializer takes a seed from the global Python RNG when it
    is created. Measured (D-055): that one draw per pooler moved the fixed-seed
    initial weights of every VQA, grounding and matching head. The census in
    the step scratchpad compared every head's weights and outputs at a fixed
    seed against ``8b4182f9d``: bit-identical with ``kernel_initializer='zeros'``.
    Since D-087 the layer itself skips the draw for a weightless strategy and
    the helper passes no initializer; the census is bit-identical again.
    """
    import random

    from dl_techniques.layers.heads.vlm.factory import _weightless_pooling

    keras.utils.set_random_seed(3)
    expected = random.random()
    keras.utils.set_random_seed(3)
    pooler = _weightless_pooling("mean", "probe_pooling")
    assert random.random() == expected, "building a mask pooler drew a random seed"
    pooler.build((None, 4, DIM))
    assert pooler.weights == []
