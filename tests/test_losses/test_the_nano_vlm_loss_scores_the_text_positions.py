"""`NanoVLMLoss` scores the trailing label-length positions of its logits.

NanoVLM's output is `[batch, vision_len + text_len, vocab]`: the projected
vision tokens come FIRST, then the text, through one causal decoder. The
loss used to refuse any length mismatch, so it could never be applied to its
own model's output (the old hand-written trainer caught that `ValueError` per
step and trained nothing). Since plan-2026-10-14T062616-81938e81 step 9 it
scores the last `text_len` positions and ignores the vision prefix.
"""

import keras
import numpy as np
import pytest

from dl_techniques.losses.nano_vlm_loss import NanoVLMLoss, text_position_logits

BATCH, VISION, TEXT, VOCAB = 2, 5, 7, 11


def _data():
    rng = np.random.default_rng(0)
    y_pred = rng.normal(size=(BATCH, VISION + TEXT, VOCAB)).astype("float32")
    y_true = rng.integers(1, VOCAB, (BATCH, TEXT)).astype("int32")
    y_true[0, -2:] = 0  # padding, ignored
    return y_true, y_pred


@pytest.mark.parametrize("label_smoothing", [0.0, 0.1])
def test_a_vision_prefix_is_dropped_and_the_text_suffix_scored(label_smoothing):
    y_true, y_pred = _data()
    loss = NanoVLMLoss(label_smoothing=label_smoothing)
    with_prefix = float(loss(y_true, y_pred))
    text_only = float(loss(y_true, y_pred[:, VISION:]))
    head = float(loss(y_true, y_pred[:, :TEXT]))
    assert with_prefix == text_only
    assert with_prefix != head


def test_the_value_is_next_token_cross_entropy_over_unpadded_targets():
    """Re-derived from the formula, not from the class."""
    y_true, y_pred = _data()
    text = y_pred[:, VISION:]
    logp = text - np.log(np.sum(np.exp(text), axis=-1, keepdims=True))
    targets = y_true[:, 1:]
    nll = -np.take_along_axis(logp[:, :-1], targets[..., None], axis=-1)[..., 0]
    mask = targets != 0
    expected = nll[mask].sum() / mask.sum()
    got = float(NanoVLMLoss()(y_true, y_pred))
    np.testing.assert_allclose(got, expected, rtol=1e-6, atol=0)


def test_fewer_logit_positions_than_labels_is_refused():
    y_true, y_pred = _data()
    with pytest.raises(ValueError, match="fewer than the"):
        NanoVLMLoss()(y_true, y_pred[:, :TEXT - 1])


def test_equal_lengths_are_returned_unchanged():
    y_true, y_pred = _data()
    text = y_pred[:, VISION:]
    assert text_position_logits(y_true, text) is text
    np.testing.assert_array_equal(
        keras.ops.convert_to_numpy(text_position_logits(y_true, y_pred)), text)
