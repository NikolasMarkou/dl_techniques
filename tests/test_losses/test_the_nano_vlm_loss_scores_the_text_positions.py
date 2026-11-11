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


# ---------------------------------------------------------------------
# plan-2026-11-11T131349-13b28b37 step 6 (D-011): the loss returns one value
# per ROW, has no Python branch on the sequence length, and clamps labels.
# ---------------------------------------------------------------------

def _ragged_dataset(vocab=VOCAB):
    import tensorflow as tf

    rng = np.random.default_rng(1)
    seqs = [rng.integers(1, vocab, int(n)).astype("int32")
            for n in (4, 6, 3, 7, 5, 6, 4, 7)]

    def gen():
        for s in seqs:
            yield s, s

    ds = tf.data.Dataset.from_generator(
        gen,
        output_signature=(tf.TensorSpec((None,), tf.int32),
                          tf.TensorSpec((None,), tf.int32)))
    # Unknown cardinality makes Keras log a phantom zero batch (loss n/(n+1)).
    return ds.padded_batch(2, padding_values=(0, 0)).apply(
        tf.data.experimental.assert_cardinality(len(seqs) // 2))


def test_a_ragged_padded_batch_fit_with_an_unknown_length_runs():
    """RED at HEAD: `if seq_len_true < 2` on a tensor raises
    `OperatorNotAllowedInGraphError` once the sequence axis is `None`."""
    inputs = keras.Input(shape=(None,), dtype="int32")
    x = keras.layers.Embedding(VOCAB, 8)(inputs)
    model = keras.Model(inputs, keras.layers.Dense(VOCAB)(x))
    model.compile(optimizer="sgd", loss=NanoVLMLoss(ignore_index=0))
    history = model.fit(_ragged_dataset(), epochs=1, verbose=0)
    assert np.isfinite(history.history["loss"][0])


def test_a_negative_ignore_index_is_finite():
    """RED at HEAD: the unclamped label -1 reaches `sparse_ce` and crashes."""
    y_true, y_pred = _data()
    y_true[:, -2:] = -1
    value = float(NanoVLMLoss(ignore_index=-1)(y_true, y_pred))
    assert np.isfinite(value)
    # Re-derived: ignoring -1 scores exactly the positions whose target >= 0.
    text = y_pred[:, VISION:]
    logp = text - np.log(np.sum(np.exp(text), axis=-1, keepdims=True))
    targets = y_true[:, 1:]
    safe = np.maximum(targets, 0)
    nll = -np.take_along_axis(logp[:, :-1], safe[..., None], axis=-1)[..., 0]
    keep = targets != -1
    np.testing.assert_allclose(value, nll[keep].sum() / keep.sum(),
                               rtol=1e-6, atol=0)


def test_reduction_none_returns_one_value_per_row():
    y_true, y_pred = _data()
    rows = keras.ops.convert_to_numpy(
        NanoVLMLoss(reduction="none")(y_true, y_pred))
    assert rows.shape == (BATCH,)
    # Per-row values average back to the batch-wide token mean EXACTLY.
    np.testing.assert_allclose(
        rows.mean(), float(NanoVLMLoss()(y_true, y_pred)), rtol=1e-6, atol=0)


def test_sample_weight_selects_rows_instead_of_scaling_the_aggregate():
    """RED at HEAD: the scalar broadcast made the weighted value exactly
    `unweighted * mean(sample_weight)`, whichever rows were weighted."""
    rng = np.random.default_rng(3)
    y_pred = rng.normal(size=(4, VISION + TEXT, VOCAB)).astype("float32")
    y_true = rng.integers(1, VOCAB, (4, TEXT)).astype("int32")
    y_true[3, 2:] = 0  # row 3 has a different valid-token count
    weights = np.array([1.0, 1.0, 1.0, 0.0], dtype="float32")
    loss = NanoVLMLoss()
    weighted = float(loss(y_true, y_pred, sample_weight=weights))
    unweighted = float(loss(y_true, y_pred))
    assert abs(weighted / unweighted - 0.75) > 1e-3
    # Re-derived: zero-weighting row 3 keeps the other rows' shares of the
    # batch-wide token mean and divides by the batch size.
    rows = keras.ops.convert_to_numpy(
        NanoVLMLoss(reduction="none")(y_true, y_pred))
    np.testing.assert_allclose(weighted, (rows * weights).sum() / 4,
                               rtol=1e-6, atol=0)


def test_a_one_token_sequence_scores_zero_without_a_python_branch():
    y_true, y_pred = _data()
    rows = keras.ops.convert_to_numpy(
        NanoVLMLoss(reduction="none")(y_true[:, :1], y_pred[:, -1:]))
    assert rows.shape == (BATCH,)
    np.testing.assert_array_equal(rows, np.zeros(BATCH, dtype="float32"))
