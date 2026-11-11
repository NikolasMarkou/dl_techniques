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


def _weighted_data():
    """Four rows with UNEQUAL valid-token counts (6, 6, 6 and 1)."""
    rng = np.random.default_rng(3)
    y_pred = rng.normal(size=(4, VISION + TEXT, VOCAB)).astype("float32")
    y_true = rng.integers(1, VOCAB, (4, TEXT)).astype("int32")
    y_true[3, 2:] = 0  # row 3 keeps one valid target, the others keep six
    return y_true, y_pred


def _reference(y_true, y_pred, ignore_index=0, label_smoothing=0.0):
    """Independent float64 NumPy derivation: ``(row_token_sums, valid_counts)``.

    Per-token next-token cross-entropy over the trailing ``seq_len`` positions
    (optionally smoothed), summed over each row's valid targets. Written from
    the definition, not from the class, so it cannot share a defect with it.
    """
    text = y_pred[:, -y_true.shape[1]:].astype("float64")
    logp = text - np.log(np.sum(np.exp(text), axis=-1, keepdims=True))
    targets = y_true[:, 1:]
    safe = np.maximum(targets, 0)
    nll = -np.take_along_axis(logp[:, :-1], safe[..., None], axis=-1)[..., 0]
    if label_smoothing:
        smooth = -logp[:, :-1].mean(axis=-1)
        nll = (1.0 - label_smoothing) * nll + label_smoothing * smooth
    keep = (targets != ignore_index).astype("float64")
    return (nll * keep).sum(axis=-1), keep.sum(axis=-1)


# DECISION plan-2026-11-11T131349-13b28b37/D-046
# The expected value below is derived from the formula, never from the loss's
# own `reduction="none"` output (the previous version of this test did that and
# could not fail). It also fixes the DOCUMENTED coupling: row i is
# `S_i / N * B` with N the batch-wide valid-token count, so a zero weight does
# not remove the row from N.
@pytest.mark.parametrize("label_smoothing", [0.0, 0.1])
@pytest.mark.parametrize("weights", [
    [1.0, 1.0, 1.0, 1.0],
    [1.0, 1.0, 1.0, 0.0],
    [0.0, 1.0, 1.0, 1.0],
    [0.5, 2.0, 1.0, 0.25],
])
def test_the_weighted_loss_equals_an_independent_float64_reference(
        label_smoothing, weights):
    y_true, y_pred = _weighted_data()
    sums, counts = _reference(y_true, y_pred, label_smoothing=label_smoothing)
    w = np.asarray(weights, dtype="float64")
    rows = sums / counts.sum() * len(w)  # S_i / N * B, N over ALL rows

    def value(reduction):
        loss = NanoVLMLoss(label_smoothing=label_smoothing, reduction=reduction)
        return keras.ops.convert_to_numpy(
            loss(y_true, y_pred, sample_weight=np.asarray(weights, "float32")))

    # Keras: sum_over_batch_size = sum(w * rows) / B; sum = sum(w * rows).
    np.testing.assert_allclose(
        value("sum_over_batch_size"), (w * rows).sum() / len(w), rtol=2e-6, atol=0)
    np.testing.assert_allclose(
        value("sum"), (w * rows).sum(), rtol=2e-6, atol=0)
    np.testing.assert_allclose(value("none"), w * rows, rtol=2e-6, atol=0)


@pytest.mark.parametrize("dead", [0, 3])
def test_a_zero_sample_weight_is_not_the_same_as_dropping_the_row(dead):
    """The documented coupling, pinned with its exact ratio.

    Zeroing row ``d``'s weight removes its contribution but not its tokens from
    the batch-wide denominator ``N``, so
    ``loss(w) == loss(batch without d) * (N - n_d) / N`` exactly, and NOT the
    ``(B - 1) / B`` of a plain row drop (the two agree only when every row has
    the same valid-token count, which this data deliberately does not).
    """
    y_true, y_pred = _weighted_data()
    weights = np.ones(4, dtype="float32")
    weights[dead] = 0.0
    keep = [i for i in range(4) if i != dead]

    weighted = float(NanoVLMLoss()(y_true, y_pred, sample_weight=weights))
    dropped = float(NanoVLMLoss()(y_true[keep], y_pred[keep]))

    _, counts = _reference(y_true, y_pred)
    ratio = (counts.sum() - counts[dead]) / counts.sum()
    np.testing.assert_allclose(weighted, dropped * ratio, rtol=2e-6, atol=0)
    # ... and the naive "same as dropping" expectation is measurably wrong:
    assert abs(weighted * 4 / 3 - dropped) > 0.05 * dropped


def test_a_one_token_sequence_scores_zero_without_a_python_branch():
    y_true, y_pred = _data()
    rows = keras.ops.convert_to_numpy(
        NanoVLMLoss(reduction="none")(y_true[:, :1], y_pred[:, -1:]))
    assert rows.shape == (BATCH,)
    np.testing.assert_array_equal(rows, np.zeros(BATCH, dtype="float32"))
