"""The bars of the language tasks: an analytic unigram, a majority share and two strict float32 comparisons.

The bar of ``language_model_pretraining`` is cut from the ``val_loss`` of a CONSTANT model that predicts the unigram
distribution of the selection targets, and the bar of ``text_classification`` from the share of the majority class of the
selection split. Neither needs a model. A constant Keras model over a 50,261-wide vocabulary at batch 8 and 255 positions is
0.4 GB of logits a forward (3 GB at 100,277), where the unigram is a count of the targets: O(vocabulary) memory. The module
scope imports numpy only (the check against a compiled constant model is a test).

Interface contract
------------------
``unigram_val_loss(label_array, vocab_size)``
    :param label_array: the integer target ids of every selection window, any shape; ``vocab_size``: the model's vocabulary.
    :return: ``-sum p log p`` in nats, ``p`` the share of each id among the targets: the ``val_loss`` of a model whose logits
        are ``log p`` at every position under the stock sparse cross-entropy (an in-sample unigram, the strictest constant
        model), computed from a chunked count and never from a ``(rows, vocab_size)`` array.
    :raises ValueError: no target, a non-integer array, a ``vocab_size`` below 1, an id outside ``[0, vocab_size)``.

``majority_share(labels)``
    :return: the share of the most frequent class among ``labels`` (non-negative integer class ids), a float64 ratio whose
        float32 is what Keras' accuracy divides for a model that answers the majority class. :raises ValueError: no label, a
        negative or non-integer one.

``strictly_below(reached, threshold, n_batches)``
    :param reached: a ``val_loss`` (Keras' float32); ``threshold``: the unigram (float64) or the epoch-zero loss; ``n_batches``:
        the batches of the selection pass.
    :return: ``train.common.forecast_run.beats_bar``: whether ``float32(reached)`` is below ``float32(threshold)`` by MORE than
        ``tie_band(n_batches)`` float32 steps (16 up to 16 batches, 96 at 500, 208 at 2,500). A model that merely equals the
        unigram, which Keras scores up to 14 float32 steps either side of the analytic number (measured, 1,250 batches), does not
        meet the bar, and neither does a NaN.

``strictly_above(reached, threshold)``
    :return: whether ``float32(reached)`` is strictly above ``float32(threshold)``. No band: Keras' accuracy of a model that answers
        the majority class is ``float32(k) / float32(n)``, the float32 of the share to the last bit (every count below 2**24).
"""

import numpy as np

#: Targets counted per pass: a count never holds more than this many ids as one ``intp`` array (8 MB).
CHUNK_ELEMENTS = 1 << 20


def unigram_val_loss(label_array: np.ndarray, vocab_size: int) -> float:
    """The entropy of the target ids in nats; see the module contract."""
    # DECISION plan-2026-11-24T092348-7806a290/D-006
    # The unigram is COUNTED, not evaluated: do NOT compile a constant Keras model at the real vocabulary to read its loss (the
    # logits tensor is 0.4 GB at 50,261 ids and 3 GB at 100,277, a batch at a time), and do NOT count in a ``(rows, vocab)``
    # array. A test holds this number equal to the `val_loss` of a compiled constant model on an ascii vocabulary, which is what
    # proves the closed form is the loss Keras computes. The count is in-sample (the selection targets themselves), the
    # strictest constant model, as the BEiT bar's.
    ids = np.asarray(label_array)
    if ids.dtype.kind not in "iu":
        raise ValueError(f"the targets must be integer ids, got dtype {ids.dtype}")
    if vocab_size < 1:
        raise ValueError(f"vocab_size must be at least 1, got {vocab_size}")
    if ids.size == 0:
        raise ValueError("no target ids: there is no unigram to count")
    flat = ids.reshape(-1)
    counts = np.zeros(vocab_size, np.int64)
    for start in range(0, flat.size, CHUNK_ELEMENTS):
        part = flat[start:start + CHUNK_ELEMENTS]
        low, high = int(part.min()), int(part.max())
        if low < 0 or high >= vocab_size:
            raise ValueError(
                f"target id {low if low < 0 else high} is outside [0, {vocab_size}): the tokenizer's vocabulary is larger than "
                "the vocab_size given")
        counts += np.bincount(part, minlength=vocab_size)
    share = counts[counts > 0] / flat.size
    return float(-(share * np.log(share)).sum())


def majority_share(labels: np.ndarray) -> float:
    """The share of the most frequent class; see the module contract."""
    ids = np.asarray(labels)
    if ids.size == 0:
        raise ValueError("no labels: there is no majority class")
    if ids.dtype.kind not in "iu" or ids.min() < 0:
        raise ValueError(f"labels must be non-negative integer class ids, got dtype {ids.dtype}")
    return float(np.bincount(ids.reshape(-1)).max() / ids.size)


def strictly_below(reached: float, threshold: float, n_batches: int) -> bool:
    """Whether ``reached`` is below ``threshold`` by more than the tie band of ``n_batches`` batches; see the module contract."""
    # DECISION plan-2026-11-24T092348-7806a290/D-006
    # DECISION plan-2026-11-24T092348-7806a290/D-020
    # STRICTLY below, in float32, and by MORE than the band of the one loss bar the repo has (`forecast_run.beats_bar`, the depth
    # task's too). Do NOT write `<=` or a bare `<`: the threshold here is a float64 number counted in numpy, the loss a float32
    # mean Keras accumulates batch by batch, and a compiled constant model measured 7 steps below the analytic unigram in 9 of 24
    # cases (14 steps at 1,250 batches), so a bare compare calls a model that is exactly the unigram a success by rounding. Do
    # NOT narrow the band to "be strict": `scratch_step3/ulp_gap*.txt` is the measurement, and the band is 10 times the worst gap.
    from train.common.forecast_run import beats_bar

    return beats_bar(reached, threshold, n_batches)


def strictly_above(reached: float, threshold: float) -> bool:
    """Whether ``reached`` is strictly above ``threshold`` in float32 (an accuracy bar); see the module contract."""
    # DECISION plan-2026-11-24T092348-7806a290/D-006
    # No band, unlike the loss: the share is a float64 ratio of two counts and Keras' accuracy the float32 ratio of the same
    # counts, equal to the last bit for every count below 2**24 (a test sweeps them), so a model that answers the majority class
    # scores exactly the bar and `>` calls it a miss. Do NOT write `>=` (that calls it a success).
    return bool(np.float32(reached) > np.float32(threshold))
