"""The bars of the language tasks: an analytic unigram, a majority share and two strict float32 comparisons.

``train.common.text_run`` cuts the success bar of ``language_model_pretraining`` (strictly below ``min(unigram, epoch zero)``)
and of ``text_classification`` (strictly above the majority-class share of the selection split) without a model and without a
vocabulary-wide logits tensor: a constant model over a 50,261-wide vocabulary at batch 8 is about 0.4 GB of logits a forward
(``findings/task-contract.md``, item 4). What is pinned here:

* ``unigram_val_loss`` is the entropy of the target ids, equal to an independent python oracle, equal to the ``val_loss`` of a
  COMPILED constant Keras model (a bias-only head over an ascii vocabulary under the stock loss, ragged last batch) within 1e-5,
  equal across chunk sizes, and computed in a few megabytes at a 100,277-wide vocabulary (no ``(rows, vocab)`` array);
* ``majority_share`` of the synthetic 60/40 bank is 0.6, and the float32 of the float64 share is what Keras divides;
* ``strictly_above`` calls equality and a NaN a miss in float32, and a constant classifier scores the majority share to the last
  bit, so it misses the bar that its own accuracy clears when the bar is the chance level;
* ``strictly_below`` is the loss bar of the repo (``forecast_run.beats_bar``, its tie band widening with the batches): a COMPILED
  constant unigram model lands a few float32 steps either side of the analytic number, below it in most of the cases measured
  here, so a bare float32 compare would call a model that is exactly the unigram a success (the control counts those cases).
"""

import math
import tracemalloc
from collections import Counter

import keras
import numpy as np
import pytest

from dl_techniques.utils.logger import logger
from train.api import text_tokenizer as tt
from train.common import text_data as td
from train.common import text_run as tr


def oracle_entropy(ids) -> float:
    """-sum p log p by a python loop over a Counter: nothing in common with the numpy implementation."""
    counts = Counter(int(i) for i in np.asarray(ids).reshape(-1))
    total = sum(counts.values())
    return -sum((c / total) * math.log(c / total) for c in counts.values())


def language_windows(seq_len=16, docs=200):
    """Select windows of the synthetic bank at the ascii vocabulary: ``(x, y, vocab_size)``."""
    opened = td.open_tokenizer(tt.tokenizer_spec("ascii95"))
    texts, _ = td.synthetic_documents(2, docs, td.LM)
    x, y = td.lm_fixed_windows(td.tokenize_documents(texts, opened), seq_len, opened.spec.eot_id)
    return x, y, opened.spec.vocab_size


def constant_language_model(log_p: np.ndarray, seq_len: int) -> keras.Model:
    """A model whose logits are ``log_p`` at every position of every row: a zero kernel over ones, the bias is the constant."""
    inputs = keras.Input((seq_len,), dtype="float32")
    ones = keras.layers.Lambda(lambda t: keras.ops.expand_dims(keras.ops.ones_like(t), -1))(inputs)
    head = keras.layers.Dense(len(log_p))
    outputs = head(ones)
    head.set_weights([np.zeros((1, len(log_p)), np.float32), log_p.astype(np.float32)])
    model = keras.Model(inputs, outputs)
    model.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True))
    return model


# ---------------------------------------------------------------------
# the unigram
# ---------------------------------------------------------------------


def test_the_unigram_is_the_entropy_of_the_targets():
    rng = np.random.RandomState(0)
    for vocab, size in ((101, 5000), (7, 300), (50261, 40000)):
        weights = rng.dirichlet(np.ones(min(vocab, 40)) * 0.3)
        ids = np.zeros(size, np.int32)
        ids[:] = rng.choice(len(weights), size=size, p=weights)
        got = tr.unigram_val_loss(ids.reshape(-1, 4) if size % 4 == 0 else ids, vocab)
        assert got == pytest.approx(oracle_entropy(ids), abs=1e-12)
    assert tr.unigram_val_loss(np.full((3, 5), 9, np.int32), 20) == 0.0, "one id only: nothing to predict"
    assert tr.unigram_val_loss(np.array([[0, 1], [0, 1]]), 2) == pytest.approx(math.log(2))


def test_the_analytic_unigram_equals_the_val_loss_of_a_compiled_constant_model():
    x, y, vocab = language_windows()
    analytic = tr.unigram_val_loss(y, vocab)
    share = np.bincount(y.reshape(-1), minlength=vocab) / y.size
    model = constant_language_model(np.log(np.maximum(share, 1e-30)), x.shape[1])
    loaded = td.select_dataset((x.astype(np.float32), y), 13)
    assert x.shape[0] % 13 != 0, "the last batch is ragged, as a real selection pass"
    compiled = float(model.evaluate(loaded, verbose=0, return_dict=True)["loss"])
    difference = abs(analytic - compiled)
    logger.info(f"UNIGRAM-CHECK analytic {analytic:.9f} compiled {compiled:.9f} max_abs_diff {difference:.3e} "
                f"windows {x.shape[0]} vocab {vocab}")
    assert difference <= 1e-5, (analytic, compiled)
    # the check can fail: the uniform model and a model of another base are far from the number
    uniform = float(constant_language_model(np.zeros(vocab), x.shape[1]).evaluate(loaded, verbose=0, return_dict=True)["loss"])
    assert uniform == pytest.approx(math.log(vocab), abs=1e-5) and abs(uniform - analytic) > 0.1
    assert abs(analytic / math.log(2) - compiled) > 0.1, "an entropy in bits is not the compiled loss"


def test_a_chunked_count_gives_the_number_of_one_pass(monkeypatch):
    ids = np.random.RandomState(1).randint(0, 50, size=(97, 31)).astype(np.int32)
    whole = tr.unigram_val_loss(ids, 50)
    for chunk in (1, 7, 1000, 10 ** 6):
        monkeypatch.setattr(tr, "CHUNK_ELEMENTS", chunk)
        assert tr.unigram_val_loss(ids, 50) == pytest.approx(whole, abs=1e-12), chunk


def test_a_unigram_over_a_huge_vocabulary_needs_no_logits_tensor():
    ids = np.random.RandomState(2).randint(0, 100277, size=(4000, 256)).astype(np.int32)   # 1 M targets, vocab of cl100k
    tracemalloc.start()
    try:
        got = tr.unigram_val_loss(ids, 100277)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert got == pytest.approx(oracle_entropy(ids), abs=1e-9)
    assert peak < 60e6, f"peak {peak / 1e6:.1f} MB: a (rows, vocab) array would be {ids.size * 100277 * 4 / 1e12:.1f} TB"


def test_a_target_outside_the_vocabulary_or_no_target_is_refused_by_name():
    with pytest.raises(ValueError, match=r"id 5.*\[0, 5\)"):
        tr.unigram_val_loss(np.array([[0, 5]]), 5)
    with pytest.raises(ValueError, match=r"id -1"):
        tr.unigram_val_loss(np.array([[0, -1]]), 5)
    with pytest.raises(ValueError, match="no target"):
        tr.unigram_val_loss(np.zeros((0, 4), np.int32), 5)
    with pytest.raises(ValueError, match="integer"):
        tr.unigram_val_loss(np.array([[0.0, 1.0]]), 5)
    with pytest.raises(ValueError, match="vocab_size"):
        tr.unigram_val_loss(np.array([[0]]), 0)


# ---------------------------------------------------------------------
# the majority share and the strict comparisons
# ---------------------------------------------------------------------


def test_the_majority_share_of_the_60_40_bank_is_0_6():
    _, labels = td.synthetic_documents(4, 100, td.CLASSIFICATION)
    assert tr.majority_share(labels) == 0.6
    assert tr.majority_share(np.array([2, 2, 1, 0, 2])) == 0.6
    assert tr.majority_share(np.array([1, 1])) == 1.0
    assert tr.majority_share(np.array([0, 1])) == 0.5
    assert tr.majority_share(np.array([0, 1, 1, 0, 2, 2])) == pytest.approx(1 / 3)
    with pytest.raises(ValueError, match="no label"):
        tr.majority_share(np.zeros(0, np.int64))
    with pytest.raises(ValueError, match="non-negative integer"):
        tr.majority_share(np.array([0, -1]))
    with pytest.raises(ValueError, match="non-negative integer"):
        tr.majority_share(np.array([0.5, 1.0]))


def test_the_float32_of_the_share_is_what_keras_divides():
    # Keras' accuracy is a float32 sum over a float32 count; the share is a float64 ratio. They agree to the last bit for
    # every count (no double-rounding case), so a model that scores the majority share exactly is not "above" it.
    for n in range(1, 500):
        counts = np.arange(0, n + 1)
        assert (np.float32(counts) / np.float32(n) == (counts / n).astype(np.float32)).all(), n


def test_a_loss_bar_needs_more_than_the_tie_band_and_a_nan_misses():
    from train.common.forecast_run import tie_band

    threshold = np.float32(2.5)
    step = np.spacing(threshold)
    for batches in (1, 16, 50, 183, 2500):
        band = tie_band(batches)
        edge = threshold - np.float32(band) * step      # exactly `band` steps below: a tie
        assert not tr.strictly_below(edge, threshold, batches)
        assert tr.strictly_below(edge - step, threshold, batches), "one step further: met"
        assert not tr.strictly_below(threshold, threshold, batches) and not tr.strictly_below(threshold + step, threshold, batches)
    assert tr.strictly_below(1.0, 2.0, 1) and not tr.strictly_below(3.0, 2.0, 1)
    low, high = 1.0, 1.0 + 1e-10     # two float64 numbers of one float32: equal, so neither is below the other
    assert not tr.strictly_below(low, high, 1) and not tr.strictly_below(high, low, 1)
    nan = float("nan")
    assert not tr.strictly_below(nan, 1.0, 1) and not tr.strictly_below(1.0, nan, 1)
    assert isinstance(tr.strictly_below(1.0, 2.0, 1), bool)


def test_an_accuracy_bar_calls_equality_and_a_nan_a_miss():
    above = tr.strictly_above
    assert above(2.0, 1.0) and not above(2.0, 2.0) and not above(1.0, 2.0)
    low, high = 1.0, 1.0 + 1e-10
    assert not above(high, low) and not above(low, high)
    assert above(np.float32(1.0) + np.spacing(np.float32(1.0)), np.float32(1.0)), "one float32 step above is above"
    nan = float("nan")
    assert not above(nan, 1.0) and not above(1.0, nan)
    assert isinstance(above(2, 1), bool)


def test_a_compiled_constant_unigram_model_never_meets_the_loss_bar_where_a_bare_compare_says_it_does():
    # The threshold is a float64 count; the loss is a float32 mean Keras accumulates over the batches. Measured on these cases
    # (and on 24 more in scratch_step3/ulp_gap.txt): the compiled number is a few float32 steps either side of the analytic one.
    rng = np.random.RandomState(0)
    vocab, seq_len, batch = 101, 64, 16
    bare, cases = [], []
    for n_windows in (200, 800):
        for trial in range(4):
            weights = rng.dirichlet(np.ones(vocab) * rng.choice([0.05, 0.3, 1.0]))
            y = rng.choice(vocab, size=(n_windows, seq_len), p=weights).astype(np.int32)
            analytic = tr.unigram_val_loss(y, vocab)
            share = np.bincount(y.reshape(-1), minlength=vocab) / y.size
            model = constant_language_model(np.log(np.maximum(share, 1e-30)), seq_len)
            loaded = td.select_dataset((np.zeros((n_windows, seq_len), np.float32), y), batch)
            compiled = float(model.evaluate(loaded, verbose=0, return_dict=True)["loss"])
            batches = int(loaded.cardinality())
            gap = (float(np.float32(compiled)) - float(np.float32(analytic))) / float(np.spacing(np.float32(analytic)))
            cases.append((n_windows, trial, gap))
            bare.append(bool(np.float32(compiled) < np.float32(analytic)))
            assert not tr.strictly_below(compiled, analytic, batches), (n_windows, trial, gap)
    logger.info(f"UNIGRAM-BAR gaps in float32 steps {[g for *_, g in cases]}; a bare compare met the bar in {sum(bare)} of {len(bare)}")
    assert sum(bare) >= 1, "the control: a bare float32 compare calls the constant model a success in some of these cases"
    assert max(abs(g) for *_, g in cases) < 16, cases


def test_a_constant_classifier_clears_chance_and_misses_the_majority_bar():
    n, num_classes, seq_len = 100, 2, 6
    _, labels = td.synthetic_documents(4, n, td.CLASSIFICATION)
    majority = tr.majority_share(labels)
    assert majority == 0.6
    chance = 1.0 / num_classes
    # a model that answers the majority class for every row: logits [10, 0] (class 0 is the 60 percent class)
    inputs = keras.Input((seq_len,), dtype="float32")
    ones = keras.layers.Lambda(lambda t: keras.ops.ones_like(t)[:, :1])(inputs)
    head = keras.layers.Dense(num_classes)
    model = keras.Model(inputs, head(ones))
    head.set_weights([np.zeros((1, num_classes), np.float32), np.array([10.0, 0.0], np.float32)])
    model.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])
    rows = np.zeros((n, seq_len), np.float32)
    scored = model.evaluate(td.select_dataset((rows, labels.astype(np.int32)), 7), verbose=0, return_dict=True)
    reached = float(scored["accuracy"])
    assert np.float32(reached) == np.float32(majority), "the constant model scores the majority share to the last bit"
    assert reached >= chance and tr.strictly_above(reached, chance), "a bar at chance calls the constant model a success"
    assert not tr.strictly_above(reached, majority), "the bar at the majority share calls it what it is"
    assert tr.strictly_above(0.61, majority) and tr.strictly_above(majority + 1e-6, majority)
    # the image task's auto bar is 2 x chance clipped to 0.95: for two classes it is 0.95, not a number a constant model reaches
    assert min(2 * chance, 0.95) == 0.95 and not reached >= 0.95
