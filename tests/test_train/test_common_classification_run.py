"""``finalize_last_epoch(sample=)``: the load check of a model whose input is not one array.

``finalize_last_epoch`` builds the sample of its reload check from one of two places: the first
batch of ``select_ds`` (no test split) or ``data.x_fig`` (a test split). Both assume an image
classifier: the first element of a ``select_ds`` batch is one array, and ``x_fig`` is one array.
The masked-image models take ``[image, mask]`` and their streams yield ``((image, mask), target,
weight)``, so both branches fail on them. MEASURED on the two-input ET and BEiT models (plan
plan-2026-11-12T015747-46e5318f, step 2 item (a)):

* no test split: ``AttributeError: 'tuple' object has no attribute 'numpy'`` at the ``select_ds`` read;
* a test split and no ``x_fig``: ``TypeError: 'NoneType' object is not subscriptable``.

``check_model_loading`` itself needs no change (a list sample round-trips with a difference of
0.0), so the fix is one optional keyword: ``sample=`` is the load-check sample, used as given
(the caller slices), and ``None`` (every caller that exists) keeps both legacy branches.

What is pinned here:

* the two measured failures stay (a call that gives no sample still reads one of the two places);
* with ``sample=[x, mask]`` the final model is saved, reloaded, its weights equal the model's and its
  scores equal the scores on that sample, on both the exact and the legacy check, with and without a
  test split, and the data record needs neither ``select_ds`` nor ``x_fig``;
* the check receives THAT sample (both elements, unchanged) and the scores of THAT sample;
* ``sample=None`` is the call that never named it, on both legacy branches (the ``x_fig`` branch is
  also pinned by ``test_the_reload_check_is_honest_about_scale_and_tf32.py``; the ``select_ds``
  branch has no other test that reaches the load check).

Writes only under ``tmp_path``.
"""

import inspect
from types import SimpleNamespace

import keras
import numpy as np
import pytest
import tensorflow as tf

from train.common import classification_run
from train.common.classification_run import LOAD_CHECK_SAMPLES

B = 8


def two_input_model(seed: int = 0) -> keras.Model:
    """A model of ``[image, mask]``: the output depends on both inputs, so a sample that loses one is seen."""
    keras.utils.set_random_seed(seed)
    image = keras.Input((4, 4, 3))
    mask = keras.Input((16,))
    hidden = keras.layers.Concatenate()([keras.layers.Flatten()(image), mask])
    hidden = keras.layers.Dense(12, activation="tanh")(hidden)
    return keras.Model([image, mask], keras.layers.Dense(5)(hidden))


def one_input_model(seed: int = 0) -> keras.Model:
    keras.utils.set_random_seed(seed)
    return keras.Sequential([keras.Input((6,)), keras.layers.Dense(4, activation="tanh"), keras.layers.Dense(3)])


@pytest.fixture
def sample():
    rng = np.random.default_rng(0)
    return [rng.standard_normal((B, 4, 4, 3)).astype("float32"), (rng.random((B, 16)) < 0.3).astype("float32")]


def tuple_stream(sample) -> "tf.data.Dataset":
    """The element shape the masked-image streams yield: ``((image, mask), target, weight)``."""
    target = np.zeros((B, 5), "float32")
    return tf.data.Dataset.from_tensor_slices(
        ((sample[0], sample[1]), target, np.ones((B,), "float32"))
    ).batch(4)


def finalize(model, run_dir, data, **kwargs):
    run_dir.mkdir(parents=True, exist_ok=True)
    return classification_run.finalize_last_epoch(
        model, SimpleNamespace(weights=model.get_weights()), data, run_dir, SimpleNamespace(history={}), 1,
        lambda m, d: {"loss": 0.0}, **kwargs,
    )


# ---------------------------------------------------------------------
# what is measured to fail without the keyword stays failing without it
# ---------------------------------------------------------------------

def test_a_tuple_element_select_stream_without_a_sample_fails_as_measured(sample, tmp_path):
    data = SimpleNamespace(test_ds=None, select_ds=tuple_stream(sample), x_fig=None)
    with pytest.raises(AttributeError, match="'tuple' object has no attribute 'numpy'"):
        finalize(two_input_model(), tmp_path, data)


def test_a_test_split_without_x_fig_and_without_a_sample_fails_as_measured(sample, tmp_path):
    data = SimpleNamespace(test_ds=tuple_stream(sample), select_ds=tuple_stream(sample), x_fig=None)
    with pytest.raises(TypeError, match="'NoneType' object is not subscriptable"):
        finalize(two_input_model(), tmp_path, data)


# ---------------------------------------------------------------------
# with a sample: saved, reloaded, equal weights, equal scores
# ---------------------------------------------------------------------

@pytest.mark.parametrize("with_test_split", [False, True], ids=["no test split", "test split"])
def test_a_list_sample_is_saved_reloaded_and_checked_for_equal_weights_and_scores(sample, tmp_path, with_test_split):
    model = two_input_model()
    # The record carries neither a usable select_ds nor an x_fig: only the sample can feed the check.
    data = SimpleNamespace(test_ds=tuple_stream(sample) if with_test_split else None, select_ds=None, x_fig=None)
    detail = {}
    _, ok = finalize(model, tmp_path, data, sample=sample, exact_reload=True, atomic=True, load_detail=detail)
    assert ok is True
    assert detail["weights_equal"] is True and detail["max_abs_diff"] == 0.0
    reloaded = keras.models.load_model(tmp_path / "final_model.keras", compile=False)
    assert all(np.array_equal(a, b) for a, b in zip(reloaded.get_weights(), model.get_weights()))
    assert np.array_equal(reloaded.predict(sample, verbose=0), model.predict(sample, verbose=0))


def test_a_list_sample_passes_the_legacy_absolute_check_too(sample, tmp_path):
    _, ok = finalize(two_input_model(), tmp_path, SimpleNamespace(test_ds=None, select_ds=None, x_fig=None), sample=sample)
    assert ok is True


def test_the_check_receives_that_sample_and_the_scores_of_that_sample(sample, tmp_path, monkeypatch):
    model = two_input_model()
    seen = []
    real = classification_run.check_model_loading

    def spy(path, test_sample, expected, *args, **kwargs):
        seen.append((test_sample, expected))
        return real(path, test_sample, expected, *args, **kwargs)

    monkeypatch.setattr(classification_run, "check_model_loading", spy)
    # Legacy records on purpose: a stream and an x_fig that hold OTHER data than the sample, so a call
    # that reads either of them in place of the sample hands the check the wrong thing.
    other = [np.ones_like(sample[0]), np.ones_like(sample[1])]
    data = SimpleNamespace(test_ds=tuple_stream(other), select_ds=tuple_stream(other), x_fig=other[0])
    finalize(model, tmp_path, data, sample=sample, exact_reload=True)
    (test_sample, expected), = seen
    assert len(test_sample) == 2
    for given, got in zip(sample, test_sample):
        assert np.array_equal(given, got)
    assert np.array_equal(expected, model.predict(sample, verbose=0))


def test_a_single_array_sample_is_used_as_given_and_not_cut(tmp_path, monkeypatch):
    """The caller slices; the keyword does not apply the legacy ``LOAD_CHECK_SAMPLES`` cut to what it is given."""
    model = one_input_model()
    given = np.random.default_rng(3).standard_normal((LOAD_CHECK_SAMPLES + 7, 6)).astype("float32")
    seen = []
    real = classification_run.check_model_loading
    monkeypatch.setattr(
        classification_run, "check_model_loading",
        lambda path, test_sample, expected, *a, **k: (seen.append(test_sample), real(path, test_sample, expected, *a, **k))[1],
    )
    data = SimpleNamespace(test_ds=None, select_ds=None, x_fig=None)
    _, ok = finalize(model, tmp_path, data, sample=given, exact_reload=True)
    assert ok is True and seen[0].shape == given.shape


def test_a_wrong_final_file_is_still_reported_false_with_a_sample(sample, tmp_path, monkeypatch):
    """The check is not a rubber stamp when a sample is given: another model's file fails it."""
    other = two_input_model(seed=1)
    real = keras.Model.save

    def save_the_other(self, path, *args, **kwargs):
        return real(other, path, *args, **kwargs)

    monkeypatch.setattr(keras.Model, "save", save_the_other)
    _, ok = finalize(two_input_model(seed=0), tmp_path, SimpleNamespace(test_ds=None, select_ds=None, x_fig=None),
                     sample=sample, exact_reload=True)
    assert ok is False


# ---------------------------------------------------------------------
# without a sample: the two legacy branches, unchanged
# ---------------------------------------------------------------------

def test_the_keyword_is_optional_and_defaults_to_none():
    assert inspect.signature(classification_run.finalize_last_epoch).parameters["sample"].default is None


def test_the_select_stream_branch_reads_the_first_batch_and_cuts_it(tmp_path, monkeypatch):
    """No test split, no sample: the load check reads ``select_ds.take(1)[0]``, cut to ``LOAD_CHECK_SAMPLES``."""
    model = one_input_model()
    n = LOAD_CHECK_SAMPLES + 9
    x = np.random.default_rng(4).standard_normal((n, 6)).astype("float32")
    stream = tf.data.Dataset.from_tensor_slices((x, np.zeros((n,), "int32"))).batch(n)
    seen = []
    real = classification_run.validate_model_loading
    monkeypatch.setattr(
        classification_run, "validate_model_loading",
        lambda path, test_sample, expected, **k: (seen.append((test_sample, expected)), real(path, test_sample, expected, **k))[1],
    )
    _, ok = finalize(model, tmp_path, SimpleNamespace(test_ds=None, select_ds=stream, x_fig=None))
    assert ok is True
    (test_sample, expected), = seen
    assert np.array_equal(test_sample, x[:LOAD_CHECK_SAMPLES])
    assert np.array_equal(expected, model.predict(x[:LOAD_CHECK_SAMPLES], verbose=0))


def test_the_x_fig_branch_reads_x_fig_cut_to_the_load_check_samples(tmp_path, monkeypatch):
    model = one_input_model()
    x_fig = np.random.default_rng(5).standard_normal((LOAD_CHECK_SAMPLES + 9, 6)).astype("float32")
    seen = []
    real = classification_run.validate_model_loading
    monkeypatch.setattr(
        classification_run, "validate_model_loading",
        lambda path, test_sample, expected, **k: (seen.append(test_sample), real(path, test_sample, expected, **k))[1],
    )
    _, ok = finalize(model, tmp_path, SimpleNamespace(test_ds=object(), select_ds=None, x_fig=x_fig))
    assert ok is True and np.array_equal(seen[0], x_fig[:LOAD_CHECK_SAMPLES])


def test_an_explicit_none_is_the_call_that_never_named_it(tmp_path):
    x_fig = np.random.default_rng(6).standard_normal((20, 6)).astype("float32")
    results = []
    for name, kwargs in (("default", {}), ("none", {"sample": None})):
        detail = {}
        results.append(finalize(
            one_input_model(), tmp_path / name, SimpleNamespace(test_ds=object(), select_ds=None, x_fig=x_fig),
            exact_reload=True, load_detail=detail, **kwargs,
        ) + (detail,))
    assert results[0] == results[1]
