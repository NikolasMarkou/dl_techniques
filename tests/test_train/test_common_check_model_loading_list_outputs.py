"""``check_model_loading`` compares a model with several outputs output by output.

A forecaster such as N-BEATS returns ``(forecast, residual)``: two arrays of different shapes. MEASURED
(plan-2026-11-13T105600-3a8319ef step 5, item a) the shared check raised on it, because it stacked the
list into one array (``ValueError: setting an array element with a sequence. The requested array has an
inhomogeneous shape``), while a dict of named outputs passed. The list or tuple branch pairs the outputs,
reports the worst gap over all of them and fails on a difference in any one.

What is pinned here, on a small two-head model written under ``tmp_path``:

* a list of outputs of DIFFERENT shapes (the error above, RED before the branch) is a faithful reload with
  a gap of exactly 0.0, for the absolute check and for the relative one with the weights compared;
* the gap reported is the WORST over the outputs, and ``scale`` is the largest absolute value over them;
* a reloaded model whose SECOND output differs is a failed check, while its FIRST output alone is equal (a
  check that read only the first output would pass), for different shapes and for equal ones;
* a tuple of expected outputs is the same as a list; a different length, or a list against one array, is a
  verdict (``ok`` False, no gap), not a crash;
* a single array and a dict of named outputs give what they always gave (the legacy ``allclose`` verdict).
"""

import keras
import numpy as np
import pytest

from train.common.evaluation import LOAD_RTOL, check_model_loading

BATCH = 6


def two_head_model(seed, second_kernel_scale=1.0, equal_shapes=False):
    """A model with outputs ``(B, 4, 1)`` and ``(B, 7)`` (or ``(B, 4, 1)`` twice for ``equal_shapes``)."""
    rng = np.random.default_rng(seed)
    inputs = keras.Input((5,))
    first = keras.layers.Dense(4, name="first")
    second = keras.layers.Dense(4 if equal_shapes else 7, name="second")
    forecast = keras.layers.Reshape((4, 1))(first(inputs))
    other = keras.layers.Reshape((4, 1))(second(inputs)) if equal_shapes else second(inputs)
    model = keras.Model(inputs, [forecast, other])
    first.set_weights([rng.standard_normal((5, 4)).astype("float32"), rng.standard_normal(4).astype("float32")])
    width = 4 if equal_shapes else 7
    second.set_weights([
        (rng.standard_normal((5, width)) * second_kernel_scale).astype("float32"),
        rng.standard_normal(width).astype("float32")])
    return model


@pytest.fixture(scope="module")
def sample():
    return np.random.default_rng(3).standard_normal((BATCH, 5)).astype("float32")


def saved(model, tmp_path, name="m.keras"):
    path = tmp_path / name
    model.save(path)
    return str(path)


def test_the_two_output_model_has_outputs_of_different_shapes(sample):
    outputs = two_head_model(0).predict(sample, verbose=0)
    assert isinstance(outputs, list) and outputs[0].shape == (BATCH, 4, 1) and outputs[1].shape == (BATCH, 7)


def test_outputs_of_different_shapes_are_a_faithful_reload_with_a_gap_of_zero(sample, tmp_path):
    model = two_head_model(0)
    expected = model.predict(sample, verbose=0)
    path = saved(model, tmp_path)
    absolute = check_model_loading(path, sample, expected)
    assert absolute.ok is True and absolute.max_abs_diff == 0.0 and absolute.weights_equal is None
    exact = check_model_loading(path, sample, expected, relative_tolerance=LOAD_RTOL, reference_model=model)
    assert exact.ok is True and exact.max_abs_diff == 0.0 and exact.weights_equal is True
    assert exact.scale == pytest.approx(max(float(np.abs(a).max()) for a in expected), rel=1e-6)


@pytest.mark.parametrize("equal_shapes", [False, True])
@pytest.mark.parametrize("relative_tolerance", [0.0, LOAD_RTOL])
def test_a_second_output_that_differs_fails_the_check_while_the_first_is_equal(
        sample, tmp_path, equal_shapes, relative_tolerance):
    model = two_head_model(0, equal_shapes=equal_shapes)
    expected = model.predict(sample, verbose=0)
    off = two_head_model(0, second_kernel_scale=1.5, equal_shapes=equal_shapes)
    got = off.predict(sample, verbose=0)
    assert np.array_equal(got[0], expected[0]), "the FIRST output is equal: a check of the first output alone passes"
    assert float(np.abs(got[1] - expected[1]).max()) > 0.1
    check = check_model_loading(saved(off, tmp_path), sample, expected, relative_tolerance=relative_tolerance)
    assert check.ok is False
    assert check.max_abs_diff == pytest.approx(float(np.abs(got[1] - expected[1]).max()), rel=1e-6)


def test_the_gap_reported_is_the_worst_over_the_outputs(sample, tmp_path):
    model = two_head_model(0)
    expected = model.predict(sample, verbose=0)
    path = saved(model, tmp_path)
    nudged = [expected[0] + 2e-6, expected[1] + 5e-6]
    check = check_model_loading(path, sample, nudged, tolerance=1e-4)
    assert check.ok is True
    assert check.max_abs_diff == pytest.approx(5e-6, rel=0.05)
    reverse = [expected[0] + 5e-6, expected[1] + 2e-6]
    assert check_model_loading(path, sample, reverse, tolerance=1e-4).max_abs_diff == pytest.approx(5e-6, rel=0.05)


def test_a_tuple_of_expected_outputs_is_the_same_as_a_list(sample, tmp_path):
    model = two_head_model(0)
    expected = model.predict(sample, verbose=0)
    path = saved(model, tmp_path)
    assert check_model_loading(path, sample, tuple(expected)).ok is True
    assert check_model_loading(path, sample, tuple(expected[::-1])).ok is False


def test_a_different_number_of_outputs_or_a_single_array_is_a_verdict_not_a_crash(sample, tmp_path):
    model = two_head_model(0)
    expected = model.predict(sample, verbose=0)
    path = saved(model, tmp_path)
    for wrong in (expected[:1], expected + [expected[0]], expected[1]):
        check = check_model_loading(path, sample, wrong)
        assert check.ok is False and check.max_abs_diff is None and check.scale is None


def test_a_single_array_and_a_dict_keep_their_verdicts(tmp_path):
    keras.utils.set_random_seed(0)
    inputs = keras.Input((4,))
    model = keras.Model(inputs, keras.layers.Dense(3)(inputs))
    one = np.random.default_rng(0).standard_normal((2, 4)).astype("float32")
    expected = model.predict(one, verbose=0)
    path = saved(model, tmp_path)
    assert check_model_loading(path, one, expected).ok is True
    assert check_model_loading(path, one, expected + 1e-3).ok is False
    # numpy's allclose with its own rtol of 1e-5 of the reloaded value: the legacy verdict of one array
    assert check_model_loading(path, one, expected * (1 + 5e-6)).ok is True
    assert check_model_loading(path, one, {"a": expected}).ok is False
    assert check_model_loading(path, one, [expected]).ok is False
    named = keras.Model(inputs, {"a": keras.layers.Dense(3)(inputs), "b": keras.layers.Dense(2)(inputs)})
    named_expected = named.predict(one, verbose=0)
    assert check_model_loading(saved(named, tmp_path, "n.keras"), one, named_expected).ok is True
    nudged = dict(named_expected, b=named_expected["b"] + 1e-3)
    assert check_model_loading(saved(named, tmp_path, "n2.keras"), one, nudged).ok is False


def named_pair(tmp_path):
    """A two-output model with NAMED outputs of different widths, its saved file, a sample and the expected dict."""
    keras.utils.set_random_seed(1)
    inputs = keras.Input((4,))
    model = keras.Model(inputs, {"a": keras.layers.Dense(3)(inputs), "b": keras.layers.Dense(2)(inputs)})
    one = np.random.default_rng(1).standard_normal((2, 4)).astype("float32")
    return saved(model, tmp_path), one, model.predict(one, verbose=0)


def test_a_dict_of_outputs_is_paired_by_key_and_never_by_position(tmp_path):
    path, one, expected = named_pair(tmp_path)
    assert check_model_loading(path, one, expected).ok is True
    # the same content in the other insertion order is the same answer: the pairs are by name (a pairing by position would
    # compare the (2, 3) output with the (2, 2) one, a shape mismatch, and fail a faithful reload)
    reordered = {key: expected[key] for key in reversed(list(expected))}
    assert list(reordered) != list(expected)
    check = check_model_loading(path, one, reordered)
    assert check.ok is True and check.max_abs_diff == 0.0


def test_a_dict_with_other_keys_than_the_models_is_a_verdict_not_a_crash_and_not_a_pass(tmp_path):
    path, one, expected = named_pair(tmp_path)
    for asked in ({"a": expected["a"]}, {**expected, "c": expected["a"]}, {"a": expected["a"], "c": expected["b"]}):
        check = check_model_loading(path, one, asked)
        assert check.ok is False and check.max_abs_diff is None and check.scale is None, sorted(asked)
