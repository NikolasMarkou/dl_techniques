"""A FITTED HKAN survives a ``.keras`` round trip, weight for weight.

An unfitted round trip proves little here: ``centers`` is rebuilt from the
seed in the config and ``mix`` from a constant, so a loader that dropped those
weights and re-ran their initializers would reproduce them. The closed-form
solution (and a data-driven center draw) exists ONLY in the weights of the
archive. Guards:

* the non-vacuity control: the fitted weights differ from a fresh build of the
  same config, including ``centers`` (data mode);
* every weight of the reloaded model equals the saved one at
  ``atol=0, rtol=0``, compared before the reloaded model is ever called;
* the number of weights (10 with both intercepts, 6 with neither) and their
  order survive, so a conditional weight cannot be loaded into another's slot;
* the reloaded model predicts identically and carries the same config.

Archives are written under ``tmp_path`` only.
"""

import os

import keras
import numpy as np
import pytest

from dl_techniques.models.general_purpose.hkan import HKAN

from . import HIDDEN, N_IN, NUM_BASIS, SLOPE, make_data

EXPECTED_NAMES = {
    True: ["coef", "block_bias", "mix", "bias", "centers"] * 2,
    False: ["coef", "mix", "centers"] * 2,
}


def _config(intercepts: bool) -> dict:
    return dict(
        hidden_units=(HIDDEN,), num_basis=NUM_BASIS, basis=("tanh", "identity"),
        slope=SLOPE, centers=("data", "random"), l2_block=(0.01, 0.1),
        use_block_bias=intercepts, use_bias=intercepts, seed=0)


@pytest.fixture(scope="module", params=[True, False], ids=["intercepts", "bias_free"])
def saved(request, tmp_path_factory):
    """A fitted model, its archive path, and the reloaded model (not yet called)."""
    intercepts = request.param
    x, y = make_data()
    model = HKAN(**_config(intercepts))
    model.fit_closed_form(x, y)
    path = os.path.join(str(tmp_path_factory.mktemp("hkan")), "fitted_hkan.keras")
    model.save(path)
    loaded = keras.models.load_model(path)
    loaded_weights = [(w.path, w.numpy().copy()) for w in loaded.weights]
    return {"intercepts": intercepts, "model": model, "loaded": loaded,
            "loaded_weights": loaded_weights, "x": x.astype("float32"), "path": path}


class TestTheFitIsWorthSaving:
    """Non-vacuity, asserted before the round trip means anything."""

    def test_the_fitted_weights_differ_from_a_fresh_build(self, saved):
        keras.utils.set_random_seed(0)
        fresh = HKAN(**_config(saved["intercepts"]))
        fresh.build((None, N_IN))
        names = [w.name for w in fresh.weights]
        differing = {
            f"{index // (len(names) // 2)}/{name}"
            for index, (name, ours, theirs) in enumerate(zip(
                names, saved["model"].get_weights(), fresh.get_weights()))
            if np.abs(ours - theirs).max() > 1e-3
        }
        # Every weight of both layers except the second layer's `random`
        # centers, which the seed in the config reproduces.
        expected = {f"{layer}/{name}" for layer in (0, 1) for name in set(names)} - {"1/centers"}
        assert differing == expected


class TestFittedRoundTrip:

    def test_the_loaded_class_and_config(self, saved):
        assert type(saved["loaded"]) is HKAN
        assert saved["loaded"].get_config() == saved["model"].get_config()

    def test_weight_count_and_order(self, saved):
        expected = EXPECTED_NAMES[saved["intercepts"]]
        assert [w.name for w in saved["model"].weights] == expected
        assert [w.name for w in saved["loaded"].weights] == expected
        assert len(saved["loaded_weights"]) == (10 if saved["intercepts"] else 6)
        assert [tuple(w.shape) for w in saved["loaded"].weights] == [
            tuple(w.shape) for w in saved["model"].weights]

    def test_every_weight_is_restored_exactly(self, saved):
        """Snapshot taken before the loaded model's first call."""
        original = [(w.path, w.numpy()) for w in saved["model"].weights]
        assert len(original) == len(saved["loaded_weights"])
        for (path, ours), (loaded_path, theirs) in zip(original, saved["loaded_weights"]):
            assert path.split("/", 1)[1] == loaded_path.split("/", 1)[1]
            assert ours.dtype == theirs.dtype == np.float32
            np.testing.assert_allclose(
                theirs, ours, atol=0, rtol=0, err_msg=f"{path} was not restored")

    def test_the_weights_are_still_those_after_a_call(self, saved):
        saved["loaded"](saved["x"], training=False)
        for (path, before), weight in zip(saved["loaded_weights"], saved["loaded"].weights):
            np.testing.assert_array_equal(weight.numpy(), before, err_msg=path)

    def test_the_reloaded_model_predicts_identically(self, saved):
        before = keras.ops.convert_to_numpy(saved["model"](saved["x"], training=False))
        after = keras.ops.convert_to_numpy(saved["loaded"](saved["x"], training=False))
        assert np.std(before) > 0.1, "the fitted model predicts a constant"
        np.testing.assert_allclose(after, before, atol=0, rtol=0)

    def test_the_archive_is_under_tmp_path(self, saved):
        assert os.path.isfile(saved["path"])
        assert f"{os.sep}results{os.sep}" not in saved["path"]


def test_an_unfitted_round_trip_restores_the_random_start(tmp_path):
    """The gradient-only start also travels in the archive: ``coef`` is a
    random draw the config cannot reproduce."""
    model = HKAN(hidden_units=(HIDDEN,), num_basis=NUM_BASIS, seed=None)
    model.build((None, N_IN))
    path = os.path.join(str(tmp_path), "unfitted_hkan.keras")
    model.save(path)
    loaded = keras.models.load_model(path)
    assert len(loaded.weights) == 10
    for ours, theirs in zip(model.get_weights(), loaded.get_weights()):
        np.testing.assert_allclose(theirs, ours, atol=0, rtol=0)
