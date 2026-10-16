"""``validate_model_loading`` raises when the archive cannot be loaded or run.

Guard for plan-2026-10-14T062616-81938e81 step 12 Part F item 6 (D-085).
Measured at d6995a932: the function wrapped the load, the predict and the
comparison in ``except Exception`` and returned ``False`` with a warning, so a
missing, corrupt or unloadable archive read exactly like a reload whose values
drifted, and every caller wrapped it in a second ``except Exception`` on top.
A value mismatch is still a ``False`` verdict (callers record it); a load or
run error is not a verdict and propagates.
"""

import numpy as np
import pytest
import keras

from train.common.evaluation import validate_model_loading


def _saved_model(tmp_path):
    keras.utils.set_random_seed(0)
    inputs = keras.Input((4,))
    model = keras.Model(inputs, keras.layers.Dense(3)(inputs))
    path = tmp_path / "model.keras"
    model.save(path)
    sample = np.random.default_rng(0).standard_normal((2, 4)).astype("float32")
    return str(path), sample, model.predict(sample, verbose=0)


def test_a_faithful_reload_validates(tmp_path):
    path, sample, expected = _saved_model(tmp_path)
    assert validate_model_loading(path, sample, expected) is True


def test_a_value_mismatch_is_a_false_verdict(tmp_path):
    path, sample, expected = _saved_model(tmp_path)
    assert validate_model_loading(path, sample, expected + 1.0) is False


def test_a_missing_archive_raises(tmp_path):
    _, sample, expected = _saved_model(tmp_path)
    with pytest.raises(ValueError):
        validate_model_loading(str(tmp_path / "absent.keras"), sample, expected)


def test_a_corrupt_archive_raises(tmp_path):
    _, sample, expected = _saved_model(tmp_path)
    corrupt = tmp_path / "corrupt.keras"
    corrupt.write_bytes(b"not a zip archive")
    with pytest.raises(Exception) as info:
        validate_model_loading(str(corrupt), sample, expected)
    assert not isinstance(info.value, AssertionError)


def test_an_input_the_reloaded_model_cannot_run_raises(tmp_path):
    path, _, expected = _saved_model(tmp_path)
    wrong_width = np.zeros((2, 5), dtype="float32")
    with pytest.raises(Exception) as info:
        validate_model_loading(path, wrong_width, expected)
    assert not isinstance(info.value, AssertionError)


def test_a_compiled_model_that_never_trained_validates(tmp_path):
    """The check reloads for inference, not to resume training.

    A compiled model saved before any ``fit`` holds an unbuilt optimizer;
    reloading it WITH its compile state warns "Skipping variable loading for
    optimizer" (a ``UserWarning``, an error under this suite's filter), which
    is about the optimizer, not about the weights this function compares.
    Measured when the D-085 change stopped swallowing load errors: the four
    stubbed-``fit`` runs of ``test_power_mlp/test_run_health.py`` failed on it.
    """
    keras.utils.set_random_seed(0)
    inputs = keras.Input((4,))
    model = keras.Model(inputs, keras.layers.Dense(3)(inputs))
    model.compile(optimizer="adam", loss="mse")
    path = tmp_path / "compiled.keras"
    model.save(path)
    sample = np.ones((2, 4), dtype="float32")
    assert validate_model_loading(str(path), sample, model.predict(sample, verbose=0)) is True
