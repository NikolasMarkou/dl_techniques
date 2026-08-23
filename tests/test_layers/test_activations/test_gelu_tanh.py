"""
Guard: ``gelu_tanh`` really is the tanh approximation, not Keras' exact default.

Reference for the approximation this module must implement:
    https://github.com/google-research/bert/blob/master/modeling.py
        cdf = 0.5 * (1.0 + tf.tanh((np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
    https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma3/configuration_gemma3.py
        hidden_activation = "gelu_pytorch_tanh"  (= F.gelu(..., approximate="tanh"))

Reference for the Keras default these assertions must distinguish it from:
    keras/src/activations/activations.py:339 -- ``def gelu(x, approximate=False)``

The two forms differ by ``max|exact - tanh| = 4.7324e-04`` at ``x ~= 2.699``
(computed in float64 with ``scipy.special.erf`` over a 2,000,001-point grid on
``[-6, 6]``). Every assertion below is written against an *independently
computed* formula, never against the implementation's own output, so a change of
form cannot pass by agreeing with itself.
"""

import keras
import numpy as np
import pytest
from keras import ops

from dl_techniques.layers.activations import gelu_tanh, resolve_activation

# ---------------------------------------------------------------------

#: max|exact-erf GELU - tanh GELU|, float64, x in [-6, 6]. Attained at x ~= 2.699.
EXPECTED_FORM_SEPARATION: float = 4.7324e-04


def _reference_tanh_gelu(x: np.ndarray) -> np.ndarray:
    """The original-BERT tanh approximation, written out from the reference."""
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))


@pytest.fixture
def grid() -> np.ndarray:
    return np.linspace(-6.0, 6.0, 20001).astype("float32")


# ---------------------------------------------------------------------


def test_gelu_tanh_matches_the_reference_tanh_formula(grid: np.ndarray) -> None:
    got = np.asarray(gelu_tanh(ops.convert_to_tensor(grid)))
    expected = _reference_tanh_gelu(grid.astype("float64")).astype("float32")
    # float32 round-off only; measured 9.54e-07.
    assert np.abs(got - expected).max() < 1e-5, (
        "gelu_tanh no longer computes the google-research/bert tanh formula"
    )


def test_gelu_tanh_is_not_the_exact_erf_form(grid: np.ndarray) -> None:
    """The whole point of this function: it must NOT be ``keras.activations.gelu``."""
    got = np.asarray(gelu_tanh(ops.convert_to_tensor(grid)))
    exact = np.asarray(keras.activations.gelu(ops.convert_to_tensor(grid)))
    separation = float(np.abs(got - exact).max())
    assert separation == pytest.approx(EXPECTED_FORM_SEPARATION, rel=0.05), (
        f"separation from the exact/erf GELU is {separation:.6e}, expected "
        f"~{EXPECTED_FORM_SEPARATION:.6e}; a separation of 0.0 means gelu_tanh "
        "has silently become the exact form"
    )


def test_keras_still_has_no_string_alias_for_the_approximation() -> None:
    """Why this module exists at all.

    ``keras.activations.get`` resolves strings only through ``ALL_OBJECTS_DICT``
    (``keras/src/activations/__init__.py``), which registers no approximate-GELU
    alias. If a future Keras adds one, this test goes red and the extended
    vocabulary in ``resolve_activation`` can be retired.
    """
    for name in ("gelu_approximate", "gelu_pytorch_tanh", "gelu_tanh"):
        with pytest.raises(ValueError):
            keras.activations.get(name)


def test_resolve_activation_maps_every_reference_spelling(grid: np.ndarray) -> None:
    for name in ("gelu_tanh", "gelu_approximate", "gelu_pytorch_tanh"):
        assert resolve_activation(name) is gelu_tanh, name
    # and does not swallow the stock vocabulary
    assert resolve_activation("relu") is keras.activations.relu
    assert resolve_activation("gelu") is keras.activations.gelu
    with pytest.raises(ValueError):
        resolve_activation("no_such_activation")


def test_gelu_tanh_round_trips_through_keras_serialization(grid: np.ndarray) -> None:
    """A lambda would fail here; the registered function must not."""
    cfg = keras.activations.serialize(gelu_tanh)
    assert keras.activations.deserialize(cfg) is gelu_tanh


def test_gelu_tanh_survives_a_saved_model_round_trip(tmp_path, grid: np.ndarray) -> None:
    model = keras.Sequential(
        [keras.layers.Input((4,)), keras.layers.Dense(4, activation=gelu_tanh)]
    )
    x = np.random.RandomState(0).randn(3, 4).astype("float32")
    before = np.asarray(model(x, training=False))
    path = tmp_path / "gelu_tanh.keras"
    model.save(path)
    reloaded = keras.models.load_model(path)
    assert reloaded.layers[-1].activation is gelu_tanh
    np.testing.assert_array_equal(before, np.asarray(reloaded(x, training=False)))

# ---------------------------------------------------------------------
