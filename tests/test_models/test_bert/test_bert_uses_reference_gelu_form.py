"""
Guard: BERT's default activation is the ORIGINAL RELEASE's tanh-approximate GELU.

Reference (the release ``bert.py``'s own References section cites, Devlin et al. 2018):
    https://github.com/google-research/bert/blob/master/modeling.py
        def gelu(x):
            cdf = 0.5 * (1.0 + tf.tanh((np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
            return x * cdf

Keras' bare ``"gelu"`` string is ``approximate=False`` -- the exact/erf form
(``keras/src/activations/activations.py:339``), which is a DIFFERENT function.
This is **inference-changing**: it alters the forward pass of every token in
every layer, not merely how training converges. See decisions.md D-500.

These assertions deliberately probe the callable that the BUILT GRAPH holds,
and compare it against an independently written-out tanh formula. A test that
only asserted ``model.hidden_act == "gelu_tanh"`` could not see which form the
graph actually runs.
"""

import keras
import numpy as np
import pytest
from keras import ops

from dl_techniques.layers.activations import gelu_tanh
from dl_techniques.models.bert.bert import BERT

# ---------------------------------------------------------------------

#: max|exact-erf GELU - tanh GELU|, float64, x in [-6, 6]. Attained at x ~= 2.699.
EXPECTED_FORM_SEPARATION: float = 4.7324e-04

_KW = dict(
    vocab_size=64, hidden_size=32, num_layers=2, num_heads=2,
    intermediate_size=64, max_position_embeddings=16, type_vocab_size=2,
)


def _inputs():
    ids = np.random.RandomState(0).randint(0, 64, size=(2, 8))
    return {
        "input_ids": ids,
        "attention_mask": np.ones_like(ids),
        "token_type_ids": np.zeros_like(ids),
    }


def _reference_tanh_gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))


# ---------------------------------------------------------------------


def test_default_hidden_act_names_the_approximation() -> None:
    assert BERT.DEFAULT_HIDDEN_ACT == "gelu_tanh"


def test_every_encoder_ffn_runs_the_tanh_form_not_the_erf_form() -> None:
    """The load-bearing assertion: which function the graph calls."""
    keras.utils.set_random_seed(1234)
    model = BERT(**_KW)
    model(_inputs(), training=False)

    grid = np.linspace(-6.0, 6.0, 20001).astype("float32")
    expected_tanh = _reference_tanh_gelu(grid.astype("float64")).astype("float32")
    exact_erf = np.asarray(keras.activations.gelu(ops.convert_to_tensor(grid)))

    assert len(model.encoder_layers) == _KW["num_layers"]
    for i, layer in enumerate(model.encoder_layers):
        fn = layer.ffn_layer.activation_fn
        got = np.asarray(fn(ops.convert_to_tensor(grid)))
        assert np.abs(got - expected_tanh).max() < 1e-5, (
            f"encoder_layer_{i}'s FFN activation is not the "
            "google-research/bert tanh approximation"
        )
        separation = float(np.abs(got - exact_erf).max())
        assert separation == pytest.approx(EXPECTED_FORM_SEPARATION, rel=0.05), (
            f"encoder_layer_{i}'s FFN activation sits {separation:.6e} from the "
            "exact/erf GELU; 0.0 would mean BERT has silently reverted to Keras' "
            "approximate=False default"
        )


def test_the_choice_actually_moves_the_forward_pass() -> None:
    """Inference-changing, proven end-to-end against weight-identical twins.

    The two models are given identical weights, so the only difference between
    them is the activation form. The delta scales with the weight magnitude, so
    the probe uses ``initializer_range=0.2``: at BERT's own 0.02 the final
    LayerNorm renormalises the perturbation down to ~4.77e-07, which is real
    (the control below is exactly 0.0) but too close to float32 noise to pin.
    """
    keras.utils.set_random_seed(1234)
    kw = dict(_KW, initializer_range=0.2)
    tanh_model = BERT(**kw)
    erf_model = BERT(**kw, hidden_act="gelu")
    inputs = _inputs()
    tanh_model(inputs, training=False)
    erf_model(inputs, training=False)
    erf_model.set_weights(tanh_model.get_weights())

    def out(m):
        return np.asarray(m(inputs, training=False)["last_hidden_state"])

    y_tanh, y_erf = out(tanh_model), out(erf_model)

    # Control: the harness itself is deterministic, so any delta below is signal.
    np.testing.assert_array_equal(y_tanh, out(tanh_model))

    delta = float(np.abs(y_tanh - y_erf).max())
    assert delta > 1e-5, (
        f"switching hidden_act moved the output by only {delta:.6e}; a delta of "
        "0.0 means the activation never reaches the graph and the fix is inert"
    )


def test_hidden_act_round_trips_as_a_plain_string(tmp_path) -> None:
    """``hidden_act`` stays a str, so ``get_config()`` needs no custom handling."""
    keras.utils.set_random_seed(1234)
    model = BERT(**_KW)
    inputs = _inputs()
    before = np.asarray(model(inputs, training=False)["last_hidden_state"])
    assert model.get_config()["hidden_act"] == "gelu_tanh"

    path = tmp_path / "bert.keras"
    model.save(path)
    reloaded = keras.models.load_model(path)
    assert reloaded.hidden_act == "gelu_tanh"
    assert reloaded.encoder_layers[0].ffn_layer.activation_fn is gelu_tanh
    np.testing.assert_array_equal(
        before, np.asarray(reloaded(inputs, training=False)["last_hidden_state"])
    )


def test_the_exact_form_is_still_reachable_for_hf_style_checkpoints() -> None:
    """``distilbert``/``modern_bert`` track HuggingFace, whose configs specify the
    exact form. The escape hatch that keeps this fix BERT-specific must work."""
    keras.utils.set_random_seed(1234)
    model = BERT(**_KW, hidden_act="gelu")
    model(_inputs(), training=False)
    assert model.encoder_layers[0].ffn_layer.activation_fn is keras.activations.gelu

# ---------------------------------------------------------------------
