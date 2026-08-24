"""
Guard: BERT's tanh-GELU default is live, serializable, and still escapable.

The reference claim itself -- *which* GELU form the built graph runs, measured
against an independently transcribed tanh formula -- has ONE home, and it is not
this file: ``tests/test_the_ported_numeric_defaults_match_their_references.py``
(rows ``bert``/``gemma`` of ``test_the_gelu_form_in_use_is_the_tanh_approximation``
plus the ``BERT.DEFAULT_HIDDEN_ACT`` scalar row). What remains here is the
BEHAVIOUR around that choice, which is BERT-specific and belongs with BERT: that
the choice moves the forward pass end to end, that ``hidden_act`` still
round-trips as a plain string, and that the exact form stays reachable for the
HuggingFace-tracking siblings.

Reference (the release ``bert.py``'s own References section cites, Devlin et al. 2018):
    https://github.com/google-research/bert/blob/master/modeling.py
        def gelu(x):
            cdf = 0.5 * (1.0 + tf.tanh((np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
            return x * cdf

Keras' bare ``"gelu"`` string is ``approximate=False`` -- the exact/erf form
(``keras/src/activations/activations.py:339``), which is a DIFFERENT function.
This is **inference-changing**: it alters the forward pass of every token in
every layer, not merely how training converges. See decisions.md D-500.

Nothing here re-asserts the form itself; see the module named above for that.
"""

import keras
import numpy as np
import pytest
from dl_techniques.layers.activations import gelu_tanh
from dl_techniques.models.bert import BERT

# ---------------------------------------------------------------------

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


# ---------------------------------------------------------------------


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
