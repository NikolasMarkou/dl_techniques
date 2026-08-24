"""
Guard: Gemma 3's tanh-GELU gate is live and serializable.

The reference claim -- *which* GELU form the built block runs -- has ONE home,
and it is not this file:
``tests/test_the_ported_numeric_defaults_match_their_references.py``, row
``gemma`` of ``test_the_gelu_form_in_use_is_the_tanh_approximation``. What
remains here is the behaviour around that choice: that it moves the block output,
and that a registered function (unlike the lambda or ``functools.partial`` that
would otherwise express it) survives a ``.keras`` round trip.

Reference:
    https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma3/configuration_gemma3.py
        Gemma3TextConfig.hidden_activation = "gelu_pytorch_tanh"
    https://github.com/huggingface/transformers/blob/main/src/transformers/activations.py
        "gelu_pytorch_tanh" -> functools.partial(nn.functional.gelu, approximate="tanh")

Keras' bare ``"gelu"`` string is ``approximate=False`` -- the exact/erf form
(``keras/src/activations/activations.py:339``) -- so the call site previously
ran a DIFFERENT function from the reference. This is **inference-changing**, not
training-only. See decisions.md D-501.

Nothing here re-asserts the form itself; see the module named above for that.
"""

import keras
import numpy as np
import pytest
from dl_techniques.layers.activations import gelu_tanh
from dl_techniques.layers.transformers.gemma3_transformer import Gemma3TransformerBlock

# ---------------------------------------------------------------------

_KW = dict(
    hidden_size=32, num_attention_heads=2, num_key_value_heads=1,
    ffn_hidden_size=64, max_seq_len=16,
)


def _x() -> np.ndarray:
    return np.random.RandomState(1).randn(2, 8, 32).astype("float32")


# ---------------------------------------------------------------------


def test_the_choice_actually_moves_the_block_output() -> None:
    """Inference-changing, proven by swapping only the activation on one block."""
    keras.utils.set_random_seed(7)
    block = Gemma3TransformerBlock(**_KW)
    x = _x()
    y_tanh = np.asarray(block(x, training=False))

    # Control: the block is deterministic, so any delta below is signal.
    np.testing.assert_array_equal(y_tanh, np.asarray(block(x, training=False)))

    shipped = block.ffn.activation
    try:
        block.ffn.activation = keras.activations.gelu   # the old, exact form
        y_erf = np.asarray(block(x, training=False))
    finally:
        block.ffn.activation = shipped

    delta = float(np.abs(y_tanh - y_erf).max())
    assert delta > 1e-5, (
        f"swapping the gate to the exact form moved the output by only "
        f"{delta:.6e}; 0.0 would mean the activation never reaches the graph"
    )


def test_the_activation_round_trips_through_a_saved_model(tmp_path) -> None:
    """A lambda or ``functools.partial`` would break here; the registered
    function must serialize by name and deserialize to the identical object."""
    keras.utils.set_random_seed(7)
    model = keras.Sequential(
        [keras.layers.Input((8, 32)), Gemma3TransformerBlock(**_KW)]
    )
    x = _x()
    before = np.asarray(model(x, training=False))
    ffn_cfg = model.layers[-1].ffn.get_config()["activation"]
    assert ffn_cfg["config"] == "dl_techniques.activations>gelu_tanh", ffn_cfg

    path = tmp_path / "gemma_block.keras"
    model.save(path)
    reloaded = keras.models.load_model(path)
    assert reloaded.layers[-1].ffn.activation is gelu_tanh
    np.testing.assert_array_equal(before, np.asarray(reloaded(x, training=False)))

# ---------------------------------------------------------------------
