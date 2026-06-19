"""Tests for auto-wiring the COMPLIANT new FFN types into TransformerLayer.

Step 8 (D-005): the position-wise new FFN types
(`squared_relu`, `lowrank`, `reglu`, `bilinear`, `monarch`) must be usable
inside a TransformerLayer WITHOUT an explicit `ffn_args` override, because
`_get_ffn_config` now maps `intermediate_size`->`hidden_dim` and
`hidden_size`->`output_dim` for them.

The token-mixing `mixer` block is NOT position-wise (its required params
`tokens_mlp_dim`/`channels_mlp_dim` do not map to intermediate_size/hidden_size,
and it mixes across tokens), so it is intentionally NOT auto-wired and must
raise when used without `ffn_args`.

`hidden_size=64`, `intermediate_size=128` are both divisible by 4 so the
`monarch` default `nblocks=4` divisibility precondition is satisfied.
"""

import os
import tempfile

import keras
import numpy as np
import pytest
from keras import layers, models, ops

from dl_techniques.layers.transformers.transformer import TransformerLayer
from dl_techniques.layers.ffn.glu_ffn import GLUFFN


HIDDEN_SIZE = 64
NUM_HEADS = 4
INTERMEDIATE_SIZE = 128  # divisible by 4 for monarch's default nblocks=4


@pytest.mark.parametrize(
    "ffn_type", ["squared_relu", "lowrank", "reglu", "bilinear", "monarch"]
)
def test_compliant_new_ffn_types_autowire(ffn_type):
    """Each compliant new FFN type builds + forward-passes inside a
    TransformerLayer WITHOUT any `ffn_args` override, producing (2,16,64)."""
    layer = TransformerLayer(
        hidden_size=HIDDEN_SIZE,
        num_heads=NUM_HEADS,
        intermediate_size=INTERMEDIATE_SIZE,
        ffn_type=ffn_type,
    )
    x = keras.random.normal((2, 16, HIDDEN_SIZE))
    y = layer(x)
    assert tuple(y.shape) == (2, 16, HIDDEN_SIZE)
    assert np.all(np.isfinite(ops.convert_to_numpy(y))), (
        f"non-finite output for ffn_type={ffn_type}"
    )


def test_reglu_bilinear_preserve_gate():
    """The no-activation branch must NOT override the alias gate: reglu keeps
    its relu gate and bilinear keeps its linear gate (not the gelu default)."""
    reglu = TransformerLayer(
        hidden_size=HIDDEN_SIZE,
        num_heads=NUM_HEADS,
        intermediate_size=INTERMEDIATE_SIZE,
        ffn_type="reglu",
    )
    assert isinstance(reglu.ffn_layer, GLUFFN)
    assert keras.activations.serialize(reglu.ffn_layer.activation) == "relu"

    bilinear = TransformerLayer(
        hidden_size=HIDDEN_SIZE,
        num_heads=NUM_HEADS,
        intermediate_size=INTERMEDIATE_SIZE,
        ffn_type="bilinear",
    )
    assert isinstance(bilinear.ffn_layer, GLUFFN)
    assert keras.activations.serialize(bilinear.ffn_layer.activation) == "linear"


def test_squared_relu_no_activation_injected():
    """squared_relu builds without error -> we did NOT inject an `activation`
    kwarg that SquaredReLUFFN (which has no such param) would reject."""
    layer = TransformerLayer(
        hidden_size=HIDDEN_SIZE,
        num_heads=NUM_HEADS,
        intermediate_size=INTERMEDIATE_SIZE,
        ffn_type="squared_relu",
    )
    x = keras.random.normal((2, 16, HIDDEN_SIZE))
    y = layer(x)
    assert tuple(y.shape) == (2, 16, HIDDEN_SIZE)


def test_mixer_not_autowired():
    """`mixer` is a token-mixing block (not position-wise) and is intentionally
    NOT auto-wired: without `ffn_args` supplying tokens_mlp_dim/channels_mlp_dim
    it must raise (at construction or build/call)."""
    with pytest.raises(Exception):
        layer = TransformerLayer(
            hidden_size=HIDDEN_SIZE,
            num_heads=NUM_HEADS,
            intermediate_size=INTERMEDIATE_SIZE,
            ffn_type="mixer",
        )
        layer(keras.random.normal((2, 16, HIDDEN_SIZE)))


def test_serialization_cycle():
    """A TransformerLayer wired with ffn_type='lowrank' (no ffn_args)
    round-trips through .keras with numerically identical outputs."""
    inputs = layers.Input(shape=(16, HIDDEN_SIZE))
    outputs = TransformerLayer(
        hidden_size=HIDDEN_SIZE,
        num_heads=NUM_HEADS,
        intermediate_size=INTERMEDIATE_SIZE,
        ffn_type="lowrank",
        dropout_rate=0.0,
        attention_dropout_rate=0.0,
    )(inputs)
    model = models.Model(inputs, outputs)

    x = keras.random.normal((2, 16, HIDDEN_SIZE))
    original = model(x, training=False)

    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "lowrank_transformer.keras")
        model.save(filepath)
        loaded = models.load_model(filepath)
        reloaded = loaded(x, training=False)

    np.testing.assert_allclose(
        ops.convert_to_numpy(original),
        ops.convert_to_numpy(reloaded),
        rtol=1e-5,
        atol=1e-5,
    )
