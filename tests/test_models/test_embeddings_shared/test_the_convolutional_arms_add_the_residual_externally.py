"""Both convolutional arms must ADD their blocks' updates, not replace the signal.

Neither wrapped block adds its own residual:

- ``CliffordNetBlock.call`` ends at ``return h_mix`` under the comment
  "transform only; residual add is external" (``clifford_block.py:1753``).
- ``ConvNextV1Block.call`` ends at ``return x`` after its LayerScale step, with
  no ``+ inputs``.

So ``x = block(x)`` does not apply a block, it REPLACES the representation with
a residual term. The failure is silent in every conventional sense: shapes
match, outputs are finite, ``get_config`` round-trips, gradients exist. Only the
MAGNITUDE moves, which is why the assertion below is a magnitude bound.

Measured on this model, ``hidden_size=64``, 4 blocks, ``K=7``, embeddings
normalized to RMS 1.0:

    arm        with the residual    updates chained (the defect)
    clifford   1.000020             6.968e-13
    convnext   1.001831             8.058e-03

The Clifford arm collapses twelve orders because its LayerScale starts at 1e-5;
the ConvNeXt arm starts at 1.0 and so collapses "only" two orders. Both are
fatal, and one bound at 0.5 separates health from either defect -- but note that
a threshold tuned to the Clifford arm's 1e-13 alone would be far too generous
to catch the ConvNeXt arm.

**Do NOT extend this guard to the transformer arm.** The exclusion is not just
"it does not need it": the oracle below is structurally incapable of seeing the
defect there. `TransformerLayer` runs post-LN in this study, so a `LayerNorm`
sets the output scale regardless of whether the residual is present. Measured
2026-08-30 by deleting BOTH residual adds from the post-LN path and chaining
four layers: the surviving-RMS ratio is **1.00330 with the residuals and
1.00330 without them** -- bit-identical. A test added here for the transformer
would pass in both directions and pin nothing.

That defect is not unguarded, it is just guarded elsewhere: injecting it reddens
`test_the_positional_signal_survives_the_embedding_sum` (dominance falls to
0.00x for `ascii_bert`) and `test_different_strategies_give_different_embeddings`.
"""

import keras
import numpy as np
import pytest

from dl_techniques.models.embeddings_experimental.shared import EmbeddingEncoder

SEQ_LEN = 32
HIDDEN = 64
NUM_LAYERS = 4

#: Ratio of stack-output RMS to embedding RMS below which the signal is
#: considered annihilated. Healthy measures ~1.0; the defect measures 8.1e-03
#: (convnext) and 7.0e-13 (clifford).
MIN_SURVIVING_RATIO = 0.5

ARMS = [
    ("clifford", {"shifts": [1, 2], "context_kernel_size": 7}),
    ("convnext", {"kernel_size": 7}),
    ("convnext_v2", {"kernel_size": 7}),
]
ARM_IDS = [arm[0] for arm in ARMS]


def _encoder(block_type, block_config):
    model = EmbeddingEncoder(
        hidden_size=HIDDEN,
        num_layers=NUM_LAYERS,
        block_type=block_type,
        block_config=block_config,
        max_position_embeddings=64,
        hidden_dropout_rate=0.0,
    )
    model.build((None, SEQ_LEN))
    return model


def _raw_update(layer, x, block_type):
    """Apply the WRAPPED block directly, bypassing the wrapper's residual."""
    if block_type.startswith("convnext"):
        lifted = keras.ops.expand_dims(x, axis=1)
        return keras.ops.squeeze(layer.block(lifted, training=False), axis=1)
    return layer.block(x, training=False)


@pytest.fixture
def token_ids():
    rng = np.random.default_rng(0)
    return rng.integers(6, 101, size=(4, SEQ_LEN)).astype("int32")


def _rms(array) -> float:
    values = keras.ops.convert_to_numpy(array)
    return float(np.sqrt((values.astype("float64") ** 2).mean()))


@pytest.mark.parametrize("block_type,block_config", ARMS, ids=ARM_IDS)
def test_the_stack_does_not_annihilate_the_signal(
    block_type, block_config, token_ids
):
    """The load-bearing assertion: signal survives the block stack."""
    model = _encoder(block_type, block_config)
    embedded = model.embeddings(token_ids, training=False)
    hidden = model({"input_ids": token_ids}, training=False)["last_hidden_state"]

    ratio = _rms(hidden) / _rms(embedded)
    assert ratio > MIN_SURVIVING_RATIO, (
        f"[{block_type}] stack output RMS is {ratio:.3e} of the embedding RMS; "
        "the wrapped blocks are transform-only, so this is the signature of a "
        "missing external residual"
    )


@pytest.mark.parametrize("block_type,block_config", ARMS, ids=ARM_IDS)
def test_chaining_the_raw_blocks_would_annihilate_it(
    block_type, block_config, token_ids
):
    """The anti-vacuity control.

    Without this, the test above could pass against a stack that does nothing at
    all. Here the defect is reproduced directly -- the raw transform-only blocks
    chained without a residual -- which is what makes the bound a real
    discriminator rather than a formality.
    """
    model = _encoder(block_type, block_config)
    embedded = model.embeddings(token_ids, training=False)

    chained = embedded
    for layer in model.encoder_layers:
        chained = _raw_update(layer, chained, block_type)

    ratio = _rms(chained) / _rms(embedded)
    assert ratio < 0.1 * MIN_SURVIVING_RATIO, (
        f"[{block_type}] expected the residual-free chain to collapse, got "
        f"ratio {ratio:.3e}; if the block has started returning x + update, "
        "this guard and the wrapper both need rewriting"
    )


@pytest.mark.parametrize("block_type,block_config", ARMS, ids=ARM_IDS)
def test_the_layer_output_is_exactly_input_plus_update(
    block_type, block_config, token_ids
):
    """The residual is an ADD, asserted directly rather than inferred.

    At inference with drop_path_rate 0 the wrapper reduces to
    ``inputs + block(inputs)``, so the identity is exact, not approximate.
    """
    model = _encoder(block_type, block_config)
    layer = model.encoder_layers[0]

    rng = np.random.default_rng(1)
    x = keras.ops.convert_to_tensor(
        rng.standard_normal((2, SEQ_LEN, HIDDEN)).astype("float32")
    )
    out = keras.ops.convert_to_numpy(
        layer(x, attention_mask=None, layer_idx=0, training=False)
    )
    update = keras.ops.convert_to_numpy(_raw_update(layer, x, block_type))
    expected = keras.ops.convert_to_numpy(x) + update

    np.testing.assert_allclose(out, expected, atol=0, rtol=0)


@pytest.mark.parametrize("block_type,block_config", ARMS, ids=ARM_IDS)
def test_each_block_actually_changes_its_input(
    block_type, block_config, token_ids
):
    """A residual that adds exactly zero would also pass the magnitude bound."""
    model = _encoder(block_type, block_config)
    x = model.embeddings(token_ids, training=False)
    for layer in model.encoder_layers:
        out = layer(x, attention_mask=None, layer_idx=0, training=False)
        delta = float(
            np.abs(
                keras.ops.convert_to_numpy(out) - keras.ops.convert_to_numpy(x)
            ).max()
        )
        assert delta > 0.0, f"[{block_type}] block contributed exactly nothing"
        x = out


@pytest.mark.parametrize("block_type,block_config", ARMS, ids=ARM_IDS)
def test_gradients_reach_the_wrapped_block_weights(
    block_type, block_config, token_ids
):
    """A block whose weights get no gradient is decorative."""
    import tensorflow as tf

    model = _encoder(block_type, block_config)
    with tf.GradientTape() as tape:
        pooled = model({"input_ids": token_ids}, training=True)["pooled_output"]
        loss = keras.ops.mean(keras.ops.square(pooled))

    marker = {
        "clifford": "clifford_block",
        "convnext": "convnext_v1_block",
        "convnext_v2": "convnext_v2_block",
    }[block_type]
    block_weights = [w for w in model.trainable_weights if marker in w.path]
    assert block_weights, f"[{block_type}] no wrapped-block weights found"

    grads = tape.gradient(loss, block_weights)
    received = {
        w.path
        for w, g in zip(block_weights, grads)
        if g is not None
        and float(np.abs(keras.ops.convert_to_numpy(g)).max()) > 0.0
    }
    assert received == {w.path for w in block_weights}, (
        f"[{block_type}] these weights received no gradient: "
        f"{sorted({w.path for w in block_weights} - received)}"
    )
