"""The Clifford stack must add its blocks' updates, not replace the signal.

``CliffordNetBlock.call`` returns ONLY the LayerScale-gated update, never
``x + update``. So ``x = block(x)`` does not apply a block -- it replaces the
representation with a residual scaled by ``layer_scale_init`` (1e-5 by
default). The failure is silent in every conventional sense: shapes match,
outputs are finite, ``get_config`` round-trips, gradients exist. Only the
MAGNITUDE moves.

Measured on this model at ``hidden_size=64``, 4 Clifford blocks, embeddings
normalized to RMS 1.0:

    with the external residual      output RMS 1.000020   (ratio 1.000020)
    without it (updates chained)    output RMS 6.996e-13  (ratio 6.996e-13)

a collapse of roughly twelve orders of magnitude. The assertion below is a
magnitude bound, because a shape, finiteness or round-trip oracle passes
through that collapse unchanged -- which is precisely how this class of defect
survives an otherwise thorough suite.
"""

import keras
import numpy as np
import pytest

from dl_techniques.models.embeddings_experimental.shared import EmbeddingEncoder

SEQ_LEN = 32
HIDDEN = 64
NUM_LAYERS = 4

#: Ratio of stack-output RMS to embedding RMS below which the signal is
#: considered annihilated. The healthy value measures ~1.0 and the defective
#: one ~7e-13, so this bound sits ~12 orders from the defect and a factor of
#: two from health.
MIN_SURVIVING_RATIO = 0.5


def _encoder(**overrides):
    config = dict(
        hidden_size=HIDDEN,
        num_layers=NUM_LAYERS,
        block_type="clifford",
        block_config={"shifts": [1, 2], "context_kernel_size": 7},
        max_position_embeddings=64,
        hidden_dropout_rate=0.0,
    )
    config.update(overrides)
    model = EmbeddingEncoder(**config)
    model.build((None, SEQ_LEN))
    return model


@pytest.fixture
def token_ids():
    rng = np.random.default_rng(0)
    return rng.integers(6, 101, size=(4, SEQ_LEN)).astype("int32")


def _rms(array) -> float:
    values = keras.ops.convert_to_numpy(array)
    return float(np.sqrt((values.astype("float64") ** 2).mean()))


def test_the_stack_does_not_annihilate_the_signal(token_ids):
    """The load-bearing assertion: signal survives the block stack."""
    model = _encoder()
    embedded = model.embeddings(token_ids, training=False)
    hidden = model({"input_ids": token_ids}, training=False)["last_hidden_state"]

    ratio = _rms(hidden) / _rms(embedded)
    assert ratio > MIN_SURVIVING_RATIO, (
        f"stack output RMS is {ratio:.3e} of the embedding RMS; the blocks are "
        "transform-only, so this is the signature of a missing external residual"
    )


def test_chaining_the_raw_blocks_would_annihilate_it(token_ids):
    """The anti-vacuity control.

    Without this, the test above could pass against a stack that does nothing
    at all. Here the defect is reproduced directly -- the raw transform-only
    blocks chained without a residual -- and the collapse is asserted, which is
    what makes the bound above a real discriminator rather than a formality.
    """
    model = _encoder()
    embedded = model.embeddings(token_ids, training=False)

    chained = embedded
    for layer in model.encoder_layers:
        chained = layer.block(chained, training=False)

    ratio = _rms(chained) / _rms(embedded)
    assert ratio < 1e-6, (
        f"expected the residual-free chain to collapse, got ratio {ratio:.3e}; "
        "if the block ever starts returning x + update, this guard and the "
        "wrapper both need rewriting"
    )


def test_each_block_actually_changes_its_input(token_ids):
    """A residual that adds exactly zero would also pass the magnitude bound."""
    model = _encoder()
    x = model.embeddings(token_ids, training=False)
    for layer in model.encoder_layers:
        out = layer(x, attention_mask=None, layer_idx=0, training=False)
        delta = float(
            np.abs(
                keras.ops.convert_to_numpy(out) - keras.ops.convert_to_numpy(x)
            ).max()
        )
        assert delta > 0.0, "block contributed exactly nothing"
        x = out


def test_gradients_reach_the_clifford_block_weights(token_ids):
    """A block whose weights get no gradient is decorative."""
    import tensorflow as tf

    model = _encoder()
    with tf.GradientTape() as tape:
        pooled = model({"input_ids": token_ids}, training=True)["pooled_output"]
        loss = keras.ops.mean(keras.ops.square(pooled))

    block_weights = [
        w
        for w in model.trainable_weights
        if "clifford_block" in w.path
    ]
    assert block_weights, "no Clifford block weights found"

    grads = tape.gradient(loss, block_weights)
    received = [
        w.path
        for w, g in zip(block_weights, grads)
        if g is not None and float(np.abs(keras.ops.convert_to_numpy(g)).max()) > 0.0
    ]
    assert len(received) == len(block_weights), (
        "these Clifford weights received no gradient: "
        f"{sorted(set(w.path for w in block_weights) - set(received))}"
    )
