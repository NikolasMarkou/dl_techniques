"""The ``bias_initializer`` and ``bias_regularizer`` of the attention pooler take effect.

Guard for plan-2026-10-14T062616-81938e81 step 5.2 (review item SP-1, D-201).
``SequencePooling`` stored and serialized both knobs but never passed them on,
and ``AttentionPooling`` took neither. Measured at bfbded6f0:
``SequencePooling('attention', bias_initializer=Constant(0.7))`` built a Dense
bias of 0.0, and ``bias_regularizer=L2(1.0)`` added no loss.

``WeightedPooling`` and the ``weighted_sum`` aggregation weights have no bias,
so the knobs reach only the attention poolers; the last test pins that.
"""

import keras
import numpy as np
import pytest

from dl_techniques.layers.sequence_pooling import AttentionPooling, SequencePooling

D = 6
BIAS = 0.7


def _losses(layer) -> float:
    return float(sum(keras.ops.convert_to_numpy(l) for l in layer.losses))


@pytest.mark.parametrize("num_heads", [1, 3])
def test_the_attention_pooler_uses_its_bias_initializer(num_heads):
    layer = AttentionPooling(
        hidden_dim=4, num_heads=num_heads,
        bias_initializer=keras.initializers.Constant(BIAS),
    )
    layer.build((None, 5, D))
    bias = keras.ops.convert_to_numpy(layer.attention_dense.bias)
    np.testing.assert_array_equal(bias, np.full((4 * num_heads,), BIAS, "float32"))


@pytest.mark.parametrize("num_heads", [1, 3])
def test_the_attention_pooler_adds_its_bias_regularizer_loss(num_heads):
    layer = AttentionPooling(
        hidden_dim=4, num_heads=num_heads,
        bias_initializer=keras.initializers.Constant(BIAS),
        bias_regularizer=keras.regularizers.L2(1.0),
    )
    layer(np.zeros((2, 5, D), "float32"))
    expected = 1.0 * BIAS ** 2 * 4 * num_heads
    np.testing.assert_allclose(_losses(layer), expected, rtol=1e-6, atol=0)


def test_without_a_bias_there_is_no_bias_weight_and_no_loss():
    layer = AttentionPooling(
        hidden_dim=4, use_bias=False,
        bias_initializer=keras.initializers.Constant(BIAS),
        bias_regularizer=keras.regularizers.L2(1.0),
    )
    layer(np.zeros((2, 5, D), "float32"))
    assert layer.attention_dense.bias is None
    assert layer.losses == []


def test_the_sequence_pooler_forwards_both_knobs_to_every_attention_pooler():
    layer = SequencePooling(
        strategy=["attention", "multi_head_attention"],
        attention_hidden_dim=4,
        attention_num_heads=2,
        bias_initializer=keras.initializers.Constant(BIAS),
        bias_regularizer=keras.regularizers.L2(1.0),
    )
    layer(np.zeros((2, 5, D), "float32"))
    for name, width in (("attention", 4), ("multi_head_attention", 8)):
        bias = keras.ops.convert_to_numpy(
            layer.learnable_components[name].attention_dense.bias
        )
        np.testing.assert_array_equal(bias, np.full((width,), BIAS, "float32"))
    np.testing.assert_allclose(_losses(layer), BIAS ** 2 * (4 + 8), rtol=1e-6, atol=0)


def test_the_bias_knobs_survive_a_config_round_trip():
    layer = AttentionPooling(
        hidden_dim=4,
        bias_initializer=keras.initializers.Constant(BIAS),
        bias_regularizer=keras.regularizers.L2(0.5),
    )
    clone = AttentionPooling.from_config(layer.get_config())
    clone(np.zeros((2, 5, D), "float32"))
    np.testing.assert_array_equal(
        keras.ops.convert_to_numpy(clone.attention_dense.bias),
        np.full((4,), BIAS, "float32"),
    )
    np.testing.assert_allclose(_losses(clone), 0.5 * BIAS ** 2 * 4, rtol=1e-6, atol=0)


def test_weighted_and_the_aggregation_weights_take_no_bias():
    layer = SequencePooling(
        strategy=["mean", "weighted"],
        aggregation_method="weighted_sum",
        weighted_max_seq_len=8,
        bias_initializer=keras.initializers.Constant(BIAS),
        bias_regularizer=keras.regularizers.L2(1.0),
    )
    layer(np.zeros((2, 5, D), "float32"))
    assert not [w.path for w in layer.weights if "bias" in w.path]
    assert layer.losses == []
