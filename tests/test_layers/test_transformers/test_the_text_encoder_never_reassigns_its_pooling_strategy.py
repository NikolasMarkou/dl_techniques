"""``TextEncoder``'s feature helpers never reassign the pooling layer's strategy.

Guard for plan-2026-10-14T062616-81938e81 step 5.6 (D-205). ``get_sequence_features``
and ``get_pooled_features`` swapped ``self.pooling_layer.strategy`` for the
call and restored it afterwards. Measured at 69b68fe5a:
``TextEncoder(output_mode='none').get_pooled_features(ids, 'attention')`` (and
on a ``'mean'`` encoder) raised ``KeyError: 'attention'``, because the swapped
strategy had no learnable component; the swap also made the layer's runtime
differ from its config for the duration of the call, and is not trace-safe.

Now a stateless strategy runs a throwaway ``SequencePooling``; a learnable one
is answered only by the pooler built at construction, and any other learnable
request raises a ``ValueError`` naming the build-time ``output_mode``.
"""

import typing

import keras
import numpy as np
import pytest

from dl_techniques.layers.sequence_pooling import SequencePooling
from dl_techniques.layers.sequence_pooling.sequence_pooling import PoolingStrategy
from dl_techniques.layers.transformers.text_encoder import TextEncoder

STRATEGIES = typing.get_args(PoolingStrategy)
LEARNABLE = {"attention", "multi_head_attention", "weighted"}
IDS = np.array([[5, 7, 9, 0], [3, 4, 0, 0]], "int32")
MASK = np.array([[1, 1, 1, 0], [1, 1, 0, 0]], "int32")


def _encoder(output_mode="mean", use_cls_token=False):
    keras.utils.set_random_seed(0)
    encoder = TextEncoder(
        vocab_size=32, embed_dim=16, depth=1, num_heads=2, max_seq_len=8,
        output_mode=output_mode, use_cls_token=use_cls_token,
    )
    encoder(IDS)
    return encoder


@pytest.mark.parametrize("strategy", sorted(set(STRATEGIES) - LEARNABLE - {"cls"}))
def test_a_stateless_strategy_pools_the_sequence_features(strategy):
    encoder = _encoder("mean")
    features = encoder.get_sequence_features(IDS, attention_mask=MASK)
    expected = SequencePooling(strategy=strategy)(features, mask=MASK)
    got = encoder.get_pooled_features(IDS, strategy, attention_mask=MASK)
    np.testing.assert_allclose(np.asarray(got), np.asarray(expected), rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("output_mode", ["none", "mean"])
@pytest.mark.parametrize("strategy", sorted(LEARNABLE))
def test_a_learnable_strategy_not_built_raises_a_value_error(output_mode, strategy):
    encoder = _encoder(output_mode)
    with pytest.raises(ValueError, match=f"output_mode='{output_mode}'"):
        encoder.get_pooled_features(IDS, strategy)


@pytest.mark.parametrize("strategy", sorted(LEARNABLE))
def test_the_built_learnable_strategy_is_the_encoders_own_output(strategy):
    encoder = _encoder(strategy)
    np.testing.assert_array_equal(
        np.asarray(encoder.get_pooled_features(IDS, strategy, attention_mask=MASK)),
        np.asarray(encoder(IDS, attention_mask=MASK)),
    )


def test_cls_without_a_cls_token_is_refused_as_at_construction():
    with pytest.raises(ValueError, match="use_cls_token"):
        _encoder("mean").get_pooled_features(IDS, "cls")


def test_an_unknown_strategy_is_refused():
    with pytest.raises(ValueError, match="bogus"):
        _encoder("mean").get_pooled_features(IDS, "bogus")


def test_no_helper_assigns_the_pooling_strategy(monkeypatch):
    encoder = _encoder("attention")
    pooling = encoder.pooling_layer
    original = SequencePooling.__setattr__

    def guarded(self, name, value):
        if self is pooling and name == "strategy":
            raise AssertionError("TextEncoder reassigned pooling_layer.strategy")
        original(self, name, value)

    monkeypatch.setattr(SequencePooling, "__setattr__", guarded)
    encoder.get_sequence_features(IDS, attention_mask=MASK)
    encoder.get_pooled_features(IDS, "max", attention_mask=MASK)
    encoder.get_pooled_features(IDS, "attention", attention_mask=MASK)
    assert pooling.strategy == ["attention"]
    assert encoder.get_config()["output_mode"] == "attention"


def test_the_sequence_features_zero_the_padded_positions():
    features = np.asarray(_encoder("mean").get_sequence_features(IDS, attention_mask=MASK))
    assert features.shape == (2, 4, 16)
    np.testing.assert_array_equal(features[MASK == 0], 0.0)
    assert np.abs(features[MASK == 1]).min() > 0.0
