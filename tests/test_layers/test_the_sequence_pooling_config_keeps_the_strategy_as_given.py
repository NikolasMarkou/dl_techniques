"""``SequencePooling.get_config`` stores ``strategy`` exactly as it was given.

Guard for plan-2026-10-14T062616-81938e81 step 5.4 (D-123 item d, D-203).
Measured at 69b68fe5a: ``SequencePooling('mean').get_config()['strategy']`` was
``['mean']``, so a config written from a string strategy differed from the one
the caller wrote. ``layer.strategy`` stays the list form the layer runs on.
"""

import pytest

from dl_techniques.layers.sequence_pooling import SequencePooling


@pytest.mark.parametrize(
    "given", ["mean", ["mean"], ["mean", "max"], "attention"], ids=repr
)
def test_the_config_holds_the_strategy_as_given(given):
    layer = SequencePooling(strategy=given, attention_hidden_dim=4)
    config = layer.get_config()
    assert config["strategy"] == given
    assert type(config["strategy"]) is type(given)
    clone = SequencePooling.from_config(config)
    assert clone.get_config()["strategy"] == given
    assert clone.strategy == (given if isinstance(given, list) else [given])
