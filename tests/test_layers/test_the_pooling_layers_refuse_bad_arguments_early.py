"""The pooling layers refuse a bad count, rate or position at construction.

Guard for plan-2026-10-14T062616-81938e81 step 5.4 (D-203). Measured at
69b68fe5a: ``AttentionPooling(hidden_dim=0)``, ``AttentionPooling(num_heads=0)``
and ``WeightedPooling(max_seq_len=0)`` constructed; ``SequencePooling('top_k_mean',
top_k=0)`` returned all NaN; ``exclude_positions=[-1]`` was a silent no-op
(``arange(seq_len) != -1`` is always true) and ``[1, 1]``, ``['a']`` and
``[1.5]`` were stored as given; a negative ``dropout_rate`` silently disabled
dropout. Negative indexing is refused, not widened (D-123).
"""

import numpy as np
import pytest

from dl_techniques.layers.sequence_pooling import (
    AttentionPooling,
    SequencePooling,
    WeightedPooling,
)

BAD_COUNTS = [0, -1, 2.0, True, None, "4", 2 ** 31]

COUNT_ENTRIES = {
    "attention.hidden_dim": lambda v: AttentionPooling(hidden_dim=v),
    "attention.num_heads": lambda v: AttentionPooling(num_heads=v),
    "weighted.max_seq_len": lambda v: WeightedPooling(max_seq_len=v),
    "sequence.attention_hidden_dim": lambda v: SequencePooling(attention_hidden_dim=v),
    "sequence.attention_num_heads": lambda v: SequencePooling(attention_num_heads=v),
    "sequence.weighted_max_seq_len": lambda v: SequencePooling(weighted_max_seq_len=v),
    "sequence.top_k": lambda v: SequencePooling(strategy="top_k_mean", top_k=v),
}


@pytest.mark.parametrize("entry", sorted(COUNT_ENTRIES))
@pytest.mark.parametrize("value", BAD_COUNTS, ids=repr)
def test_a_count_that_is_not_a_positive_int_is_refused(entry, value):
    with pytest.raises(ValueError, match="must be"):
        COUNT_ENTRIES[entry](value)


@pytest.mark.parametrize("entry", sorted(COUNT_ENTRIES))
def test_a_numpy_count_is_stored_as_a_python_int(entry):
    layer = COUNT_ENTRIES[entry](np.int64(3))
    name = entry.split(".")[1]
    value = getattr(layer, name)
    assert value == 3 and type(value) is int


@pytest.mark.parametrize("rate", [-0.1, 1.0, 1.5])
@pytest.mark.parametrize("cls", [AttentionPooling, WeightedPooling])
def test_a_dropout_rate_outside_zero_to_one_is_refused(cls, rate):
    with pytest.raises(ValueError, match="dropout_rate"):
        cls(dropout_rate=rate)


@pytest.mark.parametrize(
    "positions, match",
    [
        ([-1], "negative"),
        ([0, -2], "negative"),
        ([1, 1], "more than once"),
        (["a"], "must be an int"),
        ([1.5], "must be an int"),
        ([True], "must be an int"),
        ([np.bool_(False)], "must be an int"),
        (3, "must be a list"),
    ],
    ids=["neg", "neg_in_list", "repeat", "str", "float", "bool", "np_bool", "scalar"],
)
def test_a_bad_exclude_positions_is_refused(positions, match):
    with pytest.raises(ValueError, match=match):
        SequencePooling(strategy="mean", exclude_positions=positions)


def test_good_exclude_positions_are_stored_as_python_ints_and_take_effect():
    layer = SequencePooling(strategy="mean", exclude_positions=(np.int64(0), 2))
    assert layer.exclude_positions == [0, 2]
    assert all(type(p) is int for p in layer.exclude_positions)
    x = np.random.default_rng(0).normal(size=(2, 5, 4)).astype("float32")
    np.testing.assert_allclose(
        np.asarray(layer(x)), x[:, [1, 3, 4]].mean(axis=1), rtol=1e-6, atol=1e-6
    )
    assert SequencePooling(strategy="mean").exclude_positions == []


def test_top_k_one_is_the_smallest_accepted_and_is_finite():
    x = np.random.default_rng(0).normal(size=(2, 5, 4)).astype("float32")
    out = np.asarray(SequencePooling(strategy="top_k_mean", top_k=1)(x))
    assert np.isfinite(out).all()
