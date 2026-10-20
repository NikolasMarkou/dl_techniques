"""``SequencePooling.compute_output_shape`` reports the shape ``call`` returns.

Guard for plan-2026-10-14T062616-81938e81 step 5.3 (review items SP-3 to SP-6,
D-202). Measured at 69b68fe5a:

- ``['mean', 'flatten']`` under ``concat`` called to ``(2, 48)`` while the shape
  function said ``(2, 16)``: it counted ``flatten`` as one ``hidden_dim``.
- ``['mean_max', 'mean']`` under ``add`` failed inside ``call`` with a backend
  broadcast error while the shape function said ``(2, 8)``; ``['mean', 'none']``
  under ``add``, ``multiply`` and ``weighted_sum`` failed the same way.
- ``SequencePooling('mean', aggregation_method='invalid')`` and
  ``SequencePooling('bogus')`` constructed; the second failed only at ``call``.
- ``WeightedPooling(max_seq_len=4)`` on a length-5 input failed with a backend
  ``Incompatible shapes`` error that did not name ``max_seq_len``.

Every configuration the constructor accepts is checked here against ``call``,
at a static and at an unknown sequence length; every configuration it refuses is
a named arm.
"""

import itertools
import typing

import keras
import numpy as np
import pytest
import tensorflow as tf

from dl_techniques.layers.sequence_pooling import SequencePooling, WeightedPooling
from dl_techniques.layers.sequence_pooling.sequence_pooling import (
    AggregationMethod,
    PoolingStrategy,
)

STRATEGIES = typing.get_args(PoolingStrategy)
AGGREGATIONS = typing.get_args(AggregationMethod)
B, S, D = 2, 5, 8
WIDTH = {"mean_max": 2, "mean_std": 2, "mean_max_min": 3}


def _width(strategy):
    if strategy == "flatten":
        return "flatten"
    return WIDTH.get(strategy, 1)


def _accepted(strategies, aggregation):
    if len(strategies) == 1:
        return True
    if "none" in strategies or len(set(strategies)) != len(strategies):
        return False
    if aggregation == "concat":
        return True
    widths = {_width(s) for s in strategies}
    return len(widths) == 1 and "flatten" not in widths


def _layer(strategies, aggregation):
    return SequencePooling(
        strategy=list(strategies),
        aggregation_method=aggregation,
        attention_hidden_dim=4,
        attention_num_heads=2,
        weighted_max_seq_len=16,
        top_k=3,
    )


CASES = [((s,), "concat") for s in STRATEGIES] + [
    (pair, agg)
    for pair in itertools.combinations(STRATEGIES, 2)
    for agg in AGGREGATIONS
]
ACCEPTED = [c for c in CASES if _accepted(*c)]
REFUSED = [c for c in CASES if not _accepted(*c)]


def _id(case):
    strategies, aggregation = case
    return "+".join(strategies) + ":" + aggregation


def test_the_case_split_is_not_vacuous():
    assert len(CASES) == 18 + 153 * 4
    assert len(ACCEPTED) > 300 and len(REFUSED) > 200


@pytest.mark.parametrize("case", ACCEPTED, ids=[_id(c) for c in ACCEPTED])
def test_the_shape_function_matches_call(case):
    layer = _layer(*case)
    x = np.random.default_rng(0).normal(size=(B, S, D)).astype("float32")
    out = layer(x)
    assert tuple(layer.compute_output_shape((B, S, D))) == tuple(out.shape)

    symbolic = keras.Input((None, D))
    traced = _layer(*case)(symbolic)
    assert tuple(_layer(*case).compute_output_shape((None, None, D))) == tuple(
        traced.shape
    )


@pytest.mark.parametrize("case", REFUSED, ids=[_id(c) for c in REFUSED])
def test_a_configuration_call_cannot_run_is_refused_at_construction(case):
    strategies, aggregation = case
    with pytest.raises(ValueError, match="SequencePooling"):
        _layer(strategies, aggregation)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"strategy": "bogus"}, "unknown pooling strategy"),
        ({"strategy": ["mean", "bogus"]}, "unknown pooling strategy"),
        ({"strategy": "mean", "aggregation_method": "invalid"}, "aggregation_method"),
        ({"strategy": ["mean", "max"], "aggregation_method": "invalid"}, "aggregation_method"),
        ({"strategy": ["mean", "none"], "aggregation_method": "concat"}, "'none'"),
        ({"strategy": ["mean", "none"], "aggregation_method": "add"}, "'none'"),
        ({"strategy": ["mean_max", "mean"], "aggregation_method": "add"}, "width"),
        ({"strategy": ["mean", "flatten"], "aggregation_method": "multiply"}, "width"),
        ({"strategy": ["mean", "mean"], "aggregation_method": "concat"}, "more than once"),
        ({"strategy": []}, "at least one"),
    ],
    ids=[
        "unknown_strategy", "unknown_in_list", "aggregation_single",
        "aggregation_list", "none_concat", "none_add", "width_add",
        "flatten_multiply", "duplicate", "empty",
    ],
)
def test_the_named_refusals(kwargs, match):
    with pytest.raises(ValueError, match=match):
        SequencePooling(**kwargs)


def test_flatten_at_an_unknown_length_reports_an_unknown_width():
    layer = SequencePooling(strategy=["mean", "flatten"])
    assert tuple(layer.compute_output_shape((None, None, D))) == (None, None)
    assert tuple(layer.compute_output_shape((None, S, D))) == (None, D + S * D)


def test_a_static_length_over_max_seq_len_names_max_seq_len():
    x = np.zeros((B, 5, D), "float32")
    with pytest.raises(ValueError, match="max_seq_len"):
        WeightedPooling(max_seq_len=4)(x)
    with pytest.raises(ValueError, match="max_seq_len"):
        SequencePooling(strategy="weighted", weighted_max_seq_len=4)(x)
    with pytest.raises(ValueError, match="max_seq_len"):
        WeightedPooling(max_seq_len=4)(keras.Input((5, D)))
    # The boundary is allowed.
    assert tuple(WeightedPooling(max_seq_len=5)(x).shape) == (B, D)


def test_an_unknown_length_over_max_seq_len_fails_at_run_time():
    """The documented behaviour for a symbolic length: no assert is traced, so
    the backend refuses the broadcast when the run reaches a longer input. A
    shorter input runs."""
    inputs = keras.Input((None, D))
    model = keras.Model(inputs, WeightedPooling(max_seq_len=4)(inputs))
    assert model.predict(np.zeros((B, 3, D), "float32"), verbose=0).shape == (B, D)
    with pytest.raises(tf.errors.InvalidArgumentError, match="Incompatible shapes"):
        model.predict(np.zeros((B, 5, D), "float32"), verbose=0)
