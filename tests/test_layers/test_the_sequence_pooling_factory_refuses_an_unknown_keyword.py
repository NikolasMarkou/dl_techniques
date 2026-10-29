"""The sequence pooling factory refuses a keyword its type does not declare.

Guard for plan-2026-10-14T062616-81938e81 step 5.1 (review item SP-2, D-200).
Before the fix ``create_sequence_pooling_layer`` filtered the caller's kwargs
against the registry and dropped the rest without a word, so
``create_sequence_pooling_layer('attention', dropout=0.5)`` built a pooler with
``dropout_rate=0.0``, and ``validate_sequence_pooling_config`` accepted any key.
Every other registry factory raises (``layers/AGENTS.md``, "The factory
contract", rule 1).

The factory also checks ``strategy`` and ``aggregation_method`` names for the
``'sequence'`` type (the factory half of SP-5): an unknown name used to build
and fail only at ``call``.
"""

import re
import typing

import pytest

from dl_techniques.layers.sequence_pooling import (
    SEQUENCE_POOLING_REGISTRY,
    STRICT_DROPPED_KEY_MARKER,
    create_sequence_pooling_from_config,
    create_sequence_pooling_layer,
    validate_sequence_pooling_config,
)
from dl_techniques.layers.sequence_pooling.sequence_pooling import (
    AggregationMethod,
    PoolingStrategy,
)

TYPES = sorted(SEQUENCE_POOLING_REGISTRY)
MARKER = re.escape(STRICT_DROPPED_KEY_MARKER)


def test_the_registry_has_its_four_types():
    assert TYPES == ["attention", "mode", "sequence", "weighted"]


@pytest.mark.parametrize("pooling_type", TYPES)
def test_create_refuses_a_bogus_keyword(pooling_type):
    with pytest.raises(ValueError, match=MARKER) as exc:
        create_sequence_pooling_layer(pooling_type, bogus_key=1)
    assert "bogus_key" in str(exc.value)


@pytest.mark.parametrize("pooling_type", TYPES)
def test_validate_refuses_a_bogus_keyword(pooling_type):
    with pytest.raises(ValueError, match=MARKER):
        validate_sequence_pooling_config(pooling_type, bogus_key=1)


@pytest.mark.parametrize("pooling_type", TYPES)
def test_create_from_config_refuses_a_bogus_keyword(pooling_type):
    with pytest.raises(ValueError, match=MARKER):
        create_sequence_pooling_from_config({"type": pooling_type, "bogus_key": 1})


def test_a_misspelt_dropout_is_refused_not_dropped():
    """``dropout=`` for ``dropout_rate=`` used to build with a rate of 0.0."""
    with pytest.raises(ValueError, match=MARKER) as exc:
        create_sequence_pooling_layer("attention", hidden_dim=8, dropout=0.5)
    assert "dropout" in str(exc.value)
    layer = create_sequence_pooling_layer("attention", hidden_dim=8, dropout_rate=0.5)
    assert layer.dropout_rate == 0.5


@pytest.mark.parametrize("pooling_type", TYPES)
def test_the_keras_base_keywords_still_build(pooling_type):
    layer = create_sequence_pooling_layer(
        pooling_type, name="pool_x", dtype="float32", trainable=False
    )
    assert layer.name == "pool_x"
    assert layer.trainable is False
    from_config = create_sequence_pooling_from_config(
        {"type": pooling_type, "name": "pool_y"}
    )
    assert from_config.name == "pool_y"


@pytest.mark.parametrize("pooling_type", TYPES)
def test_every_declared_parameter_is_accepted(pooling_type):
    info = SEQUENCE_POOLING_REGISTRY[pooling_type]
    params = dict(info["optional_params"])
    validate_sequence_pooling_config(pooling_type, **params)
    layer = create_sequence_pooling_layer(pooling_type, **params)
    assert isinstance(layer, info["class"])


@pytest.mark.parametrize(
    "kwargs, bad",
    [
        ({"strategy": "bogus"}, "bogus"),
        ({"strategy": ["mean", "bogus"]}, "bogus"),
        ({"strategy": ["mean", "max"], "aggregation_method": "bogus"}, "bogus"),
        ({"strategy": "mean", "aggregation_method": "bogus"}, "bogus"),
    ],
    ids=["single", "in_list", "aggregation", "aggregation_single"],
)
def test_the_factory_refuses_an_unknown_strategy_or_aggregation(kwargs, bad):
    with pytest.raises(ValueError, match=bad):
        validate_sequence_pooling_config("sequence", **kwargs)
    with pytest.raises(ValueError, match=bad):
        create_sequence_pooling_layer("sequence", **kwargs)


def test_every_strategy_and_aggregation_name_is_accepted():
    for strategy in typing.get_args(PoolingStrategy):
        validate_sequence_pooling_config("sequence", strategy=strategy)
    for method in typing.get_args(AggregationMethod):
        validate_sequence_pooling_config(
            "sequence", strategy=["mean", "max"], aggregation_method=method
        )
