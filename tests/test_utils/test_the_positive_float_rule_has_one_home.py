"""One positive-number rule for a scale argument, imported by the poolers.

Guard for plan-2026-10-14T062616-81938e81 step 5.4 (D-203). Measured at
69b68fe5a: ``AttentionPooling(temperature=0.0)`` returned all NaN,
``WeightedPooling(temperature=-1.0)`` ran and inverted its weights, and
``SequencePooling(temperature=0.0)`` constructed; only the factory checked
``> 0``. The three poolers and the factory now bind
``dl_techniques.utils.validation.positive_float``, the sibling of
``positive_int``.
"""

import math

import numpy as np
import pytest

from dl_techniques.layers.sequence_pooling import (
    AttentionPooling,
    SequencePooling,
    WeightedPooling,
    create_sequence_pooling_layer,
)
from dl_techniques.layers.sequence_pooling import (
    attention_pooling,
    factory,
    sequence_pooling,
    weighted_pooling,
)
from dl_techniques.utils import validation

ACCEPTED = [(1, 1.0), (0.5, 0.5), (np.float32(2.0), 2.0), (np.int64(3), 3.0), (1e-6, 1e-6)]
REFUSED = [0, 0.0, -1.0, -1e-9, math.nan, math.inf, -math.inf, True, False,
           np.bool_(True), "1.0", None]


@pytest.mark.parametrize(
    "module", [attention_pooling, weighted_pooling, sequence_pooling, factory],
    ids=["attention", "weighted", "sequence", "factory"],
)
def test_each_pooling_module_binds_the_one_helper(module):
    assert module.positive_float is validation.positive_float


@pytest.mark.parametrize("value, expected", ACCEPTED, ids=repr)
def test_accepted_values_come_back_as_python_floats(value, expected):
    got = validation.positive_float(value, "t", "Owner")
    assert got == expected and type(got) is float


@pytest.mark.parametrize("value", REFUSED, ids=repr)
def test_refused_values_name_the_owner_and_the_argument(value):
    with pytest.raises(ValueError, match=r"^Owner: t must be a positive"):
        validation.positive_float(value, "t", "Owner")


ENTRIES = {
    "attention": lambda t: AttentionPooling(hidden_dim=4, temperature=t),
    "weighted": lambda t: WeightedPooling(max_seq_len=8, temperature=t),
    "sequence": lambda t: SequencePooling(strategy="mean", temperature=t),
    "factory": lambda t: create_sequence_pooling_layer("sequence", temperature=t),
}


@pytest.mark.parametrize("entry", sorted(ENTRIES))
@pytest.mark.parametrize("value", REFUSED, ids=repr)
def test_every_pooler_refuses_a_temperature_that_is_not_a_positive_number(entry, value):
    with pytest.raises(ValueError, match="temperature must be a positive"):
        ENTRIES[entry](value)


@pytest.mark.parametrize("entry", sorted(ENTRIES))
def test_every_pooler_stores_a_python_float(entry):
    layer = ENTRIES[entry](np.float32(0.5))
    assert type(layer.temperature) is float and layer.temperature == 0.5
