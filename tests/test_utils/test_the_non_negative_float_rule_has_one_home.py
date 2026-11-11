"""One non-negative-number rule for a weight argument, imported by ConfidencePenaltyLoss.

Guard for plan-2026-11-11T131349-13b28b37 step 24.1 (review finding 10).
``losses/confidence_penalty_loss.py`` kept a private ``_check_real`` that was a
near-twin of ``dl_techniques.utils.validation.positive_float`` (the same bool,
``numbers.Real``, finite and NumPy-scalar rules) and differed only by accepting
0. It now binds ``dl_techniques.utils.validation.non_negative_float``, the
sibling of ``positive_float``, and the refusal table below holds through the
helper and through the loss's own entry point.
"""

import math

import numpy as np
import pytest

from dl_techniques.losses import confidence_penalty_loss
from dl_techniques.losses.confidence_penalty_loss import ConfidencePenaltyLoss
from dl_techniques.utils import validation

ACCEPTED = [(0, 0.0), (0.0, 0.0), (1, 1.0), (0.5, 0.5), (np.float32(2.0), 2.0),
            (np.int64(3), 3.0), (np.int32(0), 0.0), (1e-6, 1e-6)]
REFUSED = [-1.0, -1e-9, -1, math.nan, math.inf, -math.inf, True, False,
           np.bool_(True), np.float32("nan"), np.float32("inf"), "1.0", None,
           [0.5]]


def test_the_loss_binds_the_one_helper_and_keeps_no_private_copy():
    assert confidence_penalty_loss.non_negative_float is validation.non_negative_float
    assert not hasattr(confidence_penalty_loss, "_check_real")


@pytest.mark.parametrize("value, expected", ACCEPTED, ids=repr)
def test_accepted_values_come_back_as_python_floats(value, expected):
    got = validation.non_negative_float(value, "w", "Owner")
    assert got == expected and type(got) is float


@pytest.mark.parametrize("value", REFUSED, ids=repr)
def test_refused_values_name_the_owner_and_the_argument(value):
    with pytest.raises(ValueError, match=r"^Owner: w must be a non-negative"):
        validation.non_negative_float(value, "w", "Owner")


def test_zero_is_the_one_value_that_separates_it_from_positive_float():
    assert validation.non_negative_float(0, "w") == 0.0
    with pytest.raises(ValueError, match="positive"):
        validation.positive_float(0, "w")


@pytest.mark.parametrize("name", ["label_smoothing", "entropy_weight", "epsilon"])
@pytest.mark.parametrize("value", REFUSED, ids=repr)
def test_the_loss_refuses_each_value_naming_the_argument_and_the_owner(name, value):
    with pytest.raises(ValueError, match=rf"^ConfidencePenaltyLoss: {name} must be"):
        ConfidencePenaltyLoss(**{name: value})
