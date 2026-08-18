"""RED proofs for the smoke-contract oracle itself.

``smoke_contract_oracle.assert_contract_rejects_a_broken_forward`` is the
instrument the de-fanged smoke tests lean on. An instrument that reports "your
guard is strong" for a guard that is actually vacuous would re-create, one level
up, exactly the defect it exists to catch -- so it gets its own guards, on tiny
throwaway models with no package dependency.

The cases below pin all three verdicts the instrument can return:

* a REAL contract on a working model              -> passes
* a VACUOUS contract (asserts nothing)            -> the instrument raises
* a contract that CRASHES rather than judges      -> the instrument raises, and
  says so in different words than the vacuous case
* a contract that rejects the real output          -> the instrument raises
  before it ever reaches a breaker
"""

import keras
import numpy as np
import pytest
from keras import ops

from .smoke_contract_oracle import (
    DEFAULT_BREAKERS,
    append_trailing_axis,
    assert_contract_rejects_a_broken_forward,
    assert_finite,
    broken_forward,
    collapse_to_scalar,
    slice_leading_axis,
)


def _tensor_model():
    """A functional model returning one ``(B, 4)`` tensor."""
    inp = keras.Input(shape=(8,))
    return keras.Model(inp, keras.layers.Dense(4)(inp), name="tensor_model")


class _DictModel(keras.Model):
    """A SUBCLASSED model returning a dict with a ``None`` value.

    The ``None`` mirrors ``DistilBERT``'s measured output
    ``{"last_hidden_state": ..., "attention_mask": None}``; the subclassing
    mirrors the half of the tree that is not functional. Both container shapes
    must survive the injection.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dense = keras.layers.Dense(4)

    def call(self, inputs, training=None):
        return {"hidden": self.dense(inputs), "mask": None}


def _x():
    return np.random.rand(2, 8).astype("float32")


# ---------------------------------------------------------------------------
# The instrument accepts a real contract
# ---------------------------------------------------------------------------
def test_a_real_shape_contract_is_accepted_on_a_functional_model():
    model = _tensor_model()

    def contract(out):
        assert not isinstance(out, (dict, list, tuple)), f"expected a tensor, got {type(out)}"
        assert tuple(out.shape) == (2, 4), tuple(out.shape)

    rejections = assert_contract_rejects_a_broken_forward(model, _x(), contract)
    assert set(rejections) == {breaker.__name__ for breaker in DEFAULT_BREAKERS}


def test_a_real_shape_contract_is_accepted_on_a_subclassed_model_returning_none():
    model = _DictModel()

    def contract(out):
        assert isinstance(out, dict), f"expected a dict, got {type(out)}"
        assert set(out) == {"hidden", "mask"}, sorted(out)
        assert tuple(out["hidden"].shape) == (2, 4), tuple(out["hidden"].shape)

    rejections = assert_contract_rejects_a_broken_forward(model, _x(), contract)
    assert len(rejections) == len(DEFAULT_BREAKERS)


# ---------------------------------------------------------------------------
# The instrument rejects a weak contract -- the whole reason it exists
# ---------------------------------------------------------------------------
def test_a_contract_that_asserts_nothing_is_reported_as_not_rejecting():
    """The vacuous case: exactly what the finiteness-only smoke tests were."""
    model = _tensor_model()

    def vacuous_contract(out):
        assert out is not None

    with pytest.raises(AssertionError, match=r"did NOT make the contract fail"):
        assert_contract_rejects_a_broken_forward(model, _x(), vacuous_contract)


def test_a_finiteness_only_contract_is_reported_as_not_rejecting():
    """`0.0` is finite, so finiteness alone accepts the collapsed output."""
    model = _tensor_model()

    with pytest.raises(AssertionError, match=r"collapse_to_scalar.*did NOT"):
        assert_contract_rejects_a_broken_forward(
            model, _x(), assert_finite, breakers=(collapse_to_scalar,)
        )


def test_a_partial_shape_contract_survives_the_leading_axis_but_not_the_trailing_one():
    """`shape[:1]`-style contracts are the yolo12 `shape[:3]` defect in miniature."""
    model = _tensor_model()

    def partial_contract(out):
        assert not isinstance(out, (dict, list, tuple)), f"got {type(out)}"
        assert tuple(out.shape[:1]) == (2,), tuple(out.shape)

    # It DOES catch the leading-axis slice ...
    assert_contract_rejects_a_broken_forward(
        model, _x(), partial_contract, breakers=(collapse_to_scalar, slice_leading_axis)
    )
    # ... and does NOT catch a wrong trailing dimension.
    with pytest.raises(AssertionError, match=r"append_trailing_axis.*did NOT"):
        assert_contract_rejects_a_broken_forward(
            model, _x(), partial_contract, breakers=(append_trailing_axis,)
        )


def test_a_contract_that_crashes_is_distinguished_from_one_that_judges():
    """A `TypeError` from indexing a scalar is the contract crashing, not judging."""
    model = _DictModel()

    def unguarded_contract(out):
        assert tuple(out["hidden"].shape) == (2, 4)

    with pytest.raises(AssertionError, match=r"that is the contract CRASHING"):
        assert_contract_rejects_a_broken_forward(
            model, _x(), unguarded_contract, breakers=(collapse_to_scalar,)
        )


def test_a_contract_that_rejects_the_real_output_fails_the_control():
    """Without this control an always-raising contract would look maximally strong."""
    model = _tensor_model()

    def always_raises(out):
        raise AssertionError("this contract is simply wrong")

    with pytest.raises(AssertionError, match=r"this contract is simply wrong"):
        assert_contract_rejects_a_broken_forward(model, _x(), always_raises)


def test_an_empty_breaker_list_is_refused():
    model = _tensor_model()
    with pytest.raises(ValueError, match=r"EMPTY breaker list"):
        assert_contract_rejects_a_broken_forward(
            model, _x(), lambda out: None, breakers=()
        )


# ---------------------------------------------------------------------------
# The injection itself
# ---------------------------------------------------------------------------
def test_broken_forward_restores_the_original_call():
    model = _tensor_model()
    x = _x()
    before = ops.convert_to_numpy(model(x, training=False))
    with broken_forward(model, collapse_to_scalar):
        assert tuple(model(x, training=False).shape) == ()
    after = ops.convert_to_numpy(model(x, training=False))
    np.testing.assert_allclose(before, after, rtol=0, atol=0)
    assert "call" not in model.__dict__, "the instance attribute was not removed"


def test_slice_leading_axis_refuses_a_rank_zero_leaf():
    """A breaker that is a no-op would make the guard look weak for free."""
    with pytest.raises(ValueError, match=r"rank-0 leaf"):
        slice_leading_axis(ops.convert_to_tensor(1.0))


def test_assert_finite_catches_nan_and_inf_through_a_dict_with_none():
    assert_finite({"a": ops.convert_to_tensor([1.0, 2.0]), "b": None})
    with pytest.raises(AssertionError, match="NaN"):
        assert_finite({"a": ops.convert_to_tensor([float("nan")]), "b": None})
    with pytest.raises(AssertionError, match="inf"):
        assert_finite([ops.convert_to_tensor([float("inf")])])
