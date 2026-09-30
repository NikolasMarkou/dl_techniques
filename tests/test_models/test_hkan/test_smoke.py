"""Build and forward smoke test of HKAN, with its assertion proven able to fail.

Guards the forward contract of the package: ``HKAN`` maps ``(B, n_in)`` to one
finite float tensor of shape ``(B, 1)``, from the unfitted state and from the
fitted one. The contract function is handed to the shared
``smoke_contract_oracle``, which breaks the model's own forward output three
ways (collapsed to a scalar, leading axis sliced, a trailing axis appended)
and requires the contract to reject each with an ``AssertionError``.
"""

import numpy as np
import pytest

from ..smoke_contract_oracle import (
    assert_contract_rejects_a_broken_forward,
    assert_finite,
)
from . import HIDDEN, N_IN, NUM_BASIS, make_data

BATCH = 4


def _build(**overrides):
    from dl_techniques.models.general_purpose.hkan import create_hkan

    config = dict(hidden_units=(HIDDEN,), num_basis=NUM_BASIS, basis="tanh",
                  slope=5.0, seed=0)
    config.update(overrides)
    return create_hkan(**config)


def _inputs() -> np.ndarray:
    return np.random.default_rng(0).uniform(0.0, 1.0, size=(BATCH, N_IN)).astype("float32")


def _assert_contract(out) -> None:
    """The smoke assertion, shared with the meta-test below."""
    assert not isinstance(out, (dict, list, tuple)), (
        f"HKAN returns a single tensor, got {type(out)}")
    assert tuple(out.shape) == (BATCH, 1), (
        f"HKAN maps (B, n_in) to (B, 1); got {tuple(out.shape)}")
    assert_finite(out)


@pytest.mark.parametrize("training", [False, True])
def test_smoke_build_and_forward(training):
    _assert_contract(_build()(_inputs(), training=training))


def test_smoke_forward_of_a_built_but_uncalled_model():
    model = _build(input_dim=N_IN)
    assert model.built
    _assert_contract(model(_inputs(), training=False))


def test_smoke_forward_after_a_closed_form_fit():
    model = _build(l2_block=0.01)
    model.fit_closed_form(*make_data())
    _assert_contract(model(_inputs(), training=False))


def test_the_smoke_contract_rejects_a_broken_forward():
    rejections = assert_contract_rejects_a_broken_forward(
        _build(), _inputs(), _assert_contract)
    assert set(rejections) == {
        "collapse_to_scalar", "slice_leading_axis", "append_trailing_axis"}
