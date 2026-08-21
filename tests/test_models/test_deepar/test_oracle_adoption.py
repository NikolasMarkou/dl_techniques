"""
Oracle adoption for ``models/time_series/deepar`` -- Phase 5 batch A.

Zero adoption of the three shared instruments before this file. All three are
adopted; no new oracle is authored.

Measured 2026-08-21 (CPU) on ``DeepAR(num_layers=1, hidden_dim=8,
likelihood='gaussian', target_dim=2, num_samples=4)`` after one real optimizer
step: **7** trainable weights, **0** dead, **0** non-finite -- the LSTM cell's
kernel / recurrent_kernel / bias plus the Gaussian head's mu and sigma
projections.

The ``sigma_projection`` pair is the reason a per-weight assertion is worth
having here. A likelihood head whose scale parameter is emitted but never enters
the loss -- or is replaced by a constant somewhere downstream -- leaves ``mu``
training normally and every output key present, finite and correctly shaped.
"""

import numpy as np
import keras
import pytest
import tensorflow as tf

from dl_techniques.models.time_series.deepar.model import DeepAR

from ..gradient_flow_oracle import (
    assert_gradients_reach_every_trainable_weight,
    default_loss,
    stop_all_gradients,
)
from ..knob_sensitivity_oracle import assert_structural_knob_changes_weights
from ..smoke_contract_oracle import (
    assert_contract_rejects_a_broken_forward,
    assert_finite,
    broken_forward,
)

T, D, C, S = 16, 2, 3, 4

#: Measured 2026-08-21 at num_layers=1, hidden_dim=8, gaussian.
GF_N_WEIGHTS = 7

#: Both halves of the likelihood head must be in the trainable set.
HEAD_FRAGMENTS = ("gaussian_head/mu_projection/", "gaussian_head/sigma_projection/")


def _batch(batch: int = 2):
    rng = np.random.default_rng(0)
    return {
        "target": rng.standard_normal((batch, T, D)).astype("float32"),
        "covariates": rng.standard_normal((batch, T, C)).astype("float32"),
    }


def _model(**overrides):
    kwargs = dict(num_layers=1, hidden_dim=8, likelihood="gaussian",
                  target_dim=D, num_samples=S)
    kwargs.update(overrides)
    model = DeepAR(**kwargs)
    model(_batch(1), training=False)
    return model


def _one_adam_step(model, inputs) -> None:
    optimizer = keras.optimizers.Adam(1e-3)
    variables = list(model.trainable_variables)
    optimizer.build(variables)
    with tf.GradientTape() as tape:
        loss = default_loss(model(inputs, training=True))
    optimizer.apply_gradients(zip(tape.gradient(loss, variables), variables))


class TestDeepARGradientFlow:

    def test_gradients_reach_every_trainable_weight_after_one_step(self):
        model = _model()
        x = _batch()
        _one_adam_step(model, x)

        report = assert_gradients_reach_every_trainable_weight(model, x)

        assert len(report) == GF_N_WEIGHTS == len(model.trainable_weights)
        for fragment in HEAD_FRAGMENTS:
            assert any(fragment in path for path in report), (
                f"no weight under {fragment!r}; the likelihood head is not in "
                f"the trainable set the count above rests on"
            )

    def test_the_gradient_assertion_can_fail(self):
        model = _model()
        with broken_forward(model, stop_all_gradients):
            with pytest.raises(AssertionError, match="received NO gradient"):
                assert_gradients_reach_every_trainable_weight(model, _batch())


class TestDeepARKnobSensitivity:

    def test_num_layers_changes_the_parameterisation(self):
        builders = {n: (lambda n=n: _model(num_layers=n)) for n in (1, 2, 3)}
        signatures = assert_structural_knob_changes_weights(builders, knob="num_layers")
        counts = [len(signatures[n]) for n in (1, 2, 3)]
        assert counts == sorted(counts) and counts[0] < counts[-1], counts

    def test_hidden_dim_changes_the_parameterisation(self):
        builders = {h: (lambda h=h: _model(hidden_dim=h)) for h in (8, 16, 32)}
        assert_structural_knob_changes_weights(builders, knob="hidden_dim")

    def test_the_knob_assertion_can_fail(self):
        builders = {"a": (lambda: _model(num_layers=1)), "b": (lambda: _model(num_layers=1))}
        with pytest.raises(AssertionError, match="is a no-op"):
            assert_structural_knob_changes_weights(builders, knob="num_layers")


class TestDeepARSmokeContract:

    def test_the_forward_contract_rejects_a_broken_forward(self):
        model = _model()
        x = _batch()
        batch = x["target"].shape[0]

        def contract(out):
            assert isinstance(out, dict), f"expected a dict, got {type(out)}"
            assert set(out) == {"mu", "sigma", "target"}, f"key set: {sorted(out)}"
            for key in ("mu", "sigma", "target"):
                assert tuple(out[key].shape) == (batch, T, D), (
                    f"{key} has shape {tuple(out[key].shape)}"
                )
            assert_finite(out)
            sigma = keras.ops.convert_to_numpy(out["sigma"])
            assert np.all(sigma > 0.0), (
                f"a Gaussian scale must be strictly positive; min={sigma.min()}"
            )

        rejections = assert_contract_rejects_a_broken_forward(model, x, contract)
        assert set(rejections) == {
            "collapse_to_scalar", "slice_leading_axis", "append_trailing_axis",
        }

    def test_an_unknown_likelihood_is_rejected_at_construction(self):
        """The argument-validation half of a smoke contract.

        The finiteness/shape half above says nothing about what the constructor
        does with a value it does not support; a silent fallback to 'gaussian'
        would pass every other test in this suite.
        """
        with pytest.raises(ValueError, match="likelihood"):
            DeepAR(num_layers=1, hidden_dim=8, likelihood="laplace",
                   target_dim=D, num_samples=S)
