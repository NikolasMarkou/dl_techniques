"""
Oracle adoption for ``models/time_series/adaptive_ema`` -- Phase 5 batch A.

This suite had ZERO adoption of the three shared instruments
(``gradient_flow_oracle``, ``knob_sensitivity_oracle``, ``smoke_contract_oracle``)
before this file. It adopts all three; it authors no new oracle.

What the adoption measured (CPU, `.venv`, 2026-08-21)
-----------------------------------------------------
* ``learnable_thresholds=True`` + a 9-quantile head -> **6** trainable weights,
  **0** dead after one real optimizer step. Two of the six are the scalars
  `model.py`'s own class docstring names (":math:`\\Rightarrow` **2** trainable
  scalars"); the other four belong to the slope featurizer and the quantile
  projection, i.e. the head the constructor REQUIRES here (it warns when
  ``learnable_thresholds=True`` is combined with no quantile head).
* ``learnable_thresholds=False`` (**the constructor default**) -> **0** trainable
  weights. The oracle REFUSES to assert on that model rather than reporting
  green, which is the behaviour under test in
  ``test_the_default_model_has_nothing_to_train``.
"""

import numpy as np
import keras
import pytest
import tensorflow as tf

from dl_techniques.models.time_series.adaptive_ema import AdaptiveEMASlopeFilterModel

from ..gradient_flow_oracle import (
    assert_gradients_reach_every_trainable_weight,
    default_loss,
    gradient_report,
    stop_all_gradients,
)
from ..knob_sensitivity_oracle import assert_value_knob_changes_output
from ..smoke_contract_oracle import (
    assert_contract_rejects_a_broken_forward,
    assert_finite,
    broken_forward,
    collapse_to_scalar,
    append_trailing_axis,
)

#: Measured 2026-08-21: `midpoint_var`, `log_half_range_var`, the slope
#: featurizer kernel/bias and the quantile projection kernel/bias. Pinned as a
#: COUNT so a weight appearing (or disappearing) is reported here rather than
#: silently changing what the gradient assertion below covers.
GF_N_TRAINABLE = 6

#: The rank-3 series outputs. `upper_threshold` / `lower_threshold` are rank-0,
#: which matters for the breaker choice below.
SERIES_KEYS = ("ema", "slope", "signal_above", "signal_below", "signal_between")

#: The quantile head's output is (B, T, num_quantiles), NOT (B, T, C).
QUANTILE_KEY = "slope_quantiles"


def _series(batch: int = 4, length: int = 64, channels: int = 1) -> np.ndarray:
    """A random walk -- a constant input would make the slope branch trivially zero."""
    rng = np.random.default_rng(0)
    return np.cumsum(
        rng.standard_normal((batch, length, channels)), axis=1
    ).astype("float32")


def _model(**overrides):
    kwargs = dict(ema_period=10, lookback_period=5, learnable_thresholds=True,
                  quantile_head_config={"num_quantiles": 9})
    kwargs.update(overrides)
    model = AdaptiveEMASlopeFilterModel(**kwargs)
    model(_series(batch=1), training=False)  # subclassed: unbuilt until first call
    return model


class TestAdaptiveEMAGradientFlow:

    def test_gradients_reach_every_trainable_weight_after_one_step(self):
        model = _model()
        x = _series()
        # A REAL optimizer step, not an init-time reading. `fit()` on a
        # dict-output model needs a matching target structure, so the step is
        # driven by the oracle's own unlabelled loss and applied with a real
        # `Adam`. Reading at init would report on the initial values, which is
        # the reading this plan's record says was wrong by 30-330 weights
        # elsewhere.
        optimizer = keras.optimizers.Adam(1e-3)
        variables = list(model.trainable_variables)
        optimizer.build(variables)
        with tf.GradientTape() as tape:
            loss = default_loss(model(x, training=True))
        optimizer.apply_gradients(zip(tape.gradient(loss, variables), variables))

        report = assert_gradients_reach_every_trainable_weight(model, x)

        assert len(report) == GF_N_TRAINABLE == len(model.trainable_weights)

    def test_the_gradient_assertion_can_fail(self):
        """RED proof: with the forward detached, all 2 weights must be convicted."""
        model = _model()
        x = _series()
        with broken_forward(model, stop_all_gradients):
            with pytest.raises(AssertionError, match="received NO gradient"):
                assert_gradients_reach_every_trainable_weight(model, x)

    def test_the_default_model_has_nothing_to_train(self):
        """`learnable_thresholds=False` is the DEFAULT and yields 0 trainable weights.

        `model.py`'s class docstring states this outright. The point of pinning
        it here is that a gradient-flow assertion over an empty weight set is
        vacuous, and the oracle refuses rather than reporting green -- so this
        test is simultaneously the documentation check and the anti-vacuity
        control for the test above.
        """
        model = AdaptiveEMASlopeFilterModel(ema_period=10, lookback_period=5)
        model(_series(batch=1), training=False)
        assert model.trainable_weights == []
        with pytest.raises(ValueError, match="no trainable weights"):
            gradient_report(model, _series())


class TestAdaptiveEMAKnobSensitivity:

    def test_ema_period_changes_the_filtered_series(self):
        """`ema_period` is a VALUE knob: identical weight signature, different output."""
        builders = {
            p: (lambda p=p: _model(ema_period=p))
            for p in (5, 10, 40)
        }
        deltas = assert_value_knob_changes_output(
            builders, _series(), knob="ema_period",
            extract=lambda out: keras.ops.convert_to_numpy(out["ema"]),
        )
        assert all(d > 0.0 for d in deltas.values())

    def test_the_knob_assertion_can_fail(self):
        """RED proof: two arms built with the SAME value must be convicted as a no-op.

        This is the defect shape the oracle exists to catch (a kwarg that never
        reaches the forward), reproduced on the REAL model rather than on a
        fixture -- if `ema_period` were dropped in `__init__`, the arms above
        would be exactly as indistinguishable as these.
        """
        builders = {
            "a": (lambda: _model(ema_period=10)),
            "b": (lambda: _model(ema_period=10)),
        }
        with pytest.raises(AssertionError, match="is a no-op"):
            assert_value_knob_changes_output(
                builders, _series(), knob="ema_period",
                extract=lambda out: keras.ops.convert_to_numpy(out["ema"]),
            )


class TestAdaptiveEMASmokeContract:

    def test_the_forward_contract_rejects_a_broken_forward(self):
        """`slice_leading_axis` is DROPPED, with the reason.

        Two of this model's seven outputs (`upper_threshold`, `lower_threshold`)
        are rank-0 scalars, and `slice_leading_axis` raises on a rank-0 leaf by
        design -- it cannot break a scalar, so demanding a rejection would be an
        artefact of the breaker. The remaining two breakers still cover the
        container/shape and the full-shape classes of under-assertion.
        """
        model = _model()
        x = _series()

        def contract(out):
            assert isinstance(out, dict), f"expected a dict, got {type(out)}"
            assert set(SERIES_KEYS).issubset(out), f"missing keys: {out.keys()}"
            for key in SERIES_KEYS:
                assert tuple(out[key].shape) == x.shape, (
                    f"{key} has shape {tuple(out[key].shape)}, expected {x.shape}"
                )
            assert tuple(out[QUANTILE_KEY].shape) == x.shape[:2] + (9,), (
                f"{QUANTILE_KEY} has shape {tuple(out[QUANTILE_KEY].shape)}"
            )
            assert_finite({k: out[k] for k in SERIES_KEYS + (QUANTILE_KEY,)})

        rejections = assert_contract_rejects_a_broken_forward(
            model, x, contract,
            breakers=(collapse_to_scalar, append_trailing_axis),
        )
        assert set(rejections) == {"collapse_to_scalar", "append_trailing_axis"}
