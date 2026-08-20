"""
RED proofs for ``lazy_build_contract_oracle``.

Every assertion in :func:`assert_lazy_build_costs_nothing` is proved here
against a model carrying exactly the defect it names, and against a healthy
twin that must stay green. The subjects model the two real shapes this plan
measured, not invented ones:

* ``_LossyReloadModel`` -- a sub-layer whose state does NOT survive the round
  trip, the ``SHGCNLinkPredictor`` shape (D-029): the archive is complete, the
  LOAD is lossy, and no warning is emitted.
* ``_InsensitiveModel`` -- a forward that ignores its own weights, the
  ``ScoreBasedNanoVLM`` shape (batch 3): every round-trip assertion passes
  while nothing is being compared.
"""

from __future__ import annotations

import keras
import numpy as np
import pytest
from keras import ops

from .lazy_build_contract_oracle import (
    assert_lazy_build_costs_nothing,
    materialization_report,
    measure_lazy_build,
    perturb_weights,
)


@keras.saving.register_keras_serializable(package="lazy_build_oracle_tests")
class _HealthyLazyModel(keras.Model):
    """No ``build()``; everything is created in ``__init__``. R-002 charges it
    and, as batch 2 measured on 8 real packages, nothing is lost."""

    def __init__(self, units: int = 4, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.dense = keras.layers.Dense(units)

    def call(self, inputs, training=None):
        return self.dense(inputs)

    def get_config(self):
        return {**super().get_config(), "units": self.units}


@keras.saving.register_keras_serializable(package="lazy_build_oracle_tests")
class _InsensitiveModel(keras.Model):
    """The forward does not read its own weights."""

    def __init__(self, units: int = 4, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.dense = keras.layers.Dense(units)

    def call(self, inputs, training=None):
        _ = self.dense(inputs)
        return ops.zeros((ops.shape(inputs)[0], self.units))

    def get_config(self):
        return {**super().get_config(), "units": self.units}


@keras.saving.register_keras_serializable(package="lazy_build_oracle_tests")
class _LossyReloadModel(keras.Model):
    """
    A scalar that is saved but never restored.

    ``get_config`` re-emits the CONSTRUCTION-time scale rather than the current
    one, so a reload silently reverts to the default -- the D-029 shape,
    reproduced without needing an unbuilt sub-layer.
    """

    def __init__(self, units: int = 4, scale: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self._initial_scale = scale
        self.dense = keras.layers.Dense(units)
        self.scale = self.add_weight(
            name="scale", shape=(), initializer=keras.initializers.Constant(scale),
            trainable=True,
        )

    def call(self, inputs, training=None):
        return self.dense(inputs) * self.scale

    def save_own_variables(self, store):
        # Deliberately writes nothing, so the reloaded scale keeps its
        # construction-time default. The archive is otherwise complete.
        return

    def load_own_variables(self, store):
        return

    def get_config(self):
        return {**super().get_config(), "units": self.units,
                "scale": self._initial_scale}


def _inputs():
    return np.random.RandomState(0).randn(3, 6).astype("float32")


# ---------------------------------------------------------------------------
# The healthy twin
# ---------------------------------------------------------------------------
def test_the_oracle_accepts_a_lazily_built_model_that_loses_nothing():
    report = assert_lazy_build_costs_nothing(
        build=_HealthyLazyModel, make_inputs=_inputs, input_shape=(None, 6)
    )
    assert report["roundtrip_max_delta"] == 0.0
    assert report["perturb_liveness"] > 0.0
    assert report["n_perturbed"] == report["n_weights"]


def test_the_healthy_model_is_charged_by_r_071_and_still_loses_nothing():
    """
    The measurement that makes this whole family a contract row rather than a
    defect row: ``.build(shape)`` materializes NOTHING, exactly as R-071 says,
    and the round trip above is still exactly 0.0.
    """
    keras.utils.set_random_seed(0)
    model = _HealthyLazyModel()
    n_after_call = len(model(_inputs()).shape) and len(model.weights)
    report = materialization_report(_HealthyLazyModel, (None, 6), n_after_call)
    assert report["n_weights_after_build"] == 0
    assert report["count_params_after_build"] == 0
    assert report["n_weights_after_call"] == 2


# ---------------------------------------------------------------------------
# The liveness arm -- the assertion that stops the rest being vacuous
# ---------------------------------------------------------------------------
def test_rejects_a_forward_that_ignores_its_own_weights():
    with pytest.raises(AssertionError, match="insensitive to its own weights"):
        assert_lazy_build_costs_nothing(
            build=_InsensitiveModel, make_inputs=_inputs
        )


def test_the_insensitive_model_passes_a_round_trip_check_without_the_liveness_arm():
    """Without the liveness arm the oracle would call the insensitive model
    healthy: its round trip is exactly 0.0 because nothing is compared."""
    report = measure_lazy_build(_InsensitiveModel, _inputs)
    assert report["roundtrip_max_delta"] == 0.0
    assert report["perturb_liveness"] == 0.0


# ---------------------------------------------------------------------------
# The lossy-reload arm -- the D-029 shape
# ---------------------------------------------------------------------------
def test_rejects_a_model_whose_state_does_not_survive_the_reload():
    with pytest.raises(AssertionError, match="The lazy build IS costing something"):
        assert_lazy_build_costs_nothing(
            build=lambda: _LossyReloadModel(scale=1.0), make_inputs=_inputs
        )


def test_the_lossy_model_is_lossy_by_a_measurable_amount_not_by_a_raise():
    """
    The finding this shape produces is a NUMBER, and it is silent: no warning,
    no exception, and the weight COUNT is unchanged -- which is why a count-
    based or archive-based check is necessary but not sufficient (batch 7).
    """
    report = measure_lazy_build(lambda: _LossyReloadModel(scale=1.0), _inputs)
    assert report["n_weights_reloaded"] == report["n_weights"]
    assert report["roundtrip_max_delta"] > 0.0


# ---------------------------------------------------------------------------
# The perturbation itself
# ---------------------------------------------------------------------------
def test_perturbation_skips_batchnorm_moving_statistics():
    """
    An ABSOLUTE perturbation of the moving statistics drove ``yolo12`` to NaN in
    batch 8 before its round trip could run. They are excluded by name.
    """
    model = keras.Sequential([keras.layers.Input((6,)), keras.layers.BatchNormalization()])
    before = {w.path: np.asarray(ops.convert_to_numpy(w)).copy() for w in model.weights}
    touched = perturb_weights(model)
    assert touched == 2, "gamma and beta must be perturbed"
    for w in model.weights:
        moved = not np.array_equal(np.asarray(ops.convert_to_numpy(w)), before[w.path])
        if "moving_" in w.path:
            assert not moved, f"{w.path} must not be perturbed"
        else:
            assert moved, f"{w.path} must be perturbed"


def test_perturbation_is_relative_with_a_floor_and_handles_scalar_weights():
    """
    A zero-variance weight would get a zero-sized perturbation under a purely
    relative rule; the ``1e-3`` floor is what makes a zeros-initialized bias
    move. Scalar (rank-0) weights are also covered -- the first draft of this
    probe crashed on one.
    """
    layer = keras.layers.Dense(3, kernel_initializer="zeros", bias_initializer="zeros")
    layer.build((None, 4))
    assert perturb_weights(layer) == 2
    assert float(np.abs(np.asarray(ops.convert_to_numpy(layer.bias))).max()) > 0.0
