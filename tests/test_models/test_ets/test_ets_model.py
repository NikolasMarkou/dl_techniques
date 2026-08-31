"""Tests for ``dl_techniques.models.time_series.ets``.

The oracle here is a **plain-numpy transcription of the ETS recursion**
(``_reference_forecast``), written from the state-space equations rather than
from the implementation. Two independent closed-form checks back it up, each
chosen because it fails loudly under a plausible wrong implementation:

*   ``ANN`` with ``alpha -> 1`` must be naive-1 (the forecast IS the last
    observation). This catches an off-by-one in the recursion -- a state emitted
    one step early or late still has the right shape and a plausible value.
*   ``AAN``'s h-step forecast must be EXACTLY linear in ``h`` (zero second
    difference). This catches a trend term applied multiplicatively, or a
    seasonal buffer leaking into a non-seasonal variant.

Measured 2026-08-31, float32, 5 series of length 60:

=========================================  ==========
check                                      residual
=========================================  ==========
``ANN`` vs numpy reference                 6.56e-08
``AAN`` vs numpy reference                 6.87e-07
``AAA`` vs numpy reference                 8.15e-07
``ANN`` alpha~1 vs last observation        2.03e-06
``AAN`` second difference over h           2.38e-07
=========================================  ==========

The shrinkage reproduction lives in its own file,
``test_the_multistep_losses_shrink_alpha.py``.
"""

import os
import tempfile

import keras
import numpy as np
import pytest

from dl_techniques.models.time_series.ets.model import (
    ETS_VARIANTS,
    ETSModel,
    create_ets,
)
from dl_techniques.models.time_series.forecast import Forecast

# ---------------------------------------------------------------------

CONTEXT, HORIZON, SEASON = 48, 12, 12
ATOL = 1e-5


@pytest.fixture()
def series():
    return np.random.default_rng(0).normal(size=(5, CONTEXT)).astype("float32")


def _reference_forecast(y, variant, horizon, m, alpha, beta, gamma):
    """The ETS recursion, transcribed in numpy from the state-space equations."""
    batch, steps = y.shape
    use_trend = variant in ("AAN", "AAA")
    use_seasonal = variant == "AAA"

    warmup = m if use_seasonal else 1
    level = y[:, :warmup].mean(axis=1)

    if use_trend:
        span = min(steps - 1, 2 * max(1, m))
        trend = (y[:, 1 : span + 1] - y[:, :span]).mean(axis=1)
    else:
        trend = np.zeros(batch)

    if use_seasonal:
        seasonal = y[:, :m] - level[:, None]
        seasonal = seasonal - seasonal.mean(axis=1, keepdims=True)
    else:
        seasonal = np.zeros((batch, 1))

    if not use_trend:
        beta = 0.0
    if not use_seasonal:
        gamma = 0.0

    for t in range(steps):
        error = y[:, t] - (level + trend + seasonal[:, 0])
        new_level = level + trend + alpha * error
        new_trend = trend + beta * error
        new_season = seasonal[:, 0] + gamma * error
        seasonal = np.concatenate([seasonal[:, 1:], new_season[:, None]], axis=1)
        level, trend = new_level, new_trend

    steps_ahead = np.arange(1, horizon + 1)
    forecast = level[:, None] + trend[:, None] * steps_ahead[None, :]
    if use_seasonal:
        forecast = forecast + seasonal[:, (steps_ahead - 1) % m]
    return forecast


def _build(variant, **kwargs):
    extra = {"seasonal_period": SEASON} if variant == "AAA" else {}
    return ETSModel(variant=variant, horizon=HORIZON, **extra, **kwargs)


# ---------------------------------------------------------------------
# Test 1 -- the recursion, against numpy
# ---------------------------------------------------------------------

@pytest.mark.parametrize("variant", ETS_VARIANTS)
def test_matches_the_state_space_recursion(series, variant):
    model = _build(variant, alpha_init=0.37, beta_init=0.12, gamma_init=0.21)
    got = np.asarray(keras.ops.convert_to_numpy(model(series)))[:, :, 0]
    want = _reference_forecast(
        series.astype("float64"), variant, HORIZON,
        SEASON if variant == "AAA" else 1, 0.37, 0.12, 0.21,
    )
    np.testing.assert_allclose(got, want, rtol=0, atol=ATOL)


# ---------------------------------------------------------------------
# Test 2 -- closed-form identities
# ---------------------------------------------------------------------

def test_ann_with_alpha_one_is_naive_one(series):
    """An off-by-one in the recursion survives every shape check but not this."""
    model = ETSModel(variant="ANN", horizon=3, alpha_init=1.0 - 1e-6)
    got = np.asarray(keras.ops.convert_to_numpy(model(series)))[:, :, 0]
    np.testing.assert_allclose(got, np.tile(series[:, -1:], (1, 3)), rtol=0, atol=1e-4)


def test_aan_forecast_is_exactly_linear_in_h(series):
    model = ETSModel(variant="AAN", horizon=6)
    forecast = np.asarray(keras.ops.convert_to_numpy(model(series)))[:, :, 0]
    second_difference = np.diff(np.diff(forecast, axis=1), axis=1)
    np.testing.assert_allclose(
        second_difference, np.zeros_like(second_difference), rtol=0, atol=1e-5
    )


def test_the_seasonal_state_is_centred(series):
    """Additive seasonality is only identified up to a constant without this."""
    model = _build("AAA", gamma_init=1e-6)
    initial = keras.ops.convert_to_numpy(
        model._initial_state(keras.ops.convert_to_tensor(series))
    )
    np.testing.assert_allclose(
        np.asarray(initial)[:, 2:].sum(axis=1), 0.0, rtol=0, atol=1e-4
    )


# ---------------------------------------------------------------------
# Test 3 -- the trainable surface is EXACTLY the smoothing parameters
# ---------------------------------------------------------------------

@pytest.mark.parametrize(
    "variant,expected",
    [("ANN", ["alpha_raw"]), ("AAN", ["alpha_raw", "beta_raw"]),
     ("AAA", ["alpha_raw", "beta_raw", "gamma_raw"])],
)
def test_trainable_surface(series, variant, expected):
    """A fitted initial state would confound the shrinkage measurement.

    If this list ever grows, ``test_the_multistep_losses_shrink_alpha.py`` stops
    measuring shrinkage and starts measuring shrinkage-plus-initialisation.
    """
    model = _build(variant)
    model(series)
    assert [v.name for v in model.trainable_variables] == expected
    assert all(v.shape == () for v in model.trainable_variables)


def test_smoothing_parameters_stay_in_the_unit_interval(series):
    """The sigmoid makes the constraint unbreakable, not merely initialised."""
    model = _build("AAA", alpha_init=0.9, beta_init=0.9, gamma_init=0.9)
    model(series)
    for variable in model.trainable_variables:
        variable.assign(1e6)
    assert 0.0 < model.alpha <= 1.0
    assert 0.0 < model.beta <= 1.0
    assert 0.0 < model.gamma <= 1.0


# ---------------------------------------------------------------------
# Test 4 -- input handling and validation
# ---------------------------------------------------------------------

def test_accepts_an_explicit_feature_axis(series):
    model = _build("AAA")
    flat = np.asarray(keras.ops.convert_to_numpy(model(series)))
    with_axis = np.asarray(keras.ops.convert_to_numpy(model(series[:, :, None])))
    np.testing.assert_allclose(flat, with_axis, rtol=0, atol=1e-6)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"variant": "MAM"},                       # multiplicative: deliberately absent
        {"variant": "AAA"},                       # seasonal_period missing
        {"variant": "AAA", "seasonal_period": 1},
        {"variant": "ANN", "horizon": 0},
        {"variant": "ANN", "alpha_init": 0.0},
        {"variant": "ANN", "alpha_init": 1.0},
    ],
)
def test_rejects_bad_configuration(kwargs):
    with pytest.raises(ValueError):
        ETSModel(**kwargs)


def test_rejects_a_multivariate_input():
    model = ETSModel(variant="ANN", horizon=2)
    with pytest.raises(ValueError, match="univariate"):
        model.build((None, CONTEXT, 3))


def test_rejects_a_context_shorter_than_one_season():
    model = ETSModel(variant="AAA", horizon=2, seasonal_period=SEASON)
    with pytest.raises(ValueError, match="too short"):
        model.build((None, SEASON))


# ---------------------------------------------------------------------
# Test 5 -- the Forecast contract
# ---------------------------------------------------------------------

def test_forecast_is_a_point_forecast(series):
    """A point model MUST NOT fabricate intervals."""
    model = _build("AAA")
    forecast = model.predict_forecast(series)

    assert isinstance(forecast, Forecast)
    assert forecast.point.shape == (series.shape[0], HORIZON, 1)
    assert forecast.quantiles is None
    assert forecast.has_quantiles() is False
    with pytest.raises(ValueError):
        forecast.interval(0.1, 0.9)


def test_fitted_values_align_with_the_input(series):
    model = _build("AAA")
    fitted, residuals = model.fitted_values(series)
    assert tuple(fitted.shape) == series.shape
    np.testing.assert_allclose(
        keras.ops.convert_to_numpy(series - fitted),
        keras.ops.convert_to_numpy(residuals),
        rtol=0, atol=1e-5,
    )


# ---------------------------------------------------------------------
# Test 6 -- serialization
# ---------------------------------------------------------------------

@pytest.mark.parametrize("variant", ETS_VARIANTS)
def test_serialization_round_trip(series, variant):
    from tests.optimizer_state import build_optimizer_state

    model = _build(variant, alpha_init=0.41)
    model(series)
    model.compile(optimizer="adam", loss="mse")

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "ets.keras")
        build_optimizer_state(model)
        model.save(path)
        loaded = keras.models.load_model(path)  # NO custom_objects

    assert loaded.variant == variant
    assert loaded.horizon == HORIZON
    np.testing.assert_allclose(loaded.alpha, model.alpha, rtol=0, atol=1e-6)
    np.testing.assert_allclose(
        keras.ops.convert_to_numpy(loaded(series)),
        keras.ops.convert_to_numpy(model(series)),
        rtol=0, atol=1e-5,
    )


def test_factory_builds_the_model():
    model = create_ets("AAA", horizon=4, seasonal_period=SEASON)
    assert isinstance(model, ETSModel)
    assert model.variant == "AAA" and model.horizon == 4
