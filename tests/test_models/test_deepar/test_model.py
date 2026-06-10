"""
Test suite for the DeepAR probabilistic forecaster + its training wrapper.

These tests exercise the iteration-4 deliverables that join DeepAR to the
unified `Forecast`/`ForecastMixin` contract:

1. `DeepAR.get_config` round-trip POST-mixin -- a regression guard that mixing
   in the stateless `ForecastMixin` did not break serialization.
2. `DeepAR._forecast` shapes -- the Monte-Carlo sample -> mean point +
   empirical-percentile quantiles + populated `Forecast.samples` contract.
3. `DeepARTrainingWrapper.get_config` round-trip -- the wrapper serializes /
   re-instantiates its base DeepAR (the D-001 add_loss wrapper).
4. `DeepARTrainingWrapper` add_loss presence -- a training-mode forward
   registers the NLL via `add_loss` (so `compile(loss=None)` trains).

All dims are deliberately tiny so the eager autoregressive sampling loop in
`_forecast` stays CPU-fast. numpy is seeded for determinism. Layer/loss numerics
are covered elsewhere and are NOT re-tested here.
"""

import keras
import pytest
import numpy as np

from dl_techniques.models.time_series.deepar.model import DeepAR
from dl_techniques.models.time_series.forecast import Forecast, ForecastMixin
from train.time_series.deepar.train_deepar import DeepARTrainingWrapper
from dl_techniques.utils.logger import logger


# Small, CPU-safe problem geometry shared across tests.
B = 2            # batch
L = 12           # conditioning / input length
H = 4            # prediction horizon
T = L + H        # teacher-forced window
D = 1            # target_dim
C = 4            # covariate_dim
S = 4            # num_samples (small -> fast eager loop)


class TestDeepAR:
    """DeepAR base-model tests (post-ForecastMixin)."""

    @pytest.fixture(autouse=True)
    def _seed(self):
        np.random.seed(0)
        keras.utils.set_random_seed(0)

    @pytest.fixture
    def model_config(self):
        return dict(
            num_layers=1,
            hidden_dim=8,
            likelihood="gaussian",
            target_dim=D,
            num_samples=S,
        )

    @pytest.fixture
    def built_model(self, model_config):
        """A DeepAR built via a training-mode dummy dict forward."""
        model = DeepAR(**model_config)
        dummy = {
            "target": np.zeros((1, T, D), dtype="float32"),
            "covariates": np.zeros((1, T, C), dtype="float32"),
        }
        model(dummy, training=False)  # build via training-mode forward
        return model

    # -- 1: get_config round-trip post-mixin -------------------------------
    def test_get_config_round_trip_post_mixin(self, model_config):
        """from_config rebuilds; the ForecastMixin must not break get_config."""
        model = DeepAR(**model_config)
        assert isinstance(model, ForecastMixin)  # mixin is on the bases

        config = model.get_config()
        rebuilt = DeepAR.from_config(config)

        for key, val in model_config.items():
            assert getattr(rebuilt, key) == val, f"{key} not preserved"
        assert rebuilt.num_layers == model_config["num_layers"]
        assert rebuilt.hidden_dim == model_config["hidden_dim"]
        assert rebuilt.likelihood == model_config["likelihood"]
        logger.info("DeepAR get_config round-trip (post-mixin) test passed.")

    # -- 2: _forecast shapes ----------------------------------------------
    def test_forecast_shapes(self, built_model):
        """_forecast returns point [B,H,D], quantiles [B,H,D,Q], non-None samples."""
        pred_dict = {
            "conditioning_target": np.random.randn(B, L, D).astype("float32"),
            "full_covariates": np.random.randn(B, T, C).astype("float32"),
        }
        fc = built_model._forecast(pred_dict)

        assert isinstance(fc, Forecast)
        q = len(fc.quantile_levels)  # default [0.1, 0.5, 0.9] -> 3
        assert q == 3
        assert fc.point.shape == (B, H, D), f"point shape {fc.point.shape}"
        assert fc.quantiles.shape == (B, H, D, q), (
            f"quantiles shape {fc.quantiles.shape}"
        )
        assert fc.samples is not None
        assert fc.samples.shape == (S, B, H, D), f"samples shape {fc.samples.shape}"
        assert np.all(np.isfinite(fc.point))
        logger.info("DeepAR _forecast shapes test passed.")


class TestDeepARTrainingWrapper:
    """DeepARTrainingWrapper tests (the D-001 add_loss wrapper)."""

    @pytest.fixture(autouse=True)
    def _seed(self):
        np.random.seed(0)
        keras.utils.set_random_seed(0)

    @pytest.fixture
    def dummy_train_inputs(self):
        return {
            "target": np.random.randn(B, T, D).astype("float32"),
            "covariates": np.random.randn(B, T, C).astype("float32"),
        }

    @pytest.fixture
    def built_wrapper(self, dummy_train_inputs):
        base = DeepAR(
            num_layers=1, hidden_dim=8, likelihood="gaussian",
            target_dim=D, num_samples=S,
        )
        wrapper = DeepARTrainingWrapper(base)
        wrapper(dummy_train_inputs, training=True)  # build the wrapper
        return wrapper

    # -- 3: wrapper get_config round-trip ---------------------------------
    def test_wrapper_get_config_round_trip(self, built_wrapper):
        """from_config(get_config()) rebuilds; rebuilt .base is a DeepAR."""
        config = built_wrapper.get_config()
        rebuilt = DeepARTrainingWrapper.from_config(config)

        assert isinstance(rebuilt, DeepARTrainingWrapper)
        assert isinstance(rebuilt.base, DeepAR)
        assert rebuilt.base.likelihood == built_wrapper.base.likelihood
        assert rebuilt.base.hidden_dim == built_wrapper.base.hidden_dim
        logger.info("DeepARTrainingWrapper get_config round-trip test passed.")

    # -- 4: wrapper add_loss present --------------------------------------
    def test_wrapper_registers_loss(self, dummy_train_inputs):
        """A training-mode call registers the NLL via add_loss (len(losses) > 0)."""
        base = DeepAR(
            num_layers=1, hidden_dim=8, likelihood="gaussian",
            target_dim=D, num_samples=S,
        )
        wrapper = DeepARTrainingWrapper(base)
        wrapper(dummy_train_inputs, training=True)

        assert len(wrapper.losses) > 0, "wrapper.call did not add_loss the NLL"
        assert all(np.isfinite(float(l)) for l in wrapper.losses)
        logger.info("DeepARTrainingWrapper add_loss presence test passed.")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
