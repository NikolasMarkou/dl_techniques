"""Tests for :class:`xLSTMForecaster` (forecasting sibling of the LM xLSTM).

Covers both head modes (quantile / point):
- forward-pass output shapes
- the unified ``Forecast`` contract via ``_forecast`` / ``predict_forecast``
- ``get_config`` / ``from_config`` round-trip (per LESSONS: build by calling the
  model once before serializing)
- the ``create_xlstm_forecaster`` factory and ``from_variant`` constructor
- constructor validation (``embed_dim % mlstm_num_heads``)

This file does NOT touch the LM test ``test_model.py`` (no regression intended).
"""

import pytest
import numpy as np
import keras

from dl_techniques.models.time_series.forecast import Forecast
from dl_techniques.models.time_series.xlstm.forecaster import (
    xLSTMForecaster,
    create_xlstm_forecaster,
)


# =============================================================================
# Shared small dimensions (CPU / GPU1-fast)
# =============================================================================

B = 2          # batch
L = 32         # context / input_length
H = 8          # prediction_length
F = 1          # num_features
EMBED_DIM = 32
NUM_LAYERS = 2
MLSTM_NUM_HEADS = 4
QLEVELS = [0.1, 0.5, 0.9]


class TestXLSTMForecaster:
    """Test suite for the xLSTMForecaster model."""

    # ----------------------------------------------------------------- fixtures
    @pytest.fixture(autouse=True)
    def _seed(self):
        """Seed numpy for deterministic inputs."""
        np.random.seed(42)

    @pytest.fixture
    def dummy_input(self) -> np.ndarray:
        """Continuous context window ``[B, L, F]``."""
        return np.random.randn(B, L, F).astype("float32")

    @pytest.fixture
    def quantile_config(self) -> dict:
        return dict(
            input_length=L,
            prediction_length=H,
            num_features=F,
            embed_dim=EMBED_DIM,
            num_layers=NUM_LAYERS,
            mlstm_num_heads=MLSTM_NUM_HEADS,
            use_quantile_head=True,
            quantile_levels=QLEVELS,
        )

    @pytest.fixture
    def point_config(self) -> dict:
        return dict(
            input_length=L,
            prediction_length=H,
            num_features=F,
            embed_dim=EMBED_DIM,
            num_layers=NUM_LAYERS,
            mlstm_num_heads=MLSTM_NUM_HEADS,
            use_quantile_head=False,
        )

    # ----------------------------------------------------------------- forward
    def test_forward_quantile(self, quantile_config, dummy_input):
        """Quantile mode forward pass -> [B, H, Q]."""
        model = xLSTMForecaster(**quantile_config)
        out = model(dummy_input)
        assert tuple(out.shape) == (B, H, len(QLEVELS))

    def test_forward_point(self, point_config, dummy_input):
        """Point mode forward pass -> [B, H, F]."""
        model = xLSTMForecaster(**point_config)
        out = model(dummy_input)
        assert tuple(out.shape) == (B, H, F)

    # ----------------------------------------------------------------- _forecast
    def test_forecast_quantile(self, quantile_config, dummy_input):
        """Quantile mode _forecast -> Forecast with quantiles + point."""
        model = xLSTMForecaster(**quantile_config)
        fc = model._forecast(dummy_input, verbose=0)

        assert isinstance(fc, Forecast)
        assert fc.point is not None
        assert fc.quantiles is not None
        assert fc.quantile_levels == QLEVELS
        # quantiles last-dim == number of levels; shape [B,H,Q] or [B,H,F,Q].
        assert fc.quantiles.shape[-1] == len(QLEVELS)
        assert fc.quantiles.shape[0] == B
        assert fc.quantiles.ndim in (3, 4)
        # point is the median, batch + horizon preserved.
        assert fc.point.shape[0] == B
        assert fc.point.shape[1] == H

    def test_forecast_point(self, point_config, dummy_input):
        """Point mode _forecast -> Forecast.point [B, H, F], quantiles None."""
        model = xLSTMForecaster(**point_config)
        fc = model._forecast(dummy_input, verbose=0)

        assert isinstance(fc, Forecast)
        assert fc.quantiles is None
        assert fc.quantile_levels is None
        assert tuple(fc.point.shape) == (B, H, F)

    def test_predict_forecast_entrypoint(self, point_config, dummy_input):
        """The ForecastMixin public entry point returns a Forecast."""
        model = xLSTMForecaster(**point_config)
        fc = model.predict_forecast(dummy_input, verbose=0)
        assert isinstance(fc, Forecast)
        assert tuple(fc.point.shape) == (B, H, F)

    # ----------------------------------------------------------------- config
    @pytest.mark.parametrize("mode", ["quantile", "point"])
    def test_config_round_trip(self, mode, quantile_config, point_config, dummy_input):
        """get_config / from_config round-trip preserves key config fields."""
        config_in = quantile_config if mode == "quantile" else point_config
        model = xLSTMForecaster(**config_in)

        # Build by calling once (LESSONS: call before serialize).
        original = model(dummy_input, training=False)

        cfg = model.get_config()
        rebuilt = xLSTMForecaster.from_config(cfg)

        # Key config fields preserved.
        assert rebuilt.input_length == model.input_length
        assert rebuilt.prediction_length == model.prediction_length
        assert rebuilt.num_features == model.num_features
        assert rebuilt.embed_dim == model.embed_dim
        assert rebuilt.num_layers == model.num_layers
        assert rebuilt.mlstm_num_heads == model.mlstm_num_heads
        assert rebuilt.use_quantile_head == model.use_quantile_head
        assert rebuilt.quantile_levels == model.quantile_levels

        # Rebuilt model produces a same-shaped output.
        rebuilt_out = rebuilt(dummy_input, training=False)
        assert tuple(rebuilt_out.shape) == tuple(original.shape)

    # ----------------------------------------------------------------- factory
    def test_create_factory(self, dummy_input):
        """create_xlstm_forecaster returns a working model."""
        model = create_xlstm_forecaster(
            input_length=L,
            prediction_length=H,
            num_features=F,
            embed_dim=EMBED_DIM,
            num_layers=NUM_LAYERS,
            mlstm_num_heads=MLSTM_NUM_HEADS,
            use_quantile_head=True,
            quantile_levels=QLEVELS,
        )
        assert isinstance(model, xLSTMForecaster)
        out = model(dummy_input)
        assert tuple(out.shape) == (B, H, len(QLEVELS))

    def test_from_variant(self, dummy_input):
        """from_variant builds a working model with overrides."""
        # Pick the first declared variant to stay robust to variant edits.
        variant = next(iter(xLSTMForecaster.MODEL_VARIANTS))
        model = xLSTMForecaster.from_variant(
            variant,
            input_length=L,
            prediction_length=H,
            num_features=F,
            # override head divisibility-safe heads for tiny dims.
            use_quantile_head=False,
        )
        assert isinstance(model, xLSTMForecaster)
        out = model(dummy_input)
        assert tuple(out.shape) == (B, H, F)

    def test_from_variant_pretrained_raises(self):
        """from_variant(pretrained=True) is not implemented."""
        variant = next(iter(xLSTMForecaster.MODEL_VARIANTS))
        with pytest.raises(NotImplementedError):
            xLSTMForecaster.from_variant(variant, pretrained=True)

    # ----------------------------------------------------------------- validation
    def test_embed_dim_not_divisible_raises(self):
        """embed_dim not divisible by mlstm_num_heads raises ValueError."""
        with pytest.raises(ValueError):
            xLSTMForecaster(
                input_length=L,
                prediction_length=H,
                embed_dim=30,            # 30 % 4 != 0
                num_layers=NUM_LAYERS,
                mlstm_num_heads=4,
            )
