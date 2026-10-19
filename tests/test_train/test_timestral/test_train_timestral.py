"""Tests of ``src/train/time_series/timestral/train_timestral.py``.

Until 2026-10-19 (plan-2026-10-14T062616-81938e81 D-170) the trainer could not run:
``_build_model`` passed ``dropout_rate``, which ``TimestralConfig`` does not have, so it
raised ``TypeError`` before training; ``main`` dropped ``--result_dir``, so a run would
have landed in the repo-root ``results/``; its three covariate flags and ``--no-onnx``
changed nothing.

One module-scoped run goes through the real ``main`` into a pytest temporary directory
with tiny settings. Nothing may be written under repo-root ``results/``: the autouse
fixture in ``tests/conftest.py`` fails any test that adds an entry there.
"""

import json
import math
import sys
from dataclasses import fields

import keras
import numpy as np
import pytest

import train.time_series.timestral.train_timestral as train_timestral
from dl_techniques.models.time_series.timestral import TimestralModel

TINY = ["--variant", "5M", "--input_length", "64", "--prediction_length", "16",
        "--patch_size", "8", "--epochs", "1", "--steps_per_epoch", "3", "--batch_size", "4",
        "--max_patterns_per_category", "2", "--no-deep-analysis", "--no-warmup",
        "--visualize_every_n_epochs", "1", "--plot_top_k_patterns", "2"]


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    root = tmp_path_factory.mktemp("timestral_runs")
    argv = sys.argv
    sys.argv = ["train_timestral", *TINY, "--result_dir", str(root)]
    try:
        train_timestral.main()
    finally:
        sys.argv = argv
    run_dirs = [p for p in root.iterdir() if p.is_dir()]
    assert len(run_dirs) == 1, run_dirs
    return run_dirs[0]


def test_the_run_lands_under_result_dir_with_its_artifacts(run):
    assert (run / "results.json").is_file()
    assert (run / "best_model.keras").is_file()
    assert any((run / "visualizations").glob("predictions_epoch_*.png"))


def test_the_held_out_metrics_compare_the_forecaster_with_both_baselines(run):
    metrics = json.loads((run / "results.json").read_text())["post_hoc_metrics"]
    for key in ("MAE", "MAE_naive", "MAE_seasonal_naive", "rMAE", "coverage"):
        assert math.isfinite(metrics[key]), key
    assert metrics["beats_naive"] == float(metrics["MAE"] < metrics["MAE_naive"])
    assert metrics["beats_seasonal_naive"] == float(
        metrics["MAE"] < metrics["MAE_seasonal_naive"])


def test_the_best_checkpoint_reloads_with_every_weight(run):
    model = keras.models.load_model(str(run / "best_model.keras"), compile=False)
    assert isinstance(model, TimestralModel)
    fresh = TimestralModel(model.config)
    fresh.build((None, 64))
    assert len(model.weights) == len(fresh.weights)
    forecast = model(np.cumsum(np.random.RandomState(0).randn(4, 64), axis=1).astype("float32"))
    assert forecast.shape == (4, 16, 21)


def test_the_config_has_no_dead_architecture_field():
    names = {f.name for f in fields(train_timestral.TimestralTrainingConfig)}
    assert "dropout_rate" not in names
    assert not any(name.startswith("use_") and "covariates" in name for name in names)


def test_build_model_builds_and_compiles_without_a_loss():
    """``compute_loss`` is the objective; a built model also saves every weight."""
    config = train_timestral.TimestralTrainingConfig(
        variant="5M", input_length=64, prediction_length=16, patch_size=8, use_warmup=False)
    trainer = train_timestral.TimestralTrainer.__new__(train_timestral.TimestralTrainer)
    trainer.config = config
    model = trainer._build_model()
    assert model.built and model.compiled
    assert model.loss is None
    assert model.config.d_model == 192 and model.config.patch_size == 8
    assert model.count_params() > 0


def test_every_split_is_batched_statically():
    """``TimestralModel`` reshapes by its batch size, so no split may end on a short batch."""
    config = train_timestral.TimestralTrainingConfig(
        variant="5M", input_length=64, prediction_length=16, batch_size=4)
    processor = train_timestral.TimestralDataProcessor(config, None, ["p"], {"p": "c"})
    assert processor.drop_remainder is True
    assert processor.normalize is False


def test_seasonal_naive_recovers_the_period_from_the_context():
    t = np.arange(140)
    series = np.stack([np.sin(2 * np.pi * t / 7), np.cos(2 * np.pi * t / 12) + 0.01 * t])
    context, future = series[:, :100], series[:, 100:]
    forecast = train_timestral.seasonal_naive(context, 40)
    np.testing.assert_allclose(forecast[0], future[0], atol=1e-9)
    assert np.abs(forecast[1] - future[1]).max() < 0.5


def test_seasonal_naive_refuses_a_context_too_short_to_backtest():
    with pytest.raises(ValueError, match="too short"):
        train_timestral.seasonal_naive(np.zeros((1, 20)), 16)


def test_forecast_windows_runs_batch_size_windows_per_forecaster_call(monkeypatch):
    """The forecaster packs a call's series into one variate-attention row set (O(rows^2))."""
    sizes = []
    real = train_timestral.TimestralForecaster.forecast

    def recording(self, context, *args, **kwargs):
        sizes.append(len(context))
        return real(self, context, *args, **kwargs)

    monkeypatch.setattr(train_timestral.TimestralForecaster, "forecast", recording)
    config = train_timestral.TimestralTrainingConfig(
        variant="5M", input_length=64, prediction_length=16, patch_size=8, batch_size=4)
    model = TimestralModel.from_variant("5M", prediction_length=16, patch_size=8)
    context = np.cumsum(np.random.RandomState(0).randn(10, 64, 1), axis=1).astype("float32")
    forecast = train_timestral.forecast_windows(model, context, config)
    assert sizes == [4, 4, 2]
    assert forecast.quantiles.shape == (10, 1, 16, 21)
    whole = real(train_timestral.TimestralForecaster(model), context[..., 0], 16)
    np.testing.assert_allclose(forecast.quantiles[:4], whole.quantiles[:4], rtol=1e-5, atol=1e-5)
