"""
Training pipeline for the Timestral probabilistic forecasting framework.

Timestral is a patch-based transformer trained with Decoupled Horizon Forecasting (DHF),
where forecasts are generated from separate prediction tokens that read the causally
encoded context, and outputs 21 non-crossing quantiles per time step.

The model trains under stock ``compile``/``fit``: ``TimestralModel.call`` is one decode
block after each context and ``TimestralModel.compute_loss`` is the objective, a pinball
loss over the quantile levels with each horizon patch in its own causal frame (D-169).
The synthetic windows carry no covariates, so this trainer trains the context-only path.
Held-out evaluation runs ``TimestralForecaster`` (the refine frame and, by default,
clampsym) on the test split and reports it against the naive last-value and the
seasonal-naive forecasts of the same windows.

References:
    - Mistral AI, 2026. Timestral: Mistral's time-series foundation models.
      (https://mistral.ai/news/timestral)
"""

import os
import sys
import math
import argparse
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import keras
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from train.common import (
    setup_gpu,
    set_seeds,
    BaseTimeSeriesTrainingConfig,
    WindowedTimeSeriesProcessor,
    TimeSeriesPerformanceCallback,
    BaseTimeSeriesTrainer,
    create_ts_argument_parser,
    _prepare_viz_data_from_processor,
)
from train.common.args import build_generator_config
from train.common.timeseries import _plot_ts_forecast, compute_post_hoc_forecast_metrics
from dl_techniques.utils.logger import logger
from dl_techniques.models.time_series.timestral import (
    create_timestral_by_variant, TimestralModel, TimestralForecaster, TimestralForecast
)
from dl_techniques.datasets.time_series import TimeSeriesGenerator

plt.style.use('default')
sns.set_palette("husl")


@dataclass
class TimestralTrainingConfig(BaseTimeSeriesTrainingConfig):
    """Configuration for Timestral training on multiple patterns.

    Inherits the shared time-series fields (data splits, optimizer/warmup knobs,
    pattern selection, category weights, visualization + deep-analysis flags)
    from BaseTimeSeriesTrainingConfig and adds the Timestral architecture
    fields below.
    """

    experiment_name: str = "timestral"
    variant: str = "28M"  # '5M', '28M', '123M', '209M'

    # Timestral architecture
    input_length: int = 1024
    prediction_length: int = 96
    patch_size: int = 32
    quantile_levels: List[float] = field(default_factory=lambda: [
        0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45,
        0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99
    ])

    # Clampsym test-time augmentation (held-out evaluation and plots only)
    symmetric_averaging: bool = True
    clip_nonnegative: bool = True

    def __post_init__(self) -> None:
        super().__post_init__()  # ratio-sum invariant
        if self.variant not in ("5M", "28M", "123M", "209M"):
            raise ValueError(f"variant must be one of 5M, 28M, 123M, 209M, got '{self.variant}'")
        if self.input_length <= 0 or self.prediction_length <= 0:
            raise ValueError("input_length and prediction_length must be positive")
        if self.patch_size <= 0:
            raise ValueError("patch_size must be positive")
        if 0.5 not in self.quantile_levels:
            raise ValueError("quantile_levels must contain 0.5 (TimestralConfig requires it)")


class TimestralDataProcessor(WindowedTimeSeriesProcessor):
    """Timestral data processor: thin subclass of WindowedTimeSeriesProcessor.

    Key differences from the base:
    1. normalize=False: the model normalizes causally per patch itself.
    2. drop_remainder=True: ``TimestralModel`` reshapes by its batch size, so every
       split is batched with a static batch dimension.
    """

    def __init__(
        self,
        config: TimestralTrainingConfig,
        generator: TimeSeriesGenerator,
        selected_patterns: List[str],
        pattern_to_category: Dict[str, str],
        num_features: int = 1,
    ):
        super().__init__(
            config,
            generator,
            selected_patterns,
            pattern_to_category=pattern_to_category,
            context_len=config.input_length,
            horizon_len=config.prediction_length,
            num_features=num_features,
            normalize=False,  # Timestral handles normalization internally
            drop_remainder=True,
        )


class TimestralPerformanceCallback(TimeSeriesPerformanceCallback):
    """Tracks and visualizes Timestral quantile forecast performance.

    Thin subclass of TimeSeriesPerformanceCallback. The base owns the
    scaffolding. Timestral keeps only the model-specific pieces: viz-data prep,
    LR tracking, and prediction plots drawn by ``TimestralForecaster`` (the
    forecast a user gets, refine frame included), not by ``model(x)``.
    """

    def __init__(self, config: TimestralTrainingConfig, processor: TimestralDataProcessor,
                 save_dir: str, model_name: str = "timestral"):
        # processor must be set BEFORE super().__init__: the base ctor calls
        # _prepare_viz_data() which reads self.processor.
        self.processor = processor
        super().__init__(config, save_dir, model_name)

    def _prepare_viz_data(self) -> Tuple[np.ndarray, np.ndarray]:
        return _prepare_viz_data_from_processor(self.processor, self.config.plot_top_k_patterns)

    def _extend_history(self, logs: dict) -> None:
        self._track_lr(logs)

    def _plot_predictions(self, epoch: int) -> None:
        test_x, test_y = self.viz_test_data
        if len(test_x) == 0:
            return

        preds = forecast_windows(self.model, test_x, self.config).quantiles[:, 0]

        quantiles = self.config.quantile_levels
        median_idx = quantiles.index(0.5)
        low_idx, high_idx = 0, len(quantiles) - 1

        num_plots = min(len(test_x), self.config.plot_top_k_patterns)
        n_cols, n_rows = 3, math.ceil(num_plots / 3)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4 * n_rows), squeeze=False)
        axes = axes.flatten()

        for i in range(num_plots):
            _plot_ts_forecast(
                axes[i],
                test_x[i].flatten(),
                test_y[i].flatten(),
                preds[i, :, median_idx],
                lower=preds[i, :, low_idx],
                upper=preds[i, :, high_idx],
                title=f'Sample {i + 1}',
                context_label='Input',
                target_label='True',
                point_label='Median',
                band_label=f'{quantiles[low_idx]}-{quantiles[high_idx]} Q',
            )

        for j in range(num_plots, len(axes)):
            axes[j].axis('off')

        plt.suptitle(f'Probabilistic Forecasts (Epoch {epoch + 1})', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f'predictions_epoch_{epoch + 1:03d}.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()


def forecast_windows(model: TimestralModel, context: np.ndarray,
                     config: TimestralTrainingConfig) -> TimestralForecast:
    """``TimestralForecaster`` over (B, T, 1) context windows, horizon ``prediction_length``.

    Interface contract (shared by the plot callback and the held-out metrics): returns
    one ``TimestralForecast`` (quantiles (B, 1, H, Q)), with clampsym as configured.
    Windows are forecast ``config.batch_size`` at a time: the forecaster packs every
    series of a call into one variate-attention row set, whose mask grows with the
    square of the row count (4416 test windows asked for one 8.2 GB tensor).
    """
    forecaster = TimestralForecaster(model)
    context = np.asarray(context)[..., 0]
    chunks = [
        forecaster.forecast(
            context[i:i + config.batch_size], config.prediction_length,
            symmetric_averaging=config.symmetric_averaging,
            clip_nonnegative=config.clip_nonnegative,
        ).quantiles
        for i in range(0, len(context), config.batch_size)
    ]
    return TimestralForecast(np.concatenate(chunks), forecaster.quantile_levels)


def seasonal_naive(context: np.ndarray, horizon: int, max_period: Optional[int] = None) -> np.ndarray:
    """Seasonal-naive forecast of (B, T) contexts, the period chosen per series from its context.

    Interface contract: for each series, every period m in [1, max_period] (default
    T // 2; m = 1 is the naive last value) is backtested on the context's last
    ``horizon`` steps, forecasting them from the ``horizon`` steps' own past; the period
    with the lowest backtest MAE forecasts the horizon, ``y[t] = x[T - m + (t mod m)]``.
    Only context values are read. Returns (B, horizon).
    """
    context = np.asarray(context, dtype=np.float64)
    b, t = context.shape
    max_period = max_period or t // 2
    if t < horizon + max_period:
        raise ValueError(f"context {t} is too short to backtest {max_period} periods over {horizon} steps")
    steps = np.arange(horizon)
    history, held = context[:, :t - horizon], context[:, t - horizon:]
    best_err = np.full(b, np.inf)
    best = np.zeros((b, horizon))
    for m in range(1, max_period + 1):
        backtest = history[:, history.shape[1] - m + steps % m]
        err = np.abs(backtest - held).mean(axis=1)
        better = err < best_err
        best_err = np.where(better, err, best_err)
        best[better] = context[better][:, t - m + steps % m]
    return best


class TimestralTrainer(BaseTimeSeriesTrainer):
    """Orchestrates Timestral training under stock ``fit``.

    Thin subclass of BaseTimeSeriesTrainer. The base owns the skeleton.
    Timestral overrides only the genuine divergences: the processor, the model
    build (+compile), the performance callback, the results prefix, and the
    held-out metrics (the forecaster against naive and seasonal-naive).
    """

    MODEL_DISPLAY_NAME = "Timestral"
    EARLY_STOPPING_PATIENCE = 30

    def _build_processor(self) -> TimestralDataProcessor:
        return TimestralDataProcessor(
            self.config, self.generator, self.selected_patterns,
            self.pattern_to_category,
        )

    def _build_performance_callback(self, viz_dir: str) -> TimestralPerformanceCallback:
        return TimestralPerformanceCallback(self.config, self.processor, viz_dir, "timestral")

    def _build_results_prefix(self) -> str:
        return f"{self.config.experiment_name}_{self.config.variant}"

    def _build_model(self) -> TimestralModel:
        """Create, build and compile the Timestral model.

        No ``loss=``: ``TimestralModel.compute_loss`` is the objective (D-169), and
        stock ``fit`` calls it. No compiled metric either: ``model(x)`` keeps every
        horizon patch in the anchor frame, so a metric on it would not score the
        forecast; the held-out metrics run the forecaster instead.
        """
        logger.info(f"Creating Timestral ({self.config.variant})")

        model = create_timestral_by_variant(
            variant=self.config.variant,
            prediction_length=self.config.prediction_length,
            patch_size=self.config.patch_size,
            quantile_levels=tuple(self.config.quantile_levels),
        )
        model.build((None, self.config.input_length))
        model.compile(optimizer=self._build_optimizer(), jit_compile=True)
        return model

    def _compute_post_hoc_metrics(self, data_pipeline: Dict[str, Any]) -> Dict[str, float]:
        """Held-out forecaster metrics, with the naive and seasonal-naive baselines.

        The base block (MAE, RMSE, sMAPE, rMAE against the naive last value, MASE,
        coverage, sharpness) on ``TimestralForecaster``'s median and quantiles, plus
        ``MAE_naive``, ``MAE_seasonal_naive`` and ``beats_naive`` /
        ``beats_seasonal_naive`` (1.0 or 0.0), all on the same test windows. A failure
        raises, as the base block does (D-183): these numbers are the run's evidence
        that the forecaster beats its baselines.
        """
        context, y_true = data_pipeline['test_data_raw']
        forecast = forecast_windows(self.model, context, self.config)
        point = forecast.median[:, 0][..., None]
        metrics = compute_post_hoc_forecast_metrics(
            y_true, point, backcast=context,
            quantiles=forecast.quantiles[:, 0][:, :, None, :],
            quantile_levels=list(forecast.quantile_levels),
        )
        horizon = self.config.prediction_length
        naive = np.repeat(context[:, -1:, 0], horizon, axis=1)
        seasonal = seasonal_naive(context[..., 0], horizon)
        metrics['MAE_naive'] = float(np.abs(y_true[..., 0] - naive).mean())
        metrics['MAE_seasonal_naive'] = float(np.abs(y_true[..., 0] - seasonal).mean())
        metrics['beats_naive'] = float(metrics['MAE'] < metrics['MAE_naive'])
        metrics['beats_seasonal_naive'] = float(metrics['MAE'] < metrics['MAE_seasonal_naive'])
        logger.info(f"Held-out forecaster vs baselines: {metrics}")
        return metrics


def build_parser() -> argparse.ArgumentParser:
    """Build the Timestral CLI parser on top of the shared TS argument parser."""
    parser = create_ts_argument_parser("Timestral Training Framework")

    # Restore Timestral's per-arg defaults where they differ from the shared parser.
    parser.set_defaults(
        experiment_name="timestral",
        max_patterns_per_category=100,
    )

    # Timestral architecture-specific arguments.
    parser.add_argument("--variant", type=str, default="28M", choices=['5M', '28M', '123M', '209M'])
    parser.add_argument("--input_length", type=int, default=1024)
    parser.add_argument("--prediction_length", type=int, default=96)
    parser.add_argument("--patch_size", type=int, default=32)
    parser.add_argument("--no-symmetric-averaging", dest="symmetric_averaging",
                        action="store_false",
                        help="Held-out evaluation without clampsym's symmetric averaging")
    parser.add_argument("--no-clip-nonnegative", dest="clip_nonnegative",
                        action="store_false",
                        help="Held-out evaluation without clampsym's non-negative clipping")
    return parser


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


def main() -> None:
    args = parse_args()
    set_seeds(args.seed)
    setup_gpu(args.gpu)

    config = TimestralTrainingConfig(
        seed=args.seed,
        result_dir=args.result_dir,
        experiment_name=args.experiment_name,
        variant=args.variant,
        input_length=args.input_length,
        prediction_length=args.prediction_length,
        patch_size=args.patch_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        steps_per_epoch=args.steps_per_epoch,
        learning_rate=args.learning_rate,
        use_warmup=args.use_warmup,
        warmup_steps=args.warmup_steps,
        warmup_start_lr=args.warmup_start_lr,
        gradient_clip_norm=args.gradient_clip_norm,
        optimizer=args.optimizer,
        max_patterns_per_category=args.max_patterns_per_category,
        visualize_every_n_epochs=args.visualize_every_n_epochs,
        plot_top_k_patterns=args.plot_top_k_patterns,
        perform_deep_analysis=args.perform_deep_analysis,
        analysis_frequency=args.analysis_frequency,
        analysis_start_epoch=args.analysis_start_epoch,
        symmetric_averaging=args.symmetric_averaging,
        clip_nonnegative=args.clip_nonnegative,
    )

    generator_config = build_generator_config(args)

    try:
        trainer = TimestralTrainer(config, generator_config)
        results = trainer.run_experiment()
        logger.info(f"Completed. Results: {results['results_dir']}")
    except Exception as e:
        logger.error(f"Failed: {e}", exc_info=True)
        sys.exit(1)
    finally:
        keras.utils.clear_session()
        sys.stdout.flush()
        sys.stderr.flush()


if __name__ == "__main__":
    main()