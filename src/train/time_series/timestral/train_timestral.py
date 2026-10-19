"""
Training pipeline for the Timestral probabilistic forecasting framework.

Timestral is a patch-based transformer trained with Decoupled Horizon Forecasting (DHF),
where forecasts are generated from separate prediction tokens that read the causally
encoded context. Supports past-only and known-future covariates, categorical covariates,
and outputs 21 non-crossing quantiles per time step.

References:
    - Mistral AI, 2024. Timestral: Time Series Transformer for Probabilistic Forecasting.
      (https://mistral.ai/news/timestral)
"""

import os
import sys
import math
import argparse
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

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
from train.common.timeseries import _plot_ts_forecast
from dl_techniques.utils.logger import logger
from dl_techniques.losses.quantile_loss import QuantileLoss
from dl_techniques.models.time_series.timestral import (
    create_timestral_by_variant, TimestralModel, TimestralConfig
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
    dropout_rate: float = 0.1

    # Covariate support
    use_past_covariates: bool = False
    use_known_covariates: bool = False
    use_categorical_covariates: bool = False

    # Clampsym test-time augmentation (eval only)
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
            logger.warning("Recommended to include 0.5 (median) in quantile_levels.")


class TimestralDataProcessor(WindowedTimeSeriesProcessor):
    """Timestral data processor: thin subclass of WindowedTimeSeriesProcessor.

    Key differences from the base:
    1. normalize=False — the model handles normalization internally
    2. Supports covariates (past, known-future, categorical)
    3. Output signature includes covariate tensors when enabled
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
        )
        self.use_past_covariates = config.use_past_covariates
        self.use_known_covariates = config.use_known_covariates
        self.use_categorical_covariates = config.use_categorical_covariates

    def _make_sample(self, window: np.ndarray, pattern_name: str) -> Tuple[Any, Any]:
        ctx = self.context_len
        # Base input: context window
        x = window[:ctx].reshape(-1, self.num_features).astype(np.float32)
        y = window[ctx:ctx + self.horizon_len].reshape(-1, self.num_features).astype(np.float32)

        # For now, we don't generate synthetic covariates in the generator
        # This is a placeholder for future covariate support
        # The generator would need to be extended to produce covariate data

        return x, y

    @property
    def output_signature(self) -> Tuple[Any, Any]:
        import tensorflow as tf
        return (
            tf.TensorSpec(shape=(self.context_len, self.num_features), dtype=tf.float32),
            tf.TensorSpec(shape=(self.horizon_len, self.num_features), dtype=tf.float32),
        )


class TimestralPerformanceCallback(TimeSeriesPerformanceCallback):
    """Tracks and visualizes Timestral quantile forecast performance.

    Thin subclass of TimeSeriesPerformanceCallback. The base owns the
    scaffolding. Timestral keeps only the model-specific pieces:
    viz-data prep, quantile metrics tracking, and 3D-quantile prediction plots.
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
        self.training_history.setdefault('mae_median', []).append(logs.get('mae_of_median', 0))
        self.training_history.setdefault('val_mae_median', []).append(logs.get('val_mae_of_median', 0))
        self._track_lr(logs)

    def _plot_predictions(self, epoch: int) -> None:
        test_x, test_y = self.viz_test_data
        if len(test_x) == 0:
            return

        preds = self.model(test_x, training=False)
        if not isinstance(preds, np.ndarray):
            preds = preds.numpy()

        quantiles = self.config.quantile_levels
        median_idx = quantiles.index(0.5) if 0.5 in quantiles else len(quantiles) // 2
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


class TimestralTrainer(BaseTimeSeriesTrainer):
    """Orchestrates Timestral training with Quantile Loss.

    Thin subclass of BaseTimeSeriesTrainer. The base owns the skeleton.
    Timestral overrides only the genuine divergences: the processor, the model
    build (+compile), the performance callback, and the model_type prefix.
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
        """Create and compile the Timestral model."""
        logger.info(f"Creating Timestral ({self.config.variant})")

        model = create_timestral_by_variant(
            variant=self.config.variant,
            prediction_length=self.config.prediction_length,
            patch_size=self.config.patch_size,
            quantile_levels=tuple(self.config.quantile_levels),
            dropout_rate=self.config.dropout_rate
        )

        optimizer = self._build_optimizer()

        loss = QuantileLoss(quantiles=self.config.quantile_levels, normalize=True)
        metrics = []
        if 0.5 in self.config.quantile_levels:
            median_idx = self.config.quantile_levels.index(0.5)
            def mae_of_median(y_true, y_pred):
                return keras.metrics.mean_absolute_error(y_true, y_pred[:, :, median_idx])
            mae_of_median.__name__ = 'mae_of_median'
            metrics.append(mae_of_median)

        model.compile(optimizer=optimizer, loss=loss, metrics=metrics, jit_compile=True)
        return model


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
    parser.add_argument("--use_past_covariates", action="store_true")
    parser.add_argument("--use_known_covariates", action="store_true")
    parser.add_argument("--use_categorical_covariates", action="store_true")
    parser.add_argument("--no-onnx", dest="export_onnx", action="store_false")
    parser.set_defaults(export_onnx=False)
    parser.add_argument("--onnx_opset_version", type=int, default=17)
    return parser


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


def main() -> None:
    args = parse_args()
    set_seeds(args.seed)
    setup_gpu(args.gpu)

    config = TimestralTrainingConfig(
        seed=args.seed,
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
        export_onnx=args.export_onnx,
        onnx_opset_version=args.onnx_opset_version,
        use_past_covariates=args.use_past_covariates,
        use_known_covariates=args.use_known_covariates,
        use_categorical_covariates=args.use_categorical_covariates,
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