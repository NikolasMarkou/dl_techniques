"""
Training pipeline for the PRISM probabilistic forecasting framework.

PRISM (Partitioned Representations for Iterative Sequence Modeling) is a
hierarchical time-series forecaster that replaces standard attention with a
learnable binary time tree combined with Haar Wavelet frequency decomposition.
Supports both point forecasting (MSE) and probabilistic forecasting via an
optional quantile head with monotonicity enforcement.

References:
    Chen et al. (2025) - PRISM: A Hierarchical Multiscale Approach for
        Time Series Forecasting (arXiv:2512.24898)
    Mallat (1989) - A theory for multiresolution signal decomposition
    Koenker & Bassett (1978) - Regression quantiles
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

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from train.common import (
    setup_gpu,
    create_callbacks as create_common_callbacks,
    set_seeds,
    create_learning_rate_schedule,
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
from dl_techniques.analyzer import AnalysisConfig
from dl_techniques.models.time_series.prism.model import PRISMModel
from dl_techniques.losses.quantile_loss import QuantileLoss
from dl_techniques.datasets.time_series import (
    TimeSeriesGenerator,
    NormalizationMethod,
)

plt.style.use('default')
sns.set_palette("husl")



# ---------------------------------------------------------------------

@dataclass
class PRISMTrainingConfig(BaseTimeSeriesTrainingConfig):
    """Configuration for PRISM training on multiple patterns.

    Inherits the shared time-series fields (data splits, optimizer/warmup knobs,
    pattern selection, category weights, visualization + deep-analysis flags)
    from :class:`BaseTimeSeriesTrainingConfig` and adds the PRISM architecture
    fields below. ``batch_size`` is re-declared because PRISM's default (64)
    differs from the base default (128); all other inherited defaults match.
    """

    experiment_name: str = "prism_forecasting"

    # Re-declared: PRISM's default differs from the base (128).
    batch_size: int = 64

    # PRISM architecture
    input_length: int = 168
    prediction_length: int = 24
    preset: str = "small"  # 'tiny', 'small', 'base', 'large'
    hidden_dim: Optional[int] = None
    num_layers: int = 2
    tree_depth: int = 2
    overlap_ratio: float = 0.25
    num_wavelet_levels: int = 3
    router_hidden_dim: int = 64
    router_temperature: float = 1.0
    dropout_rate: float = 0.1
    ffn_expansion: int = 4

    # Quantile configuration
    use_quantile_head: bool = False
    enforce_monotonicity: bool = True
    quantile_levels: List[float] = field(
        default_factory=lambda: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    )

    # PRISM-specific pattern selection
    min_data_length: int = 2000

    # ONNX export
    export_onnx: bool = False
    onnx_opset_version: int = 17

    def __post_init__(self) -> None:
        super().__post_init__()  # ratio-sum invariant
        if self.preset not in PRISMModel.MODEL_VARIANTS:
            raise ValueError(
                f"preset must be one of {list(PRISMModel.MODEL_VARIANTS.keys())}, "
                f"got '{self.preset}'"
            )
        if self.input_length <= 0 or self.prediction_length <= 0:
            raise ValueError("input_length and prediction_length must be positive")
        if self.use_quantile_head and 0.5 not in self.quantile_levels:
            logger.warning("Recommended to include 0.5 (median) in quantile_levels for evaluation.")


class PRISMDataProcessor(WindowedTimeSeriesProcessor):
    """PRISM data processor: thin subclass of :class:`WindowedTimeSeriesProcessor`.

    Uses the base's default reshape-both ``_make_sample`` / ``output_signature``
    hooks (context -> ``(input_length, num_features)``, horizon ->
    ``(prediction_length, num_features)``) with STANDARD per-instance normalization.
    """

    def __init__(
            self,
            config: PRISMTrainingConfig,
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
            normalize=True,
            normalize_method=NormalizationMethod.STANDARD,
        )


class PRISMPerformanceCallback(TimeSeriesPerformanceCallback):
    """Tracks and visualizes PRISM forecast performance.

    Thin subclass of :class:`TimeSeriesPerformanceCallback`. The base owns the
    scaffolding (``__init__`` + makedirs, ``loss``/``val_loss`` accumulation, the
    ``visualize_every_n_epochs`` gate, learning-curve delegation). PRISM keeps
    only the three model-specific pieces: viz-data prep from its processor, the
    ``metric``/``val_metric``/``lr`` extra-history tracking, and the 4D-quantile
    prediction-plot body.
    """

    def __init__(self, config: PRISMTrainingConfig, processor: PRISMDataProcessor,
                 save_dir: str, model_name: str = "prism"):
        # processor must be set BEFORE super().__init__: the base ctor calls
        # _prepare_viz_data() which reads self.processor.
        self.processor = processor
        super().__init__(config, save_dir, model_name)

    def _prepare_viz_data(self) -> Tuple[np.ndarray, np.ndarray]:
        # N3 (plan_2026-06-09_49c73926 D-003): use the shared helper rather than an
        # inline copy of the same collection loop (tirex/nbeats already use it). The
        # helper carries the D-001 tuple-guard (no-op for prism's single-array targets)
        # and the non-fatal wrapper lives in the base callback __init__. The empty case
        # returns np.array([]); _plot_predictions guards on `len(context) == 0` above.
        return _prepare_viz_data_from_processor(
            self.processor, self.config.plot_top_k_patterns)

    def _extend_history(self, logs: dict) -> None:
        self.training_history.setdefault('metric', [])
        self.training_history.setdefault('val_metric', [])
        if self.config.use_quantile_head:
            self.training_history['metric'].append(logs.get('mae_of_median', 0))
            self.training_history['val_metric'].append(logs.get('val_mae_of_median', 0))
        else:
            self.training_history['metric'].append(logs.get('mae', 0))
            self.training_history['val_metric'].append(logs.get('val_mae', 0))

        self._track_lr(logs)

    def _plot_predictions(self, epoch: int) -> None:
        context, target = self.viz_test_data
        if len(context) == 0:
            return

        predictions = self.model.predict(context, verbose=0)
        num_samples = min(self.config.plot_top_k_patterns, len(context))
        n_cols, n_rows = 3, math.ceil(num_samples / 3)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows), squeeze=False)
        axes = axes.flatten()

        if self.config.use_quantile_head:
            quantiles = self.config.quantile_levels
            median_idx = quantiles.index(0.5) if 0.5 in quantiles else len(quantiles) // 2
            low_idx, high_idx = 0, -1

        for i in range(num_samples):
            if self.config.use_quantile_head:
                _plot_ts_forecast(
                    axes[i],
                    context[i, :, 0].flatten(),
                    target[i, :, 0].flatten(),
                    predictions[i, :, 0, median_idx].flatten(),
                    lower=predictions[i, :, 0, low_idx].flatten(),
                    upper=predictions[i, :, 0, high_idx].flatten(),
                    title=f'Sample {i}',
                    context_label='Context',
                    target_label='Target',
                    point_label='Median',
                    band_label=f'{quantiles[low_idx]}-{quantiles[high_idx]} Q',
                )
            else:
                _plot_ts_forecast(
                    axes[i],
                    context[i, :, 0].flatten(),
                    target[i, :, 0].flatten(),
                    predictions[i, :, 0].flatten(),
                    title=f'Sample {i}',
                    context_label='Context',
                    target_label='Target',
                    point_label='Pred',
                )

        for j in range(num_samples, len(axes)):
            axes[j].axis('off')

        plt.suptitle(f'PRISM Forecasts (Epoch {epoch + 1})', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f'predictions_epoch_{epoch + 1:03d}.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()


class PRISMTrainer(BaseTimeSeriesTrainer):
    """Orchestrates PRISM training (point or quantile).

    Thin subclass of :class:`BaseTimeSeriesTrainer`. The base owns the skeleton
    (``__init__`` generator/pattern setup, ``_select_patterns``,
    ``_create_experiment_dir``, ``_make_callbacks``, ``_train_model``,
    ``_save_results``, ``_export_to_onnx``). PRISM overrides only the genuine
    divergences: the processor, the model build (+dummy warmup), the
    performance callback, the ``preset``/``mode`` results prefix, and the two
    callback class attrs.
    """

    MODEL_DISPLAY_NAME = "PRISM"
    EARLY_STOPPING_PATIENCE = 30

    def _build_processor(self) -> PRISMDataProcessor:
        return PRISMDataProcessor(
            self.config, self.generator, self.selected_patterns,
            self.pattern_to_category,
        )

    def _build_performance_callback(self, viz_dir: str) -> PRISMPerformanceCallback:
        return PRISMPerformanceCallback(self.config, self.processor, viz_dir, "prism")

    def _build_results_prefix(self) -> str:
        mode = "quantile" if self.config.use_quantile_head else "point"
        return f"{self.config.experiment_name}_{self.config.preset}_{mode}"

    def _build_model(self) -> PRISMModel:
        """Create and compile the PRISM model (+ dummy-input warmup)."""
        logger.info(f"Building PRISM model (preset={self.config.preset})")
        num_features = self.processor.num_features

        model_kwargs: Dict[str, Any] = {
            "variant": self.config.preset,
            # PRISMModel API kwargs are FIXED (context_len/forecast_len); only the
            # VALUE source renamed to the config's input_length/prediction_length.
            "context_len": self.config.input_length,
            "forecast_len": self.config.prediction_length,
            "num_features": num_features,
            "num_layers": self.config.num_layers,
            "tree_depth": self.config.tree_depth,
            "overlap_ratio": self.config.overlap_ratio,
            "num_wavelet_levels": self.config.num_wavelet_levels,
            "router_hidden_dim": self.config.router_hidden_dim,
            "router_temperature": self.config.router_temperature,
            "dropout_rate": self.config.dropout_rate,
            "ffn_expansion": self.config.ffn_expansion,
            "use_quantile_head": self.config.use_quantile_head,
            "num_quantiles": len(self.config.quantile_levels) if self.config.use_quantile_head else 3,
            "quantile_levels": self.config.quantile_levels if self.config.use_quantile_head else None,
            "enforce_monotonicity": self.config.enforce_monotonicity,
        }
        if self.config.hidden_dim is not None:
            model_kwargs["hidden_dim"] = self.config.hidden_dim

        model = PRISMModel.from_variant(**model_kwargs)

        optimizer = self._build_optimizer()

        model.build((None, self.config.input_length, num_features))

        if self.config.use_quantile_head:
            logger.info("Compiling with QuantileLoss")
            loss = QuantileLoss(quantiles=self.config.quantile_levels)
            metrics: List[Any] = []
            if 0.5 in self.config.quantile_levels:
                median_idx = self.config.quantile_levels.index(0.5)
                def mae_of_median(y_true, y_pred):
                    return keras.metrics.mean_absolute_error(y_true, y_pred[:, :, :, median_idx])
                mae_of_median.__name__ = 'mae_of_median'
                metrics.append(mae_of_median)
        else:
            logger.info("Compiling with MSE Loss")
            loss = 'mse'
            metrics = ['mae', 'mse']

        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        # PRISMModel (subclassed Model) needs a forward pass to finalize weights.
        dummy_input = np.zeros(
            (1, self.config.input_length, num_features), dtype='float32')
        model(dummy_input)
        return model


def build_parser() -> argparse.ArgumentParser:
    """Build the PRISM CLI parser on top of the shared TS argument parser.

    Starts from :func:`create_ts_argument_parser` (the shared TS args:
    ``--epochs``/``--batch_size``/``--steps_per_epoch``/``--learning_rate``/
    warmup/analysis/``--gpu`` etc.), restores PRISM's own defaults for the args
    whose default differs from the shared parser via ``set_defaults`` (epochs
    150, batch_size 64, steps_per_epoch 500, experiment_name "prism"), then adds
    PRISM's architecture-specific flags (preset, input/prediction lengths,
    quantile head, ONNX, per-instance normalization toggle).
    """
    parser = create_ts_argument_parser("PRISM Training Framework")

    # Restore PRISM's per-arg defaults where they differ from the shared parser
    # (shared: experiment_name=timeseries, epochs=200, batch_size=128,
    # steps_per_epoch=1000). All other shared defaults already match PRISM.
    parser.set_defaults(
        experiment_name="prism",
        epochs=150,
        batch_size=64,
        steps_per_epoch=500,
    )

    # PRISM architecture-specific arguments.
    parser.add_argument("--preset", type=str, default="small", choices=['tiny', 'small', 'base', 'large'])
    parser.add_argument("--input_length", type=int, default=168)
    parser.add_argument("--prediction_length", type=int, default=24)
    parser.add_argument("--hidden_dim", type=int, default=None)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--tree_depth", type=int, default=2)
    parser.add_argument("--num_wavelet_levels", type=int, default=3)
    parser.add_argument("--use_quantile_head", action="store_true")
    parser.add_argument("--no_monotonicity", dest="enforce_monotonicity", action="store_false")
    parser.set_defaults(enforce_monotonicity=True)
    parser.add_argument("--no-normalize", dest="normalize_per_instance", action="store_false")
    parser.set_defaults(normalize_per_instance=True)
    parser.add_argument("--no_onnx", dest="export_onnx", action="store_false")
    parser.set_defaults(export_onnx=False)
    parser.add_argument("--onnx_opset_version", type=int, default=17)
    return parser


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


def main() -> None:
    args = parse_args()
    set_seeds(args.seed)
    setup_gpu(args.gpu)

    config = PRISMTrainingConfig(
        seed=args.seed,
        experiment_name=args.experiment_name,
        preset=args.preset,
        input_length=args.input_length,
        prediction_length=args.prediction_length,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        tree_depth=args.tree_depth,
        num_wavelet_levels=args.num_wavelet_levels,
        epochs=args.epochs,
        batch_size=args.batch_size,
        steps_per_epoch=args.steps_per_epoch,
        learning_rate=args.learning_rate,
        use_warmup=args.use_warmup,
        warmup_steps=args.warmup_steps,
        warmup_start_lr=args.warmup_start_lr,
        gradient_clip_norm=args.gradient_clip_norm,
        optimizer=args.optimizer,
        normalize_per_instance=args.normalize_per_instance,
        max_patterns_per_category=args.max_patterns_per_category,
        visualize_every_n_epochs=args.visualize_every_n_epochs,
        plot_top_k_patterns=args.plot_top_k_patterns,
        perform_deep_analysis=args.perform_deep_analysis,
        analysis_frequency=args.analysis_frequency,
        analysis_start_epoch=args.analysis_start_epoch,
        use_quantile_head=args.use_quantile_head,
        enforce_monotonicity=args.enforce_monotonicity,
        export_onnx=args.export_onnx,
        onnx_opset_version=args.onnx_opset_version,
    )

    generator_config = build_generator_config(args)

    try:
        trainer = PRISMTrainer(config, generator_config)
        results = trainer.run_experiment()
        logger.info(f"Completed. Results: {results['results_dir']}")
    except Exception as e:
        logger.error(f"Failed: {e}", exc_info=True)
        sys.exit(1)
    finally:
        keras.backend.clear_session()
        sys.stdout.flush()
        sys.stderr.flush()


if __name__ == "__main__":
    main()
