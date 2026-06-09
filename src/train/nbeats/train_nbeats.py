"""
Training pipeline for the N-BEATS forecasting architecture.

N-BEATS uses a doubly residual topology of stacks and blocks to decompose time
series into trend, seasonality, and generic components. Supports both interpretable
and generic stack configurations with optional backcast reconstruction loss.

References:
    Oreshkin et al. (2019) - N-BEATS: Neural basis expansion analysis for
    interpretable time series forecasting (ICLR)
"""

import os
import sys
import json
import math
import random
import argparse
from datetime import datetime
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import keras
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf

from train.common import (
    setup_gpu,
    create_callbacks as create_common_callbacks,
    generate_training_curves,
    set_seeds,
    create_learning_rate_schedule,
    json_numpy_default,
    BaseTimeSeriesTrainingConfig,
    WindowedTimeSeriesProcessor,
)
from dl_techniques.utils.logger import logger
from dl_techniques.losses.mase_loss import MASELoss
from dl_techniques.models.nbeats import create_nbeats_model
from dl_techniques.datasets.time_series import (
    TimeSeriesGenerator,
    TimeSeriesGeneratorConfig,
    TimeSeriesNormalizer,
    NormalizationMethod
)
from dl_techniques.analyzer import AnalysisConfig
from dl_techniques.callbacks.analyzer_callback import EpochAnalyzerCallback

plt.style.use('default')
sns.set_palette("husl")


def set_random_seeds(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    set_seeds(seed)


set_random_seeds(42)


@dataclass
class NBeatsTrainingConfig(BaseTimeSeriesTrainingConfig):
    """Configuration for N-BEATS training.

    Inherits the shared time-series fields (data splits, optimizer/warmup knobs,
    pattern selection, category weights, visualization + deep-analysis flags)
    from :class:`BaseTimeSeriesTrainingConfig` and adds the N-BEATS architecture
    fields below. A handful of inherited defaults are re-declared because the
    N-BEATS originals differ from the base: ``steps_per_epoch`` (1000 vs 500),
    ``warmup_steps`` (5000 vs 1000), ``max_patterns_per_category`` (100 vs 10),
    and ``optimizer`` ('adam' vs 'adamw'). ``category_weights`` and
    ``normalize_per_instance`` match the base defaults and are dropped.
    """

    experiment_name: str = "nbeats"

    # Re-declared: N-BEATS originals differ from the base defaults.
    steps_per_epoch: int = 1000      # base default: 500
    warmup_steps: int = 5000         # base default: 1000
    max_patterns_per_category: int = 100  # base default: 10
    optimizer: str = 'adam'          # base default: 'adamw'

    # N-BEATS architecture
    backcast_length: int = 168
    forecast_length: int = 24
    input_dim: int = 1
    stack_types: List[str] = field(default_factory=lambda: ["trend", "seasonality", "generic"])
    nb_blocks_per_stack: int = 3
    hidden_layer_units: int = 128
    use_normalization: bool = True
    use_bias: bool = True
    activation: str = "gelu"
    primary_loss: Union[str, keras.losses.Loss] = "mase_loss"
    mase_seasonal_periods: int = 1

    # Regularization
    kernel_regularizer_l2: float = 1e-5
    reconstruction_loss_weight: float = 0.5
    dropout_rate: float = 0.1

    def __post_init__(self) -> None:
        super().__post_init__()  # ratio-sum invariant
        if self.backcast_length <= 0 or self.forecast_length <= 0:
            raise ValueError("backcast_length and forecast_length must be positive")


class MultiPatternDataProcessor(WindowedTimeSeriesProcessor):
    """N-BEATS data processor: subclass of :class:`WindowedTimeSeriesProcessor`.

    Overrides BOTH hooks to emit the optional backcast-reconstruction target.
    With ``reconstruction_loss_weight > 0`` the target is a 2-tuple
    ``(forecast, backcast.flatten())``; otherwise it is the single forecast
    tensor (matching the base default). STANDARD per-instance normalization.
    """

    def __init__(self, config: NBeatsTrainingConfig, generator: TimeSeriesGenerator,
                 selected_patterns: List[str], pattern_to_category: Dict[str, str],
                 num_features: int = 1):
        super().__init__(
            config,
            generator,
            selected_patterns,
            pattern_to_category=pattern_to_category,
            context_len=config.backcast_length,
            horizon_len=config.forecast_length,
            num_features=num_features,
            normalize=True,
            normalize_method=NormalizationMethod.STANDARD,
        )

    def _make_sample(self, window: np.ndarray, pattern_name: str) -> Tuple[Any, Any]:
        ctx = self.context_len
        backcast = window[:ctx].reshape(-1, self.num_features).astype(np.float32)
        forecast = window[ctx:].reshape(-1, self.num_features).astype(np.float32)
        if self.config.reconstruction_loss_weight > 0.0:
            return backcast, (forecast, backcast.flatten().astype(np.float32))
        return backcast, forecast

    @property
    def output_signature(self) -> Tuple[Any, Any]:
        x_spec = tf.TensorSpec(shape=(self.context_len, self.num_features), dtype=tf.float32)
        y_spec = tf.TensorSpec(shape=(self.horizon_len, self.num_features), dtype=tf.float32)
        if self.config.reconstruction_loss_weight > 0.0:
            rec_spec = tf.TensorSpec(shape=(self.context_len * self.num_features,), dtype=tf.float32)
            return (x_spec, (y_spec, rec_spec))
        return (x_spec, y_spec)


class PatternPerformanceCallback(keras.callbacks.Callback):
    """Monitors and visualizes N-BEATS performance on a fixed test set."""

    def __init__(self, config: NBeatsTrainingConfig, data_processor: MultiPatternDataProcessor,
                 save_dir: str, model_name: str = "model"):
        super().__init__()
        self.config = config
        self.data_processor = data_processor
        self.save_dir = save_dir
        self.model_name = model_name
        self.training_history: Dict[str, List[float]] = {
            'loss': [], 'val_loss': [], 'forecast_mae': [], 'val_forecast_mae': []
        }
        os.makedirs(save_dir, exist_ok=True)
        self.viz_test_data = self._create_viz_test_set()

    def _create_viz_test_set(self) -> Tuple[np.ndarray, np.ndarray]:
        x_list, y_list = [], []
        for x, y in self.data_processor._test_generator_raw():
            x_list.append(x)
            y_list.append(y)
            if len(x_list) >= self.config.plot_top_k_patterns:
                break
        if not x_list:
            logger.warning("Could not generate viz samples")
            return np.array([]), np.array([])
        return np.array(x_list), np.array(y_list)

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, float]] = None) -> None:
        logs = logs or {}
        self.training_history['loss'].append(logs.get('loss', 0.0))
        self.training_history['val_loss'].append(logs.get('val_loss', 0.0))
        self.training_history['forecast_mae'].append(logs.get('forecast_mae', 0.0))
        self.training_history['val_forecast_mae'].append(logs.get('val_forecast_mae', 0.0))

        if (epoch + 1) % self.config.visualize_every_n_epochs == 0:
            if self.config.create_learning_curves:
                self._plot_learning_curves(epoch)
            if self.config.create_prediction_plots and len(self.viz_test_data[0]) > 0:
                self._plot_prediction_samples(epoch)

    def _plot_learning_curves(self, epoch: int) -> None:
        generate_training_curves(
            history=self.training_history,
            results_dir=self.save_dir,
            filename=f"learning_curves_epoch_{epoch+1:03d}",
        )

    def _plot_prediction_samples(self, epoch: int) -> None:
        test_x, test_y = self.viz_test_data
        predictions_tuple = self.model(test_x, training=False)

        if isinstance(predictions_tuple, (tuple, list)):
            predictions = predictions_tuple[0]
            if isinstance(predictions, tf.Tensor):
                predictions = predictions.numpy()
        else:
            predictions = predictions_tuple.numpy()

        num_plots = min(len(test_x), self.config.plot_top_k_patterns)
        n_cols, n_rows = 3, math.ceil(num_plots / 3)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3 * n_rows), squeeze=False)
        axes = axes.flatten()

        for i in range(num_plots):
            ax = axes[i]
            bk_steps = np.arange(-self.config.backcast_length, 0)
            fc_steps = np.arange(0, self.config.forecast_length)
            ax.plot(bk_steps, test_x[i].flatten(), label='Backcast', color='blue', alpha=0.6)
            ax.plot(fc_steps, test_y[i].flatten(), label='True Future', color='green')
            ax.plot(fc_steps, predictions[i].flatten(), label='Pred Future', color='red', linestyle='--')
            ax.set_title(f'Sample {i+1}')
            if i == 0:
                ax.legend()
            ax.grid(True, alpha=0.3)

        for j in range(num_plots, len(axes)):
            axes[j].axis('off')

        plt.suptitle(f'N-BEATS Predictions (Epoch {epoch + 1})', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f'predictions_epoch_{epoch + 1:03d}.png'))
        plt.close()


class NBeatsTrainer:
    """Orchestrates N-BEATS training, evaluation, and reporting."""

    def __init__(self, config: NBeatsTrainingConfig,
                 generator_config: TimeSeriesGeneratorConfig) -> None:
        self.config = config
        self.generator_config = generator_config
        self.generator = TimeSeriesGenerator(generator_config)

        self.all_patterns = self.generator.get_task_names()
        self.pattern_categories = self.generator.get_task_categories()
        self.pattern_to_category = {
            task: cat
            for cat in self.pattern_categories
            for task in self.generator.get_tasks_by_category(cat)
        }
        self.selected_patterns = self._select_patterns()
        self.processor = MultiPatternDataProcessor(
            config, self.generator, self.selected_patterns, self.pattern_to_category
        )
        self.model: Optional[keras.Model] = None
        self.exp_dir: Optional[str] = None

    def _select_patterns(self) -> List[str]:
        if self.config.target_categories:
            candidates = {
                p for c in self.config.target_categories
                for p in self.generator.get_tasks_by_category(c)
            }
        else:
            candidates = self.all_patterns

        selected: List[str] = []
        cat_counts: Dict[str, int] = {}
        for pattern in sorted(candidates):
            cat = self.pattern_to_category.get(pattern)
            if cat and cat_counts.get(cat, 0) < self.config.max_patterns_per_category:
                selected.append(pattern)
                cat_counts[cat] = cat_counts.get(cat, 0) + 1

        if self.config.max_patterns and len(selected) > self.config.max_patterns:
            selected = random.sample(selected, self.config.max_patterns)

        logger.info(f"Selected {len(selected)} patterns for training")
        return selected

    def create_model(self) -> keras.Model:
        """Create and compile an N-BEATS model."""
        kernel_regularizer = None
        if self.config.kernel_regularizer_l2 > 0:
            kernel_regularizer = keras.regularizers.L2(self.config.kernel_regularizer_l2)

        base_model = create_nbeats_model(
            backcast_length=self.config.backcast_length,
            forecast_length=self.config.forecast_length,
            stack_types=self.config.stack_types,
            nb_blocks_per_stack=self.config.nb_blocks_per_stack,
            hidden_layer_units=self.config.hidden_layer_units,
            activation=self.config.activation,
            use_normalization=self.config.use_normalization,
            kernel_regularizer=kernel_regularizer,
            dropout_rate=self.config.dropout_rate,
            reconstruction_weight=self.config.reconstruction_loss_weight,
            input_dim=self.config.input_dim,
            output_dim=1
        )

        # Learning rate schedule
        if self.config.use_warmup:
            lr_schedule = create_learning_rate_schedule(
                self.config.learning_rate, 'cosine',
                total_epochs=self.config.epochs,
                steps_per_epoch=self.config.steps_per_epoch,
                warmup_steps=self.config.warmup_steps,
                warmup_start_lr=self.config.warmup_start_lr,
            )
        else:
            lr_schedule = self.config.learning_rate

        optimizer = keras.optimizers.get(self.config.optimizer)
        optimizer.learning_rate = lr_schedule
        if self.config.gradient_clip_norm:
            optimizer.clipnorm = self.config.gradient_clip_norm

        # Compile with appropriate loss configuration
        if self.config.reconstruction_loss_weight > 0.0:
            model = base_model
            forecast_loss = (MASELoss(seasonal_periods=self.config.mase_seasonal_periods)
                            if self.config.primary_loss == 'mase_loss'
                            else keras.losses.get(self.config.primary_loss))
            losses = [forecast_loss, keras.losses.MeanAbsoluteError(name="residual_loss", reduction="mean")]
            loss_weights = [1.0, self.config.reconstruction_loss_weight]
            metrics = [
                [keras.metrics.MeanAbsoluteError(name="forecast_mae")],
                [keras.metrics.MeanAbsoluteError(name="residual_mae")]
            ]
        else:
            inputs = keras.Input(shape=(self.config.backcast_length, self.config.input_dim))
            raw_output = base_model(inputs)
            forecast = raw_output[0] if isinstance(raw_output, (list, tuple)) else raw_output
            model = keras.Model(inputs=inputs, outputs=forecast, name="nbeats_forecast_only")
            losses = (MASELoss(seasonal_periods=self.config.mase_seasonal_periods)
                      if self.config.primary_loss == 'mase_loss'
                      else keras.losses.get(self.config.primary_loss))
            loss_weights = None
            metrics = [keras.metrics.MeanAbsoluteError(name="forecast_mae")]

        model.compile(optimizer=optimizer, loss=losses, loss_weights=loss_weights, metrics=metrics)
        return model

    def run_experiment(self) -> Dict[str, Any]:
        """Run the complete training experiment."""
        logger.info("Starting N-BEATS training experiment")
        self.exp_dir = self._create_experiment_dir()
        logger.info(f"Results: {self.exp_dir}")

        data_pipeline = self.processor.prepare_datasets()
        self.model = self.create_model()
        self.model.build((None, self.config.backcast_length, self.config.input_dim))
        logger.info(f"Model params: {self.model.count_params():,}")
        self.model.summary(print_fn=logger.info)

        training_results = self._train_model(data_pipeline, self.exp_dir)
        if self.config.save_results:
            self._save_results(training_results, self.exp_dir)

        return {
            'config': self.config, 'experiment_dir': self.exp_dir,
            'training_results': training_results, 'results_dir': self.exp_dir
        }

    def _create_experiment_dir(self) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_dir = os.path.join(self.config.result_dir, f"{self.config.experiment_name}_{timestamp}")
        os.makedirs(exp_dir, exist_ok=True)
        return exp_dir

    def _train_model(self, data_pipeline: Dict, exp_dir: str) -> Dict[str, Any]:
        viz_dir = os.path.join(exp_dir, 'visualizations')

        callbacks, _ = create_common_callbacks(
            model_name="N-BEATS",
            results_dir_prefix=exp_dir,
            monitor="val_loss",
            patience=25,
            use_lr_schedule=self.config.use_warmup,
            include_terminate_on_nan=True,
            include_analyzer=self.config.perform_deep_analysis,
            analyzer_config=AnalysisConfig(
                analyze_weights=True, analyze_spectral=True,
                analyze_calibration=False, analyze_information_flow=False,
                analyze_training_dynamics=False, verbose=False),
            analyzer_start_epoch=self.config.analysis_start_epoch,
            analyzer_epoch_frequency=self.config.analysis_frequency,
        )
        callbacks.append(PatternPerformanceCallback(self.config, self.processor, viz_dir, "nbeats"))

        history = self.model.fit(
            data_pipeline['train_ds'],
            validation_data=data_pipeline['val_ds'],
            epochs=self.config.epochs,
            steps_per_epoch=self.config.steps_per_epoch,
            validation_steps=data_pipeline['validation_steps'],
            callbacks=callbacks, verbose=1
        )

        logger.info("Evaluating on test set")
        test_results = self.model.evaluate(
            data_pipeline['test_ds'], steps=data_pipeline['test_steps'],
            verbose=1, return_dict=True
        )

        return {
            'history': history.history,
            'test_metrics': {k: float(v) for k, v in test_results.items()},
            'final_epoch': len(history.history['loss'])
        }

    def _save_results(self, results: Dict, exp_dir: str) -> None:
        serializable = {
            'history': results['history'],
            'test_metrics': results['test_metrics'],
            'final_epoch': results['final_epoch'],
            'config': self.config.__dict__
        }
        with open(os.path.join(exp_dir, 'results.json'), 'w') as f:
            json.dump(serializable, f, indent=4, default=json_numpy_default)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="N-BEATS Training")
    parser.add_argument("--experiment_name", type=str, default="nbeats")
    parser.add_argument("--backcast_length", type=int, default=168)
    parser.add_argument("--forecast_length", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--steps_per_epoch", type=int, default=1000)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--optimizer", type=str, default="adamw")
    parser.add_argument("--gradient_clip_norm", type=float, default=1.0)
    parser.add_argument("--no-normalize", dest="normalize_per_instance", action="store_false")
    parser.set_defaults(normalize_per_instance=True)
    parser.add_argument("--stack_types", nargs='+', default=["trend", "seasonality", "generic"])
    parser.add_argument("--hidden_layer_units", type=int, default=256)
    parser.add_argument("--reconstruction_loss_weight", type=float, default=0.5)
    parser.add_argument("--no-warmup", dest="use_warmup", action="store_false")
    parser.set_defaults(use_warmup=True)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--warmup_start_lr", type=float, default=1e-6)
    parser.add_argument("--no-deep-analysis", dest="perform_deep_analysis", action="store_false")
    parser.set_defaults(perform_deep_analysis=True)
    parser.add_argument("--analysis_frequency", type=int, default=10)
    parser.add_argument("--analysis_start_epoch", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=None, help="GPU device index")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_gpu(args.gpu)

    config = NBeatsTrainingConfig(
        experiment_name=args.experiment_name,
        backcast_length=args.backcast_length,
        forecast_length=args.forecast_length,
        stack_types=args.stack_types,
        hidden_layer_units=args.hidden_layer_units,
        epochs=args.epochs,
        batch_size=args.batch_size,
        steps_per_epoch=args.steps_per_epoch,
        learning_rate=args.learning_rate,
        optimizer=args.optimizer,
        gradient_clip_norm=args.gradient_clip_norm,
        normalize_per_instance=args.normalize_per_instance,
        reconstruction_loss_weight=args.reconstruction_loss_weight,
        use_warmup=args.use_warmup,
        warmup_steps=args.warmup_steps,
        warmup_start_lr=args.warmup_start_lr,
        perform_deep_analysis=args.perform_deep_analysis,
        analysis_frequency=args.analysis_frequency,
        analysis_start_epoch=args.analysis_start_epoch
    )

    generator_config = TimeSeriesGeneratorConfig(
        n_samples=10000, random_seed=42, default_noise_level=0.1
    )

    try:
        trainer = NBeatsTrainer(config, generator_config)
        results = trainer.run_experiment()
        logger.info(f"Completed. Results: {results['results_dir']}")
        keras.backend.clear_session()
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception as e:
        logger.error(f"Failed: {e}", exc_info=True)

    os._exit(0)


if __name__ == "__main__":
    main()
