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

from train.common import setup_gpu, create_callbacks as create_common_callbacks, generate_training_curves
from dl_techniques.utils.logger import logger
from dl_techniques.losses.mase_loss import MASELoss
from dl_techniques.models.nbeats import create_nbeats_model
from dl_techniques.optimization.warmup_schedule import WarmupSchedule
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
    random.seed(seed)
    np.random.seed(seed)
    keras.utils.set_random_seed(seed)
    tf.random.set_seed(seed)


set_random_seeds(42)


@dataclass
class NBeatsTrainingConfig:
    """Configuration for N-BEATS training."""
    result_dir: str = "results"
    save_results: bool = True
    experiment_name: str = "nbeats"
    target_categories: Optional[List[str]] = None

    # Data splits
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15

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

    # Training
    epochs: int = 150
    batch_size: int = 128
    steps_per_epoch: int = 1000
    learning_rate: float = 1e-4
    gradient_clip_norm: float = 1.0
    optimizer: str = 'adam'
    primary_loss: Union[str, keras.losses.Loss] = "mase_loss"
    mase_seasonal_periods: int = 1

    # Warmup schedule
    use_warmup: bool = True
    warmup_steps: int = 5000
    warmup_start_lr: float = 1e-6

    # Regularization
    kernel_regularizer_l2: float = 1e-5
    reconstruction_loss_weight: float = 0.5
    dropout_rate: float = 0.1

    # Pattern selection
    max_patterns: Optional[int] = None
    max_patterns_per_category: int = 100
    normalize_per_instance: bool = True

    # Category weights
    category_weights: Dict[str, float] = field(default_factory=lambda: {
        "trend": 1.0, "seasonal": 1.0, "composite": 1.2,
        "financial": 1.5, "weather": 1.3, "biomedical": 1.2,
        "industrial": 1.3, "intermittent": 1.0, "volatility": 1.1,
        "regime": 1.2, "structural": 1.1
    })

    # Visualization
    visualize_every_n_epochs: int = 5
    save_interim_plots: bool = True
    plot_top_k_patterns: int = 12
    create_learning_curves: bool = True
    create_prediction_plots: bool = True

    # Deep analysis
    perform_deep_analysis: bool = True
    analysis_frequency: int = 10
    analysis_start_epoch: int = 1

    def __post_init__(self) -> None:
        total_ratio = self.train_ratio + self.val_ratio + self.test_ratio
        if not np.isclose(total_ratio, 1.0, atol=1e-6):
            raise ValueError(f"Data ratios must sum to 1.0, got {total_ratio}")
        if self.backcast_length <= 0 or self.forecast_length <= 0:
            raise ValueError("backcast_length and forecast_length must be positive")


def _fill_nans(data: np.ndarray) -> np.ndarray:
    """Forward fill NaNs in numpy array."""
    mask = np.isnan(data)
    idx = np.where(~mask, np.arange(mask.shape[0]), 0)
    np.maximum.accumulate(idx, axis=0, out=idx)
    out = data[idx]
    out[np.isnan(out)] = 0
    return out


class MultiPatternDataProcessor:
    """
    Data processor for N-BEATS with infinite streaming training generator,
    pre-computed val/test datasets, and per-instance normalization.
    """

    def __init__(self, config: NBeatsTrainingConfig, generator: TimeSeriesGenerator,
                 selected_patterns: List[str], pattern_to_category: Dict[str, str],
                 num_features: int = 1):
        self.config = config
        self.ts_generator = generator
        self.selected_patterns = selected_patterns
        self.pattern_to_category = pattern_to_category
        self.num_features = num_features
        self.weighted_patterns, self.weights = self._prepare_weighted_sampling()

    def _prepare_weighted_sampling(self) -> Tuple[List[str], List[float]]:
        patterns, weights = [], []
        for name in self.selected_patterns:
            category = self.pattern_to_category.get(name, "unknown")
            weights.append(self.config.category_weights.get(category, 1.0))
            patterns.append(name)
        total = sum(weights)
        if total > 0:
            weights = [w / total for w in weights]
        else:
            weights = [1.0 / len(patterns)] * len(patterns)
        return patterns, weights

    def _safe_normalize(self, series: np.ndarray) -> np.ndarray:
        """Normalize with clipping for stability."""
        series = np.clip(series, -1e6, 1e6)
        if self.config.normalize_per_instance:
            normalizer = TimeSeriesNormalizer(method=NormalizationMethod.STANDARD)
            if np.isnan(series).any():
                series = _fill_nans(series)
            series = normalizer.fit_transform(series)
        series = np.clip(series, -10.0, 10.0)
        return series.astype(np.float32)

    def _training_generator(self) -> Generator[Tuple[np.ndarray, Union[np.ndarray, Tuple]], None, None]:
        """Infinite generator with buffered pattern mixing."""
        patterns_to_mix, windows_per_pattern = 50, 5
        buffer: List[Tuple[np.ndarray, Union[np.ndarray, Tuple]]] = []
        total_len = self.config.backcast_length + self.config.forecast_length

        while True:
            if not buffer:
                for name in random.choices(self.weighted_patterns, self.weights, k=patterns_to_mix):
                    data = self.ts_generator.generate_task_data(name)
                    if len(data) < total_len or not np.isfinite(data).all():
                        continue
                    train_data = data[:int(self.config.train_ratio * len(data))]
                    max_start = len(train_data) - total_len
                    if max_start <= 0:
                        continue
                    for _ in range(windows_per_pattern):
                        start = random.randint(0, max_start)
                        window = self._safe_normalize(train_data[start:start + total_len])
                        backcast = window[:self.config.backcast_length].reshape(-1, self.num_features)
                        forecast = window[self.config.backcast_length:].reshape(-1, self.num_features)
                        if self.config.reconstruction_loss_weight > 0.0:
                            buffer.append((backcast, (forecast, backcast.flatten())))
                        else:
                            buffer.append((backcast, forecast))
                random.shuffle(buffer)
            if buffer:
                yield buffer.pop()

    def _generate_fixed_dataset(self, split: str, num_samples: int
                                ) -> Tuple[np.ndarray, Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]]:
        """Pre-generate fixed dataset for val/test."""
        logger.info(f"Pre-computing {split} dataset ({num_samples} samples)")
        backcasts, forecasts, residuals = [], [], []
        total_len = self.config.backcast_length + self.config.forecast_length
        collected, cycle = 0, 0

        while collected < num_samples:
            name = self.selected_patterns[cycle % len(self.selected_patterns)]
            cycle += 1
            data = self.ts_generator.generate_task_data(name)
            if len(data) < total_len or not np.isfinite(data).all():
                continue

            train_end = int(self.config.train_ratio * len(data))
            val_end = train_end + int(self.config.val_ratio * len(data))
            split_data = data[train_end:val_end] if split == 'val' else data[val_end:]

            max_start = len(split_data) - total_len
            if max_start <= 0:
                continue
            window = self._safe_normalize(split_data[random.randint(0, max_start):][:total_len])
            backcast = window[:self.config.backcast_length].reshape(-1, self.num_features)
            forecast = window[self.config.backcast_length:].reshape(-1, self.num_features)
            backcasts.append(backcast)
            forecasts.append(forecast)
            if self.config.reconstruction_loss_weight > 0.0:
                residuals.append(backcast.flatten())
            collected += 1

        x = np.array(backcasts, dtype=np.float32)
        y_forecast = np.array(forecasts, dtype=np.float32)
        if self.config.reconstruction_loss_weight > 0.0:
            return x, (y_forecast, np.array(residuals, dtype=np.float32))
        return x, y_forecast

    def _test_generator_raw(self) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """Fresh test samples for visualization."""
        total_len = self.config.backcast_length + self.config.forecast_length
        viz_patterns = self.selected_patterns.copy()
        random.shuffle(viz_patterns)

        for name in viz_patterns:
            data = self.ts_generator.generate_task_data(name)
            if len(data) < total_len or not np.isfinite(data).all():
                continue
            test_data = data[int((self.config.train_ratio + self.config.val_ratio) * len(data)):]
            if len(test_data) < total_len:
                continue
            start = random.randint(0, len(test_data) - total_len)
            window = self._safe_normalize(test_data[start:start + total_len])
            yield (
                window[:self.config.backcast_length].reshape(-1, self.num_features),
                window[self.config.backcast_length:].reshape(-1, self.num_features)
            )

    def prepare_datasets(self) -> Dict[str, Any]:
        """Create tf.data.Datasets for train/val/test."""
        x_shape = (self.config.backcast_length, self.num_features)
        y_shape = (self.config.forecast_length, self.num_features)

        if self.config.reconstruction_loss_weight > 0.0:
            rec_shape = (self.config.backcast_length * self.num_features,)
            output_signature = (
                tf.TensorSpec(shape=x_shape, dtype=tf.float32),
                (tf.TensorSpec(shape=y_shape, dtype=tf.float32),
                 tf.TensorSpec(shape=rec_shape, dtype=tf.float32))
            )
        else:
            output_signature = (
                tf.TensorSpec(shape=x_shape, dtype=tf.float32),
                tf.TensorSpec(shape=y_shape, dtype=tf.float32)
            )

        train_ds = tf.data.Dataset.from_generator(
            self._training_generator, output_signature=output_signature
        ).batch(self.config.batch_size).prefetch(tf.data.AUTOTUNE)

        val_steps = max(50, len(self.selected_patterns))
        test_steps = max(20, len(self.selected_patterns))

        val_data = self._generate_fixed_dataset('val', val_steps * self.config.batch_size)
        if self.config.reconstruction_loss_weight > 0.0:
            val_x, (val_yf, val_yr) = val_data
            val_ds = tf.data.Dataset.from_tensor_slices((val_x, (val_yf, val_yr)))
        else:
            val_x, val_y = val_data
            val_ds = tf.data.Dataset.from_tensor_slices((val_x, val_y))
        val_ds = val_ds.batch(self.config.batch_size).cache().prefetch(tf.data.AUTOTUNE)

        test_data = self._generate_fixed_dataset('test', test_steps * self.config.batch_size)
        if self.config.reconstruction_loss_weight > 0.0:
            test_x, (test_yf, test_yr) = test_data
            test_ds = tf.data.Dataset.from_tensor_slices((test_x, (test_yf, test_yr)))
        else:
            test_x, test_y = test_data
            test_ds = tf.data.Dataset.from_tensor_slices((test_x, test_y))
        test_ds = test_ds.batch(self.config.batch_size).cache().prefetch(tf.data.AUTOTUNE)

        return {
            'train_ds': train_ds, 'val_ds': val_ds, 'test_ds': test_ds,
            'validation_steps': val_steps, 'test_steps': test_steps
        }


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
            total_steps = self.config.epochs * self.config.steps_per_epoch
            primary_schedule = keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=self.config.learning_rate,
                decay_steps=max(1, total_steps - self.config.warmup_steps),
                alpha=0.01
            )
            lr_schedule = WarmupSchedule(
                warmup_steps=self.config.warmup_steps,
                warmup_start_lr=self.config.warmup_start_lr,
                primary_schedule=primary_schedule
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
        def default(o: Any) -> str:
            if isinstance(o, (np.integer, np.floating)):
                return float(o)
            return str(o)

        serializable = {
            'history': results['history'],
            'test_metrics': results['test_metrics'],
            'final_epoch': results['final_epoch'],
            'config': self.config.__dict__
        }
        with open(os.path.join(exp_dir, 'results.json'), 'w') as f:
            json.dump(serializable, f, indent=4, default=default)


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
