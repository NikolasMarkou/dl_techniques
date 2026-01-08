"""
Orchestrate the end-to-end training and evaluation lifecycle of the N-BEATS forecasting architecture.

Architecture
------------
This component manages the Neural Basis Expansion Analysis for Time Series (N-BEATS),
a pure deep neural architecture utilizing backward and forward residual links. The model
is organized into a sequence of "stacks," where each stack comprises multiple basic blocks.

The architecture employs a specific "doubly residual" topology designed to decompose
time series into distinct components:
1. Backcast Residual: Each block attempts to reconstruct the input history. The residual
   (the unexplained signal) is subtracted from the input and passed to the next block.
   This acts as a filter, allowing subsequent layers to focus solely on signal dynamics
   not yet captured.
2. Forecast Aggregation: Each block predicts a partial forecast vector. The final global
   prediction is the element-wise sum of these partial forecasts.

This implementation orchestrates a "meta-learning" style workflow, training a single
global model on diverse synthetic patterns (e.g., Trend, Seasonality, Volatility) via
instance-wise normalization. It supports both "Generic" stacks (fully learnable basis)
and "Interpretable" stacks (fixed functional forms) to separate trend and seasonality.

Integration
-----------
Updated to use the unified `dl_techniques.datasets.time_series` module for
synthetic data generation and normalization.

References
----------
Oreshkin, B. N., Carpov, D., Chapados, N., & Bengio, Y. (2019).
N-BEATS: Neural basis expansion analysis for interpretable time series forecasting.
In International Conference on Learning Representations (ICLR).
"""

import os
import json
import math
import random
import argparse
import sys
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

# ---------------------------------------------------------------------
# Local Imports
# ---------------------------------------------------------------------

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

# ---------------------------------------------------------------------

plt.style.use('default')
sns.set_palette("husl")


def set_random_seeds(seed: int = 42) -> None:
    """Set random seeds for major libraries to ensure reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    keras.utils.set_random_seed(seed)
    tf.random.set_seed(seed)


set_random_seeds(42)


# ---------------------------------------------------------------------


@dataclass
class NBeatsTrainingConfig:
    """Configuration dataclass for N-BEATS training."""
    # General experiment configuration
    result_dir: str = "results"
    save_results: bool = True
    experiment_name: str = "nbeats"

    # Pattern selection configuration
    target_categories: Optional[List[str]] = None

    # Data configuration
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # N-BEATS specific configuration
    backcast_length: int = 168
    forecast_length: int = 24
    input_dim: int = 1

    # Model architecture
    stack_types: List[str] = field(
        default_factory=lambda: ["trend", "seasonality", "generic"]
    )
    nb_blocks_per_stack: int = 3
    hidden_layer_units: int = 128
    use_normalization: bool = True
    use_bias: bool = True
    activation: str = "gelu"

    # Training configuration
    epochs: int = 150
    batch_size: int = 128
    steps_per_epoch: int = 1000
    learning_rate: float = 1e-4
    gradient_clip_norm: float = 1.0
    optimizer: str = 'adam'
    primary_loss: Union[str, keras.losses.Loss] = "mase_loss"
    mase_seasonal_periods: int = 1

    # Learning rate schedule with warmup
    use_warmup: bool = True
    warmup_steps: int = 5000
    warmup_start_lr: float = 1e-6

    # Regularization
    kernel_regularizer_l2: float = 1e-5
    reconstruction_loss_weight: float = 0.5
    dropout_rate: float = 0.1

    # Pattern selection & Data Processing
    max_patterns: Optional[int] = None
    max_patterns_per_category: int = 100
    normalize_per_instance: bool = True

    # Category weights for balanced sampling
    category_weights: Dict[str, float] = field(default_factory=lambda: {
        "trend": 1.0, "seasonal": 1.0, "composite": 1.2,
        "financial": 1.5, "weather": 1.3, "biomedical": 1.2,
        "industrial": 1.3, "intermittent": 1.0, "volatility": 1.1,
        "regime": 1.2, "structural": 1.1
    })

    # Visualization configuration
    visualize_every_n_epochs: int = 5
    save_interim_plots: bool = True
    plot_top_k_patterns: int = 12
    create_learning_curves: bool = True
    create_prediction_plots: bool = True

    # Deep Analysis Configuration
    perform_deep_analysis: bool = True
    analysis_frequency: int = 10
    analysis_start_epoch: int = 1

    def __post_init__(self) -> None:
        """Validate configuration parameters after initialization."""
        total_ratio = self.train_ratio + self.val_ratio + self.test_ratio
        if not np.isclose(total_ratio, 1.0, atol=1e-6):
            raise ValueError(f"Data ratios must sum to 1.0, got {total_ratio}")
        if self.backcast_length <= 0 or self.forecast_length <= 0:
            raise ValueError("backcast_length and forecast_length must be positive")


def _fill_nans(data: np.ndarray) -> np.ndarray:
    """
    Forward fill NaNs in numpy array.

    :param data: Input array potentially containing NaNs.
    :return: Array with NaNs forward-filled.
    """
    mask = np.isnan(data)
    idx = np.where(~mask, np.arange(mask.shape[0]), 0)
    np.maximum.accumulate(idx, axis=0, out=idx)
    out = data[idx]
    out[np.isnan(out)] = 0
    return out


class MultiPatternDataProcessor:
    """
    Robust data processor for N-BEATS using the unified TimeSeriesGenerator and Normalizer.

    Features:
    - Infinite streaming generator for Training (maximum diversity).
    - Pre-computed in-memory arrays for Validation/Test (eliminates end-of-epoch lag).
    - Robust per-instance normalization using TimeSeriesNormalizer.
    """

    def __init__(
            self,
            config: NBeatsTrainingConfig,
            generator: TimeSeriesGenerator,
            selected_patterns: List[str],
            pattern_to_category: Dict[str, str],
            num_features: int = 1
    ):
        self.config = config
        self.ts_generator = generator
        self.selected_patterns = selected_patterns
        self.pattern_to_category = pattern_to_category
        self.num_features = num_features
        self.weighted_patterns, self.weights = self._prepare_weighted_sampling()

    def _prepare_weighted_sampling(self) -> Tuple[List[str], List[float]]:
        """Prepare lists for weighted random pattern selection."""
        patterns, weights = [], []
        for pattern_name in self.selected_patterns:
            category = self.pattern_to_category.get(pattern_name, "unknown")
            weight = self.config.category_weights.get(category, 1.0)
            patterns.append(pattern_name)
            weights.append(weight)

        total_weight = sum(weights)
        if total_weight > 0:
            normalized_weights = [w / total_weight for w in weights]
        else:
            num_patterns = len(patterns)
            normalized_weights = [1.0 / num_patterns] * num_patterns
        return patterns, normalized_weights

    def _safe_normalize(self, series: np.ndarray) -> np.ndarray:
        """
        Normalize series instance using TimeSeriesNormalizer.
        Includes clipping to prevent extreme outliers from destabilizing training.

        :param series: Input numpy array (n_timesteps, features) or (n_timesteps,).
        :return: Normalized array as float32.
        """
        series = np.clip(series, -1e6, 1e6)

        if self.config.normalize_per_instance:
            normalizer = TimeSeriesNormalizer(method=NormalizationMethod.STANDARD)
            if np.isnan(series).any():
                series = _fill_nans(series)
            series = normalizer.fit_transform(series)

        series = np.clip(series, -10.0, 10.0)
        return series.astype(np.float32)

    def _training_generator(
        self
    ) -> Generator[Tuple[np.ndarray, Union[np.ndarray, Tuple]], None, None]:
        """
        Infinite generator for training data with high diversity.
        Uses a buffer to mix patterns and reduce generator overhead.
        """
        patterns_to_mix = 50
        windows_per_pattern = 5
        buffer: List[Tuple[np.ndarray, Union[np.ndarray, Tuple]]] = []

        total_window_len = self.config.backcast_length + self.config.forecast_length

        while True:
            if not buffer:
                selected_patterns = random.choices(
                    self.weighted_patterns, self.weights, k=patterns_to_mix
                )

                for pattern_name in selected_patterns:
                    data = self.ts_generator.generate_task_data(pattern_name)

                    if len(data) < total_window_len or not np.isfinite(data).all():
                        continue

                    train_size = int(self.config.train_ratio * len(data))
                    train_data = data[:train_size]

                    max_start_idx = len(train_data) - total_window_len
                    if max_start_idx <= 0:
                        continue

                    for _ in range(windows_per_pattern):
                        start_idx = random.randint(0, max_start_idx)
                        window = train_data[start_idx: start_idx + total_window_len]

                        window = self._safe_normalize(window)

                        backcast = window[:self.config.backcast_length].reshape(
                            -1, self.num_features
                        )
                        forecast = window[self.config.backcast_length:].reshape(
                            -1, self.num_features
                        )

                        if self.config.reconstruction_loss_weight > 0.0:
                            residual_target = backcast.flatten()
                            buffer.append((backcast, (forecast, residual_target)))
                        else:
                            buffer.append((backcast, forecast))

                random.shuffle(buffer)

            if buffer:
                yield buffer.pop()

    def _generate_fixed_dataset(
        self, split: str, num_samples: int
    ) -> Tuple[np.ndarray, Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]]:
        """
        Pre-generate a fixed dataset in memory to prevent generator lag.

        :param split: 'val' or 'test'.
        :param num_samples: Total number of samples to generate.
        :return: Tuple of (Inputs, Targets) as numpy arrays.
        """
        logger.info(f"Pre-computing {split} dataset ({num_samples} samples)...")

        backcasts: List[np.ndarray] = []
        forecasts: List[np.ndarray] = []
        residuals: List[np.ndarray] = []

        total_window_len = self.config.backcast_length + self.config.forecast_length
        samples_collected = 0
        pattern_cycle = 0

        while samples_collected < num_samples:
            pattern_idx = pattern_cycle % len(self.selected_patterns)
            pattern_name = self.selected_patterns[pattern_idx]
            pattern_cycle += 1

            data = self.ts_generator.generate_task_data(pattern_name)

            if len(data) < total_window_len or not np.isfinite(data).all():
                continue

            train_end = int(self.config.train_ratio * len(data))
            val_end = train_end + int(self.config.val_ratio * len(data))

            if split == 'val':
                split_data = data[train_end:val_end]
            else:
                split_data = data[val_end:]

            max_start_idx = len(split_data) - total_window_len
            if max_start_idx <= 0:
                continue

            start_idx = random.randint(0, max_start_idx)
            window = split_data[start_idx: start_idx + total_window_len]
            window = self._safe_normalize(window)

            backcast = window[:self.config.backcast_length].reshape(
                -1, self.num_features
            )
            forecast = window[self.config.backcast_length:].reshape(
                -1, self.num_features
            )

            backcasts.append(backcast)
            forecasts.append(forecast)

            if self.config.reconstruction_loss_weight > 0.0:
                residuals.append(backcast.flatten())

            samples_collected += 1

        x = np.array(backcasts, dtype=np.float32)
        y_forecast = np.array(forecasts, dtype=np.float32)

        if self.config.reconstruction_loss_weight > 0.0:
            y_residual = np.array(residuals, dtype=np.float32)
            return x, (y_forecast, y_residual)

        return x, y_forecast

    def _test_generator_raw(
        self
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Helper specifically for visualization callback to pull fresh test samples.
        """
        total_window_len = self.config.backcast_length + self.config.forecast_length

        viz_patterns = self.selected_patterns.copy()
        random.shuffle(viz_patterns)

        for pattern_name in viz_patterns:
            data = self.ts_generator.generate_task_data(pattern_name)

            if len(data) < total_window_len or not np.isfinite(data).all():
                continue

            start_test = int(
                (self.config.train_ratio + self.config.val_ratio) * len(data)
            )
            test_data = data[start_test:]

            if len(test_data) < total_window_len:
                continue

            max_start_idx = len(test_data) - total_window_len
            start_idx = random.randint(0, max_start_idx)

            window = test_data[start_idx: start_idx + total_window_len]
            window = self._safe_normalize(window)

            yield (
                window[:self.config.backcast_length].reshape(-1, self.num_features),
                window[self.config.backcast_length:].reshape(-1, self.num_features)
            )

    def prepare_datasets(self) -> Dict[str, Any]:
        """Create tf.data.Datasets."""
        x_shape = (self.config.backcast_length, self.num_features)
        y_shape = (self.config.forecast_length, self.num_features)

        if self.config.reconstruction_loss_weight > 0.0:
            rec_shape = (self.config.backcast_length * self.num_features,)
            output_signature = (
                tf.TensorSpec(shape=x_shape, dtype=tf.float32),
                (
                    tf.TensorSpec(shape=y_shape, dtype=tf.float32),
                    tf.TensorSpec(shape=rec_shape, dtype=tf.float32)
                )
            )
        else:
            output_signature = (
                tf.TensorSpec(shape=x_shape, dtype=tf.float32),
                tf.TensorSpec(shape=y_shape, dtype=tf.float32)
            )

        train_ds = tf.data.Dataset.from_generator(
            self._training_generator, output_signature=output_signature
        ).batch(self.config.batch_size).prefetch(tf.data.AUTOTUNE)

        validation_steps = max(50, len(self.selected_patterns))
        test_steps = max(20, len(self.selected_patterns))

        num_val_samples = validation_steps * self.config.batch_size
        num_test_samples = test_steps * self.config.batch_size

        val_data = self._generate_fixed_dataset('val', num_val_samples)
        if self.config.reconstruction_loss_weight > 0.0:
            val_x, (val_y_forecast, val_y_residual) = val_data
            val_ds = tf.data.Dataset.from_tensor_slices(
                (val_x, (val_y_forecast, val_y_residual))
            )
        else:
            val_x, val_y = val_data
            val_ds = tf.data.Dataset.from_tensor_slices((val_x, val_y))
        val_ds = val_ds.batch(self.config.batch_size).cache().prefetch(tf.data.AUTOTUNE)

        test_data = self._generate_fixed_dataset('test', num_test_samples)
        if self.config.reconstruction_loss_weight > 0.0:
            test_x, (test_y_forecast, test_y_residual) = test_data
            test_ds = tf.data.Dataset.from_tensor_slices(
                (test_x, (test_y_forecast, test_y_residual))
            )
        else:
            test_x, test_y = test_data
            test_ds = tf.data.Dataset.from_tensor_slices((test_x, test_y))
        test_ds = test_ds.batch(self.config.batch_size).cache().prefetch(tf.data.AUTOTUNE)

        return {
            'train_ds': train_ds,
            'val_ds': val_ds,
            'test_ds': test_ds,
            'validation_steps': validation_steps,
            'test_steps': test_steps
        }


class PatternPerformanceCallback(keras.callbacks.Callback):
    """Callback for monitoring and visualizing performance on a fixed test set."""

    def __init__(
            self,
            config: NBeatsTrainingConfig,
            data_processor: MultiPatternDataProcessor,
            save_dir: str,
            model_name: str = "model"
    ):
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
        """Generate a diverse visualization test set."""
        logger.info("Creating a diverse, fixed visualization test set...")

        x_list: List[np.ndarray] = []
        y_list: List[np.ndarray] = []

        test_gen = self.data_processor._test_generator_raw()

        for _ in range(self.config.plot_top_k_patterns):
            try:
                x, y = next(test_gen)
                x_list.append(x)
                y_list.append(y)
            except StopIteration:
                break

        if not x_list:
            logger.warning("Could not generate viz samples.")
            return np.array([]), np.array([])

        logger.info(f"Created {len(x_list)} diverse samples from different patterns.")
        return np.array(x_list), np.array(y_list)

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, float]] = None) -> None:
        logs = logs or {}
        self.training_history['loss'].append(logs.get('loss', 0.0))
        self.training_history['val_loss'].append(logs.get('val_loss', 0.0))
        self.training_history['forecast_mae'].append(logs.get('forecast_mae', 0.0))
        self.training_history['val_forecast_mae'].append(logs.get('val_forecast_mae', 0.0))

        if (epoch + 1) % self.config.visualize_every_n_epochs == 0:
            self._create_interim_plots(epoch)

    def _create_interim_plots(self, epoch: int) -> None:
        if self.config.create_learning_curves:
            self._plot_learning_curves(epoch)
        if self.config.create_prediction_plots and len(self.viz_test_data[0]) > 0:
            self._plot_prediction_samples(epoch)

    def _plot_learning_curves(self, epoch: int) -> None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].plot(self.training_history['loss'], label='Train Loss')
        axes[0].plot(self.training_history['val_loss'], label='Val Loss')
        axes[0].set_title(f'Total Loss (Epoch {epoch+1})')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(self.training_history['forecast_mae'], label='Train MAE')
        axes[1].plot(self.training_history['val_forecast_mae'], label='Val MAE')
        axes[1].set_title('Forecast MAE')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            os.path.join(self.save_dir, f'learning_curves_epoch_{epoch+1:03d}.png')
        )
        plt.close()

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
        n_cols = 3
        n_rows = math.ceil(num_plots / n_cols)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3 * n_rows), squeeze=False)
        axes = axes.flatten()

        for i in range(num_plots):
            ax = axes[i]
            backcast_steps = np.arange(-self.config.backcast_length, 0)
            forecast_steps = np.arange(0, self.config.forecast_length)

            ax.plot(
                backcast_steps, test_x[i].flatten(),
                label='Backcast', color='blue', alpha=0.6
            )
            ax.plot(
                forecast_steps, test_y[i].flatten(),
                label='True Future', color='green'
            )
            ax.plot(
                forecast_steps, predictions[i].flatten(),
                label='Pred Future', color='red', linestyle='--'
            )

            ax.set_title(f'Sample {i+1}')
            if i == 0:
                ax.legend()
            ax.grid(True, alpha=0.3)

        for j in range(num_plots, len(axes)):
            axes[j].axis('off')

        plt.suptitle(f'N-BEATS Predictions (Epoch {epoch + 1})', fontsize=14)
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.save_dir, f'predictions_epoch_{epoch + 1:03d}.png')
        )
        plt.close()


class NBeatsTrainer:
    """Orchestrates the training, evaluation, and reporting of N-BEATS models."""

    def __init__(
            self, config: NBeatsTrainingConfig, generator_config: TimeSeriesGeneratorConfig
    ) -> None:
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
            patterns_to_consider = {
                p for c in self.config.target_categories
                for p in self.generator.get_tasks_by_category(c)
            }
        else:
            patterns_to_consider = self.all_patterns

        selected: List[str] = []
        category_counts: Dict[str, int] = {}

        for pattern in sorted(patterns_to_consider):
            category = self.pattern_to_category.get(pattern)
            if (category and
                    category_counts.get(category, 0) <
                    self.config.max_patterns_per_category):
                selected.append(pattern)
                category_counts[category] = category_counts.get(category, 0) + 1

        if self.config.max_patterns and len(selected) > self.config.max_patterns:
            selected = random.sample(selected, self.config.max_patterns)

        logger.info(f"Selected {len(selected)} patterns for training.")
        return selected

    def create_model(self) -> keras.Model:
        """Create and compile an N-BEATS Keras model."""
        kernel_regularizer = None
        if self.config.kernel_regularizer_l2 > 0:
            kernel_regularizer = keras.regularizers.L2(
                self.config.kernel_regularizer_l2
            )

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

        if self.config.use_warmup:
            total_steps = self.config.epochs * self.config.steps_per_epoch
            primary_steps = max(1, total_steps - self.config.warmup_steps)

            primary_schedule = keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=self.config.learning_rate,
                decay_steps=primary_steps,
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

        if self.config.reconstruction_loss_weight > 0.0:
            model = base_model

            if self.config.primary_loss == 'mase_loss':
                forecast_loss = MASELoss(
                    seasonal_periods=self.config.mase_seasonal_periods
                )
            else:
                forecast_loss = keras.losses.get(self.config.primary_loss)

            losses = [
                forecast_loss,
                keras.losses.MeanAbsoluteError(name="residual_loss", reduction="mean")
            ]
            loss_weights = [1.0, self.config.reconstruction_loss_weight]
            metrics = [
                [keras.metrics.MeanAbsoluteError(name="forecast_mae")],
                [keras.metrics.MeanAbsoluteError(name="residual_mae")]
            ]
        else:
            inputs = keras.Input(
                shape=(self.config.backcast_length, self.config.input_dim)
            )
            raw_output = base_model(inputs)

            if isinstance(raw_output, (list, tuple)):
                forecast = raw_output[0]
            else:
                forecast = raw_output

            model = keras.Model(
                inputs=inputs, outputs=forecast, name="nbeats_forecast_only"
            )

            if self.config.primary_loss == 'mase_loss':
                losses = MASELoss(seasonal_periods=self.config.mase_seasonal_periods)
            else:
                losses = keras.losses.get(self.config.primary_loss)

            loss_weights = None
            metrics = [keras.metrics.MeanAbsoluteError(name="forecast_mae")]

        model.compile(
            optimizer=optimizer,
            loss=losses,
            loss_weights=loss_weights,
            metrics=metrics
        )

        return model

    def run_experiment(self) -> Dict[str, Any]:
        """Run the complete training experiment."""
        logger.info("Starting N-BEATS training experiment")
        self.exp_dir = self._create_experiment_dir()
        logger.info(f"Results will be saved to: {self.exp_dir}")

        data_pipeline = self.processor.prepare_datasets()

        self.model = self.create_model()
        self.model.build((None, self.config.backcast_length, self.config.input_dim))
        logger.info("Model created successfully")
        logger.info(f"Model parameters: {self.model.count_params():,}")
        self.model.summary(print_fn=logger.info)

        training_results = self._train_model(data_pipeline, self.exp_dir)

        if self.config.save_results:
            self._save_results(training_results, self.exp_dir)

        return {
            'config': self.config,
            'experiment_dir': self.exp_dir,
            'training_results': training_results,
            'results_dir': self.exp_dir
        }

    def _create_experiment_dir(self) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = f"{self.config.experiment_name}_{timestamp}"
        exp_dir = os.path.join(self.config.result_dir, exp_name)
        os.makedirs(exp_dir, exist_ok=True)
        return exp_dir

    def _train_model(self, data_pipeline: Dict, exp_dir: str) -> Dict[str, Any]:
        logger.info("Starting model training...")

        viz_dir = os.path.join(exp_dir, 'visualizations')
        performance_cb = PatternPerformanceCallback(
            self.config, self.processor, viz_dir, "nbeats"
        )

        model_path = os.path.join(exp_dir, 'best_model.keras')

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=25, restore_best_weights=True, verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                filepath=model_path, monitor='val_loss', save_best_only=True, verbose=1
            ),
            performance_cb,
            keras.callbacks.TerminateOnNaN()
        ]

        if not self.config.use_warmup:
            callbacks.append(keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=10, min_lr=1e-7
            ))

        if self.config.perform_deep_analysis:
            logger.info("Adding Deep Model Analysis callback.")

            analysis_config = AnalysisConfig(
                analyze_weights=True,
                analyze_spectral=True,
                analyze_calibration=False,
                analyze_information_flow=False,
                analyze_training_dynamics=False,
                verbose=False
            )

            analysis_dir = os.path.join(exp_dir, 'deep_analysis')

            analyzer_cb = EpochAnalyzerCallback(
                output_dir=analysis_dir,
                analysis_config=analysis_config,
                start_epoch=self.config.analysis_start_epoch,
                epoch_frequency=self.config.analysis_frequency,
                model_name="N-BEATS"
            )
            callbacks.append(analyzer_cb)

        history = self.model.fit(
            data_pipeline['train_ds'],
            validation_data=data_pipeline['val_ds'],
            epochs=self.config.epochs,
            steps_per_epoch=self.config.steps_per_epoch,
            validation_steps=data_pipeline['validation_steps'],
            callbacks=callbacks,
            verbose=1
        )

        logger.info("Evaluating on test set...")
        test_results = self.model.evaluate(
            data_pipeline['test_ds'],
            steps=data_pipeline['test_steps'],
            verbose=1,
            return_dict=True
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

        def default(o: Any) -> str:
            if isinstance(o, (np.integer, np.floating)):
                return float(o)
            return str(o)

        with open(os.path.join(exp_dir, 'results.json'), 'w') as f:
            json.dump(serializable, f, indent=4, default=default)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for N-BEATS training."""
    parser = argparse.ArgumentParser(
        description="N-BEATS Training (Unified TimeSeriesGenerator)"
    )

    parser.add_argument(
        "--experiment_name", type=str, default="nbeats", help="Name for logging."
    )
    parser.add_argument("--backcast_length", type=int, default=168)
    parser.add_argument("--forecast_length", type=int, default=24)

    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--steps_per_epoch", type=int, default=1000)

    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--optimizer", type=str, default="adamw")
    parser.add_argument("--gradient_clip_norm", type=float, default=1.0)

    parser.add_argument(
        "--no-normalize", dest="normalize_per_instance", action="store_false",
        help="Disable per-instance normalization."
    )
    parser.set_defaults(normalize_per_instance=True)

    parser.add_argument(
        "--stack_types", nargs='+', default=["trend", "seasonality", "generic"]
    )
    parser.add_argument("--hidden_layer_units", type=int, default=256)
    parser.add_argument(
        "--reconstruction_loss_weight", type=float, default=0.5,
        help="Weight for backcast reconstruction loss."
    )

    parser.add_argument(
        "--no-warmup", dest="use_warmup", action="store_false",
        help="Disable learning rate warmup."
    )
    parser.set_defaults(use_warmup=True)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--warmup_start_lr", type=float, default=1e-6)

    parser.add_argument(
        "--no-deep-analysis", dest="perform_deep_analysis", action="store_false",
        help="Disable periodic deep model analysis."
    )
    parser.set_defaults(perform_deep_analysis=True)
    parser.add_argument("--analysis_frequency", type=int, default=10)
    parser.add_argument("--analysis_start_epoch", type=int, default=1)

    return parser.parse_args()


def main() -> None:
    """Main function to configure and run the N-BEATS training experiment."""
    args = parse_args()

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
        n_samples=10000,
        random_seed=42,
        default_noise_level=0.1
    )

    trainer = NBeatsTrainer(config, generator_config)
    results = trainer.run_experiment()
    logger.info(f"Experiment completed! Results saved to: {results['results_dir']}")
    sys.exit(0)


if __name__ == "__main__":
    main()