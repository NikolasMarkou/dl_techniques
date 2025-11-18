"""
Comprehensive N-BEATS Training Framework for Multiple Time Series Patterns
"""

import os
import json
import math
import random
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
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.losses.mase_loss import MASELoss
from dl_techniques.optimization.warmup_schedule import WarmupSchedule
from dl_techniques.models.nbeats.model import NBeatsNet, create_nbeats_model
from dl_techniques.datasets.time_series import TimeSeriesConfig, TimeSeriesGenerator

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
    """Configuration dataclass for N-BEATS training on multiple patterns."""
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

    # Model architecture
    stack_types: List[str] = field(
        default_factory=lambda: ["trend", "seasonality", "generic"]
    )
    nb_blocks_per_stack: int = 2
    hidden_layer_units: int = 128
    use_normalization: bool = True
    use_bias: bool = True
    activation: str = "relu"

    # Training configuration
    epochs: int = 150
    batch_size: int = 128
    steps_per_epoch: int = 500
    learning_rate: float = 1e-4
    gradient_clip_norm: float = 1.0
    optimizer: str = 'adam'
    primary_loss: Union[str, keras.losses.Loss] = keras.losses.MeanAbsoluteError(
        reduction="mean"
    )
    mase_seasonal_periods: int = 1

    # Learning rate schedule with warmup
    use_warmup: bool = True
    warmup_steps: int = 1000
    warmup_start_lr: float = 1e-6

    # Regularization
    kernel_regularizer_l2: float = 0.0
    reconstruction_loss_weight: float = 0.0
    dropout_rate: float = 0.0

    # Pattern selection
    max_patterns: Optional[int] = None
    max_patterns_per_category: int = 10
    min_data_length: int = 2000
    normalize_per_instance: bool = False

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

    # Data augmentation configuration
    multiplicative_noise_std: float = 0.0
    additive_noise_std: float = 0.0
    enable_multiplicative_noise: bool = False
    enable_additive_noise: bool = False

    def __post_init__(self) -> None:
        """Validate configuration parameters after initialization."""
        total_ratio = self.train_ratio + self.val_ratio + self.test_ratio
        if not np.isclose(total_ratio, 1.0, atol=1e-6):
            raise ValueError(f"Data ratios must sum to 1.0, got {total_ratio}")
        if self.backcast_length <= 0 or self.forecast_length <= 0:
            raise ValueError("backcast_length and forecast_length must be positive")


class MultiPatternDataProcessor:
    """Streams multi-pattern time series data using Python generators."""

    def __init__(
            self,
            config: NBeatsTrainingConfig,
            generator: TimeSeriesGenerator,
            selected_patterns: List[str],
            pattern_to_category: Dict[str, str]
    ):
        self.config = config
        self.ts_generator = generator
        self.selected_patterns = selected_patterns
        self.pattern_to_category = pattern_to_category
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

    def _training_generator(
        self, forecast_length: int
    ) -> Generator[Tuple[np.ndarray, Union[np.ndarray, Tuple]], None, None]:
        """Create an infinite generator for training data."""
        while True:
            pattern_name = random.choices(
                self.weighted_patterns, self.weights, k=1
            )[0]
            data = self.ts_generator.generate_task_data(pattern_name)
            train_size = int(self.config.train_ratio * len(data))
            train_data = data[:train_size]

            max_start_idx = max(
                len(train_data) - self.config.backcast_length - forecast_length, 0
            )

            if max_start_idx <= 0:
                continue

            start_idx = random.randint(0, max_start_idx)
            backcast = train_data[
                start_idx: start_idx + self.config.backcast_length
            ]
            forecast = train_data[
                start_idx + self.config.backcast_length:
                start_idx + self.config.backcast_length + forecast_length
            ]

            backcast_out = backcast.astype(np.float32).reshape(-1, 1)
            forecast_out = forecast.astype(np.float32).reshape(-1, 1)

            if self.config.reconstruction_loss_weight > 0.0:
                residual_target = np.zeros(
                    (self.config.backcast_length,), dtype=np.float32
                )
                yield backcast_out, (forecast_out, residual_target)
            else:
                yield backcast_out, forecast_out

    def _evaluation_generator(
            self,
            forecast_length: int,
            split: str
    ) -> Generator[Tuple[np.ndarray, Union[np.ndarray, Tuple]], None, None]:
        """Create an infinite repeating generator for validation or test data."""
        if split == 'val':
            start_ratio = self.config.train_ratio
            end_ratio = self.config.train_ratio + self.config.val_ratio
        elif split == 'test':
            start_ratio = self.config.train_ratio + self.config.val_ratio
            end_ratio = 1.0
        else:
            raise ValueError("Split must be 'val' or 'test'")

        all_sequences = []
        for pattern_name in self.selected_patterns:
            data = self.ts_generator.generate_task_data(pattern_name)
            start_idx = int(start_ratio * len(data))
            end_idx = int(end_ratio * len(data))
            split_data = data[start_idx:end_idx]

            seq_len = self.config.backcast_length + forecast_length
            for i in range(len(split_data) - seq_len + 1):
                backcast = split_data[i: i + self.config.backcast_length]
                forecast = split_data[
                    i + self.config.backcast_length: i + seq_len
                ]
                backcast_out = backcast.astype(np.float32).reshape(-1, 1)
                forecast_out = forecast.astype(np.float32).reshape(-1, 1)

                if self.config.reconstruction_loss_weight > 0.0:
                    residual_target = np.zeros(
                        (self.config.backcast_length,), dtype=np.float32
                    )
                    all_sequences.append(
                        (backcast_out, (forecast_out, residual_target))
                    )
                else:
                    all_sequences.append((backcast_out, forecast_out))

        if not all_sequences:
            logger.warning(f"No sequences found for {split} split!")
            dummy_backcast = np.zeros((self.config.backcast_length, 1), dtype=np.float32)
            dummy_forecast = np.zeros((forecast_length, 1), dtype=np.float32)
            while True:
                if self.config.reconstruction_loss_weight > 0.0:
                    dummy_residual = np.zeros(
                        (self.config.backcast_length,), dtype=np.float32
                    )
                    yield dummy_backcast, (dummy_forecast, dummy_residual)
                else:
                    yield dummy_backcast, dummy_forecast
        else:
            logger.info(f"Created {len(all_sequences)} sequences for {split} split")
            idx = 0
            while True:
                yield all_sequences[idx]
                idx = (idx + 1) % len(all_sequences)

    def get_evaluation_steps(self, forecast_length: int, split: str) -> int:
        """Calculate the number of steps for a full evaluation pass."""
        total_samples = 0
        if split == 'val':
            start_ratio = self.config.train_ratio
            end_ratio = self.config.train_ratio + self.config.val_ratio
        elif split == 'test':
            start_ratio = self.config.train_ratio + self.config.val_ratio
            end_ratio = 1.0
        else:
            return 1

        for _ in self.selected_patterns:
            data_len = self.ts_generator.config.n_samples
            start_idx = int(start_ratio * data_len)
            end_idx = int(end_ratio * data_len)
            split_len = end_idx - start_idx
            num_sequences = (
                split_len - self.config.backcast_length - forecast_length + 1
            )
            if num_sequences > 0:
                total_samples += num_sequences

        steps = max(1, math.ceil(total_samples / self.config.batch_size))
        logger.info(f"Calculated {steps} steps for {split} split ({total_samples} samples)")
        return steps

    def _normalize_instance(
            self,
            backcast: tf.Tensor,
            targets: Union[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]]
    ) -> Tuple[tf.Tensor, Union[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]]]:
        """
        Standardize an individual time series instance (z-score normalization).

        This function is designed to be used with `tf.data.Dataset.map`. It
        calculates the mean and standard deviation from the `backcast` ONLY
        and applies this normalization to both the `backcast` and the `forecast`
        part of the `targets`.

        It correctly handles two different structures for `targets`:
        1. A single tensor (the forecast) when reconstruction loss is disabled.
        2. A tuple of tensors (forecast, residual_target) when reconstruction
           loss is enabled. The residual_target is passed through unmodified.
        """
        # Calculate normalization statistics from the backcast only.
        mean = tf.reduce_mean(backcast)
        std = tf.maximum(tf.math.reduce_std(backcast), 1e-7)

        # Normalize the backcast.
        normalized_backcast = (backcast - mean) / std

        # Normalize the forecast component of the targets, preserving structure.
        if self.config.reconstruction_loss_weight > 0.0:
            # Handle the (forecast, residual_target) tuple structure.
            forecast, residual_target = targets
            normalized_forecast = (forecast - mean) / std
            # Return the data in the same nested structure.
            return normalized_backcast, (normalized_forecast, residual_target)
        else:
            # Handle the single forecast tensor structure.
            forecast = targets
            normalized_forecast = (forecast - mean) / std
            return normalized_backcast, normalized_forecast

    def prepare_datasets(self, forecast_length: int) -> Dict[str, Any]:
        """Create the complete tf.data pipeline for a given forecast horizon."""
        if self.config.reconstruction_loss_weight > 0.0:
            logger.info("Adapting data pipeline for reconstruction loss.")
            output_signature = (
                tf.TensorSpec(shape=(self.config.backcast_length, 1), dtype=tf.float32),
                (
                    tf.TensorSpec(shape=(forecast_length, 1), dtype=tf.float32),
                    tf.TensorSpec(shape=(self.config.backcast_length,), dtype=tf.float32)
                )
            )
        else:
            output_signature = (
                tf.TensorSpec(shape=(self.config.backcast_length, 1), dtype=tf.float32),
                tf.TensorSpec(shape=(forecast_length, 1), dtype=tf.float32)
            )

        train_ds = tf.data.Dataset.from_generator(
            generator=lambda: self._training_generator(forecast_length),
            output_signature=output_signature
        )

        val_ds = tf.data.Dataset.from_generator(
            generator=lambda: self._evaluation_generator(forecast_length, 'val'),
            output_signature=output_signature
        )
        test_ds = tf.data.Dataset.from_generator(
            generator=lambda: self._evaluation_generator(forecast_length, 'test'),
            output_signature=output_signature
        )

        if self.config.normalize_per_instance:
            logger.info("Applying per-instance standardization to all datasets.")
            train_ds = train_ds.map(
                self._normalize_instance,
                num_parallel_calls=tf.data.AUTOTUNE
            )
            val_ds = val_ds.map(
                self._normalize_instance,
                num_parallel_calls=tf.data.AUTOTUNE
            )
            test_ds = test_ds.map(
                self._normalize_instance,
                num_parallel_calls=tf.data.AUTOTUNE
            )

        train_ds = (
            train_ds.shuffle(
                self.config.batch_size * 1000
            )
            .batch(self.config.batch_size)
            .shuffle(128)
            .prefetch(tf.data.AUTOTUNE)
        )
        val_ds = val_ds.batch(self.config.batch_size).prefetch(tf.data.AUTOTUNE)
        test_ds = test_ds.batch(self.config.batch_size).prefetch(tf.data.AUTOTUNE)

        val_steps = self.get_evaluation_steps(forecast_length, 'val')
        test_steps = self.get_evaluation_steps(forecast_length, 'test')

        return {
            'train_ds': train_ds,
            'val_ds': val_ds,
            'test_ds': test_ds,
            'validation_steps': val_steps,
            'test_steps': test_steps
        }


class PatternPerformanceCallback(keras.callbacks.Callback):
    """Callback for monitoring and visualizing performance on a fixed test set."""

    def __init__(
            self,
            config: NBeatsTrainingConfig,
            data_processor: MultiPatternDataProcessor,
            forecast_length: int,
            save_dir: str,
            model_name: str = "model"
    ):
        super().__init__()
        self.config = config
        self.data_processor = data_processor
        self.forecast_length = forecast_length
        self.save_dir = save_dir
        self.model_name = model_name
        # Initialize history with explicit, named metrics
        self.training_history = {
            'epoch': [], 'lr': [],
            'loss': [], 'val_loss': [],
            'forecast_mae': [], 'val_forecast_mae': [],
            'residual_mae': [], 'val_residual_mae': [],
        }
        os.makedirs(save_dir, exist_ok=True)
        self.viz_test_data = self._create_viz_test_set()

    def _create_viz_test_set(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generates a small, fixed, and diverse test set for visualizations."""
        logger.info("Creating a diverse, fixed visualization test set...")
        num_samples = self.config.plot_top_k_patterns
        patterns_to_sample = self.data_processor.selected_patterns.copy()
        random.shuffle(patterns_to_sample)

        x_list, y_list, patterns_sampled = [], [], 0
        start_ratio = self.config.train_ratio + self.config.val_ratio
        end_ratio = 1.0

        for pattern_name in patterns_to_sample:
            if patterns_sampled >= num_samples:
                break
            data = self.data_processor.ts_generator.generate_task_data(pattern_name)
            start_idx = int(start_ratio * len(data))
            end_idx = int(end_ratio * len(data))
            test_data = data[start_idx:end_idx]
            sequence_length = self.config.backcast_length + self.forecast_length

            if len(test_data) >= sequence_length:
                max_start = len(test_data) - sequence_length
                sample_start_idx = random.randint(0, max_start)
                backcast = test_data[
                    sample_start_idx: sample_start_idx + self.config.backcast_length
                ]
                forecast = test_data[
                    sample_start_idx + self.config.backcast_length:
                    sample_start_idx + sequence_length
                ]

                if not (np.isnan(backcast).any() or np.isnan(forecast).any()):
                    x_list.append(backcast.astype(np.float32).reshape(-1, 1))
                    y_list.append(forecast.astype(np.float32).reshape(-1, 1))
                    patterns_sampled += 1

        if not x_list:
            logger.warning("Could not generate any visualization samples.")
            return (
                np.array([]).reshape(0, self.config.backcast_length, 1),
                np.array([]).reshape(0, self.forecast_length, 1)
            )

        logger.info(
            f"Successfully created viz set with {len(x_list)} diverse samples."
        )
        return np.array(x_list), np.array(y_list)

    def on_epoch_end(
            self, epoch: int, logs: Optional[Dict[str, float]] = None
    ) -> None:
        """Actions to perform at the end of each epoch."""
        logs = logs or {}

        # Update history with new, clearly named metrics
        self.training_history['epoch'].append(epoch)
        self.training_history['lr'].append(self.model.optimizer.learning_rate)
        self.training_history['loss'].append(logs.get('loss', 0.0))
        self.training_history['val_loss'].append(logs.get('val_loss', 0.0))
        self.training_history['forecast_mae'].append(logs.get('forecast_mae', 0.0))
        self.training_history['val_forecast_mae'].append(logs.get('val_forecast_mae', 0.0))
        # Only append residual mae if it's being tracked (reconstruction is on)
        if 'residual_mae' in logs:
            self.training_history['residual_mae'].append(logs.get('residual_mae', 0.0))
            self.training_history['val_residual_mae'].append(logs.get('val_residual_mae', 0.0))

        if (epoch + 1) % self.config.visualize_every_n_epochs == 0:
            logger.info(
                f"Creating visualizations for {self.model_name} at epoch {epoch + 1}"
            )
            self._create_interim_plots(epoch)

    def _create_interim_plots(self, epoch: int) -> None:
        """Generate and save all configured plots for the current epoch."""
        try:
            os.makedirs(self.save_dir, exist_ok=True)

            if self.config.create_learning_curves:
                self._plot_learning_curves(epoch)
            if self.config.create_prediction_plots and self.viz_test_data[0].shape[0] > 0:
                self._plot_prediction_samples(epoch)
        except Exception as e:
            logger.warning(
                f"Failed to create interim plots for {self.model_name}: {e}"
            )

    # Updated plotting for clearer, more comprehensive metrics
    def _plot_learning_curves(self, epoch: int) -> None:
        """Plot and save comprehensive training and validation learning curves."""
        fig, axes = plt.subplots(2, 2, figsize=(18, 10))
        axes = axes.flatten()
        epochs_range = self.training_history['epoch']

        # Plot 1: Total Loss
        axes[0].plot(epochs_range, self.training_history['loss'], label='Training Loss')
        axes[0].plot(epochs_range, self.training_history['val_loss'], label='Validation Loss')
        axes[0].set_title(f'Total Loss (Epoch {epoch + 1})')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Plot 2: Forecast MAE
        axes[1].plot(epochs_range, self.training_history['forecast_mae'], label='Training Forecast MAE')
        axes[1].plot(epochs_range, self.training_history['val_forecast_mae'], label='Validation Forecast MAE')
        axes[1].set_title(f'Forecast MAE (Epoch {epoch + 1})')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('MAE')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # Plot 3: Residual MAE (only if reconstruction is enabled)
        if self.training_history['residual_mae']:
            axes[2].plot(epochs_range, self.training_history['residual_mae'], label='Training Residual MAE')
            axes[2].plot(epochs_range, self.training_history['val_residual_mae'], label='Validation Residual MAE')
            axes[2].set_title(f'Residual MAE (Epoch {epoch + 1})')
            axes[2].set_xlabel('Epoch')
            axes[2].set_ylabel('MAE')
        else:
            axes[2].text(0.5, 0.5, 'Reconstruction Loss Disabled', ha='center', va='center')
            axes[2].set_title('Residual MAE')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        # Plot 4: Learning Rate
        axes[3].plot(epochs_range, self.training_history['lr'], label='Learning Rate')
        axes[3].set_title(f'Learning Rate Schedule (Epoch {epoch + 1})')
        axes[3].set_xlabel('Epoch')
        axes[3].set_ylabel('LR')
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            os.path.join(self.save_dir, f'learning_curves_epoch_{epoch + 1:03d}.png')
        )
        plt.close()

    def _plot_prediction_samples(self, epoch: int) -> None:
        """Plot and save prediction samples against true values."""
        if self.viz_test_data[0].shape[0] == 0:
            return

        test_x, test_y = self.viz_test_data

        predictions_tuple = self.model(test_x, training=False)
        if isinstance(predictions_tuple, tuple):
            predictions = predictions_tuple[0].numpy()
        else:
            predictions = predictions_tuple.numpy()

        num_plots = min(len(test_x), self.config.plot_top_k_patterns)
        n_cols = 3
        n_rows = math.ceil(num_plots / n_cols)
        # Use squeeze=False to ensure axes is always a 2D array for consistency
        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(18, 4 * n_rows), squeeze=False
        )
        # Flatten the axes array to easily iterate over it, regardless of grid size
        axes = axes.flatten()

        for i in range(num_plots):
            ax = axes[i]
            backcast_x = np.arange(-self.config.backcast_length, 0)
            forecast_x = np.arange(0, self.forecast_length)
            ax.plot(backcast_x, test_x[i].flatten(), label='Input', color='blue')
            ax.plot(forecast_x, test_y[i].flatten(), label='True', color='green')
            ax.plot(
                forecast_x, predictions[i].flatten(), label='Predicted',
                color='red', linestyle='--'
            )
            ax.set_title(f'Sample {i + 1}')
            ax.legend()
            ax.grid(True, alpha=0.3)

        # Turn off any unused subplots
        for j in range(num_plots, len(axes)):
            axes[j].axis('off')

        plt.suptitle(f'Prediction Samples (Epoch {epoch + 1})', fontsize=16)
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.save_dir, f'predictions_epoch_{epoch + 1:03d}.png')
        )
        plt.close()


class NBeatsTrainer:
    """Orchestrates the training, evaluation, and reporting of N-BEATS models."""

    def __init__(
            self, config: NBeatsTrainingConfig, ts_config: TimeSeriesConfig
    ) -> None:
        self.config = config
        self.ts_config = ts_config
        self.generator = TimeSeriesGenerator(ts_config)
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

    def _select_patterns(self) -> List[str]:
        """Select time series patterns based on the configuration."""
        if self.config.target_categories:
            patterns_to_consider = {
                p for c in self.config.target_categories
                for p in self.generator.get_tasks_by_category(c)
            }
        else:
            patterns_to_consider = self.all_patterns

        selected, category_counts = [], {}
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

    def create_model(self, forecast_length: int) -> NBeatsNet:
        """Create and compile an N-BEATS Keras model."""
        kernel_regularizer = None
        if self.config.kernel_regularizer_l2 > 0:
            kernel_regularizer = keras.regularizers.L2(
                self.config.kernel_regularizer_l2
            )

        model = create_nbeats_model(
            backcast_length=self.config.backcast_length,
            forecast_length=forecast_length,
            stack_types=self.config.stack_types,
            nb_blocks_per_stack=self.config.nb_blocks_per_stack,
            hidden_layer_units=self.config.hidden_layer_units,
            activation=self.config.activation,
            use_normalization=self.config.use_normalization,
            kernel_regularizer=kernel_regularizer,
            dropout_rate=self.config.dropout_rate,
            reconstruction_weight=self.config.reconstruction_loss_weight
        )

        if self.config.use_warmup:
            logger.info(
                f"Using learning rate schedule with warmup. "
                f"Warmup steps: {self.config.warmup_steps}, "
                f"Start LR: {self.config.warmup_start_lr}, "
                f"Target LR: {self.config.learning_rate}"
            )
            primary_schedule = keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=self.config.learning_rate,
                decay_steps=self.config.steps_per_epoch * self.config.epochs,
                alpha=0.1
            )
            lr_schedule = WarmupSchedule(
                warmup_steps=self.config.warmup_steps,
                warmup_start_lr=self.config.warmup_start_lr,
                primary_schedule=primary_schedule
            )
        else:
            logger.info(f"Using fixed learning rate: {self.config.learning_rate}")
            lr_schedule = self.config.learning_rate

        optimizer_cls = keras.optimizers.get(self.config.optimizer).__class__
        optimizer = optimizer_cls(
            learning_rate=lr_schedule,
            clipnorm=self.config.gradient_clip_norm
        )
        logger.info(f"Using optimizer: {optimizer.__class__.__name__}")

        if isinstance(self.config.primary_loss, str):
            if self.config.primary_loss == 'mase_loss':
                forecast_loss = MASELoss(
                    seasonal_periods=self.config.mase_seasonal_periods
                )
            else:
                forecast_loss = keras.losses.get(self.config.primary_loss)
        else:
            forecast_loss = self.config.primary_loss

        # Use explicitly named metrics for clear reporting
        forecast_metrics = [
            keras.metrics.MeanAbsoluteError(name="forecast_mae"),
            keras.metrics.MeanSquaredError(name="forecast_mse"),
        ]

        if self.config.reconstruction_loss_weight > 0.0:
            losses = [
                forecast_loss,
                keras.losses.MeanAbsoluteError(reduction="mean", name="residual_loss")
            ]
            loss_weights = [1.0, self.config.reconstruction_loss_weight]
            metrics = [
                forecast_metrics,
                [keras.metrics.MeanAbsoluteError(name="residual_mae")]
            ]
            logger.info(
                f"Using forecast loss + reconstruction loss "
                f"(weight={self.config.reconstruction_loss_weight})"
            )
        else:
            losses = [forecast_loss, None]
            loss_weights = None
            metrics = [forecast_metrics, []]
            logger.info("Using forecast loss only (no reconstruction loss)")

        model.compile(
            optimizer=optimizer,
            loss=losses,
            loss_weights=loss_weights,
            metrics=metrics
        )

        return model

    def run_experiment(self) -> Dict[str, Any]:
        """Execute the full training and evaluation experiment."""
        exp_dir = os.path.join(
            self.config.result_dir,
            f"{self.config.experiment_name}_"
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        os.makedirs(exp_dir, exist_ok=True)
        logger.info(f"Starting N-BEATS Experiment: {exp_dir}")

        results = {}
        horizon = self.config.forecast_length
        logger.info(f"Training Model for Horizon={horizon}")
        data_pipeline = self.processor.prepare_datasets(horizon)
        results[horizon] = self._train_model(data_pipeline, horizon, exp_dir)

        self._save_results(results, exp_dir)
        logger.info("N-BEATS Experiment completed successfully.")
        return {"results_dir": exp_dir, "results": results}

    def _train_model(
            self, data_pipeline: Dict, horizon: int, exp_dir: str
    ) -> Dict[str, Any]:
        """Train and evaluate a single model for a specific horizon."""
        model = self.create_model(horizon)

        model.build((None, self.config.backcast_length, 1))

        logger.info(f"Model created with {model.count_params():,} parameters")

        viz_dir = os.path.join(exp_dir, f'visualizations_h{horizon}')
        performance_cb = PatternPerformanceCallback(
            self.config, self.processor, horizon, viz_dir, "nbeats_model"
        )

        model_path = os.path.join(exp_dir, f'best_model_h{horizon}.keras')

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=30,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                filepath=model_path,
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            performance_cb,
            keras.callbacks.TerminateOnNaN()
        ]

        logger.info(
            f"Training with {self.config.steps_per_epoch} steps/epoch and "
            f"validating with {data_pipeline['validation_steps']} steps."
        )

        history = model.fit(
            data_pipeline['train_ds'],
            validation_data=data_pipeline['val_ds'],
            epochs=self.config.epochs,
            steps_per_epoch=self.config.steps_per_epoch,
            validation_steps=data_pipeline['validation_steps'],
            callbacks=callbacks,
            verbose=1
        )

        logger.info("Evaluating on test set...")
        test_results_list = model.evaluate(
            data_pipeline['test_ds'],
            steps=data_pipeline['test_steps'],
            verbose=1,
            return_dict=False # Ensure list output for zipping
        )

        # Robustly parse test results into a dictionary
        test_metrics = dict(zip(model.metrics_names, test_results_list))
        logger.info(f"Test Set Metrics: {test_metrics}")

        final_epoch = len(history.history['loss'])

        # Prepare final results dictionary
        final_results = {
            'history': history.history,
            'final_epoch': final_epoch,
            'test_metrics': {k: float(v) for k, v in test_metrics.items()}
        }

        return final_results

    def _save_results(self, results: Dict, exp_dir: str) -> None:
        """Save the final experiment results to a JSON file."""
        logger.info(f"Saving results to {exp_dir}")
        serializable_results = {}

        for h, r in results.items():
            history_serializable = {
                k: [float(v) for v in val]
                for k, val in r['history'].items()
            }
            serializable_results[h] = {
                'history': history_serializable,
                'final_epoch': r['final_epoch'],
                'test_metrics': r['test_metrics']
            }

        with open(os.path.join(exp_dir, 'results.json'), 'w') as f:
            json.dump(serializable_results, f, indent=4)


def main() -> None:
    """Main function to configure and run the N-BEATS training experiment."""
    config = NBeatsTrainingConfig(
        experiment_name="nbeats",
        activation="relu",
        backcast_length=104,
        forecast_length=12,
        stack_types=["trend", "seasonality", "generic"],
        nb_blocks_per_stack=3,
        hidden_layer_units=256,
        use_normalization=True,
        normalize_per_instance=False,
        use_bias=True,
        max_patterns_per_category=50,
        epochs=200,
        batch_size=128,
        steps_per_epoch=1000,
        learning_rate=1e-4,
        use_warmup=True,
        warmup_steps=5000,
        warmup_start_lr=1e-6,
        gradient_clip_norm=1.0,
        optimizer='adam',
        primary_loss=keras.losses.MeanAbsoluteError(
            reduction="mean"
        ),
        dropout_rate=0.1,
        kernel_regularizer_l2=1e-5,
        reconstruction_loss_weight=0.5, # Set > 0.0 to see residual metrics
        visualize_every_n_epochs=5,
        plot_top_k_patterns=12,
    )

    ts_config = TimeSeriesConfig(
        n_samples=1000,
        random_seed=42
    )

    try:
        trainer = NBeatsTrainer(config, ts_config)
        results = trainer.run_experiment()
        logger.info(f"Experiment completed! Results saved to: {results['results_dir']}")
    except Exception as e:
        logger.error(f"Experiment failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()