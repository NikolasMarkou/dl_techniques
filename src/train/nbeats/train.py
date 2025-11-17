"""
Comprehensive N-BEATS Training Framework for Multiple Time Series Patterns

This module provides a sophisticated, production-ready training framework for
N-BEATS models. It leverages a streaming data pipeline with tf.data.Dataset
to train a single model on an arbitrary number of dynamically generated time
series patterns, ensuring memory efficiency and scalability. It also includes
a flexible learning rate scheduler with a warmup phase to improve training
stability.

Classes
-------
WarmupSchedule
    Learning rate schedule with a linear warmup phase.

NBeatsTrainingConfig
    Comprehensive configuration dataclass for all training parameters.

MultiPatternDataProcessor
    Advanced data processing pipeline that uses Python generators to stream
    sequences for training and evaluation, handling on-the-fly sampling and
    balancing.

PatternPerformanceCallback
    Callback for monitoring performance and visualizing prediction samples on a
    fixed test set.

NBeatsTrainer
    Main training orchestrator for multi-pattern N-BEATS training with
    comprehensive experiment management and performance analysis.
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
# Use a non-interactive backend for matplotlib to prevent issues on servers
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

# Set plotting style for consistency
plt.style.use('default')
sns.set_palette("husl")


def set_random_seeds(seed: int = 42) -> None:
    """Set random seeds for major libraries to ensure reproducibility.

    :param seed: The integer seed value to use.
    :type seed: int
    """
    random.seed(seed)
    np.random.seed(seed)
    keras.utils.set_random_seed(seed)
    tf.random.set_seed(seed)


# Initialize random seeds at the start of the script.
set_random_seeds(42)


# ---------------------------------------------------------------------


@dataclass
class NBeatsTrainingConfig:
    """Configuration dataclass for N-BEATS training on multiple patterns.

    This class centralizes all hyperparameters and settings for the experiment,
    from data generation and splitting to model architecture and training
    procedures.

    :ivar result_dir: Directory to save experiment results.
    :ivar save_results: If True, saves results to disk.
    :ivar experiment_name: Name for the experiment run.
    :ivar target_categories: Specific categories of patterns to train on. If None, uses all.
    :ivar train_ratio: Proportion of data for training.
    :ivar val_ratio: Proportion of data for validation.
    :ivar test_ratio: Proportion of data for testing.
    :ivar backcast_length: Length of the input sequence (lookback window).
    :ivar forecast_length: Length of the output sequence to predict.
    :ivar forecast_horizons: List of different forecast lengths to train models for.
    :ivar stack_types: N-BEATS stack architecture (e.g., trend, seasonality).
    :ivar nb_blocks_per_stack: Number of blocks within each N-BEATS stack.
    :ivar hidden_layer_units: Number of units in the fully connected layers of each block.
    :ivar use_normalization: If True, applies batch normalization within the model.
    :ivar use_bias: If True, uses bias terms in model layers.
    :ivar epochs: Maximum number of training epochs.
    :ivar batch_size: Number of samples per training batch.
    :ivar steps_per_epoch: Number of training steps per epoch.
    :ivar learning_rate: The target learning rate for the optimizer.
    :ivar gradient_clip_norm: The value to clip gradients to, preventing exploding gradients.
    :ivar optimizer: The optimizer to use (e.g., 'adamw').
    :ivar primary_loss: The primary loss function for training. Defaults to 'mase_loss'.
    :ivar mase_seasonal_periods: Seasonal period for MASE loss calculation.
    :ivar use_warmup: If True, enables a learning rate warmup phase.
    :ivar warmup_steps: Number of steps for the linear warmup.
    :ivar warmup_start_lr: The initial learning rate at the start of the warmup.
    :ivar kernel_regularizer_l2: L2 regularization factor for kernel weights.
    :ivar reconstruction_loss_weight: Weight for the reconstruction loss on the final residual.
    :ivar dropout_rate: Dropout rate for regularization.
    :ivar max_patterns: The absolute maximum number of patterns to use. If None, uses all available.
    :ivar max_patterns_per_category: Maximum patterns to select from each category.
    :ivar min_data_length: Minimum length required for a time series to be included.
    :ivar normalize_per_instance: If True, applies z-score normalization to each sample.
    :ivar category_weights: Weights for sampling patterns from different categories.
    :ivar visualize_every_n_epochs: Frequency (in epochs) for generating visualization plots.
    :ivar save_interim_plots: If True, saves plots during training.
    :ivar plot_top_k_patterns: Number of prediction samples to visualize.
    :ivar create_learning_curves: If True, generates learning curve plots.
    :ivar create_prediction_plots: If True, generates prediction sample plots.
    :ivar multiplicative_noise_std: Standard deviation for multiplicative noise augmentation.
    :ivar additive_noise_std: Standard deviation for additive noise augmentation.
    :ivar enable_multiplicative_noise: If True, enables multiplicative noise.
    :ivar enable_additive_noise: If True, enables additive noise.
    """
    # General experiment configuration
    result_dir: str = "results"
    save_results: bool = True
    experiment_name: str = "nbeats"

    # Pattern selection configuration
    target_categories: Optional[List[str]] = None

    # Data configuration
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1

    # N-BEATS specific configuration
    backcast_length: int = 168
    forecast_length: int = 24
    forecast_horizons: List[int] = field(default_factory=lambda: [24])

    # Model architecture
    stack_types: List[str] = field(
        default_factory=lambda: ["trend", "seasonality", "generic"]
    )
    nb_blocks_per_stack: int = 2
    hidden_layer_units: int = 128
    use_normalization: bool = False
    use_bias: bool = False
    activation: str = "relu"

    # Training configuration
    epochs: int = 150
    batch_size: int = 128
    steps_per_epoch: int = 500
    learning_rate: float = 1e-5
    gradient_clip_norm: float = 1.0
    optimizer: str = 'adamw'
    # Default loss is now MASE. Can be overridden with another loss object or string.
    primary_loss: Union[str, keras.losses.Loss] = keras.losses.MeanAbsoluteError(reduction="mean")
    mase_seasonal_periods: int = 1  # Seasonal period for MASE loss

    # Learning rate schedule with warmup
    use_warmup: bool = False
    warmup_steps: int = 1000
    warmup_start_lr: float = 1e-6

    # Regularization
    kernel_regularizer_l2: float = 1e-5
    reconstruction_loss_weight: float = 0.0
    dropout_rate: float = 0.1

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
    visualize_every_n_epochs: int = 1
    save_interim_plots: bool = True
    plot_top_k_patterns: int = 8
    create_learning_curves: bool = True
    create_prediction_plots: bool = True

    # Data augmentation configuration
    multiplicative_noise_std: float = 0.01
    additive_noise_std: float = 0.01
    enable_multiplicative_noise: bool = False
    enable_additive_noise: bool = False

    def __post_init__(self) -> None:
        """Validate configuration parameters after initialization."""
        total_ratio = self.train_ratio + self.val_ratio + self.test_ratio
        if not np.isclose(total_ratio, 1.0, atol=1e-6):
            raise ValueError(
                f"Data ratios must sum to 1.0, got {total_ratio}")
        if self.backcast_length <= 0 or self.forecast_length <= 0:
            raise ValueError(
                "backcast_length and forecast_length must be positive")
        if not self.use_normalization:
            logger.warning(
                "Normalization is disabled. External normalization is required.")


class MultiPatternDataProcessor:
    """Streams multi-pattern time series data using Python generators.

    This class is responsible for creating efficient `tf.data.Dataset` pipelines
    that can handle a large number of time series patterns without loading them
    all into memory. It uses Python generators to dynamically sample patterns
    and extract sequences on-the-fly.

    :param config: The main training configuration object.
    :type config: NBeatsTrainingConfig
    :param generator: An instance of `TimeSeriesGenerator` to produce raw data.
    :type generator: TimeSeriesGenerator
    :param selected_patterns: A list of pattern names to be used.
    :type selected_patterns: List[str]
    :param pattern_to_category: A dictionary mapping pattern names to their categories.
    :type pattern_to_category: Dict[str, str]
    """

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
        """Prepare lists for weighted random pattern selection.

        This method assigns a sampling weight to each selected pattern based on
        its category, allowing for a balanced or prioritized selection during training.
        This is a tricky point because it's crucial for preventing the model from
        biasing towards categories with more patterns.

        :return: A tuple containing the list of patterns and their corresponding
                 normalized sampling weights.
        :rtype: Tuple[List[str], List[float]]
        """
        patterns, weights = [], []
        for pattern_name in self.selected_patterns:
            category = self.pattern_to_category.get(pattern_name, "unknown")
            weight = self.config.category_weights.get(category, 1.0)
            patterns.append(pattern_name)
            weights.append(weight)

        total_weight = sum(weights)
        if total_weight > 0:
            # Normalize weights to form a probability distribution.
            normalized_weights = [w / total_weight for w in weights]
        else:
            # Fallback to uniform distribution if weights sum to zero.
            num_patterns = len(patterns)
            normalized_weights = [1.0 / num_patterns] * num_patterns
        return patterns, normalized_weights

    def _training_generator(
        self, forecast_length: int
    ) -> Generator[Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]], None, None]:
        """
        Create an infinite generator for training data with weighted sampling.

        This generator continuously samples patterns based on category weights,
        generates a fresh time series, and yields a randomly selected pair of
        (backcast, (forecast, zeros_for_residual)).

        :param forecast_length: The length of the forecast horizon.
        :yield: A tuple containing the backcast and a tuple of (forecast, zero_target).
        """
        # Target for the residual is a zero vector matching residual shape.
        zeros_for_residual = np.zeros(
            (self.config.backcast_length,), dtype=np.float32
        )

        while True:
            pattern_name = random.choices(
                self.weighted_patterns, self.weights, k=1
            )[0]
            data = self.ts_generator.generate_task_data(pattern_name)
            train_size = int(self.config.train_ratio * len(data))
            train_data = data[:train_size]

            max_start_idx = max(
                len(train_data) -
                self.config.backcast_length - forecast_length
            , 0)

            start_idx = random.randint(0, max_start_idx)
            backcast = train_data[
                start_idx: start_idx + self.config.backcast_length
            ]
            forecast = train_data[
                start_idx + self.config.backcast_length:
                start_idx + self.config.backcast_length + forecast_length
            ]

            yield (
                backcast.astype(np.float32),
                (forecast.astype(np.float32), zeros_for_residual)
            )

    def _evaluation_generator(
            self,
            forecast_length: int,
            split: str
    ) -> Generator[Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]], None, None]:
        """Create a finite generator for validation or test data.

        This generator iterates through every selected pattern and yields all
        possible sequences from the specified data split ('val' or 'test'),
        formatted as (backcast, (forecast, zeros_for_residual)).

        :param forecast_length: The length of the forecast horizon.
        :param split: The data split to use, either 'val' or 'test'.
        :yield: A tuple containing the backcast and a tuple of (forecast, zero_target).
        :raises ValueError: If `split` is not 'val' or 'test'.
        """
        if split == 'val':
            start_ratio = self.config.train_ratio
            end_ratio = self.config.train_ratio + self.config.val_ratio
        elif split == 'test':
            start_ratio = self.config.train_ratio + self.config.val_ratio
            end_ratio = 1.0
        else:
            raise ValueError("Split must be 'val' or 'test'")

        zeros_for_residual = np.zeros(
            (self.config.backcast_length,), dtype=np.float32
        )

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
                yield (
                    backcast.astype(np.float32),
                    (forecast.astype(np.float32), zeros_for_residual)
                )

    def get_evaluation_steps(self, forecast_length: int, split: str) -> int:
        """Calculate the number of steps for a full evaluation pass.

        :param forecast_length: The length of the forecast horizon.
        :type forecast_length: int
        :param split: The data split, either 'val' or 'test'.
        :type split: str
        :return: The total number of batches for one full pass.
        :rtype: int
        """
        total_samples = 0
        if split == 'val':
            start_ratio = self.config.train_ratio
            end_ratio = self.config.train_ratio + self.config.val_ratio
        elif split == 'test':
            start_ratio = self.config.train_ratio + self.config.val_ratio
            end_ratio = 1.0
        else:
            return 0

        for _ in self.selected_patterns:
            data_len = self.ts_generator.config.n_samples
            start_idx = int(start_ratio * data_len)
            end_idx = int(end_ratio * data_len)
            split_len = end_idx - start_idx
            num_sequences = (split_len - self.config.backcast_length -
                             forecast_length + 1)
            if num_sequences > 0:
                total_samples += num_sequences
        return math.ceil(total_samples / self.config.batch_size)

    def _normalize_instance(
            self, backcast: tf.Tensor, targets: Tuple[tf.Tensor, tf.Tensor]
    ) -> Tuple[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]]:
        """Standardize an individual time series instance (z-score normalization).

        **Tricky Point**: This normalization is applied per-instance, meaning
        each backcast is normalized using its *own* mean and standard deviation.
        This is a common technique in deep learning for time series to make
        the model robust to shifts in level and scale between different series
        or different segments of the same series. An epsilon is added to the
        standard deviation to prevent division by zero for flat (constant) inputs.

        :param backcast: The input tensor.
        :param targets: A tuple containing the (forecast, zero_target) tensors.
        :return: The normalized backcast and forecast, with the zero_target passed through.
        """
        forecast, zero_target = targets
        mean = tf.math.reduce_mean(backcast)
        std = tf.math.reduce_std(backcast)
        epsilon = 1e-6  # Add epsilon for numerical stability
        normalized_backcast = (backcast - mean) / (std + epsilon)
        normalized_forecast = (forecast - mean) / (std + epsilon)
        return normalized_backcast, (normalized_forecast, zero_target)

    def _apply_noise_augmentation(
            self, x: tf.Tensor, y: Tuple[tf.Tensor, tf.Tensor]
    ) -> Tuple[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]]:
        """Apply additive and/or multiplicative noise augmentation to the input.

        :param x: The input backcast tensor.
        :param y: The tuple of target tensors.
        :return: The augmented input `x` and original targets `y`.
        """
        augmented_x = x
        if (self.config.enable_multiplicative_noise and
                self.config.multiplicative_noise_std > 0):
            mult_noise = tf.random.normal(
                tf.shape(x),
                mean=1.0,
                stddev=self.config.multiplicative_noise_std,
                dtype=x.dtype
            )
            augmented_x *= mult_noise
        if (self.config.enable_additive_noise and
                self.config.additive_noise_std > 0):
            add_noise = tf.random.normal(
                tf.shape(augmented_x),
                mean=0.0,
                stddev=self.config.additive_noise_std,
                dtype=augmented_x.dtype
            )
            augmented_x += add_noise
        return augmented_x, y

    def prepare_datasets(self, forecast_length: int) -> Dict[str, Any]:
        """Create the complete tf.data pipeline for a given forecast horizon.

        This method assembles the full `tf.data.Dataset` pipeline, integrating
        the generators, normalization, shuffling, batching, and prefetching
        for optimal performance.

        :param forecast_length: The forecast horizon length.
        :return: A dictionary containing the train, validation, and test datasets,
                 along with the number of steps for validation and testing.
        """
        # The output signature now reflects the nested structure for the labels.
        output_signature = (
            tf.TensorSpec(shape=(self.config.backcast_length, 1), dtype=tf.float32),
            (
                tf.TensorSpec(shape=(forecast_length, 1), dtype=tf.float32),
                tf.TensorSpec(shape=(self.config.backcast_length,), dtype=tf.float32)
            )
        )


        # Create datasets from generators
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

        # Apply per-instance normalization if configured
        if self.config.normalize_per_instance:
            logger.info("Applying per-instance standardization to all datasets.")
            train_ds = train_ds.map(self._normalize_instance,
                                    num_parallel_calls=tf.data.AUTOTUNE)
            val_ds = val_ds.map(self._normalize_instance,
                                  num_parallel_calls=tf.data.AUTOTUNE)
            test_ds = test_ds.map(self._normalize_instance,
                                  num_parallel_calls=tf.data.AUTOTUNE)

        # Shuffle, batch, and apply augmentations to the training set
        train_ds = (
            train_ds
            .batch(self.config.batch_size)
        )

        # Prefetch for performance
        train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
        val_ds = val_ds.batch(self.config.batch_size).prefetch(
            tf.data.AUTOTUNE)
        test_ds = test_ds.batch(self.config.batch_size).prefetch(
            tf.data.AUTOTUNE)

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
    """Callback for monitoring and visualizing performance on a fixed test set.

    This callback performs two main functions at regular intervals:
    1. Plots learning curves (loss and MAE) for training and validation.
    2. Generates prediction plots on a fixed, diverse set of test samples
       to provide a consistent visual reference of model performance over time.

    :param config: The main training configuration object.
    :param data_processor: The data processor instance.
    :param forecast_length: The forecast horizon.
    :param save_dir: Directory to save visualization plots.
    :param model_name: Name of the model for logging purposes.
    """

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
        self.training_history = {
            'epoch': [], 'loss': [], 'val_loss': [], 'mae': [], 'val_mae': []
        }
        os.makedirs(save_dir, exist_ok=True)
        # **Tricky Point**: Create a small, *fixed* visualization dataset at the
        # beginning. This is crucial for comparing prediction plots across
        # different epochs, as the underlying data remains the same.
        self.viz_test_data = self._create_viz_test_set()

    def _create_viz_test_set(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generates a small, fixed, and diverse test set for visualizations.

        :return: A tuple of NumPy arrays (X, y) for visualization.
        :rtype: Tuple[np.ndarray, np.ndarray]
        """
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
            data = self.data_processor.ts_generator.generate_task_data(
                pattern_name)
            start_idx, end_idx = int(start_ratio * len(data)), int(
                end_ratio * len(data))
            test_data = data[start_idx:end_idx]
            sequence_length = self.config.backcast_length + self.forecast_length

            if len(test_data) >= sequence_length:
                max_start = len(test_data) - sequence_length
                sample_start_idx = random.randint(0, max_start)
                backcast = test_data[
                           sample_start_idx: sample_start_idx + self.config.backcast_length]
                forecast = test_data[sample_start_idx + self.config.backcast_length:
                                     sample_start_idx + sequence_length]

                if not (np.isnan(backcast).any() or np.isnan(forecast).any()):
                    x_list.append(backcast.astype(np.float32))
                    y_list.append(forecast.astype(np.float32))
                    patterns_sampled += 1

        if not x_list:
            logger.warning("Could not generate any visualization samples.")
            return np.array([]), np.array([])

        logger.info(
            f"Successfully created viz set with {len(x_list)} diverse samples.")
        return np.array(x_list), np.array(y_list)

    def on_epoch_end(self, epoch: int,
                     logs: Optional[Dict[str, float]] = None) -> None:
        """Actions to perform at the end of each epoch.

        :param epoch: The current epoch number.
        :type epoch: int
        :param logs: Dictionary of metrics from the ended epoch.
        :type logs: Optional[Dict[str, float]]
        """
        logs = logs or {}
        for key in self.training_history:
            if key != 'epoch':
                self.training_history[key].append(logs.get(key, 0.0))
        self.training_history['epoch'].append(epoch)

        if (epoch + 1) % self.config.visualize_every_n_epochs == 0:
            logger.info(
                f"Creating visualizations for {self.model_name} at epoch {epoch + 1}")
            self._create_interim_plots(epoch)

    def _create_interim_plots(self, epoch: int) -> None:
        """Generate and save all configured plots for the current epoch."""
        try:
            # FIX: Add a defensive check to ensure the visualization directory
            # exists right before saving the plots. This prevents "No such file
            # or directory" errors in certain environments.
            os.makedirs(self.save_dir, exist_ok=True)

            if self.config.create_learning_curves:
                self._plot_learning_curves(epoch)
            if self.config.create_prediction_plots:
                self._plot_prediction_samples(epoch)
        except Exception as e:
            logger.warning(
                f"Failed to create interim plots for {self.model_name}: {e}")

    def _plot_learning_curves(self, epoch: int) -> None:
        """Plot and save the training and validation learning curves."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        epochs = self.training_history['epoch']

        axes[0].plot(epochs, self.training_history['loss'],
                     label='Training Loss')
        axes[0].plot(epochs, self.training_history['val_loss'],
                     label='Validation Loss')
        axes[0].set_title(f'Loss Curves (Epoch {epoch + 1})')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(epochs, self.training_history['mae'], label='Training MAE')
        axes[1].plot(epochs, self.training_history['val_mae'],
                     label='Validation MAE')
        axes[1].set_title(f'MAE Curves (Epoch {epoch + 1})')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir,
                                 f'learning_curves_epoch_{epoch + 1:03d}.png'))
        plt.close()

    def _plot_prediction_samples(self, epoch: int) -> None:
        """Plot and save prediction samples against true values."""
        if self.viz_test_data[0].shape[0] == 0:
            return

        test_x, test_y = self.viz_test_data
        # The model returns a tuple (forecast, residual). We only need the forecast.
        predictions, _ = self.model(test_x, training=False)
        num_plots = min(len(test_x), self.config.plot_top_k_patterns)
        n_cols = 3
        n_rows = math.ceil(num_plots / n_cols)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4 * n_rows))
        axes = axes.flatten()

        for i in range(num_plots):
            ax = axes[i]
            backcast_x = np.arange(-self.config.backcast_length, 0)
            forecast_x = np.arange(0, self.forecast_length)
            ax.plot(backcast_x, test_x[i], label='Input', color='blue')
            ax.plot(forecast_x, test_y[i], label='True', color='green')
            ax.plot(forecast_x, predictions[i], label='Predicted',
                    color='red', linestyle='--')
            ax.set_title(f'Sample {i + 1}')
            ax.legend()
            ax.grid(True, alpha=0.3)

        # Turn off empty subplots
        for j in range(num_plots, len(axes)):
            axes[j].axis('off')

        plt.suptitle(f'Prediction Samples (Epoch {epoch + 1})', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir,
                                 f'predictions_epoch_{epoch + 1:03d}.png'))
        plt.close()


class NBeatsTrainer:
    """Orchestrates the training, evaluation, and reporting of N-BEATS models.

    This is the main class that coordinates the entire experiment, from data
    selection and preprocessing to model creation, training, and result saving.

    :param config: The main training configuration object.
    :type config: NBeatsTrainingConfig
    :param ts_config: The configuration for the time series generator.
    :type ts_config: TimeSeriesConfig
    """

    def __init__(self, config: NBeatsTrainingConfig,
                 ts_config: TimeSeriesConfig) -> None:
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
            config, self.generator, self.selected_patterns,
            self.pattern_to_category
        )

    def _select_patterns(self) -> List[str]:
        """Select time series patterns based on the configuration.

        :return: A list of selected pattern names.
        :rtype: List[str]
        """
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
                    category_counts.get(category,
                                        0) < self.config.max_patterns_per_category):
                selected.append(pattern)
                category_counts[category] = category_counts.get(category, 0) + 1

        if self.config.max_patterns and len(selected) > self.config.max_patterns:
            selected = random.sample(selected, self.config.max_patterns)

        logger.info(f"Selected {len(selected)} patterns for training.")
        return selected

    def create_model(self, forecast_length: int) -> NBeatsNet:
        """Create and compile an N-BEATS Keras model with learning rate schedule.

        :param forecast_length: The forecast horizon for the model.
        :type forecast_length: int
        :return: A compiled Keras model.
        :rtype: NBeatsNet
        """
        kernel_regularizer = (
            keras.regularizers.L2(self.config.kernel_regularizer_l2)
            if self.config.kernel_regularizer_l2 > 0 else None
        )

        # --- Learning Rate Schedule Setup ---
        if self.config.use_warmup:
            logger.info(
                f"Using learning rate schedule with warmup. "
                f"Warmup steps: {self.config.warmup_steps}, "
                f"Start LR: {self.config.warmup_start_lr}, "
                f"Target LR: {self.config.learning_rate}"
            )
            primary_schedule = keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=self.config.learning_rate,
                decay_steps=100000,
                decay_rate=1.0
            )
            lr_schedule = WarmupSchedule(
                warmup_steps=self.config.warmup_steps,
                warmup_start_lr=self.config.warmup_start_lr,
                primary_schedule=primary_schedule
            )
        else:
            logger.info(f"Using fixed learning rate: {self.config.learning_rate}")
            lr_schedule = self.config.learning_rate

        # --- Loss Function Setup ---
        # This is a tricky point. We dynamically instantiate the loss function
        # based on the configuration. This allows us to easily switch between
        # 'mase_loss' and other standard Keras losses without changing the code.
        if self.config.primary_loss == 'mase_loss':
            loss_instance = MASELoss(
                seasonal_periods=self.config.mase_seasonal_periods)
            logger.info(
                "Using MASELoss with seasonal_periods="
                f"{self.config.mase_seasonal_periods}")
        else:
            loss_instance = self.config.primary_loss
            logger.info(f"Using standard loss: {loss_instance}")

        return create_nbeats_model(
            backcast_length=self.config.backcast_length,
            forecast_length=forecast_length,
            stack_types=self.config.stack_types,
            nb_blocks_per_stack=self.config.nb_blocks_per_stack,
            hidden_layer_units=self.config.hidden_layer_units,
            use_normalization=self.config.use_normalization,
            kernel_regularizer=kernel_regularizer,
            dropout_rate=self.config.dropout_rate,
            optimizer=self.config.optimizer,
            loss=loss_instance,  # Pass the instantiated loss object
            learning_rate=lr_schedule,
            gradient_clip_norm=self.config.gradient_clip_norm,
            reconstruction_weight=self.config.reconstruction_loss_weight
        )

    def run_experiment(self) -> Dict[str, Any]:
        """Execute the full training and evaluation experiment.

        :return: A dictionary containing the path to results and the results data.
        :rtype: Dict[str, Any]
        """
        exp_dir = os.path.join(
            self.config.result_dir,
            f"{self.config.experiment_name}_"
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        os.makedirs(exp_dir, exist_ok=True)
        logger.info(f"Starting N-BEATS Experiment: {exp_dir}")

        results = {}
        for horizon in self.config.forecast_horizons:
            logger.info(
                f"Training Model for Horizon={horizon}")
            data_pipeline = self.processor.prepare_datasets(horizon)
            results[horizon] = self._train_model(data_pipeline, horizon,
                                                 exp_dir)

        self._save_results(results, exp_dir)
        logger.info("N-BEATS Experiment completed successfully.")
        return {"results_dir": exp_dir, "results": results}

    def _train_model(
            self, data_pipeline: Dict, horizon: int, exp_dir: str
    ) -> Dict[str, Any]:
        """Train and evaluate a single model for a specific horizon.

        :param data_pipeline: The dictionary of datasets and steps.
        :param horizon: The current forecast horizon.
        :param exp_dir: The main experiment directory.
        :return: A dictionary with training history and test results.
        """
        model = self.create_model(horizon)
        model.build((None, self.config.backcast_length, 1))

        viz_dir = os.path.join(exp_dir, f'visualizations_h{horizon}')
        performance_cb = PatternPerformanceCallback(
            self.config, self.processor, horizon, viz_dir, "nbeats_model"
        )

        model_path = os.path.join(exp_dir, f'best_model_h{horizon}.keras')
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=20, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=10, min_lr=1e-7),
            keras.callbacks.ModelCheckpoint(
                filepath=model_path, save_best_only=True),
            performance_cb
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
        test_results = model.evaluate(
            data_pipeline['test_ds'],
            steps=data_pipeline['test_steps'],
            verbose=0
        )

        if self.config.reconstruction_loss_weight > 0.0:
            # Keras returns: [total_loss, forecast_loss, recon_loss, mae, mse, ...]
            # We report the forecast-specific loss and mae.
            test_loss = test_results[1]
            test_mae = test_results[3] if len(test_results) > 3 else None
        else:
            # Keras returns: [loss, mae, mse, ...]
            test_loss = test_results[0]
            test_mae = test_results[1] if len(test_results) > 1 else None

        final_epoch = len(history.history['loss'])

        return {
            'history': history.history,
            'test_loss': test_loss,
            'test_mae': test_mae,
            'final_epoch': final_epoch
        }

    def _save_results(self, results: Dict, exp_dir: str) -> None:
        """Save the final experiment results to a JSON file.

        :param results: The dictionary of results from the experiment.
        :param exp_dir: The directory to save the results file in.
        """
        logger.info(f"Saving results to {exp_dir}")
        serializable_results = {}
        for h, r in results.items():
            # Convert numpy floats in history to standard Python floats for JSON
            history_serializable = {
                k: [float(v) for v in val]
                for k, val in r['history'].items()
            }
            serializable_results[h] = {
                'history': history_serializable,
                'test_loss': float(r['test_loss']),
                'test_mae': float(r['test_mae']) if r['test_mae'] else None,
                'final_epoch': r['final_epoch']
            }

        with open(os.path.join(exp_dir, 'results.json'), 'w') as f:
            json.dump(serializable_results, f, indent=4)


def main() -> None:
    """Main function to configure and run the N-BEATS training experiment."""
    config = NBeatsTrainingConfig(
        activation="gelu",
        experiment_name="nbeats",
        backcast_length=104,
        forecast_horizons=[4],
        stack_types=["generic"],
        nb_blocks_per_stack=1,
        hidden_layer_units=64,
        use_normalization=True,
        normalize_per_instance=True,
        max_patterns_per_category=100,
        epochs=100,
        batch_size=256,
        steps_per_epoch=4000,
        learning_rate=1e-4,
        dropout_rate=0.1,
        kernel_regularizer_l2=1e-5,
        # Enable reconstruction loss to force the model to explain the backcast.
        reconstruction_loss_weight=0.1,
        primary_loss=keras.losses.MeanAbsoluteError(reduction="mean"),
        mase_seasonal_periods=1,
        # Enable and configure the warmup schedule
        use_warmup=True,
        warmup_steps=1000,
        warmup_start_lr=1e-6,
    )
    ts_config = TimeSeriesConfig(
        n_samples=500,
        random_seed=42
    )

    try:
        trainer = NBeatsTrainer(config, ts_config)
        trainer.run_experiment()
    except Exception as e:
        logger.error(f"Experiment failed: {e}", exc_info=True)


if __name__ == "__main__":
    main()